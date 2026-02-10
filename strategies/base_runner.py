"""
Base runner for bar_fly_trading strategies.

All strategy runners (run_*.py) inherit from BaseRunner which provides:
  - Shared CLI argument parsing (mode, data, symbols, portfolio filters, output)
  - Symbol resolution (--symbols, --symbols-file, --use-all-symbols)
  - Portfolio filtering pipeline (watchlist, price, sharpe ranking)
  - Three execution modes: backtest, daily, live
  - Backtest summary formatting and email
  - Signal CSV output (signals/pending_orders.csv)

Each concrete runner (~50-100 lines) implements:
  - STRATEGY_NAME, EMAIL_TAG
  - add_strategy_args(parser) — strategy-specific CLI args
  - create_strategy(account, symbols, args, data) — instantiate strategy class
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta

import pandas as pd

# Add bar_fly_trading and strategies to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from account.account_values import AccountValues
from account.backtest_account import BacktestAccount
from backtest import backtest
from backtest_stats import compute_stats, print_stats, write_trade_log, write_symbols, read_symbols, write_rankings
from signal_writer import SignalWriter
from ibkr.notifier import TradeNotifier
from portfolio import (
    load_data as portfolio_load_data,
    load_watchlist,
    apply_watchlist,
    run_pipeline as portfolio_pipeline,
)


class BaseRunner:
    """Base class for all strategy run_*.py files."""

    STRATEGY_NAME = "base"
    EMAIL_TAG = "[BASE]"

    # ================================================================== #
    #  CLI ARGUMENT PARSING
    # ================================================================== #

    @classmethod
    def build_parser(cls, description=None):
        """Build shared argument parser with all common options."""
        desc = description or f"Run {cls.STRATEGY_NAME} strategy (backtest, daily, or live)"
        parser = argparse.ArgumentParser(
            description=desc,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        # --- Mode ---
        parser.add_argument("--mode", type=str, default="backtest",
                            choices=["backtest", "daily", "live"],
                            help="'backtest' for historical test, 'daily' for "
                                 "short lookback scan, 'live' for AlphaVantage "
                                 "realtime (default: backtest)")

        # --- Data ---
        parser.add_argument("--data-path", type=str, default=None,
                            help="Path to CSV data (supports globs, e.g. 'all_data_*.csv')")
        parser.add_argument("--predictions", type=str, default=None,
                            help="Path to merged_predictions.csv")
        parser.add_argument("--symbols", type=str, nargs="+",
                            help="Symbols to trade (e.g., AAPL GOOGL MSFT)")
        parser.add_argument("--symbols-file", type=str, default=None,
                            help="CSV file with symbol list")
        parser.add_argument("--use-all-symbols", action="store_true",
                            help="Use all symbols available in data")

        # --- Backtest ---
        parser.add_argument("--start-date", type=str, default=None,
                            help="Backtest start date (YYYY-MM-DD)")
        parser.add_argument("--end-date", type=str, default=None,
                            help="Backtest end date (YYYY-MM-DD)")
        parser.add_argument("--start-cash", type=float, default=100000,
                            help="Initial cash (default: 100000)")
        parser.add_argument("--position-size", type=float, default=0.1,
                            help="Position size fraction (default: 0.1 = 10%%)")

        # --- Daily/Live ---
        parser.add_argument("--lookback-days", type=int, default=2,
                            help="Days to look back for daily/live mode (default: 2)")

        # --- Notifications ---
        parser.add_argument("--no-notify", action="store_true",
                            help="Skip sending email notifications")
        parser.add_argument("--summary-only", action="store_true",
                            help="Send summary email only")
        parser.add_argument("--skip-live", action="store_true",
                            help="Skip IBKR Gateway connection (for testing)")

        # --- Portfolio filters ---
        parser.add_argument("--watchlist", type=str, default=None,
                            help="Path to watchlist CSV file")
        parser.add_argument("--watchlist-mode", type=str, default='sort',
                            choices=['sort', 'filter'],
                            help="'sort' or 'filter' (default: sort)")
        parser.add_argument("--portfolio-data", type=str, default=None,
                            help="Path to data for portfolio ranking (default: --data-path)")
        parser.add_argument("--price-above", type=float, default=None,
                            help="Min stock price filter")
        parser.add_argument("--price-below", type=float, default=None,
                            help="Max stock price filter")
        parser.add_argument("--filter-field", type=str, default=None,
                            help="Column name to filter/rank on")
        parser.add_argument("--filter-above", type=float, default=None,
                            help="Min value for --filter-field")
        parser.add_argument("--filter-below", type=float, default=None,
                            help="Max value for --filter-field")
        parser.add_argument("--top-k-sharpe", type=int, default=None,
                            help="Keep top K symbols by Sharpe ratio")
        parser.add_argument("--sort-sharpe", action="store_true",
                            help="Sort symbols by Sharpe ratio")

        # --- Backtest rankings (portfolio filter) ---
        parser.add_argument("--backtest-rankings", type=str, default=None,
                            help="Path to backtest_rankings.csv for ranking/filtering symbols")
        parser.add_argument("--rank-by", type=str, default="score",
                            choices=["win_rate", "avg_return_pct", "total_pnl", "score", "trades"],
                            help="Field to rank by from rankings CSV (default: score)")
        parser.add_argument("--rank-top-k", type=int, default=None,
                            help="Keep top K symbols from backtest rankings")

        # --- Output ---
        parser.add_argument("--output-trades", type=str, default=None,
                            help="Path to write trade log CSV")
        parser.add_argument("--output-symbols", type=str, default=None,
                            help="Path to write filtered symbol list CSV")
        parser.add_argument("--output-signals", type=str, default=None,
                            help="Path to write signal CSV (default: signals/pending_orders.csv for live mode)")
        parser.add_argument("--output-rankings", type=str, default="backtest_rankings.csv",
                            help="Path to write per-symbol rankings CSV (default: backtest_rankings.csv)")

        # Let subclass add strategy-specific args
        cls.add_strategy_args(parser)

        return parser

    @classmethod
    def add_strategy_args(cls, parser):
        """Override to add strategy-specific CLI arguments."""
        pass

    # ================================================================== #
    #  SYMBOL RESOLUTION
    # ================================================================== #

    @staticmethod
    def resolve_symbols(args, data=None):
        """
        Determine symbols from CLI args.

        Priority: --symbols > --symbols-file > --watchlist > data > error

        Returns:
            set of symbol strings
        """
        if args.symbols:
            symbols = set(args.symbols)
            print(f"Using {len(symbols)} specified symbols")
            return symbols

        if args.symbols_file:
            symbols = set(read_symbols(args.symbols_file))
            print(f"Loaded {len(symbols)} symbols from {args.symbols_file}")
            return symbols

        # Watchlist as primary symbol source
        if args.watchlist:
            from portfolio import load_watchlist
            symbols = set(load_watchlist(args.watchlist))
            if symbols:
                print(f"Loaded {len(symbols)} symbols from watchlist {args.watchlist}")
                return symbols

        # Fall back to all symbols in data
        if args.use_all_symbols and data is not None:
            if isinstance(data, pd.DataFrame):
                symbols = set(data['symbol'].unique())
            else:
                loaded = portfolio_load_data(data)
                symbols = set(loaded['symbol'].unique())
            print(f"Found {len(symbols)} symbols in data")
            return symbols

        if args.use_all_symbols and args.data_path:
            loaded = portfolio_load_data(args.data_path)
            symbols = set(loaded['symbol'].unique())
            print(f"Found {len(symbols)} symbols from {args.data_path}")
            return symbols

        if args.predictions:
            loaded = portfolio_load_data(args.predictions)
            symbols = set(loaded['symbol'].unique())
            print(f"Found {len(symbols)} symbols from {args.predictions}")
            return symbols

        print("Error: No symbols specified. Use --symbols, --watchlist, or --use-all-symbols")
        sys.exit(1)

    # ================================================================== #
    #  PORTFOLIO PIPELINE
    # ================================================================== #

    @staticmethod
    def apply_portfolio_pipeline(args, symbols, data=None):
        """
        Apply watchlist, price filters, sharpe ranking.

        Args:
            args: Parsed CLI args
            symbols: set of symbol strings
            data: Optional pre-loaded DataFrame

        Returns:
            (filtered_symbols: set, filters_applied: list, ranks_applied: list)
        """
        filters_applied = []
        ranks_applied = []

        has_filters = any([
            args.watchlist, args.price_above is not None,
            args.price_below is not None, args.filter_field,
            args.top_k_sharpe is not None, args.sort_sharpe,
            getattr(args, 'backtest_rankings', None),
        ])

        if has_filters:
            portfolio_data_path = args.portfolio_data or args.data_path
            if portfolio_data_path:
                portfolio_df = data if data is not None else portfolio_load_data(portfolio_data_path)
                wl = load_watchlist(args.watchlist) if args.watchlist else None

                symbols_list = portfolio_pipeline(
                    portfolio_df,
                    symbols=list(symbols),
                    watchlist=wl,
                    watchlist_mode=args.watchlist_mode,
                    price_above=args.price_above,
                    price_below=args.price_below,
                    filter_field=args.filter_field,
                    filter_above=args.filter_above,
                    filter_below=args.filter_below,
                    top_k_sharpe=args.top_k_sharpe,
                    sort_sharpe=args.sort_sharpe,
                    backtest_rankings=getattr(args, 'backtest_rankings', None),
                    rank_by=getattr(args, 'rank_by', 'score'),
                    rank_top_k=getattr(args, 'rank_top_k', None),
                )
                symbols = set(symbols_list)
            elif getattr(args, 'backtest_rankings', None):
                # Backtest rankings don't need price data — handle standalone
                from portfolio import rank_by_backtest
                symbols_list = rank_by_backtest(
                    list(symbols),
                    args.backtest_rankings,
                    rank_field=args.rank_by,
                    top_k=args.rank_top_k,
                )
                symbols = set(symbols_list)
            elif args.watchlist:
                wl = load_watchlist(args.watchlist)
                if wl:
                    symbols = set(apply_watchlist(list(symbols), wl, args.watchlist_mode))
                    print(f"After watchlist ({args.watchlist_mode}): {len(symbols)} symbols")

        # Build metadata strings
        if args.watchlist:
            wl_count = len(load_watchlist(args.watchlist))
            mode_label = "filter" if args.watchlist_mode == "filter" else "sort"
            filters_applied.append(f"watchlist ({wl_count} symbols, {mode_label})")
        if args.price_above is not None or args.price_below is not None:
            parts = []
            if args.price_above is not None:
                parts.append(f">= ${args.price_above:.0f}")
            if args.price_below is not None:
                parts.append(f"<= ${args.price_below:.0f}")
            filters_applied.append(f"price {', '.join(parts)}")
        if args.filter_field:
            filters_applied.append(f"{args.filter_field} range")
        if args.top_k_sharpe is not None:
            ranks_applied.append(f"sharpe ratio (top {args.top_k_sharpe})")
        elif args.sort_sharpe:
            ranks_applied.append("sharpe ratio (sorted)")
        if getattr(args, 'backtest_rankings', None):
            rank_label = getattr(args, 'rank_by', 'score')
            if getattr(args, 'rank_top_k', None):
                ranks_applied.append(f"backtest {rank_label} (top {args.rank_top_k})")
            else:
                ranks_applied.append(f"backtest {rank_label} (sorted)")
        if not ranks_applied:
            ranks_applied.append("symbol order")

        return symbols, filters_applied, ranks_applied

    # ================================================================== #
    #  STRATEGY CREATION (subclass must override)
    # ================================================================== #

    def create_strategy(self, account, symbols, args, data=None):
        """
        Create and return the strategy instance.

        Subclass MUST override this method.

        Args:
            account: BacktestAccount instance
            symbols: set of symbols
            args: parsed CLI args
            data: pre-loaded DataFrame (optional)

        Returns:
            BaseStrategy subclass instance
        """
        raise NotImplementedError("Subclass must implement create_strategy()")

    # ================================================================== #
    #  MODE 1: BACKTEST
    # ================================================================== #

    def run_backtest(self, symbols, args, data, notifier,
                     filters_applied=None, ranks_applied=None):
        """
        Run full historical backtest.

        Creates account, runs backtest engine, computes stats,
        formats summary, writes outputs, sends email.
        """
        # Create account
        account = BacktestAccount(
            account_id=f"{self.STRATEGY_NAME}_backtest",
            owner_name=f"{self.STRATEGY_NAME.title()} Backtest",
            account_values=AccountValues(args.start_cash, 0, 0),
            start_date=pd.to_datetime(args.start_date),
        )

        # Create strategy with data
        strategy = self.create_strategy(account, symbols, args, data)

        # Print header
        print("\n" + "=" * 70)
        print(f"{self.STRATEGY_NAME.upper()} STRATEGY BACKTEST")
        print("=" * 70)
        print(f"\nBacktest Setup:")
        print(f"  Symbols: {len(symbols)} stocks")
        print(f"  Date Range: {args.start_date} to {args.end_date}")
        print(f"  Starting Cash: ${args.start_cash:,.2f}")
        print(f"  Position Size: {args.position_size * 100:.0f}%")
        if filters_applied:
            print(f"  Filters:       {', '.join(filters_applied)}")
        if ranks_applied:
            print(f"  Rank:          {', '.join(ranks_applied)}")
        print("\n" + "=" * 70 + "\n")

        # Run backtest
        account_values = backtest(strategy, symbols, args.start_date,
                                  args.end_date, data=data)

        # Compute and print stats
        final_value = account_values.get_total_value()
        stats = compute_stats(strategy.trade_log, args.start_cash)
        print_stats(stats, args.start_cash, final_value)

        # Format email text
        summary_text = self.format_backtest_summary(
            stats, args.start_cash, final_value, strategy,
            filters_applied, ranks_applied, args.start_date, args.end_date,
        )

        # Write output files
        if args.output_trades:
            write_trade_log(strategy.trade_log, args.output_trades)
        if args.output_symbols:
            write_symbols(symbols, args.output_symbols)
        if stats['per_symbol']:
            rankings_path = getattr(args, 'output_rankings', 'backtest_rankings.csv')
            write_rankings(stats['per_symbol'], rankings_path)

        # Write pending signals for open positions at backtest end
        output_signals = args.output_signals
        if output_signals:
            open_positions = strategy.get_open_positions()
            if open_positions:
                writer = SignalWriter(output_signals)
                for sym, pos in open_positions.items():
                    entry_str = (pos['entry_date'].strftime('%Y-%m-%d')
                                 if hasattr(pos['entry_date'], 'strftime')
                                 else str(pos['entry_date'])[:10])
                    writer.add(
                        action='BUY', symbol=sym,
                        price=pos['entry_price'],
                        strategy=self.STRATEGY_NAME,
                        reason=f"open position from {entry_str}",
                        stop_loss_pct=strategy.STOP_LOSS_PCT,
                        take_profit_pct=strategy.TAKE_PROFIT_PCT,
                        trailing_stop_pct=strategy.TRAILING_STOP_PCT,
                        max_hold_days=strategy.MAX_HOLD_DAYS,
                    )
                writer.save()
                print(f"  Wrote {len(open_positions)} pending signals to {output_signals}")

        # Send email
        if notifier:
            trade_count = stats['total_trades']
            subject = (
                f"{self.EMAIL_TAG} Backtest | "
                f"{trade_count} trades, {stats['win_rate']:.1f}% win rate, "
                f"Sharpe {stats['sharpe_ratio']:.2f}"
            )
            strategy.send_notification(subject, summary_text, notifier)

        return account_values

    # ================================================================== #
    #  MODE 2: DAILY REPORT
    # ================================================================== #

    def run_daily_report(self, symbols, args, data, notifier,
                         filters_applied=None, ranks_applied=None):
        """
        Scan recent data for signals and report what triggered.

        No backtest — loads data, checks last N days, lists signals.
        """
        # Create dummy account for strategy instantiation
        account = BacktestAccount(
            account_id=f"{self.STRATEGY_NAME}_daily",
            owner_name=f"{self.STRATEGY_NAME.title()} Daily",
            account_values=AccountValues(100000, 0, 0),
            start_date=pd.to_datetime(date.today()),
        )

        strategy = self.create_strategy(account, symbols, args, data)

        # Find signals
        signals = strategy.find_signals(lookback_days=args.lookback_days)

        # Generate summary
        summary = strategy.generate_signal_summary(
            signals, mode_label="Daily Scan",
            lookback_days=args.lookback_days,
            filters=filters_applied, ranks=ranks_applied,
        )
        print(summary)

        # Write signal CSV
        output_signals = args.output_signals
        if output_signals and signals:
            writer = SignalWriter(output_signals)
            for sig in signals:
                writer.add(
                    action=sig['action'], symbol=sig['symbol'],
                    price=sig['price'], strategy=self.STRATEGY_NAME,
                    reason=sig.get('reason', ''),
                    stop_loss_pct=strategy.STOP_LOSS_PCT,
                    take_profit_pct=strategy.TAKE_PROFIT_PCT,
                    trailing_stop_pct=strategy.TRAILING_STOP_PCT,
                    max_hold_days=strategy.MAX_HOLD_DAYS,
                )
            writer.save()

        # Send email
        if notifier:
            buy_count = sum(1 for s in signals if s.get('action') == 'BUY')
            sell_count = sum(1 for s in signals if s.get('action') == 'SELL')
            subject = (
                f"{self.EMAIL_TAG} Daily | "
                f"{len(signals)} signal(s): {buy_count} BUY, {sell_count} SELL "
                f"({date.today()})"
            )
            strategy.send_notification(subject, summary, notifier)

        return signals

    # ================================================================== #
    #  MODE 3: LIVE (AlphaVantage realtime)
    # ================================================================== #

    def run_live(self, symbols, args, notifier,
                 filters_applied=None, ranks_applied=None):
        """
        Fetch live data from AlphaVantage, scan for signals.

        Optionally loads overnight data (merged_predictions.csv) and
        merges with realtime API data.

        Writes signals to signals/pending_orders.csv by default.
        """
        # Create dummy account for strategy instantiation
        account = BacktestAccount(
            account_id=f"{self.STRATEGY_NAME}_live",
            owner_name=f"{self.STRATEGY_NAME.title()} Live",
            account_values=AccountValues(100000, 0, 0),
            start_date=pd.to_datetime(date.today()),
        )

        # Load overnight data if available (for ML predictions, signals)
        overnight_data = None
        data_path = args.predictions or args.data_path
        if data_path:
            overnight_data = portfolio_load_data(data_path)
            print(f"  Loaded overnight data: {len(overnight_data):,} rows")

            # Warn if overnight data is stale (> 3 calendar days old)
            if 'date' in overnight_data.columns:
                latest_date = pd.to_datetime(overnight_data['date']).max()
                days_old = (pd.Timestamp.now() - latest_date).days
                if days_old > 3:
                    print(f"  WARNING: Overnight data is {days_old} days old "
                          f"(latest: {latest_date.strftime('%Y-%m-%d')}). "
                          f"Run cron.py or pull_api_data.py to refresh.")

        strategy = self.create_strategy(account, symbols, args, overnight_data)

        print(f"\n{'=' * 60}")
        print(f"  {self.STRATEGY_NAME.upper()} - LIVE MODE")
        print(f"  Symbols: {len(symbols)}")
        print(f"{'=' * 60}\n")

        # Fetch bulk realtime prices (1 API call per 100 symbols)
        bulk_prices = pd.DataFrame()
        if not getattr(args, 'skip_live', False):
            from api_data.rt_utils import get_realtime_quotes_bulk
            bulk_prices = get_realtime_quotes_bulk(list(symbols))
        else:
            print("  --skip-live: skipping bulk quote fetch (using overnight data only)")

        # Merge: overnight data (technicals) + bulk prices (realtime)
        if strategy.overnight_data is not None and not bulk_prices.empty:
            scan_data = strategy.merge_data(strategy.overnight_data, bulk_prices)
        elif strategy.overnight_data is not None:
            print("  No bulk prices fetched. Using overnight data only.")
            scan_data = strategy.overnight_data
        elif not bulk_prices.empty:
            scan_data = bulk_prices
        else:
            print("  No data available (no overnight data, no bulk prices).")
            scan_data = None

        # Find signals on merged data (require_today=True: only trade on today's data)
        signals = strategy.find_signals(data=scan_data,
                                        lookback_days=args.lookback_days,
                                        require_today=True)

        # Generate summary
        summary = strategy.generate_signal_summary(
            signals, mode_label="Live Scan",
            lookback_days=args.lookback_days,
            filters=filters_applied, ranks=ranks_applied,
        )
        print(summary)

        # Write signal CSV (default: signals/pending_orders.csv)
        # Uses append=True so multiple strategies can accumulate signals
        # in the same file during RT loop (executor de-dups before execution)
        output_signals = (args.output_signals or
                          os.path.join(parent_dir, "signals", "pending_orders.csv"))
        if signals:
            writer = SignalWriter(output_signals)
            for sig in signals:
                writer.add(
                    action=sig['action'], symbol=sig['symbol'],
                    price=sig['price'], strategy=self.STRATEGY_NAME,
                    reason=sig.get('reason', ''),
                    stop_loss_pct=strategy.STOP_LOSS_PCT,
                    take_profit_pct=strategy.TAKE_PROFIT_PCT,
                    trailing_stop_pct=strategy.TRAILING_STOP_PCT,
                    max_hold_days=strategy.MAX_HOLD_DAYS,
                )
            writer.save(append=True)
            print(f"\nSignal CSV written: {output_signals}")
            print(f"  {len(signals)} signal(s) ready for execution")
            print(f"\nTo execute (paper):  python -m ibkr.execute_signals "
                  f"--signals {output_signals} --port 4002 --client-id 10")
            print(f"To dry-run:          python -m ibkr.execute_signals "
                  f"--signals {output_signals} --dry-run")
        else:
            print("\nNo signals found.")

        # Send email
        if notifier:
            subject = (
                f"{self.EMAIL_TAG} Live | "
                f"{len(signals)} signal(s) ({date.today()})"
            )
            strategy.send_notification(subject, summary, notifier)

        return signals

    # ================================================================== #
    #  SUMMARY FORMATTING
    # ================================================================== #

    @staticmethod
    def format_backtest_summary(stats, start_cash, final_value, strategy,
                                filters=None, ranks=None,
                                start_date=None, end_date=None):
        """Format backtest stats as plain text for email."""
        total_return = (final_value - start_cash) / start_cash * 100

        lines = [
            "=" * 70,
            f"BACKTEST RESULTS - {strategy.STRATEGY_NAME.upper()}",
            "=" * 70,
            "",
        ]

        if start_date and end_date:
            lines.append(f"Date Range:    {start_date} to {end_date}")
        if filters:
            lines.append(f"Filters:       {', '.join(filters)}")
        if ranks:
            lines.append(f"Rank:          {', '.join(ranks)}")

        lines.extend([
            "",
            "Portfolio:",
            f"  Starting:      ${start_cash:,.2f}",
            f"  Ending:        ${final_value:,.2f}",
            f"  Total Return:  {total_return:+.2f}%",
            "",
            "Trade Summary:",
            f"  Total Trades:  {stats['total_trades']}",
            f"  Wins:          {stats['wins']}",
            f"  Losses:        {stats['losses']}",
            f"  Win Rate:      {stats['win_rate']:.1f}%",
            f"  Total P&L:     ${stats['total_pnl']:+,.2f}",
            f"  Avg Return:    {stats['avg_return_pct']:+.2f}%",
            f"  Avg Hold:      {stats['avg_hold_days']:.1f} days",
            f"  Sharpe Ratio:  {stats['sharpe_ratio']:.2f}",
            "",
        ])

        # Per-symbol breakdown
        if stats['per_symbol']:
            lines.append("-" * 70)
            lines.append(f"{'Symbol':<8} {'Trades':>6} {'Wins':>5} {'Losses':>6} "
                         f"{'Win%':>6} {'P&L':>12} {'Avg Ret%':>9} {'Avg Hold':>9}")
            lines.append("-" * 70)

            sorted_symbols = sorted(
                stats['per_symbol'].items(),
                key=lambda x: x[1]['total_pnl'],
                reverse=True,
            )

            for symbol, s in sorted_symbols:
                lines.append(
                    f"{symbol:<8} {s['trades']:>6} {s['wins']:>5} {s['losses']:>6} "
                    f"{s['win_rate']:>5.1f}% ${s['total_pnl']:>+10,.2f} "
                    f"{s['avg_return_pct']:>+8.2f}% {s['avg_hold_days']:>8.1f}d"
                )
            lines.append("-" * 70)

        # Open positions
        open_positions = strategy.get_open_positions()
        if open_positions:
            lines.append("")
            lines.append(f"Open Positions ({len(open_positions)}):")
            for sym, pos in open_positions.items():
                lines.append(f"  {sym}: {pos['shares']} shares @ ${pos['entry_price']:.2f}")

        lines.append("")
        lines.append("=" * 70)

        return "\n".join(lines)

    # ================================================================== #
    #  MAIN ENTRY POINT
    # ================================================================== #

    @classmethod
    def main(cls):
        """
        Shared __main__ entry point.

        Parses args, resolves symbols, applies portfolio pipeline,
        dispatches to appropriate mode handler.
        """
        parser = cls.build_parser()
        args = parser.parse_args()

        # Default portfolio-data to data-path
        if not args.portfolio_data and args.data_path:
            args.portfolio_data = args.data_path

        # Load data (for symbol resolution and pipeline)
        data = None
        data_source = args.predictions or args.data_path
        if data_source:
            data = portfolio_load_data(data_source)
            print(f"Loaded {len(data):,} rows from {data_source}")

        # Resolve symbols
        symbols = cls.resolve_symbols(args, data)

        # Apply portfolio pipeline
        symbols, filters_applied, ranks_applied = cls.apply_portfolio_pipeline(
            args, symbols, data)

        print(f"Final symbol universe: {len(symbols)} symbols")

        # Write symbol list if requested
        if args.output_symbols:
            write_symbols(symbols, args.output_symbols)

        # Create notifier
        notifier = None if args.no_notify else TradeNotifier()

        # Instantiate runner
        runner = cls()

        # Dispatch to mode
        if args.mode == "backtest":
            if not args.start_date or not args.end_date:
                print("Error: --start-date and --end-date required for backtest mode")
                sys.exit(1)
            if not data_source:
                print("Error: --data-path or --predictions required")
                sys.exit(1)

            runner.run_backtest(
                symbols=symbols, args=args, data=data,
                notifier=notifier,
                filters_applied=filters_applied,
                ranks_applied=ranks_applied,
            )

        elif args.mode == "daily":
            if not data_source:
                print("Error: --data-path or --predictions required for daily mode")
                sys.exit(1)

            runner.run_daily_report(
                symbols=symbols, args=args, data=data,
                notifier=notifier,
                filters_applied=filters_applied,
                ranks_applied=ranks_applied,
            )

        elif args.mode == "live":
            runner.run_live(
                symbols=symbols, args=args,
                notifier=notifier,
                filters_applied=filters_applied,
                ranks_applied=ranks_applied,
            )