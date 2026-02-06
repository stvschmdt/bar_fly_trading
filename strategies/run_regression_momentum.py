#!/usr/bin/env python
"""
Runner script for Regression Momentum Strategy.

Supports two modes:
  1. backtest — full historical backtest via backtest.py
  2. daily   — short lookback (1-2 days), email summary of what hit

Uses the bar_fly_trading backtest framework with stockformer predictions.
Portfolio filtering/ranking via portfolio.py is applied before the backtest
to narrow the symbol universe.

Usage:
    # Full backtest
    python run_regression_momentum.py \
        --predictions merged_predictions.csv \
        --use-all-symbols \
        --data-path 'all_data_*.csv' \
        --start-date 2024-07-01 \
        --end-date 2024-12-31

    # Daily report (morning email of what hit in past 2 days)
    python run_regression_momentum.py \
        --predictions merged_predictions.csv \
        --use-all-symbols \
        --data-path 'all_data_*.csv' \
        --mode daily \
        --lookback-days 2 \
        --watchlist api_data/watchlist.csv \
        --sort-sharpe \
        --summary-only

    # Portfolio filtering: price band + top 15 by Sharpe
    python run_regression_momentum.py \
        --predictions merged_predictions.csv \
        --use-all-symbols \
        --data-path 'all_data_*.csv' \
        --price-above 25 --top-k-sharpe 15 \
        --start-date 2024-07-01 \
        --end-date 2024-12-31
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
from backtest_stats import compute_stats, print_stats, write_trade_log, write_symbols, read_symbols
from signal_writer import SignalWriter
from ibkr.notifier import TradeNotifier
from portfolio import (
    load_data as portfolio_load_data,
    load_watchlist,
    apply_watchlist,
    run_pipeline as portfolio_pipeline,
)

from regression_momentum_strategy import RegressionMomentumStrategy


def get_available_symbols(predictions_path):
    """Get list of symbols available in predictions (file or directory)."""
    if os.path.isfile(predictions_path):
        df = pd.read_csv(predictions_path, nrows=1)
        col = 'ticker' if 'ticker' in df.columns else 'symbol'
        df = pd.read_csv(predictions_path, usecols=[col])
        return set(df[col].unique())
    elif os.path.isdir(predictions_path):
        # Legacy: check for individual files
        for name in ['pred_reg_3d.csv', 'predictions_reg_3d.csv']:
            pred_file = os.path.join(predictions_path, name)
            if os.path.exists(pred_file):
                df = pd.read_csv(pred_file, nrows=1)
                col = 'ticker' if 'ticker' in df.columns else 'symbol'
                df = pd.read_csv(pred_file, usecols=[col])
                return set(df[col].unique())
    return set()


# ================================================================== #
# MODE 1: FULL BACKTEST
# ================================================================== #

def run_backtest(symbols, start_date, end_date, start_cash, predictions_path,
                 data_path=None, position_size=0.1, output_trades=None, output_symbols=None,
                 output_signals=None, filters_applied=None, ranks_applied=None):
    """Run the full Regression Momentum backtest."""
    # Load price data from CSV (no database needed)
    data = None
    if data_path:
        data = portfolio_load_data(data_path)
        print(f"  Loaded {len(data):,} rows from {data_path}")

    # Create account
    account = BacktestAccount(
        account_id="regression_momentum_backtest",
        owner_name="Regression Momentum Backtest",
        account_values=AccountValues(start_cash, 0, 0),
        start_date=pd.to_datetime(start_date)
    )

    # Create strategy
    strategy = RegressionMomentumStrategy(
        account=account,
        symbols=symbols,
        predictions_path=predictions_path,
        position_size=position_size
    )

    # Print header
    print("\n" + "=" * 70)
    print("REGRESSION MOMENTUM STRATEGY BACKTEST")
    print("=" * 70)
    print(f"""
Strategy Parameters:
  Entry: pred_reg_3d > 1% AND pred_reg_10d > 2% AND adx_signal > 0
  Exit:  pred_reg_3d < 0 OR cci_signal < 0 OR hold >= 13 days
  Min Hold: 2 days | Max Hold: 13 days
  Position Size: {position_size * 100:.0f}% of portfolio

Backtest Setup:
  Symbols: {len(symbols)} stocks
  Date Range: {start_date} to {end_date}
  Starting Cash: ${start_cash:,.2f}
  Predictions: {predictions_path}""")

    if filters_applied:
        print(f"  Filters:       {', '.join(filters_applied)}")
    if ranks_applied:
        print(f"  Rank:          {', '.join(ranks_applied)}")
    print()
    print("=" * 70 + "\n")

    # Run backtest
    account_values = backtest(strategy, symbols, start_date, end_date, data=data)

    # Compute and print stats
    final_value = account_values.get_total_value()
    stats = compute_stats(strategy.trade_log, start_cash)
    print_stats(stats, start_cash, final_value)

    # Open positions
    open_positions = strategy.get_open_positions()
    if open_positions:
        print(f"\nOpen Positions ({len(open_positions)}):")
        for symbol, pos in open_positions.items():
            print(f"  {symbol}: {pos['shares']} shares @ ${pos['entry_price']:.2f}")
        print()

    # Write output files
    if output_trades:
        write_trade_log(strategy.trade_log, output_trades)
    if output_symbols:
        write_symbols(symbols, output_symbols)

    # Write pending signals for the last day's orders (for live execution bridge)
    if output_signals and strategy.trade_log:
        writer = SignalWriter(output_signals)
        last_date = max(t['entry_date'] for t in strategy.trade_log)
        for t in strategy.trade_log:
            if t['entry_date'] == last_date:
                writer.add(
                    action='BUY',
                    symbol=t['symbol'],
                    price=t['entry_price'],
                    strategy='regression_momentum',
                    reason=f"pred backtest entry {last_date}",
                )
        writer.save()

    return account_values


# ================================================================== #
# MODE 2: DAILY REPORT (short lookback + email summary)
# ================================================================== #

def generate_regression_summary(strategy, stats, start_cash, final_value,
                                lookback_days, filters_applied=None,
                                ranks_applied=None):
    """Generate a formatted text summary of regression momentum results."""
    lines = [
        "=" * 85,
        "REGRESSION MOMENTUM STRATEGY SUMMARY",
        "=" * 85,
        f"Scan Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Lookback:      {lookback_days} trading days",
        f"Total Trades:  {len(strategy.trade_log)}",
    ]

    if filters_applied:
        lines.append(f"Filters:       {', '.join(filters_applied)}")
    if ranks_applied:
        lines.append(f"Rank:          {', '.join(ranks_applied)}")

    lines.append(f"Symbols:       {len(strategy.symbols)}")
    lines.append("")

    # Separate entries and exits from the trade log
    # Entries that are still open (in positions)
    # Completed trades (have exit_date)
    entries = [t for t in strategy.trade_log if 'exit_date' in t and t.get('exit_date')]
    open_positions = strategy.get_open_positions()

    # ENTRIES section - show completed trades with entry info
    lines.append(f"COMPLETED TRADES ({len(entries)}):")
    lines.append("-" * 85)
    if entries:
        lines.append(f"{'Symbol':<8} {'Entry':<12} {'Exit':<12} {'Entry$':>10} {'Exit$':>10} {'Return':>8} {'Hold':>6}")
        lines.append("-" * 85)
        for t in entries:
            ret_str = f"{t['return_pct']:+.2f}%"
            hold_str = f"{t['hold_days']}d"
            lines.append(
                f"{t['symbol']:<8} {t['entry_date']:<12} {t['exit_date']:<12} "
                f"${t['entry_price']:>8.2f} ${t['exit_price']:>8.2f} {ret_str:>8} {hold_str:>6}"
            )
    else:
        lines.append("  (none)")

    lines.append("")

    # OPEN POSITIONS section
    lines.append(f"OPEN POSITIONS ({len(open_positions)}):")
    lines.append("-" * 85)
    if open_positions:
        lines.append(f"{'Symbol':<8} {'Entry Date':<12} {'Entry$':>10} {'Shares':>8}")
        lines.append("-" * 85)
        for symbol, pos in open_positions.items():
            entry_date_str = pos['entry_date'].strftime('%Y-%m-%d') if hasattr(pos['entry_date'], 'strftime') else str(pos['entry_date'])[:10]
            lines.append(
                f"{symbol:<8} {entry_date_str:<12} ${pos['entry_price']:>8.2f} {pos['shares']:>8}"
            )
    else:
        lines.append("  (none)")

    lines.append("")

    # Stats summary
    if stats:
        lines.append("-" * 85)
        lines.append("STATS:")
        total_trades = stats.get('total_trades', 0)
        win_rate = stats.get('win_rate', 0)
        avg_return = stats.get('avg_return_pct', 0)
        total_pnl = stats.get('total_pnl', 0)
        lines.append(f"  Trades: {total_trades}  |  Win Rate: {win_rate:.1f}%  |  Avg Return: {avg_return:+.2f}%  |  PnL: ${total_pnl:,.2f}")

    lines.append("")
    lines.append("=" * 85)

    return "\n".join(lines)


def run_daily_report(symbols, predictions_path, data_path, position_size=0.1,
                     lookback_days=2, start_cash=100000, notifier=None,
                     filters_applied=None, ranks_applied=None):
    """
    Run a short lookback backtest and email a summary of what hit.

    This is the "morning report" mode: run over the past N days to see
    which symbols triggered entry/exit signals.
    """
    # Calculate date range: pad with extra calendar days to ensure we cover
    # enough trading days (weekends, holidays)
    end_date = date.today().strftime('%Y-%m-%d')
    start_date = (date.today() - timedelta(days=lookback_days + 5)).strftime('%Y-%m-%d')

    # Load price data
    data = None
    if data_path:
        data = portfolio_load_data(data_path)
        print(f"  Loaded {len(data):,} rows from {data_path}")

    # Create account
    account = BacktestAccount(
        account_id="regression_daily",
        owner_name="Regression Momentum Daily",
        account_values=AccountValues(start_cash, 0, 0),
        start_date=pd.to_datetime(start_date)
    )

    # Create strategy
    strategy = RegressionMomentumStrategy(
        account=account,
        symbols=symbols,
        predictions_path=predictions_path,
        position_size=position_size
    )

    # Run short backtest
    account_values = backtest(strategy, symbols, start_date, end_date, data=data)

    # Compute stats
    final_value = account_values.get_total_value()
    stats = compute_stats(strategy.trade_log, start_cash)

    # Generate summary
    summary = generate_regression_summary(
        strategy, stats, start_cash, final_value,
        lookback_days, filters_applied, ranks_applied
    )

    # Print summary
    print(summary)

    # Email via TradeNotifier
    if notifier:
        trade_count = len(strategy.trade_log)
        open_count = len(strategy.get_open_positions())
        subject = (
            f"[REGRESSION] Momentum Signals Summary - "
            f"{trade_count} trade(s), {open_count} open ({date.today()})"
        )
        if notifier._send_email(subject, summary):
            print(f"\nEmail sent: {subject}")
        else:
            print("\nWarning: Failed to send email")

    return strategy


# ================================================================== #
# MAIN
# ================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Regression Momentum Strategy (backtest or daily report)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full backtest
  python run_regression_momentum.py --predictions merged_predictions.csv \\
      --use-all-symbols --data-path 'all_data_*.csv' \\
      --start-date 2024-07-01 --end-date 2024-12-31

  # Daily report (morning email)
  python run_regression_momentum.py --predictions merged_predictions.csv \\
      --use-all-symbols --data-path 'all_data_*.csv' \\
      --mode daily --lookback-days 2 \\
      --watchlist api_data/watchlist.csv --sort-sharpe --summary-only

  # Portfolio filter: price > $25, top 15 by Sharpe
  python run_regression_momentum.py --predictions merged_predictions.csv \\
      --use-all-symbols --data-path 'all_data_*.csv' \\
      --price-above 25 --top-k-sharpe 15 \\
      --start-date 2024-07-01 --end-date 2024-12-31
        """
    )

    # --- Mode ---
    parser.add_argument("--mode", type=str, default="backtest",
                        choices=["backtest", "daily"],
                        help="'backtest' for historical test, 'daily' for short lookback + email (default: backtest)")

    # --- Data ---
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to merged_predictions.csv or predictions directory")
    parser.add_argument("--symbols", type=str, nargs="+",
                        help="Symbols to trade (e.g., AAPL GOOGL MSFT)")
    parser.add_argument("--symbols-file", type=str, default=None,
                        help="CSV file with symbol list (output of portfolio or prior backtest)")
    parser.add_argument("--use-all-symbols", action="store_true",
                        help="Use all symbols available in predictions")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to price data CSV(s) for backtest (supports globs, e.g. 'all_data_*.csv')")

    # --- Backtest-specific ---
    parser.add_argument("--start-date", type=str, default=None,
                        help="Backtest start date (YYYY-MM-DD) — required for backtest mode")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Backtest end date (YYYY-MM-DD) — required for backtest mode")
    parser.add_argument("--start-cash", type=float, default=100000,
                        help="Initial cash balance (default: 100000)")
    parser.add_argument("--position-size", type=float, default=0.1,
                        help="Position size as fraction of portfolio (default: 0.1 = 10%%)")

    # --- Daily report-specific ---
    parser.add_argument("--lookback-days", type=int, default=2,
                        help="Days to look back for daily report mode (default: 2)")
    parser.add_argument("--no-notify", action="store_true",
                        help="Skip sending email notification")
    parser.add_argument("--summary-only", action="store_true",
                        help="Send summary email (daily mode, default behavior)")

    # --- Portfolio ranker args ---
    parser.add_argument("--portfolio-data", type=str, default=None,
                        help="Path to data CSV for portfolio ranking (default: same as --data-path)")
    parser.add_argument("--watchlist", type=str, default=None,
                        help="Path to watchlist CSV file")
    parser.add_argument("--watchlist-mode", type=str, default='sort',
                        choices=['sort', 'filter'],
                        help="'sort' = watchlist first, 'filter' = watchlist only (default: sort)")
    parser.add_argument("--price-above", type=float, default=None,
                        help="Min stock price filter (inclusive)")
    parser.add_argument("--price-below", type=float, default=None,
                        help="Max stock price filter (inclusive)")
    parser.add_argument("--filter-field", type=str, default=None,
                        help="Column name to filter/rank on (e.g. rsi_14, beta, pe_ratio)")
    parser.add_argument("--filter-above", type=float, default=None,
                        help="Min value for --filter-field (inclusive)")
    parser.add_argument("--filter-below", type=float, default=None,
                        help="Max value for --filter-field (inclusive)")
    parser.add_argument("--top-k-sharpe", type=int, default=None,
                        help="Keep top K symbols ranked by Sharpe ratio")
    parser.add_argument("--sort-sharpe", action="store_true",
                        help="Sort symbols by Sharpe ratio (no cutoff, ordering only)")

    # --- Output files ---
    parser.add_argument("--output-trades", type=str, default=None,
                        help="Path to write trade log CSV")
    parser.add_argument("--output-symbols", type=str, default=None,
                        help="Path to write filtered symbol list CSV")
    parser.add_argument("--output-signals", type=str, default=None,
                        help="Path to write pending signal CSV for live execution")

    args = parser.parse_args()

    # Determine symbols
    if args.use_all_symbols:
        symbols = get_available_symbols(args.predictions)
        if not symbols:
            print(f"Error: No symbols found in {args.predictions}")
            sys.exit(1)
        print(f"Using all {len(symbols)} symbols from predictions")
    elif args.symbols_file:
        symbols = set(read_symbols(args.symbols_file))
        print(f"Loaded {len(symbols)} symbols from {args.symbols_file}")
    elif args.symbols:
        symbols = set(args.symbols)
    else:
        print("Error: Must specify --symbols, --symbols-file, or --use-all-symbols")
        sys.exit(1)

    # Default portfolio-data to data-path
    if not args.portfolio_data and args.data_path:
        args.portfolio_data = args.data_path

    # --- Portfolio ranking pipeline ---
    has_portfolio_filters = any([
        args.watchlist, args.price_above is not None, args.price_below is not None,
        args.filter_field, args.top_k_sharpe is not None, args.sort_sharpe,
    ])

    if has_portfolio_filters:
        portfolio_df = None
        if args.portfolio_data:
            print(f"Loading portfolio data: {args.portfolio_data}")
            portfolio_df = portfolio_load_data(args.portfolio_data)

        wl = load_watchlist(args.watchlist) if args.watchlist else None

        if portfolio_df is not None:
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
            )
            symbols = set(symbols_list)
        elif wl:
            symbols = set(apply_watchlist(list(symbols), wl, args.watchlist_mode))
            print(f"After watchlist ({args.watchlist_mode}): {len(symbols)} symbols")

    # Build filter/rank metadata for summary display
    filters_applied = []
    ranks_applied = []
    if args.watchlist:
        wl_count = len(load_watchlist(args.watchlist)) if args.watchlist else 0
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
    if not ranks_applied:
        ranks_applied.append("symbol order")

    # Write symbol list before backtest (portfolio-filtered set)
    if args.output_symbols:
        write_symbols(symbols, args.output_symbols)

    # Dispatch
    if args.mode == "backtest":
        if not args.start_date or not args.end_date:
            print("Error: --start-date and --end-date required for backtest mode")
            sys.exit(1)

        run_backtest(
            symbols=symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            start_cash=args.start_cash,
            predictions_path=args.predictions,
            data_path=args.data_path,
            position_size=args.position_size,
            output_trades=args.output_trades,
            output_signals=args.output_signals,
            filters_applied=filters_applied,
            ranks_applied=ranks_applied,
        )

    elif args.mode == "daily":
        notifier = None if args.no_notify else TradeNotifier()

        run_daily_report(
            symbols=symbols,
            predictions_path=args.predictions,
            data_path=args.data_path,
            position_size=args.position_size,
            lookback_days=args.lookback_days,
            start_cash=args.start_cash,
            notifier=notifier,
            filters_applied=filters_applied,
            ranks_applied=ranks_applied,
        )
