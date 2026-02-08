#!/usr/bin/env python
"""
Runner script for Oversold Bounce Strategy (S1).

Pure technical strategy — no ML predictions. Uses CSV indicator data only.

Supports three modes:
  1. backtest — full historical backtest via backtest.py
  2. daily   — short lookback (1-2 days), email summary of what hit
  3. signals — one-shot signal evaluation (for cron / live execution)

Entry: RSI < 35 AND bollinger_bands_signal == 1 AND close < bbands_lower_20
Exit:  hold >= 3 days OR RSI > 55

Usage:
    # Full backtest
    python run_oversold_bounce.py \\
        --data-path 'all_data_*.csv' \\
        --use-all-symbols \\
        --start-date 2024-07-01 --end-date 2025-12-31

    # Daily report
    python run_oversold_bounce.py \\
        --data-path 'all_data_*.csv' \\
        --use-all-symbols \\
        --mode daily --lookback-days 2 --summary-only

    # Signal scan
    python run_oversold_bounce.py \\
        --data-path 'all_data_*.csv' \\
        --use-all-symbols \\
        --mode signals --output-signals signals/oversold_bounce.csv
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta

import pandas as pd

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

from oversold_bounce_strategy import OversoldBounceStrategy


def get_available_symbols(data_path):
    """Get list of symbols available in CSV data."""
    import glob as glob_mod
    if '*' in data_path:
        files = glob_mod.glob(data_path)
        if not files:
            return set()
        df = pd.concat([pd.read_csv(f, usecols=['symbol']) for f in files], ignore_index=True)
    elif os.path.isfile(data_path):
        df = pd.read_csv(data_path, usecols=['symbol'])
    else:
        return set()
    return set(df['symbol'].unique())


# ================================================================== #
# MODE 1: FULL BACKTEST
# ================================================================== #

def run_backtest(symbols, start_date, end_date, start_cash, data_path,
                 position_size=0.1, max_hold_days=3,
                 output_trades=None, output_symbols=None, output_signals=None,
                 filters_applied=None, ranks_applied=None, notifier=None):
    """Run the full Oversold Bounce backtest."""
    data = None
    if data_path:
        data = portfolio_load_data(data_path)
        print(f"  Loaded {len(data):,} rows from {data_path}")

    account = BacktestAccount(
        account_id="oversold_bounce_backtest",
        owner_name="Oversold Bounce Backtest",
        account_values=AccountValues(start_cash, 0, 0),
        start_date=pd.to_datetime(start_date)
    )

    strategy = OversoldBounceStrategy(
        account=account,
        symbols=symbols,
        data=data,
        position_size=position_size,
        max_hold_days=max_hold_days
    )

    print("\n" + "=" * 70)
    print("OVERSOLD BOUNCE STRATEGY BACKTEST (S1)")
    print("=" * 70)
    print(f"""
Strategy Parameters:
  Entry: RSI < 35 AND bollinger_bands_signal == 1 AND close < bbands_lower_20
  Exit:  hold >= {max_hold_days} days OR RSI > 55 (early recovery)
  Position Size: {position_size * 100:.0f}% of portfolio
  Max Positions: {strategy.MAX_POSITIONS}

Backtest Setup:
  Symbols: {len(symbols)} stocks
  Date Range: {start_date} to {end_date}
  Starting Cash: ${start_cash:,.2f}
  Data: {data_path}""")

    if filters_applied:
        print(f"  Filters:       {', '.join(filters_applied)}")
    if ranks_applied:
        print(f"  Rank:          {', '.join(ranks_applied)}")
    print()
    print("=" * 70 + "\n")

    account_values = backtest(strategy, symbols, start_date, end_date, data=data)

    final_value = account_values.get_total_value()
    stats = compute_stats(strategy.trade_log, start_cash)
    print_stats(stats, start_cash, final_value)

    open_positions = strategy.get_open_positions()
    if open_positions:
        print(f"\nOpen Positions ({len(open_positions)}):")
        for symbol, pos in open_positions.items():
            print(f"  {symbol}: {pos['shares']} shares @ ${pos['entry_price']:.2f}")
        print()

    if output_trades:
        write_trade_log(strategy.trade_log, output_trades)
    if output_symbols:
        write_symbols(symbols, output_symbols)

    # Write pending signals for open positions at backtest end
    if output_signals:
        open_positions = strategy.get_open_positions()
        if open_positions:
            writer = SignalWriter(output_signals)
            for symbol, pos in open_positions.items():
                entry_str = pos['entry_date'].strftime('%Y-%m-%d') if hasattr(pos['entry_date'], 'strftime') else str(pos['entry_date'])[:10]
                writer.add(
                    action='BUY', symbol=symbol,
                    price=pos['entry_price'],
                    strategy='oversold_bounce',
                    reason=f"open position from {entry_str}",
                )
            writer.save()
            print(f"  Wrote {len(open_positions)} pending signals to {output_signals}")

    if notifier:
        total_trades = stats.get('total_trades', 0)
        total_return = stats.get('total_return_pct', 0)
        subject = (
            f"[OVERSOLD_BOUNCE] Backtest Results - "
            f"{total_trades} trades, {total_return:+.2f}% ({start_date} to {end_date})"
        )
        body = generate_summary(
            strategy, stats, start_cash, final_value,
            lookback_days=0, filters_applied=filters_applied,
            ranks_applied=ranks_applied
        )
        if notifier._send_email(subject, body):
            print(f"\nEmail sent: {subject}")

    return account_values


# ================================================================== #
# MODE 2: DAILY REPORT
# ================================================================== #

def generate_summary(strategy, stats, start_cash, final_value,
                     lookback_days, filters_applied=None, ranks_applied=None):
    """Generate a formatted text summary of oversold bounce results."""
    lines = [
        "=" * 85,
        "OVERSOLD BOUNCE STRATEGY SUMMARY (S1)",
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
    lines.append(f"Entry Rule:    RSI < 35, BB signal = 1, close < lower band")
    lines.append(f"Exit Rule:     hold >= 3d OR RSI > 55")
    lines.append("")

    entries = [t for t in strategy.trade_log if 'exit_date' in t and t.get('exit_date')]
    open_positions = strategy.get_open_positions()

    lines.append(f"COMPLETED TRADES ({len(entries)}):")
    lines.append("-" * 85)
    if entries:
        lines.append(f"{'Symbol':<8} {'Entry':<12} {'Exit':<12} {'Entry$':>10} {'Exit$':>10} {'Return':>8} {'Hold':>6} {'Reason':<20}")
        lines.append("-" * 85)
        for t in entries:
            ret_str = f"{t['return_pct']:+.2f}%"
            hold_str = f"{t['hold_days']}d"
            reason = t.get('exit_reason', '')
            lines.append(
                f"{t['symbol']:<8} {t['entry_date']:<12} {t['exit_date']:<12} "
                f"${t['entry_price']:>8.2f} ${t['exit_price']:>8.2f} {ret_str:>8} {hold_str:>6} {reason:<20}"
            )
    else:
        lines.append("  (none)")

    lines.append("")

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


def run_daily_report(symbols, data_path, position_size=0.1,
                     lookback_days=2, start_cash=100000, notifier=None,
                     filters_applied=None, ranks_applied=None):
    """Run a short lookback backtest and email a summary of what hit."""
    end_date = date.today().strftime('%Y-%m-%d')
    start_date = (date.today() - timedelta(days=lookback_days + 5)).strftime('%Y-%m-%d')

    data = None
    if data_path:
        data = portfolio_load_data(data_path)
        print(f"  Loaded {len(data):,} rows from {data_path}")

    account = BacktestAccount(
        account_id="oversold_bounce_daily",
        owner_name="Oversold Bounce Daily",
        account_values=AccountValues(start_cash, 0, 0),
        start_date=pd.to_datetime(start_date)
    )

    strategy = OversoldBounceStrategy(
        account=account,
        symbols=symbols,
        data=data,
        position_size=position_size
    )

    account_values = backtest(strategy, symbols, start_date, end_date, data=data)

    final_value = account_values.get_total_value()
    stats = compute_stats(strategy.trade_log, start_cash)

    summary = generate_summary(
        strategy, stats, start_cash, final_value,
        lookback_days, filters_applied, ranks_applied
    )

    print(summary)

    if notifier:
        trade_count = len(strategy.trade_log)
        open_count = len(strategy.get_open_positions())
        subject = (
            f"[OVERSOLD_BOUNCE] Signals Summary - "
            f"{trade_count} trade(s), {open_count} open ({date.today()})"
        )
        if notifier._send_email(subject, summary):
            print(f"\nEmail sent: {subject}")
        else:
            print("\nWarning: Failed to send email")

    return strategy


# ================================================================== #
# MODE 3: ONE-SHOT SIGNAL GENERATION
# ================================================================== #

def run_signal_eval(symbols, data_path, position_size=0.1,
                    output_signals=None, prices_source='live', notifier=None):
    """Evaluate strategy once for today and write signal CSV."""
    account = BacktestAccount(
        account_id="oversold_bounce_signal_runner",
        owner_name="Oversold Bounce Signal Runner",
        account_values=AccountValues(100000, 0, 0),
        start_date=pd.to_datetime(date.today())
    )

    # Load indicator data
    data = None
    if data_path:
        data = portfolio_load_data(data_path)

    strategy = OversoldBounceStrategy(
        account=account,
        symbols=symbols,
        data=data,
        position_size=position_size,
    )

    # Get current prices
    if isinstance(prices_source, str) and os.path.exists(prices_source):
        current_prices = pd.read_csv(prices_source)
    elif data is not None:
        data['date'] = pd.to_datetime(data['date'])
        latest_date = data['date'].max()
        latest = data[data['date'] == latest_date].copy()
        if 'open' not in latest.columns:
            latest['open'] = latest['adjusted_close']
        current_prices = latest[['symbol', 'open']].copy()
    else:
        from api_data.rt_utils import get_stock_quote
        rows = []
        for sym in symbols:
            try:
                quote = get_stock_quote(sym)
                if quote and 'price' in quote:
                    rows.append({'symbol': sym, 'open': quote['price']})
            except Exception as e:
                print(f"  Warning: Could not get price for {sym}: {e}")
        current_prices = pd.DataFrame(rows)

    if current_prices.empty:
        print("No prices available, cannot evaluate.")
        return []

    today = date.today()
    print(f"\n[{strategy.STRATEGY_NAME}] Signal evaluation for {today}")
    print(f"  Symbols: {len(symbols)}")
    print(f"  Output: {output_signals or '(print only)'}\n")

    signals = strategy.run_signals(
        current_prices=current_prices,
        trade_date=today,
        output_path=output_signals,
    )

    if notifier:
        subject = (
            f"[OVERSOLD_BOUNCE] Signal Scan - "
            f"{len(signals)} signal(s) ({today})"
        )
        lines = [f"Oversold Bounce Signal Scan: {today}", f"Symbols: {len(symbols)}", ""]
        if signals:
            for s in signals:
                lines.append(f"  BUY {s['symbol']} @ ${s['price']:.2f}: {s['reason']}")
        else:
            lines.append("  No signals (hold/do nothing)")
        if notifier._send_email(subject, "\n".join(lines)):
            print(f"\nEmail sent: {subject}")

    return signals


# ================================================================== #
# MAIN
# ================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Oversold Bounce Strategy (S1) — pure technical, no ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full backtest
  python run_oversold_bounce.py --data-path 'all_data_*.csv' \\
      --use-all-symbols --start-date 2024-07-01 --end-date 2025-12-31

  # Daily report
  python run_oversold_bounce.py --data-path 'all_data_*.csv' \\
      --use-all-symbols --mode daily --lookback-days 2 --summary-only

  # Signal scan
  python run_oversold_bounce.py --data-path 'all_data_*.csv' \\
      --use-all-symbols --mode signals --output-signals signals/oversold_bounce.csv
        """
    )

    # --- Mode ---
    parser.add_argument("--mode", type=str, default="backtest",
                        choices=["backtest", "daily", "signals"],
                        help="Strategy mode (default: backtest)")

    # --- Data ---
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to price/indicator CSV(s) (supports globs)")
    parser.add_argument("--symbols", type=str, nargs="+",
                        help="Symbols to trade (e.g., AAPL GOOGL MSFT)")
    parser.add_argument("--symbols-file", type=str, default=None,
                        help="CSV file with symbol list")
    parser.add_argument("--use-all-symbols", action="store_true",
                        help="Use all symbols available in data")

    # --- Backtest-specific ---
    parser.add_argument("--start-date", type=str, default=None,
                        help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--start-cash", type=float, default=100000,
                        help="Initial cash balance (default: 100000)")
    parser.add_argument("--position-size", type=float, default=0.1,
                        help="Position size as fraction of portfolio (default: 0.1)")
    parser.add_argument("--max-hold-days", type=int, default=3,
                        help="Max days to hold (default: 3)")

    # --- Daily/signals ---
    parser.add_argument("--lookback-days", type=int, default=2,
                        help="Days to look back for daily report (default: 2)")
    parser.add_argument("--no-notify", action="store_true",
                        help="Skip sending email notification")
    parser.add_argument("--summary-only", action="store_true",
                        help="Send summary email (daily mode)")

    # --- Portfolio ranker ---
    parser.add_argument("--portfolio-data", type=str, default=None,
                        help="Path to data CSV for portfolio ranking (default: --data-path)")
    parser.add_argument("--watchlist", type=str, default=None,
                        help="Path to watchlist CSV file")
    parser.add_argument("--watchlist-mode", type=str, default='sort',
                        choices=['sort', 'filter'],
                        help="Watchlist mode (default: sort)")
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
                        help="Keep top K symbols ranked by Sharpe ratio")
    parser.add_argument("--sort-sharpe", action="store_true",
                        help="Sort symbols by Sharpe ratio")

    # --- Output ---
    parser.add_argument("--output-trades", type=str, default=None,
                        help="Path to write trade log CSV")
    parser.add_argument("--output-symbols", type=str, default=None,
                        help="Path to write filtered symbol list CSV")
    parser.add_argument("--output-signals", type=str, default=None,
                        help="Path to write pending signal CSV")
    parser.add_argument("--prices-csv", type=str, default=None,
                        help="CSV with current prices (signals mode)")

    args = parser.parse_args()

    # Determine symbols
    if args.use_all_symbols:
        if not args.data_path:
            print("Error: --data-path required with --use-all-symbols")
            sys.exit(1)
        symbols = get_available_symbols(args.data_path)
        if not symbols:
            print(f"Error: No symbols found in {args.data_path}")
            sys.exit(1)
        print(f"Using all {len(symbols)} symbols from data")
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

    # Build filter/rank metadata
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

    if args.output_symbols:
        write_symbols(symbols, args.output_symbols)

    # Notifier
    notifier = None if args.no_notify else TradeNotifier()

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
            data_path=args.data_path,
            position_size=args.position_size,
            max_hold_days=args.max_hold_days,
            output_trades=args.output_trades,
            output_signals=args.output_signals,
            filters_applied=filters_applied,
            ranks_applied=ranks_applied,
            notifier=notifier,
        )

    elif args.mode == "daily":
        run_daily_report(
            symbols=symbols,
            data_path=args.data_path,
            position_size=args.position_size,
            lookback_days=args.lookback_days,
            start_cash=args.start_cash,
            notifier=notifier,
            filters_applied=filters_applied,
            ranks_applied=ranks_applied,
        )

    elif args.mode == "signals":
        prices_source = args.prices_csv or args.data_path or 'live'
        run_signal_eval(
            symbols=symbols,
            data_path=args.data_path,
            position_size=args.position_size,
            output_signals=args.output_signals,
            prices_source=prices_source,
            notifier=notifier,
        )