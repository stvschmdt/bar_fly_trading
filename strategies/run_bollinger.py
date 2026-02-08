#!/usr/bin/env python
"""
Runner script for Bollinger Band Backtest Strategy.

Supports three modes:
  1. backtest — full historical backtest via backtest.py with per-symbol P&L, win rate, Sharpe
  2. daily   — short lookback (1-2 days), email summary of unique symbols that triggered
  3. signals — one-shot signal evaluation for cron / live execution

Uses the bar_fly_trading backtest framework with Bollinger band crossover + RSI signals.
Portfolio filtering/ranking via portfolio.py is applied before the backtest
to narrow the symbol universe.

Usage:
    # Full backtest from Jan 2024
    python strategies/run_bollinger.py \
        --data-path 'all_data_*.csv' \
        --start-date 2024-01-01 --end-date 2025-12-31 \
        --watchlist api_data/watchlist.csv --watchlist-mode filter

    # Daily report (morning email of what hit in past 2 days)
    python strategies/run_bollinger.py \
        --data-path 'all_data_*.csv' \
        --mode daily --lookback-days 2 \
        --watchlist api_data/watchlist.csv --watchlist-mode filter \
        --summary-only

    # One-shot signal scan (for cron / live execution)
    python strategies/run_bollinger.py \
        --data-path 'all_data_*.csv' \
        --mode signals --symbols AAPL NVDA MSFT \
        --output-signals signals/pending_orders.csv

    # Portfolio filtering: price > $25, top 15 by Sharpe
    python strategies/run_bollinger.py \
        --data-path 'all_data_*.csv' \
        --start-date 2024-01-01 --end-date 2025-12-31 \
        --price-above 25 --top-k-sharpe 15
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

from bollinger_backtest_strategy import BollingerBacktestStrategy


# ================================================================== #
# MODE 1: FULL BACKTEST
# ================================================================== #

def run_backtest(symbols, start_date, end_date, start_cash, data_path,
                 position_size=0.05, max_hold_days=20,
                 output_trades=None, output_symbols=None, output_signals=None,
                 filters_applied=None, ranks_applied=None,
                 notifier=None):
    """Run the full Bollinger Band backtest."""
    # Load price data from CSV
    data = None
    if data_path:
        data = portfolio_load_data(data_path)
        print(f"  Loaded {len(data):,} rows from {data_path}")

    # Create account
    account = BacktestAccount(
        account_id="bollinger_backtest",
        owner_name="Bollinger Band Backtest",
        account_values=AccountValues(start_cash, 0, 0),
        start_date=pd.to_datetime(start_date)
    )

    # Create strategy — pass data so it can load indicator values (BB, RSI)
    strategy = BollingerBacktestStrategy(
        account=account,
        symbols=symbols,
        data=data,
        position_size=position_size,
        max_hold_days=max_hold_days,
        end_date=end_date,
    )

    # Print header
    print("\n" + "=" * 70)
    print("BOLLINGER BAND STRATEGY BACKTEST")
    print("=" * 70)
    print(f"""
Strategy Parameters:
  Entry: Price crosses below lower BB AND RSI <= {strategy.RSI_BUY_MAX}
  Exit:  Price reaches middle BB OR RSI > 70 OR hold >= {max_hold_days} days
  Position Size: {position_size * 100:.0f}% of portfolio

Backtest Setup:
  Symbols: {len(symbols)} stocks
  Date Range: {start_date} to {end_date}
  Starting Cash: ${start_cash:,.2f}""")

    if filters_applied:
        print(f"  Filters:       {', '.join(filters_applied)}")
    if ranks_applied:
        print(f"  Rank:          {', '.join(ranks_applied)}")
    print()
    print("=" * 70 + "\n")

    # Run backtest (positions are force-closed at end by backtest.py)
    account_values = backtest(strategy, symbols, start_date, end_date, data=data)

    # Compute and print stats
    final_value = account_values.get_total_value()
    stats = compute_stats(strategy.trade_log, start_cash)
    print_stats(stats, start_cash, final_value)

    # Format stats text for email
    summary_text = format_stats_text(stats, start_cash, final_value,
                                     strategy, filters_applied, ranks_applied,
                                     start_date, end_date)

    # Write output files
    if output_trades:
        write_trade_log(strategy.trade_log, output_trades)
    if output_symbols:
        write_symbols(symbols, output_symbols)

    # Write pending signals for live execution bridge
    if output_signals and strategy.trade_log:
        writer = SignalWriter(output_signals)
        last_date = max(t['entry_date'] for t in strategy.trade_log)
        for t in strategy.trade_log:
            if t['entry_date'] == last_date:
                writer.add(
                    action='BUY',
                    symbol=t['symbol'],
                    price=t['entry_price'],
                    strategy='bollinger',
                    reason=f"bollinger entry {last_date}",
                )
        writer.save()

    # Send email
    if notifier:
        trade_count = stats['total_trades']
        subject = (
            f"Bollinger Band (Backtest) | "
            f"{trade_count} trades, {stats['win_rate']:.1f}% win rate, "
            f"Sharpe {stats['sharpe_ratio']:.2f}"
        )
        if notifier._send_email(subject, summary_text):
            print(f"\nEmail sent: {subject}")
        else:
            print("\nWarning: Failed to send email")

    return account_values


# ================================================================== #
# MODE 2: DAILY REPORT (scan for crossovers, no backtest)
# ================================================================== #

def scan_crossovers(data, symbols, lookback_days=2):
    """
    Scan recent data for Bollinger band crossover signals.

    No backtest, no account, no position tracking — just checks which
    symbols crossed their bands in the last N trading days.

    Returns:
        List of dicts: {symbol, signal, date, close, bb_lower, bb_upper, rsi}
    """
    RSI_BUY_MAX = 40
    RSI_SELL_MIN = 60

    data = data[data['symbol'].isin(symbols)].copy()
    data['date'] = pd.to_datetime(data['date'])

    signals = []

    for symbol in symbols:
        sym_data = data[data['symbol'] == symbol].sort_values('date')
        if len(sym_data) < 2:
            continue

        # Last N+1 trading days (need prev day for crossover check)
        recent = sym_data.tail(lookback_days + 1)
        if len(recent) < 2:
            continue

        for idx in range(1, len(recent)):
            today = recent.iloc[idx]
            prev = recent.iloc[idx - 1]

            close = today['adjusted_close']
            bb_lower = today['bbands_lower_20']
            bb_upper = today['bbands_upper_20']
            bb_middle = today.get('bbands_middle_20', (bb_lower + bb_upper) / 2)
            prev_close = prev['adjusted_close']
            prev_bb_lower = prev['bbands_lower_20']
            prev_bb_upper = prev['bbands_upper_20']

            if pd.isna(bb_lower) or pd.isna(bb_upper):
                continue

            rsi = today.get('rsi_14', None)
            if pd.isna(rsi):
                rsi = None

            # BUY: price crossed below lower band
            if close <= bb_lower and prev_close > prev_bb_lower:
                if rsi is not None and rsi > RSI_BUY_MAX:
                    continue
                signals.append({
                    'symbol': symbol,
                    'signal': 'BUY',
                    'date': str(today['date'].date()),
                    'close': close,
                    'bb_lower': bb_lower,
                    'bb_upper': bb_upper,
                    'bb_middle': bb_middle,
                    'rsi': rsi,
                })

            # SELL: price crossed above upper band
            elif close >= bb_upper and prev_close < prev_bb_upper:
                if rsi is not None and rsi < RSI_SELL_MIN:
                    continue
                signals.append({
                    'symbol': symbol,
                    'signal': 'SELL',
                    'date': str(today['date'].date()),
                    'close': close,
                    'bb_lower': bb_lower,
                    'bb_upper': bb_upper,
                    'bb_middle': bb_middle,
                    'rsi': rsi,
                })

    return signals


def run_daily_report(symbols, data_path, lookback_days=2, notifier=None,
                     filters_applied=None, ranks_applied=None, **_ignored):
    """
    Scan recent data for crossovers and report which symbols triggered.

    No backtest — just loads data, checks last N days, lists what hit.
    """
    # Load price data
    data = portfolio_load_data(data_path)
    print(f"  Loaded {len(data):,} rows from {data_path}")

    # Scan for crossovers
    signals = scan_crossovers(data, symbols, lookback_days=lookback_days)

    # Generate summary
    summary = generate_daily_summary(signals, symbols, lookback_days,
                                     filters_applied, ranks_applied)
    print(summary)

    # Email
    if notifier:
        buy_count = sum(1 for s in signals if s['signal'] == 'BUY')
        sell_count = sum(1 for s in signals if s['signal'] == 'SELL')
        subject = (
            f"Bollinger Band (Daily Watch) | "
            f"{len(signals)} signal(s): {buy_count} BUY, {sell_count} SELL "
            f"({date.today()})"
        )
        if notifier._send_email(subject, summary):
            print(f"\nEmail sent: {subject}")
        else:
            print("\nWarning: Failed to send email")

    return signals


def generate_daily_summary(signals, symbols, lookback_days,
                           filters_applied=None, ranks_applied=None):
    """Generate a daily summary of crossover signals."""
    lines = [
        "=" * 85,
        "BOLLINGER BAND STRATEGY - DAILY SCAN",
        "=" * 85,
        f"Scan Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Lookback:      {lookback_days} trading day(s)",
        f"Universe:      {len(symbols)} symbols scanned",
    ]

    if filters_applied:
        lines.append(f"Filters:       {', '.join(filters_applied)}")
    if ranks_applied:
        lines.append(f"Rank:          {', '.join(ranks_applied)}")
    lines.append("")

    buy_signals = [s for s in signals if s['signal'] == 'BUY']
    sell_signals = [s for s in signals if s['signal'] == 'SELL']

    lines.append(f"BUY signals ({len(buy_signals)}):")
    lines.append("-" * 85)
    if buy_signals:
        lines.append(f"{'Symbol':<8} {'Date':<12} {'Close':>10} {'BB Lower':>10} {'BB Middle':>10} {'RSI':>6}")
        lines.append("-" * 85)
        for s in sorted(buy_signals, key=lambda x: x['symbol']):
            rsi_str = f"{s['rsi']:.1f}" if s['rsi'] is not None else "N/A"
            lines.append(
                f"{s['symbol']:<8} {s['date']:<12} ${s['close']:>8.2f} "
                f"${s['bb_lower']:>8.2f} ${s['bb_middle']:>8.2f} {rsi_str:>6}"
            )
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append(f"SELL signals ({len(sell_signals)}):")
    lines.append("-" * 85)
    if sell_signals:
        lines.append(f"{'Symbol':<8} {'Date':<12} {'Close':>10} {'BB Upper':>10} {'BB Middle':>10} {'RSI':>6}")
        lines.append("-" * 85)
        for s in sorted(sell_signals, key=lambda x: x['symbol']):
            rsi_str = f"{s['rsi']:.1f}" if s['rsi'] is not None else "N/A"
            lines.append(
                f"{s['symbol']:<8} {s['date']:<12} ${s['close']:>8.2f} "
                f"${s['bb_upper']:>8.2f} ${s['bb_middle']:>8.2f} {rsi_str:>6}"
            )
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("=" * 85)

    return "\n".join(lines)


def format_stats_text(stats, start_cash, final_value, strategy,
                      filters_applied=None, ranks_applied=None,
                      start_date=None, end_date=None):
    """Format backtest stats as plain text for email."""
    total_return = (final_value - start_cash) / start_cash * 100

    lines = [
        "=" * 70,
        "BACKTEST RESULTS FOR STRATEGY - BOLLINGER BAND",
        "=" * 70,
        "",
    ]

    if start_date and end_date:
        lines.append(f"Date Range:    {start_date} to {end_date}")
    if filters_applied:
        lines.append(f"Filters:       {', '.join(filters_applied)}")
    if ranks_applied:
        lines.append(f"Rank:          {', '.join(ranks_applied)}")

    lines.extend([
        "",
        f"Portfolio:",
        f"  Starting:      ${start_cash:,.2f}",
        f"  Ending:        ${final_value:,.2f}",
        f"  Total Return:  {total_return:+.2f}%",
        "",
        f"Trade Summary:",
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
        for symbol, pos in open_positions.items():
            lines.append(f"  {symbol}: {pos['shares']} shares @ ${pos['entry_price']:.2f}")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# ================================================================== #
# MODE 3: ONE-SHOT SIGNAL GENERATION (for cron / live execution)
# ================================================================== #

def run_signal_eval(symbols, data_path, position_size=0.05,
                    output_signals=None, prices_source='live', notifier=None):
    """
    Evaluate strategy once for today and write signal CSV.

    Args:
        symbols: Set of symbols to evaluate
        data_path: Path to price data CSV(s) with indicator columns
        position_size: Position size fraction
        output_signals: Path to write signal CSV (None = print only)
        prices_source: 'live' to fetch from rt_utils, or path to CSV
    """
    account = BacktestAccount(
        account_id="bollinger_signal_runner",
        owner_name="Bollinger Signal Runner",
        account_values=AccountValues(100000, 0, 0),
        start_date=pd.to_datetime(date.today())
    )

    # Load indicator data from CSV
    data = portfolio_load_data(data_path) if data_path else None

    strategy = BollingerBacktestStrategy(
        account=account,
        symbols=symbols,
        data=data,
        position_size=position_size,
    )

    # Get current prices
    if isinstance(prices_source, str) and os.path.exists(prices_source):
        current_prices = pd.read_csv(prices_source)
    elif data is not None:
        # Use latest prices from the loaded data
        data['date'] = pd.to_datetime(data['date'])
        latest_date = data['date'].max()
        current_prices = data[data['date'] == latest_date][['symbol', 'open']].copy()
        if 'open' not in current_prices.columns and 'adjusted_close' in data.columns:
            current_prices['open'] = data[data['date'] == latest_date]['adjusted_close']
    else:
        # Fetch live prices via rt_utils
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

    # Email signal results
    if notifier:
        subject = (
            f"[BOLLINGER] Signal Scan - "
            f"{len(signals)} signal(s) ({today})"
        )
        lines = [f"Bollinger Band Signal Scan: {today}", f"Symbols: {len(symbols)}", ""]
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
        description="Run Bollinger Band Strategy (backtest or daily report)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # --- Mode ---
    parser.add_argument("--mode", type=str, default="backtest",
                        choices=["backtest", "daily", "signals"],
                        help="'backtest' for historical test, 'daily' for short lookback + email, 'signals' for one-shot evaluation (default: backtest)")

    # --- Data ---
    parser.add_argument("--predictions", type=str, default=None,
                        help="Path to merged_predictions.csv (not used by bollinger, accepted for CLI parity)")
    parser.add_argument("--symbols", type=str, nargs="+",
                        help="Symbols to trade (e.g., AAPL GOOGL MSFT)")
    parser.add_argument("--symbols-file", type=str, default=None,
                        help="CSV file with symbol list (output of portfolio or prior backtest)")
    parser.add_argument("--use-all-symbols", action="store_true",
                        help="Use all symbols available in data")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to price data CSV(s) (supports globs, e.g. 'all_data_*.csv')")

    # --- Backtest-specific ---
    parser.add_argument("--start-date", type=str, default=None,
                        help="Backtest start date (YYYY-MM-DD) — required for backtest mode")
    parser.add_argument("--end-date", type=str, default=None,
                        help="Backtest end date (YYYY-MM-DD) — required for backtest mode")
    parser.add_argument("--start-cash", type=float, default=100000,
                        help="Initial cash balance (default: 100000)")
    parser.add_argument("--position-size", type=float, default=0.05,
                        help="Position size as fraction of portfolio (default: 0.05 = 5%%)")
    parser.add_argument("--max-hold-days", type=int, default=20,
                        help="Max days to hold a position (default: 20)")

    # --- Daily report-specific ---
    parser.add_argument("--lookback-days", type=int, default=2,
                        help="Days to look back for daily report mode (default: 2)")
    parser.add_argument("--no-notify", action="store_true",
                        help="Skip sending email notification")
    parser.add_argument("--summary-only", action="store_true",
                        help="Send summary email (daily mode)")

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
                        help="Column name to filter/rank on (e.g. rsi_14, beta)")
    parser.add_argument("--filter-above", type=float, default=None,
                        help="Min value for --filter-field (inclusive)")
    parser.add_argument("--filter-below", type=float, default=None,
                        help="Max value for --filter-field (inclusive)")
    parser.add_argument("--top-k-sharpe", type=int, default=None,
                        help="Keep top K symbols ranked by Sharpe ratio")
    parser.add_argument("--sort-sharpe", action="store_true",
                        help="Sort symbols by Sharpe ratio (no cutoff)")

    # --- Output files ---
    parser.add_argument("--output-trades", type=str, default=None,
                        help="Path to write trade log CSV")
    parser.add_argument("--output-symbols", type=str, default=None,
                        help="Path to write filtered symbol list CSV")
    parser.add_argument("--output-signals", type=str, default=None,
                        help="Path to write pending signal CSV for live execution")

    # --- Signal mode ---
    parser.add_argument("--prices-csv", type=str, default=None,
                        help="CSV with current prices (signals mode, optional)")

    args = parser.parse_args()

    # Validate: data-path is required for all modes (bollinger uses CSV indicators)
    if not args.data_path:
        print("Error: --data-path is required for bollinger strategy")
        sys.exit(1)

    # Default portfolio-data to data-path
    if not args.portfolio_data and args.data_path:
        args.portfolio_data = args.data_path

    # Determine symbols
    if args.symbols:
        symbols = set(args.symbols)
        print(f"Using {len(symbols)} specified symbols")
    elif args.symbols_file:
        symbols = set(read_symbols(args.symbols_file))
        print(f"Loaded {len(symbols)} symbols from {args.symbols_file}")
    elif args.use_all_symbols or (not args.symbols and not args.symbols_file):
        # Default: use all symbols from data
        print(f"Loading symbols from: {args.data_path}")
        all_data = portfolio_load_data(args.data_path)
        symbols = set(all_data['symbol'].unique())
        print(f"Found {len(symbols)} symbols in data")

    # --- Portfolio ranking pipeline ---
    has_portfolio_filters = any([
        args.watchlist, args.price_above is not None, args.price_below is not None,
        args.filter_field, args.top_k_sharpe is not None, args.sort_sharpe,
    ])

    if has_portfolio_filters:
        portfolio_df = all_data  # reuse already-loaded data

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

    # Write symbol list
    if args.output_symbols:
        write_symbols(symbols, args.output_symbols)

    # Notifier for all modes
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
            output_symbols=args.output_symbols,
            output_signals=args.output_signals,
            filters_applied=filters_applied,
            ranks_applied=ranks_applied,
            notifier=notifier,
        )

    elif args.mode == "daily":

        run_daily_report(
            symbols=symbols,
            data_path=args.data_path,
            lookback_days=args.lookback_days,
            notifier=notifier,
            filters_applied=filters_applied,
            ranks_applied=ranks_applied,
        )

    elif args.mode == "signals":
        prices_source = args.prices_csv or 'live'
        run_signal_eval(
            symbols=symbols,
            data_path=args.data_path,
            position_size=args.position_size,
            output_signals=args.output_signals,
            prices_source=prices_source,
            notifier=notifier,
        )
