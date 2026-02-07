#!/usr/bin/env python
"""
Runner for the Template Strategy.

Copy this file alongside your copied template_strategy.py and update
the import to point at your strategy class.

Supports two modes:
  1. backtest  — full historical backtest via backtest.py
  2. signals   — evaluate once for today, write signal CSV for IBKR executor

Usage:
    # Backtest with CSV data
    python run_template.py --predictions data.csv --symbols AAPL NVDA \
        --data-path 'all_data_*.csv' \
        --start-date 2024-07-01 --end-date 2024-12-31

    # Backtest, all symbols from predictions
    python run_template.py --predictions data.csv --use-all-symbols \
        --data-path 'all_data_*.csv' \
        --start-date 2024-07-01 --end-date 2024-12-31

    # Portfolio filtering: price band + top 15 by Sharpe
    python run_template.py --predictions data.csv --use-all-symbols \
        --data-path 'all_data_*.csv' \
        --price-above 25 --top-k-sharpe 15 \
        --start-date 2024-07-01 --end-date 2024-12-31

    # Live signal generation (single evaluation, writes signal CSV)
    python run_template.py --predictions data.csv --symbols AAPL NVDA \
        --mode signals --output-signals signals/pending_orders.csv
"""

import argparse
import os
import sys
from datetime import date

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
from portfolio import (
    load_data as portfolio_load_data,
    load_watchlist,
    apply_watchlist,
    run_pipeline as portfolio_pipeline,
)

# ------------------------------------------------------------------ #
# CUSTOMIZE: Change this import to your strategy class
# ------------------------------------------------------------------ #
from template_strategy import TemplateStrategy


def get_available_symbols(predictions_path):
    """Get list of symbols available in a predictions CSV."""
    df = pd.read_csv(predictions_path, nrows=1)
    col = 'ticker' if 'ticker' in df.columns else 'symbol'
    df = pd.read_csv(predictions_path, usecols=[col])
    return set(df[col].unique())


# ================================================================== #
# MODE 1: BACKTEST
# ================================================================== #

def run_backtest(symbols, start_date, end_date, start_cash, predictions_path,
                 data_path=None, position_size=0.1, output_trades=None,
                 output_symbols=None, output_signals=None,
                 filters_applied=None, ranks_applied=None):
    """Run the full backtest."""
    # Load price data from CSV (no database needed)
    data = None
    if data_path:
        data = portfolio_load_data(data_path)
        print(f"  Loaded {len(data):,} rows from {data_path}")

    account = BacktestAccount(
        account_id="template_backtest",
        owner_name="Template Strategy Backtest",
        account_values=AccountValues(start_cash, 0, 0),
        start_date=pd.to_datetime(start_date)
    )

    strategy = TemplateStrategy(
        account=account,
        symbols=symbols,
        predictions_path=predictions_path,
        position_size=position_size,
    )

    print("\n" + "=" * 70)
    print(f"{strategy.STRATEGY_NAME.upper()} STRATEGY BACKTEST")
    print("=" * 70)
    print(f"""
Backtest Setup:
  Symbols: {len(symbols)} stocks
  Date Range: {start_date} to {end_date}
  Starting Cash: ${start_cash:,.2f}
  Position Size: {position_size * 100:.0f}%
  Data: {predictions_path}""")

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

    if output_signals and strategy.trade_log:
        writer = SignalWriter(output_signals)
        last_date = max(t['entry_date'] for t in strategy.trade_log)
        for t in strategy.trade_log:
            if t['entry_date'] == last_date:
                writer.add(
                    action='BUY',
                    symbol=t['symbol'],
                    price=t['entry_price'],
                    strategy=strategy.STRATEGY_NAME,
                    reason=f"backtest entry {last_date}",
                )
        writer.save()

    return account_values


# ================================================================== #
# MODE 2: LIVE SIGNAL GENERATION
# ================================================================== #

def run_signals(symbols, predictions_path, position_size=0.1,
                output_signals=None, prices_source='live'):
    """
    Evaluate strategy once for today and write signal CSV.

    This is what you'd call from a cron job or scheduler:
        python run_template.py --mode signals --symbols AAPL NVDA \
            --predictions data.csv --output-signals signals/pending_orders.csv

    Args:
        symbols: Set of symbols to evaluate
        predictions_path: Path to data file
        position_size: Position size fraction
        output_signals: Path to write signal CSV (None = print only)
        prices_source: 'live' to fetch from rt_utils, or path to CSV
    """
    # Create a lightweight account for position sizing
    # In production, this would read actual account values from IBKR
    account = BacktestAccount(
        account_id="signal_runner",
        owner_name="Signal Runner",
        account_values=AccountValues(100000, 0, 0),  # placeholder
        start_date=pd.to_datetime(date.today())
    )

    strategy = TemplateStrategy(
        account=account,
        symbols=symbols,
        predictions_path=predictions_path,
        position_size=position_size,
    )

    # Get current prices
    if isinstance(prices_source, str) and os.path.exists(prices_source):
        current_prices = pd.read_csv(prices_source)
    else:
        # Fetch live prices via rt_utils (no database needed)
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

    return signals


# ================================================================== #
# MAIN
# ================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Template Strategy (backtest or signal generation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest with CSV data
  python run_template.py --predictions data.csv --symbols AAPL NVDA \\
      --data-path 'all_data_*.csv' --start-date 2024-07-01 --end-date 2024-12-31

  # Portfolio filter: price > $25, top 15 by Sharpe
  python run_template.py --predictions data.csv --use-all-symbols \\
      --data-path 'all_data_*.csv' --price-above 25 --top-k-sharpe 15 \\
      --start-date 2024-07-01 --end-date 2024-12-31

  # Signal generation (one-shot, for cron)
  python run_template.py --mode signals --predictions data.csv \\
      --symbols AAPL NVDA --output-signals signals/pending_orders.csv
        """
    )

    # Mode
    parser.add_argument("--mode", type=str, default="backtest",
                        choices=["backtest", "signals"],
                        help="'backtest' for historical test, 'signals' for live evaluation")

    # Data
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to data CSV your strategy needs")
    parser.add_argument("--symbols", type=str, nargs="+",
                        help="Symbols to trade")
    parser.add_argument("--symbols-file", type=str, default=None,
                        help="CSV file with symbol list")
    parser.add_argument("--use-all-symbols", action="store_true",
                        help="Use all symbols from predictions file")

    # Backtest-specific
    parser.add_argument("--start-date", type=str,
                        help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str,
                        help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--start-cash", type=float, default=100000,
                        help="Initial cash (default: 100000)")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to price data CSV(s) for backtest (supports globs, e.g. 'all_data_*.csv')")

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

    # Common
    parser.add_argument("--position-size", type=float, default=0.1,
                        help="Position size as fraction (default: 0.1 = 10%%)")

    # Output
    parser.add_argument("--output-trades", type=str, default=None,
                        help="Path to write trade log CSV (backtest mode)")
    parser.add_argument("--output-symbols", type=str, default=None,
                        help="Path to write symbol list CSV")
    parser.add_argument("--output-signals", type=str, default=None,
                        help="Path to write signal CSV")

    # Signal mode: price source
    parser.add_argument("--prices-csv", type=str, default=None,
                        help="CSV with current prices (signals mode, optional)")

    args = parser.parse_args()

    # Resolve symbols
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
            output_symbols=args.output_symbols,
            output_signals=args.output_signals,
            filters_applied=filters_applied,
            ranks_applied=ranks_applied,
        )

    elif args.mode == "signals":
        prices_source = args.prices_csv or 'live'
        run_signals(
            symbols=symbols,
            predictions_path=args.predictions,
            position_size=args.position_size,
            output_signals=args.output_signals,
            prices_source=prices_source,
        )
