#!/usr/bin/env python
"""
Runner for the Template Strategy.

Copy this file alongside your copied template_strategy.py and update
the import to point at your strategy class.

Supports two modes:
  1. backtest  — full historical backtest via backtest.py
  2. signals   — evaluate once for today, write signal CSV for IBKR executor

Usage:
    # Backtest
    python run_template.py --predictions data.csv --symbols AAPL NVDA \
        --start-date 2024-07-01 --end-date 2024-12-31

    # Backtest, all symbols from predictions
    python run_template.py --predictions data.csv --use-all-symbols \
        --start-date 2024-07-01 --end-date 2024-12-31

    # Live signal generation (single evaluation, writes signal CSV)
    python run_template.py --predictions data.csv --symbols AAPL NVDA \
        --mode signals --output-signals signals/pending_orders.csv

    # Dry run — just print what signals would be generated
    python run_template.py --predictions data.csv --symbols AAPL NVDA \
        --mode signals
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
from api_data.storage import connect_database
from backtest import backtest
from backtest_stats import compute_stats, print_stats, write_trade_log, write_symbols, read_symbols
from signal_writer import SignalWriter

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
                 db='local', position_size=0.1, output_trades=None,
                 output_symbols=None, output_signals=None):
    """Run the full backtest."""
    connect_database(db)

    account = BacktestAccount(
        account_id="template_backtest",
        account_name="Template Strategy Backtest",
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
  Data: {predictions_path}
""")
    print("=" * 70 + "\n")

    account_values = backtest(strategy, symbols, start_date, end_date)

    final_value = account_values.cash + account_values.stock_value + account_values.options_value
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
        prices_source: 'live' to fetch from API, or path to CSV
    """
    # Create a lightweight account for position sizing
    # In production, this would read actual account values from IBKR
    account = BacktestAccount(
        account_id="signal_runner",
        account_name="Signal Runner",
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
        # Fetch live prices from the database / API
        connect_database('local')
        from api_data.storage import get_last_price
        rows = []
        for sym in symbols:
            try:
                price = get_last_price(sym)
                if price:
                    rows.append({'symbol': sym, 'open': price})
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
  # Backtest
  python run_template.py --predictions data.csv --symbols AAPL NVDA \\
      --start-date 2024-07-01 --end-date 2024-12-31

  # Signal generation (one-shot, for cron)
  python run_template.py --mode signals --predictions data.csv \\
      --symbols AAPL NVDA --output-signals signals/pending_orders.csv

  # Signal generation, dry run (just print)
  python run_template.py --mode signals --predictions data.csv --symbols AAPL
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
    parser.add_argument("--db", type=str, default='local',
                        choices=['local', 'remote'],
                        help="Database to use (default: local)")

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
            db=args.db,
            position_size=args.position_size,
            output_trades=args.output_trades,
            output_symbols=args.output_symbols,
            output_signals=args.output_signals,
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
