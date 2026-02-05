#!/usr/bin/env python
"""
Runner script for Regression Momentum Strategy backtest.

Uses the bar_fly_trading backtest framework with stockformer predictions.

Usage:
    python run_regression_momentum.py \
        --symbols AAPL GOOGL MSFT NVDA \
        --start-date 2024-07-01 \
        --end-date 2024-12-31 \
        --start-cash 100000

    # Or use all symbols from predictions:
    python run_regression_momentum.py \
        --start-date 2024-07-01 \
        --end-date 2024-12-31 \
        --start-cash 100000 \
        --use-all-symbols

    # Use watchlist to filter symbols:
    python run_regression_momentum.py \
        --use-all-symbols \
        --watchlist api_data/watchlist.csv \
        --watchlist-mode filter \
        --start-date 2024-07-01 \
        --end-date 2024-12-31

    # Use watchlist to sort output (keep all, but prioritize watchlist order):
    python run_regression_momentum.py \
        --use-all-symbols \
        --watchlist api_data/watchlist.csv \
        --watchlist-mode sort \
        --start-date 2024-07-01 \
        --end-date 2024-12-31
"""

import argparse
import os
import sys

import pandas as pd

# Add bar_fly_trading to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bar_fly_trading')))

from account.account_values import AccountValues
from account.backtest_account import BacktestAccount
from api_data.storage import connect_database
from backtest import backtest

from regression_momentum_strategy import RegressionMomentumStrategy


# Default paths
DEFAULT_PREDICTIONS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'mlr', 'stockformer', 'output', 'predictions')
)


def load_watchlist(watchlist_path):
    """
    Load watchlist from CSV file and return ordered list of symbols.

    Args:
        watchlist_path: Path to CSV file with 'Symbol' column

    Returns:
        List of symbols in watchlist order
    """
    if not os.path.exists(watchlist_path):
        print(f"Warning: Watchlist file not found: {watchlist_path}")
        return []

    df = pd.read_csv(watchlist_path)

    # Handle different column name conventions
    symbol_col = None
    for col in ['Symbol', 'symbol', 'SYMBOL', 'ticker', 'Ticker', 'TICKER']:
        if col in df.columns:
            symbol_col = col
            break

    if symbol_col is None:
        print(f"Warning: No symbol column found in watchlist. Expected 'Symbol' or 'ticker'")
        return []

    return df[symbol_col].tolist()


def apply_watchlist(symbols, watchlist, mode='sort'):
    """
    Apply watchlist to symbols set.

    Args:
        symbols: Set of symbols to trade
        watchlist: Ordered list of watchlist symbols
        mode: 'sort' or 'filter'
            - sort: Keep all symbols, but order by watchlist (watchlist first, then others)
            - filter: Only keep symbols that are in watchlist, ordered by watchlist

    Returns:
        Ordered list of symbols
    """
    if not watchlist:
        return list(symbols)

    watchlist_set = set(watchlist)
    symbols_set = set(symbols)

    if mode == 'filter':
        # Only keep symbols in watchlist, maintain watchlist order
        result = [s for s in watchlist if s in symbols_set]
        print(f"Watchlist filter: {len(symbols)} symbols -> {len(result)} symbols")
        return result
    else:  # sort mode
        # Watchlist symbols first (in order), then remaining symbols alphabetically
        in_watchlist = [s for s in watchlist if s in symbols_set]
        not_in_watchlist = sorted(symbols_set - watchlist_set)
        result = in_watchlist + not_in_watchlist
        print(f"Watchlist sort: {len(in_watchlist)} watchlist symbols first, {len(not_in_watchlist)} others")
        return result


def get_available_symbols(predictions_dir):
    """Get list of symbols available in predictions."""
    pred_file = os.path.join(predictions_dir, 'pred_reg_3d.csv')
    if not os.path.exists(pred_file):
        return set()

    df = pd.read_csv(pred_file, usecols=['ticker'] if 'ticker' in pd.read_csv(pred_file, nrows=1).columns else ['symbol'])
    col = 'ticker' if 'ticker' in df.columns else 'symbol'
    return set(df[col].unique())


def run_backtest(symbols, start_date, end_date, start_cash, predictions_dir, db='local', position_size=0.1):
    """
    Run the Regression Momentum backtest.

    Args:
        symbols: Set of symbols to trade
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        start_cash: Initial cash balance
        predictions_dir: Path to predictions directory
        db: Database to use ('local' or 'remote')
        position_size: Fraction of portfolio per position

    Returns:
        Final AccountValues
    """
    # Connect to database
    connect_database(db)

    # Create account
    account = BacktestAccount(
        account_id="regression_momentum_backtest",
        account_name="Regression Momentum Backtest",
        account_values=AccountValues(start_cash, 0, 0),
        start_date=pd.to_datetime(start_date)
    )

    # Create strategy
    strategy = RegressionMomentumStrategy(
        account=account,
        symbols=symbols,
        predictions_dir=predictions_dir,
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
  Predictions: {predictions_dir}
""")
    print("=" * 70 + "\n")

    # Run backtest
    account_values = backtest(strategy, symbols, start_date, end_date)

    # Print results
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    final_value = account_values.cash + account_values.stock_value + account_values.options_value
    total_return = (final_value - start_cash) / start_cash * 100

    print(f"""
Final Account Values:
  Cash:          ${account_values.cash:,.2f}
  Stock Value:   ${account_values.stock_value:,.2f}
  Options Value: ${account_values.options_value:,.2f}
  ─────────────────────────────────
  Total Value:   ${final_value:,.2f}

Performance:
  Starting:      ${start_cash:,.2f}
  Ending:        ${final_value:,.2f}
  Total Return:  {total_return:+.2f}%

Open Positions: {len(strategy.get_open_positions())}
""")

    # List open positions if any
    open_positions = strategy.get_open_positions()
    if open_positions:
        print("Open Positions:")
        for symbol, pos in open_positions.items():
            print(f"  {symbol}: {pos['shares']} shares @ ${pos['entry_price']:.2f}")

    print("=" * 70 + "\n")

    return account_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Regression Momentum Strategy Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest specific symbols
  python run_regression_momentum.py --symbols AAPL NVDA --start-date 2024-07-01 --end-date 2024-12-31

  # Backtest all available symbols
  python run_regression_momentum.py --use-all-symbols --start-date 2024-07-01 --end-date 2024-12-31
        """
    )

    parser.add_argument("--symbols", type=str, nargs="+",
                        help="Symbols to trade (e.g., AAPL GOOGL MSFT)")
    parser.add_argument("--use-all-symbols", action="store_true",
                        help="Use all symbols available in predictions")
    parser.add_argument("--start-date", type=str, required=True,
                        help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True,
                        help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--start-cash", type=float, default=100000,
                        help="Initial cash balance (default: 100000)")
    parser.add_argument("--predictions-dir", type=str, default=DEFAULT_PREDICTIONS_DIR,
                        help="Directory containing prediction CSV files")
    parser.add_argument("--position-size", type=float, default=0.1,
                        help="Position size as fraction of portfolio (default: 0.1 = 10%%)")
    parser.add_argument("--db", type=str, default='local', choices=['local', 'remote'],
                        help="Database to use (default: local)")
    parser.add_argument("--watchlist", type=str, default=None,
                        help="Path to watchlist CSV file (optional)")
    parser.add_argument("--watchlist-mode", type=str, default='sort', choices=['sort', 'filter'],
                        help="Watchlist mode: 'sort' (default) keeps all symbols but orders by watchlist, "
                             "'filter' only keeps symbols in watchlist")

    args = parser.parse_args()

    # Determine symbols
    if args.use_all_symbols:
        symbols = get_available_symbols(args.predictions_dir)
        if not symbols:
            print(f"Error: No symbols found in {args.predictions_dir}")
            sys.exit(1)
        print(f"Using all {len(symbols)} symbols from predictions")
    elif args.symbols:
        symbols = set(args.symbols)
    else:
        print("Error: Must specify --symbols or --use-all-symbols")
        sys.exit(1)

    # Apply watchlist if provided
    if args.watchlist:
        watchlist = load_watchlist(args.watchlist)
        if watchlist:
            symbols = set(apply_watchlist(symbols, watchlist, args.watchlist_mode))
            print(f"After watchlist ({args.watchlist_mode}): {len(symbols)} symbols")

    # Run backtest
    run_backtest(
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        start_cash=args.start_cash,
        predictions_dir=args.predictions_dir,
        db=args.db,
        position_size=args.position_size
    )