#!/usr/bin/env python
"""
Runner script for Regression Momentum Strategy backtest.

Uses the bar_fly_trading backtest framework with stockformer predictions.
Portfolio filtering/ranking via portfolio.py is applied before the backtest
to narrow the symbol universe.

Usage:
    python run_regression_momentum.py \
        --symbols AAPL GOOGL MSFT NVDA \
        --start-date 2024-07-01 \
        --end-date 2024-12-31 \
        --start-cash 100000

    # Or use all symbols from predictions:
    python run_regression_momentum.py \
        --use-all-symbols \
        --start-date 2024-07-01 \
        --end-date 2024-12-31

    # Portfolio filtering: price band + top 15 by Sharpe
    python run_regression_momentum.py \
        --use-all-symbols \
        --portfolio-data all_data_0.csv \
        --price-above 25 --top-k-sharpe 15 \
        --start-date 2024-07-01 \
        --end-date 2024-12-31

    # Watchlist filter + field filter (beta < 1.5)
    python run_regression_momentum.py \
        --use-all-symbols \
        --portfolio-data all_data_0.csv \
        --watchlist api_data/watchlist.csv --watchlist-mode filter \
        --filter-field beta --filter-below 1.5 \
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
from portfolio import (
    load_data as portfolio_load_data,
    load_watchlist,
    apply_watchlist,
    run_pipeline as portfolio_pipeline,
)

from regression_momentum_strategy import RegressionMomentumStrategy


# Default paths
DEFAULT_PREDICTIONS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'mlr', 'stockformer', 'output', 'predictions')
)


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

  # Portfolio filter: price > $25, top 15 by Sharpe
  python run_regression_momentum.py --use-all-symbols --portfolio-data all_data_0.csv \\
      --price-above 25 --top-k-sharpe 15 --start-date 2024-07-01 --end-date 2024-12-31

  # Watchlist + field filter (beta < 1.5)
  python run_regression_momentum.py --use-all-symbols --portfolio-data all_data_0.csv \\
      --watchlist api_data/watchlist.csv --watchlist-mode filter \\
      --filter-field beta --filter-below 1.5 --start-date 2024-07-01 --end-date 2024-12-31
        """
    )

    # --- Backtest args ---
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

    # --- Portfolio ranker args ---
    parser.add_argument("--portfolio-data", type=str, default=None,
                        help="Path to data CSV for portfolio ranking (e.g. all_data_0.csv)")
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

    # --- Portfolio ranking pipeline ---
    has_portfolio_filters = any([
        args.watchlist, args.price_above is not None, args.price_below is not None,
        args.filter_field, args.top_k_sharpe is not None,
    ])

    if has_portfolio_filters:
        # Load portfolio data (required for price/field/sharpe filters)
        portfolio_df = None
        if args.portfolio_data:
            print(f"Loading portfolio data: {args.portfolio_data}")
            portfolio_df = portfolio_load_data(args.portfolio_data)

        wl = load_watchlist(args.watchlist) if args.watchlist else None

        if portfolio_df is not None:
            # Full pipeline: watchlist + price + field + sharpe
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
            )
            symbols = set(symbols_list)
        elif wl:
            # Watchlist-only (no data CSV needed)
            symbols = set(apply_watchlist(list(symbols), wl, args.watchlist_mode))
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