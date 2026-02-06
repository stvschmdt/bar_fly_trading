"""
Runner script for ML Prediction Strategy backtest.

Uses existing bar_fly_trading framework with portfolio filtering support.

Usage:
    python run_backtest.py \
        --predictions ../mlr/stockformer/output/predictions/pred_bin_3d.csv \
        --symbols AAPL GOOGL MSFT \
        --start-date 2024-07-01 \
        --end-date 2024-12-31 \
        --start-cash 100000

    # Use all symbols from predictions file:
    python run_backtest.py \
        --predictions merged_predictions.csv \
        --use-all-symbols \
        --start-date 2024-07-01 \
        --end-date 2024-12-31

    # Portfolio filtering: price band + top 15 by Sharpe
    python run_backtest.py \
        --predictions merged_predictions.csv \
        --use-all-symbols \
        --portfolio-data all_data_0.csv \
        --price-above 25 --top-k-sharpe 15 \
        --start-date 2024-07-01 \
        --end-date 2024-12-31
"""

import argparse
import os
import sys

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
from portfolio import (
    load_data as portfolio_load_data,
    load_watchlist,
    apply_watchlist,
    run_pipeline as portfolio_pipeline,
)

from ml_prediction_strategy import MLPredictionStrategy


def get_available_symbols(predictions_path):
    """Get list of symbols available in a predictions CSV."""
    df = pd.read_csv(predictions_path, nrows=1)
    col = 'ticker' if 'ticker' in df.columns else 'symbol'
    df = pd.read_csv(predictions_path, usecols=[col])
    return set(df[col].unique())


def run_ml_backtest(predictions_path, symbols, start_date, end_date, start_cash,
                    db='local', position_size=0.1, output_trades=None):
    """
    Run backtest using ML predictions.

    Args:
        predictions_path: Path to stockformer predictions CSV
        symbols: Set of symbols to trade
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        start_cash: Initial cash balance
        db: Database to use ('local' or 'remote')
        position_size: Fraction of portfolio per position
        output_trades: Path to write trade log CSV (optional)

    Returns:
        Final AccountValues
    """
    # Connect to database
    connect_database(db)

    # Create account
    account = BacktestAccount(
        account_id="ml_backtest",
        account_name="ML Prediction Backtest",
        account_values=AccountValues(start_cash, 0, 0),
        start_date=pd.to_datetime(start_date)
    )

    # Create strategy with predictions
    strategy = MLPredictionStrategy(
        account=account,
        symbols=symbols,
        predictions_path=predictions_path,
        position_size=position_size
    )

    # Print header
    print(f"\n{'='*70}")
    print(f"ML PREDICTION BACKTEST")
    print(f"{'='*70}")
    print(f"""
Strategy Parameters:
  Entry: pred_class = 1 (positive return predicted)
  Exit:  pred_class = 0 (negative return predicted)
  Position Size: {position_size * 100:.0f}% of portfolio

Backtest Setup:
  Symbols: {len(symbols)} stocks
  Date Range: {start_date} to {end_date}
  Starting Cash: ${start_cash:,.2f}
  Predictions: {predictions_path}
""")
    print(f"{'='*70}\n")

    account_values = backtest(strategy, symbols, start_date, end_date)

    # Compute and print stats
    final_value = account_values.cash + account_values.stock_value + account_values.options_value
    stats = compute_stats(strategy.trade_log, start_cash)
    print_stats(stats, start_cash, final_value)

    # Open positions
    open_positions = {s: p for s, p in strategy.holdings.items() if p['shares'] > 0}
    if open_positions:
        print(f"\nOpen Positions ({len(open_positions)}):")
        for symbol, pos in open_positions.items():
            print(f"  {symbol}: {pos['shares']} shares @ ${pos['entry_price']:.2f}")
        print()

    # Write output files
    if output_trades:
        write_trade_log(strategy.trade_log, output_trades)

    return account_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ML Prediction Backtest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Backtest specific symbols
  python run_backtest.py --predictions pred_bin_3d.csv --symbols AAPL NVDA --start-date 2024-07-01 --end-date 2024-12-31

  # Backtest all symbols from predictions
  python run_backtest.py --predictions merged_predictions.csv --use-all-symbols --start-date 2024-07-01 --end-date 2024-12-31

  # Portfolio filter: price > $25, top 15 by Sharpe
  python run_backtest.py --predictions merged_predictions.csv --use-all-symbols \\
      --portfolio-data all_data_0.csv --price-above 25 --top-k-sharpe 15 \\
      --start-date 2024-07-01 --end-date 2024-12-31
        """
    )

    # --- Backtest args ---
    parser.add_argument("--predictions", type=str, required=True,
                       help="Path to stockformer predictions CSV")
    parser.add_argument("--symbols", type=str, nargs="+",
                       help="Symbols to trade (e.g., AAPL GOOGL MSFT)")
    parser.add_argument("--symbols-file", type=str, default=None,
                       help="CSV file with symbol list (output of portfolio or prior backtest)")
    parser.add_argument("--use-all-symbols", action="store_true",
                       help="Use all symbols available in predictions file")
    parser.add_argument("--start-date", type=str, required=True,
                       help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True,
                       help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--start-cash", type=float, default=100000,
                       help="Initial cash balance (default: 100000)")
    parser.add_argument("--position-size", type=float, default=0.1,
                       help="Position size as fraction of portfolio (default: 0.1 = 10%%)")
    parser.add_argument("--db", type=str, default='local',
                       choices=['local', 'remote'],
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

    # --- Output files ---
    parser.add_argument("--output-trades", type=str, default=None,
                       help="Path to write trade log CSV")
    parser.add_argument("--output-symbols", type=str, default=None,
                       help="Path to write filtered symbol list CSV")

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

    # --- Portfolio ranking pipeline ---
    has_portfolio_filters = any([
        args.watchlist, args.price_above is not None, args.price_below is not None,
        args.filter_field, args.top_k_sharpe is not None,
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
            )
            symbols = set(symbols_list)
        elif wl:
            symbols = set(apply_watchlist(list(symbols), wl, args.watchlist_mode))
            print(f"After watchlist ({args.watchlist_mode}): {len(symbols)} symbols")

    # Write symbol list before backtest (portfolio-filtered set)
    if args.output_symbols:
        write_symbols(symbols, args.output_symbols)

    # Run backtest
    run_ml_backtest(
        predictions_path=args.predictions,
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        start_cash=args.start_cash,
        db=args.db,
        position_size=args.position_size,
        output_trades=args.output_trades,
    )
