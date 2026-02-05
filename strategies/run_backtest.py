"""
Runner script for ML Prediction Strategy backtest.

Uses existing bar_fly_trading framework without modifications.

Usage:
    python run_backtest.py \
        --predictions ../mlr/stockformer/output/predictions/pred_bin_3d.csv \
        --symbols AAPL GOOGL MSFT \
        --start-date 2024-07-01 \
        --end-date 2024-12-31 \
        --start-cash 100000
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

from ml_prediction_strategy import MLPredictionStrategy


def run_ml_backtest(predictions_path, symbols, start_date, end_date, start_cash, db='local'):
    """
    Run backtest using ML predictions.

    Args:
        predictions_path: Path to stockformer predictions CSV
        symbols: Set of symbols to trade
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        start_cash: Initial cash balance
        db: Database to use ('local' or 'remote')

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
        position_size=0.1  # 10% per position
    )

    # Run backtest using existing backtest() function
    print(f"\n{'='*60}")
    print(f"ML PREDICTION BACKTEST")
    print(f"{'='*60}")
    print(f"Predictions: {predictions_path}")
    print(f"Symbols: {symbols}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Starting cash: ${start_cash:,.2f}")
    print(f"{'='*60}\n")

    account_values = backtest(strategy, symbols, start_date, end_date)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Final account values: {account_values}")

    return account_values


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ML Prediction Backtest")

    parser.add_argument("--predictions", type=str, required=True,
                       help="Path to stockformer predictions CSV")
    parser.add_argument("--symbols", type=str, nargs="+", required=True,
                       help="Symbols to trade")
    parser.add_argument("--start-date", type=str, required=True,
                       help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True,
                       help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--start-cash", type=float, default=100000,
                       help="Initial cash balance (default: 100000)")
    parser.add_argument("--db", type=str, default='local',
                       choices=['local', 'remote'],
                       help="Database to use (default: local)")

    args = parser.parse_args()

    run_ml_backtest(
        predictions_path=args.predictions,
        symbols=set(args.symbols),
        start_date=args.start_date,
        end_date=args.end_date,
        start_cash=args.start_cash,
        db=args.db
    )
