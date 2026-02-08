#!/usr/bin/env python
"""
Runner for Oversold Reversal Strategy.

ML-enhanced strategy — requires binary 3d predictions from stockformer.

Supports three modes:
  1. backtest — full historical backtest via backtest.py
  2. daily   — short lookback scan, email summary of signals
  3. live    — fetch from AlphaVantage API, scan, email signals

Usage:
    # Full backtest (with ML predictions)
    python strategies/run_oversold_reversal.py \
        --predictions merged_predictions.csv --use-all-symbols \
        --start-date 2024-01-01 --end-date 2025-12-31

    # Full backtest (technical only, no ML)
    python strategies/run_oversold_reversal.py \
        --data-path 'all_data_*.csv' --use-all-symbols \
        --start-date 2016-01-01 --end-date 2024-12-31

    # Daily report (morning email of what hit in past 2 days)
    python strategies/run_oversold_reversal.py \
        --predictions merged_predictions.csv --use-all-symbols \
        --mode daily --lookback-days 2 --summary-only

    # Live scan (AlphaVantage realtime)
    python strategies/run_oversold_reversal.py \
        --watchlist api_data/watchlist.csv --watchlist-mode filter \
        --predictions merged_predictions.csv \
        --mode live --summary-only

    # Portfolio filtering: price > $25, top 15 by Sharpe
    python strategies/run_oversold_reversal.py \
        --predictions merged_predictions.csv --use-all-symbols \
        --start-date 2024-01-01 --end-date 2025-12-31 \
        --price-above 25 --top-k-sharpe 15
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_runner import BaseRunner
from oversold_reversal_strategy import OversoldReversalStrategy


class OversoldReversalRunner(BaseRunner):
    STRATEGY_NAME = "oversold_reversal"
    EMAIL_TAG = "[OVERSOLD_REVERSAL]"

    def create_strategy(self, account, symbols, args, data=None):
        return OversoldReversalStrategy(
            account=account,
            symbols=symbols,
            data=data,
            position_size=args.position_size,
        )


if __name__ == "__main__":
    OversoldReversalRunner.main()
