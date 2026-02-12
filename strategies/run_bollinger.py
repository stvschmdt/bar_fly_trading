#!/usr/bin/env python
"""
Runner for Bollinger Band Strategy.

Supports three modes:
  1. backtest — full historical backtest via backtest.py
  2. daily   — short lookback scan, email summary of crossover signals
  3. live    — fetch from AlphaVantage API, scan, email signals

Usage:
    # Full backtest
    python strategies/run_bollinger.py \
        --data-path 'all_data_*.csv' --use-all-symbols \
        --start-date 2016-01-01 --end-date 2024-12-31

    # Daily report (morning email of what hit in past 2 days)
    python strategies/run_bollinger.py \
        --data-path 'all_data_*.csv' --use-all-symbols \
        --mode daily --lookback-days 2 --summary-only

    # Live scan (AlphaVantage realtime)
    python strategies/run_bollinger.py \
        --watchlist api_data/watchlist.csv --watchlist-mode filter \
        --mode live --summary-only

    # Portfolio filtering: price > $25, top 15 by Sharpe
    python strategies/run_bollinger.py \
        --data-path 'all_data_*.csv' --use-all-symbols \
        --start-date 2024-01-01 --end-date 2025-12-31 \
        --price-above 25 --top-k-sharpe 15
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_runner import BaseRunner
from bollinger_bands_strategy import BollingerBandsStrategy


class BollingerRunner(BaseRunner):
    STRATEGY_NAME = "bollinger"
    EMAIL_TAG = "[BOLLINGER]"

    @classmethod
    def add_strategy_args(cls, parser):
        parser.add_argument("--max-hold-days", type=int, default=20,
                            help="Max days to hold a position (default: 20)")

    def create_strategy(self, account, symbols, args, data=None):
        return BollingerBandsStrategy(
            account=account,
            symbols=symbols,
            data=data,
            position_size=args.position_size,
            max_hold_days=args.max_hold_days,
            end_date=getattr(args, 'end_date', None),
        )


if __name__ == "__main__":
    BollingerRunner.main()