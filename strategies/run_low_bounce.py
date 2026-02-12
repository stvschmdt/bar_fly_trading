#!/usr/bin/env python
"""
Runner for 52-Week Low Bounce Strategy (M6).

Pure technical strategy — no ML predictions required.

Supports three modes:
  1. backtest — full historical backtest via backtest.py
  2. daily   — short lookback scan, email summary of signals
  3. live    — fetch from AlphaVantage API, scan, email signals

Usage:
    # Full backtest
    python strategies/run_low_bounce.py \
        --data-path 'all_data_*.csv' --use-all-symbols \
        --start-date 2016-01-01 --end-date 2024-12-31

    # Daily report (morning email of what hit in past 2 days)
    python strategies/run_low_bounce.py \
        --data-path 'all_data_*.csv' --use-all-symbols \
        --mode daily --lookback-days 2 --summary-only

    # Live scan (AlphaVantage realtime)
    python strategies/run_low_bounce.py \
        --watchlist api_data/watchlist.csv --watchlist-mode filter \
        --mode live --summary-only

    # Portfolio filtering: price > $25, top 15 by Sharpe
    python strategies/run_low_bounce.py \
        --data-path 'all_data_*.csv' --use-all-symbols \
        --start-date 2024-01-01 --end-date 2025-12-31 \
        --price-above 25 --top-k-sharpe 15
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_runner import BaseRunner
from low_bounce_strategy import LowBounceStrategy


class LowBounceRunner(BaseRunner):
    STRATEGY_NAME = "low_bounce"
    EMAIL_TAG = "[LOW_BOUNCE]"

    @classmethod
    def add_strategy_args(cls, parser):
        parser.add_argument("--max-hold-days", type=int, default=30,
                            help="Max days to hold a position (default: 30)")

    def create_strategy(self, account, symbols, args, data=None):
        return LowBounceStrategy(
            account=account,
            symbols=symbols,
            data=data,
            position_size=args.position_size,
            max_hold_days=args.max_hold_days,
        )


if __name__ == "__main__":
    LowBounceRunner.main()
