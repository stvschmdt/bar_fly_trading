#!/usr/bin/env python
"""
Runner for Oversold Bounce Strategy (S1).

Pure technical strategy — no ML predictions required.

Supports three modes:
  1. backtest — full historical backtest via backtest.py
  2. daily   — short lookback scan, email summary of signals
  3. live    — fetch from AlphaVantage API, scan, email signals

Usage:
    # Full backtest
    python strategies/run_oversold_bounce.py \
        --data-path 'all_data_*.csv' --use-all-symbols \
        --start-date 2016-01-01 --end-date 2024-12-31

    # Daily report (morning email of what hit in past 2 days)
    python strategies/run_oversold_bounce.py \
        --data-path 'all_data_*.csv' --use-all-symbols \
        --mode daily --lookback-days 2 --summary-only

    # Live scan (AlphaVantage realtime)
    python strategies/run_oversold_bounce.py \
        --watchlist api_data/watchlist.csv --watchlist-mode filter \
        --mode live --summary-only --skip-live

    # Portfolio filtering: top 15 by Sharpe
    python strategies/run_oversold_bounce.py \
        --data-path 'all_data_*.csv' --use-all-symbols \
        --start-date 2024-01-01 --end-date 2025-12-31 \
        --top-k-sharpe 15
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_runner import BaseRunner
from oversold_bounce_strategy import OversoldBounceStrategy


class OversoldBounceRunner(BaseRunner):
    STRATEGY_NAME = "oversold_bounce"
    EMAIL_TAG = "[OVERSOLD_BOUNCE]"

    @classmethod
    def add_strategy_args(cls, parser):
        parser.add_argument("--max-hold-days", type=int, default=3,
                            help="Max days to hold a position (default: 3)")

    def create_strategy(self, account, symbols, args, data=None):
        return OversoldBounceStrategy(
            account=account,
            symbols=symbols,
            data=data,
            position_size=args.position_size,
            max_hold_days=args.max_hold_days,
        )


if __name__ == "__main__":
    OversoldBounceRunner.main()