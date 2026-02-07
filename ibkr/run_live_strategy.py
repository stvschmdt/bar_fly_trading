#!/usr/bin/env python
"""
Live trading runner for Regression Momentum Strategy.

Integrates stockformer predictions with IBKR live trading.

Usage:
    # Paper trading (default)
    python run_live_strategy.py --symbols AAPL NVDA MSFT

    # Live trading (use with caution!)
    python run_live_strategy.py --symbols AAPL --live

    # Run once (for cron scheduling)
    python run_live_strategy.py --symbols AAPL --once
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, date
from typing import Optional

import pandas as pd

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibkr.config import IBKRConfig, TradingConfig, PredictionConfig
from ibkr.trade_executor import TradeExecutor
from ibkr.models import TradeSignal, OrderAction
from ibkr.execute_signals import check_live_trading_allowed, TRADING_MODE_CONF

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'logs/live_trading_{date.today()}.log')
    ]
)
logger = logging.getLogger(__name__)


class LiveRegressionMomentumStrategy:
    """
    Live trading implementation of Regression Momentum Strategy.

    Entry: pred_reg_3d > 1% AND pred_reg_10d > 2% AND adx_signal > 0
    Exit:  pred_reg_3d < 0 OR cci_signal < 0 OR hold >= 13 days
    Hold:  2-13 days
    """

    # Strategy parameters
    ENTRY_REG_3D_THRESHOLD = 0.01   # 1%
    ENTRY_REG_10D_THRESHOLD = 0.02  # 2%
    EXIT_REG_3D_THRESHOLD = 0.0
    MIN_HOLD_DAYS = 2
    MAX_HOLD_DAYS = 13

    def __init__(
        self,
        executor: TradeExecutor,
        predictions_config: PredictionConfig
    ):
        """
        Initialize live strategy.

        Args:
            executor: Trade executor instance
            predictions_config: Configuration for loading predictions
        """
        self.executor = executor
        self.predictions_config = predictions_config
        self.predictions: Optional[pd.DataFrame] = None

        self._load_predictions()

    def _load_predictions(self) -> None:
        """Load and merge prediction files."""
        reg_3d_path = self.predictions_config.reg_3d_path
        reg_10d_path = self.predictions_config.reg_10d_path

        if not os.path.exists(reg_3d_path) or not os.path.exists(reg_10d_path):
            logger.error(f"Prediction files not found:")
            logger.error(f"  {reg_3d_path}")
            logger.error(f"  {reg_10d_path}")
            return

        try:
            # Load predictions
            reg_3d = pd.read_csv(reg_3d_path)
            reg_10d = pd.read_csv(reg_10d_path)

            # Standardize columns
            reg_3d['date'] = pd.to_datetime(reg_3d['date']).dt.strftime('%Y-%m-%d')
            reg_10d['date'] = pd.to_datetime(reg_10d['date']).dt.strftime('%Y-%m-%d')

            if 'ticker' in reg_3d.columns:
                reg_3d['symbol'] = reg_3d['ticker']
            if 'ticker' in reg_10d.columns:
                reg_10d['symbol'] = reg_10d['ticker']

            # Extract needed columns
            reg_3d_sub = reg_3d[['date', 'symbol', 'pred_return', 'adx_signal', 'cci_signal']].copy()
            reg_3d_sub.columns = ['date', 'symbol', 'pred_reg_3d', 'adx_signal', 'cci_signal']

            reg_10d_sub = reg_10d[['date', 'symbol', 'pred_return']].copy()
            reg_10d_sub.columns = ['date', 'symbol', 'pred_reg_10d']

            # Merge
            self.predictions = reg_3d_sub.merge(reg_10d_sub, on=['date', 'symbol'], how='inner')

            logger.info(f"Loaded predictions: {len(self.predictions)} rows")
            logger.info(f"Date range: {self.predictions['date'].min()} to {self.predictions['date'].max()}")
            logger.info(f"Symbols: {self.predictions['symbol'].nunique()}")

        except Exception as e:
            logger.error(f"Failed to load predictions: {e}")

    def get_prediction(self, symbol: str, trade_date: Optional[str] = None) -> Optional[dict]:
        """Get prediction for a symbol on a date."""
        if self.predictions is None:
            return None

        date_str = trade_date or date.today().strftime('%Y-%m-%d')
        mask = (self.predictions['date'] == date_str) & (self.predictions['symbol'] == symbol)
        rows = self.predictions[mask]

        if len(rows) == 0:
            return None

        row = rows.iloc[0]
        return {
            'pred_reg_3d': row['pred_reg_3d'],
            'pred_reg_10d': row['pred_reg_10d'],
            'adx_signal': row['adx_signal'],
            'cci_signal': row['cci_signal']
        }

    def check_entry(self, symbol: str, trade_date: Optional[str] = None) -> bool:
        """Check if entry conditions are met for a symbol."""
        pred = self.get_prediction(symbol, trade_date)
        if pred is None:
            return False

        return (
            pred['pred_reg_3d'] > self.ENTRY_REG_3D_THRESHOLD and
            pred['pred_reg_10d'] > self.ENTRY_REG_10D_THRESHOLD and
            pred['adx_signal'] > 0
        )

    def check_exit(self, symbol: str, hold_days: int, trade_date: Optional[str] = None) -> bool:
        """Check if exit conditions are met for a symbol."""
        # Must hold minimum days
        if hold_days < self.MIN_HOLD_DAYS:
            return False

        # Exit at max hold
        if hold_days >= self.MAX_HOLD_DAYS:
            return True

        pred = self.get_prediction(symbol, trade_date)
        if pred is None:
            return False

        return (
            pred['pred_reg_3d'] < self.EXIT_REG_3D_THRESHOLD or
            pred['cci_signal'] < 0
        )

    def evaluate(self, symbols: set[str], trade_date: Optional[str] = None) -> None:
        """
        Evaluate and execute trading signals for all symbols.

        Args:
            symbols: Set of symbols to evaluate
            trade_date: Date to evaluate (defaults to today)
        """
        date_str = trade_date or date.today().strftime('%Y-%m-%d')
        logger.info(f"Evaluating signals for {date_str}")

        # Check if trading is allowed
        allowed, reason = self.executor.risk_manager.is_trading_allowed()
        if not allowed:
            logger.warning(f"Trading not allowed: {reason}")
            return

        # Get current positions
        positions = self.executor.position_manager.get_open_positions()

        # 1. Check exits first
        for symbol, position in positions.items():
            hold_days = self.executor.position_manager.get_hold_days(symbol) or 0

            if self.check_exit(symbol, hold_days, trade_date):
                pred = self.get_prediction(symbol, trade_date)
                reason = f"Exit: hold={hold_days}d"
                if pred:
                    reason += f", pred_3d={pred['pred_reg_3d']:.3f}, cci={pred['cci_signal']}"

                logger.info(f"EXIT signal for {symbol}: {reason}")
                result = self.executor.execute_sell(symbol, reason=reason)

                if result.success:
                    logger.info(f"EXIT executed: {symbol} @ ${result.order.avg_fill_price:.2f}")
                else:
                    logger.error(f"EXIT failed: {symbol} - {result.error_message}")

        # Refresh positions after exits
        positions = self.executor.position_manager.get_open_positions()

        # 2. Check entries
        for symbol in symbols:
            # Skip if already have position
            if symbol in positions:
                continue

            if self.check_entry(symbol, trade_date):
                pred = self.get_prediction(symbol, trade_date)
                reason = f"Entry: pred_3d={pred['pred_reg_3d']:.3f}, pred_10d={pred['pred_reg_10d']:.3f}, adx={pred['adx_signal']}"

                logger.info(f"ENTRY signal for {symbol}: {reason}")
                result = self.executor.execute_buy(symbol, reason=reason)

                if result.success:
                    logger.info(f"ENTRY executed: {symbol} {result.order.filled_shares} shares @ ${result.order.avg_fill_price:.2f}")
                else:
                    logger.error(f"ENTRY failed: {symbol} - {result.error_message}")

                # Check if we've hit position limit
                if len(self.executor.position_manager.get_open_positions()) >= self.executor.trading_config.max_positions:
                    logger.info("Max positions reached, stopping entries")
                    break

    def run_continuous(self, symbols: set[str], interval_minutes: int = 5) -> None:
        """
        Run strategy continuously with periodic evaluation.

        Args:
            symbols: Symbols to trade
            interval_minutes: Minutes between evaluations
        """
        logger.info(f"Starting continuous run, evaluating every {interval_minutes} minutes")

        try:
            while True:
                self.evaluate(symbols)

                # Wait for next evaluation
                logger.info(f"Sleeping {interval_minutes} minutes until next evaluation...")
                time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            logger.info("Received interrupt, stopping...")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Regression Momentum Strategy with IBKR",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--symbols", type=str, nargs="+", required=True,
                        help="Symbols to trade")
    parser.add_argument("--live", action="store_true",
                        help="Use live trading (default: paper)")
    parser.add_argument("--gateway", action="store_true",
                        help="Use IB Gateway instead of TWS")
    parser.add_argument("--client-id", type=int, default=1,
                        help="IBKR client ID")
    parser.add_argument("--once", action="store_true",
                        help="Run once and exit (for cron)")
    parser.add_argument("--interval", type=int, default=5,
                        help="Minutes between evaluations (continuous mode)")
    parser.add_argument("--predictions-dir", type=str,
                        help="Directory containing prediction files")
    parser.add_argument("--position-size", type=float, default=0.10,
                        help="Position size as fraction of portfolio")
    parser.add_argument("--max-positions", type=int, default=10,
                        help="Maximum concurrent positions")
    parser.add_argument("--max-daily-loss", type=float, default=5000,
                        help="Maximum daily loss in dollars")

    args = parser.parse_args()

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    # Configure IBKR connection
    if args.live:
        if not check_live_trading_allowed():
            logger.error("=" * 60)
            logger.error("LIVE TRADING BLOCKED")
            logger.error(f"Edit {TRADING_MODE_CONF} and set TRADING_MODE=live")
            logger.error("=" * 60)
            sys.exit(1)
        if args.gateway:
            ibkr_config = IBKRConfig.live_gateway(args.client_id)
        else:
            ibkr_config = IBKRConfig.live_tws(args.client_id)
        logger.warning("=" * 60)
        logger.warning("LIVE TRADING MODE - REAL MONEY AT RISK")
        logger.warning("=" * 60)
    else:
        if args.gateway:
            ibkr_config = IBKRConfig.paper_gateway(args.client_id)
        else:
            ibkr_config = IBKRConfig.paper_tws(args.client_id)
        logger.info("Paper trading mode")

    # Configure trading parameters
    trading_config = TradingConfig(
        symbols=set(args.symbols),
        position_size=args.position_size,
        max_positions=args.max_positions,
        max_daily_loss=args.max_daily_loss
    )

    # Configure predictions
    if args.predictions_dir:
        predictions_config = PredictionConfig(predictions_dir=args.predictions_dir)
    else:
        predictions_config = PredictionConfig.default()

    # Create and run executor
    with TradeExecutor(ibkr_config, trading_config) as executor:
        strategy = LiveRegressionMomentumStrategy(executor, predictions_config)

        if args.once:
            # Run once and exit
            strategy.evaluate(set(args.symbols))
        else:
            # Run continuously
            strategy.run_continuous(set(args.symbols), args.interval)


if __name__ == "__main__":
    main()
