"""
Regression Momentum Strategy for bar_fly_trading backtest framework.

Top-performing strategy from stockformer analysis with 68.1% win rate.

Entry Conditions:
    - pred_reg_3d > 1% (3-day regression predicts >1% return)
    - pred_reg_10d > 2% (10-day regression predicts >2% return)
    - adx_signal > 0 (trend strength confirmation)

Exit Conditions:
    - pred_reg_3d < 0 (3-day regression turns negative)
    - OR cci_signal < 0 (CCI indicates reversal)
    - OR max hold of 13 days reached

Hold Constraints: 2-13 days
"""

import os
import sys
from datetime import datetime, timedelta

import pandas as pd

# Add bar_fly_trading to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bar_fly_trading')))

from order import Order, StockOrder, OrderOperation
from strategy.base_strategy import BaseStrategy


class RegressionMomentumStrategy(BaseStrategy):
    """
    Strategy using regression model predictions with momentum confirmation.

    Win Rate: 68.1% (backtested)
    Avg Return per Trade: +1.64%
    Avg Hold: 5.2 days
    """

    # Strategy parameters
    ENTRY_REG_3D_THRESHOLD = 0.01   # 1% predicted return
    ENTRY_REG_10D_THRESHOLD = 0.02  # 2% predicted return
    EXIT_REG_3D_THRESHOLD = 0.0     # Exit when prediction turns negative
    MIN_HOLD_DAYS = 2
    MAX_HOLD_DAYS = 13

    def __init__(self, account, symbols, predictions_dir=None, position_size=0.1):
        """
        Initialize the strategy.

        Args:
            account: BacktestAccount instance
            symbols: Set of symbols to trade
            predictions_dir: Directory containing prediction CSV files
            position_size: Fraction of portfolio per position (default 0.1 = 10%)
        """
        super().__init__(account, symbols)
        self.predictions_dir = predictions_dir
        self.position_size = position_size

        # Track positions: {symbol: {'shares': int, 'entry_date': date, 'entry_price': float}}
        self.positions = {}

        # Load predictions if path provided
        self.predictions = None
        if predictions_dir:
            self._load_predictions()

    def set_predictions_dir(self, path):
        """Set predictions directory after initialization."""
        self.predictions_dir = path
        self._load_predictions()

    def _load_predictions(self):
        """Load and merge regression predictions from multiple files."""
        if not self.predictions_dir:
            return

        # Load 3-day regression predictions
        reg_3d_path = os.path.join(self.predictions_dir, 'pred_reg_3d.csv')
        reg_10d_path = os.path.join(self.predictions_dir, 'pred_reg_10d.csv')

        if not os.path.exists(reg_3d_path) or not os.path.exists(reg_10d_path):
            print(f"[RegressionMomentumStrategy] Warning: Prediction files not found")
            print(f"  Expected: {reg_3d_path}")
            print(f"  Expected: {reg_10d_path}")
            return

        # Load regression predictions
        reg_3d = pd.read_csv(reg_3d_path)
        reg_10d = pd.read_csv(reg_10d_path)

        # Standardize column names
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

        # Merge predictions
        self.predictions = reg_3d_sub.merge(reg_10d_sub, on=['date', 'symbol'], how='inner')

        # Filter to symbols we're trading
        self.predictions = self.predictions[self.predictions['symbol'].isin(self.symbols)]

        print(f"[RegressionMomentumStrategy] Loaded predictions:")
        print(f"  Rows: {len(self.predictions):,}")
        print(f"  Symbols: {self.predictions['symbol'].nunique()}")
        print(f"  Date range: {self.predictions['date'].min()} to {self.predictions['date'].max()}")

    def get_prediction(self, symbol, date):
        """Get prediction data for a symbol on a specific date."""
        if self.predictions is None:
            return None

        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]

        mask = (self.predictions['date'] == date_str) & (self.predictions['symbol'] == symbol)
        rows = self.predictions[mask]

        if len(rows) == 0:
            return None

        return rows.iloc[0]

    def _check_entry_conditions(self, pred):
        """Check if entry conditions are met."""
        if pred is None:
            return False

        # Entry: pred_reg_3d > 1% AND pred_reg_10d > 2% AND adx_signal > 0
        return (pred['pred_reg_3d'] > self.ENTRY_REG_3D_THRESHOLD and
                pred['pred_reg_10d'] > self.ENTRY_REG_10D_THRESHOLD and
                pred['adx_signal'] > 0)

    def _check_exit_conditions(self, pred, hold_days):
        """Check if exit conditions are met."""
        # Must hold minimum days
        if hold_days < self.MIN_HOLD_DAYS:
            return False

        # Exit at max hold
        if hold_days >= self.MAX_HOLD_DAYS:
            return True

        if pred is None:
            return False

        # Exit: pred_reg_3d < 0 OR cci_signal < 0
        return (pred['pred_reg_3d'] < self.EXIT_REG_3D_THRESHOLD or
                pred['cci_signal'] < 0)

    def evaluate(self, date: datetime.date, current_prices: pd.DataFrame, options_data: pd.DataFrame) -> list[Order]:
        """
        Evaluate trading signals for the day.

        Called by backtest.py for each trading day.

        Args:
            date: Current date
            current_prices: DataFrame with columns [symbol, open, adjusted_close, high, low]
            options_data: DataFrame with options data (not used)

        Returns:
            List of Order objects to execute
        """
        orders = []
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]
        current_date = pd.to_datetime(date)

        for symbol in self.symbols:
            # Get current price
            price_row = current_prices[current_prices['symbol'] == symbol]
            if len(price_row) == 0:
                continue
            current_price = float(price_row['open'].iloc[0])

            # Get prediction for today
            pred = self.get_prediction(symbol, date)

            # Check if we have a position
            has_position = symbol in self.positions and self.positions[symbol]['shares'] > 0

            if has_position:
                # Calculate hold days
                entry_date = self.positions[symbol]['entry_date']
                hold_days = (current_date - entry_date).days

                # Check exit conditions
                if self._check_exit_conditions(pred, hold_days):
                    shares = self.positions[symbol]['shares']
                    entry_price = self.positions[symbol]['entry_price']
                    pct_return = (current_price - entry_price) / entry_price * 100

                    orders.append(StockOrder(symbol, OrderOperation.SELL, shares, current_price, date_str))

                    print(f"  EXIT {symbol}: held {hold_days}d, return: {pct_return:+.2f}%")

                    # Clear position
                    del self.positions[symbol]

            else:
                # Check entry conditions
                if self._check_entry_conditions(pred):
                    shares = self.account.get_max_buyable_shares(current_price, self.position_size)
                    if shares > 0:
                        orders.append(StockOrder(symbol, OrderOperation.BUY, shares, current_price, date_str))

                        # Track position
                        self.positions[symbol] = {
                            'shares': shares,
                            'entry_date': current_date,
                            'entry_price': current_price
                        }

                        print(f"  ENTRY {symbol}: pred_3d={pred['pred_reg_3d']:.3f}, pred_10d={pred['pred_reg_10d']:.3f}")

        return orders

    def get_open_positions(self):
        """Return current open positions."""
        return self.positions.copy()