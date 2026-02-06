"""
ML Prediction Strategy for bar_fly_trading backtest framework.

Uses stockformer model predictions to make trading decisions.
Integrates with existing Account and BacktestAccount classes.
"""

import os
import sys
from datetime import datetime

import pandas as pd

# Add bar_fly_trading to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bar_fly_trading')))

from order import Order, StockOrder, OrderOperation
from strategy.base_strategy import BaseStrategy


class MLPredictionStrategy(BaseStrategy):
    """
    Strategy that trades based on ML model predictions.

    Buy when: predicted class = 1 (positive return expected)
    Sell when: predicted class = 0 (negative return expected)

    Args:
        account: BacktestAccount instance
        symbols: Set of symbols to trade
        predictions_path: Path to stockformer predictions CSV
        position_size: Fraction of portfolio per position (default 0.1 = 10%)
    """

    def __init__(self, account, symbols, predictions_path=None, position_size=0.1):
        super().__init__(account, symbols)
        self.predictions_path = predictions_path
        self.position_size = position_size
        self.holdings = {}  # {symbol: {'shares': int, 'entry_date': date, 'entry_price': float}}

        # Completed trade log for stats
        self.trade_log = []

        if predictions_path:
            self.predictions = self._load_predictions()
        else:
            self.predictions = None

    def set_predictions_path(self, path):
        """Set predictions path after initialization."""
        self.predictions_path = path
        self.predictions = self._load_predictions()

    def _load_predictions(self):
        """Load and index predictions by date and symbol."""
        df = pd.read_csv(self.predictions_path)

        # Ensure date column is string for matching
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        # Handle column name variations
        if 'symbol' not in df.columns and 'ticker' in df.columns:
            df['symbol'] = df['ticker']

        # Filter to only symbols we're trading
        df = df[df['symbol'].isin(self.symbols)]

        print(f"[MLPredictionStrategy] Loaded {len(df)} predictions for {df['symbol'].nunique()} symbols")
        print(f"[MLPredictionStrategy] Date range: {df['date'].min()} to {df['date'].max()}")

        return df

    def get_prediction(self, symbol, date):
        """Get prediction for a symbol on a specific date."""
        if self.predictions is None:
            return None

        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]

        mask = (self.predictions['date'] == date_str) & (self.predictions['symbol'] == symbol)
        row = self.predictions[mask]

        if len(row) == 0:
            return None

        return row.iloc[0]

    def evaluate(self, date: datetime.date, current_prices: pd.DataFrame, options_data: pd.DataFrame) -> list[Order]:
        """
        Evaluate trading signals for the day.

        Called by backtest.py for each trading day.

        Args:
            date: Current date
            current_prices: DataFrame with columns [symbol, open, adjusted_close, high, low]
            options_data: DataFrame with options data (not used in this strategy)

        Returns:
            List of Order objects to execute
        """
        orders = []
        date_str = date.strftime('%Y-%m-%d')

        for symbol in self.symbols:
            # Get current price
            price_row = current_prices[current_prices['symbol'] == symbol]
            if len(price_row) == 0:
                continue
            current_price = float(price_row['open'].iloc[0])

            # Get prediction
            pred = self.get_prediction(symbol, date)
            if pred is None:
                continue

            pred_class = int(pred['pred_class'])
            has_position = symbol in self.holdings and self.holdings[symbol]['shares'] > 0

            # Simple buy/sell logic
            if pred_class == 1 and not has_position:
                # BUY signal
                shares = self.account.get_max_buyable_shares(current_price, self.position_size)
                if shares > 0:
                    orders.append(StockOrder(symbol, OrderOperation.BUY, shares, current_price, date_str))
                    self.holdings[symbol] = {
                        'shares': shares,
                        'entry_date': pd.to_datetime(date),
                        'entry_price': current_price,
                    }

            elif pred_class == 0 and has_position:
                # SELL signal
                pos = self.holdings[symbol]
                shares = pos['shares']
                entry_price = pos['entry_price']
                entry_date = pos['entry_date']
                hold_days = (pd.to_datetime(date) - entry_date).days
                pct_return = (current_price - entry_price) / entry_price * 100

                orders.append(StockOrder(symbol, OrderOperation.SELL, shares, current_price, date_str))

                self.trade_log.append({
                    'symbol': symbol,
                    'entry_date': entry_date.strftime('%Y-%m-%d'),
                    'exit_date': date_str,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'shares': shares,
                    'pnl': (current_price - entry_price) * shares,
                    'return_pct': pct_return,
                    'hold_days': hold_days,
                })

                del self.holdings[symbol]

        return orders
