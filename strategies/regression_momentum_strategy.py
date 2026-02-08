"""
Regression Momentum Strategy for bar_fly_trading.

ML-enhanced strategy — requires regression predictions from stockformer.

Entry Conditions:
    - pred_reg_3d > 1% (3-day regression predicts >1% return)
    - pred_reg_10d > 2% (10-day regression predicts >2% return)
    - adx_signal > 0 (trend strength confirmation)

Exit Conditions:
    - pred_reg_3d < 0 (3-day regression turns negative)
    - OR cci_signal < 0 (CCI indicates reversal)
    - OR max hold of 13 days reached

Hold Constraints: 2-13 days

Data: overnight CSV (merged_predictions.csv required for ML columns)
Live API: GLOBAL_QUOTE only (price overlay on overnight technicals + ML)
"""

import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from order import Order, StockOrder, OrderOperation
from base_strategy import BaseStrategy


# Column name mappings for merged predictions
MERGED_COLUMN_MAP = {
    'pred_return_reg_3d': 'pred_reg_3d',
    'pred_return_reg_10d': 'pred_reg_10d',
}


class RegressionMomentumStrategy(BaseStrategy):
    """
    Strategy using regression model predictions with momentum confirmation.

    Win Rate: 68.1% (backtested)
    Avg Return per Trade: +1.64%
    Avg Hold: 5.2 days
    """

    STRATEGY_NAME = "regression_momentum"
    REQUIRED_COLUMNS = [
        'date', 'symbol', 'adjusted_close',
    ]
    ML_COLUMNS = ['pred_reg_3d', 'pred_reg_10d', 'adx_signal', 'cci_signal']

    # Strategy parameters
    ENTRY_REG_3D_THRESHOLD = 0.01
    ENTRY_REG_10D_THRESHOLD = 0.02
    EXIT_REG_3D_THRESHOLD = 0.0
    MIN_HOLD_DAYS = 2
    MAX_HOLD_DAYS = 13
    MAX_POSITIONS = 10

    def __init__(self, account, symbols, data=None, predictions_path=None,
                 position_size=0.1, max_hold_days=13):
        super().__init__(account, symbols)
        self.position_size = position_size
        self.MAX_HOLD_DAYS = max_hold_days

        if data is not None:
            self._load_from_df(data)
        elif predictions_path:
            self.load_overnight_data(predictions_path)

    def _load_from_df(self, df):
        """Load from pre-loaded DataFrame, mapping merged column names."""
        df = self._normalize_columns(df.copy())

        # Map merged prediction column names
        rename = {}
        for src, dst in MERGED_COLUMN_MAP.items():
            if src in df.columns and dst not in df.columns:
                rename[src] = dst
        if rename:
            df = df.rename(columns=rename)

        if self.symbols:
            df = df[df['symbol'].isin(self.symbols)].copy()
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            print(f"[{self.STRATEGY_NAME}] Warning: Missing columns: {missing}")
            return

        ml_missing = [c for c in self.ML_COLUMNS if c not in df.columns]
        if ml_missing:
            print(f"[{self.STRATEGY_NAME}] Warning: Missing ML columns: {ml_missing}")
            print(f"  Strategy requires merged_predictions.csv with regression models")

        self.overnight_data = df
        print(f"[{self.STRATEGY_NAME}] Loaded data: {len(df):,} rows, "
              f"{df['symbol'].nunique()} symbols, "
              f"{df['date'].min().date()} to {df['date'].max().date()}")

    # ------------------------------------------------------------------ #
    #  REALTIME API
    # ------------------------------------------------------------------ #

    def fetch_realtime(self, symbol):
        """Fetch live quote from AlphaVantage (1 API call)."""
        from api_data.rt_utils import get_realtime_quote
        quote = get_realtime_quote(symbol)
        return pd.DataFrame([{
            'symbol': quote['symbol'],
            'adjusted_close': quote['price'],
            'date': pd.to_datetime(quote['latest_trading_day']),
        }])

    # ------------------------------------------------------------------ #
    #  ENTRY / EXIT LOGIC
    # ------------------------------------------------------------------ #

    def check_entry(self, row):
        """pred_3d > 1% AND pred_10d > 2% AND adx > 0."""
        if row is None:
            return False

        pred_3d = row.get('pred_reg_3d', None)
        pred_10d = row.get('pred_reg_10d', None)
        adx = row.get('adx_signal', None)

        if pd.isna(pred_3d) or pd.isna(pred_10d) or pd.isna(adx):
            return False

        return (pred_3d > self.ENTRY_REG_3D_THRESHOLD and
                pred_10d > self.ENTRY_REG_10D_THRESHOLD and
                adx > 0)

    def check_exit(self, row, hold_days, entry_price=None):
        """Exit: pred_3d < 0 OR cci < 0 OR max hold."""
        if hold_days < self.MIN_HOLD_DAYS:
            return False, ""
        if hold_days >= self.MAX_HOLD_DAYS:
            return True, f"max hold {self.MAX_HOLD_DAYS}d"

        if row is None:
            return False, ""

        pred_3d = row.get('pred_reg_3d', None) if hasattr(row, 'get') else None
        cci = row.get('cci_signal', None) if hasattr(row, 'get') else None

        if pd.notna(pred_3d) and pred_3d < self.EXIT_REG_3D_THRESHOLD:
            return True, f"pred_3d turned negative ({pred_3d:.3f})"
        if pd.notna(cci) and cci < 0:
            return True, f"CCI reversal ({cci:.0f})"

        return False, ""

    def entry_reason(self, row):
        """Generate entry reason string."""
        pred_3d = row.get('pred_reg_3d', None)
        pred_10d = row.get('pred_reg_10d', None)
        adx = row.get('adx_signal', None)
        parts = []
        if pd.notna(pred_3d):
            parts.append(f"pred_3d={pred_3d:.3f}")
        if pd.notna(pred_10d):
            parts.append(f"pred_10d={pred_10d:.3f}")
        if pd.notna(adx):
            parts.append(f"adx={adx:.0f}")
        return ", ".join(parts)

    # ------------------------------------------------------------------ #
    #  BACKTEST INTERFACE
    # ------------------------------------------------------------------ #

    def evaluate(self, date: datetime.date, current_prices: pd.DataFrame,
                 options_data: pd.DataFrame) -> list[Order]:
        """Evaluate trading signals for a single backtest day."""
        orders = []
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]
        current_date = pd.to_datetime(date)

        for symbol in self.symbols:
            price_row = current_prices[current_prices['symbol'] == symbol]
            if len(price_row) == 0:
                continue
            current_price = float(price_row['open'].iloc[0])

            indicators = self.get_indicators(symbol, date)
            has_position = symbol in self.positions and self.positions[symbol]['shares'] > 0

            if has_position:
                entry_date = self.positions[symbol]['entry_date']
                hold_days = (current_date - entry_date).days

                should_exit, exit_reason = self.check_exit(indicators, hold_days)
                if should_exit:
                    shares = self.positions[symbol]['shares']
                    entry_price = self.positions[symbol]['entry_price']
                    pct_return = (current_price - entry_price) / entry_price * 100

                    orders.append(StockOrder(symbol, OrderOperation.SELL, shares,
                                            current_price, date_str))

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
                        'exit_reason': exit_reason,
                    })

                    print(f"  EXIT {symbol}: held {hold_days}d, return: {pct_return:+.2f}% ({exit_reason})")
                    del self.positions[symbol]

            else:
                if len(self.positions) >= self.MAX_POSITIONS:
                    continue

                if self.check_entry(indicators):
                    available_cash = self.account.account_values.cash_balance
                    if available_cash < current_price:
                        continue
                    shares = self.account.get_max_buyable_shares(
                        current_price, self.position_size)
                    if shares > 0:
                        orders.append(StockOrder(symbol, OrderOperation.BUY, shares,
                                                current_price, date_str))

                        self.positions[symbol] = {
                            'shares': shares,
                            'entry_date': current_date,
                            'entry_price': current_price,
                        }

                        pred_3d = indicators.get('pred_reg_3d', None)
                        pred_10d = indicators.get('pred_reg_10d', None)
                        print(f"  ENTRY {symbol}: pred_3d={pred_3d:.3f}, pred_10d={pred_10d:.3f}")

        return orders
