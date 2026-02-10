"""
Oversold Reversal Strategy for bar_fly_trading.

ML-enhanced strategy — requires binary 3d predictions from stockformer.

Entry Conditions:
    - rsi_14 < 40 (oversold)
    - bull_bear_delta <= -2 (bearish technicals)
    - bin_3d predicts UP (model confirms reversion is coming)

Exit Conditions:
    - hold >= 3 days (fixed 3-day hold period)
    - OR rsi_14 > 55 (early recovery)

Hold Constraints: 1-3 days

The logic: stock is technically oversold (RSI + bearish delta), but the
ML model says the reversion is coming. Filters out false bottoms.

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
    'pred_class_bin_3d': 'pred_bin_3d',
    'prob_1_bin_3d': 'prob_up_3d',
}


class OversoldReversalStrategy(BaseStrategy):
    """
    Strategy combining oversold technicals with ML model confirmation.

    Win Rate: 63.0% (backtested on v2 inference data)
    Avg Return per Trade: +0.56%
    Sharpe: 0.84
    """

    STRATEGY_NAME = "oversold_reversal"
    REQUIRED_COLUMNS = [
        'date', 'symbol', 'adjusted_close',
        'rsi_14', 'bull_bear_delta',
    ]
    ML_COLUMNS = ['pred_bin_3d', 'prob_up_3d']

    # Entry thresholds
    RSI_ENTRY_THRESHOLD = 40
    DELTA_ENTRY_THRESHOLD = -2
    PROB_UP_THRESHOLD = 0.55
    RSI_EXIT_THRESHOLD = 55
    MAX_HOLD_DAYS = 7
    MIN_HOLD_DAYS = 1
    MAX_POSITIONS = 10

    # Exit safety overrides (tighter for short-term reversal plays)
    STOP_LOSS_PCT = -0.05       # -5%
    TAKE_PROFIT_PCT = 0.10      # +10%

    def __init__(self, account, symbols, data=None, predictions_path=None,
                 position_size=0.1):
        super().__init__(account, symbols)
        self.position_size = position_size

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
            print(f"  Strategy requires merged_predictions.csv with binary 3d model")

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
        """RSI < 40 AND delta <= -2 AND bin_3d predicts UP."""
        if row is None:
            return False

        rsi = row.get('rsi_14', None)
        delta = row.get('bull_bear_delta', None)
        model_up = row.get('pred_bin_3d', None)

        if pd.isna(rsi) or pd.isna(delta) or pd.isna(model_up):
            return False

        prob_up = row.get('prob_up_3d', None)
        if pd.notna(prob_up) and prob_up < self.PROB_UP_THRESHOLD:
            return False

        return (rsi < self.RSI_ENTRY_THRESHOLD and
                delta <= self.DELTA_ENTRY_THRESHOLD and
                model_up == 1)

    def check_exit(self, row, hold_days, entry_price=None):
        """Exit: max hold 3d OR RSI recovered > 55."""
        if hold_days < self.MIN_HOLD_DAYS:
            return False, ""
        if hold_days >= self.MAX_HOLD_DAYS:
            return True, f"max hold {self.MAX_HOLD_DAYS}d"

        if row is None:
            return False, ""

        rsi = row.get('rsi_14', None) if hasattr(row, 'get') else None
        if pd.notna(rsi) and rsi > self.RSI_EXIT_THRESHOLD:
            return True, f"RSI recovered ({rsi:.1f})"

        return False, ""

    def entry_reason(self, row):
        """Generate entry reason string."""
        rsi = row.get('rsi_14', None)
        delta = row.get('bull_bear_delta', None)
        prob = row.get('prob_up_3d', None)
        parts = []
        if pd.notna(rsi):
            parts.append(f"RSI={rsi:.1f}")
        if pd.notna(delta):
            parts.append(f"delta={delta:.0f}")
        if pd.notna(prob):
            parts.append(f"prob_up={prob:.3f}")
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
                entry_price = self.positions[symbol]['entry_price']
                hold_days = (current_date - entry_date).days

                should_exit, exit_reason = self.check_exit(indicators, hold_days, entry_price)
                # Safety backstop: stop-loss, take-profit, trailing stop
                if not should_exit:
                    should_exit, exit_reason = self.check_exit_safety(
                        symbol, current_price, entry_price)
                if should_exit:
                    shares = self.positions[symbol]['shares']
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
                    self.record_exit(symbol, current_date)
                    del self.positions[symbol]

            else:
                if len(self.positions) >= self.MAX_POSITIONS:
                    continue
                allowed, _ = self.is_reentry_allowed(symbol, current_date)
                if not allowed:
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
                        self.record_entry(symbol)

                        rsi_val = indicators.get('rsi_14', None)
                        delta_val = indicators.get('bull_bear_delta', None)
                        prob_val = indicators.get('prob_up_3d', None)
                        parts = [f"RSI={rsi_val:.1f}", f"delta={delta_val:.0f}"]
                        if pd.notna(prob_val):
                            parts.append(f"prob_up={prob_val:.3f}")
                        print(f"  ENTRY {symbol}: {', '.join(parts)}")

        return orders
