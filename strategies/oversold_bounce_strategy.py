"""
Oversold Bounce Strategy (S1) for bar_fly_trading.

Pure technical strategy — no ML predictions required.
Uses RSI, Bollinger Bands signal, and price-below-lower-band for entries.

Entry Conditions (BUY — ALL must be true):
    - RSI(14) < 35
    - bollinger_bands_signal == 1  (mean-reversion signal from data pipeline)
    - adjusted_close < bbands_lower_20

Exit Conditions (SELL):
    - Hold >= 3 days (fixed short-term hold)
    - OR early exit: hold >= 1 day AND RSI > 55 (recovery)

Data: overnight CSV (merged_predictions or all_data) + live AlphaVantage
Live API: GLOBAL_QUOTE + BBANDS + RSI (3 calls per symbol)
"""

import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from order import Order, StockOrder, OrderOperation
from base_strategy import BaseStrategy


class OversoldBounceStrategy(BaseStrategy):
    """
    Oversold Bounce strategy (S1): pure technical mean-reversion.

    Buys when RSI is deeply oversold, BB signal fires, and price is below lower band.
    Sells after a fixed 3-day hold or on early RSI recovery.
    """

    STRATEGY_NAME = "oversold_bounce"
    REQUIRED_COLUMNS = [
        'date', 'symbol', 'adjusted_close',
        'rsi_14', 'bollinger_bands_signal', 'bbands_lower_20',
    ]

    # Entry thresholds
    RSI_ENTRY_MAX = 35
    MIN_VOLUME = 500_000   # Minimum daily volume to filter illiquid names
    MAX_HOLD_DAYS = 5
    MIN_HOLD_DAYS = 1
    RSI_EXIT_THRESHOLD = 55
    MAX_POSITIONS = 10

    # Exit safety overrides (tighter than base for short-term bounces)
    STOP_LOSS_PCT = -0.05       # -5%
    TAKE_PROFIT_PCT = 0.08      # +8%

    def __init__(self, account, symbols, data=None, data_path=None,
                 position_size=0.1, max_hold_days=5):
        super().__init__(account, symbols)
        self.position_size = position_size
        self.MAX_HOLD_DAYS = max_hold_days

        # Load data via base class
        if data is not None:
            self._load_from_df(data)
        elif data_path:
            self.load_overnight_data(data_path)

    def _load_from_df(self, df):
        """Load from pre-loaded DataFrame."""
        df = self._normalize_columns(df.copy())
        if self.symbols:
            df = df[df['symbol'].isin(self.symbols)].copy()
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            print(f"[{self.STRATEGY_NAME}] Warning: Missing columns: {missing}")
            return
        self.overnight_data = df
        print(f"[{self.STRATEGY_NAME}] Loaded data: {len(df):,} rows, "
              f"{df['symbol'].nunique()} symbols, "
              f"{df['date'].min().date()} to {df['date'].max().date()}")

    # ------------------------------------------------------------------ #
    #  REALTIME API
    # ------------------------------------------------------------------ #

    def fetch_realtime(self, symbol):
        """Fetch live BB + RSI + quote from AlphaVantage (3 API calls)."""
        from api_data.rt_utils import fetch_realtime_bollinger
        return fetch_realtime_bollinger(symbol)

    # ------------------------------------------------------------------ #
    #  ENTRY / EXIT LOGIC
    # ------------------------------------------------------------------ #

    def check_entry(self, row):
        """RSI < 35 AND bb_signal == 1 AND close < bb_lower."""
        rsi = row.get('rsi_14', None)
        bb_signal = row.get('bollinger_bands_signal', None)
        close = row.get('adjusted_close')
        bb_lower = row.get('bbands_lower_20', None)

        if pd.isna(rsi) or pd.isna(bb_signal) or pd.isna(bb_lower):
            return False

        volume = row.get('volume', None)
        if pd.notna(volume) and volume < self.MIN_VOLUME:
            return False

        return rsi < self.RSI_ENTRY_MAX and bb_signal == 1 and close < bb_lower

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
        rsi_str = f"RSI={rsi:.1f}" if pd.notna(rsi) else "RSI=N/A"
        return f"{rsi_str}, BB_signal=1, close < lower band"

    # ------------------------------------------------------------------ #
    #  BACKTEST INTERFACE
    # ------------------------------------------------------------------ #

    def evaluate(self, date: datetime.date, current_prices: pd.DataFrame,
                 options_data: pd.DataFrame) -> list[Order]:
        """Evaluate trading signals for a single backtest day."""
        orders = []
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]
        current_date = pd.to_datetime(date)
        cash_committed = 0.0

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

                if self.check_entry(indicators) if indicators is not None else False:
                    available_cash = self.account.account_values.cash_balance - cash_committed
                    shares = int(available_cash * self.position_size // current_price)
                    order_cost = shares * current_price
                    if shares > 0 and order_cost <= available_cash:
                        orders.append(StockOrder(symbol, OrderOperation.BUY, shares,
                                                current_price, date_str))
                        cash_committed += order_cost

                        self.positions[symbol] = {
                            'shares': shares,
                            'entry_date': current_date,
                            'entry_price': current_price,
                        }
                        self.record_entry(symbol)

                        rsi_val = indicators.get('rsi_14', None)
                        rsi_str = f", RSI={rsi_val:.1f}" if pd.notna(rsi_val) else ""
                        print(f"  ENTRY {symbol}: oversold bounce{rsi_str}")

        return orders