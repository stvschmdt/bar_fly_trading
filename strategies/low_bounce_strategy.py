"""
52-Week Low Bounce Strategy (M6) for bar_fly_trading.

Pure technical strategy — no ML predictions required.
Uses 52-week low proximity, RSI, and bull_bear_delta for entries.

Entry Conditions (BUY — ALL must be true):
    - adjusted_close < 52_week_low * 1.10  (within 10% of 52-week low)
    - RSI(14) < 40
    - bull_bear_delta <= 0  (bearish sentiment, contrarian entry)

Exit Conditions (SELL):
    - Hold >= 30 days (fixed medium-term hold)
    - OR early exit: hold >= 3 days AND RSI > 60 (strong recovery)
    - OR early exit: return > 15% (take profit)

Data: overnight CSV (merged_predictions or all_data) + live AlphaVantage
Live API: GLOBAL_QUOTE only (1 call per symbol, or bulk)
"""

import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from order import Order, StockOrder, OrderOperation
from base_strategy import BaseStrategy


class LowBounceStrategy(BaseStrategy):
    """
    52-Week Low Bounce strategy (M6): pure technical contrarian.

    Buys when price is near 52-week low with bearish sentiment (contrarian).
    Sells after a 30-day hold, on strong RSI recovery, or on take-profit.
    """

    STRATEGY_NAME = "low_bounce"
    REQUIRED_COLUMNS = [
        'date', 'symbol', 'adjusted_close',
        'rsi_14', 'bull_bear_delta', '52_week_low',
    ]

    # Entry thresholds
    LOW_PROXIMITY = 1.10
    RSI_ENTRY_MAX = 40
    DELTA_ENTRY_MAX = 0
    MIN_VOLUME = 500_000   # Minimum daily volume to filter illiquid names
    MAX_HOLD_DAYS = 30
    MIN_HOLD_DAYS = 3
    RSI_EXIT_THRESHOLD = 60
    MAX_POSITIONS = 10

    # Exit safety overrides (wider for longer-hold contrarian plays)
    STOP_LOSS_PCT = -0.10       # -10%
    TAKE_PROFIT_PCT = 0.20      # +20%
    TRAILING_STOP_PCT = -0.08   # -8% from high-water mark
    TRAILING_ACTIVATION_PCT = 0.04  # Start trailing after +4%

    def __init__(self, account, symbols, data=None, data_path=None,
                 position_size=0.1, max_hold_days=30):
        super().__init__(account, symbols)
        self.position_size = position_size
        self.MAX_HOLD_DAYS = max_hold_days

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
        """Close < 52w_low * 1.10 AND RSI < 40 AND delta <= 0."""
        if row is None:
            return False

        close = row.get('adjusted_close')
        low_52w = row.get('52_week_low', None)
        rsi = row.get('rsi_14', None)
        delta = row.get('bull_bear_delta', None)

        if pd.isna(low_52w) or pd.isna(rsi) or pd.isna(delta):
            return False
        if low_52w <= 0:
            return False

        volume = row.get('volume', None)
        if pd.notna(volume) and volume < self.MIN_VOLUME:
            return False

        return (close < low_52w * self.LOW_PROXIMITY and
                rsi < self.RSI_ENTRY_MAX and
                delta <= self.DELTA_ENTRY_MAX)

    def check_exit(self, row, hold_days, entry_price=None):
        """Exit: max hold 30d OR RSI > 60.

        Note: take-profit and stop-loss are handled by check_exit_safety()
        in the base class using TAKE_PROFIT_PCT / STOP_LOSS_PCT (decimal).
        """
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
        low_52w = row.get('52_week_low', None)
        rsi = row.get('rsi_14', None)
        delta = row.get('bull_bear_delta', None)
        parts = []
        if pd.notna(low_52w):
            parts.append(f"near 52w low (${low_52w:.2f})")
        if pd.notna(rsi):
            parts.append(f"RSI={rsi:.1f}")
        if pd.notna(delta):
            parts.append(f"delta={delta:.0f}")
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

                should_exit, exit_reason = self.check_exit(
                    indicators, hold_days, entry_price)
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
                        delta_val = indicators.get('bull_bear_delta', None)
                        rsi_str = f", RSI={rsi_val:.1f}" if pd.notna(rsi_val) else ""
                        delta_str = f", delta={delta_val:.0f}" if pd.notna(delta_val) else ""
                        print(f"  ENTRY {symbol}: near 52w low{rsi_str}{delta_str}")

        return orders
