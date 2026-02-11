"""
Bollinger Band Strategy for bar_fly_trading.

Uses Bollinger band crossover with RSI confirmation for mean-reversion entries.

Entry Conditions (BUY):
    - Price crossed below lower band (prev_close > prev_bb_lower AND close <= bb_lower)
    - RSI <= 30 (deeply oversold)
    - Volume >= 500K (liquidity filter)

Exit Conditions (SELL):
    - Price reaches middle band (mean reversion target)
    - OR RSI > 70 (overbought)
    - OR max hold days reached

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


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger band mean-reversion strategy.

    Buys when price crosses below lower band (oversold).
    Sells when price reverts to middle band or hold limit is reached.
    """

    STRATEGY_NAME = "bollinger"
    REQUIRED_COLUMNS = [
        'date', 'symbol', 'adjusted_close',
        'bbands_lower_20', 'bbands_upper_20',
    ]

    # Entry thresholds
    RSI_BUY_MAX = 30       # RSI must be <= this for BUY entry (textbook oversold)
    RSI_SELL_MIN = 60      # RSI must be >= this for SELL entry
    MIN_VOLUME = 500_000   # Minimum daily volume to filter illiquid names

    # Exit thresholds
    MAX_HOLD_DAYS = 20
    MIN_HOLD_DAYS = 1

    # Exit safety overrides
    STOP_LOSS_PCT = -0.07       # -7%
    TAKE_PROFIT_PCT = 0.12      # +12%
    TRAILING_STOP_PCT = -0.05   # -5% from high-water mark
    TRAILING_ACTIVATION_PCT = 0.03  # Start trailing after +3%

    def __init__(self, account, symbols, data=None, data_path=None,
                 position_size=0.05, max_hold_days=20, end_date=None):
        super().__init__(account, symbols)
        self.position_size = position_size
        self.MAX_HOLD_DAYS = max_hold_days
        self.end_date = pd.to_datetime(end_date) if end_date else None

        # Previous day's data for crossover detection (backtest only)
        self._prev_day = {}

        # Load data via base class
        if data is not None:
            self.load_overnight_data_from_df(data)
        elif data_path:
            self.load_overnight_data(data_path)

    def load_overnight_data_from_df(self, df):
        """Load from pre-loaded DataFrame (used when runner already loaded data)."""
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
        """
        Check if entry conditions met on a single row.

        For signal scanning (daily/live mode). Does NOT check crossover
        (needs two rows). Instead checks: close <= bb_lower AND RSI <= 40.
        """
        close = row.get('adjusted_close')
        bb_lower = row.get('bbands_lower_20')
        rsi = row.get('rsi_14')

        if pd.isna(close) or pd.isna(bb_lower):
            return False

        if close > bb_lower:
            return False

        if pd.notna(rsi) and rsi > self.RSI_BUY_MAX:
            return False

        volume = row.get('volume', None)
        if pd.notna(volume) and volume < self.MIN_VOLUME:
            return False

        return True

    def check_entry_crossover(self, symbol, indicators):
        """
        Check crossover entry: price crossed below lower BB.

        Used by evaluate() during backtest (needs prev day state).
        """
        if indicators is None:
            return False

        close = indicators['adjusted_close']
        bb_lower = indicators['bbands_lower_20']

        if pd.isna(bb_lower):
            return False

        rsi = indicators.get('rsi_14', None)
        if pd.notna(rsi) and rsi > self.RSI_BUY_MAX:
            return False

        volume = indicators.get('volume', None)
        if pd.notna(volume) and volume < self.MIN_VOLUME:
            return False

        prev = self._prev_day.get(symbol)
        if prev is None:
            return False

        prev_close = prev['adjusted_close']
        prev_bb_lower = prev['bbands_lower_20']

        if pd.isna(prev_bb_lower):
            return False

        return close <= bb_lower and prev_close > prev_bb_lower

    def check_exit(self, row, hold_days, entry_price=None):
        """Exit: middle band reached, RSI overbought, or max hold."""
        if hold_days < self.MIN_HOLD_DAYS:
            return False, ""
        if hold_days >= self.MAX_HOLD_DAYS:
            return True, f"max hold {self.MAX_HOLD_DAYS}d"

        if row is None:
            return False, ""

        close = row.get('adjusted_close') if hasattr(row, 'get') else row['adjusted_close']
        bb_middle = row.get('bbands_middle_20', None) if hasattr(row, 'get') else row.get('bbands_middle_20')
        rsi = row.get('rsi_14', None) if hasattr(row, 'get') else row.get('rsi_14')

        if pd.notna(bb_middle) and close >= bb_middle:
            return True, "middle band reached"
        if pd.notna(rsi) and rsi > 70:
            return True, f"RSI overbought ({rsi:.1f})"

        return False, ""

    def entry_reason(self, row):
        """Generate entry reason string."""
        rsi = row.get('rsi_14', None)
        rsi_str = f", RSI={rsi:.1f}" if pd.notna(rsi) else ""
        return f"close below lower BB{rsi_str}"

    # ------------------------------------------------------------------ #
    #  SIGNAL SCANNING (override for crossover detection)
    # ------------------------------------------------------------------ #

    def find_signals(self, data=None, lookback_days=2, require_today=False):
        """
        Scan for BB crossover signals.

        Bollinger needs 2-row comparison for crossover detection,
        so this overrides the base to check consecutive pairs.
        """
        df = data if data is not None else self.overnight_data
        if df is None or df.empty:
            print(f"[{self.STRATEGY_NAME}] No data to scan")
            return []

        today_str = pd.Timestamp.now().strftime('%Y-%m-%d')

        signals = []
        skipped_stale = 0
        for symbol in self.symbols:
            # Re-entry cooldown check
            allowed, _ = self.is_reentry_allowed(symbol)
            if not allowed:
                continue

            # Skip if already holding
            if symbol in self.positions and self.positions[symbol].get('shares', 0) > 0:
                continue

            sym_data = df[df['symbol'] == symbol].sort_values('date')
            if len(sym_data) < 2:
                continue

            recent = sym_data.tail(lookback_days + 1)
            if len(recent) < 2:
                continue

            for idx in range(1, len(recent)):
                today = recent.iloc[idx]
                prev = recent.iloc[idx - 1]

                # Guard: only emit signals from today's row
                row_date = str(today['date'])[:10]
                if require_today and row_date != today_str:
                    skipped_stale += 1
                    continue

                close = today['adjusted_close']
                bb_lower = today.get('bbands_lower_20')
                prev_close = prev['adjusted_close']
                prev_bb_lower = prev.get('bbands_lower_20')

                if pd.isna(bb_lower) or pd.isna(prev_bb_lower):
                    continue

                rsi = today.get('rsi_14', None)
                if pd.isna(rsi):
                    rsi = None

                # BUY crossover: price crossed below lower band
                if close <= bb_lower and prev_close > prev_bb_lower:
                    if rsi is not None and rsi > self.RSI_BUY_MAX:
                        continue
                    rsi_str = f", RSI={rsi:.1f}" if rsi is not None else ""
                    signals.append({
                        'action': 'BUY',
                        'symbol': symbol,
                        'price': float(close),
                        'reason': f"close crossed below lower BB{rsi_str}",
                        'date': str(today['date'])[:10],
                    })

                # SELL crossover: price crossed above upper band
                bb_upper = today.get('bbands_upper_20')
                prev_bb_upper = prev.get('bbands_upper_20')
                if (pd.notna(bb_upper) and pd.notna(prev_bb_upper) and
                        close >= bb_upper and prev_close < prev_bb_upper):
                    if rsi is not None and rsi < self.RSI_SELL_MIN:
                        continue
                    rsi_str = f", RSI={rsi:.1f}" if rsi is not None else ""
                    signals.append({
                        'action': 'SELL',
                        'symbol': symbol,
                        'price': float(close),
                        'reason': f"close crossed above upper BB{rsi_str}",
                        'date': str(today['date'])[:10],
                    })

        if require_today and skipped_stale > 0:
            print(f"  [{self.STRATEGY_NAME}] Skipped {skipped_stale} stale rows (not dated {today_str})")

        return signals

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

                    reason_str = f" ({exit_reason})" if exit_reason else ""
                    print(f"  EXIT {symbol}: held {hold_days}d, return: {pct_return:+.2f}%{reason_str}")
                    self.record_exit(symbol, current_date)
                    del self.positions[symbol]

            else:
                allowed, _ = self.is_reentry_allowed(symbol, current_date)
                if not allowed:
                    if indicators is not None:
                        self._prev_day[symbol] = {
                            'adjusted_close': indicators['adjusted_close'],
                            'bbands_lower_20': indicators['bbands_lower_20'],
                            'bbands_upper_20': indicators['bbands_upper_20'],
                        }
                    continue
                if self.check_entry_crossover(symbol, indicators):
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

                        rsi_val = indicators.get('rsi_14', None) if indicators is not None else None
                        rsi_str = f", RSI={rsi_val:.1f}" if pd.notna(rsi_val) else ""
                        print(f"  ENTRY {symbol}: close crossed below lower BB{rsi_str}")

            # Store today as prev for crossover detection
            if indicators is not None:
                self._prev_day[symbol] = {
                    'adjusted_close': indicators['adjusted_close'],
                    'bbands_lower_20': indicators['bbands_lower_20'],
                    'bbands_upper_20': indicators['bbands_upper_20'],
                }

        return orders


# Backward compatibility alias
BollingerBacktestStrategy = BollingerBandsStrategy