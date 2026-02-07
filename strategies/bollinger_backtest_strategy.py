"""
Bollinger Band Backtest Strategy for bar_fly_trading backtest framework.

Uses Bollinger band crossover with RSI confirmation for mean-reversion entries.

Entry Conditions (BUY):
    - Price crossed below lower band (prev_close > prev_bb_lower AND close <= bb_lower)
    - RSI <= 40 (not overbought)

Exit Conditions (SELL):
    - Price reaches middle band (mean reversion target)
    - OR RSI > 70 (overbought)
    - OR max hold days reached

This is the backtest-compatible version of BollingerShadowStrategy.
It inherits from BaseStrategy and plugs into backtest.py via evaluate().
"""

import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from order import Order, StockOrder, OrderOperation
from base_strategy import BaseStrategy


class BollingerBacktestStrategy(BaseStrategy):
    """
    Bollinger band mean-reversion strategy for backtesting.

    Buys when price crosses below lower band (oversold).
    Sells when price reverts to middle band or hold limit is reached.
    """

    # Entry thresholds
    RSI_BUY_MAX = 40       # RSI must be <= this for BUY entry
    RSI_SELL_MIN = 60      # RSI must be >= this for SELL (short) entry

    # Exit thresholds
    MAX_HOLD_DAYS = 20     # Force exit after N days
    MIN_HOLD_DAYS = 1      # Must hold at least N days

    def __init__(self, account, symbols, data=None, data_path=None,
                 position_size=0.05, max_hold_days=20, end_date=None):
        """
        Initialize the strategy.

        Args:
            account: BacktestAccount instance
            symbols: Set of symbols to trade
            data: Pre-loaded DataFrame with all_data (optional)
            data_path: Path to all_data CSV(s) — used if data is None
            position_size: Fraction of portfolio per position (default 0.05 = 5%)
            max_hold_days: Max days to hold a position (default 20)
            end_date: Backtest end date — force-close all positions on this day
        """
        super().__init__(account, symbols)
        self.position_size = position_size
        self.MAX_HOLD_DAYS = max_hold_days
        self.end_date = pd.to_datetime(end_date) if end_date else None

        # Track positions: {symbol: {'shares': int, 'entry_date': date, 'entry_price': float}}
        self.positions = {}

        # Completed trade log for stats
        self.trade_log = []

        # Previous day's data for crossover detection
        self._prev_day = {}  # symbol -> {adjusted_close, bbands_lower_20, bbands_upper_20}

        # Load indicator data (bollinger bands, RSI, etc.)
        self.indicator_data = None
        if data is not None:
            self._load_indicator_data(data)
        elif data_path:
            self._load_from_path(data_path)

    def _load_from_path(self, data_path):
        """Load indicator data from CSV path (supports globs)."""
        import glob as glob_mod
        if '*' in data_path:
            files = glob_mod.glob(data_path)
            if not files:
                print(f"[BollingerBacktest] Warning: No files matching {data_path}")
                return
            dfs = [pd.read_csv(f) for f in files]
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_csv(data_path)
        self._load_indicator_data(df)

    def _load_indicator_data(self, df):
        """Load and index indicator data for fast lookup."""
        df = df[df['symbol'].isin(self.symbols)].copy()
        df['date'] = pd.to_datetime(df['date'])

        # Ensure required columns exist
        required = ['date', 'symbol', 'adjusted_close', 'bbands_lower_20', 'bbands_upper_20']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[BollingerBacktest] Warning: Missing columns: {missing}")
            return

        self.indicator_data = df
        print(f"[BollingerBacktest] Loaded indicator data: {len(df):,} rows, "
              f"{df['symbol'].nunique()} symbols, "
              f"{df['date'].min().date()} to {df['date'].max().date()}")

    def _get_indicators(self, symbol, date):
        """Get indicator values for a symbol on a specific date."""
        if self.indicator_data is None:
            return None

        date_ts = pd.to_datetime(date)
        mask = (self.indicator_data['symbol'] == symbol) & (self.indicator_data['date'] == date_ts)
        rows = self.indicator_data[mask]

        if len(rows) == 0:
            return None
        return rows.iloc[0]

    def _check_buy_crossover(self, symbol, indicators):
        """Check if price crossed below lower Bollinger band (BUY signal)."""
        if indicators is None:
            return False

        close = indicators['adjusted_close']
        bb_lower = indicators['bbands_lower_20']

        if pd.isna(bb_lower):
            return False

        # RSI filter
        rsi = indicators.get('rsi_14', None)
        if pd.notna(rsi) and rsi > self.RSI_BUY_MAX:
            return False

        # Check crossover: current close <= lower band AND previous close > previous lower band
        prev = self._prev_day.get(symbol)
        if prev is None:
            return False

        prev_close = prev['adjusted_close']
        prev_bb_lower = prev['bbands_lower_20']

        if pd.isna(prev_bb_lower):
            return False

        return close <= bb_lower and prev_close > prev_bb_lower

    def _check_exit(self, symbol, indicators, hold_days):
        """Check if we should exit a position."""
        if hold_days < self.MIN_HOLD_DAYS:
            return False

        # Max hold reached
        if hold_days >= self.MAX_HOLD_DAYS:
            return True

        if indicators is None:
            return False

        close = indicators['adjusted_close']
        bb_middle = indicators.get('bbands_middle_20', None)
        rsi = indicators.get('rsi_14', None)

        # Exit: price reached middle band (mean reversion target)
        if pd.notna(bb_middle) and close >= bb_middle:
            return True

        # Exit: RSI indicates overbought
        if pd.notna(rsi) and rsi > 70:
            return True

        return False

    def evaluate(self, date: datetime.date, current_prices: pd.DataFrame,
                 options_data: pd.DataFrame) -> list[Order]:
        """
        Evaluate trading signals for the day.

        Called by backtest.py for each trading day.

        Args:
            date: Current date
            current_prices: DataFrame with [symbol, open, adjusted_close, high, low]
            options_data: DataFrame with options data (not used)

        Returns:
            List of Order objects to execute
        """
        orders = []
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]
        current_date = pd.to_datetime(date)
        cash_committed = 0.0  # Track cash allocated to orders this day

        for symbol in self.symbols:
            # Get current price from backtest engine
            price_row = current_prices[current_prices['symbol'] == symbol]
            if len(price_row) == 0:
                continue
            current_price = float(price_row['open'].iloc[0])

            # Get indicator data for today
            indicators = self._get_indicators(symbol, date)

            has_position = symbol in self.positions and self.positions[symbol]['shares'] > 0

            if has_position:
                # Check exit conditions
                entry_date = self.positions[symbol]['entry_date']
                hold_days = (current_date - entry_date).days

                if self._check_exit(symbol, indicators, hold_days):
                    shares = self.positions[symbol]['shares']
                    entry_price = self.positions[symbol]['entry_price']
                    pct_return = (current_price - entry_price) / entry_price * 100

                    orders.append(StockOrder(symbol, OrderOperation.SELL, shares,
                                            current_price, date_str))

                    # Record completed trade
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

                    print(f"  EXIT {symbol}: held {hold_days}d, return: {pct_return:+.2f}%")
                    del self.positions[symbol]

            else:
                # Check entry conditions (buy crossover)
                if self._check_buy_crossover(symbol, indicators):
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

                        rsi_val = indicators.get('rsi_14', None) if indicators is not None else None
                        rsi_str = f", RSI={rsi_val:.1f}" if pd.notna(rsi_val) else ""
                        print(f"  ENTRY {symbol}: close crossed below lower BB{rsi_str}")

            # Store today's data as "previous" for next day's crossover check
            if indicators is not None:
                self._prev_day[symbol] = {
                    'adjusted_close': indicators['adjusted_close'],
                    'bbands_lower_20': indicators['bbands_lower_20'],
                    'bbands_upper_20': indicators['bbands_upper_20'],
                }

        return orders

    def get_open_positions(self):
        """Return current open positions."""
        return self.positions.copy()
