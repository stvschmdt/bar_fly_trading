"""
52-Week Low Bounce Strategy (M6) for bar_fly_trading backtest framework.

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

Data source: raw CSV data (all_data_*.csv). No predictions needed.
"""

import os
import sys
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from order import Order, StockOrder, OrderOperation
from base_strategy import BaseStrategy
from signal_writer import SignalWriter


class LowBounceStrategy(BaseStrategy):
    """
    52-Week Low Bounce strategy (M6): pure technical contrarian.

    Buys when price is near 52-week low with bearish sentiment (contrarian).
    Sells after a 30-day hold, on strong RSI recovery, or on take-profit.
    """

    STRATEGY_NAME = "low_bounce"

    # Entry thresholds
    LOW_PROXIMITY = 1.10         # within 10% of 52-week low
    RSI_ENTRY_MAX = 40           # RSI must be < this
    DELTA_ENTRY_MAX = 0          # bull_bear_delta must be <= this
    MAX_HOLD_DAYS = 30           # Fixed 30-day hold
    MIN_HOLD_DAYS = 3            # Must hold at least 3 days
    RSI_EXIT_THRESHOLD = 60      # Early exit if RSI recovers above this
    TAKE_PROFIT_PCT = 15.0       # Take profit at 15% return
    MAX_POSITIONS = 10           # Max concurrent positions

    def __init__(self, account, symbols, data=None, data_path=None,
                 position_size=0.1, max_hold_days=30):
        super().__init__(account, symbols)
        self.position_size = position_size
        self.MAX_HOLD_DAYS = max_hold_days

        self.positions = {}
        self.trade_log = []

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
                print(f"[{self.STRATEGY_NAME}] Warning: No files matching {data_path}")
                return
            dfs = [pd.read_csv(f) for f in files]
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_csv(data_path)
        self._load_indicator_data(df)

    def _load_indicator_data(self, df):
        """Load and index indicator data for fast lookup."""
        # Normalize column names
        if 'ticker' in df.columns and 'symbol' not in df.columns:
            df = df.rename(columns={'ticker': 'symbol'})
        if 'close' in df.columns and 'adjusted_close' not in df.columns:
            df = df.rename(columns={'close': 'adjusted_close'})

        df = df[df['symbol'].isin(self.symbols)].copy()
        df['date'] = pd.to_datetime(df['date'])

        required = ['date', 'symbol', 'adjusted_close', 'rsi_14', 'bull_bear_delta', '52_week_low']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[{self.STRATEGY_NAME}] Warning: Missing columns: {missing}")
            return

        self.indicator_data = df
        print(f"[{self.STRATEGY_NAME}] Loaded indicator data: {len(df):,} rows, "
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

    def _check_entry(self, indicators):
        """Check if entry conditions are met (all three required)."""
        if indicators is None:
            return False

        close = indicators['adjusted_close']
        low_52w = indicators.get('52_week_low', None)
        rsi = indicators.get('rsi_14', None)
        delta = indicators.get('bull_bear_delta', None)

        if pd.isna(low_52w) or pd.isna(rsi) or pd.isna(delta):
            return False

        if low_52w <= 0:
            return False

        return (close < low_52w * self.LOW_PROXIMITY and
                rsi < self.RSI_ENTRY_MAX and
                delta <= self.DELTA_ENTRY_MAX)

    def _check_exit(self, indicators, hold_days, entry_price, current_price):
        """Check if exit conditions are met."""
        if hold_days < self.MIN_HOLD_DAYS:
            return False, ""

        # Max hold reached
        if hold_days >= self.MAX_HOLD_DAYS:
            return True, f"max hold {self.MAX_HOLD_DAYS}d"

        # Take profit
        if entry_price > 0:
            pct_return = (current_price - entry_price) / entry_price * 100
            if pct_return >= self.TAKE_PROFIT_PCT:
                return True, f"take profit ({pct_return:.1f}%)"

        if indicators is None:
            return False, ""

        # Early exit: RSI strong recovery
        rsi = indicators.get('rsi_14', None)
        if pd.notna(rsi) and rsi > self.RSI_EXIT_THRESHOLD:
            return True, f"RSI recovered ({rsi:.1f})"

        return False, ""

    def evaluate(self, date: datetime.date, current_prices: pd.DataFrame,
                 options_data: pd.DataFrame) -> list[Order]:
        orders = []
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]
        current_date = pd.to_datetime(date)
        cash_committed = 0.0

        for symbol in self.symbols:
            price_row = current_prices[current_prices['symbol'] == symbol]
            if len(price_row) == 0:
                continue
            current_price = float(price_row['open'].iloc[0])

            indicators = self._get_indicators(symbol, date)
            has_position = symbol in self.positions and self.positions[symbol]['shares'] > 0

            if has_position:
                entry_date = self.positions[symbol]['entry_date']
                entry_price = self.positions[symbol]['entry_price']
                hold_days = (current_date - entry_date).days

                should_exit, exit_reason = self._check_exit(indicators, hold_days, entry_price, current_price)
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
                    del self.positions[symbol]

            else:
                # Check entry conditions + position limits
                if len(self.positions) >= self.MAX_POSITIONS:
                    continue

                if self._check_entry(indicators):
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
                        delta_val = indicators.get('bull_bear_delta', None) if indicators is not None else None
                        rsi_str = f", RSI={rsi_val:.1f}" if pd.notna(rsi_val) else ""
                        delta_str = f", delta={delta_val:.0f}" if pd.notna(delta_val) else ""
                        print(f"  ENTRY {symbol}: near 52w low{rsi_str}{delta_str}")

        return orders

    def get_open_positions(self):
        """Return current open positions."""
        return self.positions.copy()

    def run_signals(self, current_prices, trade_date=None, output_path=None):
        """
        One-shot signal evaluation. Checks for 52-week low bounce conditions
        in the loaded indicator_data for the most recent date.

        Args:
            current_prices: DataFrame with [symbol, open] at minimum
            trade_date: Date to evaluate (defaults to today)
            output_path: Where to write signal CSV (None = print only)

        Returns:
            list of signal dicts (may be empty)
        """
        from datetime import datetime as dt
        trade_date = trade_date or dt.now()
        date_str = (trade_date.strftime('%Y-%m-%d')
                    if hasattr(trade_date, 'strftime') else str(trade_date)[:10])

        writer = SignalWriter(output_path) if output_path else None
        signals = []

        if self.indicator_data is None:
            print(f"[{self.STRATEGY_NAME}] No indicator data loaded")
            return signals

        for symbol in self.symbols:
            sym_data = self.indicator_data[
                self.indicator_data['symbol'] == symbol
            ].sort_values('date')

            if len(sym_data) == 0:
                continue

            latest = sym_data.iloc[-1]

            close = latest['adjusted_close']
            low_52w = latest.get('52_week_low', None)
            rsi = latest.get('rsi_14', None)
            delta = latest.get('bull_bear_delta', None)

            if pd.isna(low_52w) or pd.isna(rsi) or pd.isna(delta):
                continue

            if low_52w <= 0:
                continue

            if close < low_52w * self.LOW_PROXIMITY and rsi < self.RSI_ENTRY_MAX and delta <= self.DELTA_ENTRY_MAX:
                reason = f"near 52w low (${low_52w:.2f}), RSI={rsi:.1f}, delta={delta:.0f}"

                price_row = current_prices[current_prices['symbol'] == symbol]
                current_price = float(price_row['open'].iloc[0]) if len(price_row) > 0 else close

                sig = {'action': 'BUY', 'symbol': symbol, 'shares': 0,
                       'price': current_price, 'reason': reason}
                signals.append(sig)

                if writer:
                    writer.add('BUY', symbol, shares=0, price=current_price,
                               strategy=self.STRATEGY_NAME, reason=reason)

                print(f"  SIGNAL BUY {symbol} @ ${current_price:.2f}: {reason}")

        if writer and signals:
            writer.save()
        elif not signals:
            print(f"[{self.STRATEGY_NAME}] {date_str}: No signals (hold/do nothing)")

        return signals