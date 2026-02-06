"""
Template Strategy for bar_fly_trading backtest framework.

Copy this file and fill in your entry/exit logic. Works in two modes:

  1. Backtest: `evaluate()` is called by backtest.py for each trading day.
  2. Live/on-the-fly: `run_signals()` evaluates once for today and writes a
     signal CSV that the IBKR executor can consume.

Customization points (search for "CUSTOMIZE"):
  - STRATEGY_NAME / parameters at the top of the class
  - _check_entry_conditions()  — when to open a position
  - _check_exit_conditions()   — when to close a position (custom logic beyond guardrails)
  - _load_data()               — load whatever data your strategy needs

Built-in guardrails (configured via class constants):
  - Stop-loss: exit at max % loss (default -5%)
  - Take-profit: exit at target % gain (default +10%)
  - Max hold days: time-based exit
  - Re-entry cooldown: prevent buy/sell/buy churn on same symbol
  - Max positions: cap concurrent open positions
  - Cash check: skip entries when insufficient buying power

Usage:
    # Backtest (via run_template.py)
    python run_template.py --predictions data.csv --symbols AAPL NVDA \\
        --start-date 2024-07-01 --end-date 2024-12-31

    # Live signal generation (single evaluation, writes signal CSV)
    python run_template.py --predictions data.csv --symbols AAPL NVDA \\
        --mode signals --output-signals signals/pending_orders.csv
"""

import os
import sys
from datetime import datetime

import pandas as pd

# Add project root and strategies dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from order import Order, StockOrder, OrderOperation
from base_strategy import BaseStrategy
from signal_writer import SignalWriter


class TemplateStrategy(BaseStrategy):
    """
    Template strategy — copy and customize.

    CUSTOMIZE: Rename this class and set your parameters below.
    """

    # ------------------------------------------------------------------ #
    # CUSTOMIZE: Strategy metadata
    # ------------------------------------------------------------------ #
    STRATEGY_NAME = "template"

    # ------------------------------------------------------------------ #
    # CUSTOMIZE: Entry parameters
    # ------------------------------------------------------------------ #
    # Add whatever thresholds your entry logic needs.
    # Example: ENTRY_THRESHOLD = 0.01

    # ------------------------------------------------------------------ #
    # CUSTOMIZE: Exit parameters (guardrails)
    # ------------------------------------------------------------------ #
    STOP_LOSS_PCT = -5.0       # Exit if position drops below -5% (set None to disable)
    TAKE_PROFIT_PCT = 10.0     # Exit if position gains above +10% (set None to disable)
    MAX_HOLD_DAYS = 20         # Exit after 20 days (set None to disable)
    MIN_HOLD_DAYS = 1          # Don't exit before this many days

    # ------------------------------------------------------------------ #
    # CUSTOMIZE: Position management
    # ------------------------------------------------------------------ #
    ALLOW_REENTRY = True       # Allow re-entering a symbol after exit?
    REENTRY_COOLDOWN_DAYS = 3  # Days to wait before re-entering same symbol
    MAX_POSITIONS = 10         # Max concurrent open positions (None = unlimited)
    MAX_ENTRIES_PER_DAY = 5    # Max new entries per evaluation (prevents runaway)

    def __init__(self, account, symbols, predictions_path=None, position_size=0.1):
        """
        Args:
            account: BacktestAccount instance
            symbols: Set of symbols to trade
            predictions_path: Path to data CSV your strategy needs
            position_size: Fraction of portfolio per position (0.1 = 10%)
        """
        super().__init__(account, symbols)
        self.predictions_path = predictions_path
        self.position_size = position_size

        # Track positions: {symbol: {'shares', 'entry_date', 'entry_price'}}
        self.positions = {}

        # Track recent exits for re-entry cooldown: {symbol: exit_date}
        self.recent_exits = {}

        # Completed trade log for backtest_stats
        self.trade_log = []

        # Load data
        self.data = None
        if predictions_path:
            self._load_data()

    # ================================================================== #
    # DATA LOADING — CUSTOMIZE
    # ================================================================== #

    def _load_data(self):
        """
        CUSTOMIZE: Load whatever data your strategy needs.

        Default implementation loads a CSV with date/symbol columns.
        Replace this with your own data loading.
        """
        path = self.predictions_path
        if not path or not os.path.exists(path):
            print(f"[{self.STRATEGY_NAME}] Warning: Data not found: {path}")
            return

        df = pd.read_csv(path)
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        if 'ticker' in df.columns and 'symbol' not in df.columns:
            df['symbol'] = df['ticker']

        self.data = df[df['symbol'].isin(self.symbols)].copy()

        print(f"[{self.STRATEGY_NAME}] Loaded {len(self.data):,} rows "
              f"for {self.data['symbol'].nunique()} symbols")

    def get_row(self, symbol, date):
        """Get the data row for a symbol on a date. Returns None if missing."""
        if self.data is None:
            return None

        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]
        mask = (self.data['date'] == date_str) & (self.data['symbol'] == symbol)
        rows = self.data[mask]
        if len(rows) == 0:
            return None
        return rows.iloc[0]

    # ================================================================== #
    # ENTRY CONDITIONS — CUSTOMIZE
    # ================================================================== #

    def _check_entry_conditions(self, symbol, row, current_price, date):
        """
        CUSTOMIZE: Return True when you want to open a position.

        This is called AFTER the built-in guardrails pass (re-entry cooldown,
        max positions, cash check). You only need your signal logic here.

        Args:
            symbol: Ticker symbol
            row: Data row from get_row() (may be None if no data for this date)
            current_price: Current stock price (open)
            date: Current date

        Returns:
            True to enter, False to skip
        """
        # -------------------------------------------------------------- #
        # Replace this with your entry logic. Examples:
        #
        #   # ML prediction entry:
        #   if row is None:
        #       return False
        #   return row['pred_class'] == 1
        #
        #   # Threshold entry:
        #   if row is None:
        #       return False
        #   return (row['pred_return'] > 0.01 and row['adx_signal'] > 0)
        #
        #   # Bollinger band entry:
        #   if row is None:
        #       return False
        #   return current_price <= row['bb_lower']
        # -------------------------------------------------------------- #

        if row is None:
            return False

        # Placeholder — replace with your logic
        return False

    # ================================================================== #
    # EXIT CONDITIONS — CUSTOMIZE
    # ================================================================== #

    def _check_exit_custom(self, symbol, row, current_price, entry_price, hold_days, date):
        """
        CUSTOMIZE: Additional exit logic beyond the built-in guardrails.

        The built-in guardrails (stop-loss, take-profit, max-hold) are checked
        first in _check_exit_conditions(). This method is for your custom
        signal-based exits.

        Return True to exit, False to keep holding.
        """
        # -------------------------------------------------------------- #
        # Add your custom exit logic here. Examples:
        #
        #   # Exit on negative prediction:
        #   if row is not None and row['pred_return'] < 0:
        #       return True
        #
        #   # Exit on indicator reversal:
        #   if row is not None and row['cci_signal'] < 0:
        #       return True
        #
        #   # Exit when price crosses above upper bollinger:
        #   if row is not None and current_price >= row['bb_upper']:
        #       return True
        # -------------------------------------------------------------- #

        return False

    def _check_exit_conditions(self, symbol, row, current_price, entry_price, hold_days, date):
        """
        Full exit check: built-in guardrails + custom logic.

        Usually no need to modify this method — put custom logic in
        _check_exit_custom() instead.
        """
        # Respect minimum hold
        if hold_days < self.MIN_HOLD_DAYS:
            return False

        pct_change = (current_price - entry_price) / entry_price * 100

        # Built-in: stop-loss
        if self.STOP_LOSS_PCT is not None and pct_change <= self.STOP_LOSS_PCT:
            return True

        # Built-in: take-profit
        if self.TAKE_PROFIT_PCT is not None and pct_change >= self.TAKE_PROFIT_PCT:
            return True

        # Built-in: max hold days
        if self.MAX_HOLD_DAYS is not None and hold_days >= self.MAX_HOLD_DAYS:
            return True

        # Custom exit logic
        return self._check_exit_custom(symbol, row, current_price,
                                       entry_price, hold_days, date)

    # ================================================================== #
    # GUARDRAILS — usually no changes needed
    # ================================================================== #

    def _can_enter(self, symbol, current_price, current_date, entries_today):
        """
        Pre-entry guardrail checks. Returns (allowed, reason).

        Checks: re-entry cooldown, max positions, max entries/day, cash.
        """
        # Already have position
        if symbol in self.positions and self.positions[symbol]['shares'] > 0:
            return False, "already holding"

        # Re-entry cooldown
        if not self.ALLOW_REENTRY and symbol in self.recent_exits:
            return False, "re-entry disabled"

        if self.ALLOW_REENTRY and symbol in self.recent_exits:
            days_since_exit = (current_date - self.recent_exits[symbol]).days
            if days_since_exit < self.REENTRY_COOLDOWN_DAYS:
                return False, f"cooldown ({days_since_exit}/{self.REENTRY_COOLDOWN_DAYS}d)"

        # Max positions
        if self.MAX_POSITIONS is not None:
            open_count = sum(1 for p in self.positions.values() if p['shares'] > 0)
            if open_count >= self.MAX_POSITIONS:
                return False, f"max positions ({self.MAX_POSITIONS})"

        # Max entries per day
        if self.MAX_ENTRIES_PER_DAY is not None and entries_today >= self.MAX_ENTRIES_PER_DAY:
            return False, f"max entries/day ({self.MAX_ENTRIES_PER_DAY})"

        # Cash check — can we afford at least 1 share?
        buyable = self.account.get_max_buyable_shares(current_price, self.position_size)
        if buyable <= 0:
            return False, "insufficient cash"

        return True, ""

    # ================================================================== #
    # CORE EVALUATE — usually no changes needed
    # ================================================================== #

    def evaluate(self, date: datetime.date, current_prices: pd.DataFrame,
                 options_data: pd.DataFrame) -> list[Order]:
        """
        Evaluate trading signals for one day.

        Called by backtest.py each trading day. Also used by run_signals()
        for live signal generation.
        """
        orders = []
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]
        current_date = pd.to_datetime(date)
        entries_today = 0

        # PHASE 1: Check exits first (frees up cash and position slots)
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            if pos['shares'] <= 0:
                continue

            price_row = current_prices[current_prices['symbol'] == symbol]
            if len(price_row) == 0:
                continue
            current_price = float(price_row['open'].iloc[0])

            row = self.get_row(symbol, date)
            hold_days = (current_date - pos['entry_date']).days

            if self._check_exit_conditions(symbol, row, current_price,
                                           pos['entry_price'], hold_days, date):
                shares = pos['shares']
                entry_price = pos['entry_price']
                pct_return = (current_price - entry_price) / entry_price * 100

                orders.append(StockOrder(symbol, OrderOperation.SELL,
                                        shares, current_price, date_str))

                self.trade_log.append({
                    'symbol': symbol,
                    'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
                    'exit_date': date_str,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'shares': shares,
                    'pnl': (current_price - entry_price) * shares,
                    'return_pct': pct_return,
                    'hold_days': hold_days,
                })

                print(f"  EXIT {symbol}: held {hold_days}d, return: {pct_return:+.2f}%")

                # Track exit for re-entry cooldown
                self.recent_exits[symbol] = current_date
                del self.positions[symbol]

        # PHASE 2: Check entries
        for symbol in self.symbols:
            price_row = current_prices[current_prices['symbol'] == symbol]
            if len(price_row) == 0:
                continue
            current_price = float(price_row['open'].iloc[0])

            # Guardrail checks
            allowed, reason = self._can_enter(symbol, current_price,
                                              current_date, entries_today)
            if not allowed:
                continue

            row = self.get_row(symbol, date)

            if self._check_entry_conditions(symbol, row, current_price, date):
                shares = self.account.get_max_buyable_shares(
                    current_price, self.position_size)
                if shares > 0:
                    orders.append(StockOrder(symbol, OrderOperation.BUY,
                                            shares, current_price, date_str))
                    self.positions[symbol] = {
                        'shares': shares,
                        'entry_date': current_date,
                        'entry_price': current_price,
                    }
                    entries_today += 1
                    print(f"  ENTRY {symbol}: {shares} shares @ ${current_price:.2f}")

        return orders

    # ================================================================== #
    # LIVE SIGNAL GENERATION
    # ================================================================== #

    def run_signals(self, current_prices: pd.DataFrame,
                    trade_date=None, output_path=None):
        """
        Evaluate once for today and write signal CSV (or nothing).

        Call this from a cron job or scheduler. It will:
          - Check exits for any tracked positions
          - Check entries for symbols without positions
          - Write a signal CSV only if there are actions to take

        Args:
            current_prices: DataFrame with [symbol, open] at minimum
            trade_date: Date to evaluate (defaults to today)
            output_path: Where to write signal CSV (None = print only)

        Returns:
            list of signal dicts (may be empty)
        """
        trade_date = trade_date or datetime.now()
        date_str = (trade_date.strftime('%Y-%m-%d')
                    if hasattr(trade_date, 'strftime') else str(trade_date)[:10])
        current_date = pd.to_datetime(trade_date)

        writer = SignalWriter(output_path) if output_path else None
        signals = []
        entries_today = 0

        # PHASE 1: Check exits
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            if pos['shares'] <= 0:
                continue

            price_row = current_prices[current_prices['symbol'] == symbol]
            if len(price_row) == 0:
                continue
            current_price = float(price_row['open'].iloc[0])

            row = self.get_row(symbol, trade_date)
            hold_days = (current_date - pos['entry_date']).days

            if self._check_exit_conditions(symbol, row, current_price,
                                           pos['entry_price'], hold_days, trade_date):
                pct_return = (current_price - pos['entry_price']) / pos['entry_price'] * 100
                reason = f"exit: hold={hold_days}d, pnl={pct_return:+.1f}%"

                sig = {'action': 'SELL', 'symbol': symbol, 'shares': pos['shares'],
                       'price': current_price, 'reason': reason}
                signals.append(sig)

                if writer:
                    writer.add('SELL', symbol, shares=pos['shares'],
                               price=current_price,
                               strategy=self.STRATEGY_NAME, reason=reason)

                # Track exit for re-entry cooldown
                self.recent_exits[symbol] = current_date

                print(f"  SIGNAL SELL {symbol}: {reason}")

        # PHASE 2: Check entries
        for symbol in self.symbols:
            price_row = current_prices[current_prices['symbol'] == symbol]
            if len(price_row) == 0:
                continue
            current_price = float(price_row['open'].iloc[0])

            # Guardrail checks
            allowed, reason = self._can_enter(symbol, current_price,
                                              current_date, entries_today)
            if not allowed:
                continue

            row = self.get_row(symbol, trade_date)

            if self._check_entry_conditions(symbol, row, current_price, trade_date):
                reason = "entry signal"
                sig = {'action': 'BUY', 'symbol': symbol, 'shares': 0,
                       'price': current_price, 'reason': reason}
                signals.append(sig)
                entries_today += 1

                if writer:
                    writer.add('BUY', symbol, shares=0,
                               price=current_price,
                               strategy=self.STRATEGY_NAME, reason=reason)

                print(f"  SIGNAL BUY {symbol} @ ${current_price:.2f}: {reason}")

        if writer and signals:
            writer.save()
        elif not signals:
            print(f"[{self.STRATEGY_NAME}] {date_str}: No signals (hold/do nothing)")

        return signals

    def get_open_positions(self):
        """Return current open positions."""
        return self.positions.copy()
