"""
Base strategy class for bar_fly_trading.

All strategies inherit from BaseStrategy which provides:
  - Overnight data loading (merged_predictions.csv or all_data_*.csv fallback)
  - Realtime API fetching (abstract — each strategy defines its own API calls)
  - Data merge (overnight CSV + live API overlay)
  - Signal scanning (find_signals / run_signals)
  - Backtest interface (evaluate — called by backtest.py)
  - Summary generation and email notifications

Data Model:
  - merged_predictions.csv: PRIMARY (148 cols — technicals + signals + ML)
  - all_data_*.csv:         FALLBACK (89 cols — technicals + signals, no ML)
  - AlphaVantage API:       LIVE overlay (price, BB, RSI per strategy)
"""

import glob as glob_mod
import logging
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

# Add parent dirs to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

from account.account import Account
from order import Order
from signal_writer import SignalWriter

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Subclasses MUST implement:
      - STRATEGY_NAME: str
      - REQUIRED_COLUMNS: list[str]
      - evaluate(date, current_prices, options_data) -> list[Order]
      - check_entry(row) -> bool
      - fetch_realtime(symbol) -> pd.DataFrame

    Subclasses SHOULD override:
      - check_exit(row, hold_days, entry_price) -> (bool, str)
      - entry_reason(row) -> str
    """

    STRATEGY_NAME = "base"
    REQUIRED_COLUMNS = ['date', 'symbol', 'adjusted_close']
    MAX_HOLD_DAYS = 20
    MIN_HOLD_DAYS = 1
    MAX_POSITIONS = 10

    # ── Exit safety defaults (override in subclass) ──────────────────
    STOP_LOSS_PCT = -0.08       # -8% hard stop (backstop)
    TAKE_PROFIT_PCT = 0.15      # +15% take profit (backstop)
    TRAILING_STOP_PCT = None    # None = disabled; e.g. -0.08 for -8%
    TRAILING_ACTIVATION_PCT = 0.0  # Start trailing after position is up this much (0.0 = immediate)

    # ── Instrument type (override in subclass or via runner flag) ──
    INSTRUMENT_TYPE = 'stock'   # 'stock' or 'option'

    # ── Re-entry cooldown (override in subclass) ──────────────────
    REENTRY_COOLDOWN_DAYS = 1   # 1 = no same-day re-entry; 0 = allow immediate

    def __init__(self, account: Account, symbols: set[str]):
        self.account = account
        self.symbols = symbols
        self.positions = {}     # {symbol: {shares, entry_date, entry_price}}
        self.trade_log = []     # completed trades
        self.overnight_data = None  # DataFrame from CSV
        self._trailing_highs = {}  # {symbol: highest price since entry}
        self._recent_exits = {}    # {symbol: exit_date} for cooldown tracking

    # ------------------------------------------------------------------ #
    #  OVERNIGHT DATA LOADING
    # ------------------------------------------------------------------ #

    def load_overnight_data(self, path, filter_symbols=True):
        """
        Load merged_predictions.csv or all_data_*.csv.

        Handles glob patterns, normalizes columns (ticker→symbol,
        close→adjusted_close), optionally filters to self.symbols,
        and validates REQUIRED_COLUMNS.

        Args:
            path: CSV path or glob pattern
            filter_symbols: If True, keep only rows for self.symbols
        """
        df = self._load_csv(path)
        if df is None:
            return

        df = self._normalize_columns(df)

        if filter_symbols and self.symbols:
            df = df[df['symbol'].isin(self.symbols)].copy()

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            print(f"[{self.STRATEGY_NAME}] Warning: Missing columns: {missing}")
            return

        self.overnight_data = df
        print(f"[{self.STRATEGY_NAME}] Loaded overnight data: {len(df):,} rows, "
              f"{df['symbol'].nunique()} symbols, "
              f"{df['date'].min().date()} to {df['date'].max().date()}")

    def load_all_data_fallback(self, path, lookback_days=1):
        """
        Load all_data_*.csv, keep only latest N days per symbol.

        This is the fallback when merged_predictions.csv isn't available.
        No ML columns, but has all technical indicators and signals.
        """
        df = self._load_csv(path)
        if df is None:
            return

        df = self._normalize_columns(df)

        if self.symbols:
            df = df[df['symbol'].isin(self.symbols)].copy()

        # Keep only latest N days per symbol
        if lookback_days and 'date' in df.columns:
            latest_dates = df['date'].drop_duplicates().nlargest(lookback_days)
            df = df[df['date'].isin(latest_dates)].copy()

        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            print(f"[{self.STRATEGY_NAME}] Warning (fallback): Missing columns: {missing}")
            return

        self.overnight_data = df
        print(f"[{self.STRATEGY_NAME}] Loaded fallback data: {len(df):,} rows, "
              f"{df['symbol'].nunique()} symbols (lookback={lookback_days}d)")

    def _load_csv(self, path):
        """Load one or more CSVs from path (supports globs)."""
        if not path:
            return None
        if '*' in str(path):
            files = sorted(glob_mod.glob(str(path)))
            if not files:
                print(f"[{self.STRATEGY_NAME}] Warning: No files matching {path}")
                return None
            dfs = [pd.read_csv(f) for f in files]
            return pd.concat(dfs, ignore_index=True)
        if not os.path.exists(path):
            print(f"[{self.STRATEGY_NAME}] Warning: File not found: {path}")
            return None
        return pd.read_csv(path)

    @staticmethod
    def _normalize_columns(df):
        """Normalize common column naming differences."""
        if 'ticker' in df.columns and 'symbol' not in df.columns:
            df = df.rename(columns={'ticker': 'symbol'})
        if 'close' in df.columns and 'adjusted_close' not in df.columns:
            df = df.rename(columns={'close': 'adjusted_close'})
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df

    # ------------------------------------------------------------------ #
    #  INDICATOR LOOKUP (used by evaluate() during backtest)
    # ------------------------------------------------------------------ #

    def get_indicators(self, symbol, date):
        """
        Get a single row of indicator data for symbol on date.

        Used by evaluate() during backtesting. Returns a Series or None.
        """
        if self.overnight_data is None:
            return None

        date_ts = pd.to_datetime(date)
        mask = ((self.overnight_data['symbol'] == symbol) &
                (self.overnight_data['date'] == date_ts))
        rows = self.overnight_data[mask]

        if len(rows) == 0:
            return None
        return rows.iloc[0]

    # ------------------------------------------------------------------ #
    #  REALTIME API
    # ------------------------------------------------------------------ #

    @abstractmethod
    def fetch_realtime(self, symbol) -> pd.DataFrame:
        """
        Fetch live data for a single symbol from AlphaVantage.

        Each strategy defines which API calls it needs:
          - Bollinger: GLOBAL_QUOTE + BBANDS + RSI (3 calls)
          - Oversold bounce: same as Bollinger
          - Low bounce: GLOBAL_QUOTE + OVERVIEW (2 calls)
          - Regression momentum: GLOBAL_QUOTE only (1 call)

        Returns:
            DataFrame with columns matching overnight_data format.
            Typically 1-2 rows (yesterday + today for crossover detection).
        """

    def fetch_realtime_batch(self, symbols=None) -> pd.DataFrame:
        """
        Fetch realtime data for multiple symbols.

        Uses rt_utils.fetch_realtime_batch() for progress display and
        error handling.
        """
        from api_data.rt_utils import fetch_realtime_batch
        target_symbols = list(symbols or self.symbols)
        return fetch_realtime_batch(target_symbols, self.fetch_realtime)

    # ------------------------------------------------------------------ #
    #  DATA MERGE (overnight + realtime)
    # ------------------------------------------------------------------ #

    def merge_data(self, overnight_df, realtime_df):
        """
        Combine overnight CSV data with live API data.

        Live data overwrites stale fields (price) while preserving
        overnight-only fields (ML predictions, computed signals like
        bull_bear_delta, bollinger_bands_signal, RSI, BB).

        Handles two realtime formats:
          - Bulk quotes: columns [symbol, price, volume, timestamp]
          - Per-symbol fetch: columns matching overnight format

        Args:
            overnight_df: DataFrame from merged_predictions or all_data
            realtime_df: DataFrame from bulk quotes or fetch_realtime_batch

        Returns:
            Merged DataFrame with live prices overlaid on overnight data
        """
        if overnight_df is None or overnight_df.empty:
            return realtime_df
        if realtime_df is None or realtime_df.empty:
            return overnight_df

        # Map bulk quote column names to overnight column names
        col_map = {'price': 'adjusted_close'}

        merged = overnight_df.copy()
        for symbol in realtime_df['symbol'].unique():
            rt_rows = realtime_df[realtime_df['symbol'] == symbol]
            if rt_rows.empty:
                continue

            rt_latest = rt_rows.iloc[-1]

            # Find matching overnight rows for this symbol
            mask = merged['symbol'] == symbol
            if not mask.any():
                continue

            # Overlay realtime values onto the latest overnight row
            latest_idx = merged.loc[mask, 'date'].idxmax()

            # Map and overlay columns
            skip_cols = {'symbol', 'timestamp', 'date'}
            for col in rt_rows.columns:
                if col in skip_cols:
                    continue
                val = rt_latest.get(col)
                if pd.isna(val) or val == 0:
                    continue
                target_col = col_map.get(col, col)
                if target_col in merged.columns:
                    merged.loc[latest_idx, target_col] = val

        return merged

    # ------------------------------------------------------------------ #
    #  BACKTEST INTERFACE (called by backtest.py)
    # ------------------------------------------------------------------ #

    @abstractmethod
    def evaluate(self, date: datetime.date, current_prices: pd.DataFrame,
                 options_data: pd.DataFrame) -> list[Order]:
        """
        Evaluate trading signals for a single day during backtesting.

        Called by backtest.py for each trading day.

        Args:
            date: Current date
            current_prices: DataFrame with [symbol, open, adjusted_close, ...]
            options_data: DataFrame with options data (may be empty)

        Returns:
            List of Order objects to execute
        """

    # ------------------------------------------------------------------ #
    #  SIGNAL SCANNING
    # ------------------------------------------------------------------ #

    @abstractmethod
    def check_entry(self, row) -> bool:
        """
        Check if entry conditions are met for a single data row.

        Args:
            row: pandas Series with indicator/prediction values

        Returns:
            True if all entry conditions are satisfied
        """

    def check_exit(self, row, hold_days, entry_price=None):
        """
        Check if exit conditions are met.

        Base implementation provides max_hold check.
        Override in subclass for strategy-specific exit logic.
        After the subclass check, check_exit_safety() is always called
        as a backstop for stop-loss and take-profit.

        Args:
            row: pandas Series with indicator values (may be None)
            hold_days: Number of days position has been held
            entry_price: Entry price for P&L calculations

        Returns:
            (should_exit: bool, reason: str)
        """
        if hold_days < self.MIN_HOLD_DAYS:
            return False, ""
        if hold_days >= self.MAX_HOLD_DAYS:
            return True, f"max hold {self.MAX_HOLD_DAYS}d"
        return False, ""

    def check_exit_safety(self, symbol, current_price, entry_price):
        """
        Non-overridable safety backstop for stop-loss, take-profit,
        and trailing stop.  Called AFTER strategy-specific check_exit().

        Args:
            symbol: Ticker symbol (for trailing stop tracking)
            current_price: Current market price
            entry_price: Entry price of the position

        Returns:
            (should_exit: bool, reason: str)
        """
        if entry_price is None or entry_price <= 0 or current_price <= 0:
            return False, ""

        pct_change = (current_price - entry_price) / entry_price

        # Hard stop loss
        if self.STOP_LOSS_PCT is not None and pct_change <= self.STOP_LOSS_PCT:
            return True, f"stop_loss ({pct_change:+.1%} <= {self.STOP_LOSS_PCT:.0%})"

        # Take profit
        if self.TAKE_PROFIT_PCT is not None and pct_change >= self.TAKE_PROFIT_PCT:
            return True, f"take_profit ({pct_change:+.1%} >= {self.TAKE_PROFIT_PCT:+.0%})"

        # Trailing stop (with activation threshold + progressive tightening)
        if self.TRAILING_STOP_PCT is not None:
            # B: Only activate trailing after position is up enough
            if pct_change >= self.TRAILING_ACTIVATION_PCT:
                high = self._trailing_highs.get(symbol, entry_price)
                if current_price > high:
                    high = current_price
                    self._trailing_highs[symbol] = high

                # C: Progressive tightening — interpolate trailing pct between
                # TRAILING_STOP_PCT (at activation) and half that (at take profit)
                if self.TAKE_PROFIT_PCT and self.TAKE_PROFIT_PCT > self.TRAILING_ACTIVATION_PCT:
                    progress = (pct_change - self.TRAILING_ACTIVATION_PCT) / (self.TAKE_PROFIT_PCT - self.TRAILING_ACTIVATION_PCT)
                    progress = min(max(progress, 0.0), 1.0)
                    effective_trail = self.TRAILING_STOP_PCT * (1.0 - 0.5 * progress)
                else:
                    effective_trail = self.TRAILING_STOP_PCT

                drawdown = (current_price - high) / high
                if drawdown <= effective_trail:
                    return True, f"trailing_stop ({drawdown:+.1%} from high ${high:.2f}, trail={effective_trail:.1%})"

        return False, ""

    def record_entry(self, symbol):
        """Initialize trailing stop tracking for a new position."""
        if symbol in self.positions:
            self._trailing_highs[symbol] = self.positions[symbol]['entry_price']

    def record_exit(self, symbol, exit_date=None):
        """Clean up trailing stop tracking and record exit for cooldown."""
        self._trailing_highs.pop(symbol, None)
        self._recent_exits[symbol] = exit_date or datetime.now()

    def is_reentry_allowed(self, symbol, current_date=None):
        """
        Check if re-entry is allowed for a symbol based on cooldown period.

        Args:
            symbol: Ticker symbol
            current_date: Current date (datetime or date object)

        Returns:
            (allowed: bool, reason: str)
        """
        if self.REENTRY_COOLDOWN_DAYS <= 0:
            return True, ""

        if symbol not in self._recent_exits:
            return True, ""

        exit_date = self._recent_exits[symbol]
        current = current_date or datetime.now()

        # Normalize to date objects for comparison
        if hasattr(exit_date, 'date'):
            exit_date = exit_date.date()
        if hasattr(current, 'date'):
            current = current.date()

        days_since = (current - exit_date).days
        if days_since < self.REENTRY_COOLDOWN_DAYS:
            return False, (f"cooldown ({days_since}d since exit, "
                           f"need {self.REENTRY_COOLDOWN_DAYS}d)")

        return True, ""

    def entry_reason(self, row):
        """
        Generate human-readable entry reason from a data row.

        Override in subclass for strategy-specific reason strings.
        """
        return f"{self.STRATEGY_NAME} entry"

    def find_signals(self, data=None, lookback_days=2, require_today=False):
        """
        Scan data for entry signals.

        Iterates symbols, gets latest rows within lookback window,
        and calls check_entry() on each.

        Args:
            data: DataFrame to scan (defaults to self.overnight_data)
            lookback_days: Number of trading days to scan
            require_today: If True, only emit signals from rows dated today

        Returns:
            list of signal dicts: {action, symbol, price, reason, date}
        """
        df = data if data is not None else self.overnight_data
        if df is None or df.empty:
            print(f"[{self.STRATEGY_NAME}] No data to scan")
            return []

        today_str = pd.Timestamp.now().strftime('%Y-%m-%d')

        signals = []
        skipped_stale = 0
        for symbol in self.symbols:
            sym_data = df[df['symbol'] == symbol].sort_values('date')
            if sym_data.empty:
                continue

            # Re-entry cooldown check
            allowed, cooldown_reason = self.is_reentry_allowed(symbol)
            if not allowed:
                continue

            # Skip if already holding
            if symbol in self.positions and self.positions[symbol].get('shares', 0) > 0:
                continue

            # Get recent rows within lookback window
            recent = sym_data.tail(lookback_days)
            for _, row in recent.iterrows():
                row_date = str(row.get('date', ''))[:10]

                # Guard: only trade on today's data
                if require_today and row_date != today_str:
                    skipped_stale += 1
                    continue

                if self.check_entry(row):
                    price = row.get('adjusted_close', 0)
                    reason = self.entry_reason(row)
                    signals.append({
                        'action': 'BUY',
                        'symbol': symbol,
                        'price': float(price),
                        'reason': reason,
                        'date': row_date,
                    })

        if require_today and skipped_stale > 0:
            print(f"  [{self.STRATEGY_NAME}] Skipped {skipped_stale} stale rows (not dated {today_str})")

        return signals

    def run_signals(self, current_prices=None, trade_date=None,
                    output_path=None):
        """
        One-shot signal evaluation. Writes signal CSV via SignalWriter.

        Args:
            current_prices: DataFrame with [symbol, open] for live prices
            trade_date: Date to evaluate (defaults to now)
            output_path: Where to write signal CSV (None = print only)

        Returns:
            list of signal dicts
        """
        trade_date = trade_date or datetime.now()
        date_str = (trade_date.strftime('%Y-%m-%d')
                    if hasattr(trade_date, 'strftime') else str(trade_date)[:10])

        signals = self.find_signals()

        # Update prices from current_prices if available
        if current_prices is not None and not current_prices.empty:
            for sig in signals:
                price_row = current_prices[
                    current_prices['symbol'] == sig['symbol']
                ]
                if len(price_row) > 0:
                    sig['price'] = float(price_row['open'].iloc[0])

        # Write signal CSV
        if output_path and signals:
            writer = SignalWriter(output_path)
            for sig in signals:
                writer.add(
                    action=sig['action'],
                    symbol=sig['symbol'],
                    price=sig['price'],
                    strategy=self.STRATEGY_NAME,
                    reason=sig['reason'],
                    stop_loss_pct=self.STOP_LOSS_PCT,
                    take_profit_pct=self.TAKE_PROFIT_PCT,
                    trailing_stop_pct=self.TRAILING_STOP_PCT,
                    max_hold_days=self.MAX_HOLD_DAYS,
                    instrument_type=self.INSTRUMENT_TYPE,
                )
            writer.save()

        # Print results
        if signals:
            for sig in signals:
                print(f"  SIGNAL {sig['action']} {sig['symbol']} "
                      f"@ ${sig['price']:.2f}: {sig['reason']}")
        else:
            print(f"[{self.STRATEGY_NAME}] {date_str}: No signals")

        return signals

    # ------------------------------------------------------------------ #
    #  POSITION TRACKING
    # ------------------------------------------------------------------ #

    def get_open_positions(self):
        """Return current open positions."""
        return self.positions.copy()

    # ------------------------------------------------------------------ #
    #  SUMMARY & NOTIFICATIONS
    # ------------------------------------------------------------------ #

    def generate_signal_summary(self, signals, mode_label="Scan",
                                lookback_days=None, filters=None, ranks=None):
        """
        Format signal table for email/console.

        Args:
            signals: list of signal dicts from find_signals()
            mode_label: "Daily Scan", "Live Scan", etc.
            lookback_days: Days scanned (for display)
            filters: list of filter strings applied
            ranks: list of rank strings applied

        Returns:
            Formatted string
        """
        lines = [
            "=" * 85,
            f"{self.STRATEGY_NAME.upper()} STRATEGY - {mode_label.upper()}",
            "=" * 85,
            f"Scan Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Universe:      {len(self.symbols)} symbols scanned",
        ]

        if lookback_days is not None:
            lines.append(f"Lookback:      {lookback_days} trading day(s)")
        if filters:
            lines.append(f"Filters:       {', '.join(filters)}")
        if ranks:
            lines.append(f"Rank:          {', '.join(ranks)}")
        lines.append("")

        buy_signals = [s for s in signals if s.get('action') == 'BUY']
        sell_signals = [s for s in signals if s.get('action') == 'SELL']

        lines.append(f"BUY signals ({len(buy_signals)}):")
        lines.append("-" * 85)
        if buy_signals:
            lines.append(f"{'Symbol':<8} {'Date':<12} {'Price':>10} {'Reason'}")
            lines.append("-" * 85)
            for s in sorted(buy_signals, key=lambda x: x['symbol']):
                lines.append(
                    f"{s['symbol']:<8} {s.get('date', 'N/A'):<12} "
                    f"${s['price']:>8.2f}   {s.get('reason', '')}"
                )
        else:
            lines.append("  (none)")

        if sell_signals:
            lines.append("")
            lines.append(f"SELL signals ({len(sell_signals)}):")
            lines.append("-" * 85)
            lines.append(f"{'Symbol':<8} {'Date':<12} {'Price':>10} {'Reason'}")
            lines.append("-" * 85)
            for s in sorted(sell_signals, key=lambda x: x['symbol']):
                lines.append(
                    f"{s['symbol']:<8} {s.get('date', 'N/A'):<12} "
                    f"${s['price']:>8.2f}   {s.get('reason', '')}"
                )

        lines.append("")
        lines.append("=" * 85)

        return "\n".join(lines)

    @staticmethod
    def send_notification(subject, body, notifier=None):
        """
        Send email via TradeNotifier.

        Args:
            subject: Email subject line
            body: Email body text
            notifier: TradeNotifier instance (None = skip)

        Returns:
            True if sent successfully, False otherwise
        """
        if notifier is None:
            return False
        if notifier._send_email(subject, body):
            print(f"\nEmail sent: {subject}")
            return True
        else:
            print("\nWarning: Failed to send email")
            return False