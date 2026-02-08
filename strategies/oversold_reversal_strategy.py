"""
Oversold Reversal Strategy for bar_fly_trading backtest framework.

Best-performing strategy from v2 analysis with 63% win rate, Sharpe 0.84.

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
"""

import os
import sys
from datetime import datetime, timedelta

import pandas as pd

# Add project root and strategies dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from order import Order, StockOrder, OrderOperation
from base_strategy import BaseStrategy
from signal_writer import SignalWriter


# Column name mappings for merged vs individual prediction files
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

    The model acts as a filter — when technicals say "oversold" but the
    binary model still predicts UP, that's a signal the reversion will hold.
    """

    STRATEGY_NAME = "oversold_reversal"

    # Entry thresholds
    RSI_ENTRY_THRESHOLD = 40        # RSI must be below this (oversold)
    DELTA_ENTRY_THRESHOLD = -2      # bull_bear_delta must be <= this (bearish)
    PROB_UP_THRESHOLD = 0.50        # model probability of UP (default: majority class)

    # Exit thresholds
    RSI_EXIT_THRESHOLD = 55         # Early exit if RSI recovers above this
    MAX_HOLD_DAYS = 3               # Fixed 3-day hold period
    MIN_HOLD_DAYS = 1               # Must hold at least 1 day
    MAX_POSITIONS = 10              # Max concurrent positions (10 x 10% = 100%)

    def __init__(self, account, symbols, predictions_path=None, position_size=0.1):
        """
        Initialize the strategy.

        Args:
            account: BacktestAccount instance
            symbols: Set of symbols to trade
            predictions_path: Path to merged_predictions.csv or predictions directory
            position_size: Fraction of portfolio per position (default 0.1 = 10%)
        """
        super().__init__(account, symbols)
        self.predictions_path = predictions_path
        self.position_size = position_size

        # Track positions: {symbol: {'shares': int, 'entry_date': date, 'entry_price': float}}
        self.positions = {}

        # Completed trade log for stats
        self.trade_log = []

        # Load predictions if path provided
        self.predictions = None
        if predictions_path:
            self._load_predictions()

    def _load_predictions(self):
        """Load predictions from a merged CSV or predictions directory."""
        path = self.predictions_path
        if not path:
            return

        if os.path.isfile(path):
            self._load_from_merged(path)
        elif os.path.isdir(path):
            self._load_from_dir(path)
        else:
            print(f"[OversoldReversal] Warning: Path not found: {path}")

    def _load_from_merged(self, csv_path):
        """Load predictions from a single merged_predictions.csv file."""
        df = pd.read_csv(csv_path)

        # Standardize columns
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        if 'ticker' in df.columns and 'symbol' not in df.columns:
            df['symbol'] = df['ticker']

        # Map merged column names to strategy's internal names
        rename = {}
        for src, dst in MERGED_COLUMN_MAP.items():
            if src in df.columns:
                rename[src] = dst

        if not rename:
            print(f"[OversoldReversal] Warning: No prediction columns found in {csv_path}")
            print(f"  Expected columns like: {list(MERGED_COLUMN_MAP.keys())}")
            return

        df = df.rename(columns=rename)

        # Keep needed columns (technical indicators + model predictions)
        needed = ['date', 'symbol', 'rsi_14', 'bull_bear_delta', 'pred_bin_3d', 'prob_up_3d']
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"[OversoldReversal] Warning: Missing columns: {missing}")
            return

        self.predictions = df[needed].copy()
        self.predictions = self.predictions[self.predictions['symbol'].isin(self.symbols)]

        print(f"[OversoldReversal] Loaded from merged file: {csv_path}")
        print(f"  Rows: {len(self.predictions):,}")
        print(f"  Symbols: {self.predictions['symbol'].nunique()}")
        print(f"  Date range: {self.predictions['date'].min()} to {self.predictions['date'].max()}")

    def _load_from_dir(self, predictions_dir):
        """Load predictions from individual prediction files in a directory."""
        bin_3d_path = os.path.join(predictions_dir, 'predictions_bin_3d.csv')
        if not os.path.exists(bin_3d_path):
            bin_3d_path = os.path.join(predictions_dir, 'pred_bin_3d.csv')

        if not os.path.exists(bin_3d_path):
            print(f"[OversoldReversal] Warning: Binary 3d prediction file not found")
            print(f"  Checked: {predictions_dir}/predictions_bin_3d.csv")
            return

        df = pd.read_csv(bin_3d_path)
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        if 'ticker' in df.columns:
            df['symbol'] = df['ticker']

        # Map column names
        rename = {}
        if 'pred_class' in df.columns:
            rename['pred_class'] = 'pred_bin_3d'
        if 'pred_class_bin_3d' in df.columns:
            rename['pred_class_bin_3d'] = 'pred_bin_3d'
        if 'prob_1' in df.columns:
            rename['prob_1'] = 'prob_up_3d'
        if 'prob_1_bin_3d' in df.columns:
            rename['prob_1_bin_3d'] = 'prob_up_3d'

        df = df.rename(columns=rename)

        needed = ['date', 'symbol', 'rsi_14', 'bull_bear_delta', 'pred_bin_3d', 'prob_up_3d']
        missing = [c for c in needed if c not in df.columns]
        if missing:
            print(f"[OversoldReversal] Warning: Missing columns in dir: {missing}")
            return

        self.predictions = df[needed].copy()
        self.predictions = self.predictions[self.predictions['symbol'].isin(self.symbols)]

        print(f"[OversoldReversal] Loaded from directory: {predictions_dir}")
        print(f"  Rows: {len(self.predictions):,}")
        print(f"  Symbols: {self.predictions['symbol'].nunique()}")
        print(f"  Date range: {self.predictions['date'].min()} to {self.predictions['date'].max()}")

    def get_prediction(self, symbol, date):
        """Get prediction data for a symbol on a specific date."""
        if self.predictions is None:
            return None

        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]
        mask = (self.predictions['date'] == date_str) & (self.predictions['symbol'] == symbol)
        rows = self.predictions[mask]

        if len(rows) == 0:
            return None

        return rows.iloc[0]

    def _check_entry_conditions(self, pred):
        """
        Check if entry conditions are met.

        Entry: RSI < 40 AND bull_bear_delta <= -2 AND bin_3d predicts UP
        """
        if pred is None:
            return False

        rsi = pred['rsi_14']
        delta = pred['bull_bear_delta']
        model_up = pred['pred_bin_3d']

        # All three conditions must be true
        return (rsi < self.RSI_ENTRY_THRESHOLD and
                delta <= self.DELTA_ENTRY_THRESHOLD and
                model_up == 1)

    def _check_exit_conditions(self, pred, hold_days):
        """
        Check if exit conditions are met.

        Exit: hold >= 3 days OR (hold >= 1 day AND RSI > 55)
        """
        if hold_days < self.MIN_HOLD_DAYS:
            return False

        # Always exit at max hold
        if hold_days >= self.MAX_HOLD_DAYS:
            return True

        # Early exit if RSI recovered
        if pred is not None:
            rsi = pred['rsi_14']
            if rsi > self.RSI_EXIT_THRESHOLD:
                return True

        return False

    def evaluate(self, date: datetime.date, current_prices: pd.DataFrame, options_data: pd.DataFrame) -> list[Order]:
        """
        Evaluate trading signals for the day.

        Called by backtest.py for each trading day.

        Args:
            date: Current date
            current_prices: DataFrame with columns [symbol, open, adjusted_close, high, low]
            options_data: DataFrame with options data (not used)

        Returns:
            List of Order objects to execute
        """
        orders = []
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]
        current_date = pd.to_datetime(date)

        for symbol in self.symbols:
            # Get current price
            price_row = current_prices[current_prices['symbol'] == symbol]
            if len(price_row) == 0:
                continue
            current_price = float(price_row['open'].iloc[0])

            # Get prediction for today
            pred = self.get_prediction(symbol, date)

            # Check if we have a position
            has_position = symbol in self.positions and self.positions[symbol]['shares'] > 0

            if has_position:
                # Calculate hold days
                entry_date = self.positions[symbol]['entry_date']
                hold_days = (current_date - entry_date).days

                # Check exit conditions
                if self._check_exit_conditions(pred, hold_days):
                    shares = self.positions[symbol]['shares']
                    entry_price = self.positions[symbol]['entry_price']
                    pct_return = (current_price - entry_price) / entry_price * 100

                    exit_reason = "max hold 3d"
                    if pred is not None and pred['rsi_14'] > self.RSI_EXIT_THRESHOLD:
                        exit_reason = f"RSI recovered ({pred['rsi_14']:.1f})"

                    orders.append(StockOrder(symbol, OrderOperation.SELL, shares, current_price, date_str))

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
                        'exit_reason': exit_reason,
                    })

                    print(f"  EXIT {symbol}: held {hold_days}d, return: {pct_return:+.2f}% ({exit_reason})")

                    # Clear position
                    del self.positions[symbol]

            else:
                # Check entry conditions + position/cash guards
                if len(self.positions) >= self.MAX_POSITIONS:
                    continue  # At capacity, skip new entries but keep checking exits
                if self._check_entry_conditions(pred):
                    if self.account.account_values.cash_balance < current_price:
                        continue
                    shares = self.account.get_max_buyable_shares(current_price, self.position_size)
                    if shares > 0:
                        orders.append(StockOrder(symbol, OrderOperation.BUY, shares, current_price, date_str))

                        # Track position
                        self.positions[symbol] = {
                            'shares': shares,
                            'entry_date': current_date,
                            'entry_price': current_price
                        }

                        print(f"  ENTRY {symbol}: RSI={pred['rsi_14']:.1f}, delta={pred['bull_bear_delta']:.0f}, prob_up={pred['prob_up_3d']:.3f}")

        return orders

    def run_signals(self, current_prices, trade_date=None, output_path=None):
        """
        One-shot signal evaluation for today. Writes signal CSV if any triggers.

        Used by --mode signals for cron/scheduler integration.

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

        writer = SignalWriter(output_path) if output_path else None
        signals = []

        for symbol in self.symbols:
            price_row = current_prices[current_prices['symbol'] == symbol]
            if len(price_row) == 0:
                continue
            current_price = float(price_row['open'].iloc[0])

            pred = self.get_prediction(symbol, trade_date)

            if self._check_entry_conditions(pred):
                reason = (
                    f"RSI={pred['rsi_14']:.1f}, "
                    f"delta={pred['bull_bear_delta']:.0f}, "
                    f"prob_up={pred['prob_up_3d']:.3f}"
                )
                sig = {'action': 'BUY', 'symbol': symbol, 'shares': 0,
                       'price': current_price, 'reason': reason}
                signals.append(sig)

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
