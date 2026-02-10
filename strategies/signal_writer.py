"""
Signal file I/O for bridging strategies to live execution.

The signal CSV is the contract between the strategy layer and the IBKR
execution layer.  Strategy runners write pending signal files; the executor
reads them, places orders, and archives the results.

Signal CSV columns:
    action      - BUY or SELL
    symbol      - ticker symbol
    shares      - number of shares (0 = let executor auto-size)
    price       - signal/reference price (0 = use market price)
    strategy    - strategy name that generated the signal
    reason      - human-readable reason
    timestamp   - ISO 8601 when signal was generated

Usage from a strategy runner:
    from signal_writer import SignalWriter

    writer = SignalWriter("signals/pending_orders.csv")
    writer.add("BUY", "AAPL", strategy="regression_momentum",
               reason="pred_3d=0.032, pred_10d=0.041")
    writer.add("SELL", "NVDA", shares=75, strategy="regression_momentum",
               reason="max_hold_13d")
    writer.save()
"""

import os
from datetime import datetime

import pandas as pd


SIGNAL_COLUMNS = [
    'action', 'symbol', 'shares', 'price', 'strategy', 'reason', 'timestamp',
    'stop_loss_pct', 'take_profit_pct', 'trailing_stop_pct', 'max_hold_days',
]


class SignalWriter:
    """Accumulate trade signals and write to CSV."""

    def __init__(self, filepath=None):
        """
        Args:
            filepath: Output CSV path. If None, must pass to save().
        """
        self.filepath = filepath
        self.signals = []

    def add(self, action, symbol, shares=0, price=0.0, strategy="", reason="",
            stop_loss_pct=None, take_profit_pct=None, trailing_stop_pct=None,
            max_hold_days=None):
        """
        Add a trade signal.

        Args:
            action: "BUY" or "SELL"
            symbol: Ticker symbol
            shares: Number of shares (0 = let executor auto-size from account)
            price: Reference/signal price (0 = use live market price)
            strategy: Strategy name that generated this signal
            reason: Human-readable reason
            stop_loss_pct: Stop loss as decimal (e.g. -0.07 for -7%)
            take_profit_pct: Take profit as decimal (e.g. 0.12 for +12%)
            trailing_stop_pct: Trailing stop as decimal (e.g. -0.08 for -8%)
            max_hold_days: Maximum hold period in trading days
        """
        self.signals.append({
            'action': action.upper(),
            'symbol': symbol.upper(),
            'shares': int(shares),
            'price': round(float(price), 4),
            'strategy': strategy,
            'reason': reason,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'stop_loss_pct': stop_loss_pct if stop_loss_pct is not None else '',
            'take_profit_pct': take_profit_pct if take_profit_pct is not None else '',
            'trailing_stop_pct': trailing_stop_pct if trailing_stop_pct is not None else '',
            'max_hold_days': max_hold_days if max_hold_days is not None else '',
        })

    def save(self, filepath=None, append=False):
        """Write signals to CSV atomically.  Creates parent dirs if needed.

        Args:
            filepath: Output path (uses self.filepath if None)
            append: If True and file exists, append to existing signals
        """
        path = filepath or self.filepath
        if not path:
            raise ValueError("No filepath specified")
        if not self.signals:
            print("No signals to write.")
            return

        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        df = pd.DataFrame(self.signals, columns=SIGNAL_COLUMNS)

        # Append: read existing file and concatenate
        if append and os.path.exists(path):
            try:
                existing = pd.read_csv(path)
                df = pd.concat([existing, df], ignore_index=True)
                print(f"Appending {len(self.signals)} signal(s) to {len(existing)} existing")
            except Exception as e:
                print(f"Warning: could not read existing {path}: {e}, overwriting")

        # Atomic write: write to temp file then rename to prevent
        # partial reads when execute_signals is in watch mode
        tmp_path = path + '.tmp'
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
        print(f"Wrote {len(df)} signal(s) to {path}")

    def clear(self):
        self.signals = []


def read_signals(filepath):
    """
    Read a signal CSV file.

    Returns:
        list of dicts with keys matching SIGNAL_COLUMNS
    """
    if not os.path.exists(filepath):
        return []

    df = pd.read_csv(filepath)
    # Ensure expected columns exist
    for col in ['action', 'symbol']:
        if col not in df.columns:
            raise ValueError(f"Signal file missing required column: {col}")

    # Fill optional columns with defaults
    if 'shares' not in df.columns:
        df['shares'] = 0
    if 'price' not in df.columns:
        df['price'] = 0.0
    if 'strategy' not in df.columns:
        df['strategy'] = ''
    if 'reason' not in df.columns:
        df['reason'] = ''
    if 'timestamp' not in df.columns:
        df['timestamp'] = ''

    # Exit param columns (backward compatible with old CSVs)
    exit_cols = ['stop_loss_pct', 'take_profit_pct', 'trailing_stop_pct', 'max_hold_days']
    for col in exit_cols:
        if col not in df.columns:
            df[col] = ''

    # Pandas reads empty CSV cells as NaN — convert back to '' for exit params
    df[exit_cols] = df[exit_cols].fillna('')

    return df[SIGNAL_COLUMNS].to_dict('records')
