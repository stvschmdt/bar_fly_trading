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

    def add(self, action, symbol, shares=0, price=0.0, strategy="", reason=""):
        """
        Add a trade signal.

        Args:
            action: "BUY" or "SELL"
            symbol: Ticker symbol
            shares: Number of shares (0 = let executor auto-size from account)
            price: Reference/signal price (0 = use live market price)
            strategy: Strategy name that generated this signal
            reason: Human-readable reason
        """
        self.signals.append({
            'action': action.upper(),
            'symbol': symbol.upper(),
            'shares': int(shares),
            'price': round(float(price), 4),
            'strategy': strategy,
            'reason': reason,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        })

    def save(self, filepath=None):
        """Write signals to CSV.  Creates parent directories if needed."""
        path = filepath or self.filepath
        if not path:
            raise ValueError("No filepath specified")
        if not self.signals:
            print("No signals to write.")
            return

        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        df = pd.DataFrame(self.signals, columns=SIGNAL_COLUMNS)
        df.to_csv(path, index=False)
        print(f"Wrote {len(self.signals)} signal(s) to {path}")

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

    return df[SIGNAL_COLUMNS].to_dict('records')
