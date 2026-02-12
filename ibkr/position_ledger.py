"""
Position ledger for tracking live positions with bracket order state.

Persists position state to a JSON file on disk so that:
  - Bracket order IDs (stop-loss + take-profit) survive restarts
  - Trailing stop high-water marks are maintained across sessions
  - The exit monitor can enforce max hold days

positions.json contains ONLY open positions.  When a position is closed
it is removed from positions.json and appended to a daily CSV file at
``signals/executed/closed_YYYYMMDD.csv``.  A new file is created each
trading day; all closes during that day append to the same file.

File location: signals/positions.json (same dir as pending_orders.csv)
"""

import csv
import fcntl
import json
import logging
import os
import shutil
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_LEDGER_PATH = os.path.join(_PROJECT_ROOT, 'signals', 'positions.json')
DEFAULT_CLOSED_DIR = os.path.join(_PROJECT_ROOT, 'signals', 'executed')


def make_ledger_key(symbol: str, instrument_type: str = 'stock',
                    contract_type: str = '', strike: float = 0,
                    expiration: str = '') -> str:
    """Composite key for the position ledger.

    Stocks:  'AAPL'
    Options: 'BITO_C_10.0_20260320'
    """
    if instrument_type == 'option' and contract_type and strike and expiration:
        exp_compact = expiration.replace('-', '')
        right = contract_type[0].upper()  # 'C' or 'P'
        return f"{symbol}_{right}_{strike}_{exp_compact}"
    return symbol


class PositionLedger:
    """
    Persist live position state (entry info, exit params, bracket order IDs)
    to a JSON file on disk.  File-lock safe for concurrent access.

    Invariants:
      - The file is never deleted; ``save()`` always writes, even if empty.
      - ``remove_position()`` removes from positions.json and appends to
        a daily closed CSV (``signals/executed/closed_YYYYMMDD.csv``).
      - The open ``positions`` dict should always mirror actual IBKR holdings.
    """

    def __init__(self, filepath: str = DEFAULT_LEDGER_PATH,
                 closed_dir: str = DEFAULT_CLOSED_DIR):
        self.filepath = filepath
        self.closed_dir = closed_dir
        self._positions: dict[str, dict] = {}
        self._version: int = 1

    def load(self) -> dict[str, dict]:
        """Load positions from JSON file. Returns empty dict if missing."""
        if not os.path.exists(self.filepath):
            self._positions = {}
            return self._positions

        try:
            with open(self.filepath, 'r') as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                data = json.load(f)
                fcntl.flock(f, fcntl.LOCK_UN)

            self._version = data.get('version', 1)
            self._positions = data.get('positions', {})

            # Backfill fields for old entries
            for pos in self._positions.values():
                pos.setdefault('instrument_type', 'stock')
                pos.setdefault('contract_type', '')
                pos.setdefault('strike', 0.0)
                pos.setdefault('expiration', '')
                pos.setdefault('trailing_activation_pct', 0.0)
                pos.setdefault('trailing_active',
                               pos.get('trailing_stop_pct') is not None)

            # Migration: drop closed_positions from old files on next save
        except json.JSONDecodeError as e:
            logger.error(f"Corrupt ledger file {self.filepath}: {e}")
            backup = self.filepath + f'.corrupt.{int(datetime.now().timestamp())}'
            shutil.copy2(self.filepath, backup)
            logger.error(f"Backed up corrupt file to {backup}")
            self._positions = {}
        except Exception as e:
            logger.error(f"Failed to load ledger: {e}")
            self._positions = {}

        return self._positions

    def save(self) -> None:
        """Atomically write positions to JSON file with file lock.

        Always writes the file, even when ``positions`` is empty, so the
        ledger skeleton is never lost.  Only open positions are stored;
        closed positions go to daily CSV files.
        """
        os.makedirs(os.path.dirname(self.filepath) or '.', exist_ok=True)

        data = {
            'version': self._version,
            'updated_at': datetime.now().isoformat(timespec='seconds'),
            'positions': self._positions,
        }

        tmp_path = self.filepath + '.tmp'
        with open(tmp_path, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(data, f, indent=2, default=str)
            fcntl.flock(f, fcntl.LOCK_UN)
        os.replace(tmp_path, self.filepath)

    def add_position(self, symbol: str, entry_price: float, entry_date: str,
                     shares: int, strategy: str,
                     stop_loss_pct: Optional[float], take_profit_pct: Optional[float],
                     trailing_stop_pct: Optional[float], max_hold_days: Optional[int],
                     stop_order_id: int, profit_order_id: int,
                     parent_order_id: int = -1,
                     trailing_activation_pct: float = 0.0,
                     instrument_type: str = 'stock',
                     contract_type: str = '',
                     strike: float = 0.0,
                     expiration: str = '') -> str:
        """Add a new position to the ledger. Returns the ledger key."""
        key = make_ledger_key(symbol, instrument_type, contract_type,
                              strike, expiration)
        if key in self._positions:
            logger.warning(f"Overwriting existing ledger entry for {key}")

        # Compute absolute exit price levels for easy monitoring
        stop_price = round(entry_price * (1 + stop_loss_pct), 4) if stop_loss_pct else None
        take_profit_price = round(entry_price * (1 + take_profit_pct), 4) if take_profit_pct else None

        self._positions[key] = {
            'symbol': symbol,
            'entry_price': entry_price,
            'entry_date': entry_date,
            'shares': shares,
            'strategy': strategy,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'trailing_stop_pct': trailing_stop_pct,
            'trailing_activation_pct': trailing_activation_pct,
            'trailing_active': False,
            'max_hold_days': max_hold_days,
            'stop_price': stop_price,
            'take_profit_price': take_profit_price,
            'stop_order_id': stop_order_id,
            'profit_order_id': profit_order_id,
            'high_water_mark': entry_price,
            'parent_order_id': parent_order_id,
            'instrument_type': instrument_type,
            'contract_type': contract_type,
            'strike': strike,
            'expiration': expiration,
        }
        return key

    def remove_position(self, symbol: str, reason: str = 'unknown',
                        exit_price: Optional[float] = None) -> Optional[dict]:
        """Remove a position from the ledger and append to today's closed CSV.

        The entry is removed from ``positions.json`` and appended to
        ``signals/executed/closed_YYYYMMDD.csv``.  Each day gets its own
        file; all closes during that day append to the same file.

        Args:
            symbol: Ticker or ledger key to close.
            reason: Why the position was closed (e.g. 'bracket_filled',
                    'max_hold', 'manual_close', 'trailing_stop').
            exit_price: Fill price on the exit (if known).

        Returns:
            The closed position dict, or None if symbol wasn't in the ledger.
        """
        entry = self._positions.pop(symbol, None)
        if entry is None:
            return None

        now = datetime.now()
        entry['exit_date'] = now.strftime('%Y-%m-%d')
        entry['exit_time'] = now.isoformat(timespec='seconds')
        entry['exit_reason'] = reason
        if exit_price is not None:
            entry['exit_price'] = exit_price

        self._append_to_daily_closed(entry)
        return entry

    def _append_to_daily_closed(self, entry: dict) -> None:
        """Append a closed position record to today's daily CSV file."""
        os.makedirs(self.closed_dir, exist_ok=True)
        today = datetime.now().strftime('%Y%m%d')
        csv_path = os.path.join(self.closed_dir, f'closed_{today}.csv')

        fieldnames = [
            'symbol', 'instrument_type', 'contract_type', 'strike', 'expiration',
            'strategy', 'entry_price', 'entry_date', 'shares',
            'stop_loss_pct', 'take_profit_pct', 'trailing_stop_pct',
            'trailing_activation_pct', 'trailing_active', 'max_hold_days',
            'stop_price', 'take_profit_price',
            'exit_price', 'exit_date', 'exit_time', 'exit_reason',
            'high_water_mark', 'stop_order_id', 'profit_order_id',
        ]

        write_header = not os.path.exists(csv_path)
        try:
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames,
                                        extrasaction='ignore')
                if write_header:
                    writer.writeheader()
                writer.writerow(entry)
        except Exception as e:
            logger.error(f"Failed to write closed position to {csv_path}: {e}")

    def get_position(self, symbol: str) -> Optional[dict]:
        """Get a single position entry."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> dict[str, dict]:
        """Get all positions."""
        return self._positions.copy()

    def update_stop_order(self, symbol: str, new_stop_order_id: int) -> None:
        """Update the stop order ID after modifying the stop price."""
        if symbol in self._positions:
            self._positions[symbol]['stop_order_id'] = new_stop_order_id

    def update_high_water_mark(self, symbol: str, new_high: float) -> None:
        """Update trailing stop high-water mark."""
        if symbol in self._positions:
            self._positions[symbol]['high_water_mark'] = new_high

    def set_trailing_active(self, key: str, active: bool) -> None:
        """Mark a position's trailing stop as activated."""
        if key in self._positions:
            self._positions[key]['trailing_active'] = active

    def get_expired_positions(self, as_of: Optional[date] = None) -> list[dict]:
        """Return positions that have exceeded their max_hold_days."""
        as_of = as_of or date.today()
        expired = []
        for pos in self._positions.values():
            max_days = pos.get('max_hold_days')
            if max_days is None:
                continue
            entry = datetime.strptime(pos['entry_date'], '%Y-%m-%d').date()
            hold_days = (as_of - entry).days
            if hold_days >= max_days:
                expired.append({**pos, 'hold_days': hold_days})
        return expired

    def get_trailing_stop_positions(self) -> list[dict]:
        """Return positions that have trailing_stop_pct set (non-null)."""
        return [
            pos for pos in self._positions.values()
            if pos.get('trailing_stop_pct') is not None
        ]

    def __len__(self) -> int:
        return len(self._positions)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._positions

    def __repr__(self) -> str:
        return f"PositionLedger({len(self._positions)} open)"
