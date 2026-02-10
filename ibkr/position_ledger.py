"""
Position ledger for tracking live positions with bracket order state.

Persists position state to a JSON file on disk so that:
  - Bracket order IDs (stop-loss + take-profit) survive restarts
  - Trailing stop high-water marks are maintained across sessions
  - The exit monitor can enforce max hold days
  - Closed positions are archived (never deleted) for audit trail

The ledger file is NEVER deleted — even with zero open positions, the
skeleton JSON persists.  Only the executor and exit monitor should write
to this file.  Closing a position archives it to ``closed_positions``
rather than discarding it.

File location: signals/positions.json (same dir as pending_orders.csv)
"""

import fcntl
import json
import logging
import os
import shutil
from datetime import date, datetime
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_LEDGER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'signals', 'positions.json'
)


class PositionLedger:
    """
    Persist live position state (entry info, exit params, bracket order IDs)
    to a JSON file on disk.  File-lock safe for concurrent access.

    Invariants:
      - The file is never deleted; ``save()`` always writes, even if empty.
      - ``remove_position()`` archives to ``closed_positions`` (never discards).
      - The open ``positions`` dict should always mirror actual IBKR holdings.
    """

    def __init__(self, filepath: str = DEFAULT_LEDGER_PATH):
        self.filepath = filepath
        self._positions: dict[str, dict] = {}
        self._closed: list[dict] = []
        self._version: int = 1

    def load(self) -> dict[str, dict]:
        """Load positions from JSON file. Returns empty dict if missing."""
        if not os.path.exists(self.filepath):
            self._positions = {}
            self._closed = []
            return self._positions

        try:
            with open(self.filepath, 'r') as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                data = json.load(f)
                fcntl.flock(f, fcntl.LOCK_UN)

            self._version = data.get('version', 1)
            self._positions = data.get('positions', {})
            self._closed = data.get('closed_positions', [])
        except json.JSONDecodeError as e:
            logger.error(f"Corrupt ledger file {self.filepath}: {e}")
            # Back up the corrupt file
            backup = self.filepath + f'.corrupt.{int(datetime.now().timestamp())}'
            shutil.copy2(self.filepath, backup)
            logger.error(f"Backed up corrupt file to {backup}")
            self._positions = {}
            self._closed = []
        except Exception as e:
            logger.error(f"Failed to load ledger: {e}")
            self._positions = {}
            self._closed = []

        return self._positions

    def save(self) -> None:
        """Atomically write positions to JSON file with file lock.

        Always writes the file, even when ``positions`` is empty, so the
        ledger skeleton is never lost.
        """
        os.makedirs(os.path.dirname(self.filepath) or '.', exist_ok=True)

        data = {
            'version': self._version,
            'updated_at': datetime.now().isoformat(timespec='seconds'),
            'positions': self._positions,
            'closed_positions': self._closed,
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
                     parent_order_id: int = -1) -> None:
        """Add a new position to the ledger."""
        if symbol in self._positions:
            logger.warning(f"Overwriting existing ledger entry for {symbol}")

        self._positions[symbol] = {
            'symbol': symbol,
            'entry_price': entry_price,
            'entry_date': entry_date,
            'shares': shares,
            'strategy': strategy,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'trailing_stop_pct': trailing_stop_pct,
            'max_hold_days': max_hold_days,
            'stop_order_id': stop_order_id,
            'profit_order_id': profit_order_id,
            'high_water_mark': entry_price,
            'parent_order_id': parent_order_id,
        }

    def remove_position(self, symbol: str, reason: str = 'unknown',
                        exit_price: Optional[float] = None) -> Optional[dict]:
        """Archive a position to closed_positions and remove from active.

        The entry is NEVER discarded — it moves to the ``closed_positions``
        list with exit metadata so the ledger maintains a full audit trail.

        Args:
            symbol: Ticker to close.
            reason: Why the position was closed (e.g. 'bracket_filled',
                    'max_hold', 'manual_close', 'trailing_stop').
            exit_price: Fill price on the exit (if known).

        Returns:
            The archived position dict, or None if symbol wasn't in the ledger.
        """
        entry = self._positions.pop(symbol, None)
        if entry is None:
            return None

        entry['exit_date'] = datetime.now().strftime('%Y-%m-%d')
        entry['exit_time'] = datetime.now().isoformat(timespec='seconds')
        entry['exit_reason'] = reason
        if exit_price is not None:
            entry['exit_price'] = exit_price

        self._closed.append(entry)
        return entry

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

    def get_closed_positions(self) -> list[dict]:
        """Return full history of closed positions (oldest first)."""
        return list(self._closed)

    def prune_closed_before(self, cutoff_date: str) -> int:
        """Remove closed position records older than *cutoff_date* (YYYY-MM-DD).

        Returns the number of records pruned.
        """
        before = len(self._closed)
        self._closed = [
            c for c in self._closed
            if c.get('exit_date', '9999-99-99') >= cutoff_date
        ]
        return before - len(self._closed)

    def __len__(self) -> int:
        return len(self._positions)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self._positions

    def __repr__(self) -> str:
        return (f"PositionLedger({len(self._positions)} open, "
                f"{len(self._closed)} closed)")
