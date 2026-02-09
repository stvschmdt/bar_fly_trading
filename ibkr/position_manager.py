"""
Position management for IBKR trading.

Tracks current positions, P&L, and hold durations.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from ib_insync import IB, PortfolioItem

from .connection import IBKRConnection
from .models import Position

logger = logging.getLogger(__name__)


class PositionManager:
    """
    Manages position tracking and P&L calculation.

    Tracks:
    - Current positions from IBKR
    - Entry dates for hold duration
    - Unrealized and realized P&L
    """

    def __init__(self, connection: IBKRConnection):
        """
        Initialize position manager.

        Args:
            connection: IBKR connection instance
        """
        self.connection = connection
        self._positions: dict[str, Position] = {}
        self._entry_dates: dict[str, datetime] = {}  # Track when we entered positions
        self._realized_pnl: dict[str, float] = {}  # Track realized P&L by symbol

    @property
    def ib(self) -> IB:
        """Get IB instance from connection."""
        return self.connection.ib

    def refresh_positions(self) -> dict[str, Position]:
        """
        Refresh positions from IBKR.

        Uses ib.positions() which works across all client sessions, unlike
        ib.portfolio() which only returns data for the current subscription.

        Returns:
            Dict mapping symbol to Position
        """
        if not self.connection.is_connected:
            logger.error("Cannot refresh positions: not connected")
            return self._positions

        try:
            # Use ib.positions() — works across sessions/client IDs.
            # ib.portfolio() is session-specific and often returns empty.
            all_positions = self.ib.positions()

            # Filter to target account if configured
            target_account = self.connection.config.account
            if not target_account:
                # Auto-detect: prefer DU sub-accounts (paper) over DFO (FA master)
                accounts = self.ib.managedAccounts()
                target_account = next((a for a in accounts if a.startswith('DU')), None)
                if not target_account:
                    target_account = next((a for a in accounts if a.startswith('U')), None)

            for item in all_positions:
                if item.contract.secType != "STK":
                    continue  # Only track stocks

                # Filter by account if we have a target
                if target_account and item.account != target_account:
                    continue

                symbol = item.contract.symbol
                shares = int(item.position)
                avg_cost = item.avgCost
                market_value = shares * avg_cost  # Approximate; positions() lacks live mktValue

                position = Position(
                    symbol=symbol,
                    shares=shares,
                    avg_cost=avg_cost,
                    market_value=market_value,
                    unrealized_pnl=0.0,  # Not available from positions()
                    realized_pnl=self._realized_pnl.get(symbol, 0.0),
                    entry_date=self._entry_dates.get(symbol)
                )

                # Track new positions
                if symbol not in self._positions or self._positions[symbol].is_flat:
                    if position.shares != 0:
                        self._entry_dates[symbol] = datetime.now()
                        position.entry_date = self._entry_dates[symbol]
                        logger.info(f"New position detected: {symbol} {position.shares} shares")

                # Clear entry date if position closed
                if position.is_flat and symbol in self._entry_dates:
                    del self._entry_dates[symbol]

                self._positions[symbol] = position

            # Remove positions no longer held
            current_symbols = set()
            for item in all_positions:
                if item.contract.secType != "STK":
                    continue
                if target_account and item.account != target_account:
                    continue
                current_symbols.add(item.contract.symbol)

            for symbol in list(self._positions.keys()):
                if symbol not in current_symbols:
                    if symbol in self._entry_dates:
                        del self._entry_dates[symbol]
                    del self._positions[symbol]

            logger.debug(f"Refreshed {len(self._positions)} positions")
            return self._positions

        except Exception as e:
            logger.error(f"Failed to refresh positions: {e}")
            return self._positions

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.

        Args:
            symbol: Stock ticker

        Returns:
            Position or None if no position
        """
        self.refresh_positions()
        return self._positions.get(symbol)

    def get_all_positions(self) -> dict[str, Position]:
        """Get all current positions."""
        self.refresh_positions()
        return self._positions.copy()

    def get_open_positions(self) -> dict[str, Position]:
        """Get only non-zero positions."""
        self.refresh_positions()
        return {
            symbol: pos for symbol, pos in self._positions.items()
            if not pos.is_flat
        }

    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in a symbol."""
        pos = self.get_position(symbol)
        return pos is not None and not pos.is_flat

    def get_hold_days(self, symbol: str) -> Optional[int]:
        """
        Get number of days a position has been held.

        Args:
            symbol: Stock ticker

        Returns:
            Number of days held, or None if no position
        """
        if symbol not in self._entry_dates:
            return None

        entry = self._entry_dates[symbol]
        delta = datetime.now() - entry
        return delta.days

    def record_entry(self, symbol: str, entry_time: Optional[datetime] = None) -> None:
        """
        Record position entry time.

        Args:
            symbol: Stock ticker
            entry_time: Entry timestamp (uses now if None)
        """
        self._entry_dates[symbol] = entry_time or datetime.now()
        logger.info(f"Recorded entry for {symbol} at {self._entry_dates[symbol]}")

    def record_exit(self, symbol: str, realized_pnl: float) -> None:
        """
        Record position exit and realized P&L.

        Args:
            symbol: Stock ticker
            realized_pnl: P&L from closing position
        """
        if symbol in self._entry_dates:
            hold_days = self.get_hold_days(symbol)
            del self._entry_dates[symbol]
            logger.info(f"Recorded exit for {symbol}, held {hold_days} days, P&L=${realized_pnl:.2f}")

        # Accumulate realized P&L
        if symbol not in self._realized_pnl:
            self._realized_pnl[symbol] = 0.0
        self._realized_pnl[symbol] += realized_pnl

    def get_total_unrealized_pnl(self) -> float:
        """Get total unrealized P&L across all positions."""
        self.refresh_positions()
        return sum(pos.unrealized_pnl for pos in self._positions.values())

    def get_total_realized_pnl(self) -> float:
        """Get total realized P&L."""
        return sum(self._realized_pnl.values())

    def get_total_market_value(self) -> float:
        """Get total market value of all positions."""
        self.refresh_positions()
        return sum(pos.market_value for pos in self._positions.values())

    def get_positions_to_exit(self, max_hold_days: int) -> list[Position]:
        """
        Get positions that have exceeded max hold days.

        Args:
            max_hold_days: Maximum days to hold

        Returns:
            List of positions exceeding hold limit
        """
        positions_to_exit = []
        for symbol, pos in self.get_open_positions().items():
            hold_days = self.get_hold_days(symbol)
            if hold_days is not None and hold_days >= max_hold_days:
                positions_to_exit.append(pos)
                logger.info(f"{symbol} exceeded max hold ({hold_days} >= {max_hold_days} days)")
        return positions_to_exit

    def get_positions_at_min_hold(self, min_hold_days: int) -> list[Position]:
        """
        Get positions that have met minimum hold requirement.

        Args:
            min_hold_days: Minimum days to hold

        Returns:
            List of positions meeting minimum hold
        """
        positions = []
        for symbol, pos in self.get_open_positions().items():
            hold_days = self.get_hold_days(symbol)
            if hold_days is not None and hold_days >= min_hold_days:
                positions.append(pos)
        return positions

    def get_position_summary(self) -> dict:
        """Get summary of current positions."""
        self.refresh_positions()
        open_positions = self.get_open_positions()

        return {
            "total_positions": len(open_positions),
            "total_market_value": self.get_total_market_value(),
            "total_unrealized_pnl": self.get_total_unrealized_pnl(),
            "total_realized_pnl": self.get_total_realized_pnl(),
            "positions": {
                symbol: {
                    "shares": pos.shares,
                    "avg_cost": pos.avg_cost,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "hold_days": self.get_hold_days(symbol)
                }
                for symbol, pos in open_positions.items()
            }
        }

    def print_positions(self) -> None:
        """Print current positions to log."""
        summary = self.get_position_summary()

        logger.info("=" * 60)
        logger.info("CURRENT POSITIONS")
        logger.info("=" * 60)

        for symbol, info in summary["positions"].items():
            logger.info(
                f"  {symbol}: {info['shares']} shares @ ${info['avg_cost']:.2f} "
                f"(P&L: ${info['unrealized_pnl']:.2f}, {info['hold_days']}d)"
            )

        logger.info("-" * 60)
        logger.info(f"  Total Value:    ${summary['total_market_value']:.2f}")
        logger.info(f"  Unrealized P&L: ${summary['total_unrealized_pnl']:.2f}")
        logger.info(f"  Realized P&L:   ${summary['total_realized_pnl']:.2f}")
        logger.info("=" * 60)
