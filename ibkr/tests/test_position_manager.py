"""Unit tests for IBKR position manager."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch

from ibkr.connection import IBKRConnection
from ibkr.position_manager import PositionManager
from ibkr.models import Position


@pytest.fixture
def mock_connection():
    """Create mock IBKR connection."""
    connection = Mock(spec=IBKRConnection)
    connection.is_connected = True
    connection.ib = MagicMock()
    return connection


@pytest.fixture
def position_manager(mock_connection):
    """Create position manager with mock connection."""
    with patch.object(PositionManager, '__init__', lambda self, conn: None):
        manager = PositionManager.__new__(PositionManager)
        manager.connection = mock_connection
        manager._positions = {}
        manager._entry_dates = {}
        manager._realized_pnl = {}
    return manager


class TestPositionTracking:
    """Tests for basic position tracking."""

    def test_no_position_initially(self, position_manager):
        """Test no position exists initially."""
        assert not position_manager.has_position("AAPL")

    def test_get_position_not_found(self, position_manager):
        """Test getting non-existent position."""
        # Need to mock refresh_positions to not call IBKR
        position_manager.refresh_positions = Mock(return_value={})
        pos = position_manager.get_position("AAPL")
        assert pos is None

    def test_get_all_positions_empty(self, position_manager):
        """Test getting all positions when empty."""
        position_manager.refresh_positions = Mock(return_value={})
        positions = position_manager.get_all_positions()
        assert positions == {}

    def test_get_open_positions(self, position_manager):
        """Test getting only open positions."""
        position_manager._positions = {
            "AAPL": Position("AAPL", 100, 150.0),
            "GOOGL": Position("GOOGL", 0, 0.0),  # Flat position
            "MSFT": Position("MSFT", 50, 300.0)
        }
        position_manager.refresh_positions = Mock(return_value=position_manager._positions)

        open_pos = position_manager.get_open_positions()
        assert "AAPL" in open_pos
        assert "MSFT" in open_pos
        assert "GOOGL" not in open_pos  # Flat position excluded


class TestEntryTracking:
    """Tests for position entry tracking."""

    def test_record_entry(self, position_manager):
        """Test recording position entry."""
        position_manager.record_entry("AAPL")
        assert "AAPL" in position_manager._entry_dates
        assert isinstance(position_manager._entry_dates["AAPL"], datetime)

    def test_record_entry_with_time(self, position_manager):
        """Test recording entry with specific time."""
        entry_time = datetime(2024, 1, 15, 10, 30)
        position_manager.record_entry("GOOGL", entry_time)
        assert position_manager._entry_dates["GOOGL"] == entry_time

    def test_get_hold_days(self, position_manager):
        """Test getting hold days for position."""
        # Set entry 5 days ago
        entry_time = datetime.now() - timedelta(days=5)
        position_manager._entry_dates["AAPL"] = entry_time

        hold_days = position_manager.get_hold_days("AAPL")
        assert hold_days == 5

    def test_get_hold_days_no_position(self, position_manager):
        """Test getting hold days for non-existent position."""
        hold_days = position_manager.get_hold_days("UNKNOWN")
        assert hold_days is None


class TestExitTracking:
    """Tests for position exit tracking."""

    def test_record_exit(self, position_manager):
        """Test recording position exit."""
        # First record entry
        position_manager.record_entry("AAPL")
        assert "AAPL" in position_manager._entry_dates

        # Record exit
        position_manager.record_exit("AAPL", 500.0)

        # Entry date should be cleared
        assert "AAPL" not in position_manager._entry_dates
        # Realized P&L should be tracked
        assert position_manager._realized_pnl["AAPL"] == 500.0

    def test_accumulate_realized_pnl(self, position_manager):
        """Test accumulating realized P&L across trades."""
        position_manager.record_exit("AAPL", 500.0)
        position_manager.record_exit("AAPL", -200.0)
        position_manager.record_exit("AAPL", 300.0)

        assert position_manager._realized_pnl["AAPL"] == 600.0


class TestPnLCalculations:
    """Tests for P&L calculations."""

    def test_total_unrealized_pnl(self, position_manager):
        """Test total unrealized P&L calculation."""
        position_manager._positions = {
            "AAPL": Position("AAPL", 100, 150.0, unrealized_pnl=500.0),
            "GOOGL": Position("GOOGL", 50, 2800.0, unrealized_pnl=-200.0)
        }
        position_manager.refresh_positions = Mock(return_value=position_manager._positions)

        total = position_manager.get_total_unrealized_pnl()
        assert total == 300.0

    def test_total_realized_pnl(self, position_manager):
        """Test total realized P&L calculation."""
        position_manager._realized_pnl = {
            "AAPL": 500.0,
            "GOOGL": -200.0,
            "MSFT": 300.0
        }

        total = position_manager.get_total_realized_pnl()
        assert total == 600.0

    def test_total_market_value(self, position_manager):
        """Test total market value calculation."""
        position_manager._positions = {
            "AAPL": Position("AAPL", 100, 150.0, market_value=15000.0),
            "GOOGL": Position("GOOGL", 50, 2800.0, market_value=140000.0)
        }
        position_manager.refresh_positions = Mock(return_value=position_manager._positions)

        total = position_manager.get_total_market_value()
        assert total == 155000.0


class TestHoldDayFilters:
    """Tests for filtering positions by hold days."""

    def test_positions_to_exit_max_hold(self, position_manager):
        """Test finding positions exceeding max hold days."""
        position_manager._positions = {
            "AAPL": Position("AAPL", 100, 150.0),
            "GOOGL": Position("GOOGL", 50, 2800.0),
            "MSFT": Position("MSFT", 75, 300.0)
        }
        # AAPL held 15 days (should exit at 13)
        position_manager._entry_dates["AAPL"] = datetime.now() - timedelta(days=15)
        # GOOGL held 5 days (keep)
        position_manager._entry_dates["GOOGL"] = datetime.now() - timedelta(days=5)
        # MSFT held 14 days (should exit)
        position_manager._entry_dates["MSFT"] = datetime.now() - timedelta(days=14)

        position_manager.refresh_positions = Mock(return_value=position_manager._positions)
        position_manager.get_open_positions = Mock(return_value=position_manager._positions)

        to_exit = position_manager.get_positions_to_exit(max_hold_days=13)
        symbols = [p.symbol for p in to_exit]
        assert "AAPL" in symbols
        assert "MSFT" in symbols
        assert "GOOGL" not in symbols

    def test_positions_at_min_hold(self, position_manager):
        """Test finding positions meeting minimum hold."""
        position_manager._positions = {
            "AAPL": Position("AAPL", 100, 150.0),
            "GOOGL": Position("GOOGL", 50, 2800.0),
        }
        # AAPL held 3 days (meets min 2)
        position_manager._entry_dates["AAPL"] = datetime.now() - timedelta(days=3)
        # GOOGL held 1 day (doesn't meet min 2)
        position_manager._entry_dates["GOOGL"] = datetime.now() - timedelta(days=1)

        position_manager.refresh_positions = Mock(return_value=position_manager._positions)
        position_manager.get_open_positions = Mock(return_value=position_manager._positions)

        at_min = position_manager.get_positions_at_min_hold(min_hold_days=2)
        symbols = [p.symbol for p in at_min]
        assert "AAPL" in symbols
        assert "GOOGL" not in symbols


class TestPositionSummary:
    """Tests for position summary generation."""

    def test_position_summary(self, position_manager):
        """Test generating position summary."""
        position_manager._positions = {
            "AAPL": Position("AAPL", 100, 150.0, market_value=15500.0, unrealized_pnl=500.0)
        }
        position_manager._entry_dates["AAPL"] = datetime.now() - timedelta(days=5)
        position_manager._realized_pnl = {"AAPL": 1000.0}

        position_manager.refresh_positions = Mock(return_value=position_manager._positions)
        position_manager.get_open_positions = Mock(return_value=position_manager._positions)

        summary = position_manager.get_position_summary()

        assert summary["total_positions"] == 1
        assert summary["total_market_value"] == 15500.0
        assert summary["total_unrealized_pnl"] == 500.0
        assert summary["total_realized_pnl"] == 1000.0
        assert "AAPL" in summary["positions"]
        assert summary["positions"]["AAPL"]["hold_days"] == 5
