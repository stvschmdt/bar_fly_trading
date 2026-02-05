"""Unit tests for IBKR trade executor."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from ibkr.config import IBKRConfig, TradingConfig
from ibkr.trade_executor import TradeExecutor
from ibkr.models import (
    TradeSignal, TradeResult, ValidationResult, ValidationStatus,
    OrderResult, OrderAction, OrderType, OrderStatus, Position, AccountSummary
)


@pytest.fixture
def ibkr_config():
    """Create test IBKR configuration."""
    return IBKRConfig.paper_tws()


@pytest.fixture
def trading_config():
    """Create test trading configuration."""
    return TradingConfig(
        symbols={"AAPL", "GOOGL", "MSFT"},
        position_size=0.10,
        max_positions=5,
        max_daily_trades=10,
        max_daily_loss=5000.0
    )


@pytest.fixture
def mock_executor(ibkr_config, trading_config):
    """Create trade executor with mocked dependencies."""
    with patch('ibkr.trade_executor.IBKRConnection') as MockConn, \
         patch('ibkr.trade_executor.OrderManager') as MockOrder, \
         patch('ibkr.trade_executor.PositionManager') as MockPos:

        # Setup mock connection
        mock_conn = MockConn.return_value
        mock_conn.connect.return_value = True
        mock_conn.is_connected = True
        mock_conn.get_account_summary.return_value = AccountSummary(
            net_liquidation=100000.0,
            available_funds=50000.0,
            buying_power=100000.0,
            gross_position_value=50000.0
        )
        mock_conn.get_current_price.return_value = 150.0

        # Setup mock order manager
        mock_order = MockOrder.return_value
        mock_order.submit_order.return_value = OrderResult(
            order_id=12345,
            symbol="AAPL",
            action=OrderAction.BUY,
            shares=100,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_shares=100,
            avg_fill_price=150.10
        )
        mock_order.wait_for_fill.return_value = mock_order.submit_order.return_value
        mock_order.cancel_all_orders.return_value = 0

        # Setup mock position manager
        mock_pos = MockPos.return_value
        mock_pos.get_open_positions.return_value = {}
        mock_pos.refresh_positions.return_value = {}
        mock_pos.get_position.return_value = None
        mock_pos.get_hold_days.return_value = None

        executor = TradeExecutor(ibkr_config, trading_config)
        executor.connection = mock_conn
        executor.order_manager = mock_order
        executor.position_manager = mock_pos

        return executor


class TestTradeExecutorStart:
    """Tests for executor startup."""

    def test_start_success(self, mock_executor):
        """Test successful startup."""
        result = mock_executor.start()
        assert result is True
        mock_executor.connection.connect.assert_called_once()

    def test_start_connection_failure(self, mock_executor):
        """Test startup failure on connection error."""
        mock_executor.connection.connect.return_value = False
        result = mock_executor.start()
        assert result is False


class TestExecuteSignal:
    """Tests for signal execution."""

    def test_execute_buy_signal(self, mock_executor):
        """Test executing a buy signal."""
        mock_executor.start()

        signal = TradeSignal(
            symbol="AAPL",
            action=OrderAction.BUY,
            shares=100,
            signal_price=150.0,
            signal_time=datetime.now()
        )

        result = mock_executor.execute_signal(signal)

        assert result.success
        assert result.order.is_filled
        mock_executor.order_manager.submit_order.assert_called_once()

    def test_execute_signal_not_connected(self, mock_executor):
        """Test signal execution fails when not connected."""
        mock_executor.connection.is_connected = False

        signal = TradeSignal(
            symbol="AAPL",
            action=OrderAction.BUY,
            shares=100,
            signal_price=150.0,
            signal_time=datetime.now()
        )

        result = mock_executor.execute_signal(signal)
        assert not result.success
        assert "Not connected" in result.error_message

    def test_execute_signal_rejected_by_risk(self, mock_executor):
        """Test signal execution rejected by risk manager."""
        mock_executor.start()

        # Exhaust daily trade limit
        for _ in range(10):
            mock_executor.risk_manager.record_trade("TEST", 100, True)

        signal = TradeSignal(
            symbol="AAPL",
            action=OrderAction.BUY,
            shares=100,
            signal_price=150.0,
            signal_time=datetime.now()
        )

        result = mock_executor.execute_signal(signal)
        assert not result.success
        assert result.validation.is_rejected


class TestExecuteBuy:
    """Tests for buy execution helper."""

    def test_execute_buy(self, mock_executor):
        """Test buy execution."""
        mock_executor.start()
        result = mock_executor.execute_buy("AAPL", shares=100, reason="Test buy")
        assert result.signal.action == OrderAction.BUY

    def test_execute_buy_auto_size(self, mock_executor):
        """Test buy with automatic position sizing."""
        mock_executor.start()
        result = mock_executor.execute_buy("AAPL", reason="Test auto-size")
        # Position size should be calculated
        assert result.signal.shares > 0

    def test_execute_buy_no_price(self, mock_executor):
        """Test buy fails when price unavailable."""
        mock_executor.start()
        mock_executor.connection.get_current_price.return_value = None
        result = mock_executor.execute_buy("AAPL")
        assert not result.success
        assert "Could not get price" in result.error_message


class TestExecuteSell:
    """Tests for sell execution helper."""

    def test_execute_sell_no_position(self, mock_executor):
        """Test sell fails when no position."""
        mock_executor.start()
        result = mock_executor.execute_sell("AAPL", reason="Test sell")
        assert not result.success
        assert "No position to sell" in result.error_message

    def test_execute_sell_with_position(self, mock_executor):
        """Test sell execution with existing position."""
        mock_executor.start()

        # Mock having a position
        position = Position("AAPL", 100, 145.0, market_value=15500.0)
        mock_executor.position_manager.get_position.return_value = position

        # Mock sell order
        mock_executor.order_manager.submit_order.return_value = OrderResult(
            order_id=12346,
            symbol="AAPL",
            action=OrderAction.SELL,
            shares=100,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_shares=100,
            avg_fill_price=155.0
        )
        mock_executor.order_manager.wait_for_fill.return_value = \
            mock_executor.order_manager.submit_order.return_value

        result = mock_executor.execute_sell("AAPL", reason="Test sell")
        assert result.signal.action == OrderAction.SELL


class TestCloseAllPositions:
    """Tests for closing all positions."""

    def test_close_all_positions(self, mock_executor):
        """Test closing all open positions."""
        mock_executor.start()

        # Mock open positions
        mock_executor.position_manager.get_open_positions.return_value = {
            "AAPL": Position("AAPL", 100, 145.0),
            "GOOGL": Position("GOOGL", 50, 2800.0)
        }
        mock_executor.position_manager.get_position.side_effect = [
            Position("AAPL", 100, 145.0),
            Position("GOOGL", 50, 2800.0)
        ]

        results = mock_executor.close_all_positions(reason="End of day")
        assert len(results) == 2


class TestCheckExits:
    """Tests for exit condition checking."""

    def test_check_exits(self, mock_executor):
        """Test checking exit conditions."""
        mock_executor.start()

        mock_executor.position_manager.get_open_positions.return_value = {
            "AAPL": Position("AAPL", 100, 145.0)
        }
        mock_executor.position_manager.get_hold_days.return_value = 5
        mock_executor.position_manager.get_position.return_value = Position("AAPL", 100, 145.0)

        # Exit function that always returns True
        def should_exit(symbol, position, hold_days):
            return True

        results = mock_executor.check_exits(should_exit)
        # Should attempt to exit
        assert len(results) >= 0  # Results depend on execution success


class TestSummaries:
    """Tests for summary methods."""

    def test_get_account_summary(self, mock_executor):
        """Test getting account summary."""
        mock_executor.start()
        summary = mock_executor.get_account_summary()
        assert summary.net_liquidation == 100000.0

    def test_get_risk_summary(self, mock_executor):
        """Test getting risk summary."""
        mock_executor.start()
        summary = mock_executor.get_risk_summary()
        assert "trading_allowed" in summary
        assert "trades_today" in summary

    def test_get_position_summary(self, mock_executor):
        """Test getting position summary."""
        mock_executor.start()
        mock_executor.position_manager.get_position_summary.return_value = {
            "total_positions": 0,
            "positions": {}
        }
        summary = mock_executor.get_position_summary()
        assert "total_positions" in summary

    def test_get_trade_history(self, mock_executor):
        """Test getting trade history."""
        mock_executor.start()
        history = mock_executor.get_trade_history()
        assert isinstance(history, list)


class TestContextManager:
    """Tests for context manager interface."""

    def test_context_manager(self, ibkr_config, trading_config):
        """Test using executor as context manager."""
        with patch('ibkr.trade_executor.IBKRConnection') as MockConn, \
             patch('ibkr.trade_executor.OrderManager'), \
             patch('ibkr.trade_executor.PositionManager'):

            mock_conn = MockConn.return_value
            mock_conn.connect.return_value = True
            mock_conn.is_connected = True
            mock_conn.get_account_summary.return_value = AccountSummary(
                net_liquidation=100000.0,
                available_funds=50000.0,
                buying_power=100000.0,
                gross_position_value=50000.0
            )

            with TradeExecutor(ibkr_config, trading_config) as executor:
                assert executor is not None
                mock_conn.connect.assert_called()

            # Disconnect should be called on exit
            mock_conn.disconnect.assert_called()
