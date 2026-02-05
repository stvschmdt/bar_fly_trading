"""Unit tests for IBKR order manager."""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from ibkr.config import IBKRConfig, TradingConfig
from ibkr.connection import IBKRConnection
from ibkr.order_manager import OrderManager
from ibkr.models import TradeSignal, OrderAction, OrderType, OrderStatus


@pytest.fixture
def mock_connection():
    """Create mock IBKR connection."""
    connection = Mock(spec=IBKRConnection)
    connection.is_connected = True
    connection.ib = MagicMock()
    return connection


@pytest.fixture
def trading_config():
    """Create test trading configuration."""
    return TradingConfig(
        symbols={"AAPL", "GOOGL"},
        use_market_orders=True,
        limit_offset_pct=0.001,
        order_timeout=60
    )


@pytest.fixture
def order_manager(mock_connection, trading_config):
    """Create order manager with mock connection."""
    with patch.object(OrderManager, '__init__', lambda self, conn, cfg: None):
        manager = OrderManager.__new__(OrderManager)
        manager.connection = mock_connection
        manager.config = trading_config
        manager._pending_orders = {}
        manager._completed_orders = {}
        manager._order_callbacks = {}
    return manager


@pytest.fixture
def buy_signal():
    """Create test buy signal."""
    return TradeSignal(
        symbol="AAPL",
        action=OrderAction.BUY,
        shares=100,
        signal_price=150.0,
        signal_time=datetime.now()
    )


@pytest.fixture
def sell_signal():
    """Create test sell signal."""
    return TradeSignal(
        symbol="AAPL",
        action=OrderAction.SELL,
        shares=100,
        signal_price=155.0,
        signal_time=datetime.now()
    )


class TestCreateMarketOrder:
    """Tests for market order creation."""

    def test_create_buy_market_order(self, order_manager, buy_signal):
        """Test creating a buy market order."""
        order = order_manager.create_market_order(buy_signal)
        assert order.action == "BUY"
        assert order.totalQuantity == 100
        assert order.orderType == "MKT"

    def test_create_sell_market_order(self, order_manager, sell_signal):
        """Test creating a sell market order."""
        order = order_manager.create_market_order(sell_signal)
        assert order.action == "SELL"
        assert order.totalQuantity == 100

    def test_create_market_order_with_override_shares(self, order_manager, buy_signal):
        """Test creating market order with overridden shares."""
        order = order_manager.create_market_order(buy_signal, shares=50)
        assert order.totalQuantity == 50


class TestCreateLimitOrder:
    """Tests for limit order creation."""

    def test_create_buy_limit_order(self, order_manager, buy_signal):
        """Test creating a buy limit order."""
        order = order_manager.create_limit_order(buy_signal)
        assert order.action == "BUY"
        assert order.orderType == "LMT"
        # Buy limit should be slightly above signal price
        expected_price = round(150.0 * 1.001, 2)
        assert order.lmtPrice == expected_price

    def test_create_sell_limit_order(self, order_manager, sell_signal):
        """Test creating a sell limit order."""
        order = order_manager.create_limit_order(sell_signal)
        assert order.action == "SELL"
        # Sell limit should be slightly below signal price
        expected_price = round(155.0 * 0.999, 2)
        assert order.lmtPrice == expected_price

    def test_create_limit_order_with_custom_price(self, order_manager, buy_signal):
        """Test creating limit order with custom price."""
        order = order_manager.create_limit_order(buy_signal, limit_price=149.50)
        assert order.lmtPrice == 149.50


class TestCreateStopOrder:
    """Tests for stop order creation."""

    def test_create_stop_order(self, order_manager, sell_signal):
        """Test creating a stop order."""
        order = order_manager.create_stop_order(sell_signal, stop_price=145.0)
        assert order.action == "SELL"
        assert order.orderType == "STP"
        assert order.auxPrice == 145.0


class TestOrderTracking:
    """Tests for order tracking."""

    def test_get_open_orders_empty(self, order_manager):
        """Test getting open orders when none exist."""
        orders = order_manager.get_open_orders()
        assert orders == []

    def test_get_order_status_not_found(self, order_manager):
        """Test getting status for non-existent order."""
        result = order_manager.get_order_status(99999)
        assert result is None

    def test_execution_summary_empty(self, order_manager):
        """Test execution summary with no orders."""
        summary = order_manager.get_execution_summary()
        assert summary["total_orders"] == 0
        assert summary["pending"] == 0


class TestOrderCallbacks:
    """Tests for order completion callbacks."""

    def test_register_callback(self, order_manager):
        """Test registering order completion callback."""
        callback = Mock()
        order_manager.on_order_complete(12345, callback)
        assert 12345 in order_manager._order_callbacks
        assert callback in order_manager._order_callbacks[12345]

    def test_multiple_callbacks(self, order_manager):
        """Test registering multiple callbacks for same order."""
        callback1 = Mock()
        callback2 = Mock()
        order_manager.on_order_complete(12345, callback1)
        order_manager.on_order_complete(12345, callback2)
        assert len(order_manager._order_callbacks[12345]) == 2


class TestSubmitOrderNotConnected:
    """Tests for order submission when not connected."""

    def test_submit_order_not_connected(self, order_manager, buy_signal):
        """Test order submission fails when not connected."""
        order_manager.connection.is_connected = False
        result = order_manager.submit_order(buy_signal)
        assert result is None


class TestCancelOrder:
    """Tests for order cancellation."""

    def test_cancel_order_not_connected(self, order_manager):
        """Test cancel fails when not connected."""
        order_manager.connection.is_connected = False
        result = order_manager.cancel_order(12345)
        assert result is False

    def test_cancel_all_orders_not_connected(self, order_manager):
        """Test cancel all returns 0 when not connected."""
        order_manager.connection.is_connected = False
        count = order_manager.cancel_all_orders()
        assert count == 0
