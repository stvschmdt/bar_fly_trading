"""
Integration tests for IBKR live Gateway connection.

These tests connect to a real IB Gateway and perform actual operations.
Run with: pytest tests/test_integration_live.py -v -s

NOTE: These tests require:
1. IB Gateway running and authenticated on the configured host
2. Market hours for buy/sell tests (will skip if market closed)
3. Sufficient account balance for trades

Configure IBKR_GATEWAY_HOST environment variable or use default.
"""

import os
import pytest
from datetime import datetime, time
import pytz

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibkr.config import IBKRConfig
from ibkr.connection import IBKRConnection
from ibkr.models import OrderAction, OrderType, TradeSignal


# Configuration - can be overridden with environment variables
# Default to 127.0.0.1 (use SSH tunnel: ssh -L 4001:127.0.0.1:4001 $EC2_USER@$EC2_IP -N)
GATEWAY_HOST = os.environ.get("IBKR_GATEWAY_HOST", "127.0.0.1")
GATEWAY_PORT = int(os.environ.get("IBKR_GATEWAY_PORT", "4001"))
CLIENT_ID = int(os.environ.get("IBKR_CLIENT_ID", "1"))

# Test symbol
TEST_SYMBOL = "NKE"


def is_market_hours() -> bool:
    """Check if US market is currently open (9:30 AM - 4:00 PM ET, Mon-Fri)."""
    et = pytz.timezone('America/New_York')
    now = datetime.now(et)

    # Check if weekday (Monday=0, Friday=4)
    if now.weekday() > 4:
        return False

    market_open = time(9, 30)
    market_close = time(16, 0)
    current_time = now.time()

    return market_open <= current_time <= market_close


@pytest.fixture(scope="module")
def gateway_config():
    """Create configuration for the remote Gateway."""
    return IBKRConfig.remote_gateway(
        host=GATEWAY_HOST,
        port=GATEWAY_PORT,
        client_id=CLIENT_ID,
        readonly=False
    )


@pytest.fixture(scope="module")
def connection(gateway_config):
    """Create and manage connection to IB Gateway."""
    conn = IBKRConnection(gateway_config)
    connected = conn.connect()

    if not connected:
        pytest.skip(f"Could not connect to IB Gateway at {GATEWAY_HOST}:{GATEWAY_PORT}")

    yield conn

    conn.disconnect()


class TestAccountBalance:
    """Test account balance retrieval."""

    def test_get_account_summary(self, connection):
        """Test fetching account summary with balance information."""
        summary = connection.get_account_summary()

        assert summary is not None, "Failed to get account summary"

        # Print account details for verification
        print("\n" + "=" * 60)
        print("ACCOUNT SUMMARY")
        print("=" * 60)
        print(f"Net Liquidation Value: ${summary.net_liquidation:,.2f}")
        print(f"Available Funds:       ${summary.available_funds:,.2f}")
        print(f"Buying Power:          ${summary.buying_power:,.2f}")
        print(f"Gross Position Value:  ${summary.gross_position_value:,.2f}")
        print(f"Realized P&L:          ${summary.realized_pnl:,.2f}")
        print(f"Unrealized P&L:        ${summary.unrealized_pnl:,.2f}")
        print("=" * 60)

        # Verify we have valid positive values for key metrics
        assert summary.net_liquidation > 0, "Net liquidation should be positive"
        assert summary.buying_power >= 0, "Buying power should be non-negative"

    def test_account_has_sufficient_funds(self, connection):
        """Test that account has funds available for trading."""
        summary = connection.get_account_summary()

        assert summary is not None
        assert summary.available_funds > 100, "Account should have at least $100 available"


class TestPortfolioPositions:
    """Test portfolio position retrieval."""

    def test_get_portfolio_positions(self, connection):
        """Test fetching current portfolio positions."""
        # Get portfolio from ib_insync directly
        portfolio = connection.ib.portfolio()

        print("\n" + "=" * 60)
        print("PORTFOLIO POSITIONS")
        print("=" * 60)

        if not portfolio:
            print("No positions currently held")
        else:
            print(f"{'Symbol':<10} {'Shares':>10} {'Avg Cost':>12} {'Market Value':>14} {'Unrealized P&L':>16}")
            print("-" * 60)
            for item in portfolio:
                print(f"{item.contract.symbol:<10} "
                      f"{item.position:>10.0f} "
                      f"${item.averageCost:>10.2f} "
                      f"${item.marketValue:>12.2f} "
                      f"${item.unrealizedPNL:>14.2f}")

        print("=" * 60)

        # Test passes regardless of position count - just verify we can query
        assert isinstance(portfolio, list)

    def test_get_positions_dict(self, connection):
        """Test getting positions as a dictionary by symbol."""
        portfolio = connection.ib.portfolio()

        positions_dict = {}
        for item in portfolio:
            symbol = item.contract.symbol
            positions_dict[symbol] = {
                "shares": item.position,
                "avg_cost": item.averageCost,
                "market_value": item.marketValue,
                "unrealized_pnl": item.unrealizedPNL
            }

        print(f"\nPositions as dict: {positions_dict}")

        # Verify structure
        for symbol, data in positions_dict.items():
            assert "shares" in data
            assert "avg_cost" in data
            assert "market_value" in data


class TestBuyNKE:
    """Test buying a share of NKE."""

    @pytest.mark.skipif(not is_market_hours(), reason="Market is closed")
    def test_buy_one_share_nke(self, connection):
        """Test buying 1 share of NKE."""
        from ib_insync import Stock, MarketOrder

        # Create contract
        contract = Stock(TEST_SYMBOL, "SMART", "USD")
        connection.ib.qualifyContracts(contract)

        # Get current price for reference
        ticker = connection.ib.reqMktData(contract, snapshot=True)
        connection.ib.sleep(2)
        current_price = ticker.marketPrice() or ticker.last
        connection.ib.cancelMktData(contract)

        print("\n" + "=" * 60)
        print(f"BUYING 1 SHARE OF {TEST_SYMBOL}")
        print("=" * 60)
        print(f"Current Price: ${current_price:.2f}" if current_price else "Price unavailable")

        # Create and place market order
        order = MarketOrder("BUY", 1)
        trade = connection.ib.placeOrder(contract, order)

        # Wait for fill (up to 30 seconds)
        timeout = 30
        start = datetime.now()
        while not trade.isDone() and (datetime.now() - start).seconds < timeout:
            connection.ib.sleep(1)

        print(f"Order Status: {trade.orderStatus.status}")
        print(f"Filled: {trade.orderStatus.filled}")
        print(f"Avg Fill Price: ${trade.orderStatus.avgFillPrice:.2f}" if trade.orderStatus.avgFillPrice else "N/A")
        print("=" * 60)

        # Check result
        if trade.orderStatus.status == "Filled":
            assert trade.orderStatus.filled == 1, "Should have filled 1 share"
            print(f"SUCCESS: Bought 1 share of {TEST_SYMBOL} at ${trade.orderStatus.avgFillPrice:.2f}")
        else:
            # Order may not fill after hours
            print(f"Order status: {trade.orderStatus.status} (may be expected outside market hours)")
            # Don't fail the test - just report
            assert trade.orderStatus.status in ["Filled", "Submitted", "PreSubmitted", "PendingSubmit"]

    def test_buy_one_share_nke_anytime(self, connection):
        """Test buying 1 share of NKE (submits order regardless of market hours)."""
        from ib_insync import Stock, MarketOrder

        # Create contract
        contract = Stock(TEST_SYMBOL, "SMART", "USD")
        connection.ib.qualifyContracts(contract)

        print("\n" + "=" * 60)
        print(f"SUBMITTING BUY ORDER FOR 1 SHARE OF {TEST_SYMBOL}")
        print(f"Market Hours: {'Yes' if is_market_hours() else 'No'}")
        print("=" * 60)

        # Create and place market order
        order = MarketOrder("BUY", 1)
        trade = connection.ib.placeOrder(contract, order)

        # Wait a bit for status
        connection.ib.sleep(3)

        print(f"Order ID: {trade.order.orderId}")
        print(f"Order Status: {trade.orderStatus.status}")
        print(f"Filled: {trade.orderStatus.filled}")
        if trade.orderStatus.avgFillPrice:
            print(f"Avg Fill Price: ${trade.orderStatus.avgFillPrice:.2f}")
        print("=" * 60)

        # Verify order was accepted
        assert trade.order.orderId > 0, "Order should have received an ID"
        assert trade.orderStatus.status in ["Filled", "Submitted", "PreSubmitted", "PendingSubmit", "Inactive"]

        # If not filled, cancel to clean up
        if trade.orderStatus.status != "Filled":
            print("Order not filled, cancelling...")
            connection.ib.cancelOrder(trade.order)
            connection.ib.sleep(2)


class TestSellNKE:
    """Test selling a share of NKE."""

    @pytest.mark.skipif(not is_market_hours(), reason="Market is closed")
    def test_sell_one_share_nke(self, connection):
        """Test selling 1 share of NKE (requires existing position)."""
        from ib_insync import Stock, MarketOrder

        # Check if we have a position to sell
        portfolio = connection.ib.portfolio()
        nke_position = None
        for item in portfolio:
            if item.contract.symbol == TEST_SYMBOL:
                nke_position = item
                break

        print("\n" + "=" * 60)
        print(f"SELLING 1 SHARE OF {TEST_SYMBOL}")
        print("=" * 60)

        if not nke_position or nke_position.position < 1:
            print(f"No {TEST_SYMBOL} position to sell (have {nke_position.position if nke_position else 0} shares)")
            pytest.skip(f"No {TEST_SYMBOL} position to sell")

        print(f"Current Position: {nke_position.position} shares")
        print(f"Position P&L: ${nke_position.unrealizedPNL:.2f}")

        # Create contract
        contract = Stock(TEST_SYMBOL, "SMART", "USD")
        connection.ib.qualifyContracts(contract)

        # Create and place market sell order
        order = MarketOrder("SELL", 1)
        trade = connection.ib.placeOrder(contract, order)

        # Wait for fill (up to 30 seconds)
        timeout = 30
        start = datetime.now()
        while not trade.isDone() and (datetime.now() - start).seconds < timeout:
            connection.ib.sleep(1)

        print(f"Order Status: {trade.orderStatus.status}")
        print(f"Filled: {trade.orderStatus.filled}")
        print(f"Avg Fill Price: ${trade.orderStatus.avgFillPrice:.2f}" if trade.orderStatus.avgFillPrice else "N/A")
        print("=" * 60)

        if trade.orderStatus.status == "Filled":
            assert trade.orderStatus.filled == 1, "Should have filled 1 share"
            print(f"SUCCESS: Sold 1 share of {TEST_SYMBOL} at ${trade.orderStatus.avgFillPrice:.2f}")
        else:
            print(f"Order status: {trade.orderStatus.status}")
            assert trade.orderStatus.status in ["Filled", "Submitted", "PreSubmitted", "PendingSubmit"]

    def test_sell_one_share_nke_anytime(self, connection):
        """Test selling 1 share of NKE (submits order regardless of market hours)."""
        from ib_insync import Stock, MarketOrder

        # Check if we have a position to sell
        portfolio = connection.ib.portfolio()
        nke_position = None
        for item in portfolio:
            if item.contract.symbol == TEST_SYMBOL:
                nke_position = item
                break

        print("\n" + "=" * 60)
        print(f"SUBMITTING SELL ORDER FOR 1 SHARE OF {TEST_SYMBOL}")
        print(f"Market Hours: {'Yes' if is_market_hours() else 'No'}")
        print("=" * 60)

        if not nke_position or nke_position.position < 1:
            print(f"No {TEST_SYMBOL} position to sell")
            pytest.skip(f"No {TEST_SYMBOL} position to sell")

        print(f"Current Position: {nke_position.position} shares")

        # Create contract
        contract = Stock(TEST_SYMBOL, "SMART", "USD")
        connection.ib.qualifyContracts(contract)

        # Create and place market sell order
        order = MarketOrder("SELL", 1)
        trade = connection.ib.placeOrder(contract, order)

        # Wait a bit for status
        connection.ib.sleep(3)

        print(f"Order ID: {trade.order.orderId}")
        print(f"Order Status: {trade.orderStatus.status}")
        print(f"Filled: {trade.orderStatus.filled}")
        if trade.orderStatus.avgFillPrice:
            print(f"Avg Fill Price: ${trade.orderStatus.avgFillPrice:.2f}")
        print("=" * 60)

        # Verify order was accepted
        assert trade.order.orderId > 0, "Order should have received an ID"

        # If not filled, cancel to clean up
        if trade.orderStatus.status != "Filled":
            print("Order not filled, cancelling...")
            connection.ib.cancelOrder(trade.order)
            connection.ib.sleep(2)


class TestConnectionStatus:
    """Test connection health and status."""

    def test_connection_is_active(self, connection):
        """Verify connection is active."""
        assert connection.is_connected, "Connection should be active"
        print(f"\nConnection active: {connection.is_connected}")
        print(f"Connection duration: {connection.connection_duration:.1f} seconds")

    def test_can_get_market_data(self, connection):
        """Test that we can receive market data."""
        from ib_insync import Stock

        contract = Stock("AAPL", "SMART", "USD")
        connection.ib.qualifyContracts(contract)

        ticker = connection.ib.reqMktData(contract, snapshot=True)
        connection.ib.sleep(2)

        price = ticker.marketPrice() or ticker.last or ticker.close
        connection.ib.cancelMktData(contract)

        print(f"\nAAPL price: ${price:.2f}" if price else "Price unavailable (market may be closed)")

        # During market hours, we should get a price
        # Outside market hours, we may not - don't fail the test
        if is_market_hours():
            assert price is not None and price > 0, "Should get AAPL price during market hours"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "-s"])