"""
Interactive Brokers Paper Trading Module.

Provides simple interface for placing trades via IB API.
Designed for paper trading with proper verification.

Requirements:
    pip install ib_insync

Usage:
    from ib_trader import IBTrader

    trader = IBTrader(paper=True)
    trader.connect()

    # Buy 1 share of AAPL
    result = trader.buy_stock("AAPL", quantity=1)
    print(result)

    trader.disconnect()
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status codes."""
    PENDING = "PendingSubmit"
    SUBMITTED = "Submitted"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    ERROR = "Error"
    UNKNOWN = "Unknown"


@dataclass
class TradeResult:
    """Result of a trade execution."""
    success: bool
    order_id: Optional[int]
    symbol: str
    quantity: int
    side: str  # "BUY" or "SELL"
    status: OrderStatus
    fill_price: Optional[float]
    message: str

    def __str__(self):
        if self.success:
            return (f"[{self.status.value}] {self.side} {self.quantity} {self.symbol} "
                   f"@ ${self.fill_price:.2f} (Order ID: {self.order_id})")
        return f"[{self.status.value}] {self.side} {self.quantity} {self.symbol} - {self.message}"


class IBTrader:
    """
    Interactive Brokers trading client.

    Handles connection, order placement, and verification.

    Args:
        host: IB Gateway/TWS host (default: 127.0.0.1)
        port: IB Gateway/TWS port (paper: 7497, live: 7496)
        client_id: Unique client ID for this connection
        paper: If True, use paper trading port (7497)
        timeout: Connection timeout in seconds
    """

    # Default ports
    PAPER_PORT = 7497
    LIVE_PORT = 7496

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        client_id: int = 1,
        paper: bool = True,
        timeout: int = 10
    ):
        self.host = host
        self.port = port or (self.PAPER_PORT if paper else self.LIVE_PORT)
        self.client_id = client_id
        self.paper = paper
        self.timeout = timeout
        self.ib = None
        self._connected = False

        logger.info(f"IBTrader initialized (paper={paper}, port={self.port})")

    @property
    def connected(self) -> bool:
        """Check if connected to IB."""
        if self.ib is None:
            return False
        return self.ib.isConnected()

    def connect(self) -> bool:
        """
        Connect to IB Gateway/TWS.

        Returns:
            True if connection successful, False otherwise.
        """
        try:
            from ib_insync import IB
        except ImportError:
            logger.error("ib_insync not installed. Run: pip install ib_insync")
            return False

        try:
            self.ib = IB()
            self.ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=self.timeout
            )
            self._connected = True
            logger.info(f"Connected to IB at {self.host}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """Disconnect from IB."""
        if self.ib and self.connected:
            self.ib.disconnect()
            logger.info("Disconnected from IB")
        self._connected = False

    def _create_stock_contract(self, symbol: str, exchange: str = "SMART", currency: str = "USD"):
        """Create a stock contract."""
        from ib_insync import Stock
        return Stock(symbol, exchange, currency)

    def _verify_order_fill(self, trade, timeout: int = 30) -> TradeResult:
        """
        Wait for order to fill and verify result.

        Args:
            trade: IB trade object
            timeout: Max seconds to wait for fill

        Returns:
            TradeResult with fill details
        """
        import time

        start_time = time.time()
        order = trade.order
        symbol = trade.contract.symbol

        while time.time() - start_time < timeout:
            self.ib.sleep(0.5)  # Allow IB to process

            status = trade.orderStatus.status

            if status == "Filled":
                fill_price = trade.orderStatus.avgFillPrice
                logger.info(f"Order filled: {symbol} @ ${fill_price:.2f}")
                return TradeResult(
                    success=True,
                    order_id=order.orderId,
                    symbol=symbol,
                    quantity=int(order.totalQuantity),
                    side="BUY" if order.action == "BUY" else "SELL",
                    status=OrderStatus.FILLED,
                    fill_price=fill_price,
                    message="Order filled successfully"
                )

            elif status in ("Cancelled", "ApiCancelled"):
                logger.warning(f"Order cancelled: {symbol}")
                return TradeResult(
                    success=False,
                    order_id=order.orderId,
                    symbol=symbol,
                    quantity=int(order.totalQuantity),
                    side="BUY" if order.action == "BUY" else "SELL",
                    status=OrderStatus.CANCELLED,
                    fill_price=None,
                    message="Order was cancelled"
                )

            elif status == "Inactive":
                logger.warning(f"Order inactive: {symbol}")
                return TradeResult(
                    success=False,
                    order_id=order.orderId,
                    symbol=symbol,
                    quantity=int(order.totalQuantity),
                    side="BUY" if order.action == "BUY" else "SELL",
                    status=OrderStatus.ERROR,
                    fill_price=None,
                    message="Order is inactive - check account/permissions"
                )

        # Timeout
        logger.warning(f"Order timeout: {symbol} (status: {trade.orderStatus.status})")
        return TradeResult(
            success=False,
            order_id=order.orderId,
            symbol=symbol,
            quantity=int(order.totalQuantity),
            side="BUY" if order.action == "BUY" else "SELL",
            status=OrderStatus.PENDING,
            fill_price=None,
            message=f"Order timeout - current status: {trade.orderStatus.status}"
        )

    def buy_stock(
        self,
        symbol: str,
        quantity: int = 1,
        order_type: str = "MKT",
        limit_price: Optional[float] = None,
        wait_for_fill: bool = True
    ) -> TradeResult:
        """
        Buy shares of a stock.

        Args:
            symbol: Stock ticker (e.g., "AAPL")
            quantity: Number of shares to buy
            order_type: "MKT" for market, "LMT" for limit
            limit_price: Required if order_type is "LMT"
            wait_for_fill: If True, wait for fill confirmation

        Returns:
            TradeResult with execution details
        """
        if not self.connected:
            return TradeResult(
                success=False,
                order_id=None,
                symbol=symbol,
                quantity=quantity,
                side="BUY",
                status=OrderStatus.ERROR,
                fill_price=None,
                message="Not connected to IB"
            )

        try:
            from ib_insync import MarketOrder, LimitOrder

            # Create contract
            contract = self._create_stock_contract(symbol)

            # Qualify contract (verify it exists)
            self.ib.qualifyContracts(contract)

            # Create order
            if order_type == "MKT":
                order = MarketOrder("BUY", quantity)
            elif order_type == "LMT":
                if limit_price is None:
                    return TradeResult(
                        success=False,
                        order_id=None,
                        symbol=symbol,
                        quantity=quantity,
                        side="BUY",
                        status=OrderStatus.ERROR,
                        fill_price=None,
                        message="Limit price required for LMT order"
                    )
                order = LimitOrder("BUY", quantity, limit_price)
            else:
                return TradeResult(
                    success=False,
                    order_id=None,
                    symbol=symbol,
                    quantity=quantity,
                    side="BUY",
                    status=OrderStatus.ERROR,
                    fill_price=None,
                    message=f"Unsupported order type: {order_type}"
                )

            # Place order
            logger.info(f"Placing order: BUY {quantity} {symbol} ({order_type})")
            trade = self.ib.placeOrder(contract, order)

            if wait_for_fill:
                return self._verify_order_fill(trade)

            # Return immediately without waiting
            return TradeResult(
                success=True,
                order_id=order.orderId,
                symbol=symbol,
                quantity=quantity,
                side="BUY",
                status=OrderStatus.SUBMITTED,
                fill_price=None,
                message="Order submitted (not waiting for fill)"
            )

        except Exception as e:
            logger.error(f"Error placing buy order: {e}")
            return TradeResult(
                success=False,
                order_id=None,
                symbol=symbol,
                quantity=quantity,
                side="BUY",
                status=OrderStatus.ERROR,
                fill_price=None,
                message=str(e)
            )

    def sell_stock(
        self,
        symbol: str,
        quantity: int = 1,
        order_type: str = "MKT",
        limit_price: Optional[float] = None,
        wait_for_fill: bool = True
    ) -> TradeResult:
        """
        Sell shares of a stock.

        Args:
            symbol: Stock ticker (e.g., "AAPL")
            quantity: Number of shares to sell
            order_type: "MKT" for market, "LMT" for limit
            limit_price: Required if order_type is "LMT"
            wait_for_fill: If True, wait for fill confirmation

        Returns:
            TradeResult with execution details
        """
        if not self.connected:
            return TradeResult(
                success=False,
                order_id=None,
                symbol=symbol,
                quantity=quantity,
                side="SELL",
                status=OrderStatus.ERROR,
                fill_price=None,
                message="Not connected to IB"
            )

        try:
            from ib_insync import MarketOrder, LimitOrder

            contract = self._create_stock_contract(symbol)
            self.ib.qualifyContracts(contract)

            if order_type == "MKT":
                order = MarketOrder("SELL", quantity)
            elif order_type == "LMT":
                if limit_price is None:
                    return TradeResult(
                        success=False,
                        order_id=None,
                        symbol=symbol,
                        quantity=quantity,
                        side="SELL",
                        status=OrderStatus.ERROR,
                        fill_price=None,
                        message="Limit price required for LMT order"
                    )
                order = LimitOrder("SELL", quantity, limit_price)
            else:
                return TradeResult(
                    success=False,
                    order_id=None,
                    symbol=symbol,
                    quantity=quantity,
                    side="SELL",
                    status=OrderStatus.ERROR,
                    fill_price=None,
                    message=f"Unsupported order type: {order_type}"
                )

            logger.info(f"Placing order: SELL {quantity} {symbol} ({order_type})")
            trade = self.ib.placeOrder(contract, order)

            if wait_for_fill:
                return self._verify_order_fill(trade)

            return TradeResult(
                success=True,
                order_id=order.orderId,
                symbol=symbol,
                quantity=quantity,
                side="SELL",
                status=OrderStatus.SUBMITTED,
                fill_price=None,
                message="Order submitted (not waiting for fill)"
            )

        except Exception as e:
            logger.error(f"Error placing sell order: {e}")
            return TradeResult(
                success=False,
                order_id=None,
                symbol=symbol,
                quantity=quantity,
                side="SELL",
                status=OrderStatus.ERROR,
                fill_price=None,
                message=str(e)
            )

    def get_positions(self) -> list:
        """Get current positions."""
        if not self.connected:
            logger.error("Not connected to IB")
            return []

        positions = self.ib.positions()
        return [
            {
                "symbol": pos.contract.symbol,
                "quantity": pos.position,
                "avg_cost": pos.avgCost,
                "market_value": pos.position * pos.avgCost
            }
            for pos in positions
        ]

    def get_account_summary(self) -> dict:
        """Get account summary."""
        if not self.connected:
            logger.error("Not connected to IB")
            return {}

        summary = self.ib.accountSummary()
        result = {}
        for item in summary:
            result[item.tag] = item.value
        return result


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="IB Paper Trading Test")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol")
    parser.add_argument("--quantity", type=int, default=1, help="Number of shares")
    parser.add_argument("--action", type=str, default="buy", choices=["buy", "sell"],
                       help="Trade action")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="IB host")
    parser.add_argument("--port", type=int, default=7497, help="IB port (7497=paper, 7496=live)")

    args = parser.parse_args()

    # Create trader
    trader = IBTrader(host=args.host, port=args.port, paper=(args.port == 7497))

    # Connect
    if not trader.connect():
        print("Failed to connect to IB. Make sure TWS/Gateway is running.")
        exit(1)

    try:
        # Execute trade
        if args.action == "buy":
            result = trader.buy_stock(args.symbol, args.quantity)
        else:
            result = trader.sell_stock(args.symbol, args.quantity)

        print(f"\nTrade Result: {result}")

        # Show positions
        print("\nCurrent Positions:")
        for pos in trader.get_positions():
            print(f"  {pos['symbol']}: {pos['quantity']} shares @ ${pos['avg_cost']:.2f}")

    finally:
        trader.disconnect()
