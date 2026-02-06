"""
Order management for IBKR trading.

Handles order creation, submission, monitoring, and cancellation.
"""

import logging
import time
from datetime import datetime
from typing import Optional, Callable

from ib_insync import (
    IB, Trade, Order, MarketOrder, LimitOrder, StopOrder,
    Stock, OrderStatus as IBOrderStatus
)

from .connection import IBKRConnection
from .config import TradingConfig
from .models import (
    TradeSignal, OrderResult, OrderAction, OrderType, OrderStatus
)

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Manages order lifecycle with IBKR.

    Handles:
    - Order creation (market, limit, stop)
    - Order submission
    - Order status monitoring
    - Order cancellation
    - Fill confirmation
    """

    def __init__(self, connection: IBKRConnection, config: TradingConfig):
        """
        Initialize order manager.

        Args:
            connection: IBKR connection instance
            config: Trading configuration
        """
        self.connection = connection
        self.config = config
        self._pending_orders: dict[int, OrderResult] = {}
        self._completed_orders: dict[int, OrderResult] = {}
        self._order_callbacks: dict[int, list[Callable]] = {}

        # Register for order status updates
        self.ib.orderStatusEvent += self._on_order_status

    @property
    def ib(self) -> IB:
        """Get IB instance from connection."""
        return self.connection.ib

    @property
    def target_account(self) -> Optional[str]:
        """Get target trading account (prefers DU sub-accounts for paper trading)."""
        if self.connection.config.account:
            return self.connection.config.account
        try:
            all_accounts = self.ib.managedAccounts()
            # Prefer DU sub-accounts (paper) over DFO (FA master)
            account = next((a for a in all_accounts if a.startswith('DU')), None)
            if not account:
                account = next((a for a in all_accounts if a.startswith('U')), None)
            if not account and all_accounts:
                account = all_accounts[0]
            return account
        except Exception:
            return None

    def create_market_order(
        self,
        signal: TradeSignal,
        shares: Optional[int] = None
    ) -> Order:
        """
        Create a market order from a trade signal.

        Args:
            signal: Trade signal
            shares: Override number of shares (uses signal.shares if None)

        Returns:
            IB Order object
        """
        action = "BUY" if signal.action == OrderAction.BUY else "SELL"
        quantity = shares or signal.shares

        order = MarketOrder(action, quantity)
        order.tif = "DAY"  # Time in force: day order
        # FA multi-account: must specify target account
        acct = self.target_account
        if acct:
            order.account = acct
            logger.info(f"Market order: {action} {quantity} → account {acct}")

        return order

    def create_limit_order(
        self,
        signal: TradeSignal,
        limit_price: Optional[float] = None,
        shares: Optional[int] = None
    ) -> Order:
        """
        Create a limit order from a trade signal.

        Args:
            signal: Trade signal
            limit_price: Limit price (calculated from signal if None)
            shares: Override number of shares

        Returns:
            IB Order object
        """
        action = "BUY" if signal.action == OrderAction.BUY else "SELL"
        quantity = shares or signal.shares

        # Calculate limit price with offset
        if limit_price is None:
            offset = self.config.limit_offset_pct
            if signal.action == OrderAction.BUY:
                # Buy slightly above market
                limit_price = signal.signal_price * (1 + offset)
            else:
                # Sell slightly below market
                limit_price = signal.signal_price * (1 - offset)

        # Round to 2 decimal places
        limit_price = round(limit_price, 2)

        order = LimitOrder(action, quantity, limit_price)
        order.tif = "DAY"
        acct = self.target_account
        if acct:
            order.account = acct

        return order

    def create_stop_order(
        self,
        signal: TradeSignal,
        stop_price: float,
        shares: Optional[int] = None
    ) -> Order:
        """
        Create a stop order.

        Args:
            signal: Trade signal
            stop_price: Stop trigger price
            shares: Override number of shares

        Returns:
            IB Order object
        """
        action = "BUY" if signal.action == OrderAction.BUY else "SELL"
        quantity = shares or signal.shares

        order = StopOrder(action, quantity, round(stop_price, 2))
        order.tif = "DAY"
        acct = self.target_account
        if acct:
            order.account = acct

        return order

    def submit_order(
        self,
        signal: TradeSignal,
        shares: Optional[int] = None,
        order_type: OrderType = OrderType.MARKET
    ) -> Optional[OrderResult]:
        """
        Submit an order to IBKR.

        Args:
            signal: Trade signal
            shares: Number of shares (uses signal.shares if None)
            order_type: Type of order to submit

        Returns:
            OrderResult with order details, or None on failure
        """
        if not self.connection.is_connected:
            logger.error("Cannot submit order: not connected to IBKR")
            return None

        shares = shares or signal.shares

        # Create contract
        contract = Stock(signal.symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)

        # Create order based on type
        if order_type == OrderType.MARKET or self.config.use_market_orders:
            order = self.create_market_order(signal, shares)
        elif order_type == OrderType.LIMIT:
            order = self.create_limit_order(signal, shares=shares)
        else:
            order = self.create_market_order(signal, shares)

        try:
            # Submit order
            trade = self.ib.placeOrder(contract, order)

            logger.info(
                f"Order submitted: {signal.action.value} {shares} {signal.symbol} "
                f"@ {order_type.value}"
            )

            # Create result object
            result = OrderResult(
                order_id=trade.order.orderId,
                symbol=signal.symbol,
                action=signal.action,
                shares=shares,
                order_type=order_type,
                status=OrderStatus.PENDING_SUBMIT,
                submitted_time=datetime.now()
            )

            # Track pending order
            self._pending_orders[result.order_id] = result

            return result

        except Exception as e:
            logger.error(f"Failed to submit order for {signal.symbol}: {e}")
            return OrderResult(
                order_id=-1,
                symbol=signal.symbol,
                action=signal.action,
                shares=shares,
                order_type=order_type,
                status=OrderStatus.ERROR,
                error_message=str(e)
            )

    def wait_for_fill(
        self,
        order_result: OrderResult,
        timeout: Optional[int] = None
    ) -> OrderResult:
        """
        Wait for an order to fill.

        Args:
            order_result: Order to wait for
            timeout: Timeout in seconds (uses config default if None)

        Returns:
            Updated OrderResult
        """
        timeout = timeout or self.config.order_timeout
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Process IB events
            self.ib.sleep(0.5)

            # Check if order is complete
            if order_result.order_id in self._completed_orders:
                return self._completed_orders[order_result.order_id]

            # Update from pending orders
            if order_result.order_id in self._pending_orders:
                order_result = self._pending_orders[order_result.order_id]
                if order_result.is_complete:
                    return order_result

        # Timeout reached
        logger.warning(f"Order {order_result.order_id} timed out after {timeout}s")
        order_result.status = OrderStatus.INACTIVE
        order_result.error_message = f"Timeout after {timeout}s"
        return order_result

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: ID of order to cancel

        Returns:
            True if cancellation request sent successfully
        """
        if not self.connection.is_connected:
            logger.error("Cannot cancel order: not connected")
            return False

        try:
            # Find the trade object
            for trade in self.ib.openTrades():
                if trade.order.orderId == order_id:
                    self.ib.cancelOrder(trade.order)
                    logger.info(f"Cancel request sent for order {order_id}")
                    return True

            logger.warning(f"Order {order_id} not found in open trades")
            return False

        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    def cancel_all_orders(self) -> int:
        """
        Cancel all open orders.

        Returns:
            Number of orders cancelled
        """
        if not self.connection.is_connected:
            return 0

        cancelled = 0
        for trade in self.ib.openTrades():
            try:
                self.ib.cancelOrder(trade.order)
                cancelled += 1
            except Exception as e:
                logger.error(f"Failed to cancel order {trade.order.orderId}: {e}")

        logger.info(f"Cancelled {cancelled} open orders")
        return cancelled

    def get_open_orders(self) -> list[OrderResult]:
        """Get all open/pending orders."""
        return [
            result for result in self._pending_orders.values()
            if result.is_pending
        ]

    def get_order_status(self, order_id: int) -> Optional[OrderResult]:
        """
        Get current status of an order.

        Args:
            order_id: Order ID to check

        Returns:
            OrderResult or None if not found
        """
        if order_id in self._completed_orders:
            return self._completed_orders[order_id]
        if order_id in self._pending_orders:
            return self._pending_orders[order_id]
        return None

    def _on_order_status(self, trade: Trade) -> None:
        """Handle order status update from IBKR."""
        order_id = trade.order.orderId

        # Map IBKR status to our status
        status_map = {
            "PendingSubmit": OrderStatus.PENDING_SUBMIT,
            "PendingCancel": OrderStatus.PENDING_CANCEL,
            "PreSubmitted": OrderStatus.PRE_SUBMITTED,
            "Submitted": OrderStatus.SUBMITTED,
            "Cancelled": OrderStatus.CANCELLED,
            "Filled": OrderStatus.FILLED,
            "Inactive": OrderStatus.INACTIVE,
        }

        ib_status = trade.orderStatus.status
        status = status_map.get(ib_status, OrderStatus.ERROR)

        # Update or create order result
        if order_id in self._pending_orders:
            result = self._pending_orders[order_id]
        else:
            # Unknown order - create minimal result
            result = OrderResult(
                order_id=order_id,
                symbol=trade.contract.symbol,
                action=OrderAction.BUY if trade.order.action == "BUY" else OrderAction.SELL,
                shares=int(trade.order.totalQuantity),
                order_type=OrderType.MARKET,
                status=status
            )
            self._pending_orders[order_id] = result

        # Update result
        result.status = status
        result.filled_shares = int(trade.orderStatus.filled)
        result.avg_fill_price = trade.orderStatus.avgFillPrice

        logger.info(
            f"Order {order_id} status: {status.value}, "
            f"filled: {result.filled_shares}/{result.shares} "
            f"@ ${result.avg_fill_price:.2f}"
        )

        # Move to completed if done
        if result.is_complete:
            result.filled_time = datetime.now()
            self._completed_orders[order_id] = result
            if order_id in self._pending_orders:
                del self._pending_orders[order_id]

            # Call registered callbacks
            if order_id in self._order_callbacks:
                for callback in self._order_callbacks[order_id]:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"Order callback error: {e}")
                del self._order_callbacks[order_id]

    def on_order_complete(self, order_id: int, callback: Callable[[OrderResult], None]) -> None:
        """
        Register callback for when an order completes.

        Args:
            order_id: Order ID to watch
            callback: Function to call with OrderResult when complete
        """
        if order_id not in self._order_callbacks:
            self._order_callbacks[order_id] = []
        self._order_callbacks[order_id].append(callback)

    def get_execution_summary(self) -> dict:
        """Get summary of order executions."""
        filled = [r for r in self._completed_orders.values() if r.is_filled]
        cancelled = [r for r in self._completed_orders.values() if r.status == OrderStatus.CANCELLED]
        errors = [r for r in self._completed_orders.values() if r.status == OrderStatus.ERROR]

        return {
            "total_orders": len(self._completed_orders),
            "filled": len(filled),
            "cancelled": len(cancelled),
            "errors": len(errors),
            "pending": len(self._pending_orders),
            "fill_rate": len(filled) / len(self._completed_orders) if self._completed_orders else 0
        }
