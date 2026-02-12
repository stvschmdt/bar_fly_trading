"""
Trade execution orchestrator for IBKR trading.

Coordinates signal generation, validation, and order execution.
"""

import logging
import time
from datetime import datetime
from typing import Optional

from .connection import IBKRConnection
from .config import IBKRConfig, TradingConfig
from .models import (
    TradeSignal, TradeResult, ValidationResult, OrderResult,
    OrderAction, OrderType, Position, AccountSummary,
    ValidationStatus, OrderStatus
)
from .risk_manager import RiskManager
from .order_manager import OrderManager
from .position_manager import PositionManager
from .notifier import TradeNotifier, NotificationConfig

logger = logging.getLogger(__name__)


class TradeExecutor:
    """
    Main orchestrator for trade execution.

    Flow:
    1. Receive trade signal
    2. Validate against risk rules
    3. Submit order to IBKR
    4. Wait for fill confirmation
    5. Update position tracking
    6. Return trade result
    """

    def __init__(
        self,
        ibkr_config: IBKRConfig,
        trading_config: TradingConfig,
        notification_config: Optional[NotificationConfig] = None,
        enable_notifications: bool = True
    ):
        """
        Initialize trade executor.

        Args:
            ibkr_config: IBKR connection configuration
            trading_config: Trading parameters and limits
            notification_config: Notification settings (loads from env if None)
            enable_notifications: Whether to send trade notifications
        """
        self.ibkr_config = ibkr_config
        self.trading_config = trading_config

        # Initialize components
        self.connection = IBKRConnection(ibkr_config)
        self.risk_manager = RiskManager(trading_config)
        self.order_manager: Optional[OrderManager] = None
        self.position_manager: Optional[PositionManager] = None

        # Initialize notifier
        self.enable_notifications = enable_notifications
        self.notifier = TradeNotifier(notification_config) if enable_notifications else None

        # Track execution
        self._trade_history: list[TradeResult] = []
        self._session_start: Optional[datetime] = None

    def start(self) -> bool:
        """
        Start the trade executor.

        Connects to IBKR and initializes managers.

        Returns:
            True if started successfully
        """
        logger.info("Starting Trade Executor...")

        # Connect to IBKR
        if not self.connection.connect():
            logger.error("Failed to connect to IBKR")
            return False

        # Initialize managers that need connection
        self.order_manager = OrderManager(self.connection, self.trading_config)
        self.position_manager = PositionManager(self.connection)

        # Reset daily stats
        self.risk_manager.reset_daily_stats()

        # Refresh positions
        self.position_manager.refresh_positions()
        self.risk_manager.update_positions(self.position_manager.get_open_positions())

        self._session_start = datetime.now()

        # Log startup info
        account = self.connection.get_account_summary()
        if account:
            logger.info(f"Account: ${account.net_liquidation:,.2f} net liquidation")
            logger.info(f"Available: ${account.available_funds:,.2f}")

        positions = self.position_manager.get_open_positions()
        logger.info(f"Open positions: {len(positions)}")

        allowed, reason = self.risk_manager.is_trading_allowed()
        logger.info(f"Trading allowed: {allowed} ({reason})")

        return True

    def stop(self) -> None:
        """Stop the trade executor and disconnect."""
        logger.info("Stopping Trade Executor...")

        # Cancel any open orders
        if self.order_manager:
            self.order_manager.cancel_all_orders()

        # Disconnect
        self.connection.disconnect()

        # Log session summary
        self._log_session_summary()

    def execute_signal(
        self,
        signal: TradeSignal,
        wait_for_fill: bool = True
    ) -> TradeResult:
        """
        Execute a trade signal.

        Args:
            signal: Trade signal to execute
            wait_for_fill: Whether to wait for order fill

        Returns:
            TradeResult with execution details
        """
        start_time = time.time()

        logger.info(f"Executing signal: {signal.action.value} {signal.shares} {signal.symbol}")

        # Check connection
        if not self.connection.is_connected:
            return TradeResult(
                signal=signal,
                validation=ValidationResult(
                    status=ValidationStatus.REJECTED,
                    signal=signal,
                    messages=["Not connected to IBKR"]
                ),
                success=False,
                error_message="Not connected to IBKR"
            )

        # Get current account and positions
        account = self.connection.get_account_summary()
        positions = self.position_manager.get_open_positions()

        # Check bid-ask spread before validation
        bid_ask = self.connection.get_bid_ask(signal.symbol)
        if bid_ask is not None:
            bid, ask, last = bid_ask
            mid = (bid + ask) / 2
            if mid > 0:
                spread_pct = (ask - bid) / mid
                if spread_pct > self.trading_config.max_spread_pct:
                    logger.warning(
                        f"Spread too wide for {signal.symbol}: "
                        f"bid=${bid:.2f} ask=${ask:.2f} spread={spread_pct:.2%} "
                        f"(max {self.trading_config.max_spread_pct:.2%})"
                    )
                    return TradeResult(
                        signal=signal,
                        validation=ValidationResult(
                            status=ValidationStatus.REJECTED,
                            signal=signal,
                            messages=[
                                f"Bid-ask spread {spread_pct:.2%} exceeds "
                                f"{self.trading_config.max_spread_pct:.2%} limit "
                                f"(bid=${bid:.2f}, ask=${ask:.2f})"
                            ]
                        ),
                        success=False,
                        error_message=f"Spread too wide: {spread_pct:.2%}"
                    )
                # Use ask for BUY cost calculations, bid for SELL
                if signal.action == OrderAction.BUY:
                    logger.info(f"Using ask price for {signal.symbol}: ${ask:.2f} (last=${last:.2f})")
                else:
                    logger.info(f"Using bid price for {signal.symbol}: ${bid:.2f} (last=${last:.2f})")

        # Validate signal
        validation = self.risk_manager.validate_trade(signal, account, positions)

        if validation.is_rejected:
            logger.warning(f"Trade rejected: {validation.messages}")
            return TradeResult(
                signal=signal,
                validation=validation,
                success=False,
                error_message="; ".join(validation.messages)
            )

        # Use adjusted shares if needed
        shares_to_trade = validation.adjusted_shares or signal.shares

        # Determine order type: stocks default to market, options to limit
        use_market = self.trading_config.use_market_orders or self.trading_config.stock_market_orders
        order_type = OrderType.MARKET if use_market else OrderType.LIMIT

        # Submit order
        order_result = self.order_manager.submit_order(
            signal,
            shares=shares_to_trade,
            order_type=order_type
        )

        if order_result is None or order_result.status == OrderStatus.ERROR:
            return TradeResult(
                signal=signal,
                validation=validation,
                order=order_result,
                success=False,
                error_message=order_result.error_message if order_result else "Order submission failed"
            )

        # Wait for fill if requested
        if wait_for_fill:
            order_result = self.order_manager.wait_for_fill(order_result)

        # Calculate execution metrics
        execution_time_ms = int((time.time() - start_time) * 1000)
        slippage = 0.0
        if order_result.is_filled and order_result.avg_fill_price > 0:
            slippage = order_result.avg_fill_price - signal.signal_price
            if signal.action == OrderAction.SELL:
                slippage = -slippage  # Negative slippage is good for sells

        # Update position tracking
        if order_result.is_filled:
            if signal.action == OrderAction.BUY:
                self.position_manager.record_entry(signal.symbol)
            else:
                # Calculate realized P&L for sell
                position = self.position_manager.get_position(signal.symbol)
                if position:
                    realized_pnl = (order_result.avg_fill_price - position.avg_cost) * order_result.filled_shares
                    self.position_manager.record_exit(signal.symbol, realized_pnl)
                    self.risk_manager.record_trade(signal.symbol, realized_pnl, realized_pnl > 0)

            # Refresh positions
            self.position_manager.refresh_positions()
            self.risk_manager.update_positions(self.position_manager.get_open_positions())

        # Create result
        result = TradeResult(
            signal=signal,
            validation=validation,
            order=order_result,
            execution_time_ms=execution_time_ms,
            slippage=slippage,
            success=order_result.is_filled
        )

        # Track history
        self._trade_history.append(result)

        logger.info(
            f"Trade {'SUCCESS' if result.success else 'FAILED'}: "
            f"{signal.symbol} filled {order_result.filled_shares}/{shares_to_trade} "
            f"@ ${order_result.avg_fill_price:.2f} "
            f"(slippage: ${slippage:.3f}, time: {execution_time_ms}ms)"
        )

        # Send trade notification
        if self.notifier:
            self.notifier.notify_trade(
                action=signal.action.value,
                symbol=signal.symbol,
                shares=order_result.filled_shares or shares_to_trade,
                price=order_result.avg_fill_price or signal.signal_price,
                status="FILLED" if result.success else order_result.status.value,
                order_id=order_result.order_id,
                error=result.error_message
            )

        return result

    def execute_buy(
        self,
        symbol: str,
        shares: Optional[int] = None,
        reason: str = "",
        fallback_price: float = 0.0
    ) -> TradeResult:
        """
        Execute a buy order.

        Args:
            symbol: Stock ticker
            shares: Number of shares (auto-sized if None)
            reason: Reason for trade
            fallback_price: Price to use if IBKR market data unavailable (e.g. from AV quote)

        Returns:
            TradeResult
        """
        # Get current price (fall back to signal price if IBKR data unavailable)
        price = self.connection.get_current_price(symbol)
        if price is None and fallback_price > 0:
            logger.info(f"Using fallback price for {symbol}: ${fallback_price:.2f}")
            price = fallback_price
        if price is None:
            signal = TradeSignal(
                symbol=symbol,
                action=OrderAction.BUY,
                shares=1,
                signal_price=0.0,
                signal_time=datetime.now(),
                reason=reason
            )
            return TradeResult(
                signal=signal,
                validation=ValidationResult(
                    status=ValidationStatus.REJECTED,
                    signal=signal,
                    messages=[f"Could not get price for {symbol}"]
                ),
                success=False,
                error_message=f"Could not get price for {symbol}"
            )

        # Calculate position size if not specified
        if shares is None:
            account = self.connection.get_account_summary()
            if account:
                shares = self.risk_manager.calculate_position_size(symbol, price, account)
            else:
                shares = 0

        if shares <= 0:
            signal = TradeSignal(
                symbol=symbol,
                action=OrderAction.BUY,
                shares=1,
                signal_price=price,
                signal_time=datetime.now(),
                reason=reason
            )
            return TradeResult(
                signal=signal,
                validation=ValidationResult(
                    status=ValidationStatus.REJECTED,
                    signal=signal,
                    messages=["Insufficient funds for purchase"]
                ),
                success=False,
                error_message="Insufficient funds"
            )

        signal = TradeSignal(
            symbol=symbol,
            action=OrderAction.BUY,
            shares=shares,
            signal_price=price,
            signal_time=datetime.now(),
            reason=reason
        )

        return self.execute_signal(signal)

    def execute_sell(
        self,
        symbol: str,
        shares: Optional[int] = None,
        reason: str = "",
        fallback_price: float = 0.0
    ) -> TradeResult:
        """
        Execute a sell order.

        Args:
            symbol: Stock ticker
            shares: Number of shares (sells all if None)
            reason: Reason for trade
            fallback_price: Price to use if IBKR market data unavailable (e.g. from AV quote)

        Returns:
            TradeResult
        """
        # Get current position
        position = self.position_manager.get_position(symbol)
        if position is None or position.is_flat:
            signal = TradeSignal(
                symbol=symbol,
                action=OrderAction.SELL,
                shares=1,
                signal_price=max(fallback_price, 0.01),
                signal_time=datetime.now(),
                reason=reason
            )
            return TradeResult(
                signal=signal,
                validation=ValidationResult(
                    status=ValidationStatus.REJECTED,
                    signal=signal,
                    messages=[f"No position to sell for {symbol}"]
                ),
                success=False,
                error_message=f"No position to sell for {symbol}"
            )

        # Get current price (fall back to signal price if IBKR data unavailable)
        price = self.connection.get_current_price(symbol)
        if price is None and fallback_price > 0:
            logger.info(f"Using fallback price for {symbol}: ${fallback_price:.2f}")
            price = fallback_price
        if price is None:
            price = position.market_value / position.shares if position.shares else 0

        # Default to selling entire position
        shares = shares or position.shares

        signal = TradeSignal(
            symbol=symbol,
            action=OrderAction.SELL,
            shares=shares,
            signal_price=price,
            signal_time=datetime.now(),
            reason=reason
        )

        return self.execute_signal(signal)

    def close_all_positions(self, reason: str = "Close all") -> list[TradeResult]:
        """
        Close all open positions.

        Args:
            reason: Reason for closing

        Returns:
            List of TradeResults
        """
        results = []
        positions = self.position_manager.get_open_positions()

        for symbol, position in positions.items():
            result = self.execute_sell(symbol, position.shares, reason)
            results.append(result)

        return results

    def check_exits(self, check_function) -> list[TradeResult]:
        """
        Check positions for exit conditions and execute.

        Args:
            check_function: Function(symbol, position, hold_days) -> bool

        Returns:
            List of TradeResults for exits executed
        """
        results = []
        positions = self.position_manager.get_open_positions()

        for symbol, position in positions.items():
            hold_days = self.position_manager.get_hold_days(symbol) or 0

            if check_function(symbol, position, hold_days):
                result = self.execute_sell(symbol, reason=f"Exit condition met (hold: {hold_days}d)")
                results.append(result)

        return results

    def get_account_summary(self) -> Optional[AccountSummary]:
        """Get current account summary."""
        return self.connection.get_account_summary()

    def get_risk_summary(self) -> dict:
        """Get current risk status."""
        account = self.connection.get_account_summary()
        return self.risk_manager.get_risk_summary(account)

    def get_position_summary(self) -> dict:
        """Get current position summary."""
        return self.position_manager.get_position_summary()

    def get_trade_history(self) -> list[TradeResult]:
        """Get trade history for this session."""
        return self._trade_history.copy()

    def _log_session_summary(self) -> None:
        """Log session summary statistics."""
        if not self._session_start:
            return

        duration = datetime.now() - self._session_start
        total_trades = len(self._trade_history)
        successful = sum(1 for t in self._trade_history if t.success)
        failed = total_trades - successful

        total_slippage = sum(t.slippage for t in self._trade_history if t.success)
        avg_execution_time = (
            sum(t.execution_time_ms for t in self._trade_history if t.execution_time_ms) / total_trades
            if total_trades > 0 else 0
        )

        logger.info("=" * 60)
        logger.info("SESSION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Duration: {duration}")
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Total slippage: ${total_slippage:.2f}")
        logger.info(f"Avg execution time: {avg_execution_time:.0f}ms")
        logger.info("=" * 60)

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
