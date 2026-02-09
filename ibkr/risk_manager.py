"""
Risk management and pre-trade validation for IBKR trading.

Validates trades against position limits, daily loss limits, and other risk rules.
"""

import logging
from datetime import datetime, date
from typing import Optional

from .config import TradingConfig
from .models import (
    TradeSignal, ValidationResult, ValidationStatus,
    Position, AccountSummary, DailyStats, OrderAction
)

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Pre-trade risk validation and position sizing.

    Validates:
    - Position size limits
    - Maximum positions
    - Daily loss limits
    - Daily trade count limits
    - Symbol restrictions
    """

    def __init__(self, config: TradingConfig):
        """
        Initialize risk manager.

        Args:
            config: Trading configuration with risk limits
        """
        self.config = config
        self._daily_stats: dict[date, DailyStats] = {}
        self._positions: dict[str, Position] = {}

    def validate_trade(
        self,
        signal: TradeSignal,
        account: Optional[AccountSummary] = None,
        current_positions: Optional[dict[str, Position]] = None
    ) -> ValidationResult:
        """
        Validate a trade signal against risk rules.

        Args:
            signal: Trade signal to validate
            account: Current account summary
            current_positions: Current open positions

        Returns:
            ValidationResult with approval status and any adjustments
        """
        messages = []
        adjusted_shares = signal.shares
        max_allowed = signal.shares

        # Update internal position tracking
        if current_positions:
            self._positions = current_positions

        # 1. Check if symbol is in allowed list
        if self.config.symbols and signal.symbol not in self.config.symbols:
            return ValidationResult(
                status=ValidationStatus.REJECTED,
                signal=signal,
                messages=[f"Symbol {signal.symbol} not in allowed symbols list"]
            )

        # 2. Check daily trade limit
        today = date.today()
        daily_stats = self._get_daily_stats(today)
        if daily_stats.trades_executed >= self.config.max_daily_trades:
            return ValidationResult(
                status=ValidationStatus.REJECTED,
                signal=signal,
                messages=[f"Daily trade limit reached ({self.config.max_daily_trades})"]
            )

        # 3. Check daily loss limit
        if daily_stats.realized_pnl <= -self.config.max_daily_loss:
            return ValidationResult(
                status=ValidationStatus.REJECTED,
                signal=signal,
                messages=[f"Daily loss limit reached (${self.config.max_daily_loss:.2f})"]
            )

        # 4. For BUY orders, check position limits
        if signal.action == OrderAction.BUY:
            # Check max positions
            open_positions = len([p for p in self._positions.values() if not p.is_flat])
            if open_positions >= self.config.max_positions:
                return ValidationResult(
                    status=ValidationStatus.REJECTED,
                    signal=signal,
                    messages=[f"Maximum positions reached ({self.config.max_positions})"]
                )

            # Check if we already have a position in this symbol
            if signal.symbol in self._positions and not self._positions[signal.symbol].is_flat:
                messages.append(f"Already have position in {signal.symbol}")
                return ValidationResult(
                    status=ValidationStatus.REJECTED,
                    signal=signal,
                    messages=messages
                )

            # Check position value limit
            if account:
                # Use buffered price: price * (1 + fee_buffer) for cash calculations
                # This reserves headroom for commissions, slippage, and ask spread
                buffered_price = signal.signal_price * (1 + self.config.fee_buffer_pct)
                position_value = signal.shares * signal.signal_price

                # Check against max position value
                if position_value > self.config.max_position_value:
                    max_allowed = int(self.config.max_position_value / signal.signal_price)
                    adjusted_shares = min(adjusted_shares, max_allowed)
                    messages.append(
                        f"Position value ${position_value:.2f} exceeds max "
                        f"${self.config.max_position_value:.2f}, adjusted to {adjusted_shares} shares"
                    )

                # Check against position size % of portfolio
                max_by_pct = int(
                    (account.net_liquidation * self.config.position_size) / signal.signal_price
                )
                if adjusted_shares > max_by_pct:
                    adjusted_shares = max_by_pct
                    messages.append(
                        f"Position size exceeds {self.config.position_size*100:.0f}% limit, "
                        f"adjusted to {adjusted_shares} shares"
                    )

                # Check available funds (with fee + slippage buffer)
                required_funds = adjusted_shares * buffered_price
                if required_funds > account.available_funds:
                    max_by_funds = int(account.available_funds / buffered_price)
                    adjusted_shares = min(adjusted_shares, max_by_funds)
                    messages.append(
                        f"Insufficient funds after {self.config.fee_buffer_pct*100:.1f}% buffer "
                        f"(need ${required_funds:.2f}, have ${account.available_funds:.2f}), "
                        f"adjusted to {adjusted_shares} shares"
                    )

                # Check minimum order value
                order_value = adjusted_shares * signal.signal_price
                if adjusted_shares > 0 and order_value < self.config.min_order_value:
                    return ValidationResult(
                        status=ValidationStatus.REJECTED,
                        signal=signal,
                        messages=messages + [
                            f"Order value ${order_value:.2f} below minimum "
                            f"${self.config.min_order_value:.2f}"
                        ]
                    )

                # Check max symbol exposure (existing + new order)
                existing_value = 0.0
                if signal.symbol in self._positions:
                    existing_value = abs(self._positions[signal.symbol].market_value)
                new_total = existing_value + (adjusted_shares * signal.signal_price)
                max_exposure = account.net_liquidation * self.config.max_symbol_exposure_pct
                if new_total > max_exposure:
                    max_new_shares = int((max_exposure - existing_value) / signal.signal_price)
                    if max_new_shares <= 0:
                        return ValidationResult(
                            status=ValidationStatus.REJECTED,
                            signal=signal,
                            messages=messages + [
                                f"Symbol exposure ${new_total:.2f} would exceed "
                                f"{self.config.max_symbol_exposure_pct*100:.0f}% limit "
                                f"(${max_exposure:.2f})"
                            ]
                        )
                    adjusted_shares = min(adjusted_shares, max_new_shares)
                    messages.append(
                        f"Symbol exposure capped at "
                        f"{self.config.max_symbol_exposure_pct*100:.0f}%, "
                        f"adjusted to {adjusted_shares} shares"
                    )

                # Check daily loss % limit
                if account.net_liquidation > 0:
                    loss_pct = abs(daily_stats.realized_pnl) / account.net_liquidation
                    if loss_pct >= self.config.max_daily_loss_pct:
                        return ValidationResult(
                            status=ValidationStatus.REJECTED,
                            signal=signal,
                            messages=[
                                f"Daily loss {loss_pct*100:.2f}% exceeds limit "
                                f"{self.config.max_daily_loss_pct*100:.1f}%"
                            ]
                        )

        # 5. For SELL orders, verify we have the position
        if signal.action == OrderAction.SELL:
            if signal.symbol not in self._positions or self._positions[signal.symbol].is_flat:
                return ValidationResult(
                    status=ValidationStatus.REJECTED,
                    signal=signal,
                    messages=[f"No position to sell for {signal.symbol}"]
                )

            # Adjust shares to what we actually have
            held_shares = self._positions[signal.symbol].shares
            if signal.shares > held_shares:
                adjusted_shares = held_shares
                messages.append(
                    f"Requested {signal.shares} shares but only hold {held_shares}, "
                    f"adjusted to {adjusted_shares}"
                )

        # Final validation
        if adjusted_shares <= 0:
            return ValidationResult(
                status=ValidationStatus.REJECTED,
                signal=signal,
                messages=messages + ["No shares to trade after adjustments"]
            )

        # Approved with possible adjustments
        status = ValidationStatus.WARNING if messages else ValidationStatus.APPROVED
        return ValidationResult(
            status=status,
            signal=signal,
            messages=messages,
            adjusted_shares=adjusted_shares,
            max_allowed_shares=max_allowed
        )

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        account: AccountSummary
    ) -> int:
        """
        Calculate appropriate position size for a symbol.

        Uses buffered price (price * (1 + fee_buffer)) so that the resulting
        order leaves headroom for commissions, slippage, and ask spread.

        Args:
            symbol: Stock ticker
            price: Current price (last or ask)
            account: Account summary

        Returns:
            Number of shares to buy
        """
        if price <= 0 or account.net_liquidation <= 0:
            return 0

        buffered_price = price * (1 + self.config.fee_buffer_pct)

        # Calculate based on position size %
        target_value = account.net_liquidation * self.config.position_size

        # Cap at max position value
        target_value = min(target_value, self.config.max_position_value)

        # Cap at available funds
        target_value = min(target_value, account.available_funds)

        shares = int(target_value / buffered_price)

        # Enforce minimum order value
        if shares > 0 and shares * price < self.config.min_order_value:
            return 0

        return max(0, shares)

    def record_trade(self, symbol: str, pnl: float, is_win: bool) -> None:
        """
        Record a completed trade for daily stats.

        Args:
            symbol: Stock ticker
            pnl: Realized P&L from trade
            is_win: Whether trade was profitable
        """
        today = date.today()
        stats = self._get_daily_stats(today)
        stats.trades_executed += 1
        stats.realized_pnl += pnl

        if is_win:
            stats.winning_trades += 1
        else:
            stats.losing_trades += 1

        # Update max drawdown
        if stats.realized_pnl < stats.max_drawdown:
            stats.max_drawdown = stats.realized_pnl

        logger.info(
            f"Recorded trade: {symbol} P&L=${pnl:.2f}, "
            f"Daily total=${stats.realized_pnl:.2f}, "
            f"Trades today={stats.trades_executed}"
        )

    def update_positions(self, positions: dict[str, Position]) -> None:
        """Update internal position tracking."""
        self._positions = positions

    def get_daily_stats(self) -> DailyStats:
        """Get today's trading statistics."""
        return self._get_daily_stats(date.today())

    def _get_daily_stats(self, day: date) -> DailyStats:
        """Get or create daily stats for a date."""
        if day not in self._daily_stats:
            self._daily_stats[day] = DailyStats(date=datetime.combine(day, datetime.min.time()))
        return self._daily_stats[day]

    def reset_daily_stats(self) -> None:
        """Reset daily statistics (call at start of trading day)."""
        today = date.today()
        self._daily_stats[today] = DailyStats(date=datetime.combine(today, datetime.min.time()))
        logger.info("Daily stats reset")

    def is_trading_allowed(self) -> tuple[bool, str]:
        """
        Check if trading is currently allowed.

        Returns:
            Tuple of (allowed, reason)
        """
        today = date.today()
        stats = self._get_daily_stats(today)

        if stats.trades_executed >= self.config.max_daily_trades:
            return False, f"Daily trade limit reached ({self.config.max_daily_trades})"

        if stats.realized_pnl <= -self.config.max_daily_loss:
            return False, f"Daily loss limit reached (${self.config.max_daily_loss:.2f})"

        return True, "Trading allowed"

    def get_risk_summary(self, account: Optional[AccountSummary] = None) -> dict:
        """
        Get current risk status summary.

        Args:
            account: Optional account summary for additional metrics

        Returns:
            Dict with risk metrics
        """
        stats = self.get_daily_stats()
        open_positions = len([p for p in self._positions.values() if not p.is_flat])

        summary = {
            "trades_today": stats.trades_executed,
            "trades_remaining": self.config.max_daily_trades - stats.trades_executed,
            "daily_pnl": stats.realized_pnl,
            "daily_loss_remaining": self.config.max_daily_loss + stats.realized_pnl,
            "open_positions": open_positions,
            "positions_remaining": self.config.max_positions - open_positions,
            "win_rate_today": stats.win_rate,
            "trading_allowed": self.is_trading_allowed()[0]
        }

        if account:
            summary["portfolio_value"] = account.net_liquidation
            summary["available_funds"] = account.available_funds
            summary["daily_loss_pct"] = (
                abs(stats.realized_pnl) / account.net_liquidation * 100
                if account.net_liquidation > 0 else 0
            )

        return summary
