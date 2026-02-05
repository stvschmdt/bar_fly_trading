"""Unit tests for IBKR models."""

import pytest
from datetime import datetime

from ibkr.models import (
    TradeSignal, ValidationResult, Position, OrderResult, TradeResult,
    AccountSummary, DailyStats, OrderAction, OrderType, OrderStatus,
    ValidationStatus
)


class TestTradeSignal:
    """Tests for TradeSignal dataclass."""

    def test_create_buy_signal(self):
        """Test creating a buy signal."""
        signal = TradeSignal(
            symbol="AAPL",
            action=OrderAction.BUY,
            shares=100,
            signal_price=150.0,
            signal_time=datetime.now(),
            reason="Test buy"
        )
        assert signal.symbol == "AAPL"
        assert signal.action == OrderAction.BUY
        assert signal.shares == 100
        assert signal.signal_price == 150.0

    def test_create_sell_signal(self):
        """Test creating a sell signal."""
        signal = TradeSignal(
            symbol="GOOGL",
            action=OrderAction.SELL,
            shares=50,
            signal_price=2800.0,
            signal_time=datetime.now()
        )
        assert signal.action == OrderAction.SELL
        assert signal.shares == 50

    def test_signal_with_predictions(self):
        """Test signal with ML predictions."""
        signal = TradeSignal(
            symbol="NVDA",
            action=OrderAction.BUY,
            shares=25,
            signal_price=500.0,
            signal_time=datetime.now(),
            pred_reg_3d=0.015,
            pred_reg_10d=0.025,
            adx_signal=1,
            cci_signal=0.5
        )
        assert signal.pred_reg_3d == 0.015
        assert signal.pred_reg_10d == 0.025
        assert signal.adx_signal == 1

    def test_invalid_shares_raises_error(self):
        """Test that zero or negative shares raises error."""
        with pytest.raises(ValueError):
            TradeSignal(
                symbol="AAPL",
                action=OrderAction.BUY,
                shares=0,
                signal_price=150.0,
                signal_time=datetime.now()
            )

        with pytest.raises(ValueError):
            TradeSignal(
                symbol="AAPL",
                action=OrderAction.BUY,
                shares=-10,
                signal_price=150.0,
                signal_time=datetime.now()
            )

    def test_invalid_price_raises_error(self):
        """Test that zero or negative price raises error."""
        with pytest.raises(ValueError):
            TradeSignal(
                symbol="AAPL",
                action=OrderAction.BUY,
                shares=100,
                signal_price=0,
                signal_time=datetime.now()
            )


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_approved_result(self):
        """Test approved validation result."""
        signal = TradeSignal(
            symbol="AAPL",
            action=OrderAction.BUY,
            shares=100,
            signal_price=150.0,
            signal_time=datetime.now()
        )
        result = ValidationResult(
            status=ValidationStatus.APPROVED,
            signal=signal
        )
        assert result.is_approved
        assert not result.is_rejected

    def test_rejected_result(self):
        """Test rejected validation result."""
        signal = TradeSignal(
            symbol="AAPL",
            action=OrderAction.BUY,
            shares=100,
            signal_price=150.0,
            signal_time=datetime.now()
        )
        result = ValidationResult(
            status=ValidationStatus.REJECTED,
            signal=signal,
            messages=["Daily limit exceeded"]
        )
        assert result.is_rejected
        assert not result.is_approved
        assert "Daily limit exceeded" in result.messages

    def test_warning_result_with_adjusted_shares(self):
        """Test warning result with share adjustment."""
        signal = TradeSignal(
            symbol="AAPL",
            action=OrderAction.BUY,
            shares=1000,
            signal_price=150.0,
            signal_time=datetime.now()
        )
        result = ValidationResult(
            status=ValidationStatus.WARNING,
            signal=signal,
            messages=["Shares adjusted due to position limit"],
            adjusted_shares=500,
            max_allowed_shares=500
        )
        assert not result.is_rejected
        assert result.adjusted_shares == 500


class TestPosition:
    """Tests for Position dataclass."""

    def test_long_position(self):
        """Test long position properties."""
        pos = Position(
            symbol="AAPL",
            shares=100,
            avg_cost=145.0,
            market_value=15000.0,
            unrealized_pnl=500.0
        )
        assert pos.is_long
        assert not pos.is_short
        assert not pos.is_flat
        assert pos.cost_basis == 14500.0

    def test_short_position(self):
        """Test short position properties."""
        pos = Position(
            symbol="TSLA",
            shares=-50,
            avg_cost=200.0,
            market_value=-9500.0,
            unrealized_pnl=500.0
        )
        assert pos.is_short
        assert not pos.is_long
        assert not pos.is_flat

    def test_flat_position(self):
        """Test flat (no) position."""
        pos = Position(
            symbol="MSFT",
            shares=0,
            avg_cost=0.0
        )
        assert pos.is_flat
        assert not pos.is_long
        assert not pos.is_short


class TestOrderResult:
    """Tests for OrderResult dataclass."""

    def test_filled_order(self):
        """Test filled order properties."""
        result = OrderResult(
            order_id=12345,
            symbol="AAPL",
            action=OrderAction.BUY,
            shares=100,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_shares=100,
            avg_fill_price=150.25,
            filled_time=datetime.now()
        )
        assert result.is_filled
        assert result.is_complete
        assert not result.is_pending
        assert result.fill_ratio == 1.0

    def test_partial_fill(self):
        """Test partially filled order."""
        result = OrderResult(
            order_id=12345,
            symbol="AAPL",
            action=OrderAction.BUY,
            shares=100,
            order_type=OrderType.LIMIT,
            status=OrderStatus.SUBMITTED,
            filled_shares=50,
            avg_fill_price=150.0
        )
        assert not result.is_filled
        assert result.is_pending
        assert result.fill_ratio == 0.5

    def test_pending_order(self):
        """Test pending order status."""
        result = OrderResult(
            order_id=12345,
            symbol="NVDA",
            action=OrderAction.SELL,
            shares=25,
            order_type=OrderType.MARKET,
            status=OrderStatus.SUBMITTED,
            submitted_time=datetime.now()
        )
        assert result.is_pending
        assert not result.is_complete

    def test_error_order(self):
        """Test error order status."""
        result = OrderResult(
            order_id=-1,
            symbol="INVALID",
            action=OrderAction.BUY,
            shares=100,
            order_type=OrderType.MARKET,
            status=OrderStatus.ERROR,
            error_message="Invalid symbol"
        )
        assert result.is_complete
        assert not result.is_filled
        assert result.error_message == "Invalid symbol"


class TestTradeResult:
    """Tests for TradeResult dataclass."""

    def test_successful_trade(self):
        """Test successful trade result."""
        signal = TradeSignal(
            symbol="AAPL",
            action=OrderAction.BUY,
            shares=100,
            signal_price=150.0,
            signal_time=datetime.now()
        )
        validation = ValidationResult(
            status=ValidationStatus.APPROVED,
            signal=signal
        )
        order = OrderResult(
            order_id=12345,
            symbol="AAPL",
            action=OrderAction.BUY,
            shares=100,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            filled_shares=100,
            avg_fill_price=150.10
        )
        result = TradeResult(
            signal=signal,
            validation=validation,
            order=order,
            execution_time_ms=250,
            slippage=0.10,
            success=True
        )
        assert result.success
        assert result.slippage_pct == pytest.approx(0.0667, rel=0.01)

    def test_failed_trade(self):
        """Test failed trade result."""
        signal = TradeSignal(
            symbol="AAPL",
            action=OrderAction.BUY,
            shares=100,
            signal_price=150.0,
            signal_time=datetime.now()
        )
        validation = ValidationResult(
            status=ValidationStatus.REJECTED,
            signal=signal,
            messages=["Insufficient funds"]
        )
        result = TradeResult(
            signal=signal,
            validation=validation,
            success=False,
            error_message="Insufficient funds"
        )
        assert not result.success
        assert result.order is None


class TestAccountSummary:
    """Tests for AccountSummary dataclass."""

    def test_account_summary(self):
        """Test account summary properties."""
        summary = AccountSummary(
            net_liquidation=100000.0,
            total_cash=45000.0,
            available_funds=50000.0,
            buying_power=100000.0,
            gross_position_value=50000.0,
            realized_pnl=1000.0,
            unrealized_pnl=500.0
        )
        assert summary.total_pnl == 1500.0


class TestDailyStats:
    """Tests for DailyStats dataclass."""

    def test_daily_stats_win_rate(self):
        """Test daily stats win rate calculation."""
        stats = DailyStats(
            date=datetime.now(),
            trades_executed=10,
            realized_pnl=500.0,
            winning_trades=7,
            losing_trades=3
        )
        assert stats.win_rate == 0.7

    def test_daily_stats_no_trades(self):
        """Test daily stats with no trades."""
        stats = DailyStats(date=datetime.now())
        assert stats.win_rate == 0.0
        assert stats.trades_executed == 0
