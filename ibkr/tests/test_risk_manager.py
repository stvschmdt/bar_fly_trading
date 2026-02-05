"""Unit tests for IBKR risk manager."""

import pytest
from datetime import datetime, date

from ibkr.config import TradingConfig
from ibkr.risk_manager import RiskManager
from ibkr.models import (
    TradeSignal, ValidationStatus, Position, AccountSummary, OrderAction
)


@pytest.fixture
def trading_config():
    """Create test trading configuration."""
    return TradingConfig(
        symbols={"AAPL", "GOOGL", "MSFT", "NVDA"},
        position_size=0.10,
        max_positions=5,
        max_position_value=50000.0,
        max_daily_trades=10,
        max_daily_loss=5000.0,
        max_daily_loss_pct=0.02
    )


@pytest.fixture
def risk_manager(trading_config):
    """Create test risk manager."""
    return RiskManager(trading_config)


@pytest.fixture
def account_summary():
    """Create test account summary."""
    return AccountSummary(
        net_liquidation=100000.0,
        available_funds=50000.0,
        buying_power=100000.0,
        gross_position_value=50000.0
    )


@pytest.fixture
def buy_signal():
    """Create test buy signal."""
    return TradeSignal(
        symbol="AAPL",
        action=OrderAction.BUY,
        shares=100,
        signal_price=150.0,
        signal_time=datetime.now(),
        reason="Test buy"
    )


@pytest.fixture
def sell_signal():
    """Create test sell signal."""
    return TradeSignal(
        symbol="AAPL",
        action=OrderAction.SELL,
        shares=100,
        signal_price=155.0,
        signal_time=datetime.now(),
        reason="Test sell"
    )


class TestValidateTrade:
    """Tests for trade validation."""

    def test_approve_valid_buy(self, risk_manager, buy_signal, account_summary):
        """Test approving a valid buy order."""
        result = risk_manager.validate_trade(buy_signal, account_summary)
        assert result.is_approved or result.status == ValidationStatus.WARNING
        assert not result.is_rejected

    def test_reject_invalid_symbol(self, risk_manager, account_summary):
        """Test rejecting trade for invalid symbol."""
        signal = TradeSignal(
            symbol="INVALID",
            action=OrderAction.BUY,
            shares=100,
            signal_price=100.0,
            signal_time=datetime.now()
        )
        result = risk_manager.validate_trade(signal, account_summary)
        assert result.is_rejected
        assert "not in allowed symbols" in result.messages[0]

    def test_reject_daily_trade_limit(self, risk_manager, buy_signal, account_summary):
        """Test rejecting trade when daily limit reached."""
        # Simulate reaching daily limit
        for _ in range(10):
            risk_manager.record_trade("AAPL", 100, True)

        result = risk_manager.validate_trade(buy_signal, account_summary)
        assert result.is_rejected
        assert "Daily trade limit" in result.messages[0]

    def test_reject_daily_loss_limit(self, risk_manager, buy_signal, account_summary):
        """Test rejecting trade when daily loss limit reached."""
        # Simulate large loss
        risk_manager.record_trade("GOOGL", -5000, False)

        result = risk_manager.validate_trade(buy_signal, account_summary)
        assert result.is_rejected
        assert "Daily loss limit" in result.messages[0]

    def test_reject_max_positions(self, risk_manager, buy_signal, account_summary):
        """Test rejecting trade when max positions reached."""
        # Add 5 positions (max)
        positions = {
            "GOOGL": Position("GOOGL", 100, 100.0),
            "MSFT": Position("MSFT", 50, 300.0),
            "NVDA": Position("NVDA", 25, 500.0),
            "AMD": Position("AMD", 200, 80.0),
            "INTC": Position("INTC", 300, 30.0),
        }
        risk_manager.update_positions(positions)

        result = risk_manager.validate_trade(buy_signal, account_summary, positions)
        assert result.is_rejected
        assert "Maximum positions" in result.messages[0]

    def test_reject_existing_position(self, risk_manager, buy_signal, account_summary):
        """Test rejecting buy when already have position."""
        positions = {"AAPL": Position("AAPL", 100, 145.0)}
        risk_manager.update_positions(positions)

        result = risk_manager.validate_trade(buy_signal, account_summary, positions)
        assert result.is_rejected
        assert "Already have position" in result.messages[0]

    def test_adjust_shares_for_position_value(self, risk_manager, account_summary):
        """Test share adjustment for max position value."""
        # Signal for 1000 shares at $100 = $100k > $50k max
        signal = TradeSignal(
            symbol="AAPL",
            action=OrderAction.BUY,
            shares=1000,
            signal_price=100.0,
            signal_time=datetime.now()
        )
        result = risk_manager.validate_trade(signal, account_summary)
        # Should adjust to 500 shares ($50k max / $100)
        assert result.adjusted_shares <= 500
        assert "exceeds max" in result.messages[0].lower() or "adjusted" in result.messages[0].lower()

    def test_adjust_shares_for_available_funds(self, risk_manager):
        """Test share adjustment for available funds."""
        account = AccountSummary(
            net_liquidation=100000.0,
            available_funds=5000.0,  # Only $5k available
            buying_power=100000.0,
            gross_position_value=95000.0
        )
        signal = TradeSignal(
            symbol="AAPL",
            action=OrderAction.BUY,
            shares=100,  # 100 * $150 = $15k > $5k available
            signal_price=150.0,
            signal_time=datetime.now()
        )
        result = risk_manager.validate_trade(signal, account)
        assert result.adjusted_shares <= 33  # ~$5k / $150

    def test_reject_sell_no_position(self, risk_manager, sell_signal, account_summary):
        """Test rejecting sell when no position."""
        result = risk_manager.validate_trade(sell_signal, account_summary)
        assert result.is_rejected
        assert "No position to sell" in result.messages[0]

    def test_approve_sell_with_position(self, risk_manager, sell_signal, account_summary):
        """Test approving sell when position exists."""
        positions = {"AAPL": Position("AAPL", 100, 145.0)}
        risk_manager.update_positions(positions)

        result = risk_manager.validate_trade(sell_signal, account_summary, positions)
        assert result.is_approved or result.status == ValidationStatus.WARNING

    def test_adjust_sell_shares_to_position(self, risk_manager, account_summary):
        """Test adjusting sell shares to actual position."""
        positions = {"AAPL": Position("AAPL", 50, 145.0)}  # Only have 50
        risk_manager.update_positions(positions)

        signal = TradeSignal(
            symbol="AAPL",
            action=OrderAction.SELL,
            shares=100,  # Trying to sell 100
            signal_price=155.0,
            signal_time=datetime.now()
        )
        result = risk_manager.validate_trade(signal, account_summary, positions)
        assert result.adjusted_shares == 50


class TestCalculatePositionSize:
    """Tests for position size calculation."""

    def test_calculate_position_size(self, risk_manager, account_summary):
        """Test basic position size calculation."""
        shares = risk_manager.calculate_position_size("AAPL", 150.0, account_summary)
        # 10% of $100k = $10k, $10k / $150 = 66 shares
        assert shares == 66

    def test_position_size_caps_at_max_value(self, risk_manager, account_summary):
        """Test position size capped at max value."""
        # With 10% of $100k = $10k, but max is $50k
        # For a $10 stock: $50k / $10 = 5000 shares max
        # But 10% rule: $10k / $10 = 1000 shares
        shares = risk_manager.calculate_position_size("PENNY", 10.0, account_summary)
        assert shares == 1000  # Limited by 10% rule

    def test_position_size_caps_at_available_funds(self, risk_manager):
        """Test position size capped at available funds."""
        account = AccountSummary(
            net_liquidation=100000.0,
            available_funds=5000.0,  # Only $5k available
            buying_power=100000.0,
            gross_position_value=95000.0
        )
        shares = risk_manager.calculate_position_size("AAPL", 150.0, account)
        assert shares == 33  # $5k / $150

    def test_position_size_zero_price(self, risk_manager, account_summary):
        """Test position size with zero price."""
        shares = risk_manager.calculate_position_size("AAPL", 0, account_summary)
        assert shares == 0


class TestRecordTrade:
    """Tests for trade recording."""

    def test_record_winning_trade(self, risk_manager):
        """Test recording a winning trade."""
        risk_manager.record_trade("AAPL", 500.0, True)
        stats = risk_manager.get_daily_stats()
        assert stats.trades_executed == 1
        assert stats.realized_pnl == 500.0
        assert stats.winning_trades == 1
        assert stats.losing_trades == 0

    def test_record_losing_trade(self, risk_manager):
        """Test recording a losing trade."""
        risk_manager.record_trade("AAPL", -200.0, False)
        stats = risk_manager.get_daily_stats()
        assert stats.trades_executed == 1
        assert stats.realized_pnl == -200.0
        assert stats.winning_trades == 0
        assert stats.losing_trades == 1

    def test_record_multiple_trades(self, risk_manager):
        """Test recording multiple trades."""
        risk_manager.record_trade("AAPL", 500.0, True)
        risk_manager.record_trade("GOOGL", -200.0, False)
        risk_manager.record_trade("MSFT", 300.0, True)

        stats = risk_manager.get_daily_stats()
        assert stats.trades_executed == 3
        assert stats.realized_pnl == 600.0
        assert stats.winning_trades == 2
        assert stats.losing_trades == 1
        assert stats.win_rate == pytest.approx(0.667, rel=0.01)


class TestTradingAllowed:
    """Tests for trading allowed checks."""

    def test_trading_allowed_initial(self, risk_manager):
        """Test trading is allowed initially."""
        allowed, reason = risk_manager.is_trading_allowed()
        assert allowed
        assert "allowed" in reason.lower()

    def test_trading_not_allowed_trade_limit(self, risk_manager):
        """Test trading not allowed at trade limit."""
        for _ in range(10):
            risk_manager.record_trade("AAPL", 100, True)

        allowed, reason = risk_manager.is_trading_allowed()
        assert not allowed
        assert "trade limit" in reason.lower()

    def test_trading_not_allowed_loss_limit(self, risk_manager):
        """Test trading not allowed at loss limit."""
        risk_manager.record_trade("AAPL", -5000, False)

        allowed, reason = risk_manager.is_trading_allowed()
        assert not allowed
        assert "loss limit" in reason.lower()


class TestRiskSummary:
    """Tests for risk summary."""

    def test_risk_summary(self, risk_manager, account_summary):
        """Test risk summary generation."""
        risk_manager.record_trade("AAPL", 500, True)
        risk_manager.record_trade("GOOGL", -200, False)

        summary = risk_manager.get_risk_summary(account_summary)

        assert summary["trades_today"] == 2
        assert summary["trades_remaining"] == 8
        assert summary["daily_pnl"] == 300.0
        assert summary["trading_allowed"] is True
        assert summary["portfolio_value"] == 100000.0


class TestResetDailyStats:
    """Tests for daily stats reset."""

    def test_reset_daily_stats(self, risk_manager):
        """Test resetting daily stats."""
        risk_manager.record_trade("AAPL", 500, True)
        risk_manager.record_trade("GOOGL", -200, False)

        risk_manager.reset_daily_stats()

        stats = risk_manager.get_daily_stats()
        assert stats.trades_executed == 0
        assert stats.realized_pnl == 0.0
        assert stats.winning_trades == 0
