"""Unit tests for IBKR configuration."""

import pytest
import os

from ibkr.config import (
    IBKRConfig, TradingConfig, PredictionConfig, LoggingConfig,
    create_paper_config, create_live_config, DEFAULT_SYMBOLS
)


class TestIBKRConfig:
    """Tests for IBKRConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = IBKRConfig()
        assert config.host == "127.0.0.1"
        assert config.port == 7497  # Paper TWS default
        assert config.client_id == 1
        assert config.timeout == 30
        assert config.readonly is False

    def test_paper_tws_config(self):
        """Test paper TWS configuration."""
        config = IBKRConfig.paper_tws(client_id=5)
        assert config.port == 7497
        assert config.client_id == 5
        assert config.is_paper
        assert not config.is_live

    def test_live_tws_config(self):
        """Test live TWS configuration."""
        config = IBKRConfig.live_tws(client_id=2)
        assert config.port == 7496
        assert config.client_id == 2
        assert config.is_live
        assert not config.is_paper

    def test_paper_gateway_config(self):
        """Test paper Gateway configuration."""
        config = IBKRConfig.paper_gateway()
        assert config.port == 4002
        assert config.is_paper

    def test_live_gateway_config(self):
        """Test live Gateway configuration."""
        config = IBKRConfig.live_gateway()
        assert config.port == 4001
        assert config.is_live


class TestTradingConfig:
    """Tests for TradingConfig dataclass."""

    def test_default_trading_config(self):
        """Test default trading configuration."""
        config = TradingConfig()
        assert config.position_size == 0.10
        assert config.max_positions == 10
        assert config.max_position_value == 50000.0
        assert config.max_daily_trades == 20
        assert config.max_daily_loss == 5000.0
        assert config.max_daily_loss_pct == 0.02

    def test_trading_config_with_symbols(self):
        """Test trading config with symbols."""
        symbols = {"AAPL", "GOOGL", "MSFT"}
        config = TradingConfig(symbols=symbols)
        assert config.symbols == symbols

    def test_strategy_parameters(self):
        """Test strategy-specific parameters."""
        config = TradingConfig()
        assert config.entry_reg_3d_threshold == 0.01
        assert config.entry_reg_10d_threshold == 0.02
        assert config.exit_reg_3d_threshold == 0.0
        assert config.min_hold_days == 2
        assert config.max_hold_days == 13

    def test_invalid_position_size(self):
        """Test that invalid position size raises error."""
        with pytest.raises(ValueError):
            TradingConfig(position_size=0)

        with pytest.raises(ValueError):
            TradingConfig(position_size=1.5)

        with pytest.raises(ValueError):
            TradingConfig(position_size=-0.1)

    def test_invalid_max_positions(self):
        """Test that invalid max positions raises error."""
        with pytest.raises(ValueError):
            TradingConfig(max_positions=0)

        with pytest.raises(ValueError):
            TradingConfig(max_positions=-1)

    def test_invalid_daily_loss_pct(self):
        """Test that invalid daily loss pct raises error."""
        with pytest.raises(ValueError):
            TradingConfig(max_daily_loss_pct=0)

        with pytest.raises(ValueError):
            TradingConfig(max_daily_loss_pct=1.5)


class TestPredictionConfig:
    """Tests for PredictionConfig dataclass."""

    def test_default_prediction_config(self):
        """Test default prediction configuration."""
        config = PredictionConfig()
        assert config.reg_3d_file == "pred_reg_3d.csv"
        assert config.reg_10d_file == "pred_reg_10d.csv"

    def test_prediction_paths(self):
        """Test prediction file path construction."""
        config = PredictionConfig(predictions_dir="/data/predictions")
        assert config.reg_3d_path == "/data/predictions/pred_reg_3d.csv"
        assert config.reg_10d_path == "/data/predictions/pred_reg_10d.csv"

    def test_default_factory(self):
        """Test default factory method."""
        config = PredictionConfig.default()
        assert "stockformer" in config.predictions_dir
        assert "predictions" in config.predictions_dir


class TestLoggingConfig:
    """Tests for LoggingConfig dataclass."""

    def test_default_logging_config(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        assert config.log_level == "INFO"
        assert config.log_trades is True
        assert config.log_orders is True

    def test_log_dir_creation(self, tmp_path):
        """Test that log directory is created."""
        log_dir = str(tmp_path / "test_logs")
        config = LoggingConfig(log_dir=log_dir)
        assert os.path.exists(log_dir)


class TestFactoryFunctions:
    """Tests for configuration factory functions."""

    def test_create_paper_config(self):
        """Test paper configuration factory."""
        ibkr, trading = create_paper_config()
        assert ibkr.is_paper
        assert trading.symbols == DEFAULT_SYMBOLS

    def test_create_paper_config_custom_symbols(self):
        """Test paper config with custom symbols."""
        symbols = {"AAPL", "TSLA"}
        ibkr, trading = create_paper_config(symbols)
        assert trading.symbols == symbols

    def test_create_live_config(self):
        """Test live configuration factory."""
        ibkr, trading = create_live_config()
        assert ibkr.is_live
        # Live config should be more conservative
        assert trading.position_size == 0.05
        assert trading.max_positions == 5
        assert trading.max_daily_loss == 2000

    def test_default_symbols(self):
        """Test default symbols set."""
        assert "AAPL" in DEFAULT_SYMBOLS
        assert "MSFT" in DEFAULT_SYMBOLS
        assert "NVDA" in DEFAULT_SYMBOLS
        assert len(DEFAULT_SYMBOLS) == 10
