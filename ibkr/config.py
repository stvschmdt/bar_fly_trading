"""
Configuration settings for IBKR trading module.

Provides connection settings and trading parameters.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class IBKRConfig:
    """
    IBKR TWS/Gateway connection configuration.

    Attributes:
        host: TWS/Gateway host address
        port: TWS/Gateway port (7497=paper TWS, 7496=live TWS, 4002=paper Gateway, 4001=live Gateway)
        client_id: Unique client ID for this connection
        timeout: Connection timeout in seconds
        readonly: If True, no orders will be submitted
        account: Specific account ID (None for default)
    """
    host: str = "127.0.0.1"
    port: int = 7497  # Paper trading TWS default
    client_id: int = 1
    timeout: int = 30
    readonly: bool = False
    account: Optional[str] = None

    # Port reference:
    # 7497 = TWS Paper Trading
    # 7496 = TWS Live Trading
    # 4002 = IB Gateway Paper Trading
    # 4001 = IB Gateway Live Trading

    @classmethod
    def paper_tws(cls, client_id: int = 1) -> "IBKRConfig":
        """Create config for TWS paper trading."""
        return cls(port=7497, client_id=client_id)

    @classmethod
    def live_tws(cls, client_id: int = 1) -> "IBKRConfig":
        """Create config for TWS live trading."""
        return cls(port=7496, client_id=client_id, readonly=False)

    @classmethod
    def paper_gateway(cls, client_id: int = 1) -> "IBKRConfig":
        """Create config for IB Gateway paper trading."""
        return cls(port=4002, client_id=client_id)

    @classmethod
    def live_gateway(cls, client_id: int = 1, host: str = "127.0.0.1") -> "IBKRConfig":
        """Create config for IB Gateway live trading."""
        return cls(host=host, port=4001, client_id=client_id, readonly=False)

    @classmethod
    def remote_gateway(cls, host: str, port: int = 4001, client_id: int = 1, readonly: bool = False) -> "IBKRConfig":
        """Create config for remote IB Gateway connection."""
        return cls(host=host, port=port, client_id=client_id, readonly=readonly)

    @property
    def is_paper(self) -> bool:
        """Check if this is a paper trading configuration."""
        return self.port in (7497, 4002)

    @property
    def is_live(self) -> bool:
        """Check if this is a live trading configuration."""
        return self.port in (7496, 4001)


@dataclass
class TradingConfig:
    """
    Trading parameters and risk limits.

    Attributes:
        symbols: Set of symbols to trade
        position_size: Fraction of portfolio per position (0.1 = 10%)
        max_positions: Maximum concurrent positions
        max_position_value: Maximum value per position
        max_daily_trades: Maximum trades per day
        max_daily_loss: Maximum daily loss before halting
        max_daily_loss_pct: Maximum daily loss as % of portfolio
        use_market_orders: If True, use market orders; else limit orders
        limit_offset_pct: Limit price offset from market (e.g., 0.001 = 0.1%)
        order_timeout: Seconds to wait for order fill before canceling
    """
    symbols: set[str] = field(default_factory=set)
    position_size: float = 0.10  # 10% of portfolio per position
    max_positions: int = 10
    max_position_value: float = 50000.0  # Max $50k per position
    max_daily_trades: int = 20
    max_daily_loss: float = 5000.0  # Max $5k daily loss
    max_daily_loss_pct: float = 0.02  # Max 2% daily loss
    use_market_orders: bool = True
    limit_offset_pct: float = 0.001  # 0.1% limit offset
    order_timeout: int = 60  # 60 second timeout

    # Strategy-specific parameters (from RegressionMomentumStrategy)
    entry_reg_3d_threshold: float = 0.01   # 1% predicted return
    entry_reg_10d_threshold: float = 0.02  # 2% predicted return
    exit_reg_3d_threshold: float = 0.0     # Exit when prediction turns negative
    min_hold_days: int = 2
    max_hold_days: int = 13

    def __post_init__(self):
        if self.position_size <= 0 or self.position_size > 1:
            raise ValueError(f"position_size must be between 0 and 1, got {self.position_size}")
        if self.max_positions <= 0:
            raise ValueError(f"max_positions must be positive, got {self.max_positions}")
        if self.max_daily_loss_pct <= 0 or self.max_daily_loss_pct > 1:
            raise ValueError(f"max_daily_loss_pct must be between 0 and 1, got {self.max_daily_loss_pct}")


@dataclass
class PredictionConfig:
    """
    Configuration for loading ML predictions.

    Attributes:
        predictions_dir: Directory containing prediction CSV files
        reg_3d_file: Filename for 3-day regression predictions
        reg_10d_file: Filename for 10-day regression predictions
    """
    predictions_dir: str = ""
    reg_3d_file: str = "pred_reg_3d.csv"
    reg_10d_file: str = "pred_reg_10d.csv"

    @classmethod
    def default(cls) -> "PredictionConfig":
        """Create config with default stockformer output path."""
        default_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', 'mlr', 'stockformer', 'output', 'predictions')
        )
        return cls(predictions_dir=default_dir)

    @property
    def reg_3d_path(self) -> str:
        return os.path.join(self.predictions_dir, self.reg_3d_file)

    @property
    def reg_10d_path(self) -> str:
        return os.path.join(self.predictions_dir, self.reg_10d_file)


@dataclass
class LoggingConfig:
    """
    Logging configuration.

    Attributes:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_trades: Whether to log all trades to file
        log_orders: Whether to log all orders to file
    """
    log_dir: str = "./logs"
    log_level: str = "INFO"
    log_trades: bool = True
    log_orders: bool = True

    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)


# Default configurations for quick setup
DEFAULT_SYMBOLS = {
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "AMD", "INTC", "NFLX"
}


def create_paper_config(symbols: Optional[set[str]] = None) -> tuple[IBKRConfig, TradingConfig]:
    """Create default paper trading configuration."""
    return (
        IBKRConfig.paper_tws(),
        TradingConfig(symbols=symbols or DEFAULT_SYMBOLS)
    )


def create_live_config(symbols: Optional[set[str]] = None) -> tuple[IBKRConfig, TradingConfig]:
    """Create default live trading configuration with conservative limits."""
    return (
        IBKRConfig.live_tws(),
        TradingConfig(
            symbols=symbols or DEFAULT_SYMBOLS,
            position_size=0.05,  # More conservative: 5%
            max_positions=5,     # Fewer positions
            max_daily_trades=10, # Fewer trades
            max_daily_loss=2000, # Lower loss limit
            max_daily_loss_pct=0.01  # 1% max loss
        )
    )
