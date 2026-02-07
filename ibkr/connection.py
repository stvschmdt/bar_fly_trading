"""
IBKR connection management using ib_insync.

Handles connection lifecycle, auto-reconnect, and connection status.
"""

import logging
import time
from typing import Optional, Callable
from datetime import datetime

from ib_insync import IB, Contract, Stock, util

from .config import IBKRConfig
from .models import AccountSummary

logger = logging.getLogger(__name__)


class IBKRConnection:
    """
    Manages connection to Interactive Brokers TWS/Gateway.

    Provides:
    - Connection establishment and teardown
    - Auto-reconnect on disconnect
    - Connection status monitoring
    - Account information retrieval
    """

    def __init__(self, config: IBKRConfig):
        """
        Initialize connection manager.

        Args:
            config: IBKR connection configuration
        """
        self.config = config
        self.ib = IB()
        self._connected = False
        self._last_connect_time: Optional[datetime] = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5
        self._reconnect_delay = 5  # seconds

        # Callbacks
        self._on_connect_callbacks: list[Callable] = []
        self._on_disconnect_callbacks: list[Callable] = []

        # Set up event handlers
        self.ib.connectedEvent += self._on_connected
        self.ib.disconnectedEvent += self._on_disconnected
        self.ib.errorEvent += self._on_error

    def connect(self) -> bool:
        """
        Connect to TWS/Gateway.

        Returns:
            True if connection successful, False otherwise
        """
        if self._connected:
            logger.info("Already connected to IBKR")
            return True

        try:
            logger.info(f"Connecting to IBKR at {self.config.host}:{self.config.port} "
                       f"(client_id={self.config.client_id})")

            self.ib.connect(
                host=self.config.host,
                port=self.config.port,
                clientId=self.config.client_id,
                timeout=self.config.timeout,
                readonly=self.config.readonly
            )

            self._connected = True
            self._last_connect_time = datetime.now()
            self._reconnect_attempts = 0

            logger.info("Successfully connected to IBKR")

            # Log account info
            if self.config.account:
                logger.info(f"Using account: {self.config.account}")
            else:
                accounts = self.ib.managedAccounts()
                logger.info(f"Available accounts: {accounts}")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from TWS/Gateway."""
        if self._connected:
            logger.info("Disconnecting from IBKR")
            self.ib.disconnect()
            self._connected = False

    def reconnect(self) -> bool:
        """
        Attempt to reconnect after disconnect.

        Returns:
            True if reconnection successful
        """
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error(f"Max reconnection attempts ({self._max_reconnect_attempts}) reached")
            return False

        self._reconnect_attempts += 1
        logger.info(f"Reconnection attempt {self._reconnect_attempts}/{self._max_reconnect_attempts}")

        # Wait before reconnecting
        time.sleep(self._reconnect_delay)

        # Ensure old connection is closed
        try:
            self.ib.disconnect()
        except Exception:
            pass

        return self.connect()

    @property
    def is_connected(self) -> bool:
        """Check if currently connected."""
        return self._connected and self.ib.isConnected()

    @property
    def connection_duration(self) -> Optional[float]:
        """Get connection duration in seconds."""
        if not self._last_connect_time:
            return None
        return (datetime.now() - self._last_connect_time).total_seconds()

    def get_account_summary(self) -> Optional[AccountSummary]:
        """
        Get current account summary.

        Returns:
            AccountSummary with current account values
        """
        if not self.is_connected:
            logger.error("Not connected to IBKR")
            return None

        try:
            if self.config.account:
                account = self.config.account
            else:
                # Auto-detect: prefer DU sub-accounts (paper) over DFO (FA master)
                all_accounts = self.ib.managedAccounts()
                account = next((a for a in all_accounts if a.startswith('DU')), None)
                if not account:
                    account = next((a for a in all_accounts if a.startswith('U')), None)
                if not account:
                    account = all_accounts[0]
            summary = self.ib.accountSummary(account)

            # Extract numeric values only (skip string fields like AccountType)
            values = {}
            for item in summary:
                if item.value:
                    try:
                        values[item.tag] = float(item.value)
                    except ValueError:
                        pass  # Skip non-numeric values

            return AccountSummary(
                net_liquidation=values.get("NetLiquidation", 0.0),
                total_cash=values.get("TotalCashValue", 0.0),
                available_funds=values.get("AvailableFunds", 0.0),
                buying_power=values.get("BuyingPower", 0.0),
                gross_position_value=values.get("GrossPositionValue", 0.0),
                realized_pnl=values.get("RealizedPnL", 0.0),
                unrealized_pnl=values.get("UnrealizedPnL", 0.0)
            )

        except Exception as e:
            logger.error(f"Failed to get account summary: {e}")
            return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Current price or None if unavailable
        """
        if not self.is_connected:
            return None

        try:
            contract = Stock(symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)

            # Request market data
            ticker = self.ib.reqMktData(contract, snapshot=True)
            self.ib.sleep(1)  # Wait for data

            price = ticker.marketPrice()
            self.ib.cancelMktData(contract)

            if util.isNan(price):
                # Try last price if market price unavailable
                price = ticker.last if not util.isNan(ticker.last) else None

            return price

        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {e}")
            return None

    def get_current_prices(self, symbols: set[str]) -> dict[str, float]:
        """
        Get current prices for multiple symbols.

        Args:
            symbols: Set of ticker symbols

        Returns:
            Dict mapping symbol to price
        """
        prices = {}
        for symbol in symbols:
            price = self.get_current_price(symbol)
            if price is not None:
                prices[symbol] = price
        return prices

    def create_stock_contract(self, symbol: str) -> Contract:
        """
        Create and qualify a stock contract.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Qualified Contract object
        """
        contract = Stock(symbol, "SMART", "USD")
        if self.is_connected:
            self.ib.qualifyContracts(contract)
        return contract

    def on_connect(self, callback: Callable) -> None:
        """Register callback for connect events."""
        self._on_connect_callbacks.append(callback)

    def on_disconnect(self, callback: Callable) -> None:
        """Register callback for disconnect events."""
        self._on_disconnect_callbacks.append(callback)

    def _on_connected(self) -> None:
        """Internal handler for connect event."""
        logger.info("IBKR connected event received")
        self._connected = True
        for callback in self._on_connect_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Connect callback error: {e}")

    def _on_disconnected(self) -> None:
        """Internal handler for disconnect event."""
        logger.warning("IBKR disconnected event received")
        self._connected = False
        for callback in self._on_disconnect_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Disconnect callback error: {e}")

    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract: Contract) -> None:
        """Internal handler for error events."""
        # Filter out non-critical errors
        if errorCode in (2104, 2106, 2158):  # Market data connection messages
            logger.debug(f"IBKR info {errorCode}: {errorString}")
        else:
            logger.error(f"IBKR error {errorCode} (reqId={reqId}): {errorString}")

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
        return False

    def sleep(self, seconds: float) -> None:
        """Sleep while processing events."""
        self.ib.sleep(seconds)
