#!/usr/bin/env python3
"""
Bollinger Band Shadow Strategy

Scans all_data files for stocks that crossed Bollinger bands on the most recent day.
- Lower band crossover → BUY signal (oversold)
- Upper band crossover → SELL signal (overbought)

This is a SHADOW strategy - it does NOT execute trades.
It ALWAYS connects to IB Gateway to get real account data (cash, positions).
It sends notification emails with proposed orders for review.

Requirements:
    - SSH tunnel to AWS Gateway must be running
    - Notification env vars must be set (IBKR_SMTP_*, IBKR_NOTIFY_EMAIL)

Usage:
    # Start SSH tunnel first
    ssh -L 4001:127.0.0.1:4001 sschmidt@54.90.246.184 -N &

    # Run strategy (connects to Gateway, sends emails) - SHADOW MODE
    python bollinger_shadow_strategy.py --data-path ~/proj/bar_fly_trading/all_data.csv

    # Test without Gateway connection (for debugging data/signals only)
    python bollinger_shadow_strategy.py --data-path ~/data/all_data.csv --skip-live

    # EXECUTE trades for specific symbols (must explicitly list each symbol)
    python bollinger_shadow_strategy.py --data-path ~/proj/bar_fly_trading/all_data.csv --execute AAPL
    python bollinger_shadow_strategy.py --data-path ~/proj/bar_fly_trading/all_data.csv --execute AAPL,MSFT

    # Use watchlist to filter/sort signals
    python bollinger_shadow_strategy.py --data-path ~/data/all_data.csv --skip-live --no-notify \
        --watchlist api_data/watchlist.csv --watchlist-mode filter

    # Sort signals by watchlist order (watchlist symbols first)
    python bollinger_shadow_strategy.py --data-path ~/data/all_data.csv --skip-live --no-notify \
        --watchlist api_data/watchlist.csv --watchlist-mode sort
"""

import argparse
import glob
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

# Add parent directory for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

try:
    from ibkr.config import IBKRConfig, TradingConfig
    from ibkr.connection import IBKRConnection
    from ibkr.notifier import TradeNotifier
    from ibkr.trade_executor import TradeExecutor
    from ibkr.models import TradeSignal, OrderAction
except ModuleNotFoundError:
    from config import IBKRConfig, TradingConfig
    from connection import IBKRConnection
    from notifier import TradeNotifier
    from trade_executor import TradeExecutor
    from models import TradeSignal, OrderAction

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_watchlist(watchlist_path: str) -> list[str]:
    """
    Load watchlist from CSV file and return ordered list of symbols.

    Args:
        watchlist_path: Path to CSV file with 'Symbol' column

    Returns:
        List of symbols in watchlist order
    """
    if not os.path.exists(watchlist_path):
        logger.warning(f"Watchlist file not found: {watchlist_path}")
        return []

    df = pd.read_csv(watchlist_path)

    # Handle different column name conventions
    symbol_col = None
    for col in ['Symbol', 'symbol', 'SYMBOL', 'ticker', 'Ticker', 'TICKER']:
        if col in df.columns:
            symbol_col = col
            break

    if symbol_col is None:
        logger.warning(f"No symbol column found in watchlist. Expected 'Symbol' or 'ticker'")
        return []

    return df[symbol_col].tolist()


@dataclass
class BollingerSignal:
    """A Bollinger band crossover signal."""
    symbol: str
    signal_type: str  # "BUY" or "SELL"
    close_price: float
    bb_lower: float
    bb_upper: float
    bb_middle: float
    signal_date: str
    shares_proposed: int = 0
    estimated_cost: float = 0.0
    reason: str = ""
    can_execute: bool = True
    block_reason: str = ""
    # Additional context
    bb_width_pct: float = 0.0  # Band width as % of middle
    distance_from_band_pct: float = 0.0  # How far past the band
    volume: float = 0.0
    rsi: Optional[float] = None
    prev_close: float = 0.0
    bull_bear_delta: Optional[float] = None  # Bull/bear sentiment indicator


class BollingerShadowStrategy:
    """
    Shadow strategy that scans for Bollinger band crossovers.

    Does NOT execute trades - only sends notifications.
    """

    # Default position size as percentage of available cash
    POSITION_SIZE_PCT = 0.05  # 5% of cash per position

    # Minimum cash to maintain
    MIN_CASH_RESERVE = 1000.0

    def __init__(
        self,
        data_path: str,
        notifier: Optional[TradeNotifier] = None,
        position_size_pct: float = 0.05,
        shares_per_trade: int = 10,
        watchlist: Optional[list[str]] = None,
        watchlist_mode: str = 'sort'
    ):
        """
        Initialize the shadow strategy.

        Args:
            data_path: Path to all_data CSV file(s), supports glob patterns
            notifier: Optional notifier for sending alerts
            position_size_pct: Percentage of cash to use per position
            shares_per_trade: Default shares per trade if not calculated
            watchlist: Optional ordered list of symbols for filtering/sorting
            watchlist_mode: 'sort' (default) or 'filter'
        """
        self.data_path = data_path
        self.notifier = notifier  # Can be None to disable notifications
        self.position_size_pct = position_size_pct
        self.shares_per_trade = shares_per_trade
        self.watchlist = watchlist or []
        self.watchlist_mode = watchlist_mode

        self.data: Optional[pd.DataFrame] = None
        self.signals: list[BollingerSignal] = []

        # Account info (populated if live_check is enabled)
        self.cash_balance: Optional[float] = None
        self.portfolio: dict[str, int] = {}  # symbol -> shares
        self.account_id: Optional[str] = None

    def load_data(self) -> bool:
        """Load data from CSV file(s)."""
        try:
            # Handle glob patterns
            if '*' in self.data_path:
                files = glob.glob(self.data_path)
                if not files:
                    logger.error(f"No files found matching: {self.data_path}")
                    return False

                dfs = []
                for f in files:
                    logger.info(f"Loading: {f}")
                    dfs.append(pd.read_csv(f))
                self.data = pd.concat(dfs, ignore_index=True)
            else:
                if not os.path.exists(self.data_path):
                    logger.error(f"File not found: {self.data_path}")
                    return False
                logger.info(f"Loading: {self.data_path}")
                self.data = pd.read_csv(self.data_path)

            # Ensure date column is datetime
            self.data['date'] = pd.to_datetime(self.data['date'])

            # Check required columns
            required = ['date', 'symbol', 'adjusted_close', 'bbands_lower_20', 'bbands_upper_20']
            missing = [c for c in required if c not in self.data.columns]
            if missing:
                logger.error(f"Missing required columns: {missing}")
                return False

            logger.info(f"Loaded {len(self.data)} rows, {self.data['symbol'].nunique()} symbols")
            return True

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False

    def get_live_account_info(self, host: str = "127.0.0.1", port: int = 4001) -> bool:
        """
        Connect to IB Gateway and get live account info.

        Args:
            host: Gateway host (default: localhost via SSH tunnel)
            port: Gateway port (default: 4001 for live)

        Returns:
            True if successful
        """
        try:
            config = IBKRConfig.remote_gateway(host=host, port=port, client_id=99)
            connection = IBKRConnection(config)

            if not connection.connect():
                logger.warning("Could not connect to IB Gateway - skipping live checks")
                return False

            try:
                # Get account info
                accounts = connection.ib.managedAccounts()
                self.account_id = accounts[0] if accounts else None

                # Get account summary for cash balance
                summary = connection.ib.accountSummary(self.account_id)
                for item in summary:
                    if item.tag == "TotalCashValue":
                        try:
                            self.cash_balance = float(item.value)
                        except ValueError:
                            pass

                # Get portfolio positions
                positions = connection.ib.positions()
                for pos in positions:
                    if pos.contract.secType == "STK":
                        self.portfolio[pos.contract.symbol] = int(pos.position)

                logger.info(f"Account: {self.account_id}")
                logger.info(f"Cash Balance: ${self.cash_balance:,.2f}")
                logger.info(f"Positions: {len(self.portfolio)} stocks")

                return True

            finally:
                connection.disconnect()

        except Exception as e:
            logger.warning(f"Failed to get live account info: {e}")
            return False

    def find_signals(self, lookback_days: int = 2) -> list[BollingerSignal]:
        """
        Find Bollinger band crossover signals on recent data.

        Args:
            lookback_days: Number of trading days to look back (default: 2)
                          Uses actual trading days (rows), not calendar days,
                          so weekends/holidays are handled automatically.

        Returns:
            List of BollingerSignal objects
        """
        if self.data is None:
            logger.error("No data loaded")
            return []

        self.signals = []

        # Get the most recent date in data
        max_date = self.data['date'].max()

        logger.info(f"Scanning for signals on most recent {lookback_days} trading day(s) ending {max_date.date()}")

        # Get unique symbols
        symbols = self.data['symbol'].unique()

        for symbol in symbols:
            symbol_data = self.data[self.data['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date')

            # Get the most recent row
            if len(symbol_data) < 2:
                continue

            # Use trading days (rows) not calendar days - handles weekends/holidays
            # Get the last N trading days for this symbol
            recent_data = symbol_data.tail(lookback_days + 1)  # +1 to have prev day for comparison

            if len(recent_data) < 2:
                continue

            latest = recent_data.iloc[-1]
            prev = recent_data.iloc[-2]

            # Skip if this symbol's latest date is not the overall max date
            # (symbol may have stopped trading)
            if latest['date'] != max_date:
                continue

            close = latest['adjusted_close']
            bb_lower = latest['bbands_lower_20']
            bb_upper = latest['bbands_upper_20']
            bb_middle = latest.get('bbands_middle_20', (bb_lower + bb_upper) / 2)

            prev_close = prev['adjusted_close']
            prev_bb_lower = prev['bbands_lower_20']
            prev_bb_upper = prev['bbands_upper_20']

            # Skip if missing BB data
            if pd.isna(bb_lower) or pd.isna(bb_upper):
                continue

            # Get additional context data
            volume = latest.get('volume', 0)
            rsi = latest.get('rsi_14', None)
            if pd.isna(rsi):
                rsi = None

            bull_bear_delta = latest.get('bull_bear_delta', None)
            if pd.isna(bull_bear_delta):
                bull_bear_delta = None

            # Calculate band width as % of middle
            bb_width_pct = ((bb_upper - bb_lower) / bb_middle * 100) if bb_middle > 0 else 0

            signal = None

            # Check for LOWER band crossover (BUY signal)
            # Price crossed below lower band AND RSI <= 40 (not overbought)
            if close <= bb_lower and prev_close > prev_bb_lower:
                # RSI filter: for BUY, RSI should not be > 40
                if rsi is not None and rsi > 40:
                    continue  # Skip - RSI too high for a buy signal

                distance_pct = ((bb_lower - close) / close * 100) if close > 0 else 0
                signal = BollingerSignal(
                    symbol=symbol,
                    signal_type="BUY",
                    close_price=close,
                    bb_lower=bb_lower,
                    bb_upper=bb_upper,
                    bb_middle=bb_middle,
                    signal_date=str(latest['date'].date()),
                    reason=f"Price crossed below lower BB (RSI: {rsi:.1f})" if rsi else "Price crossed below lower BB",
                    bb_width_pct=bb_width_pct,
                    distance_from_band_pct=distance_pct,
                    volume=volume,
                    rsi=rsi,
                    prev_close=prev_close,
                    bull_bear_delta=bull_bear_delta
                )

            # Check for UPPER band crossover (SELL signal)
            # Price crossed above upper band AND RSI >= 60 (not oversold)
            elif close >= bb_upper and prev_close < prev_bb_upper:
                # RSI filter: for SELL, RSI should not be < 60
                if rsi is not None and rsi < 60:
                    continue  # Skip - RSI too low for a sell signal

                distance_pct = ((close - bb_upper) / close * 100) if close > 0 else 0
                signal = BollingerSignal(
                    symbol=symbol,
                    signal_type="SELL",
                    close_price=close,
                    bb_lower=bb_lower,
                    bb_upper=bb_upper,
                    bb_middle=bb_middle,
                    signal_date=str(latest['date'].date()),
                    reason=f"Price crossed above upper BB (RSI: {rsi:.1f})" if rsi else "Price crossed above upper BB",
                    bb_width_pct=bb_width_pct,
                    distance_from_band_pct=distance_pct,
                    volume=volume,
                    rsi=rsi,
                    prev_close=prev_close,
                    bull_bear_delta=bull_bear_delta
                )

            if signal:
                self._validate_signal(signal)
                self.signals.append(signal)

        logger.info(f"Found {len(self.signals)} signals")

        # Apply watchlist filtering/sorting if configured
        if self.watchlist:
            self.signals = self._apply_watchlist(self.signals)

        return self.signals

    def _apply_watchlist(self, signals: list[BollingerSignal]) -> list[BollingerSignal]:
        """
        Apply watchlist to signals list.

        Args:
            signals: List of BollingerSignal objects

        Returns:
            Filtered or sorted list of signals
        """
        if not self.watchlist:
            return signals

        watchlist_set = set(self.watchlist)
        # Create order mapping: symbol -> position in watchlist
        watchlist_order = {sym: i for i, sym in enumerate(self.watchlist)}

        if self.watchlist_mode == 'filter':
            # Only keep signals for symbols in watchlist
            filtered = [s for s in signals if s.symbol in watchlist_set]
            # Sort by watchlist order
            filtered.sort(key=lambda s: watchlist_order.get(s.symbol, len(self.watchlist)))
            logger.info(f"Watchlist filter: {len(signals)} signals -> {len(filtered)} signals")
            return filtered
        else:  # sort mode
            # Sort signals: watchlist symbols first (in order), then others alphabetically
            def sort_key(s):
                if s.symbol in watchlist_order:
                    return (0, watchlist_order[s.symbol])
                return (1, s.symbol)
            sorted_signals = sorted(signals, key=sort_key)
            in_watchlist = sum(1 for s in signals if s.symbol in watchlist_set)
            logger.info(f"Watchlist sort: {in_watchlist} watchlist signals first, {len(signals) - in_watchlist} others")
            return sorted_signals

    def _validate_signal(self, signal: BollingerSignal) -> None:
        """Validate a signal against account constraints."""

        if signal.signal_type == "BUY":
            # Calculate proposed shares
            if self.cash_balance is not None:
                available_cash = self.cash_balance - self.MIN_CASH_RESERVE
                max_position_value = available_cash * self.position_size_pct
                signal.shares_proposed = max(1, int(max_position_value / signal.close_price))
                signal.estimated_cost = signal.shares_proposed * signal.close_price

                # Check if we have enough cash
                if available_cash < signal.estimated_cost:
                    signal.can_execute = False
                    signal.block_reason = f"Insufficient cash: ${available_cash:.2f} available, need ${signal.estimated_cost:.2f}"
            else:
                # No live data - use default shares
                signal.shares_proposed = self.shares_per_trade
                signal.estimated_cost = signal.shares_proposed * signal.close_price
                signal.block_reason = "No live account data - using default shares"

        elif signal.signal_type == "SELL":
            # Check if we hold this stock
            shares_held = self.portfolio.get(signal.symbol, 0)

            if shares_held > 0:
                signal.shares_proposed = shares_held
                signal.estimated_cost = shares_held * signal.close_price  # Proceeds
            else:
                if self.portfolio:  # We have portfolio data
                    signal.can_execute = False
                    signal.block_reason = f"No position in {signal.symbol} to sell"
                    signal.shares_proposed = 0
                else:
                    # No live data - use default
                    signal.shares_proposed = self.shares_per_trade
                    signal.estimated_cost = signal.shares_proposed * signal.close_price
                    signal.block_reason = "No live account data - cannot verify position"

    def send_summary_notification(self) -> bool:
        """
        Send a single summary email with all signals in table format.

        Returns:
            True if sent successfully
        """
        if not self.notifier:
            logger.debug("Notifications disabled")
            return False

        if not self.signals:
            logger.info("No signals to notify")
            return False

        summary = self.generate_summary()
        subject = f"[SHADOW] Bollinger Signals Summary - {len(self.signals)} signal(s) ({datetime.now().strftime('%Y-%m-%d')})"

        if self.notifier._send_email(subject, summary):
            logger.info(f"Summary notification sent with {len(self.signals)} signal(s)")
            return True
        else:
            logger.warning("Failed to send summary notification")
            return False

    def send_notifications(self) -> int:
        """
        Send individual notification emails for each signal.

        Returns:
            Number of notifications sent
        """
        if not self.notifier:
            logger.debug("Notifications disabled")
            return 0

        if not self.signals:
            logger.info("No signals to notify")
            return 0

        sent = 0

        for signal in self.signals:
            subject = f"[SHADOW] {signal.signal_type} Signal: {signal.symbol} ({signal.signal_date})"

            # Price change from previous day
            price_change = signal.close_price - signal.prev_close
            price_change_pct = (price_change / signal.prev_close * 100) if signal.prev_close > 0 else 0

            body_lines = [
                "=" * 55,
                "BOLLINGER BAND SHADOW SIGNAL",
                "=" * 55,
                "",
                f"Signal Type:     {signal.signal_type}",
                f"Symbol:          {signal.symbol}",
                f"Signal Date:     {signal.signal_date}",
                f"Generated:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "-" * 55,
                "PRICE DATA:",
                "-" * 55,
                f"  Close Price:   ${signal.close_price:.2f}",
                f"  Prev Close:    ${signal.prev_close:.2f}",
                f"  Day Change:    ${price_change:+.2f} ({price_change_pct:+.2f}%)",
                f"  Volume:        {signal.volume:,.0f}",
            ]

            if signal.rsi is not None:
                rsi_status = "OVERSOLD" if signal.rsi < 30 else "OVERBOUGHT" if signal.rsi > 70 else "NEUTRAL"
                body_lines.append(f"  RSI (14):      {signal.rsi:.1f} ({rsi_status})")

            if signal.bull_bear_delta is not None:
                bbd_status = "BULLISH" if signal.bull_bear_delta > 0 else "BEARISH" if signal.bull_bear_delta < 0 else "NEUTRAL"
                body_lines.append(f"  Bull/Bear:     {int(signal.bull_bear_delta):+d} ({bbd_status})")

            body_lines.extend([
                "",
                "-" * 55,
                "BOLLINGER BANDS (20-day):",
                "-" * 55,
                f"  Upper:         ${signal.bb_upper:.2f}",
                f"  Middle:        ${signal.bb_middle:.2f}",
                f"  Lower:         ${signal.bb_lower:.2f}",
                f"  Band Width:    {signal.bb_width_pct:.1f}%",
                f"  Distance:      {signal.distance_from_band_pct:.2f}% past band",
                "",
                f"Trigger:         {signal.reason}",
                "",
                "-" * 55,
                "PROPOSED ORDER:",
                "-" * 55,
                f"  Action:        {signal.signal_type} {signal.shares_proposed} shares",
                f"  Est. Value:    ${signal.estimated_cost:,.2f}",
                f"  Can Execute:   {'YES' if signal.can_execute else 'NO'}",
            ])

            if signal.block_reason:
                body_lines.append(f"  Note:          {signal.block_reason}")

            if self.cash_balance is not None:
                body_lines.extend([
                    "",
                    "-" * 55,
                    "ACCOUNT STATUS:",
                    "-" * 55,
                    f"  Account:       {self.account_id}",
                    f"  Cash Balance:  ${self.cash_balance:,.2f}",
                    f"  Positions:     {len(self.portfolio)} stocks",
                ])

                if signal.symbol in self.portfolio:
                    body_lines.append(f"  {signal.symbol} held:    {self.portfolio[signal.symbol]} shares")

            body_lines.extend([
                "",
                "=" * 55,
                "THIS IS A SHADOW SIGNAL - NO TRADE EXECUTED",
                "To execute: --execute " + signal.symbol,
                "=" * 55,
            ])

            body = "\n".join(body_lines)

            # Send via notifier
            if self.notifier._send_email(subject, body):
                sent += 1
                logger.info(f"Notification sent: {signal.signal_type} {signal.symbol}")
            else:
                logger.warning(f"Failed to send notification for {signal.symbol}")

        return sent

    def generate_summary(self) -> str:
        """Generate a summary of all signals."""
        if not self.signals:
            return "No Bollinger band crossover signals found."

        # Get data date range
        data_dates = ""
        if self.data is not None:
            min_date = self.data['date'].min()
            max_date = self.data['date'].max()
            data_dates = f"Data Range: {min_date.date()} to {max_date.date()}"

        lines = [
            "=" * 85,
            "BOLLINGER BAND SHADOW STRATEGY SUMMARY",
            "=" * 85,
            f"Scan Time:     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"{data_dates}",
            f"Total Signals: {len(self.signals)}",
            "",
        ]

        buy_signals = [s for s in self.signals if s.signal_type == "BUY"]
        sell_signals = [s for s in self.signals if s.signal_type == "SELL"]

        lines.append(f"BUY Signals ({len(buy_signals)}):")
        lines.append("-" * 85)
        if buy_signals:
            lines.append(f"{'Symbol':<8} {'Date':<12} {'Close':>10} {'BB Lower':>10} {'Dist%':>7} {'RSI':>6} {'BullBear':>8} {'Status':<10}")
            lines.append("-" * 85)
            for s in buy_signals:
                status = "OK" if s.can_execute else "BLOCKED"
                rsi_str = f"{s.rsi:.1f}" if s.rsi is not None else "N/A"
                bbd_str = f"{int(s.bull_bear_delta):+d}" if s.bull_bear_delta is not None else "N/A"
                lines.append(f"{s.symbol:<8} {s.signal_date:<12} ${s.close_price:>8.2f} ${s.bb_lower:>8.2f} {s.distance_from_band_pct:>6.2f}% {rsi_str:>6} {bbd_str:>8} {status:<10}")
        else:
            lines.append("  (none)")

        lines.append("")
        lines.append(f"SELL Signals ({len(sell_signals)}):")
        lines.append("-" * 85)
        if sell_signals:
            lines.append(f"{'Symbol':<8} {'Date':<12} {'Close':>10} {'BB Upper':>10} {'Dist%':>7} {'RSI':>6} {'BullBear':>8} {'Status':<10}")
            lines.append("-" * 85)
            for s in sell_signals:
                status = "OK" if s.can_execute else "BLOCKED"
                rsi_str = f"{s.rsi:.1f}" if s.rsi is not None else "N/A"
                bbd_str = f"{int(s.bull_bear_delta):+d}" if s.bull_bear_delta is not None else "N/A"
                lines.append(f"{s.symbol:<8} {s.signal_date:<12} ${s.close_price:>8.2f} ${s.bb_upper:>8.2f} {s.distance_from_band_pct:>6.2f}% {rsi_str:>6} {bbd_str:>8} {status:<10}")
        else:
            lines.append("  (none)")

        if self.cash_balance is not None:
            lines.extend([
                "",
                "-" * 85,
                "ACCOUNT INFO:",
                f"  Account:   {self.account_id}",
                f"  Cash:      ${self.cash_balance:,.2f}",
                f"  Positions: {len(self.portfolio)} stocks",
            ])

        lines.append("=" * 85)

        return "\n".join(lines)

    def execute_trades(
        self,
        symbols_to_execute: list[str],
        host: str = "127.0.0.1",
        port: int = 4001
    ) -> dict:
        """
        Execute trades for specified symbols that have matching signals.

        Args:
            symbols_to_execute: List of symbols to execute (must have matching signals)
            host: Gateway host
            port: Gateway port

        Returns:
            Dict with execution results
        """
        results = {
            "executed": [],
            "skipped": [],
            "failed": []
        }

        # Find signals for the requested symbols
        signals_to_execute = [
            s for s in self.signals
            if s.symbol.upper() in [sym.upper() for sym in symbols_to_execute]
        ]

        if not signals_to_execute:
            logger.warning(f"No matching signals found for symbols: {symbols_to_execute}")
            results["skipped"] = symbols_to_execute
            return results

        # Warn about symbols requested but not having signals
        requested_upper = {sym.upper() for sym in symbols_to_execute}
        signal_symbols = {s.symbol.upper() for s in signals_to_execute}
        no_signal = requested_upper - signal_symbols
        if no_signal:
            logger.warning(f"No signals found for: {no_signal}")
            results["skipped"].extend(list(no_signal))

        # Check which signals can be executed
        executable_signals = [s for s in signals_to_execute if s.can_execute]
        blocked_signals = [s for s in signals_to_execute if not s.can_execute]

        for s in blocked_signals:
            logger.warning(f"Cannot execute {s.symbol}: {s.block_reason}")
            results["skipped"].append({"symbol": s.symbol, "reason": s.block_reason})

        if not executable_signals:
            logger.warning("No executable signals found")
            return results

        # Initialize trade executor
        logger.info(f"Executing {len(executable_signals)} trade(s)...")

        ibkr_config = IBKRConfig.remote_gateway(host=host, port=port, client_id=100)
        trading_config = TradingConfig(
            max_position_pct=self.position_size_pct,
            use_market_orders=True  # Use market orders for simplicity
        )

        try:
            with TradeExecutor(ibkr_config, trading_config, enable_notifications=True) as executor:
                for signal in executable_signals:
                    logger.info(f"Executing: {signal.signal_type} {signal.shares_proposed} {signal.symbol}")

                    try:
                        if signal.signal_type == "BUY":
                            result = executor.execute_buy(
                                symbol=signal.symbol,
                                shares=signal.shares_proposed,
                                reason=f"Bollinger band crossover: {signal.reason}"
                            )
                        else:  # SELL
                            result = executor.execute_sell(
                                symbol=signal.symbol,
                                shares=signal.shares_proposed,
                                reason=f"Bollinger band crossover: {signal.reason}"
                            )

                        if result.success:
                            logger.info(f"SUCCESS: {signal.symbol} - filled {result.order.filled_shares} shares")
                            results["executed"].append({
                                "symbol": signal.symbol,
                                "action": signal.signal_type,
                                "shares": result.order.filled_shares,
                                "price": result.order.avg_fill_price
                            })
                        else:
                            logger.error(f"FAILED: {signal.symbol} - {result.error_message}")
                            results["failed"].append({
                                "symbol": signal.symbol,
                                "error": result.error_message
                            })

                    except Exception as e:
                        logger.error(f"Exception executing {signal.symbol}: {e}")
                        results["failed"].append({"symbol": signal.symbol, "error": str(e)})

        except Exception as e:
            logger.error(f"Failed to initialize trade executor: {e}")
            for signal in executable_signals:
                results["failed"].append({"symbol": signal.symbol, "error": str(e)})

        return results

    def run(
        self,
        skip_live: bool = False,
        host: str = "127.0.0.1",
        port: int = 4001,
        execute_symbols: Optional[list[str]] = None,
        signal_filter: str = "all",
        summary_only: bool = False
    ) -> list[BollingerSignal]:
        """
        Run the full shadow strategy.

        Args:
            skip_live: If True, skip connecting to Gateway (for testing)
            host: Gateway host (default: localhost via SSH tunnel)
            port: Gateway port (default: 4001 for live)
            execute_symbols: List of symbols to actually execute trades for (optional)
            signal_filter: Filter signals - "all", "buy", or "sell"
            summary_only: If True, send one summary email instead of individual emails

        Returns:
            List of signals found
        """
        logger.info("Starting Bollinger Shadow Strategy")

        # Load data
        if not self.load_data():
            return []

        # Always get live account info unless explicitly skipped
        if not skip_live:
            if not self.get_live_account_info(host=host, port=port):
                logger.error("Failed to connect to IB Gateway - cannot proceed without live data")
                logger.error("Use --skip-live to run without Gateway (for testing only)")
                return []
        else:
            logger.warning("Skipping live account check - using default values")

        # Find signals
        signals = self.find_signals()

        # Apply signal type filter
        if signal_filter == "buy":
            self.signals = [s for s in self.signals if s.signal_type == "BUY"]
            signals = self.signals
            logger.info(f"Filtered to BUY signals only: {len(signals)} signal(s)")
        elif signal_filter == "sell":
            self.signals = [s for s in self.signals if s.signal_type == "SELL"]
            signals = self.signals
            logger.info(f"Filtered to SELL signals only: {len(signals)} signal(s)")

        # Print summary
        print(self.generate_summary())

        # Send notifications
        if signals:
            if summary_only:
                self.send_summary_notification()
            else:
                sent = self.send_notifications()
                logger.info(f"Sent {sent} notification(s)")

        # Execute trades if symbols specified
        if execute_symbols and signals:
            print("\n" + "=" * 60)
            print("EXECUTING TRADES FOR SPECIFIED SYMBOLS")
            print("=" * 60)

            execution_results = self.execute_trades(execute_symbols, host=host, port=port)

            # Print execution summary
            print(f"\nExecution Results:")
            print(f"  Executed: {len(execution_results['executed'])}")
            for ex in execution_results['executed']:
                print(f"    {ex['action']} {ex['shares']} {ex['symbol']} @ ${ex['price']:.2f}")

            print(f"  Skipped: {len(execution_results['skipped'])}")
            for sk in execution_results['skipped']:
                if isinstance(sk, dict):
                    print(f"    {sk['symbol']}: {sk['reason']}")
                else:
                    print(f"    {sk}: No matching signal")

            print(f"  Failed: {len(execution_results['failed'])}")
            for fl in execution_results['failed']:
                print(f"    {fl['symbol']}: {fl['error']}")

        return signals


def main():
    parser = argparse.ArgumentParser(
        description="Bollinger Band Shadow Strategy - scans for BB crossovers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to all_data CSV file(s), supports glob patterns"
    )
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="Skip connecting to IB Gateway (for testing without tunnel)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Gateway host (default: 127.0.0.1 via SSH tunnel)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4001,
        help="Gateway port (default: 4001 for live)"
    )
    parser.add_argument(
        "--position-pct",
        type=float,
        default=0.05,
        help="Position size as pct of cash (default: 0.05 = 5%%)"
    )
    parser.add_argument(
        "--shares",
        type=int,
        default=10,
        help="Default shares per trade (default: 10)"
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=2,
        help="Days to look back for signals (default: 2)"
    )
    parser.add_argument(
        "--signal-type",
        type=str,
        choices=["all", "buy", "sell"],
        default="all",
        help="Filter signals: 'buy' only, 'sell' only, or 'all' (default: all)"
    )
    parser.add_argument(
        "--no-notify",
        action="store_true",
        help="Skip sending notifications"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Send one summary email with table instead of individual signal emails"
    )
    parser.add_argument(
        "--execute",
        type=str,
        default=None,
        help="Explicitly specify symbols to execute trades for (e.g., --execute AAPL or --execute AAPL,MSFT). "
             "Each symbol must have a matching signal. No 'execute all' option - you must list each symbol."
    )
    parser.add_argument(
        "--watchlist",
        type=str,
        default=None,
        help="Path to watchlist CSV file (optional). Used to filter or sort output signals."
    )
    parser.add_argument(
        "--watchlist-mode",
        type=str,
        default='sort',
        choices=['sort', 'filter'],
        help="Watchlist mode: 'sort' (default) keeps all signals but orders by watchlist, "
             "'filter' only keeps signals for symbols in watchlist"
    )

    args = parser.parse_args()

    # Parse execute symbols
    execute_symbols = None
    if args.execute:
        execute_symbols = [s.strip().upper() for s in args.execute.split(",") if s.strip()]
        logger.info(f"Will execute trades for symbols: {execute_symbols}")

    # Load watchlist if provided
    watchlist = None
    if args.watchlist:
        watchlist = load_watchlist(args.watchlist)
        if watchlist:
            logger.info(f"Loaded watchlist with {len(watchlist)} symbols")
        else:
            logger.warning("Watchlist is empty or could not be loaded")

    # Create strategy
    notifier = None if args.no_notify else TradeNotifier()

    strategy = BollingerShadowStrategy(
        data_path=args.data_path,
        notifier=notifier,
        position_size_pct=args.position_pct,
        shares_per_trade=args.shares,
        watchlist=watchlist,
        watchlist_mode=args.watchlist_mode
    )

    # Run - always connect to Gateway unless --skip-live is set
    signals = strategy.run(
        skip_live=args.skip_live,
        host=args.host,
        port=args.port,
        execute_symbols=execute_symbols,
        signal_filter=args.signal_type,
        summary_only=args.summary_only
    )

    if not signals:
        print("\nNo signals found.")
        sys.exit(0)
    else:
        print(f"\nFound {len(signals)} signal(s)")
        sys.exit(0)


if __name__ == "__main__":
    main()
