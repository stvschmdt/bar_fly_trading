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

    # Run strategy (connects to Gateway, sends emails)
    python bollinger_shadow_strategy.py --data-path ~/proj/bar_fly_trading/all_data.csv

    # Test without Gateway connection (for debugging data/signals only)
    python bollinger_shadow_strategy.py --data-path ~/data/all_data.csv --skip-live
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
    from ibkr.config import IBKRConfig
    from ibkr.connection import IBKRConnection
    from ibkr.notifier import TradeNotifier
except ModuleNotFoundError:
    from config import IBKRConfig
    from connection import IBKRConnection
    from notifier import TradeNotifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        shares_per_trade: int = 10
    ):
        """
        Initialize the shadow strategy.

        Args:
            data_path: Path to all_data CSV file(s), supports glob patterns
            notifier: Optional notifier for sending alerts
            position_size_pct: Percentage of cash to use per position
            shares_per_trade: Default shares per trade if not calculated
        """
        self.data_path = data_path
        self.notifier = notifier or TradeNotifier()
        self.position_size_pct = position_size_pct
        self.shares_per_trade = shares_per_trade

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
            lookback_days: How many days back to check (default: 2 for today/yesterday)

        Returns:
            List of BollingerSignal objects
        """
        if self.data is None:
            logger.error("No data loaded")
            return []

        self.signals = []

        # Get the most recent date in data
        max_date = self.data['date'].max()
        cutoff_date = max_date - timedelta(days=lookback_days)

        logger.info(f"Scanning for signals from {cutoff_date.date()} to {max_date.date()}")

        # Get unique symbols
        symbols = self.data['symbol'].unique()

        for symbol in symbols:
            symbol_data = self.data[self.data['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('date')

            # Get the most recent row
            if len(symbol_data) < 2:
                continue

            latest = symbol_data.iloc[-1]
            prev = symbol_data.iloc[-2]

            # Skip if not recent enough
            if latest['date'] < cutoff_date:
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

            signal = None

            # Check for LOWER band crossover (BUY signal)
            # Price crossed below lower band
            if close <= bb_lower and prev_close > prev_bb_lower:
                signal = BollingerSignal(
                    symbol=symbol,
                    signal_type="BUY",
                    close_price=close,
                    bb_lower=bb_lower,
                    bb_upper=bb_upper,
                    bb_middle=bb_middle,
                    signal_date=str(latest['date'].date()),
                    reason=f"Price ({close:.2f}) crossed below lower BB ({bb_lower:.2f})"
                )

            # Check for UPPER band crossover (SELL signal)
            # Price crossed above upper band
            elif close >= bb_upper and prev_close < prev_bb_upper:
                signal = BollingerSignal(
                    symbol=symbol,
                    signal_type="SELL",
                    close_price=close,
                    bb_lower=bb_lower,
                    bb_upper=bb_upper,
                    bb_middle=bb_middle,
                    signal_date=str(latest['date'].date()),
                    reason=f"Price ({close:.2f}) crossed above upper BB ({bb_upper:.2f})"
                )

            if signal:
                self._validate_signal(signal)
                self.signals.append(signal)

        logger.info(f"Found {len(self.signals)} signals")
        return self.signals

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

    def send_notifications(self) -> int:
        """
        Send notification emails for all signals.

        Returns:
            Number of notifications sent
        """
        if not self.signals:
            logger.info("No signals to notify")
            return 0

        sent = 0

        for signal in self.signals:
            subject = f"[SHADOW] {signal.signal_type} Signal: {signal.symbol}"

            body_lines = [
                "=" * 50,
                "BOLLINGER BAND SHADOW SIGNAL",
                "=" * 50,
                "",
                f"Signal Type:     {signal.signal_type}",
                f"Symbol:          {signal.symbol}",
                f"Signal Date:     {signal.signal_date}",
                f"Close Price:     ${signal.close_price:.2f}",
                "",
                "Bollinger Bands (20-day):",
                f"  Upper:         ${signal.bb_upper:.2f}",
                f"  Middle:        ${signal.bb_middle:.2f}",
                f"  Lower:         ${signal.bb_lower:.2f}",
                "",
                f"Reason:          {signal.reason}",
                "",
                "-" * 50,
                "PROPOSED ORDER:",
                "-" * 50,
                f"  Action:        {signal.signal_type} {signal.shares_proposed} shares",
                f"  Est. Value:    ${signal.estimated_cost:,.2f}",
                f"  Can Execute:   {'YES' if signal.can_execute else 'NO'}",
            ]

            if signal.block_reason:
                body_lines.append(f"  Note:          {signal.block_reason}")

            if self.cash_balance is not None:
                body_lines.extend([
                    "",
                    "-" * 50,
                    "ACCOUNT STATUS:",
                    "-" * 50,
                    f"  Account:       {self.account_id}",
                    f"  Cash Balance:  ${self.cash_balance:,.2f}",
                    f"  Positions:     {len(self.portfolio)} stocks",
                ])

                if signal.symbol in self.portfolio:
                    body_lines.append(f"  {signal.symbol} held:    {self.portfolio[signal.symbol]} shares")

            body_lines.extend([
                "",
                "=" * 50,
                "THIS IS A SHADOW SIGNAL - NO TRADE EXECUTED",
                "=" * 50,
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

        lines = [
            "=" * 60,
            "BOLLINGER BAND SHADOW STRATEGY SUMMARY",
            "=" * 60,
            f"Scan Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Signals: {len(self.signals)}",
            "",
        ]

        buy_signals = [s for s in self.signals if s.signal_type == "BUY"]
        sell_signals = [s for s in self.signals if s.signal_type == "SELL"]

        lines.append(f"BUY Signals: {len(buy_signals)}")
        for s in buy_signals:
            status = "OK" if s.can_execute else "BLOCKED"
            lines.append(f"  {s.symbol}: ${s.close_price:.2f} (BB lower: ${s.bb_lower:.2f}) [{status}]")

        lines.append("")
        lines.append(f"SELL Signals: {len(sell_signals)}")
        for s in sell_signals:
            status = "OK" if s.can_execute else "BLOCKED"
            lines.append(f"  {s.symbol}: ${s.close_price:.2f} (BB upper: ${s.bb_upper:.2f}) [{status}]")

        if self.cash_balance is not None:
            lines.extend([
                "",
                "-" * 60,
                "ACCOUNT INFO:",
                f"  Cash: ${self.cash_balance:,.2f}",
                f"  Positions: {len(self.portfolio)}",
            ])

        lines.append("=" * 60)

        return "\n".join(lines)

    def run(
        self,
        skip_live: bool = False,
        host: str = "127.0.0.1",
        port: int = 4001
    ) -> list[BollingerSignal]:
        """
        Run the full shadow strategy.

        Args:
            skip_live: If True, skip connecting to Gateway (for testing)
            host: Gateway host (default: localhost via SSH tunnel)
            port: Gateway port (default: 4001 for live)

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

        # Print summary
        print(self.generate_summary())

        # Send notifications
        if signals:
            sent = self.send_notifications()
            logger.info(f"Sent {sent} notification(s)")

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
        "--no-notify",
        action="store_true",
        help="Skip sending notifications"
    )

    args = parser.parse_args()

    # Create strategy
    notifier = None if args.no_notify else TradeNotifier()

    strategy = BollingerShadowStrategy(
        data_path=args.data_path,
        notifier=notifier,
        position_size_pct=args.position_pct,
        shares_per_trade=args.shares
    )

    # Run - always connect to Gateway unless --skip-live is set
    signals = strategy.run(skip_live=args.skip_live, host=args.host, port=args.port)

    if not signals:
        print("\nNo signals found.")
        sys.exit(0)
    else:
        print(f"\nFound {len(signals)} signal(s)")
        sys.exit(0)


if __name__ == "__main__":
    main()
