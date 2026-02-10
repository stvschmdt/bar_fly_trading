"""
Exit monitor for live IBKR positions.

Periodically checks the position ledger and manages exit orders:
  1. Sync: detect bracket orders that already filled, clean up ledger
  2. Max hold: cancel brackets + market sell for expired positions
  3. Trailing stop: adjust stop order price upward as price rises

Usage:
    python -m ibkr.exit_monitor --gateway                  # paper (default)
    python -m ibkr.exit_monitor --gateway --live           # live trading
    python -m ibkr.exit_monitor --gateway --dry-run        # preview only
"""

import argparse
import logging
import os
import sys
from datetime import date, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibkr.config import IBKRConfig, TradingConfig
from ibkr.connection import IBKRConnection
from ibkr.order_manager import OrderManager
from ibkr.position_ledger import PositionLedger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class ExitMonitor:
    """
    Monitor live positions and manage exit orders.

    Responsibilities:
    1. Sync ledger with IBKR: remove entries where bracket already filled
    2. Max hold day exits: cancel brackets + market sell
    3. Trailing stop adjustments: modify stop order price upward
    """

    def __init__(self, connection: IBKRConnection, order_manager: OrderManager,
                 ledger: PositionLedger, dry_run: bool = False):
        self.connection = connection
        self.order_manager = order_manager
        self.ledger = ledger
        self.dry_run = dry_run
        self.actions: list[dict] = []

    def run_check(self) -> list[dict]:
        """Run one cycle of exit monitoring. Returns list of action dicts."""
        self.ledger.load()
        self.actions = []

        if len(self.ledger) == 0:
            logger.info("Exit monitor: no positions in ledger")
            return self.actions

        logger.info(f"Exit monitor: checking {len(self.ledger)} position(s)")

        # Step 1: Sync with IBKR — detect filled brackets
        self._sync_with_ibkr()

        # Step 2: Check max hold day exits
        self._check_max_hold_exits()

        # Step 3: Adjust trailing stops
        self._adjust_trailing_stops()

        # Save any changes
        self.ledger.save()

        return self.actions

    def _sync_with_ibkr(self):
        """Compare ledger vs IBKR positions. Remove closed positions."""
        ib = self.connection.ib
        ibkr_positions = {}
        for item in ib.positions():
            sym = item.contract.symbol
            shares = int(item.position)
            if shares != 0 and item.contract.secType == 'STK':
                ibkr_positions[sym] = shares

        for symbol in list(self.ledger.get_all_positions().keys()):
            if symbol not in ibkr_positions:
                pos = self.ledger.get_position(symbol)
                logger.info(f"  SYNC: {symbol} no longer in IBKR positions — "
                           f"bracket likely filled, removing from ledger")
                self.ledger.remove_position(symbol, reason='bracket_filled')
                self.actions.append({
                    'action': 'sync_remove',
                    'symbol': symbol,
                    'reason': 'position closed in IBKR (bracket filled)',
                    'strategy': pos.get('strategy', '') if pos else '',
                })

    def _check_max_hold_exits(self):
        """Cancel brackets and market sell for positions past max hold."""
        expired = self.ledger.get_expired_positions()

        for pos in expired:
            symbol = pos['symbol']
            hold_days = pos['hold_days']
            max_days = pos.get('max_hold_days', 0)
            shares = pos.get('shares', 0)

            logger.info(f"  MAX HOLD: {symbol} held {hold_days}d (max={max_days}d) — "
                       f"closing {shares} shares")

            if self.dry_run:
                self.actions.append({
                    'action': 'max_hold_exit',
                    'symbol': symbol,
                    'hold_days': hold_days,
                    'shares': shares,
                    'dry_run': True,
                })
                continue

            # Cancel existing bracket orders
            stop_id = pos.get('stop_order_id', -1)
            profit_id = pos.get('profit_order_id', -1)
            if stop_id > 0 or profit_id > 0:
                self.order_manager.cancel_bracket_orders(stop_id, profit_id)

            # Submit market sell
            from ib_insync import Stock, MarketOrder
            contract = Stock(symbol, "SMART", "USD")
            try:
                self.connection.ib.qualifyContracts(contract)
                order = MarketOrder("SELL", shares)
                order.tif = "DAY"
                acct = self.order_manager.target_account
                if acct:
                    order.account = acct
                trade = self.connection.ib.placeOrder(contract, order)
                logger.info(f"  Submitted market sell for {symbol}: {shares} shares "
                           f"(order_id={trade.order.orderId})")
            except Exception as e:
                logger.error(f"  Failed to submit sell for {symbol}: {e}")

            self.ledger.remove_position(symbol, reason='max_hold')
            self.actions.append({
                'action': 'max_hold_exit',
                'symbol': symbol,
                'hold_days': hold_days,
                'shares': shares,
                'dry_run': False,
            })

    def _adjust_trailing_stops(self):
        """Adjust stop order prices upward for trailing stop positions."""
        trailing_positions = self.ledger.get_trailing_stop_positions()

        for pos in trailing_positions:
            symbol = pos['symbol']
            trailing_pct = pos['trailing_stop_pct']
            old_hwm = pos.get('high_water_mark', pos['entry_price'])
            stop_id = pos.get('stop_order_id', -1)

            # Get current price
            try:
                current_price = self.connection.get_current_price(symbol)
            except Exception as e:
                logger.warning(f"  Could not get price for {symbol}: {e}")
                continue

            if current_price <= 0:
                continue

            # Update high-water mark if price is higher
            if current_price > old_hwm:
                new_hwm = current_price
                self.ledger.update_high_water_mark(symbol, new_hwm)

                # Calculate new stop price from new HWM
                new_stop_price = round(new_hwm * (1 + trailing_pct), 2)
                old_stop_price = round(old_hwm * (1 + trailing_pct), 2)

                if new_stop_price > old_stop_price and stop_id > 0:
                    logger.info(
                        f"  TRAILING: {symbol} HWM ${old_hwm:.2f} → ${new_hwm:.2f}, "
                        f"stop ${old_stop_price:.2f} → ${new_stop_price:.2f}")

                    if not self.dry_run:
                        self.order_manager.modify_stop_price(stop_id, new_stop_price)

                    self.actions.append({
                        'action': 'trailing_stop_adjust',
                        'symbol': symbol,
                        'old_hwm': old_hwm,
                        'new_hwm': new_hwm,
                        'old_stop': old_stop_price,
                        'new_stop': new_stop_price,
                        'dry_run': self.dry_run,
                    })

    def get_summary(self) -> str:
        """Format a summary of actions taken."""
        if not self.actions:
            return "Exit monitor: no actions taken"

        lines = [f"Exit monitor: {len(self.actions)} action(s)"]
        for a in self.actions:
            action = a['action']
            symbol = a.get('symbol', '?')
            dry = " [DRY RUN]" if a.get('dry_run') else ""
            if action == 'sync_remove':
                lines.append(f"  SYNC {symbol}: removed (bracket filled)")
            elif action == 'max_hold_exit':
                lines.append(f"  EXIT {symbol}: max hold {a.get('hold_days')}d{dry}")
            elif action == 'trailing_stop_adjust':
                lines.append(f"  TRAIL {symbol}: stop ${a.get('old_stop'):.2f} → "
                           f"${a.get('new_stop'):.2f}{dry}")
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Monitor positions and manage exits",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ibkr.exit_monitor --gateway                  # paper (default)
  python -m ibkr.exit_monitor --gateway --live           # live trading
  python -m ibkr.exit_monitor --gateway --dry-run        # preview only
        """)

    parser.add_argument("--gateway", action="store_true", default=True,
                        help="Connect via IB Gateway (default: True)")
    parser.add_argument("--live", action="store_true",
                        help="Use live trading account")
    parser.add_argument("--client-id", type=int, default=21,
                        help="IBKR client ID (default: 21, separate from executor)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview actions without executing")
    parser.add_argument("--ledger", type=str, default=None,
                        help="Path to position ledger JSON")

    args = parser.parse_args()

    # Select connection
    if args.live:
        config = IBKRConfig.live_gateway(client_id=args.client_id)
        print(f"LIVE account — port {config.port}")
    else:
        config = IBKRConfig.paper_gateway(client_id=args.client_id)
        print(f"Paper account — port {config.port}")

    if args.dry_run:
        print("[DRY RUN MODE — no orders will be submitted]")

    trading_config = TradingConfig()
    ledger = PositionLedger(args.ledger) if args.ledger else PositionLedger()

    # Connect
    print(f"Connecting to IBKR (client_id={config.client_id})...")
    connection = IBKRConnection(config)
    try:
        connection.connect()
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    try:
        order_manager = OrderManager(connection, trading_config)
        monitor = ExitMonitor(connection, order_manager, ledger,
                             dry_run=args.dry_run)
        actions = monitor.run_check()

        summary = monitor.get_summary()
        print(f"\n{summary}")

        if not actions:
            print("No actions needed.")

    finally:
        connection.disconnect()


if __name__ == "__main__":
    main()
