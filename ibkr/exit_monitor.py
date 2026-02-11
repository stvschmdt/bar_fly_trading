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
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibkr.config import IBKRConfig, TradingConfig
from ibkr.connection import IBKRConnection
from ibkr.notifier import TradeNotifier
from ibkr.order_manager import OrderManager
from ibkr.position_ledger import PositionLedger, make_ledger_key

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

        # Step 1: Sync with IBKR — detect filled brackets / closed options
        self._sync_with_ibkr()

        # Step 2: Check max hold day exits (stocks + options)
        self._check_max_hold_exits()

        # Step 3: Check SL/TP for options (no IBKR bracket orders)
        self._check_option_sl_tp_exits()

        # Step 4: Adjust trailing stops (stocks via order modify, options via sell)
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
            if shares == 0:
                continue
            if item.contract.secType == 'STK':
                ibkr_positions[sym] = shares
            elif item.contract.secType == 'OPT':
                right = item.contract.right  # 'C' or 'P'
                strike = item.contract.strike
                exp = item.contract.lastTradeDateOrContractMonth
                key = f"{sym}_{right}_{strike}_{exp}"
                ibkr_positions[key] = shares

        for ledger_key in list(self.ledger.get_all_positions().keys()):
            if ledger_key not in ibkr_positions:
                pos = self.ledger.get_position(ledger_key)
                itype = pos.get('instrument_type', 'stock') if pos else 'stock'
                reason = 'bracket_filled' if itype == 'stock' else 'option_closed'
                logger.info(f"  SYNC: {ledger_key} no longer in IBKR — removing")
                self.ledger.remove_position(ledger_key, reason=reason)
                self.actions.append({
                    'action': 'sync_remove',
                    'symbol': ledger_key,
                    'reason': f'position closed in IBKR ({reason})',
                    'strategy': pos.get('strategy', '') if pos else '',
                })

    def _build_contract(self, pos: dict):
        """Build ib_insync contract from ledger position data."""
        from ib_insync import Stock, Option as IBOption
        symbol = pos['symbol']
        if pos.get('instrument_type') == 'option':
            exp = str(pos.get('expiration', '')).replace('-', '')
            strike = pos.get('strike', 0)
            ctype = pos.get('contract_type', '')
            right = 'C' if ctype.lower().startswith('c') else 'P'
            return IBOption(symbol, exp, strike, right, 'SMART', currency='USD')
        return Stock(symbol, "SMART", "USD")

    def _check_max_hold_exits(self):
        """Cancel brackets and market sell for positions past max hold."""
        expired = self.ledger.get_expired_positions()

        for pos in expired:
            ledger_key = make_ledger_key(
                pos['symbol'], pos.get('instrument_type', 'stock'),
                pos.get('contract_type', ''), pos.get('strike', 0),
                pos.get('expiration', ''))
            hold_days = pos['hold_days']
            max_days = pos.get('max_hold_days', 0)
            shares = pos.get('shares', 0)
            instrument = pos.get('instrument_type', 'stock')
            unit = "contracts" if instrument == 'option' else "shares"

            logger.info(f"  MAX HOLD: {ledger_key} held {hold_days}d (max={max_days}d) — "
                       f"closing {shares} {unit}")

            if self.dry_run:
                self.actions.append({
                    'action': 'max_hold_exit',
                    'symbol': ledger_key,
                    'hold_days': hold_days,
                    'shares': shares,
                    'dry_run': True,
                })
                continue

            # Cancel existing bracket orders (stock only)
            if instrument == 'stock':
                stop_id = pos.get('stop_order_id', -1)
                profit_id = pos.get('profit_order_id', -1)
                if stop_id > 0 or profit_id > 0:
                    self.order_manager.cancel_bracket_orders(stop_id, profit_id)

            # Submit market sell
            from ib_insync import MarketOrder
            contract = self._build_contract(pos)
            try:
                self.connection.ib.qualifyContracts(contract)
                order = MarketOrder("SELL", shares)
                order.tif = "DAY"
                acct = self.order_manager.target_account
                if acct:
                    order.account = acct
                trade = self.connection.ib.placeOrder(contract, order)
                logger.info(f"  Submitted market sell for {ledger_key}: {shares} {unit} "
                           f"(order_id={trade.order.orderId})")
            except Exception as e:
                logger.error(f"  Failed to submit sell for {ledger_key}: {e}")

            self.ledger.remove_position(ledger_key, reason='max_hold')
            self.actions.append({
                'action': 'max_hold_exit',
                'symbol': ledger_key,
                'hold_days': hold_days,
                'shares': shares,
                'instrument_type': instrument,
                'strategy': pos.get('strategy', ''),
                'entry_price': pos.get('entry_price', 0),
                'dry_run': False,
            })

    def _get_price(self, pos: dict) -> Optional[float]:
        """Get current price for a position (stock or option)."""
        symbol = pos['symbol']
        if pos.get('instrument_type') == 'option':
            exp = str(pos.get('expiration', '')).replace('-', '')
            strike = pos.get('strike', 0)
            ctype = pos.get('contract_type', '')
            right = 'C' if ctype.lower().startswith('c') else 'P'
            return self.connection.get_option_price(symbol, exp, strike, right)
        return self.connection.get_current_price(symbol)

    def _sell_position(self, pos: dict, ledger_key: str, reason: str,
                       current_price: float = None):
        """Submit a market sell for a position and remove from ledger."""
        from ib_insync import MarketOrder
        shares = pos.get('shares', 0)
        contract = self._build_contract(pos)
        try:
            self.connection.ib.qualifyContracts(contract)
            order = MarketOrder("SELL", shares)
            order.tif = "DAY"
            acct = self.order_manager.target_account
            if acct:
                order.account = acct
            trade = self.connection.ib.placeOrder(contract, order)
            logger.info(f"  Submitted SELL for {ledger_key} ({reason})")
        except Exception as e:
            logger.error(f"  Failed to sell {ledger_key}: {e}")
        self.ledger.remove_position(ledger_key, reason=reason,
                                     exit_price=current_price)

    def _check_option_sl_tp_exits(self):
        """Check SL/TP for options (no IBKR bracket orders — software polled)."""
        for ledger_key, pos in list(self.ledger.get_all_positions().items()):
            if pos.get('instrument_type') != 'option':
                continue
            # Skip if somehow has bracket orders
            if pos.get('stop_order_id', -1) > 0:
                continue

            entry_price = pos['entry_price']
            sl_pct = pos.get('stop_loss_pct')
            tp_pct = pos.get('take_profit_pct')

            current_price = self._get_price(pos)
            if current_price is None or current_price <= 0 or entry_price <= 0:
                continue

            pct_change = (current_price - entry_price) / entry_price
            reason = None
            if sl_pct is not None and pct_change <= sl_pct:
                reason = f"stop_loss ({pct_change:+.1%} <= {sl_pct:.0%})"
            elif tp_pct is not None and pct_change >= tp_pct:
                reason = f"take_profit ({pct_change:+.1%} >= {tp_pct:+.0%})"

            if reason is None:
                continue

            logger.info(f"  OPT EXIT: {ledger_key} {reason} "
                       f"(entry=${entry_price:.2f}, now=${current_price:.2f})")

            if self.dry_run:
                self.actions.append({'action': 'option_sl_tp_exit',
                                     'symbol': ledger_key, 'reason': reason,
                                     'entry_price': entry_price,
                                     'current_price': current_price,
                                     'strategy': pos.get('strategy', ''),
                                     'shares': pos.get('shares', 0),
                                     'dry_run': True})
                continue

            self._sell_position(pos, ledger_key, reason, current_price)
            self.actions.append({'action': 'option_sl_tp_exit',
                                 'symbol': ledger_key, 'reason': reason,
                                 'entry_price': entry_price,
                                 'current_price': current_price,
                                 'strategy': pos.get('strategy', ''),
                                 'shares': pos.get('shares', 0),
                                 'dry_run': False})

    def _adjust_trailing_stops(self):
        """Adjust stop order prices upward for trailing stop positions."""
        trailing_positions = self.ledger.get_trailing_stop_positions()

        for pos in trailing_positions:
            ledger_key = make_ledger_key(
                pos['symbol'], pos.get('instrument_type', 'stock'),
                pos.get('contract_type', ''), pos.get('strike', 0),
                pos.get('expiration', ''))
            trailing_pct = pos['trailing_stop_pct']
            old_hwm = pos.get('high_water_mark', pos['entry_price'])
            stop_id = pos.get('stop_order_id', -1)

            current_price = self._get_price(pos)
            if current_price is None or current_price <= 0:
                continue

            # Update high-water mark if price is higher
            if current_price > old_hwm:
                new_hwm = current_price
                self.ledger.update_high_water_mark(ledger_key, new_hwm)

                new_stop_price = round(new_hwm * (1 + trailing_pct), 2)
                old_stop_price = round(old_hwm * (1 + trailing_pct), 2)

                if new_stop_price > old_stop_price:
                    logger.info(
                        f"  TRAILING: {ledger_key} HWM ${old_hwm:.2f} → ${new_hwm:.2f}, "
                        f"stop ${old_stop_price:.2f} → ${new_stop_price:.2f}")

                    if not self.dry_run:
                        if stop_id > 0:
                            # Stock: modify IBKR stop order
                            self.order_manager.modify_stop_price(stop_id, new_stop_price)
                        elif pos.get('instrument_type') == 'option':
                            # Option: check if trailing stop triggered
                            if current_price <= new_stop_price:
                                logger.info(f"  TRAILING STOP triggered for {ledger_key}")
                                self._sell_position(pos, ledger_key,
                                                    'trailing_stop', current_price)

                    self.actions.append({
                        'action': 'trailing_stop_adjust',
                        'symbol': ledger_key,
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

        lines = [f"Exit monitor: {len(self.actions)} action(s)",
                 f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                 ""]
        for a in self.actions:
            action = a['action']
            symbol = a.get('symbol', '?')
            dry = " [DRY RUN]" if a.get('dry_run') else ""
            strategy = a.get('strategy', '')
            strat_tag = f" [{strategy}]" if strategy else ""
            if action == 'sync_remove':
                lines.append(f"  SYNC {symbol}: removed ({a.get('reason', 'closed')}){strat_tag}")
            elif action == 'max_hold_exit':
                entry = a.get('entry_price', 0)
                itype = a.get('instrument_type', 'stock')
                unit = "contracts" if itype == 'option' else "shares"
                lines.append(f"  EXIT {symbol}: max hold {a.get('hold_days')}d, "
                             f"{a.get('shares', 0)} {unit}, entry=${entry:.2f}{strat_tag}{dry}")
            elif action == 'option_sl_tp_exit':
                entry = a.get('entry_price', 0)
                cur = a.get('current_price', 0)
                pnl = (cur - entry) * a.get('shares', 0) * 100 if entry > 0 else 0
                lines.append(f"  OPT EXIT {symbol}: {a.get('reason', 'sl/tp')}, "
                             f"entry=${entry:.2f} → ${cur:.2f}, "
                             f"P&L=${pnl:+,.0f}{strat_tag}{dry}")
            elif action == 'trailing_stop_adjust':
                lines.append(f"  TRAIL {symbol}: stop ${a.get('old_stop'):.2f} → "
                           f"${a.get('new_stop'):.2f}{strat_tag}{dry}")
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
    parser.add_argument("--no-notify", action="store_true",
                        help="Suppress email notifications")

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

        # Send email if actions were taken
        if actions and not args.no_notify:
            try:
                notifier = TradeNotifier()
                dry_tag = "[DRY RUN] " if args.dry_run else ""
                acct_tag = "LIVE" if args.live else "Paper"
                subject = (f"{dry_tag}[EXIT MONITOR] {len(actions)} action(s) "
                           f"({acct_tag})")
                notifier._send_email(subject, summary)
                print(f"Email sent: {subject}")
            except Exception as e:
                print(f"WARNING: Failed to send email: {e}")

    finally:
        connection.disconnect()


if __name__ == "__main__":
    main()
