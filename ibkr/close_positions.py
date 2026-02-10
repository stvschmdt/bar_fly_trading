"""
Close out IBKR positions.

Queries all open positions and submits market sell orders to flatten them.
Supports filtering by asset type (shares, options, or all).

Usage:
    python -m ibkr.close_positions --all                    # Close everything (gateway default)
    python -m ibkr.close_positions --shares                 # Close only stock positions
    python -m ibkr.close_positions --options                # Close only option contracts
    python -m ibkr.close_positions --all --dry-run          # Preview without executing
    python -m ibkr.close_positions --all --live             # Live trading via gateway
"""

import argparse
import logging
import sys
import time
from datetime import datetime

from ib_insync import IB, MarketOrder, Stock, Option, Contract

from .config import IBKRConfig
from .connection import IBKRConnection

logger = logging.getLogger(__name__)

TRADING_MODE_CONF = "ibkr/trading_mode.conf"


def check_live_trading_allowed():
    """Check trading_mode.conf for live trading authorization."""
    import os
    if not os.path.exists(TRADING_MODE_CONF):
        return False
    with open(TRADING_MODE_CONF) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or '=' not in line:
                continue
            key, val = line.split('=', 1)
            if key.strip() == 'TRADING_MODE' and val.strip().lower() == 'live':
                return True
    return False


def get_open_positions(ib: IB, account: str = None):
    """
    Get all open positions from IBKR.

    Returns list of dicts with: symbol, sec_type, shares, avg_cost, contract, account
    """
    all_positions = ib.positions()

    # Auto-detect account if not specified
    if not account:
        accounts = ib.managedAccounts()
        account = next((a for a in accounts if a.startswith('DU')), None)
        if not account:
            account = next((a for a in accounts if a.startswith('U')), None)
        if not account and accounts:
            account = accounts[0]

    positions = []
    for item in all_positions:
        # Filter by account
        if account and item.account != account:
            continue

        shares = int(item.position)
        if shares == 0:
            continue

        contract = item.contract
        sec_type = contract.secType  # STK, OPT, FUT, etc.

        pos = {
            'symbol': contract.symbol,
            'sec_type': sec_type,
            'shares': shares,
            'avg_cost': item.avgCost,
            'market_value': abs(shares) * item.avgCost,
            'contract': contract,
            'account': item.account,
        }

        # Add option-specific info
        if sec_type == 'OPT':
            pos['right'] = getattr(contract, 'right', '?')       # C or P
            pos['strike'] = getattr(contract, 'strike', 0)
            pos['expiry'] = getattr(contract, 'lastTradeDateOrContractMonth', '?')

        positions.append(pos)

    return positions, account


def filter_positions(positions, mode):
    """Filter positions by asset type."""
    if mode == 'all':
        return positions
    elif mode == 'shares':
        return [p for p in positions if p['sec_type'] == 'STK']
    elif mode == 'options':
        return [p for p in positions if p['sec_type'] == 'OPT']
    return positions


def display_positions(positions, mode):
    """Print positions table for user review."""
    if not positions:
        print(f"\nNo {mode} positions found to close.")
        return

    print(f"\n{'='*70}")
    print(f"POSITIONS TO CLOSE ({mode.upper()}) — {len(positions)} position(s)")
    print(f"{'='*70}")

    total_value = 0
    for p in positions:
        shares = p['shares']
        action = "SELL" if shares > 0 else "BUY TO COVER"
        value = p['market_value']
        total_value += value

        if p['sec_type'] == 'OPT':
            right = p.get('right', '?')
            strike = p.get('strike', 0)
            expiry = p.get('expiry', '?')
            print(f"  {action:14s} {abs(shares):>4} x {p['symbol']:6s} "
                  f"{right} ${strike:.0f} exp {expiry}  "
                  f"(avg_cost=${p['avg_cost']:.2f}  val=${value:,.2f})")
        else:
            print(f"  {action:14s} {abs(shares):>4}   {p['symbol']:6s} "
                  f"{'':26s}"
                  f"(avg_cost=${p['avg_cost']:.2f}  val=${value:,.2f})")

    print(f"{'─'*70}")
    print(f"  Total estimated value: ${total_value:,.2f}")
    print(f"{'='*70}")


def confirm_close():
    """Prompt user for YES/NO confirmation. Returns True only for exact 'YES'."""
    print("\nType YES to confirm closing these positions, or NO to abort:")
    try:
        response = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return False

    if response == "YES":
        return True
    else:
        print(f"Got '{response}' — aborting. (Must type exactly YES to confirm)")
        return False


def close_positions(ib: IB, positions, dry_run=False):
    """
    Submit market orders to close each position.

    Returns list of result dicts.
    """
    results = []

    for p in positions:
        symbol = p['symbol']
        shares = p['shares']
        contract = p['contract']

        # Determine action: sell longs, buy-to-cover shorts
        if shares > 0:
            action = 'SELL'
            qty = shares
        else:
            action = 'BUY'
            qty = abs(shares)

        if dry_run:
            print(f"  [DRY RUN] Would {action} {qty} {symbol} ({p['sec_type']})")
            results.append({
                'symbol': symbol, 'sec_type': p['sec_type'],
                'action': action, 'qty': qty,
                'status': 'dry_run', 'fill_price': 0,
            })
            continue

        # Qualify the contract before submitting
        try:
            qualified = ib.qualifyContracts(contract)
            if not qualified:
                print(f"  FAILED to qualify contract for {symbol} — skipping")
                results.append({
                    'symbol': symbol, 'sec_type': p['sec_type'],
                    'action': action, 'qty': qty,
                    'status': 'error', 'fill_price': 0,
                    'error': 'contract qualification failed',
                })
                continue
        except Exception as e:
            print(f"  FAILED to qualify {symbol}: {e} — skipping")
            results.append({
                'symbol': symbol, 'sec_type': p['sec_type'],
                'action': action, 'qty': qty,
                'status': 'error', 'fill_price': 0,
                'error': str(e),
            })
            continue

        # Submit market order
        order = MarketOrder(action, qty)
        print(f"  Submitting {action} {qty} {symbol} ({p['sec_type']})...", end=' ')

        try:
            trade = ib.placeOrder(contract, order)
            # Wait for fill (up to 60 seconds)
            timeout = 60
            start = time.time()
            while not trade.isDone() and (time.time() - start) < timeout:
                ib.sleep(1)

            if trade.orderStatus.status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                print(f"FILLED @ ${fill_price:.2f}")
                results.append({
                    'symbol': symbol, 'sec_type': p['sec_type'],
                    'action': action, 'qty': qty,
                    'status': 'filled', 'fill_price': fill_price,
                })
            else:
                status = trade.orderStatus.status
                print(f"STATUS: {status}")
                results.append({
                    'symbol': symbol, 'sec_type': p['sec_type'],
                    'action': action, 'qty': qty,
                    'status': status, 'fill_price': 0,
                })
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                'symbol': symbol, 'sec_type': p['sec_type'],
                'action': action, 'qty': qty,
                'status': 'error', 'fill_price': 0,
                'error': str(e),
            })

    return results


def print_summary(results):
    """Print execution summary."""
    if not results:
        return

    filled = [r for r in results if r['status'] == 'filled']
    dry = [r for r in results if r['status'] == 'dry_run']
    errors = [r for r in results if r['status'] == 'error']
    other = [r for r in results if r['status'] not in ('filled', 'dry_run', 'error')]

    print(f"\n{'='*50}")
    print(f"CLOSE SUMMARY")
    print(f"{'='*50}")

    if dry:
        print(f"  Dry run:  {len(dry)} position(s) previewed")
    if filled:
        print(f"  Filled:   {len(filled)} position(s) closed")
    if errors:
        print(f"  Errors:   {len(errors)} position(s) failed")
        for r in errors:
            print(f"    {r['symbol']}: {r.get('error', 'unknown')}")
    if other:
        print(f"  Pending:  {len(other)} position(s) not yet filled")
        for r in other:
            print(f"    {r['symbol']}: {r['status']}")

    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Close out IBKR positions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m ibkr.close_positions --all                    # Close everything (gateway default)
  python -m ibkr.close_positions --shares                 # Only stock positions
  python -m ibkr.close_positions --options                # Only option contracts
  python -m ibkr.close_positions --all --dry-run          # Preview only
  python -m ibkr.close_positions --shares --live          # Live via gateway
        """)

    # Position filter (mutually exclusive, required)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true",
                       help="Close all positions (shares + options)")
    group.add_argument("--shares", action="store_true",
                       help="Close only stock/share positions")
    group.add_argument("--options", action="store_true",
                       help="Close only option contract positions")

    # Connection
    parser.add_argument("--live", action="store_true",
                        help="Use live trading account (requires trading_mode.conf)")
    parser.add_argument("--gateway", action="store_true", default=True,
                        help="Connect via IB Gateway (default: True)")
    parser.add_argument("--tws", action="store_true",
                        help="Connect via TWS instead of IB Gateway")
    parser.add_argument("--client-id", type=int, default=20,
                        help="IBKR client ID (default: 20)")

    # Safety
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview positions without closing them")

    args = parser.parse_args()

    # Determine filter mode
    if args.all:
        mode = 'all'
    elif args.shares:
        mode = 'shares'
    else:
        mode = 'options'

    # Setup logging
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    # Live trading safety check
    if args.live:
        if not check_live_trading_allowed():
            print("ERROR: Live trading not enabled in ibkr/trading_mode.conf")
            print("  Set TRADING_MODE=live to enable.")
            sys.exit(1)

    # Select connection config (default: gateway; --tws overrides)
    use_gateway = not args.tws
    if args.live:
        if use_gateway:
            config = IBKRConfig.live_gateway(client_id=args.client_id)
        else:
            config = IBKRConfig.live_tws(client_id=args.client_id)
        print(f"LIVE account — port {config.port}")
    else:
        if use_gateway:
            config = IBKRConfig.paper_gateway(client_id=args.client_id)
        else:
            config = IBKRConfig.paper_tws(client_id=args.client_id)
        print(f"Paper account — port {config.port}")

    if args.dry_run:
        print("[DRY RUN MODE — no orders will be submitted]")

    # Connect and query positions
    print(f"Connecting to IBKR (client_id={config.client_id})...")

    try:
        conn = IBKRConnection(config)
        conn.connect()
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    try:
        positions, account = get_open_positions(conn.ib, config.account)
        print(f"Account: {account}")
        print(f"Total open positions: {len(positions)}")

        # Filter by mode
        filtered = filter_positions(positions, mode)

        # Display
        display_positions(filtered, mode)

        if not filtered:
            sys.exit(0)

        # Confirm
        if args.dry_run:
            results = close_positions(conn.ib, filtered, dry_run=True)
        else:
            if not confirm_close():
                sys.exit(0)
            print(f"\nClosing {len(filtered)} position(s)...\n")
            results = close_positions(conn.ib, filtered, dry_run=False)

        print_summary(results)

    finally:
        conn.disconnect()


if __name__ == "__main__":
    main()
