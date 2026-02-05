#!/usr/bin/env python3
"""
Ad-hoc Gateway Test Script

Test IB Gateway connectivity and trading operations.

Remote Access (SSH Tunnel):
    # Set up SSH tunnel to AWS Gateway server:
    ssh -L 4001:127.0.0.1:4001 sschmidt@54.90.246.184 -N &

    # Then run tests (defaults to 127.0.0.1:4001)
    python test_gateway.py --balance

Usage:
    python test_gateway.py --all           # Test balance + portfolio
    python test_gateway.py --balance       # Account balance only
    python test_gateway.py --portfolio     # Portfolio positions only
    python test_gateway.py --buy --symbol NKE --shares 1
    python test_gateway.py --sell --symbol NKE --shares 1
    python test_gateway.py --buy --symbol NKE --dry-run  # Simulate only

Examples:
    python test_gateway.py --all
    python test_gateway.py --balance --portfolio
    python test_gateway.py --buy --symbol AAPL --shares 2
    python test_gateway.py --sell --symbol AAPL --shares 1 --dry-run
"""

import argparse
import sys
import os
from datetime import datetime, time
import pytz

# Add current and parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, parent_dir)

# Try importing from ibkr package first, fall back to direct imports
try:
    from ibkr.config import IBKRConfig
    from ibkr.connection import IBKRConnection
    from ibkr.notifier import TradeNotifier
except ModuleNotFoundError:
    from config import IBKRConfig
    from connection import IBKRConnection
    from notifier import TradeNotifier


def is_market_hours() -> bool:
    """Check if US market is currently open."""
    et = pytz.timezone('America/New_York')
    now = datetime.now(et)
    if now.weekday() > 4:
        return False
    market_open = time(9, 30)
    market_close = time(16, 0)
    return market_open <= now.time() <= market_close


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def test_balance(connection: IBKRConnection, account: str = None) -> bool:
    """Test account balance retrieval for one or all accounts."""
    print_header("ACCOUNT BALANCE")

    # Get list of accounts
    accounts = connection.ib.managedAccounts()
    print(f"Managed Accounts: {accounts}\n")

    # If specific account requested, filter to just that one
    if account:
        if account not in accounts:
            print(f"ERROR: Account {account} not found in {accounts}")
            return False
        accounts = [account]

    # Show balance for each account
    for acct in accounts:
        summary_items = connection.ib.accountSummary(acct)

        # Extract numeric values
        values = {}
        for item in summary_items:
            if item.value:
                try:
                    values[item.tag] = float(item.value)
                except ValueError:
                    pass

        acct_type = "FA Master" if acct.startswith("F") else "Individual"
        print(f"--- {acct} ({acct_type}) ---")
        print(f"  Net Liquidation:  ${values.get('NetLiquidation', 0):>12,.2f}")
        print(f"  Cash Balance:     ${values.get('TotalCashValue', 0):>12,.2f}")
        print(f"  Available Funds:  ${values.get('AvailableFunds', 0):>12,.2f}")
        print(f"  Buying Power:     ${values.get('BuyingPower', 0):>12,.2f}")
        print(f"  Position Value:   ${values.get('GrossPositionValue', 0):>12,.2f}")
        print(f"  Unrealized P&L:   ${values.get('UnrealizedPnL', 0):>12,.2f}")
        print()

    print("SUCCESS: Account balance retrieved")
    return True


def test_portfolio(connection: IBKRConnection) -> bool:
    """Test portfolio positions retrieval."""
    print_header("PORTFOLIO POSITIONS")

    portfolio = connection.ib.portfolio()

    if not portfolio:
        print("No positions currently held")
    else:
        print(f"{'Symbol':<8} {'Shares':>8} {'Avg Cost':>10} {'Cur Price':>10} {'Mkt Value':>12} {'P&L':>12} {'P&L %':>8}")
        print("-" * 72)
        total_value = 0
        total_pnl = 0
        for item in portfolio:
            cur_price = item.marketValue / item.position if item.position != 0 else 0
            pnl_pct = (item.unrealizedPNL / (item.averageCost * item.position) * 100) if item.averageCost * item.position != 0 else 0
            print(f"{item.contract.symbol:<8} "
                  f"{item.position:>8.0f} "
                  f"${item.averageCost:>8.2f} "
                  f"${cur_price:>8.2f} "
                  f"${item.marketValue:>10.2f} "
                  f"${item.unrealizedPNL:>10.2f} "
                  f"{pnl_pct:>7.1f}%")
            total_value += item.marketValue
            total_pnl += item.unrealizedPNL
        print("-" * 72)
        print(f"{'TOTAL':<8} {'':<8} {'':<10} {'':<10} ${total_value:>10.2f} ${total_pnl:>10.2f}")

    print(f"\nTotal positions: {len(portfolio)}")
    print("SUCCESS: Portfolio retrieved")
    return True


def test_buy(connection: IBKRConnection, symbol: str, shares: int, dry_run: bool = False) -> bool:
    """Test buying shares."""
    from ib_insync import Stock, MarketOrder

    print_header(f"BUY {shares} SHARE(S) OF {symbol}")
    print(f"Market Hours: {'Yes' if is_market_hours() else 'No (order may not fill)'}")
    print(f"Dry Run: {dry_run}")

    # Create and qualify contract
    contract = Stock(symbol, "SMART", "USD")
    connection.ib.qualifyContracts(contract)

    # Get current price
    ticker = connection.ib.reqMktData(contract, snapshot=True)
    connection.ib.sleep(2)
    price = ticker.marketPrice() or ticker.last
    connection.ib.cancelMktData(contract)

    if price:
        print(f"Current Price: ${price:.2f}")
        print(f"Estimated Cost: ${price * shares:.2f}")

    if dry_run:
        print("\nDRY RUN: Order not submitted")
        return True

    # Place order
    order = MarketOrder("BUY", shares)
    trade = connection.ib.placeOrder(contract, order)

    print(f"\nOrder submitted (ID: {trade.order.orderId})")

    # Wait for fill (up to 30 seconds)
    timeout = 30
    start = datetime.now()
    while not trade.isDone() and (datetime.now() - start).seconds < timeout:
        connection.ib.sleep(1)
        print(f"  Status: {trade.orderStatus.status}, Filled: {trade.orderStatus.filled}")

    print(f"\nFinal Status: {trade.orderStatus.status}")
    print(f"Filled: {trade.orderStatus.filled}")
    if trade.orderStatus.avgFillPrice:
        print(f"Fill Price: ${trade.orderStatus.avgFillPrice:.2f}")

    if trade.orderStatus.status == "Filled":
        print(f"\nSUCCESS: Bought {trade.orderStatus.filled} share(s) of {symbol}")
        return True
    else:
        # Cancel unfilled order
        if trade.orderStatus.status not in ["Filled", "Cancelled", "Inactive"]:
            print("Cancelling unfilled order...")
            connection.ib.cancelOrder(trade.order)
            connection.ib.sleep(2)
        print(f"\nOrder not filled (status: {trade.orderStatus.status})")
        return False


def test_sell(connection: IBKRConnection, symbol: str, shares: int, dry_run: bool = False) -> bool:
    """Test selling shares."""
    from ib_insync import Stock, MarketOrder

    print_header(f"SELL {shares} SHARE(S) OF {symbol}")
    print(f"Market Hours: {'Yes' if is_market_hours() else 'No (order may not fill)'}")
    print(f"Dry Run: {dry_run}")

    # Check current position
    portfolio = connection.ib.portfolio()
    position = None
    for item in portfolio:
        if item.contract.symbol == symbol:
            position = item
            break

    if not position or position.position < shares:
        held = position.position if position else 0
        print(f"\nERROR: Insufficient position. Have {held} shares, need {shares}")
        return False

    print(f"Current Position: {position.position} shares")
    print(f"Position P&L: ${position.unrealizedPNL:.2f}")

    if dry_run:
        print("\nDRY RUN: Order not submitted")
        return True

    # Create and qualify contract
    contract = Stock(symbol, "SMART", "USD")
    connection.ib.qualifyContracts(contract)

    # Place order
    order = MarketOrder("SELL", shares)
    trade = connection.ib.placeOrder(contract, order)

    print(f"\nOrder submitted (ID: {trade.order.orderId})")

    # Wait for fill (up to 30 seconds)
    timeout = 30
    start = datetime.now()
    while not trade.isDone() and (datetime.now() - start).seconds < timeout:
        connection.ib.sleep(1)
        print(f"  Status: {trade.orderStatus.status}, Filled: {trade.orderStatus.filled}")

    print(f"\nFinal Status: {trade.orderStatus.status}")
    print(f"Filled: {trade.orderStatus.filled}")
    if trade.orderStatus.avgFillPrice:
        print(f"Fill Price: ${trade.orderStatus.avgFillPrice:.2f}")

    if trade.orderStatus.status == "Filled":
        print(f"\nSUCCESS: Sold {trade.orderStatus.filled} share(s) of {symbol}")
        return True
    else:
        # Cancel unfilled order
        if trade.orderStatus.status not in ["Filled", "Cancelled", "Inactive"]:
            print("Cancelling unfilled order...")
            connection.ib.cancelOrder(trade.order)
            connection.ib.sleep(2)
        print(f"\nOrder not filled (status: {trade.orderStatus.status})")
        return False


def test_notify() -> bool:
    """Test notification system (email and SMS)."""
    print_header("NOTIFICATION TEST")

    notifier = TradeNotifier()

    print("Checking notification configuration...")
    print(f"  Email configured: {notifier.config.email_enabled}")
    print(f"  SMS configured:   {notifier.config.sms_enabled}")
    print()

    if not notifier.config.email_enabled and not notifier.config.sms_enabled:
        print("ERROR: No notifications configured!")
        print()
        print("Set these environment variables:")
        print("  Email: IBKR_SMTP_SERVER, IBKR_SMTP_USER, IBKR_SMTP_PASSWORD, IBKR_NOTIFY_EMAIL")
        print("  SMS:   IBKR_TWILIO_SID, IBKR_TWILIO_TOKEN, IBKR_TWILIO_FROM, IBKR_NOTIFY_PHONE")
        return False

    print("Sending test notifications...")
    results = notifier.test_notifications()

    print()
    if results.get("email_sent"):
        print("SUCCESS: Email sent")
    elif results.get("email_configured"):
        print("FAILED: Email configured but send failed")

    if results.get("sms_sent"):
        print("SUCCESS: SMS sent")
    elif results.get("sms_configured"):
        print("FAILED: SMS configured but send failed")

    return results.get("email_sent", False) or results.get("sms_sent", False)


def main():
    parser = argparse.ArgumentParser(
        description="Test IB Gateway connectivity and trading operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Connection options
    parser.add_argument("--host", default="127.0.0.1",
                        help="Gateway host address (default: 127.0.0.1 via SSH tunnel)")
    parser.add_argument("--port", type=int, default=4001,
                        help="Gateway port (default: 4001 for live)")
    parser.add_argument("--client-id", type=int, default=1,
                        help="Client ID (default: 1)")

    # Test options
    parser.add_argument("--all", action="store_true",
                        help="Run all tests (balance, portfolio)")
    parser.add_argument("--balance", action="store_true",
                        help="Test account balance retrieval")
    parser.add_argument("--portfolio", action="store_true",
                        help="Test portfolio positions retrieval")

    # Trading options
    parser.add_argument("--buy", action="store_true",
                        help="Test buying shares")
    parser.add_argument("--sell", action="store_true",
                        help="Test selling shares")
    parser.add_argument("--symbol", default="NKE",
                        help="Stock symbol for buy/sell (default: NKE)")
    parser.add_argument("--shares", type=int, default=1,
                        help="Number of shares for buy/sell (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't actually execute trades, just simulate")
    parser.add_argument("--account", default=None,
                        help="Specific account ID (default: show all accounts)")
    parser.add_argument("--test-notify", action="store_true",
                        help="Test email/SMS notifications (requires env vars)")

    args = parser.parse_args()

    # If no test specified, show help
    if not any([args.all, args.balance, args.portfolio, args.buy, args.sell, args.test_notify]):
        parser.print_help()
        print("\nExample: python test_gateway.py --balance --portfolio")
        sys.exit(1)

    # Handle notification test (doesn't need Gateway connection)
    if args.test_notify:
        success = test_notify()
        sys.exit(0 if success else 1)

    # Create configuration
    config = IBKRConfig.remote_gateway(
        host=args.host,
        port=args.port,
        client_id=args.client_id
    )

    print_header("CONNECTING TO IB GATEWAY")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Client ID: {args.client_id}")

    # Connect
    connection = IBKRConnection(config)
    if not connection.connect():
        print(f"\nERROR: Failed to connect to Gateway at {args.host}:{args.port}")
        sys.exit(1)

    print("SUCCESS: Connected to IB Gateway")

    # Track results
    results = {}

    try:
        # Run selected tests
        if args.all or args.balance:
            results["balance"] = test_balance(connection, args.account)

        if args.all or args.portfolio:
            results["portfolio"] = test_portfolio(connection)

        if args.buy:
            results["buy"] = test_buy(connection, args.symbol, args.shares, args.dry_run)

        if args.sell:
            results["sell"] = test_sell(connection, args.symbol, args.shares, args.dry_run)

    finally:
        # Always disconnect
        print_header("DISCONNECTING")
        connection.disconnect()
        print("Disconnected from IB Gateway")

    # Summary
    print_header("TEST SUMMARY")
    all_passed = True
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()