#!/usr/bin/env python
"""
Query real-time portfolio holdings from IBKR.

Connects to TWS/Gateway, fetches all positions and account summary,
and prints a formatted report. Optionally sends via email.

Usage:
    python scripts/portfolio.py                    # print to stdout
    python scripts/portfolio.py --email            # print + send email
    python scripts/portfolio.py --gateway          # use IB Gateway (default)
    python scripts/portfolio.py --tws              # use TWS instead
    python scripts/portfolio.py --live             # live account
    python scripts/portfolio.py --client-id 15     # custom client ID
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from ibkr.config import IBKRConfig, TradingConfig
from ibkr.trade_executor import TradeExecutor
from ibkr.notifier import TradeNotifier, NotificationConfig

logger = logging.getLogger(__name__)


def format_portfolio_report(executor):
    """Build a formatted portfolio report string."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []

    lines.append("PORTFOLIO HOLDINGS")
    lines.append("=" * 60)
    lines.append(f"Time: {timestamp}")
    lines.append("")

    # Account summary
    account = executor.get_account_summary()
    if account:
        lines.append("ACCOUNT SUMMARY")
        lines.append("-" * 60)
        lines.append(f"  Net Liquidation:    ${account.net_liquidation:>14,.2f}")
        lines.append(f"  Total Cash:         ${account.total_cash:>14,.2f}")
        lines.append(f"  Available Funds:    ${account.available_funds:>14,.2f}")
        lines.append(f"  Buying Power:       ${account.buying_power:>14,.2f}")
        lines.append(f"  Gross Positions:    ${account.gross_position_value:>14,.2f}")
        lines.append(f"  Unrealized P&L:     ${account.unrealized_pnl:>14,.2f}")
        lines.append("")

    # Positions
    summary = executor.get_position_summary()
    positions = summary.get('positions', {})

    if positions:
        lines.append(f"POSITIONS ({len(positions)})")
        lines.append("-" * 60)
        lines.append(f"  {'Symbol':<8} {'Shares':>6} {'AvgCost':>10} {'Value':>12}")
        lines.append(f"  {'------':<8} {'------':>6} {'-------':>10} {'-----':>12}")

        total_value = 0
        for sym in sorted(positions.keys()):
            pos = positions[sym]
            shares = pos['shares']
            avg_cost = pos['avg_cost']
            mkt_val = pos['market_value']
            total_value += mkt_val
            lines.append(f"  {sym:<8} {shares:>6} ${avg_cost:>9.2f} ${mkt_val:>11,.2f}")

        lines.append(f"  {'------':<8} {'------':>6} {'-------':>10} {'-----':>12}")
        lines.append(f"  {'TOTAL':<8} {len(positions):>6} {'':>10} ${total_value:>11,.2f}")
    else:
        lines.append("POSITIONS: none")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Query IBKR portfolio holdings")
    parser.add_argument("--email", action="store_true",
                        help="Send portfolio report via email")
    parser.add_argument("--gateway", action="store_true", default=True,
                        help="Connect via IB Gateway (default)")
    parser.add_argument("--tws", action="store_true",
                        help="Connect via TWS instead of Gateway")
    parser.add_argument("--live", action="store_true",
                        help="Use live trading account")
    parser.add_argument("--client-id", type=int, default=15,
                        help="IBKR client ID (default: 15)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    # Configure connection
    if args.live:
        if args.tws:
            ibkr_config = IBKRConfig.live_tws(client_id=args.client_id)
        else:
            ibkr_config = IBKRConfig.live_gateway(client_id=args.client_id)
    else:
        if args.tws:
            ibkr_config = IBKRConfig.paper_tws(client_id=args.client_id)
        else:
            ibkr_config = IBKRConfig.paper_gateway(client_id=args.client_id)

    trading_config = TradingConfig()

    with TradeExecutor(ibkr_config, trading_config, enable_notifications=False) as executor:
        report = format_portfolio_report(executor)
        print(report)

        if args.email:
            notifier = TradeNotifier()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            subject = f"[PORTFOLIO] Holdings snapshot — {timestamp}"
            if notifier._send_email(subject, report):
                print("Email sent.")
            else:
                print("Email failed (check IBKR_SMTP_* env vars).")


if __name__ == "__main__":
    main()
