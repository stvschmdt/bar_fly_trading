#!/usr/bin/env python3
"""Check MySQL for stale symbols and optionally re-pull them.

Compares each symbol's latest date in core_stock against the expected
trading day (yesterday, or last Friday if weekend). Symbols that are
behind get flagged and optionally re-pulled via pull_api_data.

Usage:
    # Report only — show stale symbols
    python scripts/check_stale_symbols.py

    # Re-pull stale symbols automatically
    python scripts/check_stale_symbols.py --fix

    # Check specific symbols
    python scripts/check_stale_symbols.py -s AAPL MSFT NVDA

    # Custom expected date
    python scripts/check_stale_symbols.py --expected-date 2026-02-12

    # Also rebuild gold table CSVs after fixing
    python scripts/check_stale_symbols.py --fix --rebuild-csv
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta

import pymysql


def get_db_connection():
    """Connect to MySQL using same credentials as pull_api_data."""
    password = os.environ.get("MYSQL_PASSWORD", "")
    return pymysql.connect(
        host="127.0.0.1",
        port=3306,
        user="root",
        password=password,
        database="bar_fly_trading",
        cursorclass=pymysql.cursors.DictCursor,
    )


def last_trading_day(ref_date=None):
    """Return the most recent trading day (excludes weekends)."""
    if ref_date is None:
        ref_date = datetime.now().date()
    # If today is before market close, use yesterday
    if isinstance(ref_date, datetime):
        ref_date = ref_date.date()
    d = ref_date - timedelta(days=1)
    # Skip weekends
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d -= timedelta(days=1)
    return d


def get_all_symbol_dates(conn, symbols=None):
    """Get latest date per symbol from core_stock table."""
    with conn.cursor() as cur:
        if symbols:
            placeholders = ",".join(["%s"] * len(symbols))
            cur.execute(
                f"SELECT symbol, MAX(date) as last_date FROM core_stock "
                f"WHERE symbol IN ({placeholders}) GROUP BY symbol",
                symbols,
            )
        else:
            cur.execute(
                "SELECT symbol, MAX(date) as last_date FROM core_stock GROUP BY symbol"
            )
        return {row["symbol"]: row["last_date"] for row in cur.fetchall()}


def get_symbol_count(conn):
    """Total distinct symbols in core_stock."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(DISTINCT symbol) as cnt FROM core_stock")
        return cur.fetchone()["cnt"]


def main():
    parser = argparse.ArgumentParser(description="Check for stale symbols in MySQL")
    parser.add_argument("-s", "--symbols", nargs="+", help="Check specific symbols")
    parser.add_argument("--fix", action="store_true", help="Re-pull stale symbols")
    parser.add_argument("--rebuild-csv", action="store_true",
                        help="Rebuild gold table CSVs after fixing")
    parser.add_argument("--expected-date", help="Expected latest date (YYYY-MM-DD)")
    parser.add_argument("--max-stale-days", type=int, default=3,
                        help="Max days behind before flagging (default: 3)")
    parser.add_argument("--batch-size", type=int, default=20,
                        help="How many symbols to re-pull at once (default: 20)")
    args = parser.parse_args()

    # Determine expected date
    if args.expected_date:
        expected = datetime.strptime(args.expected_date, "%Y-%m-%d").date()
    else:
        expected = last_trading_day()

    print(f"Expected latest trading day: {expected}")
    print(f"Stale threshold: >{args.max_stale_days} days behind")
    print()

    conn = get_db_connection()
    try:
        total = get_symbol_count(conn)
        symbol_dates = get_all_symbol_dates(conn, args.symbols)

        current = []
        stale = []
        very_stale = []

        for sym, last_date in sorted(symbol_dates.items()):
            if hasattr(last_date, "date"):
                last_date = last_date.date()
            elif isinstance(last_date, str):
                last_date = datetime.strptime(last_date, "%Y-%m-%d").date()

            days_behind = (expected - last_date).days

            if days_behind <= 0:
                current.append(sym)
            elif days_behind <= args.max_stale_days:
                stale.append((sym, last_date, days_behind))
            else:
                very_stale.append((sym, last_date, days_behind))

        # Report
        print(f"Total symbols in DB: {total}")
        print(f"Checked: {len(symbol_dates)}")
        print(f"  Current ({expected}): {len(current)}")
        print(f"  Stale (1-{args.max_stale_days}d): {len(stale)}")
        print(f"  Very stale (>{args.max_stale_days}d): {len(very_stale)}")

        if stale:
            print(f"\nStale symbols ({len(stale)}):")
            for sym, dt, days in stale:
                print(f"  {sym:8s} last: {dt}  ({days}d behind)")

        if very_stale:
            print(f"\nVery stale / likely delisted ({len(very_stale)}):")
            for sym, dt, days in sorted(very_stale, key=lambda x: x[1]):
                print(f"  {sym:8s} last: {dt}  ({days}d behind)")

        # Fix mode: re-pull stale symbols
        fixable = [sym for sym, _, days in stale if days > 0]
        if args.fix and fixable:
            print(f"\n--- Re-pulling {len(fixable)} stale symbols ---")
            for i in range(0, len(fixable), args.batch_size):
                batch = fixable[i : i + args.batch_size]
                print(f"  Batch {i // args.batch_size + 1}: {' '.join(batch)}")
                cmd = [
                    sys.executable, "-m", "api_data.pull_api_data",
                    "-s",
                ] + batch
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"    ERROR: {result.stderr[-200:]}")
                else:
                    print(f"    OK")

            if args.rebuild_csv:
                print("\n--- Rebuilding gold table CSVs ---")
                cmd = [
                    sys.executable, "-m", "api_data.pull_api_data",
                    "-w", "all", "--gold-table-only",
                ]
                subprocess.run(cmd)

            print("\nDone. Re-run without --fix to verify.")
        elif args.fix and not fixable:
            print("\nNothing to fix — all symbols are current.")
        elif fixable:
            print(f"\nRun with --fix to re-pull {len(fixable)} stale symbols.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
