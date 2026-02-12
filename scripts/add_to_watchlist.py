#!/usr/bin/env python
"""
Merge symbols from a new CSV into an existing watchlist.

Handles multiple input formats:
  - Standard watchlist: "Symbol","Name","Is ETF"
  - Ideas export:       "SYMBOL, NAME","Last Price",...  (symbol+name in col 0)
  - Plain list:         one symbol per line (no header)

If a new symbol's ETF status is unknown, queries AlphaVantage OVERVIEW endpoint.

Usage:
    python scripts/add_to_watchlist.py new.csv old.csv          # merge in-place
    python scripts/add_to_watchlist.py new.csv old.csv -o out.csv  # write to separate file
    python scripts/add_to_watchlist.py new.csv old.csv --dry-run   # preview only
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_watchlist(path):
    """Read existing watchlist. Returns dict {SYMBOL: (name, is_etf)}."""
    entries = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            symbol = row[0].strip().strip('"').upper()
            name = row[1].strip().strip('"') if len(row) > 1 else ""
            is_etf = row[2].strip().strip('"') if len(row) > 2 else "0"
            entries[symbol] = (name, is_etf)
    return entries


def parse_new_symbols(path):
    """
    Parse symbols from a new CSV file. Auto-detects format.
    Returns list of (symbol, name, is_etf_or_None).
    """
    results = []

    with open(path, newline='', encoding='utf-8-sig') as f:
        content = f.read()

    lines = content.strip().split('\n')
    if not lines:
        return results

    # Detect format from first data line
    reader = csv.reader(lines)
    header = next(reader, None)
    if not header:
        return results

    headers_lower = [h.strip().strip('"').lower() for h in header]

    # Standard watchlist format: "Symbol","Name","Is ETF"
    if 'is etf' in headers_lower:
        for row in reader:
            if not row:
                continue
            symbol = row[0].strip().strip('"').upper()
            name = row[1].strip().strip('"') if len(row) > 1 else ""
            is_etf = row[2].strip().strip('"') if len(row) > 2 else None
            results.append((symbol, name, is_etf))
        return results

    # Ideas/brokerage export: "Symbol" col contains "TICKER, COMPANY NAME"
    # Detected by "Last Price" or similar columns, or comma in first data row
    for row in reader:
        if not row:
            continue
        col0 = row[0].strip().strip('"')
        if ',' in col0:
            parts = col0.split(',', 1)
            symbol = parts[0].strip().upper()
            name = parts[1].strip()
        else:
            symbol = col0.strip().upper()
            name = ""

        if not symbol:
            continue

        results.append((symbol, name, None))

    return results


def lookup_etf_status(symbol):
    """Query AlphaVantage OVERVIEW to determine if symbol is an ETF."""
    try:
        from api_data.alpha_client import alpha_client
        resp = alpha_client.fetch(function='OVERVIEW', symbol=symbol)
        asset_type = resp.get('AssetType', '').upper()
        if 'ETF' in asset_type:
            return "1"
        if resp.get('Name'):
            return "0"
        # No data — might be ETF (OVERVIEW doesn't cover ETFs well)
        # Check if name contains ETF keywords
        name = resp.get('Name', '')
        if any(kw in name.upper() for kw in ['ETF', 'TRUST', 'FUND']):
            return "1"
        return "0"
    except Exception as e:
        print(f"  WARNING: API lookup failed for {symbol}: {e}")
        return "0"


# Sector ETFs, index ETFs, and benchmarks — always sorted to bottom
PIN_BOTTOM = {
    'SPY', 'QQQ',
    'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY', 'XRT',
    'TQQQ', 'SQQQ', 'SOXL', 'TECL', 'FAS', 'NUGT', 'GUSH',
    'GLD', 'SLV', 'BITO',
    'JEPI', 'QYLD', 'DWAS', 'NUSI',
}


def write_watchlist(path, entries):
    """Write watchlist in standard format. Pins ETFs/sectors/benchmarks to bottom."""
    stocks = {s: v for s, v in entries.items() if s not in PIN_BOTTOM}
    pinned = {s: v for s, v in entries.items() if s in PIN_BOTTOM}

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["Symbol", "Name", "Is ETF"])
        for symbol in sorted(stocks.keys()):
            name, is_etf = stocks[symbol]
            writer.writerow([symbol, name, is_etf])
        for symbol in sorted(pinned.keys()):
            name, is_etf = pinned[symbol]
            writer.writerow([symbol, name, is_etf])


def main():
    parser = argparse.ArgumentParser(description="Merge symbols into a watchlist")
    parser.add_argument("new_csv", help="CSV with new symbols to add")
    parser.add_argument("old_csv", help="Existing watchlist CSV (updated in-place unless -o)")
    parser.add_argument("-o", "--output", default=None,
                        help="Write merged result to this file instead of updating old_csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without writing")
    parser.add_argument("--no-api", action="store_true",
                        help="Skip API lookups for ETF status (default to 0)")
    args = parser.parse_args()

    # Read existing watchlist
    existing = read_watchlist(args.old_csv)
    print(f"Existing watchlist: {len(existing)} symbols from {args.old_csv}")

    # Parse new symbols
    new_symbols = parse_new_symbols(args.new_csv)
    print(f"New file: {len(new_symbols)} symbols from {args.new_csv}")

    # Merge
    added = []
    skipped = []
    api_lookups = 0

    for symbol, name, is_etf in new_symbols:
        if symbol in existing:
            skipped.append(symbol)
            continue

        # Resolve unknown ETF status
        if is_etf is None:
            if args.no_api:
                is_etf = "0"
            else:
                print(f"  Looking up ETF status for {symbol}...", end=" ")
                is_etf = lookup_etf_status(symbol)
                label = "ETF" if is_etf == "1" else "stock"
                print(f"{label}")
                api_lookups += 1
                if api_lookups % 5 == 0:
                    time.sleep(1)  # rate limit

        existing[symbol] = (name, is_etf)
        added.append(symbol)

    # Report
    print(f"\nResults:")
    print(f"  Added:   {len(added)} new symbols")
    print(f"  Skipped: {len(skipped)} (already in watchlist)")
    print(f"  Total:   {len(existing)} symbols")

    if added:
        print(f"\n  New symbols: {', '.join(sorted(added))}")

    if args.dry_run:
        print("\n  [DRY RUN] No files written.")
        return

    output_path = args.output or args.old_csv
    write_watchlist(output_path, existing)
    print(f"\n  Written to {output_path}")


if __name__ == "__main__":
    main()
