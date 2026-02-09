"""
Realtime intraday data updater.

Updates all_data_*.csv files in place with fresh OHLCV from AlphaVantage
GLOBAL_QUOTE API. Technical indicators are kept from the nightly pull (they
don't change intraday). Derived columns (sma_*_pct, pe_ratio, etc.) are
recomputed against the new price.

Usage:
    python -m api_data.pull_api_data_rt                          # Watchlist symbols only
    python -m api_data.pull_api_data_rt -w all                   # SP500 + watchlist (~543 symbols)
    python -m api_data.pull_api_data_rt -s AAPL NVDA TSLA       # Specific symbols
    python -m api_data.pull_api_data_rt --dry-run
"""

import argparse
import glob
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Column groups
# ─────────────────────────────────────────────────────────────────────────────

CORE_RT_COLUMNS = ['open', 'high', 'low', 'adjusted_close', 'volume']


# ─────────────────────────────────────────────────────────────────────────────
# API fetcher (no DB dependency)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_realtime_quote(api_client, symbol):
    """
    Fetch current OHLCV via GLOBAL_QUOTE endpoint.

    Args:
        api_client: AlphaVantageClient instance
        symbol: Stock ticker

    Returns:
        Dict with keys matching CORE_RT_COLUMNS
    """
    response = api_client.fetch(function='GLOBAL_QUOTE', symbol=symbol)

    if 'Global Quote' not in response or not response['Global Quote']:
        raise ValueError(f"No quote data returned for {symbol}")

    gq = response['Global Quote']
    return {
        'open': float(gq.get('02. open', 0)),
        'high': float(gq.get('03. high', 0)),
        'low': float(gq.get('04. low', 0)),
        'adjusted_close': float(gq.get('05. price', 0)),
        'volume': int(gq.get('06. volume', 0)),
    }


def fetch_symbol_rt_data(api_client, symbol):
    """
    Fetch realtime OHLCV for a symbol (1 API call).

    Args:
        api_client: AlphaVantageClient instance
        symbol: Stock ticker

    Returns:
        Dict with 'ohlcv' sub-dict, or None on failure
    """
    try:
        ohlcv = fetch_realtime_quote(api_client, symbol)
    except Exception as e:
        logger.warning(f"Quote failed for {symbol}: {e}")
        return None

    return {'ohlcv': ohlcv}


# ─────────────────────────────────────────────────────────────────────────────
# CSV update logic
# ─────────────────────────────────────────────────────────────────────────────

def update_derived_columns(df, idx):
    """
    Recompute derived columns for a specific row after RT update.

    Updates: sma_*_pct, 52_week_*_pct, pe_ratio, adjusted_close_pct,
    volume_pct using the new price against existing (nightly) technicals.

    Args:
        df: Full DataFrame (modified in place)
        idx: Row index to update
    """
    row = df.loc[idx]
    price = row.get('adjusted_close')
    if pd.isna(price) or price == 0:
        return

    # SMA distance percentages
    for period in [20, 50, 200]:
        sma_col = f'sma_{period}'
        pct_col = f'sma_{period}_pct'
        if sma_col in df.columns and pct_col in df.columns:
            sma_val = row.get(sma_col)
            if pd.notna(sma_val) and sma_val != 0:
                df.loc[idx, pct_col] = round((price - sma_val) / sma_val * 100, 2)

    # 52-week high/low pct
    for metric in ['52_week_high', '52_week_low']:
        pct_col = f'{metric}_pct'
        if metric in df.columns and pct_col in df.columns:
            val = row.get(metric)
            if pd.notna(val) and val != 0:
                df.loc[idx, pct_col] = round((price - val) / val * 100, 2)

    # PE ratio
    if 'ttm_eps' in df.columns and 'pe_ratio' in df.columns:
        ttm = row.get('ttm_eps')
        if pd.notna(ttm) and ttm != 0:
            df.loc[idx, 'pe_ratio'] = price / ttm

    # Pct change vs previous row for same symbol
    symbol = row.get('symbol')
    if symbol is not None:
        sym_mask = df['symbol'] == symbol
        sym_indices = df.loc[sym_mask].index
        pos = sym_indices.get_loc(idx)
        if pos > 0:
            prev_idx = sym_indices[pos - 1]
            prev_close = df.loc[prev_idx, 'adjusted_close']
            if pd.notna(prev_close) and prev_close != 0:
                df.loc[idx, 'adjusted_close_pct'] = (price - prev_close) / prev_close

            prev_vol = df.loc[prev_idx, 'volume']
            if 'volume_pct' in df.columns and pd.notna(prev_vol) and prev_vol != 0:
                df.loc[idx, 'volume_pct'] = (row['volume'] - prev_vol) / prev_vol


def update_csv_file(csv_path, rt_data, dry_run=False):
    """
    Update the latest row per symbol in a CSV with realtime OHLCV data.

    Args:
        csv_path: Path to all_data_X.csv
        rt_data: Dict of {symbol: {'ohlcv': {...}}}
        dry_run: If True, print changes but don't write

    Returns:
        Number of symbols updated
    """
    df = pd.read_csv(csv_path, index_col=0)

    updated = 0
    for symbol, data in rt_data.items():
        if data is None:
            continue

        sym_mask = df['symbol'] == symbol
        if not sym_mask.any():
            continue

        # Get the last row index for this symbol
        latest_idx = df.loc[sym_mask].index[-1]

        # Update OHLCV columns
        for col, val in data.get('ohlcv', {}).items():
            if col in df.columns:
                df.loc[latest_idx, col] = val

        # Recompute derived columns against existing technicals
        update_derived_columns(df, latest_idx)

        updated += 1
        if dry_run:
            price = data.get('ohlcv', {}).get('adjusted_close', '?')
            print(f"  [DRY RUN] {symbol}: price={price}")

    if not dry_run and updated > 0:
        df.to_csv(csv_path)

    return updated


# ─────────────────────────────────────────────────────────────────────────────
# Symbol universe
# ─────────────────────────────────────────────────────────────────────────────

def load_symbol_universe():
    """
    Load the full symbol universe from sp500.csv + watchlist.csv.

    Returns:
        Set of unique ticker symbols (~543)
    """
    from constants import SP500_PATH, WATCHLIST_PATH

    symbols = set()
    for path in [SP500_PATH, WATCHLIST_PATH]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            symbols.update(df['Symbol'].str.strip().tolist())
        else:
            logger.warning(f"Symbol file not found: {path}")

    return symbols


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def find_csv_files(data_dir):
    """Find all all_data_*.csv files in a directory."""
    pattern = os.path.join(data_dir, 'all_data_*.csv')
    files = sorted(glob.glob(pattern))
    return files


def get_symbols_from_csv(csv_path):
    """Get unique symbols from a CSV without loading all data."""
    df = pd.read_csv(csv_path, index_col=0, usecols=[0, 2])  # index + symbol
    return df['symbol'].unique().tolist()


def run_rt_update(api_client, data_dir, symbols_filter=None, dry_run=False,
                  use_all_symbols=False):
    """
    Main orchestrator: fetch RT OHLCV and update all CSV files.

    Args:
        api_client: AlphaVantageClient instance
        data_dir: Directory containing all_data_*.csv files
        symbols_filter: Optional set to restrict which symbols to update
        dry_run: If True, don't write changes
        use_all_symbols: If True, use SP500 + watchlist as symbol universe

    Returns:
        Dict with stats: total_symbols, updated, failed, skipped, elapsed_seconds
    """
    csv_files = find_csv_files(data_dir)
    if not csv_files:
        print(f"No all_data_*.csv files found in {data_dir}")
        return {'total_symbols': 0, 'updated': 0, 'failed': 0,
                'skipped': 0, 'elapsed_seconds': 0}

    print(f"Found {len(csv_files)} CSV files in {data_dir}")

    # Collect all symbols across all CSV files (these are the ones we can update)
    symbol_to_files = {}
    for csv_path in csv_files:
        for sym in get_symbols_from_csv(csv_path):
            symbol_to_files.setdefault(sym, []).append(csv_path)

    csv_symbols = set(symbol_to_files.keys())

    # Determine which symbols to fetch
    if use_all_symbols:
        universe = load_symbol_universe()
        # Intersect with what's actually in CSVs (can't update what doesn't exist)
        fetchable = universe & csv_symbols
        skipped = universe - csv_symbols
        if skipped:
            print(f"  {len(skipped)} symbols not in CSVs (skipped): "
                  f"{', '.join(sorted(skipped)[:10])}"
                  f"{'...' if len(skipped) > 10 else ''}")
        # Apply explicit filter on top
        if symbols_filter:
            fetchable = fetchable & set(symbols_filter)
    else:
        fetchable = csv_symbols
        skipped = set()
        if symbols_filter:
            fetchable = fetchable & set(symbols_filter)

    all_symbols = sorted(fetchable)
    print(f"Updating {len(all_symbols)} symbols (OHLCV only, 1 API call each): "
          f"{', '.join(all_symbols[:10])}"
          f"{'...' if len(all_symbols) > 10 else ''}")

    # Fetch RT data for all symbols
    rt_data = {}
    start = time.time()
    total = len(all_symbols)
    failed = 0

    for i, symbol in enumerate(all_symbols, 1):
        elapsed = time.time() - start
        rate = elapsed / i if i > 1 else 0
        eta = rate * (total - i)
        print(f"  [{i}/{total}] {symbol} "
              f"(elapsed: {elapsed:.0f}s, eta: {eta:.0f}s) ...", end='', flush=True)

        data = fetch_symbol_rt_data(api_client, symbol)
        if data:
            rt_data[symbol] = data
            price = data['ohlcv']['adjusted_close']
            print(f" ${price:.2f}")
        else:
            failed += 1
            print(" FAILED")

    fetch_elapsed = time.time() - start
    print(f"\nFetch complete: {len(rt_data)}/{total} symbols in {fetch_elapsed:.0f}s "
          f"({failed} failed)")

    # Update each CSV file
    total_updated = 0
    for csv_path in csv_files:
        file_symbols = get_symbols_from_csv(csv_path)
        file_rt_data = {s: rt_data[s] for s in file_symbols if s in rt_data}
        if not file_rt_data:
            continue

        n = update_csv_file(csv_path, file_rt_data, dry_run=dry_run)
        if n > 0:
            basename = os.path.basename(csv_path)
            print(f"  Updated {basename}: {n} symbols")
            total_updated += n

    total_elapsed = time.time() - start
    print(f"\nDone: {total_updated} symbols updated across {len(csv_files)} files "
          f"in {total_elapsed:.0f}s")

    return {
        'total_symbols': total,
        'updated': total_updated,
        'failed': failed,
        'skipped': len(skipped) if use_all_symbols else 0,
        'elapsed_seconds': total_elapsed,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Update all_data CSVs with realtime OHLCV from GLOBAL_QUOTE'
    )
    parser.add_argument(
        '--data-dir', type=str,
        default=os.environ.get('DATA_DIR', '/home/stvschmdt/data'),
        help='Directory containing all_data_*.csv files',
    )
    parser.add_argument(
        '-w', '--watchlist', type=str, default='api_data/watchlist.csv',
        help='File containing symbols to fetch. Use `all` for S&P 500 + watchlist.',
    )
    parser.add_argument(
        '-s', '--symbols', nargs='+', type=str, default=[],
        help='List of symbols to fetch data for',
    )
    parser.add_argument(
        '-t', '--test', type=int, default=600,
        help='Number of symbols for testing functionality',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Fetch data but do not write changes',
    )
    args = parser.parse_args()

    # Import here to avoid requiring API key for tests/imports
    from api_data.collector import alpha_client

    # Resolve symbol list — same logic as pull_api_data.py
    use_all = False
    if args.symbols:
        symbols_filter = set(s.upper() for s in args.symbols)
    elif args.watchlist == 'all':
        symbols_filter = None
        use_all = True
    else:
        wl = pd.read_csv(args.watchlist)
        symbols_filter = set(wl['Symbol'].str.strip().str.upper().tolist()[:args.test])

    print("=" * 60)
    print(f"RT Data Update — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if use_all:
        print("Symbol universe: SP500 + watchlist")
    elif args.symbols:
        print(f"Symbols: {', '.join(sorted(symbols_filter))}")
    else:
        print(f"Watchlist: {args.watchlist} ({len(symbols_filter)} symbols)")
    print("=" * 60)

    stats = run_rt_update(
        api_client=alpha_client,
        data_dir=args.data_dir,
        symbols_filter=symbols_filter,
        dry_run=args.dry_run,
        use_all_symbols=use_all,
    )

    print(f"\nStats: {stats}")


if __name__ == '__main__':
    main()