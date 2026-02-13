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
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Trading calendar helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_previous_trading_day(today: pd.Timestamp) -> pd.Timestamp:
    """Get the most recent trading day before today, skipping weekends and US market holidays."""
    us_holidays = USFederalHolidayCalendar().holidays(
        start=today - pd.Timedelta(days=10),
        end=today,
    )
    cbday = CustomBusinessDay(holidays=us_holidays)
    return (today - cbday).normalize()


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

    # Drop temp column before saving
    df.drop(columns=['_parsed_date'], inplace=True, errors='ignore')

    if not dry_run and updated > 0:
        df.to_csv(csv_path, index=False)

    return updated


# ─────────────────────────────────────────────────────────────────────────────
# Bulk RT update (REALTIME_BULK_QUOTES — 1 API call per 100 symbols)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_columns(df):
    """Detect symbol and close column names for a DataFrame."""
    sym_col = 'ticker' if 'ticker' in df.columns else 'symbol'
    close_col = 'close' if ('close' in df.columns and
                            'adjusted_close' not in df.columns) else 'adjusted_close'
    return sym_col, close_col


def update_derived_columns_generic(df, idx, sym_col='symbol', close_col='adjusted_close'):
    """
    Recompute derived columns for a specific row after RT update.

    Works with both all_data_*.csv (symbol/adjusted_close) and
    merged_predictions.csv (ticker/close) column conventions.

    Args:
        df: Full DataFrame (modified in place)
        idx: Row index to update
        sym_col: Name of the symbol column
        close_col: Name of the close price column
    """
    # Use .at[] for scalar access (safe even with duplicate indices)
    try:
        price = df.at[idx, close_col]
    except Exception:
        return
    if pd.isna(price) or price == 0:
        return

    # SMA distance percentages
    for period in [20, 50, 200]:
        sma_col = f'sma_{period}'
        pct_col = f'sma_{period}_pct'
        if sma_col in df.columns and pct_col in df.columns:
            sma_val = df.at[idx, sma_col]
            if pd.notna(sma_val) and sma_val != 0:
                df.at[idx, pct_col] = round((price - sma_val) / sma_val * 100, 2)

    # 52-week high/low pct
    for metric in ['52_week_high', '52_week_low']:
        pct_col = f'{metric}_pct'
        if metric in df.columns and pct_col in df.columns:
            val = df.at[idx, metric]
            if pd.notna(val) and val != 0:
                df.at[idx, pct_col] = round((price - val) / val * 100, 2)

    # PE ratio
    if 'ttm_eps' in df.columns and 'pe_ratio' in df.columns:
        ttm = df.at[idx, 'ttm_eps']
        if pd.notna(ttm) and ttm != 0:
            df.at[idx, 'pe_ratio'] = price / ttm

    # Pct change vs previous row for same symbol
    symbol = df.at[idx, sym_col]
    if symbol is not None:
        sym_mask = df[sym_col] == symbol
        sym_indices = df.loc[sym_mask].index
        pos = sym_indices.get_loc(idx)
        if pos > 0:
            prev_idx = sym_indices[pos - 1]
            prev_close = df.at[prev_idx, close_col]
            if 'adjusted_close_pct' in df.columns and pd.notna(prev_close) and prev_close != 0:
                df.at[idx, 'adjusted_close_pct'] = (price - prev_close) / prev_close

            prev_vol = df.at[prev_idx, 'volume']
            if 'volume_pct' in df.columns and pd.notna(prev_vol) and prev_vol != 0:
                vol = df.at[idx, 'volume']
                df.at[idx, 'volume_pct'] = (vol - prev_vol) / prev_vol

            # OHLC pct change vs previous row
            for ohlc_col in ['open', 'high', 'low']:
                pct_col = f'{ohlc_col}_pct'
                if pct_col in df.columns and ohlc_col in df.columns:
                    prev_val = df.at[prev_idx, ohlc_col]
                    cur_val = df.at[idx, ohlc_col]
                    if pd.notna(prev_val) and prev_val != 0 and pd.notna(cur_val):
                        df.at[idx, pct_col] = (cur_val - prev_val) / prev_val


def update_csv_from_bulk(csv_path, bulk_df, dry_run=False):
    """
    Update (or create) today's row per symbol in a CSV using bulk quote data.

    Handles both all_data_*.csv (symbol/adjusted_close) and
    merged_predictions.csv (ticker/close) column conventions.

    If a symbol's latest row is already today, updates it in place.
    Otherwise, clones the latest row, sets date=today, updates OHLCV,
    and appends — so strategies always see a fresh row dated today.

    Args:
        csv_path: Path to CSV file
        bulk_df: DataFrame with columns: symbol, price, volume
        dry_run: If True, print changes but don't write

    Returns:
        Number of symbols updated
    """
    if bulk_df.empty:
        return 0

    df = pd.read_csv(csv_path)
    # Drop unnamed index columns, use clean integer index to avoid dupes
    unnamed_cols = [c for c in df.columns if c.startswith('Unnamed')]
    if unnamed_cols:
        df.drop(columns=unnamed_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)

    sym_col, close_col = _detect_columns(df)

    # Parse dates for latest-row detection
    has_date = 'date' in df.columns
    if has_date:
        df['_parsed_date'] = pd.to_datetime(df['date'], errors='coerce')
    else:
        df['_parsed_date'] = pd.NaT

    today = pd.Timestamp.now().normalize()  # midnight today
    today_str = today.strftime('%Y-%m-%d')

    # Build lookup from bulk quotes
    quotes = {}
    for _, row in bulk_df.iterrows():
        sym = row['symbol']
        quotes[sym] = {
            'price': row['price'],
            'volume': row.get('volume', 0),
            'open': row.get('open', 0),
            'high': row.get('high', 0),
            'low': row.get('low', 0),
        }

    # Compute expected last trading day for staleness check
    # Accounts for weekends AND US market holidays (e.g., MLK Day, Presidents Day)
    expected_last_trading_day = get_previous_trading_day(today)

    new_rows = []
    updated = 0
    stale_symbols = []
    for symbol, qdata in quotes.items():
        price = qdata['price']
        volume = qdata['volume']

        if price <= 0:
            continue

        sym_mask = df[sym_col] == symbol
        if not sym_mask.any():
            continue

        # Find latest row by date
        sym_rows = df.loc[sym_mask]
        if sym_rows['_parsed_date'].notna().any():
            latest_idx = sym_rows['_parsed_date'].idxmax()
        else:
            latest_idx = sym_rows.index[-1]

        latest_date = df.at[latest_idx, '_parsed_date']

        # If latest row is NOT today, clone it and append as today
        if has_date and pd.notna(latest_date) and latest_date.normalize() < today:
            # Staleness check: source row must be from the last trading day
            source_date = latest_date.normalize()
            if source_date < expected_last_trading_day:
                days_stale = (expected_last_trading_day - source_date).days
                stale_symbols.append((symbol, source_date.strftime('%Y-%m-%d'), days_stale))
                logger.warning(
                    f"STALE DATA: {symbol} last row is {source_date.strftime('%Y-%m-%d')} "
                    f"({days_stale}d older than expected {expected_last_trading_day.strftime('%Y-%m-%d')}). "
                    f"Skipping clone — run nightly pipeline first."
                )
                print(
                    f"  WARNING: {symbol} data stale "
                    f"(last={source_date.strftime('%Y-%m-%d')}, "
                    f"expected>={expected_last_trading_day.strftime('%Y-%m-%d')}). "
                    f"Skipping row clone."
                )
                continue
            new_row = df.loc[latest_idx].copy()
            new_row['date'] = today_str
            new_row['_parsed_date'] = today
            # Set OHLCV on the new row
            new_row[close_col] = price
            if qdata['open'] > 0:
                new_row['open'] = qdata['open']
            else:
                new_row['open'] = price  # First RT quote of the day = open
            if qdata['high'] > 0:
                new_row['high'] = qdata['high']
            else:
                new_row['high'] = price
            if qdata['low'] > 0:
                new_row['low'] = qdata['low']
            else:
                new_row['low'] = price
            if volume > 0:
                new_row['volume'] = volume

            # Fix calendar columns for the new date (don't clone yesterday's day-of-week)
            new_date = pd.to_datetime(today_str)
            if 'day_of_week_num' in df.columns:
                new_row['day_of_week_num'] = new_date.dayofweek
            if 'day_of_week_name' in df.columns:
                new_row['day_of_week_name'] = new_date.day_name()
            if 'month' in df.columns:
                new_row['month'] = new_date.month
            if 'day_of_year' in df.columns:
                new_row['day_of_year'] = new_date.dayofyear
            if 'year' in df.columns:
                new_row['year'] = new_date.year

            # Fix adjusted OHLC to match today's values (not yesterday's)
            if 'adjusted_open' in df.columns:
                new_row['adjusted_open'] = new_row['open']
            if 'adjusted_high' in df.columns:
                new_row['adjusted_high'] = new_row['high']
            if 'adjusted_low' in df.columns:
                new_row['adjusted_low'] = new_row['low']

            new_rows.append(new_row)
            updated += 1
            if dry_run:
                print(f"  [DRY RUN] {symbol}: NEW row date={today_str}, "
                      f"{close_col}=${price:.2f}")
        else:
            # Today's row already exists — update in place
            df.at[latest_idx, close_col] = price

            if 'open' in df.columns and qdata['open'] > 0:
                df.at[latest_idx, 'open'] = qdata['open']

            if 'high' in df.columns:
                if qdata['high'] > 0:
                    df.at[latest_idx, 'high'] = qdata['high']
                else:
                    existing_high = df.at[latest_idx, 'high']
                    if pd.notna(existing_high):
                        df.at[latest_idx, 'high'] = max(existing_high, price)
                    else:
                        df.at[latest_idx, 'high'] = price

            if 'low' in df.columns:
                if qdata['low'] > 0:
                    df.at[latest_idx, 'low'] = qdata['low']
                else:
                    existing_low = df.at[latest_idx, 'low']
                    if pd.notna(existing_low):
                        df.at[latest_idx, 'low'] = min(existing_low, price)
                    else:
                        df.at[latest_idx, 'low'] = price

            if volume > 0 and 'volume' in df.columns:
                df.at[latest_idx, 'volume'] = volume

            # Recompute derived columns
            update_derived_columns_generic(df, latest_idx, sym_col, close_col)

            updated += 1
            if dry_run:
                print(f"  [DRY RUN] {symbol}: UPDATE date={today_str}, "
                      f"{close_col}=${price:.2f}")

    # Append new today rows
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        # Recompute derived columns for the newly appended rows
        for i in range(len(df) - len(new_rows), len(df)):
            update_derived_columns_generic(df, i, sym_col, close_col)

    # Summary of stale symbols
    if stale_symbols:
        print(f"\n  === STALE DATA SUMMARY: {len(stale_symbols)} symbols skipped ===")
        for sym, last_date, days in stale_symbols:
            print(f"    {sym}: last data {last_date} ({days}d stale)")
        print(f"  Run nightly pipeline to refresh before RT scanning.\n")
        logger.warning(
            f"Skipped {len(stale_symbols)} stale symbols: "
            + ", ".join(s[0] for s in stale_symbols[:10])
            + ("..." if len(stale_symbols) > 10 else "")
        )

    # Drop temporary column before saving
    df.drop(columns=['_parsed_date'], inplace=True, errors='ignore')

    if not dry_run and updated > 0:
        df.to_csv(csv_path, index=False)

    return updated


def _fetch_individual_quotes(symbols_list):
    """
    Fallback: fetch quotes one-by-one via GLOBAL_QUOTE.

    Returns DataFrame with columns: symbol, price, volume (same shape as bulk).
    Also returns full OHLCV in an rt_data dict for update_csv_file().
    """
    from api_data.collector import alpha_client

    rows = []
    rt_data = {}
    total = len(symbols_list)
    failed = 0
    start = time.time()

    for i, symbol in enumerate(symbols_list, 1):
        elapsed = time.time() - start
        rate = elapsed / i if i > 1 else 0
        eta = rate * (total - i)
        print(f"  [{i}/{total}] {symbol} "
              f"(elapsed: {elapsed:.0f}s, eta: {eta:.0f}s) ...", end='', flush=True)

        data = fetch_symbol_rt_data(alpha_client, symbol)
        if data and data['ohlcv'].get('adjusted_close', 0) > 0:
            ohlcv = data['ohlcv']
            rows.append({
                'symbol': symbol,
                'open': ohlcv.get('open', 0),
                'high': ohlcv.get('high', 0),
                'low': ohlcv.get('low', 0),
                'price': ohlcv['adjusted_close'],
                'volume': ohlcv.get('volume', 0),
            })
            rt_data[symbol] = data
            print(f" ${ohlcv['adjusted_close']:.2f}")
        else:
            failed += 1
            print(" FAILED")

    print(f"  Fetched {len(rows)}/{total} quotes ({failed} failed)")

    # Second pass: retry failed symbols to maximize coverage
    if failed > 0:
        fetched_syms = {r['symbol'] for r in rows}
        retry_syms = [s for s in symbols_list if s not in fetched_syms]
        print(f"  Retry pass: {len(retry_syms)} symbols...")
        retry_ok = 0
        for symbol in retry_syms:
            data = fetch_symbol_rt_data(alpha_client, symbol)
            if data and data['ohlcv'].get('adjusted_close', 0) > 0:
                ohlcv = data['ohlcv']
                rows.append({
                    'symbol': symbol,
                    'open': ohlcv.get('open', 0),
                    'high': ohlcv.get('high', 0),
                    'low': ohlcv.get('low', 0),
                    'price': ohlcv['adjusted_close'],
                    'volume': ohlcv.get('volume', 0),
                })
                rt_data[symbol] = data
                retry_ok += 1
        print(f"  Retry recovered {retry_ok}/{len(retry_syms)} symbols "
              f"(total: {len(rows)}/{total})")

    bulk_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return bulk_df, rt_data


def run_bulk_rt_update(data_dir, predictions_path=None, symbols_filter=None,
                       dry_run=False):
    """
    Fetch RT quotes and update CSV files on disk.

    Tries REALTIME_BULK_QUOTES first (1 API call per 100 symbols).
    Falls back to individual GLOBAL_QUOTE if bulk returns zeros (no premium key).

    Call this BEFORE running strategies so they see fresh prices.

    Args:
        data_dir: Directory containing all_data_*.csv files
        predictions_path: Optional path to merged_predictions.csv
        symbols_filter: Optional set to restrict symbols
        dry_run: If True, don't write changes

    Returns:
        Dict with stats: total_symbols, quotes_received, updated, elapsed_seconds
    """
    start = time.time()

    # Collect all symbols from CSVs
    all_symbols = set()
    csv_files = find_csv_files(data_dir)
    for csv_path in csv_files:
        all_symbols.update(get_symbols_from_csv(csv_path))

    # Also collect from predictions if provided
    if predictions_path and os.path.exists(predictions_path):
        try:
            pred_df = pd.read_csv(predictions_path)
            if not pred_df.empty:
                ticker_col = 'ticker' if 'ticker' in pred_df.columns else 'symbol'
                if ticker_col in pred_df.columns:
                    all_symbols.update(pred_df[ticker_col].unique())
        except Exception:
            pass  # predictions file missing or malformed — skip

    if symbols_filter:
        all_symbols = all_symbols & set(symbols_filter)

    symbols_list = sorted(all_symbols)
    n_calls = (len(symbols_list) + 99) // 100
    print(f"Bulk RT update: {len(symbols_list)} symbols, {n_calls} API call(s)...")

    # Try bulk endpoint first
    from api_data.rt_utils import get_realtime_quotes_bulk
    bulk_df = get_realtime_quotes_bulk(symbols_list)

    # Check if bulk returned real data (not sample/demo)
    # Sample data has valid prices but wrong symbols — must detect and fall back
    rt_data = None
    is_sample = False
    if not bulk_df.empty:
        # Check if symbols match what we requested (sample returns fixed demo symbols)
        returned_syms = set(bulk_df['symbol'].unique())
        requested_syms = set(symbols_list)
        unrequested = returned_syms - requested_syms
        is_sample = len(unrequested) > 0  # Any unrequested symbol = sample data

        valid_prices = bulk_df[bulk_df['price'] > 0]
        if len(valid_prices) > 0 and not is_sample:
            print(f"  Bulk quotes: {len(valid_prices)} valid prices")
            bulk_df = valid_prices
        elif is_sample:
            print(f"  Bulk returned sample/demo data — falling back to individual GLOBAL_QUOTE")
            bulk_df, rt_data = _fetch_individual_quotes(symbols_list)
        else:
            print(f"  Bulk quotes returned all zeros — falling back to individual GLOBAL_QUOTE")
            bulk_df, rt_data = _fetch_individual_quotes(symbols_list)
    else:
        print(f"  Bulk endpoint returned empty — falling back to individual GLOBAL_QUOTE")
        bulk_df, rt_data = _fetch_individual_quotes(symbols_list)

    if bulk_df.empty:
        print("  No quotes received")
        return {'total_symbols': len(symbols_list), 'quotes_received': 0,
                'updated': 0, 'elapsed_seconds': time.time() - start}

    print(f"  Got quotes for {len(bulk_df)} symbols")

    # Update all_data_*.csv files using update_csv_from_bulk (handles today-row creation)
    total_updated = 0
    for csv_path in csv_files:
        n = update_csv_from_bulk(csv_path, bulk_df, dry_run=dry_run)
        if n > 0:
            print(f"  Updated {os.path.basename(csv_path)}: {n} symbols")
            total_updated += n

    # Update predictions file if provided
    if predictions_path and os.path.exists(predictions_path):
        n = update_csv_from_bulk(predictions_path, bulk_df, dry_run=dry_run)
        if n > 0:
            print(f"  Updated {os.path.basename(predictions_path)}: {n} symbols")
            total_updated += n

    elapsed = time.time() - start
    file_count = len(csv_files) + (1 if predictions_path else 0)
    print(f"Bulk RT update done: {total_updated} symbols across "
          f"{file_count} files in {elapsed:.1f}s")

    return {
        'total_symbols': len(symbols_list),
        'quotes_received': len(bulk_df),
        'updated': total_updated,
        'elapsed_seconds': elapsed,
    }


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
    # Read just the header to detect column layout
    header = pd.read_csv(csv_path, nrows=0)
    if 'symbol' in header.columns:
        df = pd.read_csv(csv_path, usecols=['symbol'])
        return df['symbol'].unique().tolist()
    elif 'ticker' in header.columns:
        df = pd.read_csv(csv_path, usecols=['ticker'])
        return df['ticker'].unique().tolist()
    else:
        # Legacy format with unnamed index column
        df = pd.read_csv(csv_path, index_col=0, usecols=[0, 2])
        return df.iloc[:, 0].unique().tolist()


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
        default=os.environ.get('DATA_DIR', '.'),
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
    parser.add_argument(
        '--bulk', action='store_true',
        help='Use REALTIME_BULK_QUOTES (1 call per 100 symbols, faster)',
    )
    parser.add_argument(
        '--predictions', type=str, default=None,
        help='Also update merged_predictions.csv with RT prices',
    )
    args = parser.parse_args()

    # Resolve symbol filter
    symbols_filter = None
    use_all = False
    if args.symbols:
        symbols_filter = set(s.upper() for s in args.symbols)
    elif args.watchlist == 'all':
        use_all = True
    else:
        wl = pd.read_csv(args.watchlist)
        symbols_filter = set(wl['Symbol'].str.strip().str.upper().tolist()[:args.test])

    print("=" * 60)
    print(f"RT Data Update — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    mode_str = "BULK" if args.bulk else "INDIVIDUAL"
    if use_all:
        print(f"Symbol universe: SP500 + watchlist ({mode_str})")
    elif args.symbols:
        print(f"Symbols: {', '.join(sorted(symbols_filter))} ({mode_str})")
    else:
        print(f"Watchlist: {args.watchlist} ({len(symbols_filter)} symbols, {mode_str})")
    if args.predictions:
        print(f"Predictions: {args.predictions}")
    print("=" * 60)

    if args.bulk:
        stats = run_bulk_rt_update(
            data_dir=args.data_dir,
            predictions_path=args.predictions,
            symbols_filter=symbols_filter,
            dry_run=args.dry_run,
        )
    else:
        # Import here to avoid requiring API key for tests/imports
        from api_data.collector import alpha_client
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