"""
Portfolio filtering, ranking, and selection utilities.

Library of composable ranker/filter functions to narrow down a universe
of symbols after a strategy pass. Each function accepts a list of symbols
(or a DataFrame) and returns a ranked list of symbols, optionally capped
at top_k.

Typical strategy integration:
    from portfolio import rank_by_sharpe, rank_by_price, rank_by_field, apply_watchlist

    signals = my_strategy.evaluate(...)           # -> list of symbols
    signals = rank_by_price(df, signals, above=50, below=500, top_k=30)
    signals = rank_by_sharpe(df, signals, top_k=15)
    signals = apply_watchlist(signals, watchlist, mode='filter')

CLI usage:
    python portfolio.py --data all_data_0.csv --top-k-sharpe 20
    python portfolio.py --data all_data_0.csv --price-above 50 --price-below 500 --top-k 30
    python portfolio.py --data all_data_0.csv --filter-field rsi_14 --filter-above 30 --filter-below 70 --top-k 25
    python portfolio.py --data all_data_0.csv --watchlist api_data/watchlist.csv --watchlist-mode filter
    python portfolio.py --data all_data_0.csv --top-k-sharpe 20 --price-above 25 --watchlist my_watchlist.csv

All filters are composable: apply multiple flags in one call to build a pipeline.
"""

import argparse
import glob as globmod
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load CSV data file(s). Supports glob patterns (e.g. 'all_data_*.csv').

    Handles both 'symbol' and 'ticker' column naming conventions.

    Args:
        data_path: Path to CSV file or glob pattern

    Returns:
        DataFrame with 'symbol' and 'date' columns guaranteed
    """
    if '*' in data_path:
        files = sorted(globmod.glob(data_path))
        if not files:
            raise FileNotFoundError(f"No files matching: {data_path}")
        df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    else:
        df = pd.read_csv(data_path)

    # Normalise symbol column
    if 'ticker' in df.columns and 'symbol' not in df.columns:
        df['symbol'] = df['ticker']

    # Normalise price column — all_data uses 'adjusted_close', predictions use 'close'
    if 'adjusted_close' not in df.columns and 'close' in df.columns:
        df['adjusted_close'] = df['close']

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    return df


def load_data_recent(data_path: str, n_rows: int = 5) -> pd.DataFrame:
    """
    Memory-efficient loader: reads CSV files one at a time, keeping only
    the last n_rows per symbol. Peak memory = 1 CSV file + accumulated tails.

    For live mode with --lookback-days 1, n_rows=5 gives plenty of headroom.
    """
    if '*' not in data_path:
        return load_data(data_path)

    files = sorted(globmod.glob(data_path))
    if not files:
        raise FileNotFoundError(f"No files matching: {data_path}")

    chunks = []
    for f in files:
        chunk = pd.read_csv(f)
        # Normalise symbol column
        if 'ticker' in chunk.columns and 'symbol' not in chunk.columns:
            chunk['symbol'] = chunk['ticker']
        if 'date' in chunk.columns:
            chunk['date'] = pd.to_datetime(chunk['date'])
            chunk = chunk.sort_values('date')
        # Keep only the last n_rows per symbol in this file
        chunks.append(chunk.groupby('symbol').tail(n_rows))

    df = pd.concat(chunks, ignore_index=True)

    # Normalise price column
    if 'adjusted_close' not in df.columns and 'close' in df.columns:
        df['adjusted_close'] = df['close']

    # Final dedup: keep last n_rows per symbol across all files
    if 'date' in df.columns:
        df = df.sort_values('date').groupby('symbol').tail(n_rows).reset_index(drop=True)

    return df


def load_watchlist(watchlist_path: str) -> list[str]:
    """
    Load watchlist from CSV and return ordered list of symbols.

    Accepts column names: Symbol, symbol, SYMBOL, ticker, Ticker, TICKER.
    """
    if not os.path.exists(watchlist_path):
        print(f"Warning: Watchlist file not found: {watchlist_path}")
        return []

    df = pd.read_csv(watchlist_path)

    for col in ['Symbol', 'symbol', 'SYMBOL', 'ticker', 'Ticker', 'TICKER']:
        if col in df.columns:
            return df[col].tolist()

    print("Warning: No symbol column found in watchlist")
    return []


def _get_latest(df: pd.DataFrame, symbols: list[str] = None) -> pd.DataFrame:
    """Get the most recent row per symbol, optionally filtered to a symbol list."""
    subset = df if symbols is None else df[df['symbol'].isin(set(symbols))]
    return subset.sort_values('date').groupby('symbol').tail(1)


# ---------------------------------------------------------------------------
# Watchlist filter / sort
# ---------------------------------------------------------------------------

def watchlist_filter(symbols: list[str], watchlist: list[str]) -> list[str]:
    """Keep only symbols in watchlist, preserving watchlist order."""
    symbol_set = set(symbols)
    result = [s for s in watchlist if s in symbol_set]
    print(f"Watchlist filter: {len(symbols)} -> {len(result)} symbols")
    return result


def watchlist_sort(symbols: list[str], watchlist: list[str]) -> list[str]:
    """Watchlist entries first (in order), then remaining alphabetically."""
    symbol_set = set(symbols)
    watchlist_set = set(watchlist)
    in_wl = [s for s in watchlist if s in symbol_set]
    rest = sorted(symbol_set - watchlist_set)
    print(f"Watchlist sort: {len(in_wl)} watchlist first, {len(rest)} others")
    return in_wl + rest


def apply_watchlist(symbols: list[str], watchlist: list[str], mode: str = 'sort') -> list[str]:
    """
    Apply watchlist in the requested mode.

    Args:
        symbols: Universe of symbols (e.g. strategy output)
        watchlist: Ordered watchlist
        mode: 'filter' or 'sort'

    Returns:
        Ranked list of symbols
    """
    if not watchlist:
        return symbols
    if mode == 'filter':
        return watchlist_filter(symbols, watchlist)
    return watchlist_sort(symbols, watchlist)


# ---------------------------------------------------------------------------
# Rank by price
# ---------------------------------------------------------------------------

def rank_by_price(df: pd.DataFrame, symbols: list[str] = None,
                  above: float = None, below: float = None,
                  top_k: int = None) -> list[str]:
    """
    Filter symbols by most-recent price band, return ranked list (highest first).

    Args:
        df: DataFrame with 'symbol', 'date', 'adjusted_close'
        symbols: List of symbols to consider (None = all in df)
        above: Min price inclusive (None = no lower bound)
        below: Max price inclusive (None = no upper bound)
        top_k: Return at most this many symbols (None = all that pass)

    Returns:
        List of symbols sorted by price descending, capped at top_k
    """
    if symbols is None:
        symbols = df['symbol'].unique().tolist()

    latest = _get_latest(df, symbols)

    mask = pd.Series(True, index=latest.index)
    if above is not None:
        mask &= latest['adjusted_close'] >= above
    if below is not None:
        mask &= latest['adjusted_close'] <= below

    ranked = latest.loc[mask].sort_values('adjusted_close', ascending=False)
    result = ranked['symbol'].tolist()

    if top_k is not None:
        result = result[:top_k]

    desc = []
    if above is not None:
        desc.append(f">= ${above:.2f}")
    if below is not None:
        desc.append(f"<= ${below:.2f}")
    band = ', '.join(desc) if desc else 'all'
    print(f"Price rank ({band}): {len(symbols)} -> {len(result)} symbols" +
          (f" (top {top_k})" if top_k else ""))
    return result


# ---------------------------------------------------------------------------
# Rank by field (generic)
# ---------------------------------------------------------------------------

def rank_by_field(df: pd.DataFrame, field: str, symbols: list[str] = None,
                  above: float = None, below: float = None,
                  ascending: bool = False, top_k: int = None) -> list[str]:
    """
    Filter + rank symbols by any numeric field in the data.

    Args:
        df: DataFrame with 'symbol', 'date', and the target field
        field: Column name (e.g. 'rsi_14', 'beta', 'pe_ratio', 'market_capitalization')
        symbols: List of symbols to consider (None = all in df)
        above: Min value inclusive (None = no lower bound)
        below: Max value inclusive (None = no upper bound)
        ascending: Sort order (False = highest first, True = lowest first)
        top_k: Return at most this many symbols (None = all that pass)

    Returns:
        List of symbols ranked by field value, capped at top_k
    """
    if symbols is None:
        symbols = df['symbol'].unique().tolist()

    if field not in df.columns:
        print(f"Warning: field '{field}' not found. Numeric columns:")
        for c in sorted(df.select_dtypes(include=[np.number]).columns):
            print(f"  {c}")
        return symbols[:top_k] if top_k else symbols

    latest = _get_latest(df, symbols)

    mask = pd.Series(True, index=latest.index)
    if above is not None:
        mask &= latest[field] >= above
    if below is not None:
        mask &= latest[field] <= below

    ranked = latest.loc[mask].sort_values(field, ascending=ascending)
    result = ranked['symbol'].tolist()

    if top_k is not None:
        result = result[:top_k]

    direction = 'asc' if ascending else 'desc'
    before = len(symbols)
    print(f"Field rank ({field} {direction}): {before} -> {len(result)} symbols" +
          (f" (top {top_k})" if top_k else ""))
    return result


# ---------------------------------------------------------------------------
# Rank by Sharpe ratio
# ---------------------------------------------------------------------------

def compute_sharpe(df: pd.DataFrame, symbols: list[str] = None,
                   risk_free_rate: float = 0.0) -> pd.DataFrame:
    """
    Compute annualised Sharpe ratio per symbol from daily returns.

    Uses 'adjusted_close_pct' if available, otherwise computes daily
    returns from 'adjusted_close'.

    Args:
        df: DataFrame with 'symbol', 'date', and price/return data
        symbols: Restrict to these symbols (None = all)
        risk_free_rate: Annualised risk-free rate (default 0)

    Returns:
        DataFrame [symbol, sharpe, mean_return, std_return, n_days]
        sorted by sharpe descending
    """
    subset = df if symbols is None else df[df['symbol'].isin(set(symbols))]
    records = []
    daily_rf = risk_free_rate / 252

    for symbol, grp in subset.groupby('symbol'):
        grp = grp.sort_values('date')

        if 'adjusted_close_pct' in grp.columns:
            returns = grp['adjusted_close_pct'].dropna() / 100.0
        elif 'adjusted_close' in grp.columns:
            returns = grp['adjusted_close'].pct_change().dropna()
        else:
            continue

        if len(returns) < 20:
            continue

        excess = returns - daily_rf
        mean_r = excess.mean()
        std_r = excess.std()
        sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0.0

        records.append({
            'symbol': symbol,
            'sharpe': round(sharpe, 4),
            'mean_return': round(mean_r * 252, 4),
            'std_return': round(std_r * np.sqrt(252), 4),
            'n_days': len(returns),
        })

    if not records:
        return pd.DataFrame(columns=['symbol', 'sharpe', 'mean_return', 'std_return', 'n_days'])
    return pd.DataFrame(records).sort_values('sharpe', ascending=False).reset_index(drop=True)


def rank_by_sharpe(df: pd.DataFrame, symbols: list[str] = None,
                   top_k: int = None, risk_free_rate: float = 0.0) -> list[str]:
    """
    Rank symbols by Sharpe ratio.

    Args:
        df: DataFrame with price/return data
        symbols: List of symbols to consider (None = all in df)
        top_k: Number of top symbols to keep (None = keep all, sorted)
        risk_free_rate: Annualised risk-free rate

    Returns:
        List of symbols ranked by Sharpe ratio (highest first)
    """
    if symbols is None:
        symbols = df['symbol'].unique().tolist()

    sharpe_df = compute_sharpe(df, symbols, risk_free_rate)

    if top_k is not None:
        display_df = sharpe_df.head(top_k)
        label = f"Top {top_k}"
    else:
        display_df = sharpe_df
        label = "All (sorted)"

    result = display_df['symbol'].tolist()

    print(f"\n{label} by Sharpe ratio:")
    print(f"  {'Symbol':<8} {'Sharpe':>8} {'Ann Ret':>10} {'Ann Vol':>10} {'Days':>6}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*6}")
    for _, row in display_df.iterrows():
        print(f"  {row['symbol']:<8} {row['sharpe']:>8.4f} {row['mean_return']:>9.4f} {row['std_return']:>10.4f} {row['n_days']:>6}")

    print(f"\nSharpe rank: {len(symbols)} -> {len(result)} symbols ({label.lower()})")
    return result


# ---------------------------------------------------------------------------
# Rank by backtest rankings CSV
# ---------------------------------------------------------------------------

def rank_by_backtest(symbols: list[str], rankings_path: str,
                     rank_field: str = 'score', top_k: int = None) -> list[str]:
    """
    Filter and rank symbols using a backtest rankings CSV.

    If the file is missing or any error occurs, returns symbols unchanged
    (pass-through identity behavior).

    Args:
        symbols: List of symbols to filter/rank
        rankings_path: Path to backtest_rankings.csv
        rank_field: Column to sort by (win_rate, avg_return_pct, total_pnl,
                    score, trades)
        top_k: Number of top symbols to keep (None = keep all, sorted)

    Returns:
        List of symbols ranked by rank_field descending, capped at top_k
    """
    if not symbols:
        return symbols

    try:
        if not os.path.exists(rankings_path):
            print(f"Warning: Rankings file not found: {rankings_path} (pass-through)")
            return symbols

        df = pd.read_csv(rankings_path)

        if 'symbol' not in df.columns:
            print(f"Warning: No 'symbol' column in {rankings_path} (pass-through)")
            return symbols

        if rank_field not in df.columns:
            print(f"Warning: Field '{rank_field}' not in rankings. "
                  f"Available: {', '.join(df.columns.tolist())} (pass-through)")
            return symbols

        # Filter to only symbols in our universe
        symbol_set = set(symbols)
        df = df[df['symbol'].isin(symbol_set)]

        # Sort descending by rank_field (higher is better)
        df = df.sort_values(rank_field, ascending=False)

        result = df['symbol'].tolist()

        if top_k is not None:
            result = result[:top_k]

        # Note symbols not in rankings (they get dropped)
        missing = symbol_set - set(df['symbol'])
        if missing:
            print(f"  Note: {len(missing)} symbols not in rankings (dropped): "
                  f"{', '.join(sorted(missing)[:5])}"
                  f"{'...' if len(missing) > 5 else ''}")

        label = f"top {top_k}" if top_k else "sorted"
        print(f"Backtest rank ({rank_field}, {label}): "
              f"{len(symbols)} -> {len(result)} symbols")
        return result

    except Exception as e:
        print(f"Warning: Error reading rankings: {e} (pass-through)")
        return symbols


# ---------------------------------------------------------------------------
# Pipeline: chain all rankers
# ---------------------------------------------------------------------------

def run_pipeline(df: pd.DataFrame,
                 symbols: list[str] = None,
                 watchlist: list[str] = None,
                 watchlist_mode: str = 'sort',
                 price_above: float = None,
                 price_below: float = None,
                 price_top_k: int = None,
                 filter_field: str = None,
                 filter_above: float = None,
                 filter_below: float = None,
                 filter_ascending: bool = False,
                 filter_top_k: int = None,
                 top_k_sharpe: int = None,
                 sort_sharpe: bool = False,
                 risk_free_rate: float = 0.0,
                 backtest_rankings: str = None,
                 rank_by: str = 'score',
                 rank_top_k: int = None) -> list[str]:
    """
    Run the full filter/rank pipeline. Order:
      1. Watchlist filter/sort
      2. Price rank
      3. Field rank
      4. Sharpe rank
      5. Backtest ranking

    Each step receives the symbol list from the previous step.

    Args:
        df: Source DataFrame
        symbols: Starting list of symbols (None = all in df)
        watchlist: Ordered watchlist (optional)
        watchlist_mode: 'filter' or 'sort'
        price_above / price_below / price_top_k: Price filter params
        filter_field / filter_above / filter_below / filter_ascending / filter_top_k: Field filter params
        top_k_sharpe: Top k by Sharpe (optional)
        risk_free_rate: Risk-free rate for Sharpe
        backtest_rankings: Path to backtest_rankings.csv (optional)
        rank_by: Field to rank by from rankings CSV (default: 'score')
        rank_top_k: Keep top K from backtest rankings (optional)

    Returns:
        Final ranked list of symbols
    """
    if symbols is None:
        symbols = df['symbol'].unique().tolist()

    print(f"\nStarting universe: {len(symbols)} symbols")
    print("=" * 60)

    # 1. Watchlist
    if watchlist:
        symbols = apply_watchlist(symbols, watchlist, watchlist_mode)

    # 2. Price rank
    if price_above is not None or price_below is not None:
        symbols = rank_by_price(df, symbols, above=price_above, below=price_below, top_k=price_top_k)

    # 3. Field rank
    if filter_field is not None:
        symbols = rank_by_field(df, filter_field, symbols,
                                above=filter_above, below=filter_below,
                                ascending=filter_ascending, top_k=filter_top_k)

    # 4. Sharpe rank
    if top_k_sharpe is not None:
        symbols = rank_by_sharpe(df, symbols, top_k=top_k_sharpe, risk_free_rate=risk_free_rate)
    elif sort_sharpe:
        symbols = rank_by_sharpe(df, symbols, top_k=None, risk_free_rate=risk_free_rate)

    # 5. Backtest ranking
    if backtest_rankings is not None:
        symbols = rank_by_backtest(symbols, backtest_rankings,
                                   rank_field=rank_by, top_k=rank_top_k)

    print("=" * 60)
    print(f"Final universe: {len(symbols)} symbols\n")
    return symbols


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, symbols: list[str]):
    """Print a concise summary table for the ranked symbols."""
    latest = _get_latest(df, symbols)
    # Maintain the ranking order from symbols list
    order = {s: i for i, s in enumerate(symbols)}
    latest = latest.copy()
    latest['_rank'] = latest['symbol'].map(order)
    latest = latest.sort_values('_rank')

    print(f"\n  {'#':<4} {'Symbol':<8} {'Price':>10} {'RSI':>8} {'ADX':>8} {'Beta':>8} {'Sector'}")
    print(f"  {'-'*4} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*20}")

    for rank, (_, row) in enumerate(latest.iterrows(), 1):
        price = row.get('adjusted_close', row.get('close', 0))
        rsi = row.get('rsi_14', np.nan)
        adx = row.get('adx_14', np.nan)
        beta = row.get('beta', np.nan)
        sector = row.get('sector', '')

        rsi_str = f"{rsi:.1f}" if pd.notna(rsi) else 'N/A'
        adx_str = f"{adx:.1f}" if pd.notna(adx) else 'N/A'
        beta_str = f"{beta:.2f}" if pd.notna(beta) else 'N/A'
        sector_str = str(sector)[:20] if pd.notna(sector) else ''

        print(f"  {rank:<4} {row['symbol']:<8} ${price:>8.2f} {rsi_str:>8} {adx_str:>8} {beta_str:>8} {sector_str}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Portfolio filtering, ranking, and selection utilities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Top 20 symbols by Sharpe ratio
  python portfolio.py --data all_data_0.csv --top-k-sharpe 20

  # Price band $50-$500, keep top 30
  python portfolio.py --data all_data_0.csv --price-above 50 --price-below 500 --top-k 30

  # Filter by RSI between 30 and 70, keep top 25
  python portfolio.py --data all_data_0.csv --filter-field rsi_14 --filter-above 30 --filter-below 70 --top-k 25

  # Watchlist filter + top 10 Sharpe
  python portfolio.py --data all_data_0.csv --watchlist api_data/watchlist.csv --watchlist-mode filter --top-k-sharpe 10

  # Combine: price > $25, beta < 1.5, top 15 by Sharpe
  python portfolio.py --data all_data_0.csv --price-above 25 --filter-field beta --filter-below 1.5 --top-k-sharpe 15
        """
    )

    parser.add_argument('--data', type=str, required=True,
                        help='Path to CSV data file (all_data_*.csv or predictions_*.csv)')
    parser.add_argument('--watchlist', type=str, default=None,
                        help='Path to watchlist CSV file')
    parser.add_argument('--watchlist-mode', type=str, default='sort',
                        choices=['sort', 'filter'],
                        help="'sort' = watchlist first, 'filter' = watchlist only (default: sort)")
    parser.add_argument('--price-above', type=float, default=None,
                        help='Minimum stock price (inclusive)')
    parser.add_argument('--price-below', type=float, default=None,
                        help='Maximum stock price (inclusive)')
    parser.add_argument('--top-k', type=int, default=None,
                        help='Limit price/field filter results to top K')
    parser.add_argument('--filter-field', type=str, default=None,
                        help='Column name to filter/rank on (e.g. rsi_14, beta, pe_ratio)')
    parser.add_argument('--filter-above', type=float, default=None,
                        help='Minimum value for --filter-field (inclusive)')
    parser.add_argument('--filter-below', type=float, default=None,
                        help='Maximum value for --filter-field (inclusive)')
    parser.add_argument('--filter-ascending', action='store_true',
                        help='Sort field filter ascending (lowest first) instead of descending')
    parser.add_argument('--top-k-sharpe', type=int, default=None,
                        help='Keep top K symbols ranked by Sharpe ratio')
    parser.add_argument('--risk-free-rate', type=float, default=0.0,
                        help='Annualised risk-free rate for Sharpe (default: 0.0)')
    parser.add_argument('--summary', action='store_true',
                        help='Print a summary table of the final universe')
    parser.add_argument('--output', type=str, default=None,
                        help='Save final symbol list to file (one per line)')

    args = parser.parse_args()

    # Load
    print(f"Loading {args.data}...")
    data = load_data(args.data)

    wl = load_watchlist(args.watchlist) if args.watchlist else None

    # Run pipeline
    result_symbols = run_pipeline(
        data,
        watchlist=wl,
        watchlist_mode=args.watchlist_mode,
        price_above=args.price_above,
        price_below=args.price_below,
        price_top_k=args.top_k,
        filter_field=args.filter_field,
        filter_above=args.filter_above,
        filter_below=args.filter_below,
        filter_ascending=args.filter_ascending,
        filter_top_k=args.top_k,
        top_k_sharpe=args.top_k_sharpe,
        risk_free_rate=args.risk_free_rate,
    )

    if args.summary:
        print_summary(data, result_symbols)

    if args.output:
        out_df = pd.DataFrame({'symbol': result_symbols})
        out_df.to_csv(args.output, index=False)
        print(f"Saved {len(result_symbols)} symbols to {args.output}")