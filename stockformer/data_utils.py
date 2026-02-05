"""
Data loading utilities for the StockFormer pipeline.

Functions:
    - load_panel_csvs: Load and concatenate stock panel CSVs
    - add_future_returns: Compute future return targets
    - create_dummy_embeddings: Auto-create embeddings if missing
    - load_embeddings: Load embedding CSVs
    - merge_embeddings: Merge embeddings onto main dataframe
"""

import os
from glob import glob
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CSV Loading
# =============================================================================

def load_panel_csvs(path_pattern: str) -> pd.DataFrame:
    """
    Load panel CSVs given a path or glob pattern.

    - If path_pattern is a directory: look for "all_data_*.csv" in it.
    - Else: treat path_pattern as a glob.
    - Fallback: if glob has no matches but path_pattern is a file, load it directly.

    Returns concatenated DataFrame.
    """
    if os.path.isdir(path_pattern):
        pattern = os.path.join(path_pattern, "all_data_*.csv")
        paths = sorted(glob(pattern))
    else:
        paths = sorted(glob(path_pattern))
        if len(paths) == 0 and os.path.isfile(path_pattern):
            paths = [path_pattern]

    if not paths:
        raise FileNotFoundError(f"No CSV files found for pattern: {path_pattern}")

    print(f"Loading {len(paths)} CSVs:")
    for p in paths:
        print(f"  - {p}")

    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)

    # Map common column name variations to expected names
    column_map = {}
    if "symbol" in df.columns and "ticker" not in df.columns:
        column_map["symbol"] = "ticker"
    if "adjusted_close" in df.columns and "close" not in df.columns:
        column_map["adjusted_close"] = "close"
    if column_map:
        df = df.rename(columns=column_map)
        print(f"Renamed columns: {column_map}")

    print(f"Loaded combined shape: {df.shape}")
    return df


# =============================================================================
# Future Returns (Target Variables)
# =============================================================================

def add_future_returns(
    df: pd.DataFrame,
    horizons: List[int],
    price_col: str = "close",
    group_col: str = "ticker",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Add future return columns for each horizon h in horizons.

    Creates columns like:
        future_3_day_pct = (price[t+3] / price[t]) - 1
        future_10_day_pct = (price[t+10] / price[t]) - 1
    """
    df = df.sort_values([group_col, date_col]).reset_index(drop=True)

    for h in horizons:
        col_name = f"future_{h}_day_pct"
        df[col_name] = (
            df.groupby(group_col)[price_col]
            .shift(-h)
            .div(df[price_col])
            .sub(1.0)
        )

    return df


# =============================================================================
# Embeddings
# =============================================================================

def create_dummy_embeddings(
    path: str,
    base_df: pd.DataFrame,
    emb_dim: int = 16,
    key_cols: Tuple[str, str] = ("ticker", "date"),
) -> None:
    """
    Create a simple random embedding CSV if not present.

    - One row per (ticker, date) combination found in base_df.
    - Key columns: "ticker", "date"
    - Embedding columns: e0, e1, ..., e{emb_dim-1}
    """
    ticker_col, date_col = key_cols
    if ticker_col not in base_df.columns or date_col not in base_df.columns:
        raise ValueError(
            f"Cannot auto-create embeddings: base_df must contain '{ticker_col}' and '{date_col}'"
        )

    tickers = sorted(base_df[ticker_col].dropna().unique().tolist())
    dates = sorted(base_df[date_col].dropna().unique().tolist())

    if not tickers or not dates:
        raise ValueError(
            "Cannot auto-create embeddings: no tickers or dates found in base_df"
        )

    print(f"[EMB] Auto-creating dummy embeddings at {path}")
    print(f"[EMB]   tickers: {len(tickers)}, dates: {len(dates)}, dim: {emb_dim}")

    rows = []
    rng = np.random.default_rng(42)  # deterministic for reproducibility
    for t in tickers:
        for d in dates:
            emb = rng.normal(0.0, 0.1, emb_dim).tolist()
            row = {ticker_col: t, date_col: d}
            for i, val in enumerate(emb):
                row[f"e{i}"] = float(val)
            rows.append(row)

    emb_df = pd.DataFrame(rows)

    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    emb_df.to_csv(path, index=False)
    print(f"[EMB]   wrote {len(emb_df)} rows to {path}")


def load_embeddings(
    path: Optional[str],
    key_cols: Tuple[str, str] = ("ticker", "date"),
    prefix: str = "",
    base_df: Optional[pd.DataFrame] = None,
) -> Optional[Tuple[pd.DataFrame, List[str]]]:
    """
    Load embedding CSV with key columns referencing ticker and date.

    If the CSV does not exist but a base_df is provided:
       - Automatically create a dummy embedding file at `path`.

    Embedding columns (everything except key_cols) are optionally renamed
    with the given prefix, e.g. e0 -> m_e0 for market embeddings.

    Returns a tuple of (dataframe, actual_key_cols_used) to support
    embeddings that only have 'date' (market-level) vs both 'ticker' and 'date'.
    """
    if path is None:
        return None

    if not os.path.exists(path):
        if base_df is None:
            raise FileNotFoundError(
                f"Embedding file '{path}' does not exist and base_df is None; "
                f"cannot auto-create embeddings."
            )
        create_dummy_embeddings(path=path, base_df=base_df, key_cols=key_cols)

    emb_df = pd.read_csv(path)

    # Determine which key columns are actually present
    actual_key_cols = [col for col in key_cols if col in emb_df.columns]
    if not actual_key_cols:
        raise ValueError(
            f"Embeddings file '{path}' has no recognized key columns. "
            f"Expected at least one of: {key_cols}"
        )

    emb_cols = [c for c in emb_df.columns if c not in list(key_cols)]
    emb_df = emb_df[actual_key_cols + emb_cols]

    if prefix:
        rename_map = {c: f"{prefix}{c}" for c in emb_cols}
        emb_df = emb_df.rename(columns=rename_map)

    return emb_df, actual_key_cols


def merge_embeddings(
    base_df: pd.DataFrame,
    market_result: Optional[Tuple[pd.DataFrame, List[str]]],
    sector_result: Optional[Tuple[pd.DataFrame, List[str]]],
) -> pd.DataFrame:
    """
    Merge market and sector embeddings onto base panel by their respective key columns.
    """
    df = base_df.copy()

    if market_result is not None:
        market_df, market_keys = market_result
        df = df.merge(market_df, on=market_keys, how="left")

    if sector_result is not None:
        sector_df, sector_keys = sector_result
        df = df.merge(sector_df, on=sector_keys, how="left")

    return df
