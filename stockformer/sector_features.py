"""
Sector and market ETF feature extraction for the StockFormer pipeline.

Extracts real features from SPY, QQQ, and sector ETFs (XLK, XLV, XLF, etc.)
that already exist in the panel data, replacing dummy random embeddings.

Functions:
    - extract_etf_features: Compute return/volatility features for each ETF
    - add_sector_features: Merge sector-relative and market features onto stocks
    - exclude_etf_symbols: Remove ETF rows so they aren't training targets
"""

import numpy as np
import pandas as pd


# =============================================================================
# Sector ETF Mapping
# =============================================================================

SECTOR_ETF_MAP = {
    "TECHNOLOGY": "XLK",
    "COMMUNICATION SERVICES": "XLK",
    "HEALTHCARE": "XLV",
    "LIFE SCIENCES": "XLV",
    "FINANCIAL SERVICES": "XLF",
    "FINANCE": "XLF",
    "CONSUMER CYCLICAL": "XLY",
    "CONSUMER DEFENSIVE": "XLP",
    "INDUSTRIALS": "XLI",
    "ENERGY": "XLE",
    "ENERGY & TRANSPORTATION": "XLE",
    "UTILITIES": "XLU",
    "REAL ESTATE": "XLRE",
    "BASIC MATERIALS": "XLB",
}

MARKET_ETFS = ["SPY", "QQQ"]
SECTOR_ETFS = sorted(set(SECTOR_ETF_MAP.values()))
ALL_ETFS = set(SECTOR_ETFS + MARKET_ETFS)


# =============================================================================
# ETF Feature Extraction
# =============================================================================

def extract_etf_features(
    df: pd.DataFrame,
    date_col: str = "date",
    group_col: str = "ticker",
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Extract return and volatility features for each ETF in the panel data.

    For each ETF, computes:
        - {etf}_ret_1d: 1-day return (pct_change)
        - {etf}_ret_5d: 5-day rolling return
        - {etf}_vol_5d: 5-day rolling volatility of daily returns

    Args:
        df: Full panel DataFrame (must include ETF rows)
        date_col: Date column name
        group_col: Ticker/symbol column name
        price_col: Price column for computing returns

    Returns:
        DataFrame indexed by date with ETF feature columns
    """
    etf_df = df[df[group_col].isin(ALL_ETFS)][[group_col, date_col, price_col]].copy()
    etf_df = etf_df.sort_values([group_col, date_col])

    all_features = []

    for etf in sorted(ALL_ETFS):
        mask = etf_df[group_col] == etf
        ticker_df = etf_df.loc[mask].copy()

        if len(ticker_df) < 6:
            print(f"[SECTOR] Warning: {etf} has only {len(ticker_df)} rows, skipping")
            continue

        # Compute features
        ticker_df[f"{etf}_ret_1d"] = ticker_df[price_col].pct_change()
        ticker_df[f"{etf}_ret_5d"] = (
            ticker_df[price_col] / ticker_df[price_col].shift(5) - 1.0
        )
        ticker_df[f"{etf}_vol_5d"] = (
            ticker_df[price_col].pct_change().rolling(5).std()
        )

        feat_cols = [f"{etf}_ret_1d", f"{etf}_ret_5d", f"{etf}_vol_5d"]
        all_features.append(ticker_df[[date_col] + feat_cols].set_index(date_col))

    if not all_features:
        raise ValueError("No ETF features could be extracted — check data for ETF symbols")

    # Merge all ETF features by date (one row per date)
    result = all_features[0]
    for feat_df in all_features[1:]:
        result = result.join(feat_df, how="outer")

    result = result.sort_index()
    print(f"[SECTOR] Extracted features for {len(all_features)} ETFs, "
          f"{len(result)} dates, {len(result.columns)} columns")

    return result


# =============================================================================
# Add Sector Features to Stock Data
# =============================================================================

def add_sector_features(
    df: pd.DataFrame,
    etf_features: pd.DataFrame,
    sector_col: str = "sector",
    date_col: str = "date",
    group_col: str = "ticker",
) -> pd.DataFrame:
    """
    Add sector-relative and market features to stock-level data.

    For each stock-date row:
        - sector_etf_ret_1d/5d, sector_etf_vol_5d: from matched sector ETF
        - sector_rel_ret_1d: stock's 1d return minus sector ETF's 1d return
        - spy_ret_1d/5d, spy_vol_5d: SPY features (same for all stocks on a date)

    Args:
        df: Stock panel DataFrame
        etf_features: ETF features DataFrame from extract_etf_features()
        sector_col: Column with stock's sector name
        date_col: Date column
        group_col: Ticker column

    Returns:
        DataFrame with sector/market feature columns added
    """
    df = df.copy()

    # Merge ETF features by date
    etf_flat = etf_features.reset_index()
    df = df.merge(etf_flat, on=date_col, how="left")

    # Map each stock's sector to its sector ETF
    if sector_col in df.columns:
        df["_sector_etf"] = df[sector_col].map(SECTOR_ETF_MAP)
    else:
        df["_sector_etf"] = np.nan

    # Build sector-specific columns by looking up the right ETF's features
    sector_ret_1d = np.full(len(df), np.nan)
    sector_ret_5d = np.full(len(df), np.nan)
    sector_vol_5d = np.full(len(df), np.nan)

    for etf in SECTOR_ETFS:
        mask = df["_sector_etf"] == etf
        ret_1d_col = f"{etf}_ret_1d"
        ret_5d_col = f"{etf}_ret_5d"
        vol_5d_col = f"{etf}_vol_5d"

        if ret_1d_col in df.columns:
            sector_ret_1d[mask] = df.loc[mask, ret_1d_col].values
        if ret_5d_col in df.columns:
            sector_ret_5d[mask] = df.loc[mask, ret_5d_col].values
        if vol_5d_col in df.columns:
            sector_vol_5d[mask] = df.loc[mask, vol_5d_col].values

    df["sector_etf_ret_1d"] = sector_ret_1d
    df["sector_etf_ret_5d"] = sector_ret_5d
    df["sector_etf_vol_5d"] = sector_vol_5d

    # Sector-relative return: stock's 1d return minus sector ETF's 1d return
    # Use close_1d_roc if available (computed by features.py), else pct_change
    if "close_1d_roc" in df.columns:
        stock_ret_1d = df["close_1d_roc"]
    elif "close" in df.columns:
        stock_ret_1d = df.groupby(group_col)["close"].pct_change()
    else:
        stock_ret_1d = 0.0
    df["sector_rel_ret_1d"] = stock_ret_1d - df["sector_etf_ret_1d"]

    # SPY features (same for all stocks on a given date)
    df["spy_ret_1d"] = df.get("SPY_ret_1d", np.nan)
    df["spy_ret_5d"] = df.get("SPY_ret_5d", np.nan)
    df["spy_vol_5d"] = df.get("SPY_vol_5d", np.nan)

    # Fill NaN sector/market features with 0
    sector_market_cols = [
        "sector_etf_ret_1d", "sector_etf_ret_5d", "sector_etf_vol_5d",
        "sector_rel_ret_1d", "spy_ret_1d", "spy_ret_5d", "spy_vol_5d",
    ]
    df[sector_market_cols] = df[sector_market_cols].fillna(0.0)

    # Drop intermediate ETF columns and helper
    etf_detail_cols = [
        c for c in df.columns
        if any(c.startswith(f"{etf}_") for etf in ALL_ETFS)
    ]
    df = df.drop(columns=etf_detail_cols + ["_sector_etf"], errors="ignore")

    non_zero = (df[sector_market_cols] != 0).any(axis=1).sum()
    print(f"[SECTOR] Added sector/market features: {non_zero}/{len(df)} rows have non-zero values")

    return df


# =============================================================================
# Exclude ETF Symbols from Training
# =============================================================================

def exclude_etf_symbols(
    df: pd.DataFrame,
    group_col: str = "ticker",
) -> pd.DataFrame:
    """
    Remove ETF rows so they aren't used as training/inference targets.

    Args:
        df: Stock panel DataFrame
        group_col: Ticker column

    Returns:
        DataFrame with ETF rows removed
    """
    before = len(df)
    df = df[~df[group_col].isin(ALL_ETFS)].copy()
    removed = before - len(df)
    print(f"[SECTOR] Excluded {removed} ETF rows ({sorted(ALL_ETFS)}), {len(df)} rows remaining")
    return df
