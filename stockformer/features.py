"""
Feature engineering for the StockFormer pipeline.

To add a new feature:
    1. Add an entry to FEATURE_LIST with (name, function_type, params)
    2. If needed, add a new function type in compute_feature()
    3. Add the feature name to BASE_FEATURE_COLUMNS in config.py
"""

import numpy as np
import pandas as pd


# =============================================================================
# Feature Definitions
# =============================================================================

# Each feature is defined as: (feature_name, function_type, params)
# Edit this list to add or remove features.

FEATURE_LIST = [
    # Price rate of change features
    ("close_1d_roc", "pct_change", {"column": "close", "periods": 1}),
    ("close_3d_roc", "pct_change", {"column": "close", "periods": 3}),
    ("close_10d_roc", "pct_change", {"column": "close", "periods": 10}),

    # Volume rolling mean features
    ("vol_3d_mean", "rolling_mean", {"column": "volume", "window": 3}),
    ("vol_10d_mean", "rolling_mean", {"column": "volume", "window": 10}),

    # Volatility features
    ("close_5d_vol", "rolling_std", {"column": "close_1d_roc", "window": 5}),

    # -------------------------------------------------------------------------
    # Add new features here. Examples:
    # -------------------------------------------------------------------------
    # ("close_20d_roc", "pct_change", {"column": "close", "periods": 20}),
    # ("close_10d_ema", "ema", {"column": "close", "span": 10}),
    # ("vol_20d_mean", "rolling_mean", {"column": "volume", "window": 20}),
]


# =============================================================================
# Feature Computation
# =============================================================================

def compute_feature(df, group_col, feature_name, func_type, params):
    """
    Compute a single feature and add it to the dataframe.

    To add a new function type, add an elif block below.

    Args:
        df: DataFrame with stock data
        group_col: Column to group by (usually "ticker")
        feature_name: Name for the new column
        func_type: Type of computation ("pct_change", "rolling_mean", etc.)
        params: Dict of parameters for the computation

    Returns:
        DataFrame with the new feature column added
    """

    if func_type == "pct_change":
        # Percent change over N periods
        df[feature_name] = df.groupby(group_col)[params["column"]].pct_change(params["periods"])

    elif func_type == "rolling_mean":
        # Rolling mean over a window
        df[feature_name] = (
            df.groupby(group_col)[params["column"]]
            .rolling(params["window"])
            .mean()
            .reset_index(level=0, drop=True)
        )

    elif func_type == "rolling_std":
        # Rolling standard deviation over a window
        df[feature_name] = (
            df.groupby(group_col)[params["column"]]
            .rolling(params["window"])
            .std()
            .reset_index(level=0, drop=True)
        )

    elif func_type == "ema":
        # Exponential moving average
        df[feature_name] = (
            df.groupby(group_col)[params["column"]]
            .ewm(span=params["span"], adjust=False)
            .mean()
            .reset_index(level=0, drop=True)
        )

    elif func_type == "rolling_max":
        # Rolling maximum over a window
        df[feature_name] = (
            df.groupby(group_col)[params["column"]]
            .rolling(params["window"])
            .max()
            .reset_index(level=0, drop=True)
        )

    elif func_type == "rolling_min":
        # Rolling minimum over a window
        df[feature_name] = (
            df.groupby(group_col)[params["column"]]
            .rolling(params["window"])
            .min()
            .reset_index(level=0, drop=True)
        )

    # -------------------------------------------------------------------------
    # Add new function types here. Example:
    # -------------------------------------------------------------------------
    # elif func_type == "macd":
    #     fast = df.groupby(group_col)[params["column"]].ewm(span=12).mean()
    #     slow = df.groupby(group_col)[params["column"]].ewm(span=26).mean()
    #     df[feature_name] = (fast - slow).reset_index(level=0, drop=True)

    else:
        raise ValueError(f"Unknown function type: {func_type}")

    return df


def add_all_features(df, group_col="ticker", date_col="date"):
    """
    Apply all features defined in FEATURE_LIST to the dataframe.

    Args:
        df: DataFrame with stock data (must have group_col, date_col, and required columns)
        group_col: Column to group by (default: "ticker")
        date_col: Column for dates (default: "date")

    Returns:
        DataFrame with all features added
    """
    # Sort by group and date first
    df = df.sort_values([group_col, date_col]).reset_index(drop=True)

    # Compute each feature
    for feature_name, func_type, params in FEATURE_LIST:
        df = compute_feature(df, group_col, feature_name, func_type, params)

        # Clean up: replace inf with NaN, then fill NaNs with 0.0
        df[feature_name] = df[feature_name].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df
