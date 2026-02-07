"""
Configuration for the StockFormer pipeline.

Edit DEFAULT_CONFIG to change settings.
Edit BASE_FEATURE_COLUMNS to change which features are used.
Edit OUTPUT_COLUMN_GROUPS to customize prediction output columns.
"""

# =============================================================================
# Output Column Groups (for predictions CSV)
# =============================================================================

CORE_FIELDS = [
    "date", "symbol", "open", "high", "low", "close", "volume"
]

MOVING_AVG_FIELDS = [
    "sma_20", "sma_50", "sma_200", "ema_20", "ema_50", "ema_200"
]

TECHNICAL_FIELDS = [
    "macd", "rsi_14", "adx_14", "atr_14", "cci_14",
    "bbands_upper_20", "bbands_middle_20", "bbands_lower_20"
]

OPTIONS_FIELDS = [
    "call_volume", "put_volume", "total_volume",
    "options_14_mean", "options_14_std", "pcr", "pcr_14_mean"
]

SIGNAL_FIELDS = [
    "macd_signal", "macd_zero_signal", "macd_9_ema", "adx_signal", "atr_signal",
    "pe_ratio", "pe_ratio_signal", "bollinger_bands_signal", "rsi_signal",
    "sma_cross_signal", "cci_signal", "pcr_signal", "bull_bear_delta"
]

OUTPUT_COLUMN_GROUPS = {
    "core": CORE_FIELDS,
    "moving_avg": MOVING_AVG_FIELDS,
    "technical": TECHNICAL_FIELDS,
    "options": OPTIONS_FIELDS,
    "signals": SIGNAL_FIELDS,
}


# =============================================================================
# Default Configuration
# =============================================================================

DEFAULT_CONFIG = {
    # -------------------------------------------------------------------------
    # Data paths
    # -------------------------------------------------------------------------
    "data_path": "../../data/all_data_*.csv",
    "market_path": None,      # Path to market embeddings CSV (auto-created if missing)
    "sector_path": None,      # Path to sector embeddings CSV (auto-created if missing)

    # -------------------------------------------------------------------------
    # Sequence and label settings
    # -------------------------------------------------------------------------
    "lookback": 5,            # Number of past days to feed into the model
    "horizon": 3,             # Prediction horizon in days (3, 10, or 30)
    "label_mode": "binary",   # Options: "regression", "binary", "buckets"
    "bucket_edges": [-6, -4, -2, 0, 2, 4, 6],  # Bucket edges in percent for "buckets" mode
    "mode": "correlated",     # "single" = stock features only, "correlated" = include embeddings

    # -------------------------------------------------------------------------
    # Training hyperparameters
    # -------------------------------------------------------------------------
    "batch_size": 128,
    "lr": 1e-3,
    "num_epochs": 10,
    "val_fraction": 0.2,
    "optimizer": "adam",      # Options: "adam", "adamw", "sgd"

    # -------------------------------------------------------------------------
    # Model architecture
    # -------------------------------------------------------------------------
    "d_model": 128,           # Transformer model dimension
    "nhead": 4,               # Number of attention heads
    "num_layers": 3,          # Number of transformer encoder layers
    "dim_feedforward": 256,   # Hidden dimension of feedforward network
    "dropout": 0.1,

    # -------------------------------------------------------------------------
    # System / runtime
    # -------------------------------------------------------------------------
    "num_workers": 0,         # DataLoader workers (0 = main process)
    "device": None,           # None = auto-detect (cuda if available, else cpu)

    # -------------------------------------------------------------------------
    # Output paths
    # -------------------------------------------------------------------------
    "model_out": "output/model_checkpoint.pt",
    "log_path": "output/training_log.csv",
    "output_csv": "output/predictions.csv",
}


# =============================================================================
# Base Feature Columns
# =============================================================================

# These are the core features computed from price/volume data.
# Edit this list to add or remove features.
# Note: feature names must match what's computed in features.py

BASE_FEATURE_COLUMNS = [
    # Price & Volume (0% null)
    "open", "high", "low", "close", "volume",
    "adjusted_open", "adjusted_high", "adjusted_low",

    # Moving Averages (<2% null)
    "sma_20", "sma_50", "sma_200",
    "ema_20", "ema_50", "ema_200",
    "sma_20_pct", "sma_50_pct", "sma_200_pct",

    # Technical Indicators (<1% null)
    "macd", "macd_9_ema",
    "rsi_14", "adx_14", "atr_14", "cci_14",
    "bbands_upper_20", "bbands_middle_20", "bbands_lower_20",

    # Percent Changes (0% null)
    "adjusted_close_pct", "volume_pct", "open_pct", "high_pct", "low_pct",

    # 52-Week Range (~14% null - fundamentals)
    "52_week_high", "52_week_low", "52_week_high_pct", "52_week_low_pct",

    # Economic Indicators (<2% null, except inflation=100% removed)
    "treasury_yield_2year", "treasury_yield_10year",
    "ffer", "cpi", "retail_sales", "durables",
    "unemployment", "nonfarm_payroll",

    # Fundamentals (~14% null - quarterly data)
    "reported_eps", "estimated_eps", "ttm_eps",
    "surprise", "surprise_percentage",
    "market_capitalization", "book_value", "dividend_yield",
    "eps", "price_to_book_ratio", "beta", "shares_outstanding",
    "forward_pe", "pe_ratio",

    # Analyst Ratings (~14% null)
    "analyst_rating_strong_buy", "analyst_rating_buy",
    "analyst_rating_hold", "analyst_rating_sell", "analyst_rating_strong_sell",

    # Temporal (0% null)
    "day_of_week_num", "month", "day_of_year", "year",

    # Signals - derived indicators (0% null)
    "macd_signal", "adx_signal", "atr_signal", "pe_ratio_signal",
    "bollinger_bands_signal", "rsi_signal", "sma_cross_signal",
    "cci_signal", "bull_bear_delta",
]


# =============================================================================
# Helper Functions
# =============================================================================

def get_feature_columns(df, mode):
    """
    Get all feature columns based on mode.

    In "correlated" mode, adds embedding columns (m_* and s_*) to base features.
    """
    feature_cols = BASE_FEATURE_COLUMNS.copy()

    if mode == "correlated":
        for col in df.columns:
            if col.startswith("m_") or col.startswith("s_"):
                feature_cols.append(col)

    return feature_cols


def get_target_column(horizon):
    """Get the target column name for a given horizon."""
    return f"future_{horizon}_day_pct"
