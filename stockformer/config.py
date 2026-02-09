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
    "date", "symbol", "open", "high", "low", "adjusted_close", "volume"
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
    "lookback": 30,           # Number of past days to feed into the model
    "horizon": 3,             # Prediction horizon in days (3, 10, or 30)
    "label_mode": "binary",   # Options: "regression", "binary", "buckets"
    "bucket_edges": [-6, -4, -2, 0, 2, 4, 6],  # Bucket edges in percent for "buckets" mode
    "mode": "correlated",     # "single" = stock features only, "correlated" = include embeddings

    # -------------------------------------------------------------------------
    # Training hyperparameters
    # -------------------------------------------------------------------------
    "batch_size": 128,
    "lr": 5e-4,
    "num_epochs": 50,
    "val_fraction": 0.2,
    "optimizer": "adamw",     # Options: "adam", "adamw", "sgd"
    "weight_decay": 0.01,     # AdamW weight decay (0 = disabled)
    "patience": 10,           # Early stopping patience
    "warmup_epochs": 5,       # Linear warmup epochs

    # -------------------------------------------------------------------------
    # Model architecture
    # -------------------------------------------------------------------------
    "d_model": 64,            # Transformer model dimension
    "nhead": 4,               # Number of attention heads
    "num_layers": 2,          # Number of transformer encoder layers
    "dim_feedforward": 128,   # Hidden dimension of feedforward network
    "dropout": 0.2,
    "layer_drop": 0.1,        # Stochastic depth: probability of skipping a layer (0 = disabled)

    # -------------------------------------------------------------------------
    # System / runtime
    # -------------------------------------------------------------------------
    "num_workers": 0,         # DataLoader workers (0 = main process)
    "device": None,           # None = auto-detect (cuda if available, else cpu)

    # -------------------------------------------------------------------------
    # Anti-collapse / loss settings
    # -------------------------------------------------------------------------
    "loss_name": None,            # Override loss: "focal", "label_smoothing", etc. (None = default for label_mode)
    "entropy_reg_weight": 0.05,   # Entropy regularization weight (>0 penalizes collapsed predictions)

    # -------------------------------------------------------------------------
    # Output paths
    # -------------------------------------------------------------------------
    "model_out": "model_checkpoint.pt",
    "log_path": "training_log.csv",
    "output_csv": "predictions.csv",
}


# =============================================================================
# Base Feature Columns
# =============================================================================

# All features must be stationary (relative/pct/bounded).
# No raw dollar amounts (close, sma_20, eps, market_cap, etc.) — these
# break when the model sees future data at different price levels.

BASE_FEATURE_COLUMNS = [
    # Volume (relative)
    "volume_pct",
    # Momentum / rate of change (stationary)
    "close_1d_roc",
    "close_3d_roc",
    "close_10d_roc",
    "close_5d_vol",
    "vol_3d_mean",
    "vol_10d_mean",
    # Moving average distance (pct from price — stationary)
    "sma_20_pct",
    "sma_50_pct",
    "sma_200_pct",
    # Technical indicators (bounded/stationary by construction)
    "rsi_14",
    "adx_14",
    "cci_14",
    # Relative position (already pct-based)
    "52_week_high_pct",
    "52_week_low_pct",
    "high_pct",
    "low_pct",
    "open_pct",
    "adjusted_close_pct",
    # Signals (discrete, bounded)
    "macd_signal",
    "rsi_signal",
    "adx_signal",
    "atr_signal",
    "cci_signal",
    "bollinger_bands_signal",
    "sma_cross_signal",
    "pe_ratio_signal",
    "bull_bear_delta",
    # Fundamentals (ratios only — no dollar amounts)
    "pe_ratio",
    "forward_pe",
    "beta",
    "dividend_yield",
    "price_to_book_ratio",
    # Earnings (surprise is relative)
    "surprise_percentage",
    # Analyst ratings (counts — comparable across time)
    "analyst_rating_strong_buy",
    "analyst_rating_buy",
    "analyst_rating_hold",
    "analyst_rating_sell",
    "analyst_rating_strong_sell",
    # Calendar
    "day_of_week_num",
    "month",
    # Macro (rates/pct — stationary)
    "treasury_yield_10year",
    "treasury_yield_2year",
    "unemployment",
    "nonfarm_payroll",
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
