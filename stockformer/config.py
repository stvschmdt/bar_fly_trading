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
    "lookback": 20,           # Number of past days to feed into the model
    "horizon": 3,             # Prediction horizon in days (3, 10, or 30)
    "label_mode": "binary",   # Options: "regression", "binary", "buckets"
    "bucket_edges": [-6, -4, -2, 0, 2, 4, 6],  # Bucket edges in percent for "buckets" mode
    "mode": "correlated",     # "single" = stock features only, "correlated" = include embeddings

    # -------------------------------------------------------------------------
    # Training hyperparameters
    # -------------------------------------------------------------------------
    "batch_size": 128,
    "lr": 5e-4,               # Lowered from 1e-3 to prevent early majority-class collapse
    "num_epochs": 30,
    "val_fraction": 0.2,
    "optimizer": "adamw",     # Options: "adam", "adamw", "sgd"

    # -------------------------------------------------------------------------
    # Loss function settings (anti-collapse)
    # -------------------------------------------------------------------------
    "loss_name": None,        # None = auto from label_mode (focal for cls, huber for reg)
    "focal_gamma": 2.0,       # Focal loss focusing parameter (higher = more focus on hard examples)
    "label_smoothing": 0.1,   # Smooth one-hot targets to prevent overconfident predictions
    "class_weights": "auto",  # "auto" = inverse-frequency from training data, None = uniform
    "entropy_weight": 0.1,    # Entropy regularization weight (encourages diverse predictions)

    # -------------------------------------------------------------------------
    # Model architecture
    # -------------------------------------------------------------------------
    "d_model": 128,           # Transformer model dimension
    "nhead": 4,               # Number of attention heads
    "num_layers": 3,          # Number of transformer encoder layers
    "dim_feedforward": 256,   # Hidden dimension of feedforward network
    "dropout": 0.15,          # Increased from 0.1 to reduce overfitting

    # -------------------------------------------------------------------------
    # System / runtime
    # -------------------------------------------------------------------------
    "num_workers": 0,         # DataLoader workers (0 = main process)
    "device": None,           # None = auto-detect (cuda if available, else cpu)

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

# These are the core features computed from price/volume data.
# Edit this list to add or remove features.
# Note: feature names must match what's computed in features.py

BASE_FEATURE_COLUMNS = [
    # Price & volume
    "close",
    "volume",
    "volume_pct",
    # Momentum / rate of change
    "close_1d_roc",
    "close_3d_roc",
    "close_10d_roc",
    "close_5d_vol",
    "vol_3d_mean",
    "vol_10d_mean",
    # Moving averages
    "sma_20",
    "sma_50",
    "sma_200",
    "ema_20",
    "ema_50",
    "sma_20_pct",
    "sma_50_pct",
    "sma_200_pct",
    # Technical indicators
    "rsi_14",
    "macd",
    "macd_9_ema",
    "adx_14",
    "atr_14",
    "cci_14",
    "bbands_upper_20",
    "bbands_middle_20",
    "bbands_lower_20",
    # Relative position
    "52_week_high_pct",
    "52_week_low_pct",
    "high_pct",
    "low_pct",
    "open_pct",
    "adjusted_close_pct",
    # Signals (screener composite)
    "macd_signal",
    "rsi_signal",
    "adx_signal",
    "atr_signal",
    "cci_signal",
    "bollinger_bands_signal",
    "sma_cross_signal",
    "pe_ratio_signal",
    "bull_bear_delta",
    # Fundamentals
    "pe_ratio",
    "forward_pe",
    "beta",
    # Macro
    "treasury_yield_10year",
    "treasury_yield_2year",
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
