"""
PyTorch Dataset for stock sequence data.

Classes:
    - StockSequenceDataset: Creates (sequence, label) pairs for training

Train/val split is TEMPORAL (by date), not random, to prevent look-ahead bias.
Normalization stats are computed on training data only, then applied to both.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset


# =============================================================================
# Dataset
# =============================================================================

class StockSequenceDataset(Dataset):
    """
    Dataset that produces (sequence, label) pairs for a panel of stock data.

    Each item is:
      - A [lookback, feature_dim] float tensor (the input sequence)
      - A label: either a float (regression) or class index (classification)

    Args:
        df: DataFrame with stock data
        lookback: Number of past days to include in each sequence
        target_col: Column name for the target variable
        feature_cols: List of column names to use as features
        label_mode: How to create labels
            - "regression": label is the raw future return (float)
            - "binary": label is 0 if return < 0, else 1
            - "buckets": label is bucket index based on bucket_edges
        bucket_edges: List of edges for bucket mode (in percent, e.g. [-2, 0, 2])
        group_col: Column to group by (default: "ticker")
        date_col: Column for dates (default: "date")
        market_feature_cols: Optional list of market/sector feature columns
            for cross-attention or other multi-input model types
    """

    def __init__(
        self,
        df: pd.DataFrame,
        lookback: int,
        target_col: str,
        feature_cols: List[str],
        label_mode: str = "binary",
        bucket_edges: Optional[List[float]] = None,
        group_col: str = "ticker",
        date_col: str = "date",
        market_feature_cols: Optional[List[str]] = None,
    ):
        self.df = df.sort_values([group_col, date_col]).reset_index(drop=True)
        self.lookback = lookback
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.label_mode = label_mode
        self.bucket_edges = bucket_edges
        self.group_col = group_col
        self.date_col = date_col
        self.market_feature_cols = market_feature_cols

        # Clean feature columns: forward-fill within each ticker, then zero-fill
        all_feat_cols = list(self.feature_cols)
        if self.market_feature_cols:
            all_feat_cols += self.market_feature_cols

        self.df[all_feat_cols] = (
            self.df.groupby(group_col)[all_feat_cols]
            .ffill()
            .fillna(0.0)
        )
        self.df[all_feat_cols] = (
            self.df[all_feat_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        # Store normalization stats — actual z-score is applied later
        # by make_temporal_split() using training-only stats
        self._norm_means: Optional[np.ndarray] = None
        self._norm_stds: Optional[np.ndarray] = None

        # Precompute valid indices where lookback window is available per ticker
        # Also precompute group indices for fast __getitem__ access
        self.indices: List[int] = []
        self._group_indices: dict = {}  # row_idx -> (group_indices_list, position_in_group)

        grouped = self.df.groupby(self.group_col).indices
        for group_name, idxs in grouped.items():
            idxs_list = list(idxs)
            for i in range(len(idxs_list)):
                if i >= self.lookback - 1:
                    row_idx = idxs_list[i]
                    self.indices.append(row_idx)
                    # Store precomputed group info for O(1) access
                    self._group_indices[row_idx] = (idxs_list, i)

        # Filter indices where target is not NaN
        valid_indices = []
        for idx in self.indices:
            val = self.df.loc[idx, self.target_col]
            if not pd.isna(val):
                valid_indices.append(idx)
        self.indices = valid_indices

        if not self.indices:
            raise ValueError("No valid indices with non-NaN targets found.")

        # If bucket mode, convert bucket edges from percent to fraction
        if self.label_mode == "buckets":
            if not self.bucket_edges:
                raise ValueError(
                    "bucket_edges must be provided for 'buckets' label_mode."
                )
            edges = np.array(self.bucket_edges, dtype="float32") / 100.0
            self.bucket_edges = edges.tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def _get_bucket_label(self, value: float) -> int:
        """
        Convert a future return value to a bucket index using bucket_edges.

        For edges e.g. [-0.02, 0.0, 0.02], buckets are:
          0: value < -0.02
          1: -0.02 <= value < 0.0
          2: 0.0 <= value < 0.02
          3: value >= 0.02
        """
        edges = np.array(self.bucket_edges, dtype="float32")
        return int(np.searchsorted(edges, value, side="right"))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        row_idx = self.indices[idx]

        # Use precomputed group info for O(1) access
        group_indices, pos_in_group = self._group_indices[row_idx]

        # Sequence goes from (pos_in_group - lookback + 1) ... pos_in_group
        start_pos = pos_in_group - self.lookback + 1
        seq_indices = group_indices[start_pos : pos_in_group + 1]

        seq_df = self.df.loc[seq_indices]
        seq = seq_df[self.feature_cols].to_numpy(dtype="float32")
        seq_tensor = torch.from_numpy(seq)  # [lookback, feat_dim]

        # Create label based on label_mode
        target_val = float(self.df.loc[row_idx, self.target_col])

        if self.label_mode == "regression":
            label = target_val
        elif self.label_mode == "binary":
            label = int(target_val >= 0.0)
        elif self.label_mode == "buckets":
            label = self._get_bucket_label(target_val)
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")

        # For cross-attention models, return market features as third element
        if self.market_feature_cols:
            market_seq = seq_df[self.market_feature_cols].to_numpy(dtype="float32")
            market_tensor = torch.from_numpy(market_seq)
            return seq_tensor, label, market_tensor

        return seq_tensor, label


# =============================================================================
# Normalization
# =============================================================================

def apply_normalization(
    dataset: "StockSequenceDataset",
    means: np.ndarray,
    stds: np.ndarray,
) -> None:
    """Apply z-score normalization using given stats (in-place)."""
    feats = dataset.df[dataset.feature_cols].to_numpy(dtype="float32")
    feats = (feats - means) / stds
    dataset.df[dataset.feature_cols] = feats
    dataset._norm_means = means
    dataset._norm_stds = stds


def apply_per_ticker_normalization(
    dataset: "StockSequenceDataset",
    train_dates_cutoff: str,
) -> None:
    """
    Per-ticker z-score normalization using each ticker's own training-period stats.

    This handles scale differences between tickers ($5 biotech vs $500 mega-cap)
    and keeps features stationary across time for each ticker.

    Args:
        dataset: StockSequenceDataset instance
        train_dates_cutoff: Date string; stats computed from rows with date < cutoff
    """
    df = dataset.df
    feat_cols = dataset.feature_cols
    group_col = dataset.group_col
    date_col = dataset.date_col

    # Cast feature columns to float64 so z-scored values can be written back
    for col in feat_cols:
        if df[col].dtype != np.float64:
            df[col] = df[col].astype(np.float64)

    train_mask = df[date_col] < train_dates_cutoff

    for ticker, group_idx in df.groupby(group_col).groups.items():
        ticker_mask = df.index.isin(group_idx)
        ticker_train_mask = ticker_mask & train_mask

        if ticker_train_mask.sum() < 2:
            # Not enough training data for this ticker — use global fallback
            continue

        train_feats = df.loc[ticker_train_mask, feat_cols].to_numpy(dtype="float64")
        means = train_feats.mean(axis=0)
        stds = train_feats.std(axis=0)
        stds = np.where(stds < 1e-6, 1.0, stds)

        # Normalize ALL rows for this ticker using training-period stats
        feats = df.loc[ticker_mask, feat_cols].to_numpy(dtype="float64")
        feats = (feats - means) / stds
        df.loc[ticker_mask, feat_cols] = feats

    # Store dummy global stats for compatibility
    dataset._norm_means = np.zeros(len(feat_cols))
    dataset._norm_stds = np.ones(len(feat_cols))


# =============================================================================
# Train/Val Split — TEMPORAL (no look-ahead bias)
# =============================================================================

def make_temporal_split(
    dataset: "StockSequenceDataset",
    val_fraction: float = 0.2,
) -> Tuple[Subset, Subset]:
    """
    Create train/val split by DATE (temporal), not random.

    The earliest (1 - val_fraction) of dates go to training,
    the most recent val_fraction of dates go to validation.
    Normalization stats are computed on training data only, then applied to both.

    Args:
        dataset: StockSequenceDataset instance
        val_fraction: Fraction of dates for validation (default: 0.2)

    Returns:
        Tuple of (train_subset, val_subset)
    """
    # Get the date for each valid index
    dates = dataset.df.loc[
        [dataset.indices[i] for i in range(len(dataset.indices))],
        dataset.date_col,
    ]
    sorted_unique_dates = sorted(dates.unique())
    cutoff_idx = int(len(sorted_unique_dates) * (1 - val_fraction))
    cutoff_date = sorted_unique_dates[cutoff_idx]

    # Split indices by date
    train_indices = []
    val_indices = []
    for i, row_idx in enumerate(dataset.indices):
        d = dataset.df.loc[row_idx, dataset.date_col]
        if d < cutoff_date:
            train_indices.append(i)
        else:
            val_indices.append(i)

    # Per-ticker normalization: each ticker gets z-scored using its own
    # training-period stats. This handles scale differences across tickers
    # and keeps features stationary within each ticker's history.
    apply_per_ticker_normalization(dataset, cutoff_date)

    print(
        f"Temporal split: train={len(train_indices)} samples "
        f"(dates < {cutoff_date}), val={len(val_indices)} samples "
        f"(dates >= {cutoff_date})"
    )

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


# =============================================================================
# Quantile Bucket Edges
# =============================================================================

def compute_quantile_edges(df, target_col, n_buckets=4, group_col="ticker"):
    """
    Compute bucket edges from quantiles of the target distribution.

    Ensures roughly equal samples per bucket, eliminating class imbalance.

    Args:
        df: DataFrame with target column
        target_col: Column name for the target variable
        n_buckets: Number of buckets (edges = n_buckets - 1)
        group_col: Grouping column (unused, kept for API consistency)

    Returns:
        List of bucket edges (in percent, matching bucket_edges convention)
    """
    vals = df[target_col].dropna()
    quantiles = np.linspace(0, 1, n_buckets + 1)[1:-1]  # e.g. [0.25, 0.5, 0.75] for 4 buckets
    edges = np.percentile(vals, quantiles * 100)
    # Convert from decimal to percent (bucket_edges convention)
    edges_pct = (edges * 100).tolist()
    return edges_pct


# Backward-compatible alias
def make_train_val_split(
    dataset: "StockSequenceDataset",
    val_fraction: float = 0.2,
    **kwargs,
) -> Tuple[Subset, Subset]:
    """Alias for make_temporal_split (temporal split is now the default)."""
    return make_temporal_split(dataset, val_fraction=val_fraction)
