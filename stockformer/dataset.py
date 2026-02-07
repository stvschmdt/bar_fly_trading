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
        self.df[self.feature_cols] = (
            self.df.groupby(group_col)[self.feature_cols]
            .ffill()
            .fillna(0.0)
        )
        self.df[self.feature_cols] = (
            self.df[self.feature_cols]
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

    # Compute normalization stats on TRAINING data only
    train_row_idxs = [dataset.indices[i] for i in train_indices]
    train_feats = dataset.df.loc[train_row_idxs, dataset.feature_cols].to_numpy(
        dtype="float32"
    )
    means = train_feats.mean(axis=0)
    stds = train_feats.std(axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)

    # Apply normalization to ENTIRE dataset using training stats
    apply_normalization(dataset, means, stds)

    print(
        f"Temporal split: train={len(train_indices)} samples "
        f"(dates < {cutoff_date}), val={len(val_indices)} samples "
        f"(dates >= {cutoff_date})"
    )

    return Subset(dataset, train_indices), Subset(dataset, val_indices)


# Backward-compatible alias
def make_train_val_split(
    dataset: "StockSequenceDataset",
    val_fraction: float = 0.2,
    **kwargs,
) -> Tuple[Subset, Subset]:
    """Alias for make_temporal_split (temporal split is now the default)."""
    return make_temporal_split(dataset, val_fraction=val_fraction)
