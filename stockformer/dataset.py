"""
PyTorch Dataset for stock sequence data.

Classes:
    - StockSequenceDataset: Creates (sequence, label) pairs for training
"""

import math
from typing import Any, List, Optional, Tuple

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

    For cross-attention mode (market_feature_cols provided):
      - Returns (stock_seq, market_seq, label)
      - stock_seq: [lookback, stock_feature_dim]
      - market_seq: [lookback, market_feature_dim]

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
        market_feature_cols: Optional list of market feature columns for cross-attention mode
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
        market_feature_cols: Optional[List[str]] = None,
        group_col: str = "ticker",
        date_col: str = "date",
    ):
        self.df = df.sort_values([group_col, date_col]).reset_index(drop=True)
        self.lookback = lookback
        self.target_col = target_col
        self.feature_cols = feature_cols
        self.label_mode = label_mode
        self.bucket_edges = bucket_edges
        self.market_feature_cols = market_feature_cols
        self.group_col = group_col
        self.date_col = date_col

        # Clean feature columns: replace inf with NaN, then fill NaNs with 0.0
        self.df[self.feature_cols] = (
            self.df[self.feature_cols]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        # Normalize features (z-score) to stabilize transformer numerics
        feats = self.df[self.feature_cols].to_numpy(dtype="float32")
        means = feats.mean(axis=0)
        stds = feats.std(axis=0)
        stds = np.where(stds < 1e-6, 1.0, stds)  # avoid division by zero
        feats = (feats - means) / stds
        self.df[self.feature_cols] = feats

        # Normalize market features if provided (for cross-attention mode)
        if self.market_feature_cols:
            self.df[self.market_feature_cols] = (
                self.df[self.market_feature_cols]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )
            market_feats = self.df[self.market_feature_cols].to_numpy(dtype="float32")
            m_means = market_feats.mean(axis=0)
            m_stds = market_feats.std(axis=0)
            m_stds = np.where(m_stds < 1e-6, 1.0, m_stds)
            market_feats = (market_feats - m_means) / m_stds
            self.df[self.market_feature_cols] = market_feats

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

    def __getitem__(self, idx: int):
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

        # Return market features separately for cross-attention mode
        if self.market_feature_cols:
            market_seq = seq_df[self.market_feature_cols].to_numpy(dtype="float32")
            market_tensor = torch.from_numpy(market_seq)  # [lookback, market_feat_dim]
            return seq_tensor, market_tensor, label

        return seq_tensor, label


# =============================================================================
# Train/Val Split
# =============================================================================

def make_train_val_split(
    dataset: Dataset,
    val_fraction: float = 0.2,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    """
    Create train/val split for a given dataset.

    Args:
        dataset: The dataset to split
        val_fraction: Fraction of data to use for validation (default: 0.2)
        shuffle: Whether to shuffle before splitting (default: True)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Tuple of (train_subset, val_subset)
    """
    n_total = len(dataset)
    n_val = int(math.floor(val_fraction * n_total))
    n_train = n_total - n_val

    indices = list(range(n_total))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    return Subset(dataset, train_indices), Subset(dataset, val_indices)
