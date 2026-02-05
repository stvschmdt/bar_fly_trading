# StockFormer Test Plan

## Overview

This document outlines the test plan for the refactored `stockformer` module. Tests are organized into three categories:
1. **Unit Tests** - Test individual functions in isolation
2. **Integration Tests** - Test components working together
3. **End-to-End Tests** - Test full pipeline with real data

Data location: `--data-path "../../data/all_data_*.csv"`

---

## 1. Unit Tests

### 1.1 Config Module (`config.py`)

| Test | Description | Expected Result |
|------|-------------|-----------------|
| `test_default_config_keys` | Verify all required keys exist in DEFAULT_CONFIG | All keys present |
| `test_get_feature_columns_single` | Call with mode="single" | Returns only BASE_FEATURE_COLUMNS |
| `test_get_feature_columns_correlated` | Call with df containing m_* and s_* columns | Returns base + embedding columns |
| `test_get_target_column` | Call with horizon=3, 10, 30 | Returns "future_3_day_pct", etc. |

### 1.2 Data Utils (`data_utils.py`)

| Test | Description | Expected Result |
|------|-------------|-----------------|
| `test_load_panel_csvs_glob` | Load with glob pattern | Returns concatenated DataFrame |
| `test_load_panel_csvs_missing` | Load with non-existent path | Raises FileNotFoundError |
| `test_load_panel_csvs_column_rename` | Load data with "symbol" column | Renames to "ticker" |
| `test_add_future_returns_single_horizon` | Add horizon=3 | Creates "future_3_day_pct" column |
| `test_add_future_returns_multiple` | Add horizons=[3,10,30] | Creates all three columns |
| `test_add_future_returns_values` | Check computed values | Values match (price[t+h]/price[t])-1 |

### 1.3 Features (`features.py`)

| Test | Description | Expected Result |
|------|-------------|-----------------|
| `test_compute_feature_pct_change` | Compute 1-day ROC | Values match pandas pct_change |
| `test_compute_feature_rolling_mean` | Compute 3-day vol mean | Values match pandas rolling mean |
| `test_compute_feature_rolling_std` | Compute 5-day volatility | Values match pandas rolling std |
| `test_compute_feature_unknown_type` | Call with bad func_type | Raises ValueError |
| `test_add_all_features` | Apply all features | All FEATURE_LIST columns created |
| `test_add_all_features_no_inf` | Check for inf values | No inf values in output |

### 1.4 Dataset (`dataset.py`)

| Test | Description | Expected Result |
|------|-------------|-----------------|
| `test_dataset_length` | Create dataset, check len() | Returns expected count |
| `test_dataset_getitem_shape` | Get item, check tensor shape | Shape is [lookback, feature_dim] |
| `test_dataset_label_regression` | Create with label_mode="regression" | Label is float |
| `test_dataset_label_binary` | Create with label_mode="binary" | Label is 0 or 1 |
| `test_dataset_label_buckets` | Create with label_mode="buckets" | Label is bucket index |
| `test_dataset_normalization` | Check feature values | Mean ≈ 0, std ≈ 1 |
| `test_train_val_split_sizes` | Split with val_fraction=0.2 | 80/20 split |
| `test_train_val_split_no_overlap` | Check indices | No shared indices |

### 1.5 Model (`model.py`)

| Test | Description | Expected Result |
|------|-------------|-----------------|
| `test_model_forward_shape_regression` | Forward pass, regression mode | Output shape [batch] |
| `test_model_forward_shape_binary` | Forward pass, binary mode | Output shape [batch, 2] |
| `test_model_forward_shape_buckets` | Forward pass, buckets mode | Output shape [batch, num_buckets] |
| `test_create_model_regression` | create_model with regression | output_mode="regression" |
| `test_create_model_buckets_count` | create_model with 7 bucket edges | num_buckets=8 |

### 1.6 Losses (`losses.py`)

| Test | Description | Expected Result |
|------|-------------|-----------------|
| `test_get_loss_regression` | Get loss for regression | Returns MSELoss |
| `test_get_loss_binary` | Get loss for binary | Returns CrossEntropyLoss |
| `test_get_loss_focal` | Get loss for "focal" | Returns FocalLoss |
| `test_get_loss_unknown` | Get loss for bad name | Raises ValueError |
| `test_focal_loss_forward` | FocalLoss forward pass | Returns scalar tensor |

### 1.7 Training (`training.py`)

| Test | Description | Expected Result |
|------|-------------|-----------------|
| `test_get_optimizer_adam` | Get Adam optimizer | Returns Adam instance |
| `test_get_optimizer_unknown` | Get unknown optimizer | Raises ValueError |
| `test_run_one_epoch_training` | Run 1 training epoch | Returns dict with loss, accuracy |
| `test_run_one_epoch_validation` | Run 1 val epoch (no grad) | No gradient updates |

---

## 2. Integration Tests

### 2.1 Embedding Auto-Creation

**Test: `test_embedding_auto_detection_and_creation`**

```
Given: No embedding files exist at specified paths
When: train() is called with market_path="test_market_emb.csv"
Then:
  - Detects missing file
  - Creates dummy embeddings with correct shape
  - File contains (ticker, date, e0, e1, ..., e15) columns
  - Embeddings are merged into training data
  - Training proceeds without error
```

**Test: `test_embedding_reuse_existing`**

```
Given: Embedding file already exists from previous run
When: train() is called with same market_path
Then:
  - Does not recreate file
  - Loads existing embeddings
  - Training proceeds
```

**Test: `test_embedding_column_prefix`**

```
Given: market_path and sector_path both specified
When: Embeddings are merged
Then:
  - Market columns prefixed with "m_" (m_e0, m_e1, ...)
  - Sector columns prefixed with "s_" (s_e0, s_e1, ...)
  - No column name collisions
```

### 2.2 Data Pipeline Integration

**Test: `test_full_data_pipeline`**

```
Given: Raw CSV data from ../../data/all_data_*.csv
When: Data flows through load -> features -> future_returns -> dataset
Then:
  - No NaN values in features (filled with 0)
  - No inf values in features
  - Target column exists and has valid values
  - Dataset produces valid (tensor, label) pairs
```

### 2.3 Training Integration

**Test: `test_single_epoch_training`**

```
Given: Real data, model, optimizer, loss function
When: train_model() runs for 1 epoch
Then:
  - Loss decreases or stays reasonable (< 10.0)
  - Accuracy is between 0 and 1 for classification
  - No NaN in loss
  - Model weights are updated
```

---

## 3. End-to-End Tests

### 3.1 Single Configuration E2E

**Test: `test_e2e_single_config`**

```bash
python -m stockformer.main \
    --data-path "../../data/all_data_*.csv" \
    --horizon 3 \
    --label-mode binary \
    --epochs 1 \
    --model-out test_model.pt \
    --output-csv test_predictions.csv
```

**Verify:**
- [ ] Model checkpoint created (`test_model.pt`)
- [ ] Predictions CSV created (`test_predictions.csv`)
- [ ] Predictions CSV has columns: ticker, date, pred_class, pred_expected_return, true_class
- [ ] pred_class values are 0 or 1
- [ ] No NaN in predictions

### 3.2 All Configurations E2E

**Test: `test_e2e_all_horizons`**

```bash
python -m stockformer.main \
    --data-path "../../data/all_data_*.csv" \
    --epochs 1
```

**Verify:**
- [ ] 9 model checkpoints created (reg_3d, reg_10d, reg_30d, bin_3d, ...)
- [ ] 9 prediction CSVs created
- [ ] Each has appropriate output columns for its label_mode

### 3.3 Reproducibility Test

**Test: `test_reproducibility`**

```
Given: Same data, same config, same random seed
When: Run training twice
Then:
  - Same final loss (within 1e-6)
  - Same model weights
```

---

## 4. Proposed Logging Features

### 4.1 Training Metrics Logging

Add to `training.py`:

```python
# Metrics to log per epoch
METRICS_TO_LOG = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_acc",
    "val_acc",
    "learning_rate",      # NEW: track LR for schedulers
    "epoch_time_sec",     # NEW: timing
    "gpu_memory_mb",      # NEW: memory usage (if CUDA)
]
```

### 4.2 Model Logging

Add to `model.py`:

```python
def log_model_summary(model):
    """Log model architecture and parameter count."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": total_params * 4 / (1024 * 1024),  # float32
    }
```

### 4.3 Data Logging

Add to `data_utils.py`:

```python
def log_data_summary(df, feature_cols, target_col):
    """Log data statistics for debugging."""
    return {
        "num_rows": len(df),
        "num_tickers": df["ticker"].nunique(),
        "date_range": (df["date"].min(), df["date"].max()),
        "num_features": len(feature_cols),
        "target_stats": {
            "mean": df[target_col].mean(),
            "std": df[target_col].std(),
            "min": df[target_col].min(),
            "max": df[target_col].max(),
            "pct_positive": (df[target_col] > 0).mean(),
        },
        "missing_values": df[feature_cols].isna().sum().to_dict(),
    }
```

### 4.4 Logging Output Format

Create new file `stockformer/logging_utils.py`:

```python
"""
Logging utilities.

Outputs:
  - Console: progress bars, epoch summaries
  - CSV: training_log.csv (epoch metrics)
  - JSON: run_config.json (full config for reproducibility)
"""

import json
import logging
from datetime import datetime

def setup_logging(log_dir="logs"):
    """Setup logging to console and file."""
    pass

def log_run_config(cfg, path="run_config.json"):
    """Save full config to JSON for reproducibility."""
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)

def log_epoch(epoch, metrics, log_path):
    """Append epoch metrics to CSV."""
    pass
```

---

## 5. Proposed Visualization Features

### 5.1 Training Curves

Create new file `stockformer/visualization.py`:

```python
"""
Visualization utilities for training analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(log_path, output_path="training_curves.png"):
    """
    Plot loss and accuracy curves from training log.

    Creates 2x1 subplot:
      - Top: Train/Val Loss vs Epoch
      - Bottom: Train/Val Accuracy vs Epoch
    """
    df = pd.read_csv(log_path)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Loss plot
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss")
    axes[0].plot(df["epoch"], df["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].set_title("Training and Validation Loss")

    # Accuracy plot
    axes[1].plot(df["epoch"], df["train_acc"], label="Train Acc")
    axes[1].plot(df["epoch"], df["val_acc"], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].set_title("Training and Validation Accuracy")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_prediction_distribution(pred_csv, output_path="pred_distribution.png"):
    """
    Plot distribution of predictions vs actuals.

    For classification: confusion matrix heatmap
    For regression: scatter plot with diagonal line
    """
    pass


def plot_returns_by_prediction(pred_csv, output_path="returns_by_pred.png"):
    """
    Plot actual returns grouped by predicted class.

    Shows if model predictions correlate with actual returns.
    Box plot or violin plot of actual returns per predicted bucket.
    """
    pass


def plot_feature_importance(model, feature_cols, output_path="feature_importance.png"):
    """
    Approximate feature importance using gradient-based attribution.

    Bar chart of average absolute gradient per feature.
    """
    pass
```

### 5.2 Prediction Analysis

```python
def plot_cumulative_returns(pred_csv, output_path="cumulative_returns.png"):
    """
    Plot cumulative returns of a simple strategy:
      - Go long when pred_class == 1 (or pred_expected_return > 0)
      - Compare to buy-and-hold baseline

    Line chart: Strategy vs Baseline cumulative return over time.
    """
    pass


def plot_confusion_matrix(pred_csv, output_path="confusion_matrix.png"):
    """
    Plot confusion matrix for classification predictions.

    Heatmap with true labels on y-axis, predicted on x-axis.
    """
    pass


def plot_calibration_curve(pred_csv, output_path="calibration.png"):
    """
    Plot calibration curve for binary classification.

    Shows if predicted probabilities match actual frequencies.
    """
    pass
```

### 5.3 Visualization CLI Integration

Add to `main.py`:

```python
parser.add_argument(
    "--plot",
    action="store_true",
    help="Generate visualization plots after training",
)

# After training completes:
if cfg.get("plot"):
    from .visualization import plot_training_curves, plot_prediction_distribution
    plot_training_curves(cfg["log_path"])
    plot_prediction_distribution(cfg["output_csv"])
```

---

## 6. Test Execution Plan

### Phase 1: Unit Tests (Run First)
```bash
pytest stockformer/tests/test_config.py -v
pytest stockformer/tests/test_data_utils.py -v
pytest stockformer/tests/test_features.py -v
pytest stockformer/tests/test_dataset.py -v
pytest stockformer/tests/test_model.py -v
pytest stockformer/tests/test_losses.py -v
pytest stockformer/tests/test_training.py -v
```

### Phase 2: Integration Tests
```bash
pytest stockformer/tests/test_integration.py -v
```

### Phase 3: E2E Tests (Requires Real Data)
```bash
pytest stockformer/tests/test_e2e.py -v --data-path "../../data/all_data_*.csv"
```

### Quick Smoke Test
```bash
# Single command to verify everything works
python -m stockformer.main \
    --data-path "../../data/all_data_*.csv" \
    --horizon 3 \
    --label-mode binary \
    --epochs 1 \
    --batch-size 64
```

---

## 7. Test File Structure

```
stockformer/
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Shared fixtures (sample data, configs)
│   ├── test_config.py
│   ├── test_data_utils.py
│   ├── test_features.py
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_losses.py
│   ├── test_training.py
│   ├── test_integration.py   # Embedding auto-creation, data pipeline
│   └── test_e2e.py           # Full pipeline tests
├── logging_utils.py          # NEW: logging utilities
└── visualization.py          # NEW: plotting utilities
```

---

## 8. Success Criteria

| Category | Criteria |
|----------|----------|
| Unit Tests | 100% pass, >80% code coverage |
| Integration | Embedding auto-creation works |
| E2E | 1-epoch training completes without error |
| Performance | Training on full data completes in <5 min/epoch |
| Logging | All metrics captured in CSV |
| Visualization | Plots generated without error |
