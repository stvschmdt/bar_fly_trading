"""
Inference module for the StockFormer pipeline.

This module handles all inference-related functionality:
    - Running predictions with trained models
    - Output formatting (classification, expected value, probabilities)
    - Date filtering for inference data
    - Bucket midpoint calculations

Functions:
    - compute_bucket_midpoints: Get midpoint values for each bucket
    - softmax_to_expected_value: Convert probabilities to expected return
    - format_predictions: Apply output mode to model outputs
    - filter_data_by_date: Filter dataframe by date range
    - infer: Main inference function
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import (
        confusion_matrix, classification_report, roc_auc_score,
        precision_recall_fscore_support,
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from .config import get_feature_columns, get_target_column, OUTPUT_COLUMN_GROUPS, BASE_FEATURE_COLUMNS
from .data_utils import (
    load_panel_csvs,
    add_future_returns,
    load_embeddings,
    merge_embeddings,
)
from .features import add_all_features
from .sector_features import extract_etf_features, add_sector_features, exclude_etf_symbols
from .dataset import StockSequenceDataset
from .model import create_model
from .logging_utils import Timer


# =============================================================================
# Bucket Midpoint Calculations
# =============================================================================

def compute_bucket_midpoints(bucket_edges, edge_extension=0.01):
    """
    Compute midpoint values for each bucket.

    For bucket classification, we need a representative value for each bucket
    to compute expected returns. This function calculates those midpoints.

    Args:
        bucket_edges: List of edges as fractions (e.g., [-0.02, 0, 0.02])
        edge_extension: How far beyond first/last edge for extreme buckets

    Returns:
        Array of midpoints, one per bucket (len = len(edges) + 1)

    Example:
        edges = [-0.02, 0, 0.02]  # 3 edges -> 4 buckets
        midpoints = [-0.03, -0.01, 0.01, 0.03]

        Bucket 0: (-inf, -0.02) -> midpoint -0.03
        Bucket 1: [-0.02, 0)    -> midpoint -0.01
        Bucket 2: [0, 0.02)     -> midpoint  0.01
        Bucket 3: [0.02, +inf)  -> midpoint  0.03
    """
    edges = np.array(bucket_edges, dtype=np.float32)
    midpoints = []

    # First bucket: extends to -infinity, use first_edge - extension
    midpoints.append(edges[0] - edge_extension)

    # Middle buckets: average of adjacent edges
    for i in range(len(edges) - 1):
        midpoints.append((edges[i] + edges[i + 1]) / 2.0)

    # Last bucket: extends to +infinity, use last_edge + extension
    midpoints.append(edges[-1] + edge_extension)

    return np.array(midpoints, dtype=np.float32)


def get_midpoints_for_label_mode(label_mode, bucket_edges):
    """
    Get bucket midpoints based on label mode.

    Args:
        label_mode: "regression", "binary", or "buckets"
        bucket_edges: List of bucket edges (percent, e.g., [-6, -4, -2, 0, 2, 4, 6])

    Returns:
        Array of midpoints, or None for regression
    """
    if label_mode == "regression":
        return None
    elif label_mode == "binary":
        # Binary: class 0 = negative return, class 1 = positive return
        return np.array([-0.01, 0.01], dtype=np.float32)
    elif label_mode == "buckets" and bucket_edges is not None:
        # Convert from percent to fraction
        edges = np.array(bucket_edges, dtype=np.float32) / 100.0
        return compute_bucket_midpoints(edges)
    else:
        return None


# =============================================================================
# Output Conversion Functions
# =============================================================================

def softmax_to_expected_value(probs, midpoints):
    """
    Convert probability matrix to expected values using bucket midpoints.

    This computes: E[return] = sum(prob_i * midpoint_i)

    Args:
        probs: Array of shape [N, num_classes] with probabilities
        midpoints: Array of shape [num_classes] with bucket midpoint values

    Returns:
        Array of shape [N] with expected return values
    """
    return (probs * midpoints.reshape(1, -1)).sum(axis=1)


def format_predictions(probs, labels, label_mode, output_mode, bucket_edges=None):
    """
    Format model outputs based on output mode.

    Args:
        probs: Probability array [N, num_classes] or regression values [N]
        labels: True labels array [N]
        label_mode: "regression", "binary", or "buckets"
        output_mode: "classification", "expected_value", "probabilities", "all"
        bucket_edges: List of bucket edges in percent (for buckets mode)

    Returns:
        Dict mapping column names to arrays
    """
    result = {}

    # Regression mode: output is already the predicted value
    if label_mode == "regression":
        result["pred_return"] = probs
        result["true_return"] = labels
        return result

    # Classification modes (binary or buckets)
    pred_class = np.argmax(probs, axis=1)
    midpoints = get_midpoints_for_label_mode(label_mode, bucket_edges)

    # Build output based on mode
    if output_mode == "classification":
        result["pred_class"] = pred_class
        result["true_class"] = labels

    elif output_mode == "expected_value":
        result["pred_class"] = pred_class
        if midpoints is not None:
            result["pred_expected_return"] = softmax_to_expected_value(probs, midpoints)
        result["true_class"] = labels

    elif output_mode == "probabilities":
        result["pred_class"] = pred_class
        for i in range(probs.shape[1]):
            result[f"prob_{i}"] = probs[:, i]
        result["true_class"] = labels

    elif output_mode == "all":
        result["pred_class"] = pred_class
        if midpoints is not None:
            result["pred_expected_return"] = softmax_to_expected_value(probs, midpoints)
        for i in range(probs.shape[1]):
            result[f"prob_{i}"] = probs[:, i]
        result["true_class"] = labels

    else:
        raise ValueError(f"Unknown output_mode: {output_mode}")

    return result


# =============================================================================
# Data Filtering
# =============================================================================

def filter_data_by_date(df, start_date=None, end_date=None, date_col="date"):
    """
    Filter dataframe by date range.

    Args:
        df: DataFrame with date column
        start_date: Start date string (YYYY-MM-DD) or None for no lower bound
        end_date: End date string (YYYY-MM-DD) or None for no upper bound
        date_col: Name of date column

    Returns:
        Filtered DataFrame
    """
    if start_date is None and end_date is None:
        return df

    df = df.copy()
    original_len = len(df)

    if start_date is not None:
        df = df[df[date_col] >= start_date]

    if end_date is not None:
        df = df[df[date_col] <= end_date]

    filtered_len = len(df)
    print(f"Date filter: {original_len:,} -> {filtered_len:,} rows")

    if start_date:
        print(f"  Start date: {start_date}")
    if end_date:
        print(f"  End date: {end_date}")

    return df


# =============================================================================
# Output Column Filtering
# =============================================================================

def get_output_columns(column_groups_str, available_columns):
    """
    Build list of columns to include in output based on column groups.

    Args:
        column_groups_str: Comma-separated group names (e.g., "core,signals")
                          or "all" to include everything
        available_columns: List of columns present in the dataframe

    Returns:
        List of column names to include, or None if "all"
    """
    if column_groups_str == "all":
        return None  # Include all columns

    groups = [g.strip() for g in column_groups_str.split(",")]
    columns = []

    for group in groups:
        if group in OUTPUT_COLUMN_GROUPS:
            # Add columns that exist in the dataframe
            for col in OUTPUT_COLUMN_GROUPS[group]:
                if col in available_columns and col not in columns:
                    columns.append(col)
        else:
            print(f"Warning: Unknown column group '{group}', skipping")

    return columns if columns else None


# =============================================================================
# Main Inference Function
# =============================================================================

def infer(cfg):
    """
    Run inference using a trained model.

    Args:
        cfg: Configuration dict with keys:
            - data_path: Path to CSV data
            - model_out: Path to trained model checkpoint
            - output_csv: Path to save predictions
            - horizon, label_mode, bucket_edges: Model configuration
            - output_mode: "classification", "expected_value", "probabilities", "all"
            - infer_start_date, infer_end_date: Optional date filters
            - batch_size, num_workers, device: Runtime settings

    Returns:
        DataFrame with predictions
    """
    print("=" * 60)
    print("INFERENCE")
    print("=" * 60)

    with Timer("Total inference") as total_timer:

        # Check model exists
        if not os.path.exists(cfg["model_out"]):
            raise FileNotFoundError(f"Model not found: {cfg['model_out']}")

        # Load data
        print("\nLoading data for inference...")
        with Timer("Data loading") as load_timer:
            df = load_panel_csvs(cfg["data_path"])

            # Apply date filtering if specified
            start_date = cfg.get("infer_start_date")
            end_date = cfg.get("infer_end_date")
            if start_date or end_date:
                df = filter_data_by_date(df, start_date, end_date)

            target_col = get_target_column(cfg["horizon"])

            # Always recompute future returns to match training scale (decimal form).
            # Raw CSVs may have pre-computed values in percentage form which would
            # cause true_return/true_class to use wrong scale in output.
            print(f"Computing future returns for horizon={cfg['horizon']}...")
            df = add_future_returns(df, horizons=[cfg["horizon"]])

            # Add technical features
            print("Adding technical features...")
            df = add_all_features(df)

            # Add sector/market ETF features
            print("Extracting sector/market ETF features...")
            etf_features = extract_etf_features(df)
            df = add_sector_features(df, etf_features)
            df = exclude_etf_symbols(df)

            # Load embeddings
            print("Loading embeddings...")
            market_result = None
            sector_result = None

            if cfg.get("market_path"):
                market_result = load_embeddings(cfg["market_path"], prefix="m_", base_df=df)

            if cfg.get("sector_path"):
                sector_result = load_embeddings(cfg["sector_path"], prefix="s_", base_df=df)

            if cfg["mode"] == "correlated":
                df = merge_embeddings(df, market_result, sector_result)

        print(f"Data loading time: {load_timer}")

        # ── Peek at checkpoint to auto-detect architecture ──────────────
        # This must happen BEFORE auto bucket edges and dataset creation
        # so that num_buckets and other params are correct for the data pipeline.
        device = cfg.get("device")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        print(f"Loading model from: {cfg['model_out']}")
        state_dict = torch.load(cfg["model_out"], map_location=device, weights_only=True)

        from .model import infer_arch_from_state_dict
        detected = infer_arch_from_state_dict(state_dict, model_path=cfg["model_out"])
        if detected:
            overrides = []
            for key in ("d_model", "nhead", "num_layers", "dim_feedforward"):
                if key in detected and detected[key] != cfg.get(key):
                    overrides.append(f"{key}: {cfg.get(key)} -> {detected[key]}")
                    cfg[key] = detected[key]
            if overrides:
                print(f"Auto-detected architecture from checkpoint: {', '.join(overrides)}")

        # For buckets mode, override n_buckets from checkpoint before auto-edge computation
        if cfg["label_mode"] == "buckets" and "num_buckets" in detected:
            cfg["n_buckets"] = detected["num_buckets"]
            cfg["_num_buckets_override"] = detected["num_buckets"]

        # ── Auto-compute quantile bucket edges ──────────────────────────
        if cfg["label_mode"] == "buckets" and cfg.get("bucket_edges") == "auto":
            from .dataset import compute_quantile_edges
            n_buckets = cfg.get("n_buckets", 4)
            cfg["bucket_edges"] = compute_quantile_edges(df, target_col, n_buckets=n_buckets)
            print(f"Auto bucket edges ({n_buckets} buckets): {cfg['bucket_edges']}")

        # Get feature columns based on model type
        model_type = cfg.get("model_type", "encoder")
        output_mode = cfg.get("output_mode", "all")

        if model_type == "cross_attention":
            # For cross-attention: stock features are base features only
            stock_feature_cols = BASE_FEATURE_COLUMNS.copy()
            market_feature_cols = [col for col in df.columns if col.startswith("m_") or col.startswith("s_")]
            feature_cols = stock_feature_cols
            print(f"Cross-attention mode: {len(stock_feature_cols)} stock features, {len(market_feature_cols)} market features")
        else:
            feature_cols = get_feature_columns(df, cfg["mode"])
            market_feature_cols = None

        print(f"Features: {len(feature_cols)} columns")
        print(f"Output mode: {output_mode}")

        # Create dataset (now with correct bucket_edges from checkpoint)
        dataset = StockSequenceDataset(
            df=df,
            lookback=cfg["lookback"],
            target_col=target_col,
            feature_cols=feature_cols,
            label_mode=cfg["label_mode"],
            bucket_edges=cfg.get("bucket_edges"),
            market_feature_cols=market_feature_cols,
            binary_threshold=cfg.get("binary_threshold", 0.0),
            min_return_threshold=cfg.get("min_return_threshold", 0.0),
        )

        print(f"Inference samples: {len(dataset)}")

        loader = DataLoader(
            dataset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
        )

        # ── Create model and load weights ───────────────────────────────
        if model_type == "cross_attention" and market_feature_cols:
            cfg["market_feature_dim"] = len(market_feature_cols)

        model = create_model(
            feature_dim=len(feature_cols),
            label_mode=cfg["label_mode"],
            bucket_edges=cfg.get("bucket_edges"),
            cfg=cfg,
            model_type=model_type,
        ).to(device)

        model.load_state_dict(state_dict)
        model.eval()

        # Run inference
        print("\nRunning inference...")
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                # Handle different batch formats based on model type
                if model_type == "cross_attention":
                    batch_x, batch_market, batch_y = batch
                    batch_x = batch_x.to(device)
                    batch_market = batch_market.to(device)
                else:
                    batch_x, batch_y = batch
                    batch_x = batch_x.to(device)

                if cfg["label_mode"] == "regression":
                    batch_y = batch_y.float()
                else:
                    batch_y = batch_y.long()

                if model_type == "cross_attention":
                    outputs = model(batch_x, batch_market)
                else:
                    outputs = model(batch_x)

                if cfg["label_mode"] == "regression":
                    # Regression: output is the prediction
                    all_probs.append(outputs.cpu().numpy())
                else:
                    # Classification: get softmax probabilities
                    probs = F.softmax(outputs, dim=-1)
                    all_probs.append(probs.cpu().numpy())

                all_labels.append(batch_y.numpy())

        # Concatenate results
        probs_arr = np.concatenate(all_probs, axis=0)
        labels_arr = np.concatenate(all_labels, axis=0)

        # Format predictions based on output mode
        pred_dict = format_predictions(
            probs=probs_arr,
            labels=labels_arr,
            label_mode=cfg["label_mode"],
            output_mode=output_mode,
            bucket_edges=cfg.get("bucket_edges"),
        )

    # Build output dataframe
    out_df = df.iloc[dataset.indices].copy()
    out_df = out_df.reset_index(drop=True)

    # Add prediction columns
    for col_name, values in pred_dict.items():
        out_df[col_name] = values

    # Filter columns based on output_columns setting
    output_columns_str = cfg.get("output_columns", "all")
    selected_cols = get_output_columns(output_columns_str, out_df.columns.tolist())

    if selected_cols is not None:
        # Get prediction columns (always include these)
        pred_cols = [c for c in out_df.columns if c.startswith(("pred_", "true_"))]
        # Combine selected input columns with prediction columns
        final_cols = selected_cols + [c for c in pred_cols if c not in selected_cols]
        out_df = out_df[final_cols]
        print(f"Output columns filtered: {len(final_cols)} columns ({output_columns_str})")

    # Save predictions
    output_path = cfg.get("output_csv", "predictions.csv")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    # Print summary
    print(f"\nInference time: {total_timer}")
    print(f"Predictions saved to: {output_path}")
    print(f"  Rows: {len(out_df)}")
    print(f"  Prediction columns: {list(pred_dict.keys())}")

    # Print evaluation report
    print_eval_report(pred_dict, cfg.get("label_mode", "binary"))

    return out_df


# =============================================================================
# Evaluation Report
# =============================================================================

def print_eval_report(pred_dict, label_mode):
    """
    Print comprehensive evaluation metrics after inference.

    For classification: confusion matrix, precision/recall/F1, ROC-AUC,
    class distribution, and collapse detection.
    For regression: correlation, direction accuracy, mean bias.
    """
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    if label_mode == "regression":
        _eval_regression(pred_dict)
    elif "pred_class" in pred_dict and "true_class" in pred_dict:
        _eval_classification(pred_dict, label_mode)
    else:
        print("  No evaluation data available.")

    print("=" * 60 + "\n")


def _eval_classification(pred_dict, label_mode):
    """Evaluate classification predictions."""
    pred = pred_dict["pred_class"].astype(int)
    true = pred_dict["true_class"].astype(int)
    n = len(pred)

    # Basic accuracy
    accuracy = (pred == true).mean()
    print(f"\n  Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print(f"  Samples:  {n:,}")

    # Class distribution comparison
    unique_true, counts_true = np.unique(true, return_counts=True)
    unique_pred, counts_pred = np.unique(pred, return_counts=True)
    true_dist = dict(zip(unique_true.astype(int), counts_true))
    pred_dist = dict(zip(unique_pred.astype(int), counts_pred))

    all_classes = sorted(set(true_dist.keys()) | set(pred_dist.keys()))
    print(f"\n  Class Distribution:")
    print(f"  {'Class':<8} {'True':>10} {'True%':>8} {'Pred':>10} {'Pred%':>8}")
    print(f"  {'-'*46}")
    for c in all_classes:
        tc = true_dist.get(c, 0)
        pc = pred_dist.get(c, 0)
        print(f"  {c:<8} {tc:>10,} {tc/n:>7.1%} {pc:>10,} {pc/n:>7.1%}")

    # Collapse detection
    dominant_pred_pct = max(counts_pred) / n if len(counts_pred) > 0 else 0
    if dominant_pred_pct > 0.90:
        dominant_class = unique_pred[np.argmax(counts_pred)]
        print(f"\n  *** COLLAPSED: {dominant_pred_pct:.1%} of predictions are class {dominant_class} ***")
    elif dominant_pred_pct > 0.75:
        dominant_class = unique_pred[np.argmax(counts_pred)]
        print(f"\n  ** WARNING: {dominant_pred_pct:.1%} of predictions are class {dominant_class} **")

    if not HAS_SKLEARN:
        print("\n  (Install scikit-learn for confusion matrix, F1, ROC-AUC)")
        return

    # Confusion matrix
    cm = confusion_matrix(true, pred, labels=all_classes)
    print(f"\n  Confusion Matrix (rows=true, cols=pred):")
    header = "  " + " " * 8 + "".join(f"pred_{c:>3}" for c in all_classes)
    print(header)
    for i, c in enumerate(all_classes):
        row = "  " + f"true_{c:<3}" + "".join(f"{cm[i, j]:>7,}" for j in range(len(all_classes)))
        print(row)

    # Precision, Recall, F1
    prec, rec, f1, support = precision_recall_fscore_support(
        true, pred, labels=all_classes, zero_division=0
    )
    print(f"\n  Per-Class Metrics:")
    print(f"  {'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*50}")
    for i, c in enumerate(all_classes):
        print(f"  {c:<8} {prec[i]:>10.4f} {rec[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10,}")

    # Macro / weighted averages
    macro_f1 = f1.mean()
    weighted_f1 = np.average(f1, weights=support)
    print(f"\n  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")

    # ROC-AUC (binary only, needs probabilities)
    if label_mode == "binary" and "prob_1" in pred_dict:
        try:
            auc = roc_auc_score(true, pred_dict["prob_1"])
            print(f"  ROC-AUC:     {auc:.4f}")
        except ValueError as e:
            print(f"  ROC-AUC:     N/A ({e})")

    # Probability calibration summary (if available)
    prob_cols = [k for k in pred_dict if k.startswith("prob_")]
    if prob_cols:
        print(f"\n  Probability Stats:")
        for col in sorted(prob_cols):
            vals = pred_dict[col]
            print(f"    {col}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
                  f"min={vals.min():.4f}  max={vals.max():.4f}")


def _eval_regression(pred_dict):
    """Evaluate regression predictions."""
    pred = pred_dict.get("pred_return")
    true = pred_dict.get("true_return")
    if pred is None or true is None:
        print("  No regression data available.")
        return

    pred = pred.flatten()
    true = true.flatten()
    n = len(pred)

    # Basic stats
    mse = np.mean((pred - true) ** 2)
    mae = np.mean(np.abs(pred - true))
    print(f"\n  Samples: {n:,}")
    print(f"  MSE:     {mse:.6f}")
    print(f"  MAE:     {mae:.6f}")

    # Correlation
    if np.std(pred) > 1e-10 and np.std(true) > 1e-10:
        corr = np.corrcoef(pred, true)[0, 1]
    else:
        corr = 0.0
    print(f"  Correlation: {corr:.4f}")

    # Direction accuracy
    pred_dir = np.sign(pred)
    true_dir = np.sign(true)
    # Exclude zero-return samples from direction calc
    nonzero = true_dir != 0
    if nonzero.sum() > 0:
        dir_acc = (pred_dir[nonzero] == true_dir[nonzero]).mean()
        print(f"  Direction Accuracy: {dir_acc:.4f} ({dir_acc:.2%})")
    else:
        print(f"  Direction Accuracy: N/A (all true returns are zero)")

    # Bias
    pred_mean = pred.mean()
    true_mean = true.mean()
    print(f"\n  Pred Mean:  {pred_mean:+.6f}")
    print(f"  True Mean:  {true_mean:+.6f}")
    print(f"  Bias:       {pred_mean - true_mean:+.6f}")

    # Collapse detection for regression
    pred_std = np.std(pred)
    if pred_std < 1e-5:
        print(f"\n  *** COLLAPSED: pred std={pred_std:.8f} (near-constant predictions) ***")
    elif pred_std < np.std(true) * 0.1:
        print(f"\n  ** WARNING: pred std={pred_std:.6f} << true std={np.std(true):.6f} "
              f"(predictions lack variance) **")
