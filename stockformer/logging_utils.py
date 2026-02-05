"""
Logging utilities for the StockFormer pipeline.

Functions:
    - log_data_summary: Log dataset statistics
    - log_model_summary: Log model architecture info
    - log_training_summary: Log training results
    - get_gpu_memory: Get GPU memory usage
    - Timer: Context manager for timing code blocks
"""

import json
import os
import time
from datetime import datetime

import torch


# =============================================================================
# Timer Utility
# =============================================================================

class Timer:
    """
    Context manager for timing code blocks.

    Usage:
        with Timer("Training") as t:
            train_model(...)
        print(f"Training took {t.elapsed:.1f}s")
    """

    def __init__(self, name=""):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time

    def __str__(self):
        if self.elapsed >= 60:
            minutes = int(self.elapsed // 60)
            seconds = self.elapsed % 60
            return f"{minutes}m {seconds:.1f}s"
        return f"{self.elapsed:.1f}s"


# =============================================================================
# GPU Memory Tracking
# =============================================================================

def get_gpu_memory():
    """
    Get GPU memory usage in MB.

    Returns:
        Dict with 'allocated' and 'reserved' MB, or None if no GPU.
    """
    if not torch.cuda.is_available():
        return None

    try:
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)

        return {
            "allocated_mb": round(allocated, 1),
            "reserved_mb": round(reserved, 1),
            "max_allocated_mb": round(max_allocated, 1),
        }
    except Exception:
        return None


def reset_gpu_memory_stats():
    """Reset GPU memory tracking stats."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# =============================================================================
# Data Summary Logging
# =============================================================================

def log_data_summary(df, feature_cols, target_col):
    """
    Log dataset statistics.

    Args:
        df: DataFrame with stock data
        feature_cols: List of feature column names
        target_col: Target column name

    Returns:
        Dict with data statistics
    """
    summary = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "num_features": len(feature_cols),
        "num_tickers": df["ticker"].nunique() if "ticker" in df.columns else None,
        "date_range": None,
        "target_stats": None,
    }

    # Date range
    if "date" in df.columns:
        summary["date_range"] = {
            "min": str(df["date"].min()),
            "max": str(df["date"].max()),
            "num_dates": df["date"].nunique(),
        }

    # Target statistics
    if target_col in df.columns:
        target = df[target_col].dropna()
        summary["target_stats"] = {
            "mean": round(float(target.mean()), 6),
            "std": round(float(target.std()), 6),
            "min": round(float(target.min()), 6),
            "max": round(float(target.max()), 6),
            "pct_positive": round(float((target > 0).mean()), 4),
            "pct_missing": round(float(df[target_col].isna().mean()), 4),
        }

    return summary


def print_data_summary(summary):
    """Print data summary to console."""
    print("\n" + "-" * 40)
    print("DATA SUMMARY")
    print("-" * 40)
    print(f"  Rows: {summary['num_rows']:,}")
    print(f"  Features: {summary['num_features']}")

    if summary["num_tickers"]:
        print(f"  Tickers: {summary['num_tickers']}")

    if summary["date_range"]:
        dr = summary["date_range"]
        print(f"  Date range: {dr['min']} to {dr['max']} ({dr['num_dates']} days)")

    if summary["target_stats"]:
        ts = summary["target_stats"]
        print(f"  Target mean: {ts['mean']:.4f} (std: {ts['std']:.4f})")
        print(f"  Target range: [{ts['min']:.4f}, {ts['max']:.4f}]")
        print(f"  Positive samples: {ts['pct_positive']:.1%}")


# =============================================================================
# Model Summary Logging
# =============================================================================

def log_model_summary(model):
    """
    Log model architecture and parameter count.

    Args:
        model: PyTorch model

    Returns:
        Dict with model statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_size_mb": round(total_params * 4 / (1024 * 1024), 2),
    }


def print_model_summary(summary):
    """Print model summary to console."""
    print(f"  Model parameters: {summary['total_parameters']:,}")
    print(f"  Model size: {summary['model_size_mb']:.2f} MB")


# =============================================================================
# Training Summary Logging
# =============================================================================

def log_training_summary(history, train_time, device, gpu_memory=None):
    """
    Create training summary dict.

    Args:
        history: Training history dict
        train_time: Total training time in seconds
        device: Device used (cuda/cpu)
        gpu_memory: GPU memory stats dict (optional)

    Returns:
        Dict with training summary
    """
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_epochs": len(history["epoch"]),
        "total_time_sec": round(train_time, 2),
        "device": str(device),
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "final_val_loss": history["val_loss"][-1] if history["val_loss"] else None,
        "final_train_acc": history["train_acc"][-1] if history["train_acc"] else None,
        "final_val_acc": history["val_acc"][-1] if history["val_acc"] else None,
        "best_val_loss": min(history["val_loss"]) if history["val_loss"] else None,
        "best_val_acc": max(history["val_acc"]) if history["val_acc"] else None,
    }

    if gpu_memory:
        summary["gpu_memory"] = gpu_memory

    return summary


def print_training_summary(summary):
    """Print training summary to console."""
    print("\n" + "=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)

    # Time
    time_sec = summary["total_time_sec"]
    if time_sec >= 60:
        minutes = int(time_sec // 60)
        seconds = time_sec % 60
        time_str = f"{minutes}m {seconds:.1f}s"
    else:
        time_str = f"{time_sec:.1f}s"

    print(f"  Total time: {time_str}")
    print(f"  Epochs: {summary['total_epochs']}")
    print(f"  Device: {summary['device']}")

    # Final metrics
    if summary["final_val_loss"] is not None:
        print(f"  Final val loss: {summary['final_val_loss']:.4f}")
    if summary["final_val_acc"] is not None:
        print(f"  Final val accuracy: {summary['final_val_acc']:.2%}")

    # Best metrics
    if summary["best_val_loss"] is not None:
        print(f"  Best val loss: {summary['best_val_loss']:.4f}")
    if summary["best_val_acc"] is not None:
        print(f"  Best val accuracy: {summary['best_val_acc']:.2%}")

    # GPU memory
    if summary.get("gpu_memory"):
        gm = summary["gpu_memory"]
        print(f"  GPU memory (peak): {gm['max_allocated_mb']:.0f} MB")


# =============================================================================
# Config Saving
# =============================================================================

def save_run_config(cfg, path):
    """
    Save configuration to JSON for reproducibility.

    Args:
        cfg: Config dict
        path: Output path for JSON file
    """
    # Make a copy and convert any non-serializable types
    cfg_copy = {}
    for k, v in cfg.items():
        if v is None or isinstance(v, (str, int, float, bool, list)):
            cfg_copy[k] = v
        else:
            cfg_copy[k] = str(v)

    cfg_copy["saved_at"] = datetime.now().isoformat()

    with open(path, "w") as f:
        json.dump(cfg_copy, f, indent=2)

    print(f"Saved run config to {path}")
