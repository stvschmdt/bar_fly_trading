#!/usr/bin/env python
"""
Smoke test for the stockformer module.

Runs a quick 1-epoch training and inference cycle to verify everything works.

Usage:
    python stockformer/smoke_test.py --data-path "../../data/all_data_*.csv"

    # Or as module:
    python -m stockformer.smoke_test --data-path "../../data/all_data_*.csv"
"""

import argparse
import os
import sys
import tempfile
import shutil
import time

# Ensure parent directory is in path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from stockformer.config import DEFAULT_CONFIG
from stockformer.main import train, infer


def run_smoke_test(data_path, with_embeddings=False):
    """
    Run a quick smoke test of the stockformer pipeline.

    Args:
        data_path: Path to CSV data files
        with_embeddings: If True, also test embedding auto-creation
    """
    print("=" * 60)
    print("STOCKFORMER SMOKE TEST")
    print("=" * 60)

    # Create temp directory for outputs
    temp_dir = tempfile.mkdtemp()
    print(f"\nTemp directory: {temp_dir}")

    try:
        # Setup config
        cfg = DEFAULT_CONFIG.copy()
        cfg["data_path"] = data_path
        cfg["horizon"] = 3
        cfg["label_mode"] = "binary"
        cfg["num_epochs"] = 1
        cfg["batch_size"] = 64
        cfg["d_model"] = 64
        cfg["nhead"] = 2
        cfg["num_layers"] = 2
        cfg["device"] = "cpu"
        cfg["num_workers"] = 0
        cfg["model_out"] = os.path.join(temp_dir, "model.pt")
        cfg["log_path"] = os.path.join(temp_dir, "log.csv")
        cfg["output_csv"] = os.path.join(temp_dir, "predictions.csv")

        if with_embeddings:
            cfg["mode"] = "correlated"
            cfg["market_path"] = os.path.join(temp_dir, "market_emb.csv")
        else:
            cfg["mode"] = "single"

        # Run training
        print("\n" + "-" * 40)
        print("STEP 1: Training (1 epoch)")
        print("-" * 40)

        start_time = time.time()
        train(cfg)
        train_time = time.time() - start_time

        # Verify training outputs
        assert os.path.exists(cfg["model_out"]), "Model checkpoint not created"
        assert os.path.exists(cfg["log_path"]), "Training log not created"

        log_df = pd.read_csv(cfg["log_path"])
        print(f"\nTraining completed in {train_time:.1f}s")
        print(f"  Train loss: {log_df['train_loss'].iloc[-1]:.4f}")
        print(f"  Val loss: {log_df['val_loss'].iloc[-1]:.4f}")
        print(f"  Train acc: {log_df['train_acc'].iloc[-1]:.2%}")
        print(f"  Val acc: {log_df['val_acc'].iloc[-1]:.2%}")

        # Run inference
        print("\n" + "-" * 40)
        print("STEP 2: Inference")
        print("-" * 40)

        start_time = time.time()
        infer(cfg)
        infer_time = time.time() - start_time

        # Verify inference outputs
        assert os.path.exists(cfg["output_csv"]), "Predictions not created"

        pred_df = pd.read_csv(cfg["output_csv"])
        accuracy = (pred_df["pred_class"] == pred_df["true_class"]).mean()

        print(f"\nInference completed in {infer_time:.1f}s")
        print(f"  Predictions: {len(pred_df)} rows")
        print(f"  Accuracy: {accuracy:.2%}")

        # Check for NaN
        nan_count = pred_df["pred_class"].isna().sum()
        if nan_count > 0:
            print(f"  WARNING: {nan_count} NaN predictions")

        # Check embeddings if applicable
        if with_embeddings:
            print("\n" + "-" * 40)
            print("STEP 3: Embedding Verification")
            print("-" * 40)

            assert os.path.exists(cfg["market_path"]), "Market embeddings not created"
            emb_df = pd.read_csv(cfg["market_path"])
            print(f"  Market embeddings: {len(emb_df)} rows, {len(emb_df.columns)} columns")

        # Summary
        print("\n" + "=" * 60)
        print("SMOKE TEST PASSED")
        print("=" * 60)
        print(f"\nAll checks passed. Pipeline is working correctly.")

        return True

    except Exception as e:
        print("\n" + "=" * 60)
        print("SMOKE TEST FAILED")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temp directory")


def main():
    parser = argparse.ArgumentParser(description="Stockformer smoke test")
    parser.add_argument(
        "--data-path",
        type=str,
        default="../../data/all_data_*.csv",
        help="Path to CSV data files",
    )
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Also test embedding auto-creation",
    )

    args = parser.parse_args()

    success = run_smoke_test(args.data_path, args.with_embeddings)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
