"""
Merge all 9 prediction CSVs into a single merged_predictions.csv.

Reads from: stockformer/output/predictions/pred_*.csv (9 files)
Writes to:  merged_predictions.csv (main directory)

Usage:
    python -m stockformer.merge_predictions
    python -m stockformer.merge_predictions --input-dir stockformer/output/predictions --output merged_predictions.csv
"""

import argparse
import os
from glob import glob

import numpy as np
import pandas as pd


def merge_predictions(input_dir: str = "output", output_path: str = "merged_predictions.csv"):
    """
    Merge all prediction CSVs into one file.

    Args:
        input_dir: Directory containing prediction CSVs
        output_path: Path for merged output file
    """
    # Find all prediction files (supports both pred_*.csv and predictions_*.csv)
    pattern = os.path.join(input_dir, "pred_*.csv")
    pred_files = sorted(glob(pattern))

    if not pred_files:
        # Fallback: try predictions_*.csv naming convention
        pattern = os.path.join(input_dir, "predictions_*.csv")
        pred_files = sorted(glob(pattern))

    if not pred_files:
        print(f"No prediction files found in: {input_dir}")
        print(f"  Tried: pred_*.csv and predictions_*.csv")
        return

    print(f"Found {len(pred_files)} prediction files:")
    for f in pred_files:
        print(f"  - {f}")

    # Define the key columns for merging
    key_cols = ["date", "ticker"]

    # Define base columns to keep from the first file (includes all features)
    base_cols = None
    merged = None

    for pred_file in pred_files:
        # Parse filename to get suffix (e.g., "bin_3d" from "pred_bin_3d.csv")
        basename = os.path.basename(pred_file)
        # Extract suffix: pred_bin_3d.csv -> bin_3d, predictions_bin_3d.csv -> bin_3d
        suffix = basename.replace("predictions_", "").replace("pred_", "").replace(".csv", "")

        print(f"\nProcessing {basename} (suffix: {suffix})...")
        df = pd.read_csv(pred_file)

        # Identify prediction columns (they start with pred_ or prob_ or true_)
        pred_cols = [c for c in df.columns if c.startswith(("pred_", "prob_", "true_"))]

        if merged is None:
            # First file: keep all columns, rename prediction columns with suffix
            base_cols = [c for c in df.columns if c not in pred_cols]
            merged = df[base_cols].copy()

            # Add prediction columns with suffix
            for col in pred_cols:
                merged[f"{col}_{suffix}"] = df[col]

            print(f"  Base columns: {len(base_cols)}")
            print(f"  Prediction columns: {len(pred_cols)} -> renamed with _{suffix}")
        else:
            # Subsequent files: merge only prediction columns
            pred_df = df[key_cols + pred_cols].copy()

            # Rename prediction columns with suffix
            rename_map = {col: f"{col}_{suffix}" for col in pred_cols}
            pred_df = pred_df.rename(columns=rename_map)

            # Merge on key columns
            merged = merged.merge(pred_df, on=key_cols, how="inner")
            print(f"  Added {len(pred_cols)} columns with _{suffix}")

    print(f"\n{'='*60}")
    print(f"Merged dataset:")
    print(f"  Rows: {len(merged):,}")
    print(f"  Columns: {len(merged.columns)}")

    # Count prediction columns by type
    pred_col_counts = {}
    for col in merged.columns:
        for prefix in ["pred_class_", "pred_expected_return_", "pred_return_", "prob_", "true_"]:
            if col.startswith(prefix):
                pred_col_counts[prefix] = pred_col_counts.get(prefix, 0) + 1
                break

    print(f"  Prediction column counts:")
    for prefix, count in sorted(pred_col_counts.items()):
        print(f"    {prefix}*: {count}")

    # Add ensemble consensus scoring
    merged = add_ensemble_scores(merged)

    # Save merged file
    merged.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return merged


def add_ensemble_scores(df):
    """
    Add consensus scoring across 9 models (3 horizons x 3 label modes).

    For each (date, ticker) computes:
    - expected_return_{3,10,30}d: expected return from each horizon's models
    - consensus_direction: BUY/SELL/NEUTRAL based on majority of horizons
    - consensus_confidence: 0-1 average confidence across models
    - horizon_agreement: 1-3 count of horizons agreeing on direction
    """
    horizons = ["3d", "10d", "30d"]

    for h in horizons:
        signals = []

        # Regression: direct predicted return
        reg_col = f"pred_return_reg_{h}"
        if reg_col in df.columns:
            signals.append(df[reg_col])

        # Binary: expected return from P(up) - P(down)
        p1_col = f"prob_1_bin_{h}"
        if p1_col in df.columns:
            # Map P(up) to directional signal: P(up)*0.01 - P(down)*0.01
            signals.append((df[p1_col] * 2 - 1) * 0.01)

        # Buckets: expected return from class probs × midpoints (if available)
        er_col = f"pred_expected_return_buck_{h}"
        if er_col in df.columns:
            signals.append(df[er_col])

        # Average available signals for this horizon
        if signals:
            stacked = np.column_stack([s.values for s in signals])
            df[f"expected_return_{h}"] = np.nanmean(stacked, axis=1)
        else:
            df[f"expected_return_{h}"] = np.nan

    # Consensus across horizons
    er_cols = [f"expected_return_{h}" for h in horizons if f"expected_return_{h}" in df.columns]
    if er_cols:
        er_matrix = df[er_cols].values

        # Direction per horizon: +1 = up, -1 = down, 0 = near zero
        directions = np.sign(er_matrix)

        # Horizon agreement: how many horizons agree on direction
        pos_count = (directions > 0).sum(axis=1)
        neg_count = (directions < 0).sum(axis=1)
        df["horizon_agreement"] = np.maximum(pos_count, neg_count).astype(int)

        # Consensus direction
        net_direction = directions.sum(axis=1)
        df["consensus_direction"] = np.where(
            net_direction > 0, "BUY",
            np.where(net_direction < 0, "SELL", "NEUTRAL")
        )

        # Confidence: average |expected_return| across horizons, scaled
        avg_magnitude = np.nanmean(np.abs(er_matrix), axis=1)
        # Normalize to 0-1 using sigmoid-like scaling (0.05 return → ~0.7 confidence)
        df["consensus_confidence"] = 1.0 - 1.0 / (1.0 + avg_magnitude * 50)

        print(f"\n  Ensemble scoring:")
        direction_counts = df["consensus_direction"].value_counts()
        for d, c in direction_counts.items():
            print(f"    {d}: {c:,} ({c/len(df):.1%})")
        print(f"    Avg confidence: {df['consensus_confidence'].mean():.3f}")
        print(f"    Avg horizon agreement: {df['horizon_agreement'].mean():.1f}/3")

    return df


def main():
    parser = argparse.ArgumentParser(description="Merge prediction CSVs into one file")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="stockformer/output/predictions",
        help="Directory containing prediction CSVs (default: stockformer/output/predictions)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="merged_predictions.csv",
        help="Output path for merged file (default: merged_predictions.csv)",
    )

    args = parser.parse_args()
    merge_predictions(args.input_dir, args.output)


if __name__ == "__main__":
    main()