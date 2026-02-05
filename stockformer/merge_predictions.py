"""
Merge all 9 prediction CSVs into a single merged_predictions.csv.

Reads from: output/predictions_*.csv (9 files)
Writes to:  merged_predictions.csv (main directory)

Usage:
    python -m stockformer.merge_predictions
    python -m stockformer.merge_predictions --input-dir output --output merged_predictions.csv
"""

import argparse
import os
from glob import glob

import pandas as pd


def merge_predictions(input_dir: str = "output", output_path: str = "merged_predictions.csv"):
    """
    Merge all prediction CSVs into one file.

    Args:
        input_dir: Directory containing prediction CSVs
        output_path: Path for merged output file
    """
    # Find all prediction files
    pattern = os.path.join(input_dir, "predictions_*.csv")
    pred_files = sorted(glob(pattern))

    if not pred_files:
        print(f"No prediction files found matching: {pattern}")
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
        # Parse filename to get suffix (e.g., "bin_3d" from "predictions_bin_3d.csv")
        basename = os.path.basename(pred_file)
        # Extract suffix: predictions_bin_3d.csv -> bin_3d
        suffix = basename.replace("predictions_", "").replace(".csv", "")

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

    # Save merged file
    merged.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge prediction CSVs into one file")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="output",
        help="Directory containing prediction CSVs (default: output)",
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