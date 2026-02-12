"""
Generate per-symbol historical price JSON files for the webapp chart feature.

Reads all_data_*.csv files, extracts OHLC + indicators per symbol,
writes {SYMBOL}_history.json to the data directory.

Usage:
    python -m webapp.backend.generate_history
    python -m webapp.backend.generate_history --csv-pattern "/path/all_data_*.csv"
    python -m webapp.backend.generate_history --symbols AAPL,NVDA
"""

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("BFT_DATA_DIR", Path(__file__).parent.parent / "data"))

# Columns to include in history JSON (beyond date and close)
HISTORY_COLS = {
    "adjusted_close": "close",
    "volume": "volume",
    "sma_20": "sma_20",
    "sma_50": "sma_50",
    "sma_200": "sma_200",
    "bbands_upper_20": "bb_upper",
    "bbands_lower_20": "bb_lower",
}


def generate_history(csv_pattern: str, symbols_filter: list = None):
    """Read CSVs and write per-symbol history JSON files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(glob.glob(csv_pattern))
    if not csv_files:
        logger.error(f"No CSV files found: {csv_pattern}")
        return

    # Only read columns we need
    usecols = ["date", "symbol"] + list(HISTORY_COLS.keys())

    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, usecols=usecols, parse_dates=["date"])
            frames.append(df)
        except Exception as e:
            logger.warning(f"Skipping {f}: {e}")

    if not frames:
        logger.error("No valid CSV files loaded")
        return

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["symbol", "date"])

    if symbols_filter:
        symbols_filter = [s.upper() for s in symbols_filter]
        df = df[df["symbol"].isin(symbols_filter)]

    written = 0
    for symbol, group in df.groupby("symbol"):
        records = []
        for _, row in group.iterrows():
            rec = {"date": row["date"].strftime("%Y-%m-%d")}
            for csv_col, json_key in HISTORY_COLS.items():
                val = row.get(csv_col)
                if pd.notna(val):
                    if json_key == "volume":
                        rec[json_key] = int(val)
                    else:
                        rec[json_key] = round(float(val), 2)
                else:
                    rec[json_key] = None
            records.append(rec)

        out_path = DATA_DIR / f"{symbol}_history.json"
        with open(out_path, "w") as f:
            json.dump(records, f, separators=(",", ":"))
        written += 1

    logger.info(f"Wrote {written} history files to {DATA_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="Generate per-symbol history JSONs")
    parser.add_argument("--csv-pattern", default="/home/stvschmdt/data/all_data_*.csv")
    parser.add_argument("--symbols", help="Comma-separated symbols (default: all)")
    args = parser.parse_args()

    symbols = args.symbols.split(",") if args.symbols else None
    generate_history(args.csv_pattern, symbols)
