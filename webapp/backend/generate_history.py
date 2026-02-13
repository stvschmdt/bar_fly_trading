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


def _row_to_record(row) -> dict:
    """Convert a DataFrame row to a history record dict."""
    rec = {"date": row["date"].strftime("%Y-%m-%d")}
    for csv_col, json_key in HISTORY_COLS.items():
        val = row.get(csv_col)
        if pd.notna(val):
            rec[json_key] = int(val) if json_key == "volume" else round(float(val), 2)
        else:
            rec[json_key] = None
    return rec


def generate_history(csv_pattern: str, symbols_filter: list = None):
    """Read CSVs one at a time, write history per batch (low memory)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(glob.glob(csv_pattern))
    if not csv_files:
        logger.error(f"No CSV files found: {csv_pattern}")
        return

    usecols = ["date", "symbol"] + list(HISTORY_COLS.keys())

    if symbols_filter:
        symbols_filter = {s.upper() for s in symbols_filter}

    written = 0
    for i, f in enumerate(csv_files):
        try:
            df = pd.read_csv(f, usecols=usecols, parse_dates=["date"])
            logger.info(f"  [{i+1}/{len(csv_files)}] {Path(f).name} ({len(df)} rows)")

            if symbols_filter:
                df = df[df["symbol"].isin(symbols_filter)]

            df = df.sort_values(["symbol", "date"])

            # Write each symbol's history from this CSV immediately
            for symbol, group in df.groupby("symbol"):
                records = [_row_to_record(row) for _, row in group.iterrows()]

                out_path = DATA_DIR / f"{symbol}_history.json"

                # If file already exists (symbol split across CSVs), merge
                if out_path.exists():
                    try:
                        with open(out_path) as ef:
                            existing = json.load(ef)
                        existing_dates = {r["date"] for r in existing}
                        for rec in records:
                            if rec["date"] not in existing_dates:
                                existing.append(rec)
                        records = sorted(existing, key=lambda r: r["date"])
                    except Exception:
                        pass

                with open(out_path, "w") as wf:
                    json.dump(records, wf, separators=(",", ":"))
                written += 1

            del df
        except Exception as e:
            logger.warning(f"Skipping {f}: {e}")

    logger.info(f"Wrote {written} history files to {DATA_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="Generate per-symbol history JSONs")
    parser.add_argument("--csv-pattern", default="./all_data_*.csv")
    parser.add_argument("--symbols", help="Comma-separated symbols (default: all)")
    args = parser.parse_args()

    symbols = args.symbols.split(",") if args.symbols else None
    generate_history(args.csv_pattern, symbols)
