"""
Nightly refresh of technical indicators from all_data CSVs.

Re-reads the latest row per symbol from CSVs (which get updated by the RT pipeline)
and updates only the technical section of each {SYMBOL}.json. Optionally runs LLM
summaries via Ollama.

Usage:
    python -m webapp.backend.refresh_technicals                     # technicals only
    python -m webapp.backend.refresh_technicals --with-summaries    # + LLM summaries
"""

import argparse
import glob
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("BFT_DATA_DIR", Path(__file__).parent.parent / "data"))

# Same column mapping as populate_all.py
TECHNICAL_COLS = [
    "rsi_14", "macd", "macd_9_ema", "adx_14", "cci_14", "atr_14",
    "sma_20", "sma_50", "sma_200", "ema_20", "ema_50",
    "bbands_upper_20", "bbands_lower_20", "bbands_middle_20",
    "pe_ratio", "price_to_book_ratio", "book_value", "eps",
    "beta", "dividend_yield", "forward_pe",
    "52_week_high", "52_week_low",
]

COL_RENAME = {
    "macd_9_ema": "macd_signal",
    "rsi_14": "rsi",
    "adx_14": "adx",
    "cci_14": "cci",
    "atr_14": "atr",
    "bbands_upper_20": "bbands_upper",
    "bbands_lower_20": "bbands_lower",
    "bbands_middle_20": "bbands_middle",
}


def refresh_technicals(csv_pattern: str, with_summaries: bool = False):
    """Update the technical section of all symbol JSONs from CSV data."""
    csv_files = sorted(glob.glob(csv_pattern))
    if not csv_files:
        logger.error(f"No CSV files found: {csv_pattern}")
        return

    # Load latest row per symbol
    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, parse_dates=["date"])
            frames.append(df)
        except Exception as e:
            logger.warning(f"Skipping {f}: {e}")

    if not frames:
        return

    df = pd.concat(frames, ignore_index=True)
    latest = df.sort_values("date").groupby("symbol").last().reset_index()
    logger.info(f"Loaded {len(latest)} symbols for technical refresh")

    updated = 0
    for _, row in latest.iterrows():
        symbol = row["symbol"]
        sym_file = DATA_DIR / f"{symbol}.json"
        if not sym_file.exists():
            continue

        try:
            with open(sym_file) as f:
                report = json.load(f)
        except Exception:
            continue

        # Build fresh technical section
        price = round(float(row.get("adjusted_close", 0) or 0), 2)
        tech = {"price": price}

        for col in TECHNICAL_COLS:
            if col in row.index and pd.notna(row[col]):
                key = COL_RENAME.get(col, col)
                val = row[col]
                if isinstance(val, (float, np.floating)):
                    val = round(float(val), 4)
                elif isinstance(val, (int, np.integer)):
                    val = int(val)
                tech[key] = val

        # Preserve existing summary if not regenerating
        if not with_summaries and report.get("technical", {}).get("summary"):
            tech["summary"] = report["technical"]["summary"]

        # Preserve analyst ratings
        if report.get("technical", {}).get("analyst_ratings"):
            tech["analyst_ratings"] = report["technical"]["analyst_ratings"]

        # LLM summary generation (optional, nightly)
        if with_summaries:
            try:
                from api_data.rt_utils import summarize_technical_with_llm
                summary = summarize_technical_with_llm(symbol, tech)
                if summary:
                    tech["summary"] = summary
            except Exception as e:
                logger.debug(f"LLM summary skipped for {symbol}: {e}")

        report["technical"] = tech
        report["report_date"] = str(row.get("date", ""))[:10]

        with open(sym_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        updated += 1

    logger.info(f"Updated technicals for {updated} symbols")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-pattern", default="./all_data_*.csv")
    parser.add_argument("--with-summaries", action="store_true",
                        help="Also regenerate LLM technical summaries (slow)")
    args = parser.parse_args()
    refresh_technicals(args.csv_pattern, args.with_summaries)
