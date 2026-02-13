"""
Build sector → symbol mapping from existing all_data CSV files.

Outputs:
  - sectors.json: list of sectors with ETF tickers
  - sector_{ETF}.json: list of stocks per sector

Usage:
    python -m webapp.backend.build_sector_map
    python -m webapp.backend.build_sector_map --data-dir /path/to/data
"""

import argparse
import glob
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)

# GICS sector → ETF mapping (supports both title-case and uppercase variants)
SECTOR_ETF_MAP = {
    "Financials": "XLF",
    "Financial Services": "XLF",
    "FINANCIAL SERVICES": "XLF",
    "FINANCE": "XLF",
    "Technology": "XLK",
    "Information Technology": "XLK",
    "TECHNOLOGY": "XLK",
    "Health Care": "XLV",
    "Healthcare": "XLV",
    "HEALTHCARE": "XLV",
    "LIFE SCIENCES": "XLV",
    "Energy": "XLE",
    "ENERGY": "XLE",
    "ENERGY & TRANSPORTATION": "XLE",
    "Consumer Discretionary": "XLY",
    "Consumer Cyclical": "XLY",
    "CONSUMER CYCLICAL": "XLY",
    "Industrials": "XLI",
    "INDUSTRIALS": "XLI",
    "Materials": "XLB",
    "Basic Materials": "XLB",
    "BASIC MATERIALS": "XLB",
    "Communication Services": "XLC",
    "COMMUNICATION SERVICES": "XLC",
    "Utilities": "XLU",
    "UTILITIES": "XLU",
    "Real Estate": "XLRE",
    "REAL ESTATE": "XLRE",
    "Consumer Staples": "XLP",
    "Consumer Defensive": "XLP",
    "CONSUMER DEFENSIVE": "XLP",
}

# Display order
SECTOR_ORDER = ["XLF", "XLK", "XLV", "XLE", "XLY", "XLI", "XLB", "XLC", "XLU", "XLRE", "XLP"]

ETF_TO_NAME = {v: k for k, v in SECTOR_ETF_MAP.items() if k != "Information Technology"}


def build_sector_map(csv_pattern: str, output_dir: str):
    """Parse sector column from all_data CSVs, group symbols by sector."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(glob.glob(csv_pattern))
    if not csv_files:
        logger.error(f"No CSV files found matching: {csv_pattern}")
        return

    # Read just symbol + sector columns from the most recent data
    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, usecols=["symbol", "sector", "adjusted_close", "date"])
            frames.append(df)
        except (ValueError, KeyError) as e:
            logger.warning(f"Skipping {f}: {e}")

    if not frames:
        logger.error("No valid CSV files with symbol/sector columns")
        return

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])

    # Get latest row per symbol
    latest = df.sort_values("date").groupby("symbol").last().reset_index()

    # Build sector groups
    sector_groups = {}
    for _, row in latest.iterrows():
        sector = row.get("sector", "Unknown")
        etf = SECTOR_ETF_MAP.get(sector)
        if etf is None:
            continue
        if etf not in sector_groups:
            sector_groups[etf] = []
        sector_groups[etf].append({
            "symbol": row["symbol"],
            "price": round(float(row["adjusted_close"]), 2) if pd.notna(row["adjusted_close"]) else 0,
            "change_pct": 0.0,  # Will be updated by update_quotes.py
        })

    # Sort stocks within each sector by symbol
    for etf in sector_groups:
        sector_groups[etf].sort(key=lambda x: x["symbol"])

    # Write sector_*.json files
    for etf, stocks in sector_groups.items():
        sector_file = output_dir / f"sector_{etf}.json"
        sector_data = {
            "sector_id": etf,
            "name": ETF_TO_NAME.get(etf, etf),
            "change_pct": 0.0,
            "stocks": stocks,
        }
        with open(sector_file, "w") as f:
            json.dump(sector_data, f, indent=2)
        logger.info(f"Wrote {sector_file} ({len(stocks)} stocks)")

    # Write sectors.json (overview)
    sectors_list = []
    for etf in SECTOR_ORDER:
        if etf in sector_groups:
            sectors_list.append({
                "id": etf,
                "name": ETF_TO_NAME.get(etf, etf),
                "change_pct": 0.0,
                "stock_count": len(sector_groups[etf]),
            })

    indices = [
        {"id": "SPY", "name": "S&P 500", "price": 0, "change_pct": 0.0},
        {"id": "QQQ", "name": "Nasdaq 100", "price": 0, "change_pct": 0.0},
    ]

    sectors_data = {"sectors": sectors_list, "indices": indices, "total_stocks": len(latest)}

    sectors_file = output_dir / "sectors.json"
    with open(sectors_file, "w") as f:
        json.dump(sectors_data, f, indent=2)
    logger.info(f"Wrote {sectors_file} ({len(sectors_list)} sectors, {len(latest)} total stocks)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Build sector map from CSV data")
    parser.add_argument("--csv-pattern", default="./all_data_*.csv",
                        help="Glob pattern for all_data CSV files")
    parser.add_argument("--output-dir",
                        default=str(Path(__file__).parent.parent / "data"),
                        help="Output directory for JSON files")
    args = parser.parse_args()
    build_sector_map(args.csv_pattern, args.output_dir)
