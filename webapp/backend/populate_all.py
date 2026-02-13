from __future__ import annotations

"""
One-time bulk population of all symbol JSONs from existing CSV data + yfinance.

Reads technical indicators from all_data_*.csv files (no API calls needed),
then optionally fetches current prices via yfinance in one batch call.

Usage:
    python -m webapp.backend.populate_all                    # from CSVs only
    python -m webapp.backend.populate_all --with-quotes      # CSVs + live yfinance prices
    python -m webapp.backend.populate_all --symbols AAPL,NVDA  # specific symbols only
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

# Columns to extract from CSVs for the technical section
TECHNICAL_COLS = [
    "rsi_14", "macd", "macd_9_ema", "adx_14", "cci_14", "atr_14",
    "sma_20", "sma_50", "sma_200", "ema_20", "ema_50",
    "bbands_upper_20", "bbands_lower_20", "bbands_middle_20",
    "pe_ratio", "price_to_book_ratio", "book_value", "eps",
    "beta", "dividend_yield", "forward_pe",
    "52_week_high", "52_week_low",
    "analyst_rating_strong_buy", "analyst_rating_buy",
    "analyst_rating_hold", "analyst_rating_sell", "analyst_rating_strong_sell",
]

# Map CSV column names to clean JSON field names
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


def load_latest_per_symbol(csv_pattern: str) -> pd.DataFrame:
    """Load CSVs one at a time, keeping only the latest row per symbol (low memory)."""
    csv_files = sorted(glob.glob(csv_pattern))
    if not csv_files:
        logger.error(f"No CSV files found: {csv_pattern}")
        return pd.DataFrame()

    # Dict of symbol -> latest row (as dict), only keeping the most recent
    best = {}  # symbol -> (date, row_dict)
    prev_data = {}  # symbol -> {close, pct} from previous day

    for i, f in enumerate(csv_files):
        try:
            df = pd.read_csv(f, parse_dates=["date"])
            logger.info(f"  [{i+1}/{len(csv_files)}] Reading {Path(f).name} ({len(df)} rows)")
            for symbol, group in df.groupby("symbol"):
                sorted_group = group.sort_values("date")
                row = sorted_group.iloc[-1]
                row_date = row["date"]
                if symbol not in best or row_date > best[symbol][0]:
                    best[symbol] = (row_date, row.to_dict())
                    # Track previous day's close AND pct for fallback
                    if len(sorted_group) >= 2:
                        prev_row = sorted_group.iloc[-2]
                        pc = prev_row.get("adjusted_close")
                        pp = prev_row.get("adjusted_close_pct")
                        prev_data[symbol] = {
                            "close": float(pc) if pd.notna(pc) else 0,
                            "pct": float(pp) if pd.notna(pp) else 0,
                        }
            del df  # free memory immediately
        except Exception as e:
            logger.warning(f"Skipping {f}: {e}")

    if not best:
        return pd.DataFrame()

    # Fix change_pct: live mode copies yesterday's row so pct=0 and close is identical.
    # Use previous day's pct when today's is 0 (the copied row has no real change).
    for symbol, (_, row_dict) in best.items():
        csv_pct = float(row_dict.get("adjusted_close_pct", 0) or 0)
        if csv_pct == 0 and symbol in prev_data and prev_data[symbol]["pct"] != 0:
            row_dict["adjusted_close_pct"] = prev_data[symbol]["pct"]

    latest = pd.DataFrame([v[1] for v in best.values()])
    logger.info(f"Loaded latest rows for {len(latest)} symbols from {len(csv_files)} CSV files")
    return latest


def build_symbol_json(row: pd.Series) -> dict:
    """Build a full symbol JSON from a CSV row."""
    symbol = row["symbol"]
    price = float(row.get("adjusted_close", 0)) if pd.notna(row.get("adjusted_close")) else 0

    # Technical section
    tech = {"price": round(price, 2)}
    for col in TECHNICAL_COLS:
        if col in row.index and pd.notna(row[col]):
            key = COL_RENAME.get(col, col)
            val = row[col]
            if isinstance(val, (float, np.floating)):
                val = round(float(val), 4)
            elif isinstance(val, (int, np.integer)):
                val = int(val)
            tech[key] = val

    # Metadata
    meta = {}
    for col in ["sector", "industry", "exchange", "market_capitalization", "shares_outstanding"]:
        if col in row.index and pd.notna(row[col]):
            val = row[col]
            if isinstance(val, (float, np.floating)):
                val = round(float(val), 2)
            meta[col] = val

    # Analyst ratings
    ratings = {}
    for col in ["analyst_rating_strong_buy", "analyst_rating_buy", "analyst_rating_hold",
                 "analyst_rating_sell", "analyst_rating_strong_sell"]:
        if col in row.index and pd.notna(row[col]):
            key = col.replace("analyst_rating_", "")
            ratings[key] = int(row[col])
    if ratings:
        tech["analyst_ratings"] = ratings

    report = {
        "symbol": symbol,
        "report_date": str(row.get("date", ""))[:10],
        "quote": {
            "price": round(price, 2),
            "change_pct": round(float(row.get("adjusted_close_pct", 0) or 0) * 100, 2),
            "volume": int(row.get("volume", 0) or 0),
            "last_updated": datetime.now().isoformat(timespec="seconds"),
        },
        "technical": tech,
        "news": None,       # populated weekly by generate_reports.py
        "earnings": None,   # populated quarterly
        "signal": None,     # populated by RT scan loop
        "meta": meta,
    }

    return report


def fetch_live_quotes(symbols: list[str]) -> dict:
    """Batch fetch current prices from Alpha Vantage REALTIME_BULK_QUOTES."""
    from api_data.rt_utils import get_realtime_quotes_bulk

    logger.info(f"Fetching AV bulk quotes for {len(symbols)} symbols...")
    df = get_realtime_quotes_bulk(symbols)

    if df.empty:
        logger.error("No quotes returned from Alpha Vantage")
        return {}

    quotes = {}
    now = datetime.now().isoformat(timespec="seconds")
    for _, row in df.iterrows():
        sym = row["symbol"]
        price = float(row["price"])
        if price <= 0:
            continue
        quotes[sym] = {
            "price": round(price, 2),
            "change_pct": 0.0,  # No previous close from bulk; will be computed by update_quotes
            "volume": int(row.get("volume", 0)),
            "last_updated": now,
        }

    logger.info(f"Got AV quotes for {len(quotes)}/{len(symbols)} symbols")
    return quotes


def populate_all(csv_pattern: str, with_quotes: bool = False, symbols_filter: list[str] = None):
    """Build all symbol JSONs from CSV data, optionally with live quotes."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    latest = load_latest_per_symbol(csv_pattern)
    if latest.empty:
        return

    if symbols_filter:
        latest = latest[latest["symbol"].isin(symbols_filter)]
        logger.info(f"Filtered to {len(latest)} symbols")

    # Fetch live quotes if requested
    quotes = {}
    if with_quotes:
        quotes = fetch_live_quotes(latest["symbol"].tolist())

    # Write per-symbol JSONs
    written = 0
    for _, row in latest.iterrows():
        symbol = row["symbol"]
        report = build_symbol_json(row)

        # Overlay live quote if available
        if symbol in quotes:
            report["quote"] = quotes[symbol]
            report["technical"]["price"] = quotes[symbol]["price"]

        # Preserve existing news/earnings/signal if file already exists
        existing_file = DATA_DIR / f"{symbol}.json"
        if existing_file.exists():
            try:
                with open(existing_file) as f:
                    existing = json.load(f)
                for key in ["news", "earnings", "signal"]:
                    if existing.get(key) and not report.get(key):
                        report[key] = existing[key]
            except Exception:
                pass

        with open(DATA_DIR / f"{symbol}.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        written += 1

    logger.info(f"Wrote {written} symbol JSONs to {DATA_DIR}")

    # Also update sector files with latest prices
    _update_sector_prices(latest, quotes)


def _update_sector_prices(latest: pd.DataFrame, quotes: dict):
    """Update sector JSON files with latest prices from populate."""
    sectors_file = DATA_DIR / "sectors.json"
    if not sectors_file.exists():
        logger.info("No sectors.json — run build_sector_map.py first")
        return

    with open(sectors_file) as f:
        sectors_data = json.load(f)

    for sector in sectors_data["sectors"]:
        etf = sector["id"]
        sf = DATA_DIR / f"sector_{etf}.json"
        if not sf.exists():
            continue

        with open(sf) as f:
            sd = json.load(f)

        now = datetime.now().isoformat(timespec="seconds")
        changes = []
        for stock in sd["stocks"]:
            sym = stock["symbol"]
            if sym in quotes:
                stock["price"] = quotes[sym]["price"]
                stock["change_pct"] = quotes[sym]["change_pct"]
                stock["volume"] = quotes[sym].get("volume", 0)
                stock["last_updated"] = now
            else:
                row = latest[latest["symbol"] == sym]
                if not row.empty:
                    r = row.iloc[0]
                    stock["price"] = round(float(r.get("adjusted_close", 0) or 0), 2)
                    stock["last_updated"] = now
            changes.append(stock.get("change_pct", 0))

        sd["change_pct"] = round(sum(changes) / len(changes), 2) if changes else 0
        sector["change_pct"] = sd["change_pct"]

        with open(sf, "w") as f:
            json.dump(sd, f, indent=2)

    sectors_data["last_updated"] = datetime.now().isoformat(timespec="seconds")
    with open(sectors_file, "w") as f:
        json.dump(sectors_data, f, indent=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser(description="One-time bulk populate all symbol JSONs")
    parser.add_argument("--csv-pattern", default="/home/stvschmdt/data/all_data_*.csv")
    parser.add_argument("--with-quotes", action="store_true", help="Also fetch live prices from yfinance")
    parser.add_argument("--symbols", help="Comma-separated symbols to populate (default: all)")
    args = parser.parse_args()

    symbols = args.symbols.split(",") if args.symbols else None
    populate_all(args.csv_pattern, args.with_quotes, symbols)
