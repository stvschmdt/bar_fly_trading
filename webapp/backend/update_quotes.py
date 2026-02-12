"""
Fetch real-time quotes via Alpha Vantage REALTIME_BULK_QUOTES and update JSON files.

Uses the existing rt_utils.get_realtime_quotes_bulk() which handles chunking
(100 symbols per API call, premium endpoint).

Designed to run every 30 minutes during market hours via cron.

Usage:
    python -m webapp.backend.update_quotes
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("BFT_DATA_DIR", Path(__file__).parent.parent / "data"))


def update_quotes():
    """Bulk fetch quotes via Alpha Vantage and update sector + symbol JSON files."""
    from api_data.rt_utils import get_realtime_quotes_bulk

    sectors_file = DATA_DIR / "sectors.json"
    if not sectors_file.exists():
        logger.error("sectors.json not found — run build_sector_map.py first")
        return

    with open(sectors_file) as f:
        sectors_data = json.load(f)

    # Collect all symbols from sector files
    all_symbols = set()
    sector_symbols = {}  # etf -> [symbols]

    for sector in sectors_data["sectors"]:
        etf = sector["id"]
        sector_file = DATA_DIR / f"sector_{etf}.json"
        if sector_file.exists():
            with open(sector_file) as f:
                sd = json.load(f)
            syms = [s["symbol"] for s in sd["stocks"]]
            sector_symbols[etf] = syms
            all_symbols.update(syms)

    # Add index ETFs
    index_symbols = ["SPY", "QQQ"] + list(sector_symbols.keys())
    all_symbols.update(index_symbols)

    # Also add watchlist symbols
    wl_file = DATA_DIR / "watchlist.json"
    if wl_file.exists():
        with open(wl_file) as f:
            wl = json.load(f)
        all_symbols.update(wl.get("symbols", []))

    if not all_symbols:
        logger.warning("No symbols to update")
        return

    logger.info(f"Fetching AV bulk quotes for {len(all_symbols)} symbols...")

    # Alpha Vantage REALTIME_BULK_QUOTES — auto-chunks in batches of 100
    df = get_realtime_quotes_bulk(sorted(all_symbols))

    if df.empty:
        logger.error("No quotes returned from Alpha Vantage")
        return

    # Build quotes dict: need previous close for change_pct
    # AV bulk returns current price; we compute change from yesterday's close in symbol JSONs
    quotes = {}
    for _, row in df.iterrows():
        sym = row["symbol"]
        price = float(row["price"])
        if price <= 0:
            continue

        # Get previous close from existing symbol JSON for change calculation
        prev_close = price  # fallback: 0% change
        sym_file = DATA_DIR / f"{sym}.json"
        if sym_file.exists():
            try:
                with open(sym_file) as f:
                    existing = json.load(f)
                prev = existing.get("technical", {}).get("price", 0)
                if prev and prev > 0:
                    prev_close = prev
            except Exception:
                pass

        change_pct = round(((price - prev_close) / prev_close) * 100, 2) if prev_close else 0

        quotes[sym] = {
            "price": round(price, 2),
            "change_pct": change_pct,
            "volume": int(row.get("volume", 0)),
        }

    logger.info(f"Got quotes for {len(quotes)}/{len(all_symbols)} symbols")

    # Update sector files
    for etf, syms in sector_symbols.items():
        sector_file = DATA_DIR / f"sector_{etf}.json"
        with open(sector_file) as f:
            sd = json.load(f)

        changes = []
        for stock in sd["stocks"]:
            sym = stock["symbol"]
            if sym in quotes:
                stock["price"] = quotes[sym]["price"]
                stock["change_pct"] = quotes[sym]["change_pct"]
                stock["volume"] = quotes[sym].get("volume", 0)
                changes.append(quotes[sym]["change_pct"])

        sd["change_pct"] = round(sum(changes) / len(changes), 2) if changes else 0.0

        with open(sector_file, "w") as f:
            json.dump(sd, f, indent=2)

    # Update sectors.json overview
    for sector in sectors_data["sectors"]:
        etf = sector["id"]
        sf = DATA_DIR / f"sector_{etf}.json"
        if sf.exists():
            with open(sf) as f:
                sd = json.load(f)
            sector["change_pct"] = sd["change_pct"]

    for idx in sectors_data["indices"]:
        if idx["id"] in quotes:
            idx["price"] = quotes[idx["id"]]["price"]
            idx["change_pct"] = quotes[idx["id"]]["change_pct"]

    sectors_data["last_updated"] = datetime.now().isoformat(timespec="seconds")

    with open(sectors_file, "w") as f:
        json.dump(sectors_data, f, indent=2)

    # Patch quote section in each symbol JSON
    patched = 0
    now = datetime.now().isoformat(timespec="seconds")
    for sym, q in quotes.items():
        sym_file = DATA_DIR / f"{sym}.json"
        if not sym_file.exists():
            continue
        try:
            with open(sym_file) as f:
                report = json.load(f)
            report["quote"] = {
                "price": q["price"],
                "change_pct": q["change_pct"],
                "volume": q.get("volume", 0),
                "last_updated": now,
            }
            if report.get("technical"):
                report["technical"]["price"] = q["price"]
            with open(sym_file, "w") as f:
                json.dump(report, f, indent=2)
            patched += 1
        except Exception:
            continue

    logger.info(f"Updated {len(sector_symbols)} sectors, {len(quotes)} quotes, {patched} symbol files")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    update_quotes()
