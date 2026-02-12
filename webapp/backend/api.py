"""
BFT Web API — FastAPI backend serving sector, stock, and signal data.

Reads pre-generated JSON files from a data directory.
In development, serves sample data if files don't exist yet.
"""

import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("BFT_DATA_DIR", Path(__file__).parent.parent / "data"))
FRONTEND_DIR = Path(__file__).parent.parent / "frontend" / "dist"

app = FastAPI(title="Bar Fly Trading & Investing", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Lock down in production
    allow_methods=["*"],
    allow_headers=["*"],
)


def _read_json(filename: str) -> dict:
    """Read a JSON file from the data directory."""
    path = DATA_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@app.get("/api/sectors")
def get_sectors():
    """Return all sectors with current performance."""
    data = _read_json("sectors.json")
    if data is None:
        raise HTTPException(404, "sectors.json not found — run build_sector_map.py first")
    return data


@app.get("/api/sector/{sector_id}")
def get_sector(sector_id: str):
    """Return stocks within a sector."""
    sector_id = sector_id.upper()
    data = _read_json(f"sector_{sector_id}.json")
    if data is None:
        raise HTTPException(404, f"sector_{sector_id}.json not found")
    return data


@app.get("/api/symbol/{symbol}")
def get_symbol(symbol: str):
    """Return full report for a single symbol."""
    symbol = symbol.upper()
    data = _read_json(f"{symbol}.json")
    if data is None:
        raise HTTPException(404, f"{symbol}.json not found — run generate_reports.py")
    return data


@app.get("/api/signals/today")
def get_signals():
    """Return today's trading signals."""
    data = _read_json("signals_today.json")
    if data is None:
        return {"signals": [], "date": None}
    return data


@app.get("/api/watchlist")
def get_watchlist():
    """Return the custom watchlist with current stock data."""
    wl = _read_json("watchlist.json")
    if wl is None:
        return {"name": "Custom", "symbols": [], "stocks": []}

    # Enrich with latest quote data from symbol JSONs
    stocks = []
    for sym in wl.get("symbols", []):
        sym_data = _read_json(f"{sym}.json")
        if sym_data and sym_data.get("quote"):
            stocks.append({
                "symbol": sym,
                "price": sym_data["quote"].get("price", 0),
                "change_pct": sym_data["quote"].get("change_pct", 0),
                "volume": sym_data["quote"].get("volume", 0),
            })
        else:
            stocks.append({"symbol": sym, "price": 0, "change_pct": 0, "volume": 0})

    changes = [s["change_pct"] for s in stocks if s["change_pct"] != 0]
    return {
        "sector_id": "CUSTOM",
        "name": wl.get("name", "Custom"),
        "change_pct": round(sum(changes) / len(changes), 2) if changes else 0,
        "stocks": stocks,
    }


@app.post("/api/watchlist")
def set_watchlist(payload: dict = Body(...)):
    """Upload a custom watchlist. Accepts {symbols: [...]} or {symbols: "AAPL,NVDA,..."}."""
    symbols = payload.get("symbols", [])

    # Accept comma-separated string or list
    if isinstance(symbols, str):
        symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    else:
        symbols = [s.strip().upper() for s in symbols if s.strip()]

    if not symbols:
        raise HTTPException(400, "No symbols provided")

    name = payload.get("name", "Custom")

    wl_data = {"name": name, "symbols": symbols}
    wl_path = DATA_DIR / "watchlist.json"
    with open(wl_path, "w") as f:
        json.dump(wl_data, f, indent=2)

    logger.info(f"Watchlist updated: {len(symbols)} symbols")
    return {"status": "ok", "count": len(symbols), "symbols": symbols}


@app.get("/api/health")
def health():
    """Health check."""
    return {"status": "ok", "data_dir": str(DATA_DIR)}


# Serve the React frontend (must be after API routes)
if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        """Serve React SPA — all non-API routes return index.html."""
        file_path = FRONTEND_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")
