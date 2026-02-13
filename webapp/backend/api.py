"""
BFT Web API — FastAPI backend serving sector, stock, and signal data.

Reads pre-generated JSON files from a data directory.
Auth via JWT tokens (invite-code-gated registration).
"""

import csv
import json
import logging
import os
import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .auth import router as auth_router, get_current_user, SECRET_KEY
from .build_sector_map import SECTOR_ORDER, ETF_TO_NAME
from .database import (
    init_db,
    get_user_by_email,
    get_watchlist as db_get_watchlist,
    set_watchlist as db_set_watchlist,
)

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

# Include auth routes
app.include_router(auth_router)


# ── Startup ───────────────────────────────────────────────────────

@app.on_event("startup")
def on_startup():
    init_db()
    if SECRET_KEY == "bft-dev-secret-change-in-prod":
        logger.warning("Using default JWT secret — set BFT_JWT_SECRET in production")


# ── Auth Middleware ────────────────────────────────────────────────

AUTH_WHITELIST = {"/api/auth/login", "/api/auth/register", "/api/health"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api/") and path not in AUTH_WHITELIST:
        try:
            get_current_user(request)
        except HTTPException as e:
            return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
    response = await call_next(request)
    return response


# ── Helpers ───────────────────────────────────────────────────────

def _read_json(filename: str) -> dict:
    """Read a JSON file from the data directory."""
    path = DATA_DIR / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _get_user_id(request: Request) -> int:
    """Get user ID from JWT token."""
    email = get_current_user(request)
    user = get_user_by_email(email)
    if not user:
        raise HTTPException(401, "User not found")
    return user["id"]


# ── Data Endpoints ────────────────────────────────────────────────

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


@app.get("/api/bigboard")
def get_bigboard():
    """Return all stocks across all sectors for the Big Board view."""
    all_stocks = []
    sectors = []
    for etf in SECTOR_ORDER:
        data = _read_json(f"sector_{etf}.json")
        if data is None:
            continue
        sectors.append({"id": etf, "name": ETF_TO_NAME.get(etf, etf)})
        for stock in data.get("stocks", []):
            stock["sector_id"] = etf
            all_stocks.append(stock)
    all_stocks.sort(key=lambda s: s["symbol"])
    return {"stocks": all_stocks, "sectors": sectors}


@app.get("/api/symbol/{symbol}")
def get_symbol(symbol: str):
    """Return full report for a single symbol."""
    symbol = symbol.upper()
    data = _read_json(f"{symbol}.json")
    if data is None:
        raise HTTPException(404, f"{symbol}.json not found — run generate_reports.py")
    return data


@app.get("/api/symbol/{symbol}/history")
def get_symbol_history(symbol: str):
    """Return historical price data for charting."""
    symbol = symbol.upper()
    data = _read_json(f"{symbol}_history.json")
    if data is None:
        raise HTTPException(404, f"{symbol}_history.json not found — run generate_history.py")
    return data


@app.get("/api/signals/today")
def get_signals():
    """Return today's trading signals."""
    data = _read_json("signals_today.json")
    if data is None:
        return {"signals": [], "date": None}
    return data


# ── Overnight Screener ───────────────────────────────────────────

OVERNIGHT_BASE = Path(os.environ.get("BFT_PROJECT_DIR", Path(__file__).parent.parent.parent))

def _find_overnight_dir():
    """Find the most recent overnight_v2_* directory."""
    dirs = sorted(OVERNIGHT_BASE.glob("overnight_v2_*"), reverse=True)
    return dirs[0] if dirs else None


@app.get("/api/overnight")
def get_overnight():
    """Return overnight screener metadata: signals from CSVs, available images."""
    overnight_dir = _find_overnight_dir()

    # Collect signal data from the most recent all_data CSVs
    signals = {}
    stocks = []
    signal_cols = [
        "macd_signal", "macd_zero_signal", "adx_signal", "atr_signal",
        "pe_ratio_signal", "bollinger_bands_signal", "rsi_signal",
        "sma_cross_signal", "cci_signal", "pcr_signal", "bull_bear_delta",
    ]
    csv_files = sorted(OVERNIGHT_BASE.glob("all_data_*.csv"))
    for csv_path in csv_files:
        try:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                continue
            # Get the latest date's rows
            dates = sorted(set(r.get("date", "") for r in rows if r.get("date")))
            if not dates:
                continue
            latest = dates[-1]
            for row in rows:
                if row.get("date") != latest:
                    continue
                sym = row.get("symbol", "").strip()
                if not sym or sym in signals:
                    continue
                sig = {}
                for col in signal_cols:
                    val = row.get(col, "")
                    try:
                        sig[col] = int(float(val)) if val else 0
                    except (ValueError, TypeError):
                        sig[col] = 0
                signals[sym] = sig
                stocks.append(sym)
        except Exception:
            continue

    # Scan overnight directory for available images
    stock_charts = {}
    sectors = []
    has_market_returns = False
    screener_date = None

    if overnight_dir and overnight_dir.is_dir():
        screener_date = overnight_dir.name.replace("overnight_v2_", "")
        for img in overnight_dir.iterdir():
            if not img.suffix == ".jpg":
                continue
            name = img.stem
            if name == "market_returns":
                has_market_returns = True
            elif name.startswith("_divider_"):
                continue
            elif "_sector_" in name:
                # e.g. XLK_sector_Technology_analysis
                m = re.match(r"^([A-Z]+)_sector_(.+)_analysis$", name)
                if m:
                    sectors.append({
                        "id": m.group(1),
                        "name": m.group(2),
                        "filename": img.name,
                    })
            else:
                # Per-stock chart: AAPL_daily_price, AAPL_technical_rsi, etc.
                for chart_type in ["daily_price", "daily_volume", "technical_rsi",
                                   "technical_macd", "technical_cci",
                                   "technical_off_from_highs", "technical_ttm_pe_ratio"]:
                    if name.endswith(f"_{chart_type}"):
                        sym = name[: -(len(chart_type) + 1)]
                        stock_charts.setdefault(sym, []).append(chart_type)
                        break

    stocks.sort()
    sectors.sort(key=lambda s: s["id"])

    return {
        "date": screener_date,
        "stocks": stocks,
        "signals": signals,
        "stock_charts": stock_charts,
        "sectors": sectors,
        "has_market_returns": has_market_returns,
    }


@app.get("/api/overnight/image/{filename:path}")
def get_overnight_image(filename: str):
    """Serve an image from the overnight screener directory."""
    overnight_dir = _find_overnight_dir()
    if not overnight_dir:
        raise HTTPException(404, "No overnight data available")
    # Sanitize filename
    safe_name = Path(filename).name
    file_path = overnight_dir / safe_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, f"Image not found: {safe_name}")
    return FileResponse(file_path, media_type="image/jpeg")


# ── Per-User Watchlist ────────────────────────────────────────────

@app.get("/api/watchlist")
def get_watchlist(request: Request):
    """Return the current user's watchlist with enriched stock data."""
    user_id = _get_user_id(request)
    wl = db_get_watchlist(user_id)

    if wl is None:
        return {"name": "Custom", "symbols": [], "stocks": [], "sector_id": "CUSTOM", "change_pct": 0}

    stocks = []
    for sym in wl["symbols"]:
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
def set_watchlist(request: Request, payload: dict = Body(...)):
    """Save the current user's watchlist."""
    user_id = _get_user_id(request)

    symbols = payload.get("symbols", [])
    if isinstance(symbols, str):
        symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    else:
        symbols = [s.strip().upper() for s in symbols if s.strip()]

    if not symbols:
        raise HTTPException(400, "No symbols provided")

    name = payload.get("name", "Custom")
    db_set_watchlist(user_id, symbols, name)

    logger.info(f"Watchlist updated for user {user_id}: {len(symbols)} symbols")
    return {"status": "ok", "count": len(symbols), "symbols": symbols}


@app.get("/api/health")
def health():
    """Health check."""
    return {"status": "ok", "data_dir": str(DATA_DIR)}


# ── Serve React SPA (must be after API routes) ────────────────────

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        """Serve React SPA — all non-API routes return index.html."""
        file_path = FRONTEND_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")
