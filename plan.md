# Stock Chart Feature — Implementation Plan

## Overview
Add interactive price charts with time-range toggles (5D, 1M, 3M, 6M, 1Y, 3Y) to the StockDetail page. Line chart with area fill, plus toggleable SMA and Bollinger Band overlays.

## Problem
Currently, `{SYMBOL}.json` files only store the latest single data point — no historical price data is exposed via the API. The CSVs (`all_data_*.csv`) have 10 years of daily OHLC data with pre-calculated indicators.

## Plan

### 1. Generate historical JSON files — `webapp/backend/generate_history.py`
New script that reads all 28 `all_data_*.csv` files, groups by symbol, and writes one `{SYMBOL}_history.json` per symbol to `webapp/data/`.

Each file contains an array of daily records with only the columns needed for charting:
```json
[
  {"date": "2023-01-03", "close": 125.07, "volume": 112117500, "sma_20": 128.3, "sma_50": 140.1, "sma_200": 153.4, "bb_upper": 135.2, "bb_lower": 121.4},
  ...
]
```

This keeps files small (~100KB per symbol for 3 years of data) and avoids parsing 600MB of CSVs on every API request. Run once alongside `populate_all`.

### 2. New API endpoint — `webapp/backend/api.py`
Add `GET /api/symbol/{symbol}/history` that reads `{SYMBOL}_history.json` and returns it. No query params needed — the frontend filters by date range client-side (the full dataset is small enough).

### 3. Install recharts — `webapp/frontend/`
`npm install recharts` — lightweight, React-native charting library. No heavy dependencies like d3.

### 4. Create `StockChart.jsx` component
- Time-range toggle buttons: **5D, 1M, 3M, 6M, 1Y, 3Y**
- Recharts `AreaChart` for price with gradient fill (green if up overall, red if down)
- Volume bars below the price chart (`BarChart` composited or separate)
- Toggleable overlays: **SMA 20**, **SMA 50**, **SMA 200**, **Bollinger Bands** (upper/lower as dashed lines)
- Responsive, dark-mode aware

### 5. Add `getSymbolHistory()` to `client.js`
New API client function following existing pattern.

### 6. Integrate chart into `StockDetail.jsx`
Add the chart above the existing flip card (technical/news). The page layout becomes:
- Header (symbol, price, change)
- **Chart with time-range and overlay toggles** ← new
- Flip card (technical indicators / news)

### 7. Build & test

## Files Changed
| File | Action |
|------|--------|
| `webapp/backend/generate_history.py` | **New** — script to pre-generate history JSONs |
| `webapp/backend/api.py` | Add `/api/symbol/{symbol}/history` endpoint |
| `webapp/frontend/package.json` | Add `recharts` dependency |
| `webapp/frontend/src/api/client.js` | Add `getSymbolHistory()` |
| `webapp/frontend/src/components/StockChart.jsx` | **New** — chart component |
| `webapp/frontend/src/components/StockDetail.jsx` | Import and render `StockChart` |
