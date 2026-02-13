"""
Generate sample JSON data for local development.

Run this so the frontend has something to display without needing
real API calls or CSV data.

Usage:
    python -m webapp.backend.generate_sample_data
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

SECTORS = {
    "XLF": {"name": "Financials", "stocks": ["JPM", "BAC", "WFC", "GS", "MS", "BLK", "SCHW", "AXP", "CB", "CME", "ICE", "USB", "PNC", "TFC", "COF"]},
    "XLK": {"name": "Technology", "stocks": ["AAPL", "MSFT", "NVDA", "AVGO", "ADBE", "CRM", "AMD", "INTC", "CSCO", "ORCL", "TXN", "QCOM", "AMAT", "NOW", "INTU"]},
    "XLV": {"name": "Health Care", "stocks": ["UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN", "GILD", "ISRG", "CVS", "ELV"]},
    "XLE": {"name": "Energy", "stocks": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PXD", "PSX", "VLO", "OXY"]},
    "XLY": {"name": "Consumer Discretionary", "stocks": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "CMG", "ORLY", "ROST", "DHI", "GM", "F"]},
    "XLI": {"name": "Industrials", "stocks": ["CAT", "UNP", "HON", "UPS", "BA", "GE", "RTX", "DE", "LMT", "MMM", "WM", "ETN", "ITW", "EMR", "FDX"]},
    "XLB": {"name": "Materials", "stocks": ["LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "DOW", "DD", "PPG"]},
    "XLC": {"name": "Communication Services", "stocks": ["META", "GOOG", "GOOGL", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS", "CHTR"]},
    "XLU": {"name": "Utilities", "stocks": ["NEE", "SO", "DUK", "D", "SRE", "AEP", "EXC", "XEL", "WEC", "ES"]},
    "XLRE": {"name": "Real Estate", "stocks": ["PLD", "AMT", "CCI", "EQIX", "PSA", "O", "SPG", "WELL", "DLR", "AVB"]},
    "XLP": {"name": "Consumer Staples", "stocks": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "MDLZ", "GIS", "KMB", "STZ", "SYY", "HSY", "KHC"]},
}

SAMPLE_PRICES = {
    "AAPL": 232, "MSFT": 415, "NVDA": 128, "AMZN": 225, "TSLA": 340, "META": 595,
    "GOOG": 190, "JPM": 247, "BAC": 41, "WFC": 72, "GS": 598, "UNH": 510,
    "JNJ": 155, "PFE": 26, "XOM": 112, "CVX": 155, "HD": 395, "CAT": 370,
    "SPY": 603, "QQQ": 530,
}


def generate():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    sectors_list = []
    total_stocks = 0

    for etf, info in SECTORS.items():
        stocks = []
        changes = []
        for sym in info["stocks"]:
            base_price = SAMPLE_PRICES.get(sym, random.uniform(30, 400))
            chg = round(random.uniform(-3.0, 3.0), 2)
            price = round(base_price * (1 + chg / 100), 2)
            stocks.append({
                "symbol": sym,
                "price": price,
                "change_pct": chg,
                "volume": random.randint(500_000, 50_000_000),
            })
            changes.append(chg)

        sector_avg = round(sum(changes) / len(changes), 2)
        stocks.sort(key=lambda x: x["symbol"])

        sector_data = {
            "sector_id": etf,
            "name": info["name"],
            "change_pct": sector_avg,
            "stocks": stocks,
        }
        with open(DATA_DIR / f"sector_{etf}.json", "w") as f:
            json.dump(sector_data, f, indent=2)

        sectors_list.append({
            "id": etf,
            "name": info["name"],
            "change_pct": sector_avg,
            "stock_count": len(stocks),
        })
        total_stocks += len(stocks)

    # Indices
    indices = [
        {"id": "SPY", "name": "S&P 500", "price": SAMPLE_PRICES["SPY"], "change_pct": round(random.uniform(-1.5, 2.0), 2)},
        {"id": "QQQ", "name": "Nasdaq 100", "price": SAMPLE_PRICES["QQQ"], "change_pct": round(random.uniform(-1.5, 2.0), 2)},
    ]

    sectors_json = {
        "sectors": sectors_list,
        "indices": indices,
        "total_stocks": total_stocks,
        "last_updated": "2026-02-11T15:30:00",
    }
    with open(DATA_DIR / "sectors.json", "w") as f:
        json.dump(sectors_json, f, indent=2)

    # Sample symbol reports
    for etf, info in SECTORS.items():
        for sym in info["stocks"][:3]:  # First 3 per sector
            base_price = SAMPLE_PRICES.get(sym, 150)
            report = {
                "symbol": sym,
                "report_date": "2026-02-11",
                "technical": {
                    "rsi": round(random.uniform(25, 75), 1),
                    "macd": round(random.uniform(-2, 2), 2),
                    "macd_signal": round(random.uniform(-1.5, 1.5), 2),
                    "adx": round(random.uniform(15, 35), 1),
                    "cci": round(random.uniform(-120, 120), 1),
                    "atr": round(random.uniform(1, 8), 2),
                    "sma_20": round(base_price * random.uniform(0.97, 1.03), 2),
                    "sma_50": round(base_price * random.uniform(0.95, 1.05), 2),
                    "bbands_upper": round(base_price * 1.05, 2),
                    "bbands_lower": round(base_price * 0.95, 2),
                    "pe_ratio": round(random.uniform(8, 40), 1),
                    "price": base_price,
                    "summary": f"{sym} is showing mixed signals. RSI is in neutral territory suggesting no extreme momentum conditions. The MACD is slightly positive indicating mild bullish pressure. Watch the 20-day SMA for a potential crossover signal.",
                },
                "news": {
                    "sentiment": round(random.uniform(0.3, 0.8), 2),
                    "summary": f"Recent analyst coverage for {sym} has been moderately positive, with consensus estimates reflecting steady growth expectations for the current fiscal year.",
                    "bullets": [
                        "Q4 earnings beat consensus by 3.2%",
                        "Management reaffirmed full-year guidance",
                        "Institutional ownership increased 1.5% last quarter",
                    ],
                },
                "signal": None,
            }
            with open(DATA_DIR / f"{sym}.json", "w") as f:
                json.dump(report, f, indent=2)

    # Sample signals
    signals_data = {
        "signals": [
            {"symbol": "AMZN", "type": "BUY", "strategy": "bollinger", "reason": "Price touched lower Bollinger Band with RSI oversold"},
            {"symbol": "PFE", "type": "BUY", "strategy": "oversold_bounce", "reason": "RSI < 30 with volume spike"},
            {"symbol": "NFLX", "type": "SELL", "strategy": "bollinger", "reason": "Price above upper Bollinger Band, RSI overbought"},
        ],
        "date": "2026-02-11",
        "count": 3,
    }
    with open(DATA_DIR / "signals_today.json", "w") as f:
        json.dump(signals_data, f, indent=2)

    print(f"Generated sample data in {DATA_DIR}/")
    print(f"  {len(SECTORS)} sectors, {total_stocks} stocks")
    print(f"  {sum(min(3, len(s['stocks'])) for s in SECTORS.values())} symbol reports")
    print(f"  3 sample signals")


if __name__ == "__main__":
    generate()
