from __future__ import annotations

"""
Generate per-symbol JSON reports using existing bar_fly_trading utilities.

Wraps rt_utils (get_technical_data, summarize_technical_with_llm, get_news_sentiment)
and writes {SYMBOL}.json + signals_today.json.

Usage:
    python -m webapp.backend.generate_reports
    python -m webapp.backend.generate_reports --symbols AAPL,NVDA,JPM
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("BFT_DATA_DIR", Path(__file__).parent.parent / "data"))


def get_all_symbols(csv_pattern: str) -> list[str]:
    """Get unique symbols from all_data CSV files."""
    csv_files = sorted(glob.glob(csv_pattern))
    symbols = set()
    for f in csv_files:
        try:
            df = pd.read_csv(f, usecols=["symbol"])
            symbols.update(df["symbol"].unique())
        except Exception:
            continue
    return sorted(symbols)


def generate_symbol_report(symbol: str, csv_pattern: str) -> dict:
    """Generate a report dict for a single symbol using existing rt_utils."""
    report = {
        "symbol": symbol,
        "report_date": datetime.now().strftime("%Y-%m-%d"),
        "technical": None,
        "news": None,
        "signal": None,
    }

    try:
        from api_data.rt_utils import get_technical_data
        tech = get_technical_data(symbol, csv_pattern)
        if tech:
            report["technical"] = {
                "rsi": tech.get("RSI 14"),
                "macd": tech.get("MACD"),
                "macd_signal": tech.get("MACD Signal Line"),
                "adx": tech.get("ADX 14"),
                "cci": tech.get("CCI 14"),
                "atr": tech.get("ATR 14"),
                "sma_20": tech.get("SMA 20"),
                "sma_50": tech.get("SMA 50"),
                "bbands_upper": tech.get("Bollinger Upper"),
                "bbands_lower": tech.get("Bollinger Lower"),
                "pe_ratio": tech.get("pe_ratio"),
                "price": tech.get("Price"),
            }
    except Exception as e:
        logger.warning(f"  Technical data failed for {symbol}: {e}")

    try:
        from api_data.rt_utils import summarize_technical_with_llm
        if report["technical"]:
            summary = summarize_technical_with_llm(symbol, report["technical"])
            if summary:
                report["technical"]["summary"] = summary
    except Exception as e:
        logger.debug(f"  LLM summary skipped for {symbol}: {e}")

    try:
        from api_data.rt_utils import get_news_sentiment
        news = get_news_sentiment(symbol)
        if news:
            report["news"] = news
    except Exception as e:
        logger.debug(f"  News sentiment skipped for {symbol}: {e}")

    return report


def generate_reports(csv_pattern: str, symbols: list[str] = None):
    """Generate reports for all (or specified) symbols."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if symbols is None:
        symbols = get_all_symbols(csv_pattern)

    logger.info(f"Generating reports for {len(symbols)} symbols...")

    signals_today = []

    for i, symbol in enumerate(symbols):
        logger.info(f"  [{i+1}/{len(symbols)}] {symbol}")
        try:
            report = generate_symbol_report(symbol, csv_pattern)

            # Merge into existing file (preserve quote, meta, etc. from populate_all)
            report_file = DATA_DIR / f"{symbol}.json"
            if report_file.exists():
                try:
                    with open(report_file) as f:
                        existing = json.load(f)
                    # Overlay report fields onto existing data
                    for key in ("technical", "news", "signal"):
                        if report.get(key) is not None:
                            existing[key] = report[key]
                    existing["report_date"] = report["report_date"]
                    report = existing
                except Exception:
                    pass

            with open(report_file, "w") as f:
                json.dump(report, f, indent=2)

            # Collect signals
            if report.get("signal"):
                signals_today.append({
                    "symbol": symbol,
                    "type": report["signal"].get("type"),
                    "strategy": report["signal"].get("strategy"),
                    "reason": report["signal"].get("reason", ""),
                })
        except Exception as e:
            logger.error(f"  Failed for {symbol}: {e}")

    # Write signals_today.json
    signals_file = DATA_DIR / "signals_today.json"
    with open(signals_file, "w") as f:
        json.dump({
            "signals": signals_today,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "count": len(signals_today),
        }, f, indent=2)

    logger.info(f"Done. {len(symbols)} reports, {len(signals_today)} signals.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-pattern", default="/home/stvschmdt/data/all_data_*.csv")
    parser.add_argument("--symbols", help="Comma-separated symbols (default: all)")
    args = parser.parse_args()

    symbols = args.symbols.split(",") if args.symbols else None
    generate_reports(args.csv_pattern, symbols)
