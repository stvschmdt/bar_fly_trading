#!/bin/bash
# Multi-symbol real-time report
#
# Loops over a symbol list and runs rt_utils for each symbol.
# Useful for morning review of top portfolio picks.
#
# Usage:
#   ./scripts/rt_report.sh                            # uses portfolio_picks.csv
#   ./scripts/rt_report.sh --symbols AAPL NVDA MSFT   # explicit symbols
#   ./scripts/rt_report.sh --email                     # send email for each
#   ./scripts/rt_report.sh --news-only                 # skip earnings (faster)
#
# Defaults:
#   SYMBOLS_FILE=portfolio_picks.csv
#   Flags: --news --summary (full report)

set -e

cd ~/proj/bar_fly_trading

# Defaults
SYMBOLS_FILE="${SYMBOLS_FILE:-portfolio_picks.csv}"
SYMBOLS=""
EMAIL_FLAG=""
NEWS_ONLY=""
EXTRA_FLAGS=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --symbols) shift; SYMBOLS=""; while [[ $# -gt 0 && ! "$1" == --* ]]; do SYMBOLS="$SYMBOLS $1"; shift; done ;;
        --symbols-file) SYMBOLS_FILE="$2"; shift 2 ;;
        --email) EMAIL_FLAG="--email"; shift ;;
        --news-only) NEWS_ONLY="--no-earnings"; shift ;;
        --months-out) EXTRA_FLAGS="$EXTRA_FLAGS --months-out $2"; shift 2 ;;
        --strikes) EXTRA_FLAGS="$EXTRA_FLAGS --strikes $2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Build symbol list
if [[ -n "$SYMBOLS" ]]; then
    SYMBOL_LIST=($SYMBOLS)
elif [[ -f "$SYMBOLS_FILE" ]]; then
    # Read symbols from CSV (skip header, first column)
    SYMBOL_LIST=($(tail -n +2 "$SYMBOLS_FILE" | cut -d',' -f1 | tr -d ' '))
else
    echo "Error: No symbols provided and $SYMBOLS_FILE not found"
    echo "Usage: ./scripts/rt_report.sh --symbols AAPL NVDA"
    exit 1
fi

echo "============================================"
echo "REAL-TIME REPORT: $(date)"
echo "  Symbols: ${SYMBOL_LIST[*]}"
echo "  Count: ${#SYMBOL_LIST[@]}"
echo "  Email: ${EMAIL_FLAG:-disabled}"
echo "  News only: ${NEWS_ONLY:-no (full report)}"
echo "============================================"
echo ""

TOTAL=${#SYMBOL_LIST[@]}
COUNT=0

for SYMBOL in "${SYMBOL_LIST[@]}"; do
    COUNT=$((COUNT + 1))
    echo "────────────────────────────────────────────"
    echo "[$COUNT/$TOTAL] $SYMBOL"
    echo "────────────────────────────────────────────"

    python -m api_data.rt_utils "$SYMBOL" \
        --news --summary \
        $NEWS_ONLY \
        $EMAIL_FLAG \
        $EXTRA_FLAGS \
        || echo "  WARNING: Failed for $SYMBOL, continuing..."

    echo ""

    # Rate limit: Alpha Vantage free tier is 5 calls/min
    # Each symbol makes ~3 API calls (quote, news, earnings)
    if [[ $COUNT -lt $TOTAL ]]; then
        echo "  (waiting 15s for API rate limit...)"
        sleep 15
    fi
done

echo "============================================"
echo "Report complete for ${#SYMBOL_LIST[@]} symbols"
echo "Finished at $(date)"
echo "============================================"
