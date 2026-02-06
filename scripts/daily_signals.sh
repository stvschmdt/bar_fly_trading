#!/bin/bash
# Daily/hourly signal generation + optional execution
#
# Runs strategy in signal mode (evaluate once for today), then optionally
# feeds the signal CSV to the executor.
#
# Designed for cron:
#   # Every hour during market hours (9:30am - 4pm ET, Mon-Fri)
#   30 9-15 * * 1-5 ~/proj/bar_fly_trading/scripts/daily_signals.sh
#
# Usage:
#   ./scripts/daily_signals.sh                    # generate signals only (dry run)
#   ./scripts/daily_signals.sh --execute          # generate + execute on paper
#   ./scripts/daily_signals.sh --execute --live   # generate + execute live (requires trading_mode.conf=live)
#
# Defaults:
#   PREDICTIONS=merged_predictions.csv
#   SYMBOLS_FILE=portfolio_picks.csv (or set SYMBOLS="AAPL NVDA MSFT")

set -e

cd ~/proj/bar_fly_trading

# Defaults
PREDICTIONS="${PREDICTIONS:-merged_predictions.csv}"
SYMBOLS_FILE="${SYMBOLS_FILE:-portfolio_picks.csv}"
SYMBOLS="${SYMBOLS:-}"
EXECUTE=false
LIVE_FLAG=""

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --execute) EXECUTE=true; shift ;;
        --live) LIVE_FLAG="--live"; shift ;;
        --symbols) SYMBOLS="$2"; shift 2 ;;
        --symbols-file) SYMBOLS_FILE="$2"; shift 2 ;;
        --predictions) PREDICTIONS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Build symbol arg
if [[ -n "$SYMBOLS" ]]; then
    SYMBOL_ARG="--symbols $SYMBOLS"
elif [[ -f "$SYMBOLS_FILE" ]]; then
    SYMBOL_ARG="--symbols-file $SYMBOLS_FILE"
else
    echo "Error: No symbols provided and $SYMBOLS_FILE not found"
    echo "Set SYMBOLS='AAPL NVDA' or SYMBOLS_FILE=path.csv"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SIGNALS_DIR="signals"
SIGNALS_FILE="$SIGNALS_DIR/pending_orders_${TIMESTAMP}.csv"
mkdir -p "$SIGNALS_DIR" logs

LOG="logs/daily_signals_${TIMESTAMP}.log"

echo "============================================" | tee "$LOG"
echo "DAILY SIGNAL RUN: $(date)" | tee -a "$LOG"
echo "  Predictions: $PREDICTIONS" | tee -a "$LOG"
echo "  Symbols: ${SYMBOLS:-from $SYMBOLS_FILE}" | tee -a "$LOG"
echo "  Execute: $EXECUTE $LIVE_FLAG" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Step 1: Generate signals
echo "Generating signals..." | tee -a "$LOG"
python strategies/run_template.py \
    --mode signals \
    --predictions "$PREDICTIONS" \
    $SYMBOL_ARG \
    --output-signals "$SIGNALS_FILE" \
    2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"

# Check if signal file was created (no signals = no file)
if [[ ! -f "$SIGNALS_FILE" ]]; then
    echo "No signals generated (hold/do nothing)" | tee -a "$LOG"
    exit 0
fi

SIGNAL_COUNT=$(tail -n +2 "$SIGNALS_FILE" | wc -l)
echo "$SIGNAL_COUNT signal(s) generated in $SIGNALS_FILE" | tee -a "$LOG"

# Step 2: Execute (if requested)
if [[ "$EXECUTE" == true ]]; then
    echo "" | tee -a "$LOG"
    echo "Executing signals..." | tee -a "$LOG"
    python -m ibkr.execute_signals \
        --signals "$SIGNALS_FILE" \
        $LIVE_FLAG \
        2>&1 | tee -a "$LOG"
else
    echo "" | tee -a "$LOG"
    echo "Dry run (signals written but not executed):" | tee -a "$LOG"
    python -m ibkr.execute_signals \
        --signals "$SIGNALS_FILE" \
        --dry-run \
        2>&1 | tee -a "$LOG"
    echo "" | tee -a "$LOG"
    echo "To execute: python -m ibkr.execute_signals --signals $SIGNALS_FILE" | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"
echo "Done at $(date)" | tee -a "$LOG"
