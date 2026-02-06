#!/bin/bash
# Portfolio filter → backtest → output trades + signals
#
# Takes merged predictions and runs the full backtest flow:
# 1. Portfolio filter/rank to narrow symbols
# 2. Backtest strategy on filtered symbols
# 3. Output trade log + signal CSV for execution
#
# Usage:
#   ./scripts/backtest_pipeline.sh
#   ./scripts/backtest_pipeline.sh --strategy regression_momentum
#   ./scripts/backtest_pipeline.sh --strategy ml_prediction
#   ./scripts/backtest_pipeline.sh --symbols-file my_picks.csv
#
# Defaults:
#   PREDICTIONS=merged_predictions.csv
#   STRATEGY=regression_momentum
#   PORTFOLIO_DATA=all_data_0.csv
#   START_DATE=2024-07-01
#   END_DATE=2024-12-31
#   START_CASH=100000
#   PRICE_ABOVE=25
#   TOP_K_SHARPE=15

set -e

cd ~/proj/bar_fly_trading

# Defaults (override via env vars or edit here)
PREDICTIONS="${PREDICTIONS:-merged_predictions.csv}"
STRATEGY="${STRATEGY:-regression_momentum}"
PORTFOLIO_DATA="${PORTFOLIO_DATA:-all_data_0.csv}"
START_DATE="${START_DATE:-2024-07-01}"
END_DATE="${END_DATE:-2024-12-31}"
START_CASH="${START_CASH:-100000}"
PRICE_ABOVE="${PRICE_ABOVE:-25}"
TOP_K_SHARPE="${TOP_K_SHARPE:-15}"
SYMBOLS_FILE="${SYMBOLS_FILE:-}"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --strategy) STRATEGY="$2"; shift 2 ;;
        --symbols-file) SYMBOLS_FILE="$2"; shift 2 ;;
        --start-date) START_DATE="$2"; shift 2 ;;
        --end-date) END_DATE="$2"; shift 2 ;;
        --start-cash) START_CASH="$2"; shift 2 ;;
        --predictions) PREDICTIONS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Output paths
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="backtest_results"
mkdir -p "$OUTPUT_DIR" signals

TRADES_FILE="$OUTPUT_DIR/trades_${STRATEGY}_${TIMESTAMP}.csv"
SYMBOLS_OUT="$OUTPUT_DIR/symbols_${STRATEGY}_${TIMESTAMP}.csv"
SIGNALS_FILE="signals/pending_orders.csv"

echo "============================================"
echo "BACKTEST PIPELINE"
echo "  Strategy: $STRATEGY"
echo "  Predictions: $PREDICTIONS"
echo "  Date range: $START_DATE to $END_DATE"
echo "  Starting cash: \$$START_CASH"
echo "============================================"
echo ""

# Step 1: Portfolio filter (skip if symbols-file provided)
if [[ -n "$SYMBOLS_FILE" ]]; then
    echo "Step 1: Using provided symbols file: $SYMBOLS_FILE"
    SYMBOL_ARG="--symbols-file $SYMBOLS_FILE"
else
    echo "Step 1: Portfolio filtering (price > \$$PRICE_ABOVE, top $TOP_K_SHARPE by Sharpe)..."

    PORTFOLIO_SYMBOLS="$OUTPUT_DIR/portfolio_picks_${TIMESTAMP}.csv"
    python strategies/portfolio.py \
        --data "$PORTFOLIO_DATA" \
        --price-above "$PRICE_ABOVE" \
        --top-k-sharpe "$TOP_K_SHARPE" \
        --output "$PORTFOLIO_SYMBOLS" \
        --summary

    SYMBOL_ARG="--symbols-file $PORTFOLIO_SYMBOLS"
    echo ""
fi

# Step 2: Backtest
echo "Step 2: Running $STRATEGY backtest..."
echo ""

if [[ "$STRATEGY" == "regression_momentum" ]]; then
    python strategies/run_regression_momentum.py \
        --predictions "$PREDICTIONS" \
        $SYMBOL_ARG \
        --start-date "$START_DATE" \
        --end-date "$END_DATE" \
        --start-cash "$START_CASH" \
        --output-trades "$TRADES_FILE" \
        --output-symbols "$SYMBOLS_OUT" \
        --output-signals "$SIGNALS_FILE"

elif [[ "$STRATEGY" == "ml_prediction" ]]; then
    python strategies/run_backtest.py \
        --predictions "$PREDICTIONS" \
        $SYMBOL_ARG \
        --start-date "$START_DATE" \
        --end-date "$END_DATE" \
        --start-cash "$START_CASH" \
        --output-trades "$TRADES_FILE" \
        --output-symbols "$SYMBOLS_OUT" \
        --output-signals "$SIGNALS_FILE"

else
    echo "Unknown strategy: $STRATEGY"
    echo "Options: regression_momentum, ml_prediction"
    exit 1
fi

echo ""
echo "============================================"
echo "BACKTEST COMPLETE"
echo "  Trade log:     $TRADES_FILE"
echo "  Symbol list:   $SYMBOLS_OUT"
echo "  Signals:       $SIGNALS_FILE"
echo ""
echo "Next steps:"
echo "  # Preview signals"
echo "  python -m ibkr.execute_signals --signals $SIGNALS_FILE --dry-run"
echo ""
echo "  # Execute on paper account"
echo "  python -m ibkr.execute_signals --signals $SIGNALS_FILE"
echo "============================================"
