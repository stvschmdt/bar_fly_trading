#!/bin/bash
# Intraday realtime pipeline.
#
# Fetches fresh OHLCV from AlphaVantage GLOBAL_QUOTE, updates all_data CSVs
# in place, optionally runs ML inference, then runs strategies in daily mode.
# Technical indicators are kept from the nightly pull (1 API call/symbol).
#
# Usage:
#   ./run_intraday.sh                    # Full pipeline (fetch + inference + strategies)
#   ./run_intraday.sh --skip-inference   # Skip ML inference (~4 min fetch only)
#   ./run_intraday.sh --dry-run          # Fetch data but don't write or run strategies
#   ./run_intraday.sh --symbols AAPL NVDA TSLA  # Only update specific symbols
#   ./run_intraday.sh --watchlist api_data/watchlist.csv  # Only watchlist (68 symbols, ~30 sec)
#
# Default: SP500 + watchlist (~543 symbols, ~4 min fetch at 140 req/min).
# Can run every 15 minutes during market hours via cron:
#
#   # Crontab — every 15 min, 9:31 to 3:46 ET (adjust for your timezone):
#   31,46 9 * * 1-5   cd ~/bar_fly_trading && bash run_intraday.sh --skip-inference >> /tmp/intraday.log 2>&1
#   1,16,31,46 10-14 * * 1-5 cd ~/bar_fly_trading && bash run_intraday.sh --skip-inference >> /tmp/intraday.log 2>&1
#   1,16,31 15 * * 1-5 cd ~/bar_fly_trading && bash run_intraday.sh --skip-inference >> /tmp/intraday.log 2>&1
#
#   # Or hourly at :31 past the hour:
#   31 9-15 * * 1-5  cd ~/bar_fly_trading && bash run_intraday.sh --skip-inference >> /tmp/intraday.log 2>&1

set -euo pipefail

cd "$(dirname "$0")"
PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:-.}"
MODEL_DIR="stockformer/output/models"
PRED_DIR="stockformer/output/predictions"

# Parse flags
SKIP_INFERENCE=false
DRY_RUN=false
SYMBOL_ARGS=""
WATCHLIST_ARGS="-w all"  # Default: SP500 + watchlist

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-inference) SKIP_INFERENCE=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --watchlist|-w)
            shift
            WATCHLIST_ARGS="-w $1"
            shift
            ;;
        --symbols|-s)
            shift
            SYMBOL_ARGS="-s"
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SYMBOL_ARGS="$SYMBOL_ARGS $1"
                shift
            done
            ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

echo "========================================"
echo "Intraday RT Pipeline — $(date)"
echo "skip-inference=$SKIP_INFERENCE dry-run=$DRY_RUN"
echo "========================================"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Fetch RT data and update CSVs
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Step 1: RT Data Fetch ==="

RT_ARGS="--data-dir $DATA_DIR $WATCHLIST_ARGS"
if [ "$DRY_RUN" = true ]; then
    RT_ARGS="$RT_ARGS --dry-run"
fi
if [ -n "$SYMBOL_ARGS" ]; then
    RT_ARGS="$RT_ARGS $SYMBOL_ARGS"
fi

$PYTHON -m api_data.pull_api_data_rt $RT_ARGS

if [ "$DRY_RUN" = true ]; then
    echo "Dry run — skipping remaining steps."
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: ML Inference (optional)
# ─────────────────────────────────────────────────────────────────────────────
if [ "$SKIP_INFERENCE" = false ]; then
    echo ""
    echo "=== Step 2: ML Inference ==="

    for HORIZON in 3 10 30; do
        for LMODE in regression binary; do
            if [ "$LMODE" = "regression" ]; then
                TAG="reg_${HORIZON}d"
            else
                TAG="bin_${HORIZON}d"
            fi

            MODEL="${MODEL_DIR}/model_${TAG}.pt"
            OUTPUT="${PRED_DIR}/pred_${TAG}.csv"

            if [ ! -f "$MODEL" ]; then
                echo "  SKIP: $MODEL not found"
                continue
            fi

            echo "  Inference: ${TAG}"
            $PYTHON -u -m stockformer.main \
                --data-path "${DATA_DIR}/all_data_*.csv" \
                --lookback 20 \
                --batch-size 256 \
                --horizon $HORIZON \
                --label-mode $LMODE \
                --model-out "$MODEL" \
                --output-csv "$OUTPUT" \
                --infer-only 2>&1 | tail -3
        done
    done

    # Bucket models
    for HORIZON in 3 10 30; do
        TAG="buck_${HORIZON}d"
        MODEL="${MODEL_DIR}/model_${TAG}.pt"
        OUTPUT="${PRED_DIR}/pred_${TAG}.csv"

        if [ ! -f "$MODEL" ]; then
            echo "  SKIP: $MODEL not found"
            continue
        fi

        echo "  Inference: ${TAG}"
        $PYTHON -u -m stockformer.main \
            --data-path "${DATA_DIR}/all_data_*.csv" \
            --lookback 20 \
            --batch-size 256 \
            --bucket-edges="-2,0,2" \
            --horizon $HORIZON \
            --label-mode buckets \
            --model-out "$MODEL" \
            --output-csv "$OUTPUT" \
            --infer-only 2>&1 | tail -3
    done

    # Merge predictions
    echo "  Merging predictions..."
    $PYTHON -m stockformer.merge_predictions \
        --input-dir "$PRED_DIR" \
        --output merged_predictions_rt.csv

    echo "=== Inference complete ==="
else
    echo ""
    echo "=== Step 2: ML Inference — SKIPPED ==="
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Run strategies in daily mode
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "=== Step 3: Strategies (daily mode) ==="

DATA_PATH="${DATA_DIR}/all_data_*.csv"
PRED_PATH="merged_predictions_rt.csv"
if [ "$SKIP_INFERENCE" = true ] || [ ! -f "$PRED_PATH" ]; then
    PRED_PATH="merged_predictions_v3.csv"
fi

COMMON_ARGS="--data-path '$DATA_PATH' --use-all-symbols --mode daily --lookback-days 2 --skip-live --summary-only"

echo "  Running oversold bounce..."
eval $PYTHON strategies/run_oversold_bounce.py $COMMON_ARGS 2>&1 | tail -5

echo "  Running bollinger..."
eval $PYTHON strategies/run_bollinger.py $COMMON_ARGS 2>&1 | tail -5

echo "  Running low bounce..."
eval $PYTHON strategies/run_low_bounce.py $COMMON_ARGS 2>&1 | tail -5

echo "  Running oversold reversal..."
eval $PYTHON strategies/run_oversold_reversal.py $COMMON_ARGS --predictions "$PRED_PATH" 2>&1 | tail -5

# regression_momentum disabled — missing 10d model, produces 0 trades
# echo "  Running regression momentum..."
# eval $PYTHON strategies/run_regression_momentum.py $COMMON_ARGS --predictions "$PRED_PATH" 2>&1 | tail -5

echo ""
echo "========================================"
echo "Intraday RT Pipeline COMPLETE — $(date)"
echo "========================================"
