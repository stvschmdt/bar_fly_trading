#!/bin/bash
# Inference-only + merge predictions
#
# Runs inference on all 9 pre-trained models (3 label modes x 3 horizons),
# then merges into a single merged_predictions.csv.
#
# Launches 3 parallel jobs (one per label mode), each running 3 horizons
# sequentially. Total: 9 inference runs, ~3x faster than serial.
#
# Use this when models are already trained and you have new data to score.
#
# Usage:
#   ./scripts/infer_merge.sh [--encoder|--cross-attention]
#
# Prerequisites:
#   - Trained model checkpoints in stockformer/output/models/
#   - Fresh all_data_*.csv in data directory

set -e

cd ~/proj/bar_fly_trading

# Parse args
MODEL_TYPE="encoder"
if [[ "$1" == "--cross-attention" ]]; then
    MODEL_TYPE="cross_attention"
    echo "Using cross-attention models"
elif [[ "$1" == "--encoder" ]] || [[ -z "$1" ]]; then
    MODEL_TYPE="encoder"
    echo "Using encoder models"
fi

DATA_PATH="./all_data_*.csv"
MARKET_PATH="./market_embeddings.csv"
SECTOR_PATH="./sector_embeddings.csv"
INFER_START="2025-07-01"
BATCH_SIZE=64
WORKERS=0
MODEL_DIR="stockformer/output/models"
PRED_DIR="stockformer/output/predictions"

mkdir -p logs "$MODEL_DIR" "$PRED_DIR"

echo "Starting inference at $(date)"
echo "============================================"

EMBEDDING_ARGS=""
if [[ "$MODEL_TYPE" == "cross_attention" ]]; then
    EMBEDDING_ARGS="--market-path $MARKET_PATH --sector-path $SECTOR_PATH"
fi

# Label mode tags: (mode, short_tag)
declare -A LABEL_TAGS
LABEL_TAGS[regression]="reg"
LABEL_TAGS[binary]="bin"
LABEL_TAGS[buckets]="buck"

HORIZONS=(3 10 30)
HORIZON_TAGS=(3d 10d 30d)

# Launch one parallel job per label mode; each job runs 3 horizons sequentially
for LABEL_MODE in regression binary buckets; do
    TAG="${LABEL_TAGS[$LABEL_MODE]}"
    LOG="logs/infer_${TAG}_${MODEL_TYPE}_$(date +%Y%m%d).log"

    echo "Launching $LABEL_MODE inference (3 horizons) -> $LOG"

    (
        for i in "${!HORIZONS[@]}"; do
            H="${HORIZONS[$i]}"
            HT="${HORIZON_TAGS[$i]}"
            SUFFIX="${TAG}_${HT}"
            MODEL_PATH="${MODEL_DIR}/model_${SUFFIX}.pt"
            OUTPUT_PATH="${PRED_DIR}/pred_${SUFFIX}.csv"

            if [[ ! -f "$MODEL_PATH" ]]; then
                echo "SKIP $SUFFIX: model not found at $MODEL_PATH"
                continue
            fi

            # Buckets mode needs auto bucket edges to match training
            BUCKET_ARGS=""
            if [[ "$LABEL_MODE" == "buckets" ]]; then
                BUCKET_ARGS="--bucket-edges auto"
            fi

            echo ""
            echo "=== Inference: ${LABEL_MODE} ${H}d ==="
            python -u -m stockformer.main \
                --data-path "$DATA_PATH" \
                --infer-start-date $INFER_START \
                --batch-size $BATCH_SIZE \
                --num-workers $WORKERS \
                $EMBEDDING_ARGS \
                $BUCKET_ARGS \
                --horizon $H \
                --label-mode $LABEL_MODE \
                --model-out "$MODEL_PATH" \
                --output-csv "$OUTPUT_PATH" \
                --infer-only
        done
    ) > "$LOG" 2>&1 &

    eval "PID_${LABEL_MODE}=$!"
    echo "  $LABEL_MODE PID: $!"
done

echo ""
echo "Waiting for all 3 inference jobs to complete..."
wait $PID_regression $PID_binary $PID_buckets
echo "All inference jobs finished at $(date)"

# Check for errors
FAILED=0
for LABEL_MODE in regression binary buckets; do
    TAG="${LABEL_TAGS[$LABEL_MODE]}"
    LOG="logs/infer_${TAG}_${MODEL_TYPE}_$(date +%Y%m%d).log"
    if grep -q "Error\|Exception\|Traceback" "$LOG" 2>/dev/null; then
        echo "WARNING: $LABEL_MODE may have errors — check $LOG"
        FAILED=1
    fi
done

if [[ $FAILED -eq 1 ]]; then
    echo "Some jobs had warnings. Check logs before proceeding."
fi

# Merge predictions
echo ""
echo "Merging predictions..."
python -m stockformer.merge_predictions \
    --input-dir "$PRED_DIR" \
    --output merged_predictions.csv

echo ""
echo "============================================"
echo "Done! Output: merged_predictions.csv"
echo "Finished at $(date)"
