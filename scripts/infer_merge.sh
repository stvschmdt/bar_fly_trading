#!/bin/bash
# Inference-only + merge predictions
#
# Runs inference on all 9 pre-trained models in parallel (3 label modes x 3 horizons),
# then merges into a single merged_predictions.csv.
#
# Use this when models are already trained and you have new data to score.
#
# Usage:
#   ./scripts/infer_merge.sh [--encoder|--cross-attention]
#
# Prerequisites:
#   - Trained model checkpoints in stockformer/output/
#   - Fresh all_data_*.csv in project root

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
INFER_START="2025-11-01"
BATCH_SIZE=64
WORKERS=0
OUTPUT_DIR="stockformer/output"

mkdir -p logs "$OUTPUT_DIR"

echo "Starting inference at $(date)"
echo "============================================"

EMBEDDING_ARGS=""
if [[ "$MODEL_TYPE" == "cross_attention" ]]; then
    EMBEDDING_ARGS="--market-path $MARKET_PATH --sector-path $SECTOR_PATH"
fi

# Run inference for each label mode in parallel (each does 3 horizons)
for LABEL_MODE in regression binary buckets; do
    echo "Launching $LABEL_MODE inference..."
    nohup python -u -m stockformer.main \
        --data-path "$DATA_PATH" \
        --infer-start-date $INFER_START \
        --batch-size $BATCH_SIZE \
        --num-workers $WORKERS \
        --model-type $MODEL_TYPE \
        $EMBEDDING_ARGS \
        --label-mode $LABEL_MODE \
        --infer-only \
        > "logs/infer_${LABEL_MODE}_${MODEL_TYPE}_$(date +%Y%m%d).log" 2>&1 &
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
    LOG="logs/infer_${LABEL_MODE}_${MODEL_TYPE}_$(date +%Y%m%d).log"
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
    --input-dir "$OUTPUT_DIR" \
    --output merged_predictions.csv

echo ""
echo "============================================"
echo "Done! Output: merged_predictions.csv"
echo "Finished at $(date)"
