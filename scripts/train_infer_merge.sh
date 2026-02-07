#!/bin/bash
# Full pipeline: train all 9 models in parallel, then merge predictions
#
# Wraps run_parallel_training.sh and adds the merge step when all jobs finish.
# This is the "weekly retrain" script.
#
# Usage:
#   ./scripts/train_infer_merge.sh [--encoder|--cross-attention]
#
# Timeline: ~12 hours (3 parallel training jobs, each doing 3 horizons)

set -e

cd ~/proj/bar_fly_trading

MODEL_TYPE="encoder"
if [[ "$1" == "--cross-attention" ]]; then
    MODEL_TYPE="cross_attention"
fi

OUTPUT_DIR="stockformer/output"
mkdir -p logs "$OUTPUT_DIR"

echo "============================================"
echo "FULL PIPELINE: Train + Infer + Merge"
echo "Model type: $MODEL_TYPE"
echo "Started at $(date)"
echo "============================================"
echo ""

# Step 1: Train all 9 models in parallel (this script backgrounds 3 processes)
echo "Step 1/2: Training all 9 models..."
echo ""

DATA_PATH="./all_data_*.csv"
MARKET_PATH="./market_embeddings.csv"
SECTOR_PATH="./sector_embeddings.csv"
TRAIN_END="2025-10-31"
INFER_START="2025-11-01"
BATCH_SIZE=64
EPOCHS=15
WORKERS=0

EMBEDDING_ARGS=""
if [[ "$MODEL_TYPE" == "cross_attention" ]]; then
    EMBEDDING_ARGS="--market-path $MARKET_PATH --sector-path $SECTOR_PATH"
fi

PIDS=()
for LABEL_MODE in regression binary buckets; do
    echo "  Launching $LABEL_MODE training..."
    nohup python -u -m stockformer.main \
        --data-path "$DATA_PATH" \
        --train-end-date $TRAIN_END \
        --infer-start-date $INFER_START \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --num-workers $WORKERS \
        --model-type $MODEL_TYPE \
        $EMBEDDING_ARGS \
        --label-mode $LABEL_MODE \
        > "logs/train_${LABEL_MODE}_${MODEL_TYPE}_$(date +%Y%m%d).log" 2>&1 &
    PIDS+=($!)
    echo "    PID: $!"
done

echo ""
echo "  Waiting for all training to finish..."
echo "  Monitor: tail -f logs/train_*_${MODEL_TYPE}_$(date +%Y%m%d).log"
echo ""

# Wait for all training jobs
for PID in "${PIDS[@]}"; do
    wait $PID
done

echo "Training complete at $(date)"
echo ""

# Step 2: Merge predictions
echo "Step 2/2: Merging predictions..."
python -m stockformer.merge_predictions \
    --input-dir "$OUTPUT_DIR" \
    --output merged_predictions.csv

echo ""
echo "============================================"
echo "PIPELINE COMPLETE"
echo "  Output: merged_predictions.csv"
echo "  Finished at $(date)"
echo "============================================"
