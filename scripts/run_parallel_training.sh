#!/bin/bash
# Parallel training script for stockformer — all 9 models
# Runs 3 label modes in parallel, each handling 3 horizons sequentially
# Each job specifies BOTH --horizon and --label-mode to avoid running all 9
#
# Usage:
#   ./scripts/run_parallel_training.sh [--encoder|--cross-attention] [--epochs N]
#
# Model types:
#   --encoder         Use original bidirectional encoder (default)
#   --cross-attention Use new market encoder + stock decoder with cross-attention

set -e

cd ~/proj/bar_fly_trading

# Parse command line args
MODEL_TYPE="encoder"
EPOCHS=15
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cross-attention)
            MODEL_TYPE="cross_attention"
            shift ;;
        --encoder)
            MODEL_TYPE="encoder"
            shift ;;
        --epochs)
            EPOCHS="$2"
            shift 2 ;;
        *)
            echo "Unknown arg: $1"; exit 1 ;;
    esac
done

DATA_PATH="./all_data_*.csv"
MARKET_PATH="./market_embeddings.csv"
SECTOR_PATH="./sector_embeddings.csv"
TRAIN_END="2025-10-31"
INFER_START="2025-11-01"
BATCH_SIZE=64
WORKERS=0
DATE_TAG=$(date +%Y%m%d)

mkdir -p logs

echo "Starting parallel training at $(date)"
echo "============================================"
echo "  Model type: $MODEL_TYPE"
echo "  Epochs: $EPOCHS"
echo ""

EMBEDDING_ARGS=""
if [[ "$MODEL_TYPE" == "cross_attention" ]]; then
    EMBEDDING_ARGS="--market-path $MARKET_PATH --sector-path $SECTOR_PATH"
    echo "  Using cross-attention with embeddings"
fi

# Helper: train a single model with explicit horizon + label-mode
train_one() {
    local HORIZON=$1
    local LABEL_MODE=$2
    local TAG=$3  # e.g. reg_3d, bin_10d, buck_30d

    echo "  Training ${TAG} (horizon=${HORIZON}, label_mode=${LABEL_MODE})..."
    python -u -m stockformer.main \
        --data-path "$DATA_PATH" \
        --train-end-date $TRAIN_END \
        --infer-start-date $INFER_START \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --num-workers $WORKERS \
        --model-type $MODEL_TYPE \
        $EMBEDDING_ARGS \
        --horizon $HORIZON \
        --label-mode $LABEL_MODE \
        --model-out model_checkpoint_${TAG}.pt \
        --log-path training_log_${TAG}.csv \
        --output-csv predictions_${TAG}.csv \
        --save-config
    echo "  Finished ${TAG} at $(date)"
}

# Regression: 3d -> 10d -> 30d (sequential within process)
(
    train_one 3  regression reg_3d
    train_one 10 regression reg_10d
    train_one 30 regression reg_30d
) > logs/train_regression_${MODEL_TYPE}_${DATE_TAG}.log 2>&1 &
PID_REG=$!
echo "Launched regression (3d,10d,30d) PID: $PID_REG"

# Binary: 3d -> 10d -> 30d
(
    train_one 3  binary bin_3d
    train_one 10 binary bin_10d
    train_one 30 binary bin_30d
) > logs/train_binary_${MODEL_TYPE}_${DATE_TAG}.log 2>&1 &
PID_BIN=$!
echo "Launched binary (3d,10d,30d) PID: $PID_BIN"

# Buckets: 3d -> 10d -> 30d
(
    train_one 3  buckets buck_3d
    train_one 10 buckets buck_10d
    train_one 30 buckets buck_30d
) > logs/train_buckets_${MODEL_TYPE}_${DATE_TAG}.log 2>&1 &
PID_BUCK=$!
echo "Launched buckets (3d,10d,30d) PID: $PID_BUCK"

echo ""
echo "============================================"
echo "All 3 processes launched (9 models total)"
echo "  Regression: $PID_REG"
echo "  Binary:     $PID_BIN"
echo "  Buckets:    $PID_BUCK"
echo ""
echo "Monitor:"
echo "  tail -f logs/train_*_${MODEL_TYPE}_${DATE_TAG}.log"
echo "  nvidia-smi"
echo ""

# Wait for all to finish
echo "Waiting for all jobs to complete..."
wait $PID_REG && echo "Regression done" || echo "WARNING: Regression failed"
wait $PID_BIN && echo "Binary done" || echo "WARNING: Binary failed"
wait $PID_BUCK && echo "Buckets done" || echo "WARNING: Buckets failed"

echo ""
echo "============================================"
echo "All training complete at $(date)"

# Verify all 9 prediction files exist
EXPECTED=(
    predictions_reg_3d.csv predictions_reg_10d.csv predictions_reg_30d.csv
    predictions_bin_3d.csv predictions_bin_10d.csv predictions_bin_30d.csv
    predictions_buck_3d.csv predictions_buck_10d.csv predictions_buck_30d.csv
)
MISSING=0
for f in "${EXPECTED[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "MISSING: $f"
        MISSING=$((MISSING + 1))
    fi
done

if [[ $MISSING -gt 0 ]]; then
    echo "WARNING: $MISSING of 9 prediction files missing!"
    echo "Check logs for errors"
    exit 1
fi

echo "All 9 prediction files present. Merging..."
python -m stockformer.merge_predictions --input-dir . --output merged_predictions.csv
echo "Done! merged_predictions.csv updated at $(date)"
