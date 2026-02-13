#!/bin/bash
# Unified training script for StockFormer.
#
# Usage:
#   ./run_training.sh regression        # Train 3 regression models (3d/10d/30d)
#   ./run_training.sh binary            # Train 3 binary models
#   ./run_training.sh buckets           # Train 3 bucket models
#   ./run_training.sh all               # Train all 9 models sequentially
#
# Hyperparameters are set via defaults in config.py.
# Override any param via environment variables:
#   LR=1e-3 EPOCHS=30 DROPOUT=0.1 ./run_training.sh regression
#
# Config mapping (label_mode → default loss):
#   regression → combined_regression
#   binary     → symmetric_ce
#   buckets    → soft_ordinal (with auto quantile edges)

set -euo pipefail

PYTHON="${PYTHON:-python}"
cd "$(dirname "$0")"

DATA_PATH="${DATA_PATH:-./all_data_*.csv}"
OUTPUT_BASE="${OUTPUT_BASE:-stockformer/output}"

# Hyperparameter defaults (override via env vars)
LR="${LR:-5e-4}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-256}"
DROPOUT="${DROPOUT:-0.2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LAYER_DROP="${LAYER_DROP:-0.1}"
PATIENCE="${PATIENCE:-10}"
WARMUP="${WARMUP:-5}"
ENTROPY_REG="${ENTROPY_REG:-0.3}"
TRAIN_END="${TRAIN_END:-2025-06-30}"
INFER_START="${INFER_START:-2025-07-01}"

# Default loss per label mode
declare -A LOSS_MAP=(
    [regression]=combined_regression
    [binary]=symmetric_ce
    [buckets]=soft_ordinal
)

# Allow override
LOSS_OVERRIDE="${LOSS_OVERRIDE:-}"

run_label_mode() {
    local LMODE="$1"
    local LOSS="${LOSS_OVERRIDE:-${LOSS_MAP[$LMODE]}}"

    # Bucket-specific args
    local BUCKET_ARGS=""
    if [ "$LMODE" = "buckets" ]; then
        BUCKET_ARGS="--bucket-edges auto"
    fi

    # Entropy reg only for classification
    local EREG_ARGS=""
    if [ "$LMODE" != "regression" ]; then
        EREG_ARGS="--entropy-reg-weight $ENTROPY_REG"
    fi

    for HORIZON in 3 10 30; do
        case "$LMODE" in
            regression) TAG="reg_${HORIZON}d" ;;
            binary)     TAG="bin_${HORIZON}d" ;;
            buckets)    TAG="buck_${HORIZON}d" ;;
        esac

        MODEL="${OUTPUT_BASE}/models/model_${TAG}.pt"
        LOG="${OUTPUT_BASE}/logs/training_log_${TAG}.csv"
        OUTPUT="${OUTPUT_BASE}/predictions/pred_${TAG}.csv"

        mkdir -p "$(dirname "$MODEL")" "$(dirname "$LOG")" "$(dirname "$OUTPUT")"

        echo ""
        echo "=== ${TAG} | loss=${LOSS} | $(date) ==="
        $PYTHON -u -m stockformer.main \
            --data-path "$DATA_PATH" \
            --lookback 20 \
            --batch-size "$BATCH" \
            --lr "$LR" \
            --dropout "$DROPOUT" \
            --weight-decay "$WEIGHT_DECAY" \
            --layer-drop "$LAYER_DROP" \
            --patience "$PATIENCE" \
            --warmup-epochs "$WARMUP" \
            --epochs "$EPOCHS" \
            --infer-start-date "$INFER_START" \
            --train-end-date "$TRAIN_END" \
            --horizon "$HORIZON" \
            --label-mode "$LMODE" \
            --loss-name "$LOSS" \
            $EREG_ARGS \
            $BUCKET_ARGS \
            --model-out "$MODEL" \
            --log-path "$LOG" \
            --output-csv "$OUTPUT" \
            --save-config --plot
    done
}

# Main
MODE="${1:-all}"

echo "========================================"
echo "StockFormer Training — ${MODE} — $(date)"
echo "lr=${LR} epochs=${EPOCHS} dropout=${DROPOUT} wd=${WEIGHT_DECAY}"
echo "========================================"

case "$MODE" in
    regression|binary|buckets)
        run_label_mode "$MODE"
        ;;
    all)
        for M in regression binary buckets; do
            run_label_mode "$M"
        done
        ;;
    *)
        echo "Usage: $0 {regression|binary|buckets|all}"
        echo ""
        echo "Environment variable overrides:"
        echo "  LR=1e-3 EPOCHS=30 LOSS_OVERRIDE=mse $0 regression"
        exit 1
        ;;
esac

echo ""
echo "========================================"
echo "COMPLETE — ${MODE} — $(date)"
echo "========================================"
