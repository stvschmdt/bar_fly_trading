#!/usr/bin/env bash
# =============================================================================
# Train all 9 StockFormer models (3 label_modes x 3 horizons) in parallel
#
# Runs 3 parallel groups (one per label_mode), each training 3 horizons sequentially.
# This keeps GPU memory manageable while maximizing throughput.
#
# Usage:
#   ./scripts/train_all_9.sh [--epochs N] [--data-path PATH]
#
# Defaults: 30 epochs, data from ./all_data_*.csv
# =============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."

# Defaults
EPOCHS=30
DATA_PATH="./all_data_*.csv"
BATCH_SIZE=64
TRAIN_END="2025-10-31"
INFER_START="2025-11-01"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs) EPOCHS="$2"; shift 2 ;;
        --data-path) DATA_PATH="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p logs

echo "============================================================"
echo "  StockFormer v2 - Training all 9 models"
echo "  Epochs: $EPOCHS | Batch: $BATCH_SIZE | Lookback: 20"
echo "  Optimizer: adamw | LR scheduler: cosine + warmup"
echo "  Early stopping: patience=7 | Gradient clipping: 1.0"
echo "  Temporal split (train <= $TRAIN_END, val/infer >= $INFER_START)"
echo "  Features: 46 columns (technicals + signals + fundamentals + macro)"
echo "  Model: PositionalEncoding + AttentionPooling + MLP heads"
echo "  Started: $(date)"
echo "============================================================"

# Train one model: train_one <horizon> <label_mode> <tag>
train_one() {
    local horizon="$1"
    local label_mode="$2"
    local tag="$3"

    echo "[$(date +%H:%M:%S)] Starting ${tag} (horizon=${horizon}, label_mode=${label_mode}, epochs=${EPOCHS})"

    python -u -m stockformer.main \
        --data-path "$DATA_PATH" \
        --train-end-date "$TRAIN_END" \
        --infer-start-date "$INFER_START" \
        --batch-size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --num-workers 0 \
        --horizon "$horizon" \
        --label-mode "$label_mode" \
        --model-out "model_checkpoint.pt" \
        --log-path "training_log.csv" \
        --output-csv "predictions.csv" \
        --save-config \
        --plot

    echo "[$(date +%H:%M:%S)] Finished ${tag}"
}

# Group 1: regression (3d, 10d, 30d) — sequential within group
run_regression() {
    train_one 3  regression reg_3d
    train_one 10 regression reg_10d
    train_one 30 regression reg_30d
}

# Group 2: binary (3d, 10d, 30d) — sequential within group
run_binary() {
    train_one 3  binary bin_3d
    train_one 10 binary bin_10d
    train_one 30 binary bin_30d
}

# Group 3: buckets (3d, 10d, 30d) — sequential within group
run_buckets() {
    train_one 3  buckets buck_3d
    train_one 10 buckets buck_10d
    train_one 30 buckets buck_30d
}

# Run all 3 groups in parallel
echo ""
echo "Launching 3 parallel training groups..."
echo "  Group 1: regression (3d → 10d → 30d)"
echo "  Group 2: binary    (3d → 10d → 30d)"
echo "  Group 3: buckets   (3d → 10d → 30d)"
echo ""

run_regression > logs/train_regression_$(date +%Y%m%d).log 2>&1 &
PID_REG=$!
echo "  regression PID: $PID_REG"

run_binary > logs/train_binary_$(date +%Y%m%d).log 2>&1 &
PID_BIN=$!
echo "  binary PID: $PID_BIN"

run_buckets > logs/train_buckets_$(date +%Y%m%d).log 2>&1 &
PID_BUCK=$!
echo "  buckets PID: $PID_BUCK"

echo ""
echo "Waiting for all groups to finish..."

# Wait for all groups and track failures
FAILED=0

wait $PID_REG || { echo "ERROR: regression group failed"; FAILED=$((FAILED + 1)); }
echo "[$(date +%H:%M:%S)] regression group done"

wait $PID_BIN || { echo "ERROR: binary group failed"; FAILED=$((FAILED + 1)); }
echo "[$(date +%H:%M:%S)] binary group done"

wait $PID_BUCK || { echo "ERROR: buckets group failed"; FAILED=$((FAILED + 1)); }
echo "[$(date +%H:%M:%S)] buckets group done"

echo ""
echo "============================================================"
echo "  All training complete. Failures: $FAILED"
echo "  Finished: $(date)"
echo "============================================================"

# Verify all 9 prediction files exist
echo ""
echo "Checking prediction files..."
MISSING=0
for tag in reg_3d reg_10d reg_30d bin_3d bin_10d bin_30d buck_3d buck_10d buck_30d; do
    if [[ -f "predictions_${tag}.csv" ]]; then
        rows=$(wc -l < "predictions_${tag}.csv")
        echo "  OK: predictions_${tag}.csv ($rows lines)"
    else
        echo "  MISSING: predictions_${tag}.csv"
        MISSING=$((MISSING + 1))
    fi
done

if [[ $MISSING -eq 0 ]]; then
    echo ""
    echo "All 9 prediction files present. Running merge..."
    python -m stockformer.merge_predictions --input-dir . --output merged_predictions.csv
    echo "Merged predictions saved to merged_predictions.csv"
else
    echo ""
    echo "WARNING: $MISSING prediction files missing. Skipping merge."
fi

echo ""
echo "Done!"
