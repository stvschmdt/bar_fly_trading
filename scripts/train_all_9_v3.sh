#!/bin/bash
# =============================================================================
# Train all 9 StockFormer models (v3 - anti-collapse)
# =============================================================================
#
# Changes from v2:
#   - LR: 1e-3 -> 5e-4 (slower convergence, less majority-class snap)
#   - Loss: MSE/CE -> HuberLoss (regression), FocalLoss (classification)
#   - Class weights: auto (inverse-frequency from training data)
#   - Label smoothing: 0.1 (prevents overconfident single-class predictions)
#   - Focal gamma: 2.0 (down-weights easy majority-class examples)
#   - Entropy regularization: 0.1 (penalizes collapsed output distribution)
#   - Dropout: 0.1 -> 0.15
#   - Batch size: 64 -> 128
#   - Added temperature scaling in model (learnable logit calibration)
#   - Added LayerNorm before output heads
#
# Usage:
#   ./scripts/train_all_9_v3.sh
#
# Timeline: ~8-12 hours (3 parallel training groups x 3 horizons each)

set -e

cd ~/proj/bar_fly_trading

# =============================================================================
# Configuration
# =============================================================================

DATA_PATH="./all_data_*.csv"
TRAIN_END="2025-10-31"
INFER_START="2025-11-01"

# Training hyperparameters (v3 anti-collapse)
EPOCHS=30
BATCH_SIZE=128
LR=5e-4
DROPOUT=0.15

# Loss function settings
FOCAL_GAMMA=2.0
LABEL_SMOOTHING=0.1
CLASS_WEIGHTS="auto"
ENTROPY_WEIGHT=0.1

# Output paths
MODEL_OUT="model_checkpoint.pt"
LOG_PATH="training_log.csv"
OUTPUT_CSV="predictions.csv"

WORKERS=0

mkdir -p logs

# =============================================================================
# Header
# =============================================================================

echo "============================================================"
echo "  StockFormer v3 - Training all 9 models (anti-collapse)"
echo "  Epochs: $EPOCHS | Batch: $BATCH_SIZE | LR: $LR"
echo "  Dropout: $DROPOUT | Optimizer: adamw"
echo "  Loss: FocalLoss(gamma=$FOCAL_GAMMA) + ClassWeights($CLASS_WEIGHTS)"
echo "  Label smoothing: $LABEL_SMOOTHING | Entropy reg: $ENTROPY_WEIGHT"
echo "  Temporal split (train <= $TRAIN_END, val/infer >= $INFER_START)"
echo "  Model: PositionalEncoding + LayerNorm + TempScaling + MLP heads"
echo "  Started: $(date)"
echo "============================================================"
echo ""

# =============================================================================
# Common args
# =============================================================================

# Note: DATA_PATH glob must be quoted in all python calls below to prevent shell expansion

# =============================================================================
# Launch 3 parallel training groups
# =============================================================================

echo "Launching 3 parallel training groups..."
echo "  Group 1: regression (3d -> 10d -> 30d)"
echo "  Group 2: binary    (3d -> 10d -> 30d)"
echo "  Group 3: buckets   (3d -> 10d -> 30d)"
echo ""

LOGDATE=$(date +%Y%m%d)
FAILURES=0

# Group 1: Regression
nohup python -u -m stockformer.main \
    --data-path "$DATA_PATH" \
    --train-end-date "$TRAIN_END" \
    --infer-start-date "$INFER_START" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --dropout "$DROPOUT" \
    --num-workers "$WORKERS" \
    --model-out "$MODEL_OUT" \
    --log-path "$LOG_PATH" \
    --output-csv "$OUTPUT_CSV" \
    --focal-gamma "$FOCAL_GAMMA" \
    --label-smoothing "$LABEL_SMOOTHING" \
    --class-weights "$CLASS_WEIGHTS" \
    --entropy-weight "$ENTROPY_WEIGHT" \
    --label-mode regression \
    > "logs/train_regression_v3_${LOGDATE}.log" 2>&1 &
REG_PID=$!
echo "  regression PID: $REG_PID"

# Group 2: Binary
nohup python -u -m stockformer.main \
    --data-path "$DATA_PATH" \
    --train-end-date "$TRAIN_END" \
    --infer-start-date "$INFER_START" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --dropout "$DROPOUT" \
    --num-workers "$WORKERS" \
    --model-out "$MODEL_OUT" \
    --log-path "$LOG_PATH" \
    --output-csv "$OUTPUT_CSV" \
    --focal-gamma "$FOCAL_GAMMA" \
    --label-smoothing "$LABEL_SMOOTHING" \
    --class-weights "$CLASS_WEIGHTS" \
    --entropy-weight "$ENTROPY_WEIGHT" \
    --label-mode binary \
    > "logs/train_binary_v3_${LOGDATE}.log" 2>&1 &
BIN_PID=$!
echo "  binary PID: $BIN_PID"

# Group 3: Buckets
nohup python -u -m stockformer.main \
    --data-path "$DATA_PATH" \
    --train-end-date "$TRAIN_END" \
    --infer-start-date "$INFER_START" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --dropout "$DROPOUT" \
    --num-workers "$WORKERS" \
    --model-out "$MODEL_OUT" \
    --log-path "$LOG_PATH" \
    --output-csv "$OUTPUT_CSV" \
    --focal-gamma "$FOCAL_GAMMA" \
    --label-smoothing "$LABEL_SMOOTHING" \
    --class-weights "$CLASS_WEIGHTS" \
    --entropy-weight "$ENTROPY_WEIGHT" \
    --label-mode buckets \
    > "logs/train_buckets_v3_${LOGDATE}.log" 2>&1 &
BUCK_PID=$!
echo "  buckets PID: $BUCK_PID"

echo ""
echo "Waiting for all groups to finish..."
echo "  Monitor: tail -f logs/train_*_v3_${LOGDATE}.log"
echo ""

# Wait for each group
for PID_NAME in "regression:$REG_PID" "binary:$BIN_PID" "buckets:$BUCK_PID"; do
    NAME="${PID_NAME%%:*}"
    PID="${PID_NAME##*:}"
    if wait $PID; then
        echo "[$(date +%H:%M:%S)] $NAME group done"
    else
        echo "[$(date +%H:%M:%S)] $NAME group FAILED (exit code: $?)"
        FAILURES=$((FAILURES + 1))
    fi
done

echo ""
echo "============================================================"
echo "  All training complete. Failures: $FAILURES"
echo "  Finished: $(date)"
echo "============================================================"
echo ""

# =============================================================================
# Check prediction files
# =============================================================================

echo "Checking prediction files..."
EXPECTED_FILES=(
    "predictions_reg_3d.csv"
    "predictions_reg_10d.csv"
    "predictions_reg_30d.csv"
    "predictions_bin_3d.csv"
    "predictions_bin_10d.csv"
    "predictions_bin_30d.csv"
    "predictions_buck_3d.csv"
    "predictions_buck_10d.csv"
    "predictions_buck_30d.csv"
)

MISSING=0
for f in "${EXPECTED_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        LINES=$(wc -l < "$f")
        echo "  OK: $f ($LINES lines)"
    else
        echo "  MISSING: $f"
        MISSING=$((MISSING + 1))
    fi
done

echo ""

# =============================================================================
# Merge predictions
# =============================================================================

if [[ $MISSING -gt 0 ]]; then
    echo "WARNING: $MISSING prediction files missing. Skipping merge."
else
    echo "Merging all 9 prediction files..."
    python -m stockformer.merge_predictions \
        --input-dir . \
        --output merged_predictions_v3.csv

    if [[ -f "merged_predictions_v3.csv" ]]; then
        LINES=$(wc -l < "merged_predictions_v3.csv")
        COLS=$(head -1 "merged_predictions_v3.csv" | tr ',' '\n' | wc -l)
        echo "  Merged: merged_predictions_v3.csv ($LINES rows, $COLS columns)"
    fi
fi

echo ""
echo "Done!"
