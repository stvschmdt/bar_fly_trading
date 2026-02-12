#!/bin/bash
# =============================================================================
# Train all 9 StockFormer models (3 horizons x 3 label modes)
# =============================================================================
#
# v4: Aggressive anti-collapse measures after v3 models still collapsed.
#
# Key changes from v3:
#   - Binary: focal loss (was label_smoothing) — down-weights easy examples
#   - Binary: entropy_reg=0.5 (was 0.3) — stronger diversity pressure
#   - Binary: threshold=0.015 / 1.5% (was 0.0) — cleaner label separation
#   - Regression: combined_regression loss (was directional_mse) — logcosh + DMSE
#   - Training auto-halts if dominant class > 90% for 3 consecutive epochs
#   - Inference prints full eval report (confusion matrix, F1, ROC-AUC)
#
# Unchanged from v3:
#   - Classification models: d_model=64, 2 layers (smaller to prevent memorization)
#   - Buckets: soft_ordinal loss + auto quantile edges
#   - 30d models: extra regularization (dropout=0.3, weight_decay=0.03)
#
# Usage:
#   ./scripts/train_all_9.sh
#
# Timeline: ~3-5 hours (3 parallel groups, classification models are smaller)

cd ~/proj/bar_fly_trading

# =============================================================================
# Configuration
# =============================================================================

DATA_PATH="/home/stvschmdt/data/all_data_*.csv"
TRAIN_END="2024-12-31"
INFER_START="2025-01-01"

# Output paths (inside stockformer/output/)
MODEL_DIR="stockformer/output/models"
LOG_DIR="stockformer/output/logs"
PRED_DIR="stockformer/output/predictions"

mkdir -p "$MODEL_DIR" "$LOG_DIR" "$PRED_DIR" logs

# =============================================================================
# Header
# =============================================================================

echo "============================================================"
echo "  StockFormer — Training all 9 models (v4)"
echo "  Regression:      d_model=128, layers=3, combined_regression"
echo "  Classification:  d_model=64,  layers=2, dim_ff=128"
echo "  Binary:          focal + entropy_reg=0.5, threshold=1.5%"
echo "  Buckets:         soft_ordinal + auto quantile edges"
echo "  Temporal split: train <= $TRAIN_END, infer >= $INFER_START"
echo "  Started: $(date)"
echo "============================================================"
echo ""

LOGDATE=$(date +%Y%m%d)

# Helper: run a single train+infer for one (label_mode, horizon) combo
run_one() {
    local label_mode="$1"
    local horizon="$2"
    local loss_name="$3"
    local extra_flags="$4"
    local tag="${label_mode}_${horizon}d"

    # Map label_mode to short tag for file names
    case "$label_mode" in
        regression) short="reg" ;;
        binary)     short="bin" ;;
        buckets)    short="buck" ;;
    esac
    local suffix="${short}_${horizon}d"

    echo "[$(date +%H:%M:%S)] Starting $suffix ($label_mode, ${horizon}d, loss=$loss_name)"

    python -u -m stockformer.main \
        --data-path "$DATA_PATH" \
        --train-end-date "$TRAIN_END" --infer-start-date "$INFER_START" \
        --batch-size 256 --plot --save-config \
        --loss-name "$loss_name" \
        --horizon "$horizon" --label-mode "$label_mode" \
        --model-out "$MODEL_DIR/model_${suffix}.pt" \
        --log-path "$LOG_DIR/training_log_${suffix}.csv" \
        --output-csv "$PRED_DIR/pred_${suffix}.csv" \
        $extra_flags

    echo "[$(date +%H:%M:%S)] Finished $suffix"
}
export -f run_one
export DATA_PATH TRAIN_END INFER_START MODEL_DIR LOG_DIR PRED_DIR

# =============================================================================
# Per-model hyperparameters
# =============================================================================

# Smaller model for classification (prevents memorization -> collapse)
SMALL_MODEL="--d-model 64 --num-layers 2 --dim-feedforward 128"

# 30d extra regularization (longer horizon = noisier target)
REG_30D="--dropout 0.3 --weight-decay 0.03"

# Binary classification: focal loss + strong entropy reg + 1.5% threshold for clean separation
BINARY_BASE="--entropy-reg-weight 0.5 --binary-threshold 0.015 --min-return-threshold 0.0 $SMALL_MODEL"

# Bucket classification: auto quantile edges (3 balanced classes) + strong entropy reg
BUCKET_BASE="--bucket-edges auto --n-buckets 3 --entropy-reg-weight 0.5 --min-return-threshold 0.0 $SMALL_MODEL"

# =============================================================================
# Launch 3 parallel groups
# =============================================================================

echo "Launching 3 parallel training groups..."
echo "  Group 1: regression (3d -> 10d -> 30d)  [128d, 3 layers, combined_regression]"
echo "  Group 2: binary    (3d -> 10d -> 30d)  [64d, 2 layers, focal]"
echo "  Group 3: buckets   (3d -> 10d -> 30d)  [64d, 2 layers, soft_ordinal]"
echo ""

# Group 1: Regression — combined_regression (logcosh + directional_mse, direction_weight=3.0)
# 3d and 10d: default model size (working well)
# 30d: smaller model + extra regularization (was overfitting)
(
    run_one regression 3  combined_regression ""
    run_one regression 10 combined_regression ""
    run_one regression 30 combined_regression "$SMALL_MODEL $REG_30D"
) > "logs/train_regression_${LOGDATE}.log" 2>&1 &
REG_PID=$!
echo "  regression PID: $REG_PID"

# Group 2: Binary — focal loss (down-weights easy examples, prevents class collapse)
(
    run_one binary 3  focal "$BINARY_BASE"
    run_one binary 10 focal "$BINARY_BASE"
    run_one binary 30 focal "$BINARY_BASE $REG_30D"
) > "logs/train_binary_${LOGDATE}.log" 2>&1 &
BIN_PID=$!
echo "  binary PID: $BIN_PID"

# Group 3: 3-class buckets — soft_ordinal loss, auto quantile edges
(
    run_one buckets 3  soft_ordinal "$BUCKET_BASE"
    run_one buckets 10 soft_ordinal "$BUCKET_BASE"
    run_one buckets 30 soft_ordinal "$BUCKET_BASE $REG_30D"
) > "logs/train_buckets_${LOGDATE}.log" 2>&1 &
BUCK_PID=$!
echo "  buckets PID: $BUCK_PID"

echo ""
echo "Monitor progress:"
echo "  tail -f logs/train_*_${LOGDATE}.log"
echo ""
echo "Waiting for all groups to finish..."

# Wait for each group
FAILURES=0
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
for suffix in reg_3d reg_10d reg_30d bin_3d bin_10d bin_30d buck_3d buck_10d buck_30d; do
    f="$PRED_DIR/pred_${suffix}.csv"
    if [[ -f "$f" ]]; then
        LINES=$(wc -l < "$f")
        echo "  OK: $f ($LINES lines)"
    else
        echo "  MISSING: $f"
    fi
done

# =============================================================================
# Merge predictions into merged_predictions.csv
# =============================================================================

echo ""
echo "Merging predictions..."
python -m stockformer.merge_predictions \
    --input-dir "$PRED_DIR" \
    --output merged_predictions.csv

if [[ -f "merged_predictions.csv" ]]; then
    LINES=$(wc -l < "merged_predictions.csv")
    COLS=$(head -1 "merged_predictions.csv" | tr ',' '\n' | wc -l)
    echo "  Merged: merged_predictions.csv ($LINES rows, $COLS columns)"
fi

echo ""
echo "Done!"
