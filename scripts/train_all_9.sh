#!/bin/bash
# =============================================================================
# Train all 9 StockFormer models (3 horizons x 3 label modes)
# =============================================================================
#
# Launches 3 parallel groups (regression, binary, buckets), each running
# sequentially through 3d -> 10d -> 30d horizons via explicit calls.
#
# Usage:
#   ./scripts/train_all_9.sh
#
# Timeline: ~4-6 hours (3 parallel groups, reduced model size)

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
echo "  StockFormer — Training all 9 models (v2)"
echo "  Config: lookback=60, d_model=128, num_layers=3, dim_ff=256"
echo "  Sector ETF + SPY features, noisy label filtering"
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
# Launch 3 parallel groups
# =============================================================================

echo "Launching 3 parallel training groups..."
echo "  Group 1: regression (3d -> 10d -> 30d)"
echo "  Group 2: binary    (3d -> 10d -> 30d)"
echo "  Group 3: buckets   (3d -> 10d -> 30d)"
echo ""

# Shared flags for classification models
BINARY_FLAGS="--entropy-reg-weight 0.1 --binary-threshold 0.005 --min-return-threshold 0.0025"
BUCKET_FLAGS="--bucket-edges=-1,1 --entropy-reg-weight 0.1 --min-return-threshold 0.0025"

# Group 1: Regression — directional_mse (direction_weight=3.0 from config)
(
    run_one regression 3  directional_mse
    run_one regression 10 directional_mse
    run_one regression 30 directional_mse
) > "logs/train_regression_${LOGDATE}.log" 2>&1 &
REG_PID=$!
echo "  regression PID: $REG_PID"

# Group 2: Binary — focal loss, noise filtering, asymmetric threshold
(
    run_one binary 3  focal "$BINARY_FLAGS"
    run_one binary 10 focal "$BINARY_FLAGS"
    run_one binary 30 focal "$BINARY_FLAGS"
) > "logs/train_binary_${LOGDATE}.log" 2>&1 &
BIN_PID=$!
echo "  binary PID: $BIN_PID"

# Group 3: 3-class buckets — focal loss, noise filtering
(
    run_one buckets 3  focal "$BUCKET_FLAGS"
    run_one buckets 10 focal "$BUCKET_FLAGS"
    run_one buckets 30 focal "$BUCKET_FLAGS"
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
