#!/bin/bash
# =============================================================================
# Train all 9 StockFormer models (3 horizons x 3 label modes)
# =============================================================================
#
# v5: Cross-attention + CORAL + collapse recovery
#
# Key changes from v4:
#   - Cross-attention architecture: stock encoder + gated market context
#   - d_model=128, nhead=8, 4 stock layers, 2 market layers, dim_ff=512
#   - Binary: focal gamma=1.5 + label_smoothing=0.05 (was gamma=2.0)
#   - Buckets: CORAL ordinal loss (was soft_ordinal — collapsed 100%)
#   - Collapse recovery: LR/10 + entropy boost instead of halt
#   - patience=15 (was 10), max_epochs=80 (was 50)
#   - All models: same architecture (no more small classification models)
#
# Usage:
#   ./scripts/train_all_9.sh
#
# Timeline: ~5-8 hours (3 parallel groups on DGX GPU)

cd ~/proj/bar_fly_trading

# =============================================================================
# Configuration
# =============================================================================

DATA_PATH="${DATA_PATH:-./all_data_*.csv}"
TRAIN_END="2025-12-31"
INFER_START="2025-01-01"

# Output paths (inside stockformer/output/)
MODEL_DIR="stockformer/output/models"
LOG_DIR="stockformer/output/logs"
PRED_DIR="stockformer/output/predictions"

mkdir -p "$MODEL_DIR" "$LOG_DIR" "$PRED_DIR" logs

# =============================================================================
# Research hyperparameters (from RESEARCH_CONFIG)
# =============================================================================

# Shared architecture: cross-attention with market context
ARCH="--model-type cross_attention --d-model 128 --nhead 8 --num-layers 4 --market-layers 2 --dim-feedforward 512"

# Shared training params
TRAIN="--lr 3e-4 --weight-decay 0.02 --patience 15 --epochs 25 --warmup-epochs 1 --dropout 0.15 --layer-drop 0.1"

# Collapse recovery: reduce LR by 10x + double entropy reg instead of halting
COLLAPSE="--collapse-lr-reduction 0.1 --collapse-entropy-boost 2.0"

# 30d extra regularization (longer horizon = noisier target)
REG_30D="--dropout 0.2 --weight-decay 0.03"

# =============================================================================
# Header
# =============================================================================

echo "============================================================"
echo "  StockFormer v5 — Cross-Attention + Anti-Collapse"
echo "  Architecture: d=128, h=8, layers=4+2, ff=512"
echo "  Regression:   combined_regression (directional_weight=3.0)"
echo "  Binary:       focal gamma=1.5, label_smoothing=0.05"
echo "  Buckets:      CORAL ordinal (anti-collapse)"
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
        $ARCH $TRAIN $COLLAPSE \
        $extra_flags

    echo "[$(date +%H:%M:%S)] Finished $suffix"
}
export -f run_one
export DATA_PATH TRAIN_END INFER_START MODEL_DIR LOG_DIR PRED_DIR ARCH TRAIN COLLAPSE

# =============================================================================
# Per-mode loss settings
# =============================================================================

# Binary: focal gamma=1.5 + label smoothing + 2% threshold
BINARY_FLAGS="--entropy-reg-weight 0.3 --binary-threshold 0.02 --focal-gamma 1.5 --label-smoothing 0.05"

# Buckets: CORAL ordinal loss (anti-collapse by construction) + auto quantile edges
BUCKET_FLAGS="--bucket-edges auto --n-buckets 3 --entropy-reg-weight 0.3"

# Regression: combined_regression (logcosh + directional_mse)
REG_FLAGS="--direction-weight 3.0"

# =============================================================================
# Launch 3 groups in parallel (regression | binary | buckets)
# Each group trains its 3 horizons sequentially
# =============================================================================

echo "Training 9 models in 3 parallel groups..."
echo "  Regression: 3d -> 10d -> 30d  [cross-attn, combined_regression]"
echo "  Binary:     3d -> 10d -> 30d  [cross-attn, focal gamma=1.5]"
echo "  Buckets:    3d -> 10d -> 30d  [cross-attn, CORAL ordinal]"
echo ""

LOG="logs/train_all_${LOGDATE}.log"

# Group 1: Regression (3 horizons sequentially)
(
    run_one regression 3  combined_regression "$REG_FLAGS"
    run_one regression 10 combined_regression "$REG_FLAGS"
    run_one regression 30 combined_regression "$REG_FLAGS $REG_30D"
) 2>&1 | tee "logs/train_regression_${LOGDATE}.log" &
PID_REG=$!

# Group 2: Binary (3 horizons sequentially)
(
    run_one binary 3  focal "$BINARY_FLAGS"
    run_one binary 10 focal "$BINARY_FLAGS"
    run_one binary 30 focal "$BINARY_FLAGS $REG_30D"
) 2>&1 | tee "logs/train_binary_${LOGDATE}.log" &
PID_BIN=$!

# Group 3: Buckets (3 horizons sequentially)
(
    run_one buckets 3  coral "$BUCKET_FLAGS"
    run_one buckets 10 coral "$BUCKET_FLAGS"
    run_one buckets 30 coral "$BUCKET_FLAGS $REG_30D"
) 2>&1 | tee "logs/train_buckets_${LOGDATE}.log" &
PID_BUCK=$!

echo "Launched 3 parallel groups: regression=$PID_REG, binary=$PID_BIN, buckets=$PID_BUCK"
echo "Waiting for all groups to finish..."
wait $PID_REG $PID_BIN $PID_BUCK
FAILURES=0

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
