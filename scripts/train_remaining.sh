#!/bin/bash
# Train remaining/extended models sequentially
# buck_10d is already running separately
# This script runs: buck_30d, reg_30d (30 epochs), bin_30d (30 epochs)
set -e
cd ~/proj/bar_fly_trading

echo "Starting remaining training at $(date)"
echo "============================================"

DATA_PATH="./all_data_*.csv"
TRAIN_END="2025-10-31"
INFER_START="2025-11-01"
BATCH_SIZE=64
WORKERS=0

# 1. buck_30d (15 epochs) — missing
echo ""
echo "=== buck_30d (15 epochs) ==="
python -u -m stockformer.main \
    --data-path "$DATA_PATH" \
    --train-end-date $TRAIN_END \
    --infer-start-date $INFER_START \
    --batch-size $BATCH_SIZE \
    --epochs 15 \
    --num-workers $WORKERS \
    --horizon 30 \
    --label-mode buckets \
    --model-out model_checkpoint_buck_30d.pt \
    --log-path training_log_buck_30d.csv \
    --output-csv predictions_buck_30d.csv \
    --save-config

# 2. reg_30d (30 epochs) — was still improving at 15
echo ""
echo "=== reg_30d extended (30 epochs) ==="
python -u -m stockformer.main \
    --data-path "$DATA_PATH" \
    --train-end-date $TRAIN_END \
    --infer-start-date $INFER_START \
    --batch-size $BATCH_SIZE \
    --epochs 30 \
    --num-workers $WORKERS \
    --horizon 30 \
    --label-mode regression \
    --model-out model_checkpoint_reg_30d.pt \
    --log-path training_log_reg_30d.csv \
    --output-csv predictions_reg_30d.csv \
    --save-config

# 3. bin_30d (30 epochs) — was still improving at 15
echo ""
echo "=== bin_30d extended (30 epochs) ==="
python -u -m stockformer.main \
    --data-path "$DATA_PATH" \
    --train-end-date $TRAIN_END \
    --infer-start-date $INFER_START \
    --batch-size $BATCH_SIZE \
    --epochs 30 \
    --num-workers $WORKERS \
    --horizon 30 \
    --label-mode binary \
    --model-out model_checkpoint_bin_30d.pt \
    --log-path training_log_bin_30d.csv \
    --output-csv predictions_bin_30d.csv \
    --save-config

echo ""
echo "============================================"
echo "All remaining training complete at $(date)"
echo "Now merging all predictions..."

python -m stockformer.merge_predictions \
    --input-dir . \
    --output merged_predictions.csv

echo "Done! merged_predictions.csv updated"
