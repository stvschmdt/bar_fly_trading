#!/bin/bash
# Train all 9 cross_attention models: 3 label modes × 3 horizons
# Runs 3 at a time (one per label mode), waits, then next horizon batch
#
# Models: regression/binary/buckets × 3d/10d/30d = 9 total
# Output: model_checkpoint_ca_{mode}_{horizon}d.pt

set -e
cd ~/proj/bar_fly_trading

DATA_PATH="./all_data_*.csv"

mkdir -p logs

echo "============================================"
echo "CROSS-ATTENTION TRAINING: 9 models (3 batches of 3)"
echo "Started: $(date)"
echo "============================================"

for HORIZON in 3 10 30; do
    echo ""
    echo "────────────────────────────────────────────"
    echo "BATCH: horizon=${HORIZON}d  (regression + binary + buckets)"
    echo "Started: $(date)"
    echo "────────────────────────────────────────────"

    # Launch 3 in parallel
    python -u -m stockformer.main \
        --data-path "$DATA_PATH" \
        --train-end-date 2025-10-31 --infer-start-date 2025-11-01 \
        --batch-size 64 --epochs 10 --num-workers 0 \
        --model-type cross_attention \
        --market-path ./market_embeddings.csv --sector-path ./sector_embeddings.csv \
        --label-mode regression --horizon $HORIZON \
        > logs/train_ca_regression_${HORIZON}d.log 2>&1 &
    PID_REG=$!

    python -u -m stockformer.main \
        --data-path "$DATA_PATH" \
        --train-end-date 2025-10-31 --infer-start-date 2025-11-01 \
        --batch-size 64 --epochs 10 --num-workers 0 \
        --model-type cross_attention \
        --market-path ./market_embeddings.csv --sector-path ./sector_embeddings.csv \
        --label-mode binary --horizon $HORIZON \
        > logs/train_ca_binary_${HORIZON}d.log 2>&1 &
    PID_BIN=$!

    python -u -m stockformer.main \
        --data-path "$DATA_PATH" \
        --train-end-date 2025-10-31 --infer-start-date 2025-11-01 \
        --batch-size 64 --epochs 10 --num-workers 0 \
        --model-type cross_attention \
        --market-path ./market_embeddings.csv --sector-path ./sector_embeddings.csv \
        --label-mode buckets --horizon $HORIZON \
        > logs/train_ca_buckets_${HORIZON}d.log 2>&1 &
    PID_BUCK=$!

    echo "  PIDs: regression=$PID_REG  binary=$PID_BIN  buckets=$PID_BUCK"

    # Wait for all 3 to finish
    wait $PID_REG
    echo "  [$(date)] regression ${HORIZON}d done (exit $?)"
    wait $PID_BIN
    echo "  [$(date)] binary ${HORIZON}d done (exit $?)"
    wait $PID_BUCK
    echo "  [$(date)] buckets ${HORIZON}d done (exit $?)"

    echo "  Batch ${HORIZON}d complete at $(date)"
done

echo ""
echo "============================================"
echo "ALL 9 CROSS-ATTENTION MODELS COMPLETE"
echo "Finished: $(date)"
echo "============================================"
echo ""
echo "Model files:"
ls -lh model_checkpoint_ca_*.pt 2>/dev/null || ls -lh *.pt 2>/dev/null
