#!/bin/bash
# =============================================================================
# Run 10 RL experiments in batches of 3
# Each experiment uses a unique run_name for TensorBoard comparison
# =============================================================================

cd ~/proj/bar_fly_trading
LOGDATE=$(date +%Y%m%d_%H%M)
mkdir -p logs/rl_experiments

echo "============================================================"
echo "  RL Experiment Sweep — 10 experiments, 2M steps each"
echo "  Started: $(date)"
echo "============================================================"
echo ""

EXP_DIR="bargyms/configs/experiments"

# Batch 1: PPO network size + sharpe reward (3 parallel)
echo "[Batch 1/4] Launching exp01-03..."
nohup python bargyms/train.py --config $EXP_DIR/exp01_ppo_small_net.yaml \
    > logs/rl_experiments/exp01_${LOGDATE}.log 2>&1 &
P1=$!
nohup python bargyms/train.py --config $EXP_DIR/exp02_ppo_large_net.yaml \
    > logs/rl_experiments/exp02_${LOGDATE}.log 2>&1 &
P2=$!
nohup python bargyms/train.py --config $EXP_DIR/exp03_ppo_sharpe_reward.yaml \
    > logs/rl_experiments/exp03_${LOGDATE}.log 2>&1 &
P3=$!
echo "  PIDs: $P1 $P2 $P3"
wait $P1 $P2 $P3
echo "[Batch 1] Complete: $(date)"

# Batch 2: PPO risk + short episode + minimal features (3 parallel)
echo "[Batch 2/4] Launching exp04-06..."
nohup python bargyms/train.py --config $EXP_DIR/exp04_ppo_risk_adjusted.yaml \
    > logs/rl_experiments/exp04_${LOGDATE}.log 2>&1 &
P4=$!
nohup python bargyms/train.py --config $EXP_DIR/exp05_ppo_short_episode.yaml \
    > logs/rl_experiments/exp05_${LOGDATE}.log 2>&1 &
P5=$!
nohup python bargyms/train.py --config $EXP_DIR/exp06_ppo_minimal_features.yaml \
    > logs/rl_experiments/exp06_${LOGDATE}.log 2>&1 &
P6=$!
echo "  PIDs: $P4 $P5 $P6"
wait $P4 $P5 $P6
echo "[Batch 2] Complete: $(date)"

# Batch 3: SAC experiments (2 parallel — SAC is heavier)
echo "[Batch 3/4] Launching exp07-08..."
nohup python bargyms/train.py --config $EXP_DIR/exp07_sac_large_net.yaml \
    > logs/rl_experiments/exp07_${LOGDATE}.log 2>&1 &
P7=$!
nohup python bargyms/train.py --config $EXP_DIR/exp08_sac_pnl_reward.yaml \
    > logs/rl_experiments/exp08_${LOGDATE}.log 2>&1 &
P8=$!
echo "  PIDs: $P7 $P8"
wait $P7 $P8
echo "[Batch 3] Complete: $(date)"

# Batch 4: Trade completion + long lookback (2 parallel)
echo "[Batch 4/4] Launching exp09-10..."
nohup python bargyms/train.py --config $EXP_DIR/exp09_ppo_trade_completion.yaml \
    > logs/rl_experiments/exp09_${LOGDATE}.log 2>&1 &
P9=$!
nohup python bargyms/train.py --config $EXP_DIR/exp10_ppo_long_lookback.yaml \
    > logs/rl_experiments/exp10_${LOGDATE}.log 2>&1 &
P10=$!
echo "  PIDs: $P9 $P10"
wait $P9 $P10
echo "[Batch 4] Complete: $(date)"

echo ""
echo "============================================================"
echo "  All 10 experiments complete: $(date)"
echo "  TensorBoard: tensorboard --logdir ./rl_logs"
echo "============================================================"
