"""
Evaluate a trained RL model using the same metrics as the strategies framework.

Runs the model through rolling episodes and feeds trade_log to compute_stats()
for directly comparable metrics (win rate, Sharpe, per-symbol P&L).

Usage:
    python bargyms/evaluate.py --model rl_models/best/best_model.zip \
        --config bargyms/configs/sac_continuous.yaml
"""

import argparse
import os
import sys
import yaml

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bargyms.envs.trading_env import TradingEnv, RLDataLoader
from strategies.backtest_stats import compute_stats, print_stats

ALGO_MAP = {"PPO": PPO, "SAC": SAC, "TD3": TD3}


def evaluate_model(model, data_loader, config, n_episodes=100, deterministic=True):
    """
    Run the model through multiple episodes and collect trade logs.

    Returns:
        list[dict]: All trades across episodes (compatible with compute_stats)
    """
    env = TradingEnv(data_loader, config, mode="eval")
    all_trades = []
    episode_returns = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Collect trades from this episode
        trade_log = info.get("trade_log", [])

        # Convert step-based dates to actual dates for compute_stats
        dates = data_loader.get_dates(info["symbol"])
        for trade in trade_log:
            start_idx = env._start_idx  # Episode start in data
            entry_data_idx = start_idx + trade["entry_step"]
            exit_data_idx = start_idx + trade["exit_step"]
            if entry_data_idx < len(dates) and exit_data_idx < len(dates):
                trade["entry_date"] = str(dates[entry_data_idx])[:10]
                trade["exit_date"] = str(dates[exit_data_idx])[:10]
            else:
                trade["entry_date"] = f"step_{trade['entry_step']}"
                trade["exit_date"] = f"step_{trade['exit_step']}"

        all_trades.extend(trade_log)
        episode_returns.append(info.get("episode_return_pct", 0))

    return all_trades, episode_returns


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL trading model")
    parser.add_argument("--model", required=True, help="Path to trained model (.zip)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--n-episodes", type=int, default=100, help="Number of eval episodes")
    parser.add_argument("--vecnormalize", help="Path to VecNormalize stats (.pkl)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load data
    print("Loading data...")
    data_loader = RLDataLoader(config)
    data_loader.load()

    # Load model
    algo_name = config["algorithm"]["name"].upper()
    algo_cls = ALGO_MAP[algo_name]
    print(f"Loading {algo_name} model from {args.model}")
    model = algo_cls.load(args.model)

    # Evaluate
    print(f"\nRunning {args.n_episodes} evaluation episodes...")
    all_trades, episode_returns = evaluate_model(
        model, data_loader, config, n_episodes=args.n_episodes
    )

    # Compute stats using strategies framework
    initial_cash = config["environment"].get("initial_cash", 100000)
    if all_trades:
        stats = compute_stats(all_trades, initial_cash)
        final_value = initial_cash + stats["total_pnl"]
        print_stats(stats, initial_cash, final_value)
    else:
        print("No trades were executed across all episodes.")

    # Episode-level summary
    if episode_returns:
        returns = np.array(episode_returns)
        print(f"\nEpisode Summary ({len(returns)} episodes):")
        print(f"  Mean return: {returns.mean():.2f}%")
        print(f"  Std return:  {returns.std():.2f}%")
        print(f"  Min return:  {returns.min():.2f}%")
        print(f"  Max return:  {returns.max():.2f}%")
        print(f"  Win rate:    {(returns > 0).mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
