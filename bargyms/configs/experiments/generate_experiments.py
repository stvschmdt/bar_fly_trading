"""Generate 10 RL experiment configs for hyperparameter sweep."""
import yaml
import os
import copy

BASE_DIR = os.path.dirname(__file__)

# Base config template
BASE = {
    "data": {
        "path": "all_data_*.csv",
        "symbols": None,
        "date_range": {"start": "2020-01-01", "end": "2025-12-31"},
        "val_split": 0.2,
    },
    "features": {
        "preset": "standard",
        "custom_columns": [],
        "exclude_columns": [],
        "market_context": True,
        "normalize": "zscore",
    },
    "environment": {
        "lookback_window": 20,
        "episode_length": 40,
        "action_space": "discrete",
        "initial_cash": 100000,
        "position_size": 0.10,
        "max_positions": 1,
        "allow_short": False,
        "transaction_cost_bps": 0,
        "exit_safety": {
            "enabled": True,
            "stop_loss_pct": -0.08,
            "take_profit_pct": 0.15,
            "trailing_stop_pct": None,
            "trailing_activation_pct": 0.0,
            "max_hold_days": 20,
        },
    },
    "reward": {
        "function": "pnl_pct_step",
        "params": {
            "illegal_action_penalty": -1.0,
            "holding_cost": 0.002,
            "win_bonus": 0.5,
            "loss_penalty": 0.3,
            "no_trade_penalty": 0.005,
            "large_loss_threshold": -0.03,
            "large_loss_multiplier": 3.0,
        },
    },
    "algorithm": {
        "name": "PPO",
        "policy": "MlpPolicy",
        "policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        "hyperparameters": {
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "n_steps": 2048,
            "batch_size": 256,
            "n_epochs": 10,
            "clip_range": 0.2,
            "gae_lambda": 0.95,
            "ent_coef": 0.01,
        },
    },
    "training": {
        "total_timesteps": 2000000,
        "eval_freq": 20000,
        "n_eval_episodes": 20,
        "log_interval": 10,
        "save_freq": 100000,
        "tensorboard_dir": "./rl_logs",
        "model_save_path": "./rl_models",
        "seed": 42,
    },
}

EXPERIMENTS = []

# --- Exp 1: PPO small network [64,64] ---
exp = copy.deepcopy(BASE)
exp["training"]["run_name"] = "exp01_ppo_small_net"
exp["algorithm"]["policy_kwargs"] = {"net_arch": {"pi": [64, 64], "vf": [64, 64]}}
EXPERIMENTS.append(("exp01_ppo_small_net.yaml", exp))

# --- Exp 2: PPO large network [512,512] ---
exp = copy.deepcopy(BASE)
exp["training"]["run_name"] = "exp02_ppo_large_net"
exp["algorithm"]["policy_kwargs"] = {"net_arch": {"pi": [512, 512], "vf": [512, 512]}}
EXPERIMENTS.append(("exp02_ppo_large_net.yaml", exp))

# --- Exp 3: PPO with sharpe_step reward ---
exp = copy.deepcopy(BASE)
exp["training"]["run_name"] = "exp03_ppo_sharpe_reward"
exp["reward"]["function"] = "sharpe_step"
exp["reward"]["params"]["sharpe_window"] = 20
EXPERIMENTS.append(("exp03_ppo_sharpe_reward.yaml", exp))

# --- Exp 4: PPO with risk_adjusted reward ---
exp = copy.deepcopy(BASE)
exp["training"]["run_name"] = "exp04_ppo_risk_adjusted"
exp["reward"]["function"] = "risk_adjusted"
exp["reward"]["params"]["drawdown_penalty"] = 2.0
EXPERIMENTS.append(("exp04_ppo_risk_adjusted.yaml", exp))

# --- Exp 5: PPO short episodes (20 steps) + bigger position ---
exp = copy.deepcopy(BASE)
exp["training"]["run_name"] = "exp05_ppo_short_episode"
exp["environment"]["episode_length"] = 20
exp["environment"]["position_size"] = 0.20
exp["environment"]["exit_safety"]["max_hold_days"] = 14
EXPERIMENTS.append(("exp05_ppo_short_episode.yaml", exp))

# --- Exp 6: PPO minimal features (15 cols — does less noise help?) ---
exp = copy.deepcopy(BASE)
exp["training"]["run_name"] = "exp06_ppo_minimal_features"
exp["features"]["preset"] = "minimal"
exp["features"]["market_context"] = False
EXPERIMENTS.append(("exp06_ppo_minimal_features.yaml", exp))

# --- Exp 7: SAC large network [512,512] ---
exp = copy.deepcopy(BASE)
exp["training"]["run_name"] = "exp07_sac_large_net"
exp["algorithm"]["name"] = "SAC"
exp["environment"]["action_space"] = "continuous"
exp["environment"]["dead_zone"] = 0.1
exp["algorithm"]["policy_kwargs"] = {"net_arch": {"pi": [512, 512], "qf": [512, 512]}}
exp["algorithm"]["hyperparameters"] = {
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "batch_size": 256,
    "buffer_size": 200000,
    "tau": 0.005,
    "ent_coef": "auto",
    "learning_starts": 10000,
}
exp["training"]["log_interval"] = 100
EXPERIMENTS.append(("exp07_sac_large_net.yaml", exp))

# --- Exp 8: SAC with pnl_pct_step reward (instead of sharpe) ---
exp = copy.deepcopy(BASE)
exp["training"]["run_name"] = "exp08_sac_pnl_reward"
exp["algorithm"]["name"] = "SAC"
exp["environment"]["action_space"] = "continuous"
exp["environment"]["dead_zone"] = 0.1
exp["reward"]["function"] = "pnl_pct_step"
exp["algorithm"]["policy_kwargs"] = {"net_arch": {"pi": [256, 256], "qf": [256, 256]}}
exp["algorithm"]["hyperparameters"] = {
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "batch_size": 256,
    "buffer_size": 100000,
    "tau": 0.005,
    "ent_coef": "auto",
    "learning_starts": 10000,
}
exp["training"]["log_interval"] = 100
EXPERIMENTS.append(("exp08_sac_pnl_reward.yaml", exp))

# --- Exp 9: PPO with trade_completion (sparse) reward ---
exp = copy.deepcopy(BASE)
exp["training"]["run_name"] = "exp09_ppo_trade_completion"
exp["reward"]["function"] = "trade_completion"
exp["reward"]["params"]["trade_reward_multiplier"] = 10.0
EXPERIMENTS.append(("exp09_ppo_trade_completion.yaml", exp))

# --- Exp 10: PPO longer lookback (40d) + bigger position ---
exp = copy.deepcopy(BASE)
exp["training"]["run_name"] = "exp10_ppo_long_lookback"
exp["environment"]["lookback_window"] = 40
exp["environment"]["position_size"] = 0.15
exp["algorithm"]["hyperparameters"]["learning_rate"] = 0.0001
EXPERIMENTS.append(("exp10_ppo_long_lookback.yaml", exp))

# Write all configs
for fname, config in EXPERIMENTS:
    path = os.path.join(BASE_DIR, fname)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  Written: {path}")

print(f"\n{len(EXPERIMENTS)} experiment configs generated.")
