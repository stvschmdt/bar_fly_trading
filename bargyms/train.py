"""
Config-driven RL training script.

Supports PPO, SAC, TD3 via YAML config files.
Logs trading metrics and policy introspection to TensorBoard.

Usage:
    python bargyms/train.py --config bargyms/configs/ppo_discrete.yaml
    python bargyms/train.py --config bargyms/configs/sac_continuous.yaml
    python bargyms/train.py --config bargyms/configs/td3_continuous.yaml --symbols NVDA AAPL
"""

import argparse
import os
import sys
import yaml

import numpy as np

# Add project root to path (append so script dir stays first for bargyms.envs)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter

from bargyms.envs.trading_env import TradingEnv, RLDataLoader

ALGO_MAP = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}


def load_config(path):
    """Load and validate YAML config."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Validate algo + action space compatibility
    algo = config["algorithm"]["name"].upper()
    action_space = config["environment"]["action_space"]
    if algo in ("SAC", "TD3") and action_space != "continuous":
        raise ValueError(f"{algo} requires action_space='continuous', got '{action_space}'")

    return config


class TradingMetricsCallback(BaseCallback):
    """
    Logs trading-specific metrics to TensorBoard:
    - Episode return, win rate, avg trade return, number of trades
    - Per-step portfolio value
    """

    def __init__(self, writer, verbose=0):
        super().__init__(verbose)
        self.writer = writer
        self._episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            # Log per-step scalars
            if "portfolio_value" in info:
                self.writer.add_scalar(
                    "trading/portfolio_value", info["portfolio_value"], self.num_timesteps
                )
            if "step_return" in info:
                self.writer.add_scalar(
                    "trading/step_return", info["step_return"], self.num_timesteps
                )

            # Log episode-end metrics
            if "episode_return_pct" in info:
                self._episode_count += 1
                self.writer.add_scalar(
                    "trading/episode_return_pct", info["episode_return_pct"],
                    self._episode_count,
                )
                self.writer.add_scalar(
                    "trading/final_value", info.get("final_value", 0),
                    self._episode_count,
                )
                self.writer.add_scalar(
                    "trading/n_trades", info.get("n_trades", 0),
                    self._episode_count,
                )
                if "win_rate" in info:
                    self.writer.add_scalar(
                        "trading/win_rate", info["win_rate"],
                        self._episode_count,
                    )
                if "avg_trade_return_pct" in info:
                    self.writer.add_scalar(
                        "trading/avg_trade_return_pct", info["avg_trade_return_pct"],
                        self._episode_count,
                    )

        return True


class PolicyIntrospectionCallback(BaseCallback):
    """
    Periodic policy introspection using gradient-based saliency.

    Every `introspect_freq` steps:
    - Logs value function estimates
    - Logs action probabilities (PPO) or mean action (SAC/TD3)
    - Logs feature saliency (input gradient magnitude)
    """

    def __init__(self, writer, introspect_freq=50000, verbose=0):
        super().__init__(verbose)
        self.writer = writer
        self.introspect_freq = introspect_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.introspect_freq != 0 or self.num_timesteps == 0:
            return True

        try:
            import torch

            # Get a sample observation
            obs = self.locals.get("new_obs")
            if obs is None:
                return True
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.tensor(obs[:1], dtype=torch.float32)
            else:
                obs_tensor = obs[:1].float()

            policy = self.model.policy
            policy.eval()

            with torch.no_grad():
                # Value estimate
                if hasattr(policy, "predict_values"):
                    values = policy.predict_values(obs_tensor)
                    self.writer.add_scalar(
                        "introspection/value_estimate",
                        values.mean().item(),
                        self.num_timesteps,
                    )

                # Action distribution (PPO) or mean action (SAC/TD3)
                if hasattr(policy, "get_distribution"):
                    dist = policy.get_distribution(obs_tensor)
                    if hasattr(dist.distribution, "probs"):
                        probs = dist.distribution.probs[0]
                        for i, p in enumerate(probs):
                            label = ["hold", "buy", "sell"][i] if i < 3 else f"a{i}"
                            self.writer.add_scalar(
                                f"introspection/action_prob_{label}",
                                p.item(),
                                self.num_timesteps,
                            )
                    elif hasattr(dist.distribution, "mean"):
                        self.writer.add_scalar(
                            "introspection/action_mean",
                            dist.distribution.mean[0].item(),
                            self.num_timesteps,
                        )

            # Feature saliency via input gradient
            obs_grad = obs_tensor.clone().requires_grad_(True)
            if hasattr(policy, "predict_values"):
                value = policy.predict_values(obs_grad)
                value.sum().backward()
                if obs_grad.grad is not None:
                    # Average saliency across time dimension
                    saliency = obs_grad.grad.abs().mean(dim=(0, 1))
                    self.writer.add_histogram(
                        "introspection/feature_saliency",
                        saliency.cpu().numpy(),
                        self.num_timesteps,
                    )

            policy.train()

        except Exception as e:
            if self.verbose:
                print(f"PolicyIntrospection error: {e}")

        return True


def make_env(data_loader, config, mode="train"):
    """Factory for creating monitored trading environments."""
    def _init():
        env = TradingEnv(data_loader, config, mode=mode)
        env = Monitor(env)
        return env
    return _init


def build_algo_kwargs(config):
    """Build SB3 algorithm constructor kwargs from config."""
    algo_cfg = config["algorithm"]
    hyperparams = algo_cfg.get("hyperparameters", {})
    policy_kwargs = algo_cfg.get("policy_kwargs", {})

    # Convert activation function name to torch class
    if "activation_fn" in policy_kwargs:
        import torch.nn as nn
        act_map = {"ReLU": nn.ReLU, "Tanh": nn.Tanh, "GELU": nn.GELU}
        policy_kwargs["activation_fn"] = act_map.get(
            policy_kwargs["activation_fn"], nn.ReLU
        )

    kwargs = {
        "policy": algo_cfg.get("policy", "MlpPolicy"),
    }
    if policy_kwargs:
        kwargs["policy_kwargs"] = policy_kwargs

    # Add hyperparameters (filter by algo type)
    algo_name = algo_cfg["name"].upper()
    for key, value in hyperparams.items():
        # Skip PPO-only params for off-policy algos
        if algo_name in ("SAC", "TD3") and key in ("n_steps", "n_epochs", "clip_range", "gae_lambda"):
            continue
        # Skip off-policy params for PPO
        if algo_name == "PPO" and key in ("buffer_size", "tau", "learning_starts"):
            continue
        kwargs[key] = value

    return kwargs


def main():
    parser = argparse.ArgumentParser(description="Train RL trading agent")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--symbols", nargs="+", help="Override symbols list")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.symbols:
        config["data"]["symbols"] = args.symbols

    train_cfg = config.get("training", {})
    algo_name = config["algorithm"]["name"].upper()

    # Set up directories
    tb_dir = train_cfg.get("tensorboard_dir", "./rl_logs")
    model_dir = train_cfg.get("model_save_path", "./rl_models")
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Load data
    print(f"Loading data for {algo_name} training...")
    data_loader = RLDataLoader(config)
    data_loader.load()

    if not data_loader.get_symbols():
        raise ValueError("No symbols loaded. Check data path and date range.")

    # Create vectorized environments
    print("Creating environments...")
    train_env = DummyVecEnv([make_env(data_loader, config, mode="train")])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv([make_env(data_loader, config, mode="eval")])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Create algorithm
    print(f"Creating {algo_name} agent...")
    algo_cls = ALGO_MAP.get(algo_name)
    if algo_cls is None:
        raise ValueError(f"Unknown algorithm '{algo_name}'. Available: {list(ALGO_MAP.keys())}")

    algo_kwargs = build_algo_kwargs(config)
    seed = train_cfg.get("seed", 42)

    model = algo_cls(
        env=train_env,
        tensorboard_log=tb_dir,
        seed=seed,
        verbose=1,
        **algo_kwargs,
    )

    print(f"Policy network: {model.policy}")

    # Set up callbacks
    run_name = f"{algo_name}_{config['environment']['action_space']}"
    writer = SummaryWriter(os.path.join(tb_dir, run_name))

    callbacks = [
        TradingMetricsCallback(writer=writer),
        PolicyIntrospectionCallback(writer=writer, introspect_freq=50000),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(model_dir, "best"),
            log_path=os.path.join(model_dir, "eval_logs"),
            eval_freq=train_cfg.get("eval_freq", 10000),
            n_eval_episodes=train_cfg.get("n_eval_episodes", 20),
            deterministic=True,
        ),
        CheckpointCallback(
            save_freq=train_cfg.get("save_freq", 50000),
            save_path=os.path.join(model_dir, "checkpoints"),
            name_prefix=run_name,
        ),
    ]

    # Train
    total_timesteps = train_cfg.get("total_timesteps", 500000)
    print(f"\nStarting {algo_name} training for {total_timesteps:,} timesteps...")
    print(f"  Action space: {config['environment']['action_space']}")
    print(f"  Features: {config.get('features', {}).get('preset', 'standard')}")
    print(f"  Reward: {config.get('reward', {}).get('function', 'pnl_pct_step')}")
    print(f"  Symbols: {len(data_loader.get_symbols())}")
    print(f"  TensorBoard: tensorboard --logdir {tb_dir}")
    print()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=train_cfg.get("log_interval", 10),
    )

    # Save final model + normalization stats
    final_path = os.path.join(model_dir, f"{run_name}_final")
    model.save(final_path)
    train_env.save(os.path.join(model_dir, f"{run_name}_vecnormalize.pkl"))

    # Save config alongside model for reproducibility
    config_save_path = os.path.join(model_dir, f"{run_name}_config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    writer.close()
    train_env.close()
    eval_env.close()

    print(f"\nTraining complete!")
    print(f"  Model saved: {final_path}.zip")
    print(f"  Config saved: {config_save_path}")
    print(f"  Best model: {model_dir}/best/best_model.zip")


if __name__ == "__main__":
    main()
