import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
import os
import yaml
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from register import register_env

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter



class InfoDictTensorboardCallback(BaseCallback):
    """
    Custom callback to log scalar fields in the `info` dictionary to TensorBoard.
    """
    def __init__(self, writer, verbose=0):
        super(InfoDictTensorboardCallback, self).__init__(verbose)
        self.writer = writer

    def _on_step(self) -> bool:
        # Extract the `info` dictionary for the current step
        info_dict = self.locals["infos"][0]

        # Log each item in `info_dict` to TensorBoard if it's a scalar and not in the ignored keys
        for key, value in info_dict.items():
            # Skip specific keys with known non-scalar structures
            if key in {"TimeLimit.truncated", "episode", "terminal_observation"}:
                continue
            # Log only scalar values (int, float, numpy scalar)
            if isinstance(value, (int, float, np.number)):
                self.writer.add_scalar(f"Info/{key}", value, self.num_timesteps)
            else:
                # Print the key and value type to diagnose other potential issues
                print(f"Skipping non-scalar info key '{key}' with value: {value} (type: {type(value)})")
        # Access the observations from the environment
        obs = self.locals["rollout_buffer"].observations
        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        # Get the policy network
        policy = self.model.policy

        # Forward pass to get the logits
        with torch.no_grad():
            distribution = policy.get_distribution(obs_tensor)
            logits = distribution.distribution.logits  # Logits from the Categorical distribution

        # Check for NaN or Inf in the logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise ValueError("Logits contain NaN or Inf values during training")

        return True

def load_config(config_file):
    """Loads the YAML configuration file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():


    #( Load the configuration file
    config = load_config("configs/ppo_agent.yaml")

    # Register the custom environment if needed
    register_env()

    # Create the environment (no need for render_mode since there's no visual output)
    #env = gym.make("AnalystGym-v0")
    #env = gym.make("StockTradingEnv-v0")
    env = Monitor(gym.make("GoldTradeEnv-v0"))

    # Set up a directory for monitoring results
    monitor_dir = './monitor_results'
    if not os.path.exists(monitor_dir):
        os.makedirs(monitor_dir)

    # Set up the directory for TensorBoard logs from config
    tensorboard_log_dir = config['tensorboard']['log_dir']
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)

    # Initialize TensorBoard writer
    writer = SummaryWriter(tensorboard_log_dir + "/runs")
    # Use InfoDictTensorboardCallback to log selected info fields
    info_callback = InfoDictTensorboardCallback(writer=writer)


    # Extract PPO agent parameters from config
    ppo_params = config['ppo']

    # Ensure the tensorboard_log argument is added dynamically from config
    ppo_params['tensorboard_log'] = tensorboard_log_dir

    # Initialize PPO agent with parameters from the config file
    model = PPO(env=env, **ppo_params)  # Dynamically pass all PPO params

    # Set up a callback for model evaluation (optional)
    eval_callback = EvalCallback(
        env,
        best_model_save_path=monitor_dir,
        log_path=monitor_dir,
        eval_freq=5000,  # Evaluate every 5000 steps
        deterministic=True,
        render=False
    )

    # Train the agent and log progress to TensorBoard with the interval from config
    model.learn(total_timesteps=config['training']['total_timesteps'],
                log_interval=config['training']['log_interval'],
                callback=[eval_callback, info_callback])

    # Save the trained model
    model.save("ppo_analystgym")

    # Close the environment when done
    env.close()

if __name__ == "__main__":
    main()

