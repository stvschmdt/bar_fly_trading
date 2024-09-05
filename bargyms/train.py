import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

def load_config(config_file):
    """Loads the YAML configuration file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    # Load the configuration file
    config = load_config("configs/ppo_agent.yaml")

    # Register the custom environment if needed
    from register import register_env
    register_env()

    # Create the environment (no need for render_mode since there's no visual output)
    env = gym.make("AnalystGym-v0")

    # Set up a directory for monitoring results
    monitor_dir = './monitor_results'
    if not os.path.exists(monitor_dir):
        os.makedirs(monitor_dir)

    # Set up the directory for TensorBoard logs from config
    tensorboard_log_dir = config['tensorboard']['log_dir']
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)

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
                callback=eval_callback)

    # Save the trained model
    model.save("ppo_analystgym")

    # Close the environment when done
    env.close()

if __name__ == "__main__":
    main()

