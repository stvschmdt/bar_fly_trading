import random
import numpy as np
from stable_baselines3 import PPO
from gymnasium import Env
import pandas as pd
from register import register_env
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from register import register_env

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter
from bargyms.envs.multiactiongym import BenchmarkMultiEnv

class BenchmarkInference:
    def __init__(self, model_path, csv_path, symbols, start_date, end_date):
        """
        Initialize the BenchmarkInference class.
        
        :param model_path: Path to the saved RLlib PPO model.
        :param csv_path: Path to the CSV containing market data.
        :param symbols: List of stock symbols for inference.
        :param start_date: Start date for inference (YYYY-MM-DD).
        :param end_date: End date for inference (YYYY-MM-DD).
        """
        self.model_path = model_path
        self.csv_path = csv_path
        self.symbols = symbols
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

        # Load the saved PPO model
        self.model = PPO.load(model_path)
        
        # Load the custom environment
        self.env = BenchmarkMultiEnv(csv_path=csv_path, stock_symbols=symbols)
    
    def run_inference(self):
        """
        Run inference on the custom Gym environment.
        """
        print("Starting inference...")
        total_episodes = 0
        total_rewards = 0
        reset_options = {
            "start_date": self.start_date,
            "end_date": self.end_date,
        }

        while True:
            # Reset the environment and get initial state
            state, _ = self.env.reset(options=reset_options)

            # Check if the environment's date is within the range
            #if current_date < self.start_date or current_date > self.end_date:
                #break  # Stop if out of date range

            done = False
            episode_reward = 0

            # print out symbol and date
            while not done:
                # print the start_date shifted by env current_step
                print(f"Symbol: {self.env.current_symbol}, Date: {self.start_date + pd.DateOffset(days=self.env.current_step)}")
                # Choose action using the trained model
                action, _ = self.model.predict(state)
                
                # Take action in the environment
                state, reward, done, truncated, info = self.env.step(action)
                print(f"Action: {action}, Reward: {reward}, Balance: {self.env.current_balance}")
                
                # Accumulate reward
                episode_reward += reward

                # Log info
                #self.log_info(info)

            print(f"Episode completed for {self.env.current_symbol}. Reward: {episode_reward}")
            total_rewards += episode_reward
            total_episodes += 1
            break

        print(f"Inference completed: {total_episodes} episodes, Total Reward: {total_rewards}")

    def log_info(self, info):
        """
        Log information from the environment's `info` dictionary.
        
        :param info: Dictionary of environment metadata.
        """
        for key, value in info.items():
            print(f"INFO - {key}: {value}")


if __name__ == "__main__":
    # Define the paths and parameters
    # Register the custom environment if needed
    register_env()

    # Create the environment (no need for render_mode since there's no visual output)
    #env = gym.make("AnalystGym-v0")
    #env = gym.make("StockTradingEnv-v0")
    env = gym.make("BenchmarkMultiEnv-v0")
    model_path = "ppo_analystgym.zip"
    csv_path = "../api_data/all_data.csv"
    symbols = ["AAPL", "GOOGL", "AMZN"]
    start_date = "2024-06-01"
    end_date = "2024-10-18"

    # Run inference
    inference = BenchmarkInference(model_path, csv_path, symbols, start_date, end_date)
    inference.run_inference()
