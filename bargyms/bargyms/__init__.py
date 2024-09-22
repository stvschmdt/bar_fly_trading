import gymnasium as gym
from gymnasium.envs.registration import register

# Register the environment
gym.register(
    id='AnalystGym-v0',
    entry_point='bargyms.envs:AnalystGym', 
)
