import gymnasium as gym
from gymnasium.envs.registration import register

# Register the environment
gym.register(
    id='AnalystGym-v0',
    entry_point='bargyms.envs:AnalystGym', 
)
# Register the environment
gym.register(
    id='StockTradingEnv-v0',
    entry_point='bargyms.envs:StockTradingEnv', 
)
# Register the environment
gym.register(
    id='BenchmarkEnv-v0',
    entry_point='bargyms.envs:BenchmarkEnv', 
)

# Register the environment
gym.register(
    id='BenchmarkMultiEnv-v0',
    entry_point='bargyms.envs:BenchmarkMultiEnv', 
)
