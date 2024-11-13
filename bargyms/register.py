from gymnasium.envs.registration import register

def register_env():
    register(
        id='AnalystGym-v0',
        entry_point='bargyms.envs.analystgym:AnalystGym',
        max_episode_steps=1000
    )

def register_env():
    register(
        id='StockTradingEnv-v0',
        entry_point='bargyms.envs.binarygym:StockTradingEnv',
        max_episode_steps=1000
    )

def register_env():
    register(
        id='BenchmarkEnv-v0',
        entry_point='bargyms.envs.benchmarkgym:BenchmarkEnv',
        max_episode_steps=1000
    )
