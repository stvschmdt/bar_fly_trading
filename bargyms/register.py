from gymnasium.envs.registration import register

def register_env():
    """
    Dynamically registers all custom environments.
    """
    env_specs = [
        {
            "id": "AnalystGym-v0",
            "entry_point": "bargyms.envs.analystgym:AnalystGym",
            "max_episode_steps": 1000,
        },
        {
            "id": "StockTradingEnv-v0",
            "entry_point": "bargyms.envs.binarygym:StockTradingEnv",
            "max_episode_steps": 1000,
        },
        {
            "id": "BenchmarkEnv-v0",
            "entry_point": "bargyms.envs.benchmarkgym:BenchmarkEnv",
            "max_episode_steps": 1000,
        },
        {
            "id": "BenchmarkMultiEnv-v0",
            "entry_point": "bargyms.envs.multiactiongym:BenchmarkMultiEnv",
            "max_episode_steps": 20,
        },
        {
            "id": "GoldTradeEnv-v0",
            "entry_point": "bargyms.envs.goldtradegym:GoldTradeEnv",
            "max_episode_steps": 1000,
        },
    ]

    # Dynamically register each environment
    for env_spec in env_specs:
        register(
            id=env_spec["id"],
            entry_point=env_spec["entry_point"],
            max_episode_steps=env_spec["max_episode_steps"],
        )
        print(f"Registered environment: {env_spec['id']}")

if __name__ == "__main__":
    register_env()

