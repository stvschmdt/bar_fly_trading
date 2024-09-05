from gymnasium.envs.registration import register

def register_env():
    register(
        id='AnalystGym-v0',
        entry_point='bargyms.envs.analystgym:AnalystGym',
        max_episode_steps=1000
    )

