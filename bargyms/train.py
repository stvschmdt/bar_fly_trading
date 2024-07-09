import gym
from gym import error, spaces, utils
from bargyms.envs.analystgym import AnalystGym

# Test the environment
#env = gym.make('stockgym:BarFlyGym-v0')

# register the custom environment
#select_env = "BarFlyGym-v0"
#register_env(select_env, lambda config: BarFlyGym_v0())
env = gym.make("AnalystGym-v0")

obs = env.reset()
env.render()

done = False
while True:
    action = env.action_space.sample()  # Random action selection
    obs, reward, done, _ = env.step(action)
    print('Reward:', reward)
    print('Done:', done)

