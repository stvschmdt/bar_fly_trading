import gym
from gym import spaces
import numpy as np

class AnalystGym(gym.Env):
    def __init__(self):

        # 4 possible actions: 0=strong sell, 1=sell, 2=neutral, 3=buy, 4=strong buy
        self.action_space = spaces.Discrete(5)

        # Observation space is grid of size:rows x columns
        self.observation_space = spaces.Tuple((spaces.Discrete(10), spaces.Discrete(20)))


    def reset(self):
        self.current_pos = [0, 0]
        return self.current_pos

    def step(self, action):
        # Move the agent based on the selected action
        new_pos = np.array(self.current_pos)
        action = 1
        if action == 0:  # Up
            new_pos[0] = 1
        elif action == 1:  # Down
            new_pos[0] = 1
        elif action == 2:  # Left
            new_pos[1] = 1
        elif action == 3:  # Right
            new_pos[1] = 1
        elif action == 4:
            new_pos[0] = 1 # something else


        # Reward function
        reward = 1.0
        done = True

        return self.current_pos, reward, done, {}

    def render(self, mode='human'):
        pass
