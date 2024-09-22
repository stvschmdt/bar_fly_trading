import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class AnalystGym(gym.Env):
    """Custom environment for Gym."""
    
    def __init__(self, render_mode=None):
        super(AnalystGym, self).__init__()
        
        # Define action and observation space
        # creaet a custom observation space of 5 pandas dataframe rows and 3 float columns
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1, 6), dtype=np.float32)
        # action space is a discrete space of 1 action between 0 and 2
        self.action_space = spaces.Discrete(3)


        # read in dataframe for the environment
        self.df = pd.read_csv('../api_data/all_data.csv')
        self.symbols = ['MS'] # pick one to test
        self.df = self.df[self.df['symbol'].isin(self.symbols)]
        # only use the columns we need
        self.columns =['date', 'adjusted_close', 'rsi_14', 'sma_20']
        self.df = self.df[self.columns]
        # set max steps to 10
        self.max_steps = 5
        # Store render_mode
        self.render_mode = render_mode
        self.max_total_profit = 0.0

        # Initial state
        self.state = np.zeros(self.observation_space.shape)
        # call reset to initialize the state
        self.reset()

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        # set the step to 0
        self.current_step = 0
        self.buy = 0
        self.total_profit = 0.0
        # randomly select a row from the dataframe between the 5th and the 15th to last row
        idx = np.random.randint(5, len(self.df) - 15)
        # use this to select the data for that date and the 4 days after it
        self.state = self.df.iloc[idx, 1:4].values
        # padding the state with zeros for actions
        # pad the state with zeros for the action
        self.state = np.pad(self.state, (0, 3), 'constant')
        # set target to the adjusted close of the day after the last day in the state
        self.target = self.df.iloc[idx+1, 1]
        #print(self.state, self.target)
        return self.state, {}

    def step(self, action):
        """Execute one time step within the environment."""
        # action is a number between 0 and 2 for buy, hold, sell
        # we will update the state with the action, the 4th column is buy, 5th is hold, 6th is sell
        # self.step is the time step corresponding to the row in the dataframe
        action = int(action)
        # set the action to 1 in the state
        self.state[3+action] = 1
        # set reward based on if self.target is greater than the adjusted close of the last day in the state
        # ensure there is a buy first
        if self.buy == 0 and action == 2:
            reward = -10
        elif self.buy == 1 and action == 2:
            if self.target < self.state[0]:
                reward = 5
            else:
                reward = -3
            self.buy = 0
        if action == 0:
            if self.target > self.state[0]:
                reward = 5
            else:
                reward = -3
            self.buy = 1
        # if action is 1, hold, reward is 0
        if action == 1:
            reward = 0
        self.current_step += 1
        if self.current_step == self.max_steps:
            done = True
            self.total_profit = self.total_profit + reward
            if self.total_profit > self.max_total_profit:
                self.max_total_profit = self.total_profit
                print(f"Max Total Profit Reached ---------------------------->: {self.max_total_profit}")
        else:
            done = False  # Example termination flag
            self.total_profit = self.total_profit + reward
        return self.state, reward, done, {}, {}

    def render(self, mode='human'):
        """Render the environment."""
        if self.render_mode == 'rgb_array':
            # Return an image (e.g., NumPy array)
            return np.zeros((400, 600, 3), dtype=np.uint8)  # Placeholder for an RGB image
        elif self.render_mode == 'human':
            print("Rendering in human mode")
        else:
            pass

