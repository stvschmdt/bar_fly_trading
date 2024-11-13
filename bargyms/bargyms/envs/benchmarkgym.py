import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random

class BenchmarkEnv(gym.Env):
    def __init__(self, csv_path=None, stock_symbols=None, render_mode=None, initial_balance=1):
        super(BenchmarkEnv, self).__init__()
        if csv_path is None:
            csv_path = "../api_data/all_data.csv"  # Update with actual path to your CSV
        if stock_symbols is None:
            stock_symbols = ["AAPL", "NVDA", "AMD", "GOOG", "AMZN", "QCOM", "MSFT"]
        self.render_mode = render_mode
        # Load data from CSV file
        self.data = pd.read_csv(csv_path)
        # ensure the data is sorted by date most recent first
        self.data = self.data.sort_values(by='date', ascending=False)
        
        # Store the list of symbols to be used for training
        self.stock_symbols = stock_symbols
        self.spy_symbol = 'SPY'  # SPY for benchmarking
        #self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_pl = 0
        self.total_positive_trades = 0
        self.total_negative_trades = 0
        
        self.n_days = 20
        
        self.cols = ['adjusted_open', 'adjusted_high', 'adjusted_low', 'adjusted_close', 'sma_20', 'sma_50', 'ema_20', 'ema_50', 'bbands_upper_20', 'bbands_lower_20', 'rsi_14', 'adx_14', 'atr_14', 'macd','treasury_yield_2year', 'treasury_yield_10year', '52_week_high', '52_week_low', 'beta', 'analyst_rating_strong_buy', 'analyst_rating_buy', 'analyst_rating_hold', 'analyst_rating_sell', 'analyst_rating_strong_sell', 'day_of_week_num', 'day_of_year', 'year']
        self.spy_cols = ['adjusted_open', 'adjusted_high', 'adjusted_low', 'adjusted_close', 'sma_20', 'sma_50', 'bbands_upper_20', 'bbands_lower_20']
        # Define observation and action spaces
        space_cols = len(self.cols) + len(self.spy_cols) + 3
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_days, space_cols), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        
        
        # Initialize placeholders for current state
        self.current_symbol = None
        self.current_index = None
        self.current_data_window = None
        self.spy_data = None
        self.shares_owned = 0
        self.options_owned = 0
        self.current_step = 0
        self.total_pl = 0

    def reset(self, seed=None, options=None):
        # map self.stock_symbols to the unique integer for each symbol
        self.symbols_map = {symbol: i for i, symbol in enumerate(self.stock_symbols)}
        # Select a random symbol from the input list
        self.current_symbol = random.choice(self.stock_symbols)
        # n days for episode
        
        # Filter the data for the selected symbol
        symbol_data = self.data[self.data['symbol'] == self.current_symbol].reset_index(drop=True)
        # randomly select a start index, with a minimum of 10 days from the earliest date
        min_start_index = self.n_days
        # max start date is 20 days from the end of the data
        max_start_index = len(symbol_data) - (self.n_days * 2) - 2
        # select a random start date in the range of min, max
        self.current_index = random.randint(min_start_index, max_start_index)
        # get the data for the selected date range, starting from the current index and adding 20 days
        self.current_data_window = symbol_data.iloc[self.current_index : self.current_index + (self.n_days*2)+1]

        # Get SPY data for the same date range
        self.spy_data = self.data[self.data['symbol'] == self.spy_symbol].reset_index(drop=True)
        self.spy_data = self.spy_data.iloc[self.current_index : self.current_index + (self.n_days*2)+1]
        
        # Reset account, holdings, and state
        self.initial_balance = 0
        self.initial_cost = 0
        self.final_cost = 0
        self.final_balance = 0
        self.current_pl = 0
        self.shares_owned = 0
        self.options_owned = 0
        self.current_step = 0
        self.num_trades = 0
        
        # Set up the initial 10-day observation window
        self.state_data = self.current_data_window[self.cols].values
        # get the same columns for the spy data
        self.spy_data = self.spy_data[self.spy_cols].values
        # create the action element of the state vector, which is a vector of 1 element for each day in the window
        # get the digit of the symbol we are using
        sym = np.full((len(self.state_data), 1), self.symbols_map[self.current_symbol])
        # create an account vector same as action
        account = np.full((len(self.state_data), 1), 0)
        # the action is initialized to -1 for each of the length of self.current_data_window
        actions = np.full((len(self.state_data), 1), -1)
        # horizontally stack the state data and the sym vector
        self.state_data = np.hstack((self.state_data, sym))
        # horizontally stack the state data and the spy data
        self.state_data = np.hstack((self.state_data, self.spy_data))
        # horizontally stack the state data and the account
        self.state_data = np.hstack((self.state_data, account))
        # horizontally stack the state data and the actions
        self.state = np.hstack((self.state_data, actions))
        
        self.state_data = self.state
        # print the company
        #print(self.current_symbol)
        # print the date of the first day in the window
        #print(self.current_data_window.iloc[0]['date'])
        #print(self.current_data_window.iloc[-1]['date'])
        #print(self.state)

        # return the first 10 days of the state
        # return the earliest 10 dates in the window
        self.info = {'reward':0, 'current_pl':0, 'end_pl':0}
        return self.state.astype(np.float32)[-self.n_days:], {}
        #return self.state.astype(np.float32)[0:10], {}

    # step and reward functions follow as previously outlined.
    def step(self, action):
        # Check if episode is done (10 steps have passed)
        done = self.current_step >= self.n_days - 1
        # if action is 0, do not buy or sell
        if action == 0:
            reward = 0.1
            # change the state at this step to reflect the action taken
            self.state_data[-(self.current_step + self.n_days + 1)][-1] = 0
        # if the action is to buy a share
        if action == 1:
            # make sure we havent already bought the max number of shares
            if self.shares_owned > 0:
                reward = -25.
                # change the state at this step to reflect the action taken
                self.state_data[-(self.current_step + self.n_days + 1)][-1] = 0
            else:
                # buy a share
                self.shares_owned += 10
                self.num_trades += 1
                # calculate the cost of the share -> the visible day closing
                self.initial_cost = self.current_data_window.iloc[-(self.current_step+self.n_days)]['adjusted_close'] * 100
                # reward is the cost of the share * 10
                reward = -1.
                self.state_data[-(self.current_step + self.n_days + 1)][-1] = action  
                self.state_data[-(self.current_step + self.n_days + 1)][-2] = -self.initial_cost
        elif action == 2:
            # make sure you own the stock first
            if self.shares_owned > 0:
                # sell shares -> the current day opening price
                self.final_cost = self.current_data_window.iloc[-(self.current_step+self.n_days)]['adjusted_close'] *100
                self.shares_owned -= 10
                # calculate the profit or loss
                self.current_pl += self.final_cost - self.initial_cost
                # calculate the reward
                reward = self.final_cost - self.initial_cost
                # calculate percent profit or loss
                reward_perc = reward / self.initial_cost
                self.state_data[-(self.current_step + self.n_days + 1)][-1] = action 
                self.state_data[-(self.current_step + self.n_days + 1)][-2] = reward
                # tell us about a positive trade
                if reward_perc > .03:
                    # represent options
                    reward = reward * 2.0
                if reward > 0:
                    self.total_positive_trades += 1
                    self.info['positive_trade'] = self.total_positive_trades
                else:
                    #reward = reward * 3.0 # punish losses more
                    self.total_negative_trades += 1
                    self.info['negative_trade'] = self.total_negative_trades
            else:
                reward = -25.
                self.state_data[-(self.current_step+self.n_days+1)][-1] = 0
        # if its the last step, sell all shares
        if done:
            if self.shares_owned > 0:
                self.final_cost = self.current_data_window.iloc[-(self.current_step+self.n_days+1)]['adjusted_close'] * 100
                self.shares_owned -= 10
                self.current_pl += self.final_cost - self.initial_cost
                reward = self.final_cost - self.initial_cost
                # calculate percent profit or loss
                reward_perc = reward / self.initial_cost
                # tell us about a positive trade
                if reward > 0:
                    self.info['positive_trade'] = reward
                    self.total_positive_trades += 1
                else:
                    self.info['negative_trade'] = reward
                    self.total_negative_trades += 1
                if reward_perc > .03:
                    # represent options
                    reward = reward * 2.0
                self.info['end_pl'] = self.current_pl
                self.state_data[-(self.current_step+self.n_days+1)][-1] = 0
                self.state_data[-(self.current_step + self.n_days + 1)][-2] = reward
            else:
                reward = 0.0
            self.total_pl += self.current_pl
            reward = self.current_pl
            #if self.num_trades > 0 and reward < 10:
                #reward = reward * .8
            #self.info['num_trades'] = self.num_trades
            
            # print episode pl and total pl
            # print win rate (positive trades / positive trades + negative trades)
            win_rate = self.total_positive_trades / (self.total_positive_trades + self.total_negative_trades)
            if self.current_pl != 0.0:
                # cap to 2 decimal places
                print(f"Episode P/L: {self.current_pl:.2f}, Win Rate: {win_rate:.2f}")
            self.info['win_rate'] = win_rate

        self.current_step += 1
        self.state = self.state_data[-(self.current_step + self.n_days ) : -(self.current_step)].astype(np.float32)
        #print(self.state)
        self.info['reward'] = reward
        self.info['current_pl'] = self.current_pl
        return self.state, reward, done, {}, self.info

    def render(self, mode='human'):
        """Render the environment."""
        if self.render_mode == 'rgb_array':
            # Return an image (e.g., NumPy array)
            return np.zeros((400, 600, 3), dtype=np.uint8)  # Placeholder for an RGB image
        elif self.render_mode == 'human':
            print("Rendering in human mode")
        else:
            pass

if __name__ == "__main__":
    # Initialize environment parameters
    csv_path = "../../../api_data/all_data.csv"  # Update with actual path to your CSV
    stock_symbols = ["AAPL", "NVDA", "AMD"]  # Example stock symbols
    stock_symbols = ["AAPL"]
    env = BenchmarkEnv(csv_path=csv_path, stock_symbols=stock_symbols, initial_balance=1)

    # Run the environment for a few episodes
    num_episodes = 1
    for episode in range(num_episodes):
        print(f"=== Episode {episode + 1} ===")
        
        # Reset environment and get initial state
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        # Run the episode
        while not done:
            print(f"Step {step_count}")
            # Sample a random action
            action = env.action_space.sample()
            print(f"Action: {action}")
            print(f"State:\n{state}")
            
            next_state, reward, done, _ , _ = env.step(action)
            
            # Accumulate reward and increase step count
            total_reward += reward
            step_count += 1
            
            # Print step information
            # Take a step in the environment
            print(f"Reward: {reward}")
            print(f"Total Reward so far: {total_reward}")
            print(f"Balance: {env.balance}, Current P/L: {env.current_pl}")
            print("===")
            
            # Move to next state
            state = next_state

        print(f"Episode {episode + 1} finished with Total Reward: {total_reward}")
        print("===" * 10)

