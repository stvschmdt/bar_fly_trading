import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import random

class BenchmarkMultiEnv(gym.Env):
    def __init__(self, csv_path=None, stock_symbols=None, render_mode=None, initial_balance=500000):
        super(BenchmarkMultiEnv, self).__init__()
        if csv_path is None:
            csv_path = "../api_data/all_data.csv"  # Update with actual path to your CSV
        if stock_symbols is None:
            stock_symbols = ["AMD", "NVDA", "AVGO", "AMZN", "GOOG"]
        self.render_mode = render_mode
        # Load data from CSV file
        self.data = pd.read_csv(csv_path)
        # ensure the data is sorted by date most recent first
        self.data = self.data.sort_values(by='date', ascending=False)
        # example data
        # all_data containts
        # 1-1-2023 -> 12-31-2023
        
        # Store the list of symbols to be used for training
        self.stock_symbols = stock_symbols
        self.symbols_map = {symbol: i for i, symbol in enumerate(self.stock_symbols)}
        self.symbols_pl = {symbol: 0.0 for i, symbol in enumerate(self.stock_symbols)}
        self.spy_symbol = 'SPY'  # SPY for benchmarking
        self.sector_symbol = 'XLK'  # SOXL for benchmarking chips
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_pl = 0
        self.total_positive_trades = 0
        self.total_negative_trades = 0
        self.max_balance = self.initial_balance
        
        # we give the agent n_days to learn the environment (window view)
        self.n_days = 30
        # example
        # agent sees at the beginning of an episode
        # 10-31-2023 back to 10-1-2023
        # example at step 2
        # agent sees 11-01-2023 back to 10-02-2023
        # we want to show the agent window amount of data to take an action
        self.window = 15
        # low number of discrete shares
        self.low_shares = 10
        # high number of discrete shares
        self.high_shares = 100
        
        self.cols = ['adjusted_open', 'adjusted_high', 'adjusted_low', 'adjusted_close', 'sma_20', 'sma_50', 'ema_20', 'ema_50', 'bbands_upper_20', 'bbands_lower_20', 'rsi_14', 'adx_14', 'atr_14', 'macd','treasury_yield_2year', 'treasury_yield_10year', '52_week_high', '52_week_low', 'beta', 'analyst_rating_strong_buy', 'analyst_rating_buy', 'analyst_rating_hold', 'analyst_rating_sell', 'analyst_rating_strong_sell', 'day_of_week_num', 'day_of_year', 'year']
        self.spy_cols = ['adjusted_open', 'adjusted_high', 'adjusted_low', 'adjusted_close', 'sma_20', 'sma_50', 'bbands_upper_20', 'bbands_lower_20']
        self.sector_cols = ['adjusted_open', 'adjusted_high', 'adjusted_low', 'adjusted_close', 'sma_20', 'sma_50', 'bbands_upper_20', 'bbands_lower_20']
        # Define observation and action spaces
        self.action_cols = 8
        space_cols = len(self.cols) + len(self.spy_cols) + len(self.sector_cols) + self.action_cols
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_days, space_cols), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        
        
        # Initialize placeholders for overall stats
        self.current_episode = 0
        self.total_pl = 0
        self.win_trade_pl = 0
        self.win_trade = 0
        self.loss_trade_pl =0
        self.loss_trade = 0

    def reset(self, seed=None, options=None):

        # for cleaner monitoring of training
        if self.current_episode % 250000 == 0:
            self.total_pl = 0
            self.win_trade_pl = 0
            self.win_trade = 0
            self.loss_trade_pl =0
            self.loss_trade = 0
            self.total_positive_trades = 0
            self.total_negative_trades = 0
            self.num_low = 0
            self.num_high = 0

        self.current_episode += 1
        # map self.stock_symbols to the unique integer for each symbol
        # Select a random symbol from the input list
        self.current_symbol = random.choice(self.stock_symbols)
        # n days for episode
        
        # Filter the data for the selected symbol
        symbol_data = self.data[self.data['symbol'] == self.current_symbol].reset_index(drop=True)
        
        # check for inference mode
        if options is not None and options.get('inference', True):
            # we need to set the dates
            self.start_date = options.get('start_date', '2021-01-01')
            self.end_date = options.get('end_date', '2021-12-31')
            # set current index to the start date, which is the index of the date in the data
            self.current_index = symbol_data[symbol_data['date'] == self.start_date].index[0]
        else:
            # randomly select a start index, with a minimum of 10 days from the earliest date
            min_start_index = self.n_days + 30
            # max start date is 20 days from the end of the data

            max_start_index = len(symbol_data) - (self.n_days + self.window + 1)
            # select a random start date in the range of min, max
            self.current_index = random.randint(min_start_index, max_start_index)
        # get the data for the selected date range, starting from the current index and adding 20 days
        self.current_data_window = symbol_data.iloc[self.current_index : self.current_index + (self.n_days + self.window + 1)]

        # Get SPY data for the same date range
        self.spy_data = self.data[self.data['symbol'] == self.spy_symbol].reset_index(drop=True)
        self.spy_data = self.spy_data.iloc[self.current_index : self.current_index + (self.n_days + self.window + 1)]

        # Get Sector data for the same date range
        self.sector_data = self.data[self.data['symbol'] == self.sector_symbol].reset_index(drop=True)
        self.sector_data = self.sector_data.iloc[self.current_index : self.current_index + (self.n_days + self.window + 1)]
        
        # Reset account, holdings, and state
        self.current_balance = self.initial_balance
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
        # horizontally stack the state data and the sym vector
        self.state_data = np.hstack((self.state_data, sym))
        # get the columns for the sector data
        self.sector_data = self.sector_data[self.sector_cols].values
        # horizontally stack the state data and the sector data
        self.state_data = np.hstack((self.state_data, self.sector_data))
        # horizontally stack the state data and the spy data
        self.state_data = np.hstack((self.state_data, self.spy_data))
        # the state shares is initialized to 0 for each of the length of self.current_data_window
        self.shares = np.full((len(self.state_data), 1), self.shares_owned)
        # horizontally stack the state data and the shares state
        self.state_data = np.hstack((self.state_data, self.shares))
        # create an account vector same as action
        self.account = np.full((len(self.state_data), 1), self.initial_balance)
        # horizontally stack the state data and the account
        self.state_data = np.hstack((self.state_data, self.account))
        # the action is initialized to -1 for each of the length of self.current_data_window
        actions = np.full((len(self.state_data), 5), -1)
        # horizontally stack the state data and the actions
        self.state = np.hstack((self.state_data, actions))
        
        self.state_data = self.state
        # print the company
        #print(self.current_symbol)
        # print the date of the first day in the window
        #print(self.current_data_window.iloc[0]['date'])
        #print(self.state)

        # return the first 10 days of the state
        # return the earliest 10 dates in the window
        self.info = {'reward':0, 'current_pl':0, 'end_pl':0}
        return self.state.astype(np.float32)[-self.n_days:], {}
        #return self.state.astype(np.float32)[0:10], {}

    # step and reward functions follow as previously outlined.
    def step(self, action):
        # Check if episode is done (10 steps have passed)
        done = self.current_step >= self.window - 1
        share_price = self.current_data_window.iloc[-(self.current_step+self.n_days)]['adjusted_close']
        low_cost = self.current_data_window.iloc[-(self.current_step+self.n_days)]['adjusted_close'] * self.low_shares
        high_cost = self.current_data_window.iloc[-(self.current_step+self.n_days)]['adjusted_close'] * self.high_shares

        # if action is 0, do not buy or sell
        if action == 0:
            if self.shares_owned == 0:
                reward = 1.0 # risk free rate of return
            else:
                reward = 0.0
            # change the state at this step to reflect the action taken
            self.state_data[-(self.current_step + self.n_days)][-5] = 0

        # if the action is to buy low shares
        if action == 1:
            # make sure we have enough money to buy the shares
            if self.current_balance >= low_cost:
                self.shares_owned += 10
                self.num_trades += 1
                self.num_low += 1
                # calculate the cost of the share -> the visible day closing
                #self.initial_cost = (share_price  * self.shares_owned) / self.num_trades
                # update the state
                self.state_data[-(self.current_step + self.n_days)][-4] = 1
                self.state_data[-(self.current_step + self.n_days)][-6] += -low_cost
                self.state_data[-(self.current_step + self.n_days)][-7] += self.shares_owned
                self.current_balance -= low_cost
                reward = 0.0
            else:
                print('invalid buy action')
                # change the state at this step to reflect the action taken
                self.state_data[-(self.current_step + self.n_days)][-5] = 0
                reward = -100000
                
        # take the action is to buy high shares
        elif action == 2:
            # make sure we have enough money to buy the shares
            if self.current_balance >= high_cost:
                self.shares_owned += 100
                self.num_trades += 1
                self.num_high += 1
                # calculate the cost of the share -> the visible day closing
                #self.initial_cost = (share_price  * self.shares_owned) / self.num_trades
                # assume minimal risk for entering into a position
                self.state_data[-(self.current_step + self.n_days)][-3] = 1 
                self.state_data[-(self.current_step + self.n_days)][-6] += -high_cost
                self.state_data[-(self.current_step + self.n_days)][-7] += self.shares_owned
                self.current_balance -= high_cost
                reward = -1.0 # risk free rate of return
            else:
                print('invalid buy action')
                # change the state at this step to reflect the action taken
                self.state_data[-(self.current_step + self.n_days)][-5] = 0
                reward = -100000
        
        # if the action is 3 to sell low shares
        elif action == 3:
            # make sure you own the stock first
            if self.shares_owned >= 10:
                # sell shares -> the current day opening price
                self.final_cost = low_cost
                self.shares_owned -= 10
                #self.initial_cost = (share_price  * self.shares_owned) / self.num_trades
                self.current_balance += low_cost
                # calculate the profit or loss
                # calculate the reward
                #reward = self.final_cost - self.initial_cost
                share_value = self.shares_owned * share_price
                # current pl
                reward = (self.current_balance + share_value) - self.initial_balance
                self.current_pl = (self.current_balance + share_value) - self.initial_balance
                # calculate percent profit or loss
                reward_perc = ((reward / self.initial_balance) - 1.0) * 100
                # update the state to reflect the action taken
                self.state_data[-(self.current_step + self.n_days)][-2] = 1
                self.state_data[-(self.current_step + self.n_days)][-6] += low_cost
                self.state_data[-(self.current_step + self.n_days)][-7] = self.shares_owned
                # craft reward and info around positive and negative trades
                self.info['low_trade'] = self.num_low
                if reward > 0:
                    self.total_positive_trades += 1
                    self.info['positive_trade'] = self.total_positive_trades
                    self.info['positive_amount'] = reward
                    # winning trade aka episode -> could be renamed
                    self.win_trade += 1
                    self.win_trade_pl += reward
                    self.info['avg_win_trade'] = self.win_trade_pl / self.win_trade
                    self.info['low_win_perc'] = self.num_low / self.win_trade
                    reward = reward * (1.0 + reward_perc)
                elif reward < 0:
                    self.total_negative_trades += 1
                    self.info['negative_trade'] = self.total_negative_trades
                    self.info['negative_amount'] = reward
                    self.loss_trade += 1
                    self.loss_trade_pl += reward
                    self.info['avg_loss_trade'] = self.loss_trade_pl / self.loss_trade
                    self.info['low_loss_perc'] = self.num_low / self.loss_trade
                    # shape reward to downside
                    reward = reward * (1.0 + abs(reward_perc))
                # craft reward and info around positive and negative trades
                else:
                    reward = 0.0

            else:
                reward = -100000
                self.state_data[-(self.current_step+self.n_days)][-5] = 0
                print('invalid sell action')

        # if the action is 4 to sell high shares
        elif action == 4:
            # make sure you own the stock first
            if self.shares_owned >= 100:
                # sell shares -> the current day opening price
                self.final_cost = high_cost
                self.shares_owned -= 100
                self.info['high_trade'] = self.num_high
                self.current_balance += high_cost
                # calculate the reward
                #reward = self.final_cost - self.initial_cost
                share_value = self.shares_owned * share_price
                reward = (self.current_balance + share_value) - self.initial_balance
                self.current_pl = (self.current_balance + share_value) - self.initial_balance
                # calculate percent profit or loss
                reward_perc = ((reward / self.initial_balance - 1.0) * 100)
                # update the state to reflect the action taken
                self.state_data[-(self.current_step + self.n_days)][-1] = 1 
                self.state_data[-(self.current_step + self.n_days)][-6] += high_cost
                self.state_data[-(self.current_step + self.n_days)][-7] = self.shares_owned
                # craft reward and info around positive and negative trades
                if reward > 0:
                    self.total_positive_trades += 1
                    self.info['positive_trade'] = self.total_positive_trades
                    self.info['positive_amount'] = reward
                    # winning trade aka episode -> could be renamed
                    self.win_trade += 1
                    self.win_trade_pl += reward
                    self.info['avg_win_trade'] = self.win_trade_pl / self.win_trade
                    self.info['high_win_perc'] = self.num_high / self.win_trade
                    # add a multiple to the reward
                    #reward = reward * 5.0
                    reward = reward * (1.0 + reward_perc)
                elif reward < 0:
                    self.total_negative_trades += 1
                    self.info['negative_trade'] = self.total_negative_trades
                    self.info['negative_amount'] = reward
                    self.loss_trade += 1
                    self.loss_trade_pl += reward
                    self.info['avg_loss_trade'] = self.loss_trade_pl / self.loss_trade
                    self.info['high_loss_perc'] = self.num_high / self.loss_trade
                    reward = reward * (1.0 + 2*abs(reward_perc))
                # craft reward and info around positive and negative trades
                else:
                    reward = 0.0
            else:
                reward = -100000
                self.state_data[-(self.current_step+self.n_days)][-5] = 0
                print('invalid sell action')

        # if its the last step, sell all shares
        if done:
            if self.shares_owned > 0:
                # we are done, get the final cost of the stock and close out
                self.final_cost = self.current_data_window.iloc[-(self.current_step+self.n_days)]['adjusted_close']
                # sell all shares
                share_value = self.shares_owned * self.final_cost
                self.current_balance += share_value
                self.shares_owned = 0
                reward = self.current_balance - self.initial_balance
                self.current_pl = self.current_balance - self.initial_balance
                # calculate percent profit or loss
                reward_perc = ((reward / self.initial_balance - 1.0) * 100)
                # tell us about a positive trade
                if reward > 0:
                    self.total_positive_trades += 1
                    self.info['positive_trade'] = self.total_positive_trades
                    self.info['positive_amount'] = reward
                    # winning trade aka episode -> could be renamed
                    self.win_trade += 1
                    self.win_trade_pl += reward
                    self.info['avg_win_trade'] = self.win_trade_pl / self.win_trade
                    # shape reward to downside
                    reward = reward * (1 + reward_perc)
                elif reward < 0:
                    self.total_negative_trades += 1
                    self.info['negative_trade'] = self.total_negative_trades
                    self.info['negative_amount'] = reward
                    self.loss_trade += 1
                    self.loss_trade_pl += reward
                    self.info['avg_loss_trade'] = self.loss_trade_pl / self.loss_trade
                    # shape reward to downside
                    reward = reward * (1 + 2*abs(reward_perc))
                else:
                    reward = 0.0
            else:
                reward = 0.0
                self.current_pl = self.current_balance - self.initial_balance
                reward_perc = ((reward / self.initial_balance - 1.0) * 100)
            # extra negative reward
            if reward < 0:
                reward = reward * 1.5
            self.total_pl += self.current_balance - self.initial_balance
            self.info['reward_perc'] = reward_perc
            self.info['avg_reward_perc'] = reward_perc / max((self.win_trade + self.loss_trade), 1)
    
            # print episode pl and total pl
            # print win rate (positive trades / positive trades + negative trades)
            win_rate = self.total_positive_trades / max((self.total_positive_trades + self.total_negative_trades), 1)
            self.info['win_rate'] = win_rate
            self.info['total_pl'] = self.win_trade_pl - self.loss_trade_pl
            self.info['avg_all_pl'] = self.total_pl / max((self.win_trade + self.loss_trade), 1)
            self.symbols_pl[self.current_symbol] += self.total_pl
            if reward != 0.0:
                # cap to 2 decimal places
                print(f"Ep P/L: {self.current_balance-self.initial_balance:.2f}, WR: {win_rate:.2f}, WL P/L: {self.info.get('avg_win_trade',0)-self.info.get('avg_loss_trade',0):.2f}")
            if self.current_balance > self.max_balance:
                self.max_balance = self.current_balance
                print(f"******* NEW MAX BALANCE: {self.max_balance}, {self.current_symbol} ********")

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
    env = BenchmarkMultiEnv(csv_path=csv_path, stock_symbols=stock_symbols, initial_balance=500000)

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
        for ep in range(0, env.window):
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

