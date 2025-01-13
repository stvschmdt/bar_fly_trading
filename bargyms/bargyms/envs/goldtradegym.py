import gymnasium as gym
import os
import logging
from gymnasium import spaces
import numpy as np
import pandas as pd
import random

class GoldTradeEnv(gym.Env):
    def __init__(self, csv_path=None, stock_symbols=None, render_mode=None, initial_balance=25000):
        super(GoldTradeEnv, self).__init__()
        if csv_path is None:
            csv_path = "../"  # Update with actual path to your CSV
        if stock_symbols is None:
            stock_symbols = ['NVDA', 'AVGO', 'NFLX', 'WMT', 'AMZN']
        self.render_mode = render_mode
        self.spy_symbol = 'SPY'  # SPY for benchmarking
        self.sector_symbol = 'QQQ'  # QQQ for benchmarking
        for file in os.listdir(csv_path):
            if file.startswith('all_data'):
                data = os.path.join('..', file)
                if not hasattr(self, 'data'):
                    # log reading data file
                    self.data = pd.read_csv(data)
                    logging.info(f'Reading data file: {data}')

                    # only keep data since a certain date
                    self.data = self.data[self.data['date'] >= '2016-01-01']
                else:
                    self.append_data = pd.read_csv(data)
                    logging.info(f'Reading data file: {data}')
                    # only keep data since a certain date
                    self.append_data = self.append_data[self.append_data['date'] >= '2016-01-01']
                    self.data = pd.concat([self.data, self.append_data])
        # cast date to datetime
        self.data['date'] = pd.to_datetime(self.data['date'])
        # ensure the data is sorted by date most recent first
        self.data = self.data.sort_values(by='date', ascending=False)
        # print how much data we have
        print(f"Data shape: {self.data.shape}")
        
        # Store the list of symbols to be used for training
        self.stock_symbols = stock_symbols
        self.symbols_map = {symbol: i for i, symbol in enumerate(self.stock_symbols)}
        self.symbols_pl = {symbol: 0.0 for i, symbol in enumerate(self.stock_symbols)}
        self.initial_balance = initial_balance
        self.balance = initial_balance
        
        # we give the agent n_days to learn the environment (window view)
        self.n_days = 5
        # we want to show the agent window amount of data to take an action
        self.window = 20

        # low number of discrete shares
        self.low_shares = 10
        # high number of discrete shares
        self.high_shares = 100
        
        self.cols = ['adjusted_open', 'adjusted_high', 'adjusted_low', 'adjusted_close', 'sma_20', 'sma_50', 'ema_20', 'ema_50', 'bbands_upper_20', 'bbands_lower_20', 'rsi_14', 'adx_14', 'atr_14', 'macd','pe_ratio', 'macd_signal', 'adx_signal', 'atr_signal', 'pe_ratio_signal', 'bollinger_bands_signal', 'rsi_signal', 'sma_cross_signal', 'bull_bear_delta', 'treasury_yield_2year', 'treasury_yield_10year', 'day_of_week_num', 'day_of_year', 'year']
        self.spy_cols = ['adjusted_open', 'adjusted_high', 'adjusted_low', 'adjusted_close', 'sma_20', 'sma_50', 'bbands_upper_20', 'bbands_lower_20', 'rsi_14', 'macd_signal', 'adx_signal',
       'atr_signal', 'pe_ratio_signal', 'bollinger_bands_signal', 'rsi_signal', 'sma_cross_signal', 'bull_bear_delta']
        self.sector_cols = ['adjusted_open', 'adjusted_high', 'adjusted_low', 'adjusted_close', 'sma_20', 'sma_50', 'bbands_upper_20', 'bbands_lower_20', 'rsi_14', 'macd_signal', 'adx_signal',
       'atr_signal', 'pe_ratio_signal', 'bollinger_bands_signal', 'rsi_signal', 'sma_cross_signal', 'bull_bear_delta']
        # Define observation and action spaces
        self.action_cols = 7
        # set enums for easier access to state space
        # account balance | shares owned | hold | low buy | high buy | sell
        # enums should be the corresponding index in the state space after cols, spy_cols, sector_cols
        self.account_balance_idx = len(self.cols) + len(self.spy_cols) + len(self.sector_cols) + 1 # plus one for symbol
        self.shares_owned_idx = self.account_balance_idx + 1
        self.hold_idx = self.shares_owned_idx + 1
        self.low_buy_idx = self.hold_idx + 1
        self.high_buy_idx = self.low_buy_idx + 1
        self.sell_idx = self.high_buy_idx + 1

        space_cols = len(self.cols) + len(self.spy_cols) + len(self.sector_cols) + self.action_cols
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_days, space_cols), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        
        
        # Initialize placeholders for overall stats
        self.max_balance = self.initial_balance
        self.current_episode = 0

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
            self.inference = True
            # we need to set the dates
            self.start_date = options.get('start_date', '2021-01-01')
            self.end_date = options.get('end_date', '2021-12-31')
            print('inference mode symbol', self.current_symbol)
            # print the start and end date
            print(self.start_date, self.end_date)
            # get the number of rows between the start and end date
            num_rows = len(symbol_data[(symbol_data['date'] >= self.start_date) & (symbol_data['date'] <= self.end_date)])
            print('num_rows', num_rows)
            # set window to the number of rows
            #self.window = num_rows - 1
            # get the index for the nearest date to the start date (if start date is not in the data get next earliest date)
            self.start_index_date = symbol_data[symbol_data['date'] >= self.start_date] # get the date before the start date
            # now get the latest date in the data
            self.start_index_date = self.start_index_date.iloc[-1]['date']
            #print('start date', self.start_index_date)
            # get the index of the current index
            self.start_index = symbol_data.index[symbol_data['date'] == self.start_index_date].tolist()[0]
            #print(self.start_index)
            # get the end date and index
            self.end_index_date = symbol_data[symbol_data['date'] <= self.end_date] # get the date before the end date
            # now get the latest date in the data
            self.end_index_date = self.end_index_date.iloc[0]['date']
            #print('end date', self.end_index_date)
            # get the index of the current index
            self.end_index = symbol_data.index[symbol_data['date'] == self.end_index_date].tolist()[0]
            # get the data for the selected date range, starting from the current index and adding 20 days
            #print(self.start_index, self.end_index)
            self.current_data_window = symbol_data.iloc[self.end_index : self.start_index + 1]
            # print the first row of data
            #print(self.current_data_window)
            #print("**************")
        else:
            self.inference = False
            # randomly select a start index, with a minimum of 10 days from the earliest date
            min_start_index = self.n_days + 30
            # max start date is 20 days from the end of the data
            max_start_index = len(symbol_data) - (self.n_days + self.window + 30)
            # assert that the max start index is greater than the min start index
            # select a random start date in the range of min, max
            self.current_index = random.randint(min_start_index, max_start_index)
            # print out the current index, min and max start index
            print(self.current_index, min_start_index, max_start_index)    
            # get the data for the selected date range, starting from the current index and adding 20 days
            self.current_data_window = symbol_data.iloc[self.current_index : self.current_index + (self.n_days + self.window + 1)]
            # get the date of the first day in the window
            self.start_index_date = self.current_data_window.iloc[-1]['date']
            # get the date of the last day in the window
            self.end_index_date = self.current_data_window.iloc[0]['date']
            # print start and end index dates
            #print(self.start_index_date, self.end_index_date)

        # print the -n_days date for testing
        #print(self.current_data_window.iloc[-self.n_days]['date'])
        # Get SPY data for the same date range
        self.spy_data = self.data[self.data['symbol'] == self.spy_symbol].reset_index(drop=True)
        # get the index of the self.index_date in the spy data        
        self.spy_start_index = self.spy_data.index[self.spy_data['date'] == self.start_index_date].tolist()[0]
        # get the index of the self.index_date in the spy data
        self.spy_end_index = self.spy_data.index[self.spy_data['date'] == self.end_index_date].tolist()[0]
        # get the spy data for the same date range
        #print(self.spy_start_index, self.spy_end_index)
        self.spy_data = self.spy_data.iloc[self.spy_end_index : self.spy_start_index + 1]
        #print the first day in spy data
        #print(self.spy_data.iloc[0]['date'])
        # print the -n_days date for testing
        #print(self.spy_data.iloc[-self.n_days]['date'])


        # Get sector data for the same date range
        self.sector_data = self.data[self.data['symbol'] == self.sector_symbol].reset_index(drop=True)
        # get the index of the self.index_date in the sector data        
        self.sector_start_index = self.sector_data.index[self.sector_data['date'] == self.start_index_date].tolist()[0]
        # get the index of the self.index_date in the sector data
        self.sector_end_index = self.sector_data.index[self.sector_data['date'] == self.end_index_date].tolist()[0]
        # get the sector data for the same date range
        #print(self.sector_start_index, self.sector_end_index)
        self.sector_data = self.sector_data.iloc[self.sector_end_index : self.sector_start_index + 1]
        # Reset account, holdings, and state
        self.current_balance = self.initial_balance
        self.initial_cost = 0
        self.final_cost = 0
        self.final_balance = 0
        self.current_pl = 0
        self.shares_owned = 0
        self.options_owned = 0
        self.current_step = 0
        self.enter_trade_step = -1
        self.exit_trade_step = -1
        self.num_trades = 0
        self.max_shares = 0
        self.illegal_trades = 0

        # get the max value of the symbol for the current episode
        self.max_value = self.current_data_window['adjusted_close'].max()
        # set high shares to nearest integer without going over the initial balance
        self.high_shares = int(self.initial_balance / self.max_value)
        # set low shares to 10% of high shares
        self.low_shares = int(self.high_shares * 0.1)
        
        # Set up the initial 10-day observation window
        self.state_data = self.current_data_window[self.cols].values
        # get the digit of the symbol we are using
        sym = np.full((len(self.state_data), 1), self.symbols_map[self.current_symbol])
        # horizontally stack the state data and the sym vector
        self.state_data = np.hstack((self.state_data, sym))
        # get the same columns for the spy data
        self.spy_data_arr = self.spy_data[self.spy_cols].values
        # get the columns for the sector data
        self.sector_data = self.sector_data[self.sector_cols].values
        # horizontally stack the state data and the sector data
        self.state_data = np.hstack((self.state_data, self.sector_data))
        # horizontally stack the state data and the spy data
        self.state_data = np.hstack((self.state_data, self.spy_data_arr))

        # account balance | shares owned | hold | low buy | high buy | sell
        
        # create an account vector same as action
        self.account = np.full((len(self.state_data), 1), self.initial_balance)
        # horizontally stack the state data and the account
        self.state_data = np.hstack((self.state_data, self.account))
        # the state shares is initialized to 0 for each of the length of self.current_data_window
        self.shares = np.full((len(self.state_data), 1), self.shares_owned)
        # horizontally stack the state data and the shares state
        self.state_data = np.hstack((self.state_data, self.shares))
        # the action is initialized to -1 for each of the length of self.current_data_window
        actions = np.full((len(self.state_data), 4), 0)
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

        # agent episode details for logging wrap all these in the info dict
        #self.info = {'trade_days': self.trade_days, 'max_profit': self.max_profit, 'agent_pl': self.agent_pl, 'agent_pl_perc': self.agent_pl_perc, 'spy_pl': self.spy_pl, 'spy_pl_perc': self.spy_pl_perc, 'reward':0, 'current_pl':0, 'end_pl':0}

        return np.round(self.state.astype(np.float32)[-self.n_days:], decimals=2), {}
        #return self.state.astype(np.float32)[0:10], {}

    # step and reward functions follow as previously outlined.
    def step(self, action):
        # track the current day
        self.current_day = self.current_data_window.iloc[-(self.current_step+self.n_days)]['date']
        if self.inference:
            # print the date and symbol
            print(self.current_data_window.iloc[-(self.current_step+self.n_days+1)]['date'], self.current_symbol)
            # if current date is the day before the end date, end the episode
            if self.current_data_window.iloc[-(self.current_step+self.n_days+1)]['date'] == self.end_index_date:
                done = True
            else:
                done = False
        else:
            # Check if episode is done...agent only gets so long to make a trade
            done = self.current_step >= self.window - 1

        # some accounting for ease
        share_price = self.current_data_window.iloc[-(self.current_step+self.n_days)]['adjusted_close']
        low_cost = self.current_data_window.iloc[-(self.current_step+self.n_days)]['adjusted_close'] * self.low_shares
        high_cost = self.current_data_window.iloc[-(self.current_step+self.n_days)]['adjusted_close'] * self.high_shares

        # if action is 0, do not buy or sell
        if action == 0:
            if self.shares_owned == 0:

                reward = -1.0 # risk free rate of return
            else:
                # reward is the daily gain or loss from previous day
                reward = share_price - self.current_data_window.iloc[-(self.current_step+self.n_days+1)]['adjusted_close']
                # take into account shares owned
                # synthetic theta decay of 10% of low costper day
                reward =  -0.1 * low_cost

            # change the state at this step to reflect the action taken
            self.state_data[-(self.current_step + self.n_days)][self.hold_idx] = 1

        # if the action is to buy low shares
        if action == 1:
            # make sure we have enough money to buy the shares
            if self.current_balance >= low_cost and self.shares_owned == 0:
                self.shares_owned += self.low_shares
                self.num_trades += 1
                self.trade_price = low_cost
                self.enter_trade_step = self.current_step
                self.current_balance -= low_cost
                # update the state shares owned
                self.state_data[-(self.current_step + self.n_days)][self.shares_owned_idx] = self.shares_owned
                # update the state account balance
                self.state_data[-(self.current_step + self.n_days)][self.account_balance_idx] = self.current_balance
                # update the state to reflect the low buy action
                self.state_data[-(self.current_step + self.n_days)][self.low_buy_idx] = 1
                # no reward for buying low shares
                reward = 0.0
            else:
                print('invalid buy action')
                # change the state for low buy attemtp to -1 for invalid
                self.state_data[-(self.current_step + self.n_days)][self.low_buy_idx] = -1
                reward = -10000 # punish agent for illegal trade
                self.illegal_trades = 1
                #done = True
                
        # take the action is to buy high shares
        elif action == 2:
            # make sure we have enough money to buy the shares
            if self.current_balance >= high_cost and self.shares_owned == 0:
                self.shares_owned += self.high_shares
                self.num_trades += 1
                self.trade_price = high_cost
                self.enter_trade_step = self.current_step
                self.current_balance -= high_cost
                # update the state shares owned
                self.state_data[-(self.current_step + self.n_days)][self.shares_owned_idx] = self.shares_owned
                # update the state account balance
                self.state_data[-(self.current_step + self.n_days)][self.account_balance_idx] = self.current_balance
                # update the state to reflect the high buy action
                self.state_data[-(self.current_step + self.n_days)][self.high_buy_idx] = 1
                reward = 0.0 # risk free rate of return
            else:
                print('invalid buy action')
                # change the state for high buy attemtp to -1 for invalid
                self.state_data[-(self.current_step + self.n_days)][self.high_buy_idx] = -1
                reward = -10000 # punish agent for illegal trade
                self.illegal_trades = 1
                #done = True
        
        # if the action is 3 to close out the trade
        elif action == 3:
            # make sure you own the stock first
            if self.shares_owned >= 0:
                self.max_shares = self.shares_owned
                # sell shares -> the current day opening price
                self.final_cost = share_price * self.shares_owned
                #self.initial_cost = (share_price  * self.shares_owned) / self.num_trades
                self.current_balance += self.final_cost
                self.exit_trade_step = self.current_step
                # number of days trade held for
                self.trade_days = self.exit_trade_step - self.enter_trade_step
                self.agent_pl = self.final_cost - self.trade_price
                self.shares_owned = 0
                # calculate the trade price gain or loss
                self.trade_pl_perc = ((self.final_cost - self.trade_price) / self.trade_price) * 100
                # current pl
                self.current_pl = self.current_balance - self.initial_balance
                # calculate percent profit or loss
                # update the state shares owned
                self.state_data[-(self.current_step + self.n_days)][self.shares_owned_idx] = self.shares_owned
                # update the state account balance
                self.state_data[-(self.current_step + self.n_days)][self.account_balance_idx] = self.current_balance
                # update the state to reflect the sell action
                self.state_data[-(self.current_step + self.n_days)][self.sell_idx] = 1
                # craft reward and info around positive and negative trades
                reward = self.final_cost - self.trade_price
                # round self.trade_pl_perc to nearest whole number
                reward = reward * (1.0 + round(self.trade_pl_perc))
                # extra positive reward if trade is positive and trade_days is less than 10
                if self.trade_pl_perc > 1.5 and self.trade_days < 10:
                    reward = reward * 3.0
                # extra positive reward if trade is positive and trade_days is less than 5
                if self.trade_pl_perc > .25 and self.trade_days < 5:
                    reward = reward * 1.5
                if self.trade_pl_perc > 0:
                    self.total_positive_trades += 1
                    self.info['positive_trade'] = self.total_positive_trades
                    self.info['positive_amount'] = reward
                    # winning trade aka episode -> could be renamed
                    self.win_trade += 1
                    self.win_trade_pl += self.trade_pl_perc
                    self.info['avg_win_trade'] = self.win_trade_pl / self.win_trade
                elif self.trade_pl_perc < 1.0:
                    reward = reward * 5.0
                    self.total_negative_trades += 1
                    self.info['negative_trade'] = self.total_negative_trades
                    self.info['negative_amount'] = reward
                    self.loss_trade += 1
                    self.loss_trade_pl += self.trade_pl_perc
                    self.info['avg_loss_trade'] = self.loss_trade_pl / self.loss_trade
                # craft reward and info around positive and negative trades
                else:
                    reward = reward * 1.1

            else:
                # update the state to reflect the sell action
                self.state_data[-(self.current_step + self.n_days)][self.sell_idx] = -1
                reward = -10000 # punish agent for illegal trade
                print('invalid sell action')
                self.illegal_trades = 1
            #if not self.inference:
            #    done = True
            #else:
                #done = False
                # reset the trade steps
                #self.enter_trade_step = -1
                #self.exit_trade_step = -1

        if self.inference:
            # print the date and symbol
            print(self.current_data_window.iloc[-(self.current_step+self.n_days+1)]['date'], self.current_symbol)
            # if current date is the day before the end date, end the episode
            if self.current_data_window.iloc[-(self.current_step+self.n_days+1)]['date'] == self.end_index_date:
                done = True

        # if its the last step, sell all shares
        if done:
            self.day_one_cost = self.current_data_window.iloc[-self.n_days]['adjusted_close']
            # find the max profit possible for the last n_days where a trade could have been made
            # max profit is calculated as max gain from any day in the window to any other day AFTER that day in the window
            # must check all possible entry days and exit days for largest spread in self.window

            max_profit = 0  # Initialize maximum profit

            # Iterate over potential buy points relative to -self.n_days
            for i in range(-self.n_days, -1):  # Start from -self.n_days, stop before the last index
                for j in range(i + 1, 0):  # Ensure sell (j) is after buy (i)
                    buy_price = self.current_data_window.iloc[i]['adjusted_close']
                    sell_price = self.current_data_window.iloc[j]['adjusted_close']
                    profit = sell_price - buy_price
                    max_profit = max(max_profit, profit)  # Update max profit if higher

            self.max_profit = max_profit

            # get the max adjusted_close over the last n_days
            self.max_entry = self.current_data_window.iloc[-(self.current_step+self.n_days):self.n_days]['adjusted_close'].max()
            # get the min adjusted_close over the last n_days
            self.first_entry = self.current_data_window.iloc[-self.n_days]['adjusted_close']
            # get the same for spy as we did for max_profit
            spy_max_profit = 0  # Initialize maximum profit
            # Iterate over potential buy points relative to -self.n_days
            for i in range(-self.n_days, -1):  # Start from -self.n_days, stop before the last index
                for j in range(i + 1, 0):
                    buy_price = self.spy_data.iloc[i]['adjusted_close']
                    sell_price = self.spy_data.iloc[j]['adjusted_close']
                    profit = sell_price - buy_price
                    spy_max_profit = max(spy_max_profit, profit)
            self.spy_max_profit = spy_max_profit
            self.spy_trade_entry = self.spy_data.iloc[-(self.enter_trade_step+self.n_days)]['adjusted_close']
            self.spy_trade_exit = self.spy_data.iloc[-(self.exit_trade_step+self.n_days)]['adjusted_close']
   
            # trading window is over, the agent should have closed out all trades
            if self.shares_owned > 0: # case where agent has shares left over
                self.max_shares = self.shares_owned
                # we are done, get the final cost of the stock and close out
                self.final_cost = self.current_data_window.iloc[-(self.current_step+self.n_days)]['adjusted_close']
                # sell all shares
                self.final_cost = self.shares_owned * self.final_cost
                self.current_balance += self.final_cost
                self.shares_owned = 0
                # number of days trade held for
                self.exit_trade_step = self.current_step
                self.trade_days = self.exit_trade_step - self.enter_trade_step
                self.current_pl = self.current_balance - self.initial_balance
                self.agent_pl = self.final_cost - self.trade_price
                self.trade_pl_perc = ((self.final_cost - self.trade_price) / self.trade_price) * 100
                # extra negative reward
                if self.agent_pl < 0:
                    reward = self.agent_pl
                    reward = self.agent_pl * 5.0
                    self.total_negative_trades += 1
                    self.info['negative_trade'] = self.total_negative_trades
                    self.info['negative_amount'] = self.current_pl
                    self.loss_trade += 1
                    self.loss_trade_pl += self.trade_pl_perc
                    self.info['avg_loss_trade'] = self.loss_trade_pl / self.loss_trade
                else: # agent was holding profit, but didnt close out in time
                    reward = self.agent_pl * .3
                    self.total_positive_trades += 1
                    self.info['positive_trade'] = self.total_positive_trades
                    self.info['positive_amount'] = self.current_pl
                    # winning trade aka episode -> could be renamed
                    self.win_trade += 1
                    self.win_trade_pl += self.trade_pl_perc
                    self.info['avg_win_trade'] = self.win_trade_pl / self.win_trade
            else:
                if self.enter_trade_step < 0: # no trades
                    # check if there was profit to be made
                    if self.max_profit > 500:
                        reward = -self.max_profit * self.low_shares
                    else:
                        reward = 100.0
                    self.trade_days = 0
                    self.agent_pl = 0
                    self.trade_pl_perc = 0
                else:
                    reward = 0.0
                self.current_pl = self.current_balance - self.initial_balance
            self.total_pl += self.current_balance - self.initial_balance
            self.info['reward_perc'] = ((self.current_pl-self.initial_balance) / self.initial_balance) * 100
            # print episode pl and total pl
            # print win rate (positive trades / positive trades + negative trades)
            if self.illegal_trades:
                reward = -10000
            win_rate = self.total_positive_trades / max((self.total_positive_trades + self.total_negative_trades), 1)
            self.info['win_rate'] = win_rate
            self.info['total_pl'] = self.win_trade_pl - self.loss_trade_pl
            self.info['avg_all_pl'] = self.total_pl / max((self.win_trade + self.loss_trade), 1)
            # add in all other info dict info
            self.info['current_pl'] = self.current_pl
            self.info['current_balance'] = self.current_balance
            self.info['trade_days'] = self.trade_days
            self.info['max_profit'] = self.max_profit * self.high_shares
            self.info['spy_max_profit'] = self.spy_max_profit * self.max_shares
            self.info['agent_pl'] = self.agent_pl
            self.info['agent_pl_perc'] = self.trade_pl_perc
            self.info['agent_trade_alpha'] = self.agent_pl - (self.spy_trade_entry * self.max_shares - self.spy_trade_exit * self.max_shares)
            self.info['agent_trade_alpha_avg'] = self.info['agent_trade_alpha'] / max(self.num_trades, 1)
            self.info['agent_possible_alpha'] = self.max_profit * self.max_shares - self.spy_max_profit * self.max_shares
            self.info['agent_alpha_avg'] = self.info['agent_possible_alpha'] / (self.current_episode)

            self.symbols_pl[self.current_symbol] += self.total_pl
            if reward != 0.0 and reward > -1000:
                # cap to 2 decimal places
                print(f"Ep P/L: {self.current_balance-self.initial_balance:.2f}, WR: {win_rate:.2f}, Agent P/L: {self.agent_pl:.2f}, Agent P/L %: {self.trade_pl_perc:.2f}, Max PL: {self.max_profit:.2f}, Spy Max PL: {self.spy_max_profit:.2f}, Reward: {reward:.2f}")
            if self.current_balance > self.max_balance:
                self.max_balance = self.current_balance
                print(f"******* NEW MAX BALANCE: {self.max_balance}, {self.current_symbol} ********")
        if self.illegal_trades:
            reward = -1000
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
    csv_path = "../"  # Update with actual path to your CSV
    stock_symbols = ["AAPL", "NVDA", "AMD"]  # Example stock symbols
    stock_symbols = ["AAPL"]
    env = GoldTradeEnv(csv_path=csv_path, stock_symbols=stock_symbols, initial_balance=500000)

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

