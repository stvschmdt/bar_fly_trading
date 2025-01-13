import os
import sys
from datetime import datetime

import pandas as pd

from order import Order, StockOrder, OrderOperation
from strategy.base_strategy import BaseStrategy

# print current working directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class BollingerBandsStrategy(BaseStrategy):
    def __init__(self, account, symbols):
        super().__init__(account, symbols)
        self.historical_data = pd.read_csv('api_data/all_data.csv')
        # Remove any rows where symbol is not in symbols
        self.historical_data = self.historical_data[self.historical_data['symbol'].isin(symbols)]

    def evaluate(self, date: datetime.date, current_prices: pd.DataFrame) -> list[Order]:
        orders = []
        date = date.strftime('%Y-%m-%d')
        # evenly distribute the cash among the stocks
        percent = 1.0 / len(self.symbols)
        for symbol in self.symbols:
            # Get the historical data where the symbol column = symbol and date column = date
            row = self.historical_data[(self.historical_data['symbol'] == symbol) & (self.historical_data['date'] == date)]
            current_price = float(current_prices.loc[current_prices['symbol'] == symbol, 'open'].iloc[0])
            # If the adjusted_close column in historical_data for this symbol on date is less than the bbands_lower_20 column, then buy
            if row['adjusted_close'].iloc[0] < row['bbands_lower_20'].iloc[0]:
                # buy max shares account will offer
                shares_to_buy = self.account.get_max_buyable_shares(current_price, percent)
                orders.append(StockOrder(symbol, OrderOperation.BUY, shares_to_buy, current_price, date))
            elif row['adjusted_close'].iloc[0] > row['bbands_upper_20'].iloc[0]:
                # check to make sure we own shares first
                if self.account.stock_positions.get(symbol, 0) > 0:
                    orders.append(StockOrder(symbol, OrderOperation.SELL, self.account.stock_positions.get(symbol, 0), current_price, date))
        return orders

