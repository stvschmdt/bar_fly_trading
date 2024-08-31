from datetime import datetime

import pandas as pd

from order import Order, StockOrder, OrderOperation
from strategy.base_strategy import BaseStrategy


class BollingerBandsStrategy(BaseStrategy):
    def __init__(self, account, symbols):
        super().__init__(account, symbols)
        self.historical_data = pd.read_csv('all_data.csv')
        # Remove any columns where symbol is not in symbols
        self.historical_data = self.historical_data[self.historical_data['symbol'].isin(symbols)]

    def evaluate(self, date: datetime.date, current_prices: pd.DataFrame) -> list[Order]:
        orders = []
        date = date.strftime('%Y-%m-%d')
        for symbol in self.symbols:
            # Get the historical data where the symbol column = symbol and date column = date
            row = self.historical_data[(self.historical_data['symbol'] == symbol) & (self.historical_data['date'] == date)]
            # If the adjusted_close column in historical_data for this symbol on date is less than the bbands_lower_20 column, then buy
            if row['adjusted_close'].iloc[0] < row['bbands_lower_20'].iloc[0]:
                orders.append(StockOrder(symbol, OrderOperation.BUY, 1, float(current_prices.loc[current_prices['symbol'] == symbol, 'open'].iloc[0]), date))
            elif row['adjusted_close'].iloc[0] > row['bbands_upper_20'].iloc[0]:
                orders.append(StockOrder(symbol, OrderOperation.SELL, 1, float(current_prices.loc[current_prices['symbol'] == symbol, 'open'].iloc[0]), date))
        return orders

