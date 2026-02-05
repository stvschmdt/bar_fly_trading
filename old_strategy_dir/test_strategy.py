from datetime import datetime

import pandas as pd

from order import Order, StockOrder, OrderOperation
from strategy.base_strategy import BaseStrategy


class TestStrategy(BaseStrategy):
    def evaluate(self, date: datetime.date, current_prices: pd.DataFrame, options_data: pd.DataFrame) -> list[Order]:
        # Buy 1 share of each stock in self.symbols
        orders = []
        for symbol in self.symbols:
            orders.append(StockOrder(symbol, OrderOperation.BUY, 1, current_prices.loc[current_prices['symbol'] == symbol, 'open'].iloc[0], date))
        return orders
