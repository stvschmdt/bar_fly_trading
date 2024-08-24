from datetime import datetime

import pandas as pd

from position import Position, StockPosition, OrderOperation
from strategy.base_strategy import BaseStrategy


class TestStrategy(BaseStrategy):
    def evaluate(self, date: datetime.date, current_prices: pd.DataFrame) -> list[Position]:
        # Buy 1 share of each stock in the watchlist
        positions = []
        for symbol in self.symbols:
            positions.append(StockPosition(symbol, OrderOperation.BUY, 1, current_prices.loc[current_prices['symbol'] == symbol, 'open']))
        return positions
