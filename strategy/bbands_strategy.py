from datetime import datetime

import pandas as pd

from position import Position, StockPosition, OrderOperation
from strategy.base_strategy import BaseStrategy


class BBandsStrategy(BaseStrategy):
    def evaluate(self, date: datetime.date, current_prices: pd.DataFrame) -> list[Position]:
        print(current_prices.columns)
        # Buy 1 share of each stock in the watchlist
        positions = []
        for symbol in self.symbols:
            # check if open is below bband_lower_20
            if current_prices.loc[current_prices['symbol'] == symbol, 'open'] < current_prices.loc[current_prices['symbol'] == symbol, 'bband_lower_20']:
                positions.append(StockPosition(symbol, OrderOperation.BUY, 1, current_prices.loc[current_prices['symbol'] == symbol, 'open']))
            # else if open is above bband_upper_20
            elif current_prices.loc[current_prices['symbol'] == symbol, 'open'] > current_prices.loc[current_prices['symbol'] == symbol, 'bband_upper_20']:
                positions.append(StockPosition(symbol, OrderOperation.SELL, 1, current_prices.loc[current_prices['symbol'] == symbol, 'open']))
            # otherwise just pass
            else:
                pass
        return positions
