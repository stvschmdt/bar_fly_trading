from datetime import datetime

import pandas as pd

from order import Order, StockOrder, OrderOperation
from strategy.base_strategy import BaseStrategy


class TechnicalStrategy(BaseStrategy):
    def __init__(self, account, symbols):
        super().__init__(account, symbols)
        self.historical_data = pd.read_csv('api_data/all_data.csv')
        # Remove any columns where symbol is not in symbols
        self.historical_data = self.historical_data[self.historical_data['symbol'].isin(symbols)]

    def evaluate(self, date: datetime.date, current_prices: pd.DataFrame, options_data: pd.DataFrame) -> list[Order]:
        orders = []
        date = date.strftime('%Y-%m-%d')

        # evenly distribute the cash among the stocks
        percent = 1.0 / len(self.symbols)
        
        for symbol in self.symbols:
            row = self.historical_data[(self.historical_data['symbol'] == symbol) & (self.historical_data['date'] == date)]
            current_price = float(current_prices.loc[current_prices['symbol'] == symbol, 'open'].iloc[0])

            # Extract the relevant data for the current row
            close = row['adjusted_close'].iloc[0]
            sma_20 = row['sma_20'].iloc[0]
            sma_50 = row['sma_50'].iloc[0]
            sma_200 = row['sma_200'].iloc[0]
            ema_20 = row['ema_20'].iloc[0]  
            ema_50 = row['ema_50'].iloc[0]
            ema_200 = row['ema_200'].iloc[0]
            macd = row['macd'].iloc[0]
            rsi = row['rsi_14'].iloc[0]
            bb_upper = row['bbands_upper_20'].iloc[0]   
            bb_lower = row['bbands_lower_20'].iloc[0]
            pe_ratio = row['pe_ratio'].iloc[0]
            # Bullish signals
            if (rsi < 30 and close < bb_lower) or \
               close > ema_20 > ema_50 or \
               close > sma_20 > sma_50 > sma_200 or \
               (macd > 0 and ema_20 > ema_50):
               # buy max shares account will offer
               shares_to_buy = self.account.get_max_buyable_shares(current_price, percent)
               orders.append(StockOrder(symbol, OrderOperation.BUY, shares_to_buy, current_price, date))
               
            # Bearish signals
            if (rsi > 70 and close > bb_upper) or \
               close < ema_20 < ema_50 or \
               close < sma_20 < sma_50 < sma_200 or \
               (macd < 0 and ema_20 < ema_50):
               if self.account.stock_positions.get(symbol, 0) > 0:
                   orders.append(StockOrder(symbol, OrderOperation.SELL, self.account.stock_positions.get(symbol, 0), current_price, date))
              
        return orders

