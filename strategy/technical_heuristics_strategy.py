import os
import sys
from datetime import datetime

import pandas as pd

from order import Order, OrderOperation, StockOrder
from strategy.base_strategy import BaseStrategy

# print current working directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# No symbol should make up more than 30% of our account value
# We're starting out with MGM, FAS, NFLX

# Able to buy/sell if bull_bear_delta is >=2 (buy)

# Buy triggers:
# - Bollinger band was not 1 yesterday and is today
# - RSI was not 1 yesterday, and is today
# - MACD and SMA cross - one went from not 1 to 1, the other is 1

# Sell triggers:
# - We've made 10% profit or lost 20%
# - Bollinger band or RSI is -1
# - We've held it for 45 days

class TechnicalHeuristicsStrategy(BaseStrategy):
    def __init__(self, account, symbols):
        super().__init__(account, symbols)
        self.historical_data = pd.read_csv('all_data_0.csv')
        # Sort historical data from newest to oldest
        self.historical_data = self.historical_data.sort_values(by='date', ascending=False)
        # Remove any rows where symbol is not in symbols
        self.historical_data = self.historical_data[self.historical_data['symbol'].isin(symbols)]

    def evaluate(self, date: datetime.date, current_prices: pd.DataFrame) -> list[Order]:
        orders = []
        date = date.strftime('%Y-%m-%d')

        for symbol in self.symbols:
            # Get the historical data where the symbol column = symbol for today and last trading day
            rows = self.historical_data[(self.historical_data['symbol'] == symbol) & (self.historical_data['date'] <= date)]
            today = rows.iloc[0]
            yesterday = rows.iloc[1]

            current_price = float(current_prices.loc[current_prices['symbol'] == symbol, 'open'].iloc[0])

            if symbol in self.account.stock_positions:
                # Check sell criteria
                buys = self.account.order_history.get_filtered_order_history(symbol, order_types=[StockOrder], order_operation=OrderOperation.BUY)
                # Get most recent order
                most_recent_buy = None
                for _, order_id_dict in buys.items():
                    for _, order_date_dict in order_id_dict.items():
                        for order_date, order in order_date_dict.items():
                            if most_recent_buy is None or order.order_date > most_recent_buy.order_date:
                                most_recent_buy = order

                if (current_price >= most_recent_buy.entry_price * 1.1 or current_price <= most_recent_buy.entry_price * 0.8) \
                        or (today['bollinger_bands_signal'] == -1 or today['rsi_signal'] == -1) \
                        or (datetime.strptime(date, '%Y-%m-%d') - datetime.strptime(most_recent_buy.order_date, '%Y-%m-%d')).days >= 120:
                    # Sell
                    orders.append(StockOrder(symbol, OrderOperation.SELL, self.account.stock_positions[symbol], current_price, date))

                return orders

            if today['bull_bear_delta'] >= 2:
                if (today['bollinger_bands_signal'] == 1 and yesterday['bollinger_bands_signal'] != 1) \
                    or (today['rsi_signal'] == 1 and yesterday['rsi_signal'] != 1) \
                    or ((today['macd_signal'] == 1 and yesterday['macd_signal'] != 1) or (today['sma_cross_signal'] == 1 and yesterday['sma_cross_signal'] != 1)):
                    # Buy
                    shares_to_buy = self.account.get_max_buyable_shares(current_price, 0.3)
                    orders.append(StockOrder(symbol, OrderOperation.BUY, shares_to_buy, current_price, date))

        return orders
