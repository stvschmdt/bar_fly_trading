from collections import defaultdict
from datetime import datetime, timedelta

from api_data.core_stock import fetch_daily_adjusted_data, parse_daily_adjusted_data
from simulations.historical_option_data import get_option_data
from simulations.spreads.util import find_nearest_strike, get_expiration_date, MarketClosedError

BUY_CALL_PERCENT_ABOVE_START = 0.05     # 5% above the start price
SELL_CALL_PERCENT_ABOVE_START = 0.12    # 12% above the start price
NO_OPTION_DATA_ERROR_MESSAGE = 'Please specify a valid combination of symbol and trading day'


# This simulates a bull call spread strategy, where we buy a call option and sell a call option with a higher strike price.
# Right now, we're just assuming we're holding the options until expiration, and we're calculating the profit on 1 contract.
class BullCallSpread:
    def __init__(self, alpha_client, symbol: str, option_window: int, start_date: str, end_date: str):
        self.alpha_client = alpha_client
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        # Approximate number of days in advance to buy options
        # We'll choose the closest expiration date available
        self.option_window = option_window

    def calculate(self):
        results = defaultdict(list)

        # This fetches all stock price data in history for the symbol, so we only need to do it once.
        stock_price_df = self.fetch_stock_price_data()

        profit = 0
        current_date = datetime.strptime(self.start_date, '%Y-%m-%d')

        # Just assuming for now that we're holding options for the whole period,
        # so start_date + option_window must be < end_date
        if current_date + timedelta(days=self.option_window) > datetime.strptime(self.end_date, '%Y-%m-%d'):
            return profit

        # Iterate until the current date + option_window is past the end date
        while current_date <= datetime.strptime(self.end_date, '%Y-%m-%d'):
            print(f'Iterating on date: {current_date.strftime("%Y-%m-%d")}')
            try:
                expiration_date, options_df = self.fetch_option_data(current_date.strftime('%Y-%m-%d'))
            except MarketClosedError:
                # If the market is closed, skip to the next day
                current_date += timedelta(days=1)
                continue
            except Exception as e:
                print(f"Error fetching options data for date {current_date.strftime('%Y-%m-%d')}: {e}")
                return

            # Again, assuming we're holding until the expiration date of an option for now
            if datetime.strptime(expiration_date, '%Y-%m-%d') > datetime.strptime(self.end_date, '%Y-%m-%d'):
                break

            start_price = stock_price_df.loc[current_date]['adjusted_close']
            end_price = stock_price_df.loc[expiration_date]['adjusted_close']
            buy_strike_price = find_nearest_strike(options_df['strike'].tolist(), start_price, BUY_CALL_PERCENT_ABOVE_START)
            sell_strike_price = find_nearest_strike(options_df['strike'].tolist(), start_price, SELL_CALL_PERCENT_ABOVE_START)
            buy_option_price = options_df[options_df['strike'] == buy_strike_price]['call_ask'].iloc[0]
            sell_option_price = options_df[options_df['strike'] == sell_strike_price]['call_bid'].iloc[0]

            # Calculate the profit/loss
            buy_profit = (max(0, end_price - buy_strike_price) - buy_option_price) * 100
            sell_profit = (sell_option_price - max(0, end_price - sell_strike_price)) * 100
            transaction_profit = buy_profit + sell_profit
            profit += transaction_profit

            results['dates'].append(current_date)
            results['buys'].append(buy_option_price)
            results['sells'].append(sell_option_price)
            results['incremental_profits'].append(transaction_profit)
            results['cumulative_profits'].append(profit)

            print(f'Buy profit: {buy_profit}')
            print(f'Sell profit: {sell_profit}')
            print(f'Transaction profit: {transaction_profit}')
            print(f'Total profit: {profit}')
            current_date = datetime.strptime(expiration_date, '%Y-%m-%d') + timedelta(days=1)
        return profit, results

    def fetch_stock_price_data(self):
        try:
            prices_data = fetch_daily_adjusted_data(self.alpha_client, self.symbol, False)
            prices_df = parse_daily_adjusted_data(prices_data)
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return

        return prices_df

    def fetch_option_data(self, current_date: str):
        # Fetch options data for current date
        options_data = {}
        try:
            df = get_option_data(self.alpha_client, self.symbol, current_date)
        except Exception as e:
            if NO_OPTION_DATA_ERROR_MESSAGE in options_data.get('message', ''):
                raise MarketClosedError()
            raise e

        expiration_dates = df['expiration'].drop_duplicates().sort_values().tolist()
        expiration_date = get_expiration_date(expiration_dates, current_date, self.option_window)

        return expiration_date, df[df['expiration'] == expiration_date]
