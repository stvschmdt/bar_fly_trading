from datetime import datetime

import pandas as pd

from core_stock import fetch_daily_adjusted_data, parse_daily_adjusted_data
from collector import alpha_client
from simulations.spreads import bull_call_spread
from simulations.spreads.util import parse_historical_options_data, fetch_historical_options_data, get_expiration_date


# Since options only expire on Fridays, we'll find the nearest Friday to start_date + APPROXIMATE_TIME_TO_EXPIRATION
APPROXIMATE_TIME_TO_EXPIRATION = 30
SYMBOLS = ['AAPL']
START_DATE = '2024-01-02'
END_DATE = '2024-02-31'
# END_DATE = datetime.today().strftime('%Y-%m-%d')


def main():
    for symbol in SYMBOLS:
        # Fetch historical stock price data
        try:
            prices_data = fetch_daily_adjusted_data(alpha_client, symbol, False)
            prices_df = parse_daily_adjusted_data(prices_data)
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return

        current_date = START_DATE

        # Fetch options data for current date
        try:
            # TODO: Need to handle weekends and holidays. The error will contain
            # `Please specify a valid combination of symbol and trading day` so retry next day when that happens.
            options_data = fetch_historical_options_data(alpha_client, symbol, current_date)
            df = parse_historical_options_data(options_data['data'])
        except Exception as e:
            print(f"Error fetching/parsing historical options data: {e}")
            return

        expiration_dates = df['expiration'].drop_duplicates().sort_values().tolist()
        expiration_date = get_expiration_date(expiration_dates, START_DATE, APPROXIMATE_TIME_TO_EXPIRATION)

        total_profit = 0
        # while expiration_date < END_DATE:
            # Calculate the bull call spread
        try:
            expiration_date_df = df[df['expiration'] == expiration_date]
            profit = bull_call_spread.calculate(expiration_date_df, prices_df, start_date=current_date, end_date=expiration_date)
        except Exception as e:
            print(f"Error calculating bull call spread: {e}")
            return

        total_profit += profit
            # current_date = expiration_date
            # print(f"Profit for {symbol} from {current_date} to {END_DATE}: ${round(profit, 2)}")

        print(f"Profit for {symbol} from {START_DATE} to {END_DATE}: ${round(total_profit, 2)}")


if __name__ == "__main__":
    main()
