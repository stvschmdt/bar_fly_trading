import bisect
import pandas as pd
from collector import AlphaVantageClient


def find_nearest_strike(strike_prices: list[float], current_price: float, desired_percent_deviation: float):
    if desired_percent_deviation > 0:
        idx = bisect.bisect_right(strike_prices, current_price)
    else:
        idx = bisect.bisect_left(strike_prices, current_price) - 1

    percent_difference = 0
    adjustment = 1 if desired_percent_deviation > 0 else -1
    while 0 <= idx < len(strike_prices):
        new_percent_difference = (strike_prices[idx] - current_price) / current_price
        if abs(new_percent_difference - desired_percent_deviation) > abs(percent_difference - desired_percent_deviation):
            return strike_prices[idx - adjustment]

        percent_difference = new_percent_difference
        idx += adjustment

    return ValueError(f"No strike price with desired_percent_deviation {desired_percent_deviation} of current_price "
                      f"{current_price}. Strike prices between [{strike_prices[0]}, {strike_prices[-1]}].")


def get_expiration_date(expiration_dates: list[str], start_date: str, days_to_expiration: int):
    for date in expiration_dates:
        # If the date is at least days_to_expiration days after start_date, return it
        if (pd.Timestamp(date) - pd.Timestamp(start_date)).days >= days_to_expiration:
            return date
    return ValueError(f"No expiration date found {days_to_expiration} days after start_date {start_date}.")


def parse_historical_options_data(data: list[dict]):
    updated_data = []
    left = 0    # call
    right = 1   # put

    # Condense call and put data into a single dictionary, so it'll be one row in the df
    while right < len(data):
        new_data = data[right]
        new_data['call_bid'] = data[left]['bid']
        new_data['call_ask'] = data[left]['ask']
        new_data['put_bid'] = data[right]['bid']
        new_data['put_ask'] = data[right]['ask']
        updated_data.append(new_data)
        left += 2
        right += 2

    # Convert the updated list of dictionaries to a DataFrame
    df = pd.DataFrame(updated_data)
    df = df.filter(items=['date', 'expiration', 'call_bid', 'call_ask', 'put_bid', 'put_ask', 'strike'])

    # Convert numeric columns to actual numbers
    numeric_cols = ['strike', 'call_bid', 'call_ask', 'put_bid', 'put_ask']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Display the DataFrame
    print(df.head())
    return df


def fetch_historical_options_data(api_client: AlphaVantageClient, symbol: str, start_date: str):
    params = {
        'function': 'HISTORICAL_OPTIONS',
        'symbol': symbol,
        'date': start_date,
    }
    return api_client.fetch(**params)
