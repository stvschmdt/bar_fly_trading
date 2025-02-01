import pandas as pd

from api_data.collector import AlphaVantageClient


# This is just for testing/simulation purposes.
# To fetch historical options data that we store in the DB, use api_data/historical_options.py.


def fetch_historical_options_data(api_client: AlphaVantageClient, symbol: str, start_date: str) -> dict:
    params = {
        'function': 'HISTORICAL_OPTIONS',
        'symbol': symbol,
        'date': start_date,
    }
    return api_client.fetch(**params)


def parse_historical_options_data(data: list[dict]) -> pd.DataFrame:
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
    df = df.filter(items=['contractID', 'date', 'expiration', 'call_bid', 'call_ask', 'put_bid', 'put_ask', 'strike'])

    # Convert numeric columns to actual numbers
    numeric_cols = ['strike', 'call_bid', 'call_ask', 'put_bid', 'put_ask']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    return df


def get_option_data(api_client: AlphaVantageClient, symbol: str, start_date: str) -> pd.DataFrame:
    response = fetch_historical_options_data(api_client, symbol, start_date)
    return parse_historical_options_data(response['data'])
    # TODO: Write this data to DB

