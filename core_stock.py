import pandas as pd

from collector import AlphaVantageClient
from storage import store_data


CORE_STOCK_TABLE_NAME = 'core_stock'
CORE_STOCK_COLUMNS = ['open', 'high', 'low', 'adjusted_close', 'volume']


def fetch_daily_adjusted_data(api_client: AlphaVantageClient, symbol: str, incremental: bool = True):
    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': symbol,
        'outputsize': 'compact' if incremental else 'full',
    }
    return api_client.fetch(**params)


def parse_daily_adjusted_data(data: dict):
    if 'Time Series (Daily)' not in data:
        raise ValueError("Unexpected response format: 'Time Series (Daily)' key not found")

    time_series = data['Time Series (Daily)']
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.index = pd.to_datetime(df.index)
    df.index.name = 'date'
    # Update column names to be everything after the first space, then replace the rest of the spaces with underscores
    df.columns = [col[col.index(' ')+1:].replace(' ', '_') for col in df.columns]
    # Filter down to only the columns we care about
    df = df.filter(items=CORE_STOCK_COLUMNS)
    df = df.apply(pd.to_numeric)
    return df


def update_core_stock_data(api_client: AlphaVantageClient, symbol: str, incremental: bool = True):
    response = fetch_daily_adjusted_data(api_client, symbol, incremental)
    core_stock_df = parse_daily_adjusted_data(response)
    core_stock_df['symbol'] = symbol
    print(f'{symbol} core data')
    print(core_stock_df.head())
    store_data(core_stock_df, table_name=CORE_STOCK_TABLE_NAME)
