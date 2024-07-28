import pandas as pd

from collector import AlphaVantageClient
from storage import store_data
from util import drop_existing_rows, get_last_updated_date, get_table_write_option


CORE_STOCK_TABLE_NAME = 'core_stock'
CORE_STOCK_COLUMNS = ['open', 'high', 'low', 'adjusted_close', 'volume']
DATE_COL = 'date'


def fetch_daily_adjusted_data(api_client: AlphaVantageClient, symbol: str, fetch_compact_data: bool = True):
    params = {
        'function': 'TIME_SERIES_DAILY_ADJUSTED',
        'symbol': symbol,
        'outputsize': 'compact' if fetch_compact_data else 'full',
    }
    return api_client.fetch(**params)


def parse_daily_adjusted_data(data: dict):
    if 'Time Series (Daily)' not in data:
        raise ValueError("Unexpected response format: 'Time Series (Daily)' key not found")

    time_series = data['Time Series (Daily)']
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.index = pd.to_datetime(df.index)
    df.index.name = DATE_COL
    # Update column names to be everything after the first space, then replace the rest of the spaces with underscores
    df.columns = [col[col.index(' ')+1:].replace(' ', '_') for col in df.columns]
    # Filter down to only the columns we care about
    df = df.filter(items=CORE_STOCK_COLUMNS)
    # Every value in the table is a number, so we can convert the whole table to numeric
    df = df.apply(pd.to_numeric)
    return df


def update_core_stock_data(api_client: AlphaVantageClient, symbol: str, incremental: bool = True):
    fetch_compact_data = False
    if incremental:
        last_updated_date = get_last_updated_date(CORE_STOCK_TABLE_NAME, DATE_COL, symbol)
        # If the last_updated_date is not None and is within the last 100 days, fetch compact data (just last 100 days)
        fetch_compact_data = last_updated_date and (pd.Timestamp.now() - last_updated_date).days < 100

    response = fetch_daily_adjusted_data(api_client, symbol, fetch_compact_data)
    core_stock_df = parse_daily_adjusted_data(response)
    core_stock_df['symbol'] = symbol

    if incremental:
        core_stock_df = drop_existing_rows(core_stock_df, CORE_STOCK_TABLE_NAME, DATE_COL, symbol)

    print(f'{symbol} core data')
    print(core_stock_df.head())
    store_data(core_stock_df, table_name=CORE_STOCK_TABLE_NAME, write_option=get_table_write_option(incremental))
