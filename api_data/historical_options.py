import logging

import pandas as pd

from api_data.collector import AlphaVantageClient
from api_data.storage import store_data, get_dates_for_symbol
from api_data.util import graceful_df_to_numeric, get_table_write_option, drop_existing_rows
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

HISTORICAL_OPTIONS_TABLE_NAME = 'historical_options'
DATE_COL = 'date'


def fetch_historical_options(api_client: AlphaVantageClient, symbol: str, date: str) -> dict:
    # If you don't give a date, it will return the most recent data.
    params = {
        'function': 'HISTORICAL_OPTIONS',
        'symbol': symbol,
    }
    if date:
        params['date'] = date
    return api_client.fetch(**params)


def parse_historical_options(data: list[dict]) -> pd.DataFrame:
    # Convert the updated list of dictionaries to a DataFrame
    df = pd.DataFrame(data)
    df = df.filter(items=['contractID', 'date', 'symbol', 'type', 'expiration', 'strike', 'last', 'mark', 'bid', 'ask', 'volume', 'implied_volatility', 'delta', 'gamma', 'theta', 'vega', 'rho'])
    df = df.rename(columns={'contractID': 'contract_id'})
    # Convert numeric columns to actual numbers
    df = graceful_df_to_numeric(df)

    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index)
    return df


# No incremental option for historical options data because it would take a long time to repopulate the entire table.
# We'll always operate incrementally, and if you want to refresh the entire table, you can do so manually.
def update_historical_options(api_client: AlphaVantageClient, symbol: str, start_date: str = None, end_date: str = None):
    # If no start date was given, just fetch the most recent data.
    if not start_date:
        update_historical_options_for_date(api_client, symbol, True)
        return

    # Default end_date to today if not provided
    end_date = end_date or pd.Timestamp.today().strftime('%Y-%m-%d')
    existing_dates = set(get_dates_for_symbol(symbol, HISTORICAL_OPTIONS_TABLE_NAME, DATE_COL, start_date, end_date))
    for date in pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y-%m-%d'):
        if date in existing_dates:
            logger.info(f'{symbol} historical options data already exists for {date}, skipping')
            continue

        # We don't need to drop existing data because we're already skipping dates that are in the table.
        update_historical_options_for_date(api_client, symbol, False, date)


def update_historical_options_for_date(api_client: AlphaVantageClient, symbol: str, drop_existing_data: bool, date: str = None):
    response = fetch_historical_options(api_client, symbol, date)
    if not response['data']:
        # Since we default end_date to today, we may not have data for today yet.
        logger.info(f'No historical options data found for {symbol}')
        return

    df = parse_historical_options(response['data'])
    if drop_existing_data:
        df = drop_existing_rows(df, HISTORICAL_OPTIONS_TABLE_NAME, DATE_COL, symbol)
    store_data(df, table_name=HISTORICAL_OPTIONS_TABLE_NAME, write_option=get_table_write_option(True), include_index=True)
