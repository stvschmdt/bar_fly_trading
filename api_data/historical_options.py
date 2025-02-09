import logging

import pandas as pd
from sqlalchemy.exc import ProgrammingError

from api_data.collector import AlphaVantageClient
from api_data.core_stock import CORE_STOCK_TABLE_NAME
from api_data.storage import store_data, get_dates_for_symbol, select_all_by_symbol
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
    # Get dates in DB that already have historical options data
    existing_dates = set(get_dates_for_symbol(symbol, HISTORICAL_OPTIONS_TABLE_NAME, DATE_COL, start_date, end_date))
    # Get core stock data for each date in the range for close prices, so we can trim down the strike prices we store.
    close_prices = select_all_by_symbol(CORE_STOCK_TABLE_NAME, symbols={symbol}, order_by='date ASC', start_date=start_date, end_date=end_date)[['date', 'close']]
    for date in pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y-%m-%d'):
        if date in existing_dates:
            logger.info(f'{symbol} historical options data already exists for {date}, skipping')
            continue

        # Get close price on this date
        stock_price = close_prices[close_prices['date'] == date]['close'].values[0]

        # We don't need to drop existing data because we're already skipping dates that are in the table.
        update_historical_options_for_date(api_client, symbol, False, date, stock_price)


def update_historical_options_for_date(api_client: AlphaVantageClient, symbol: str, drop_existing_data: bool,
                                       num_strikes_on_each_side: int = 10, num_expirations: int = 10,
                                       date: str = None, close_price: float = None):
    if close_price and not date:
        raise ValueError('A close price was provided without a date')

    response = fetch_historical_options(api_client, symbol, date)
    if not response['data']:
        # Since we default end_date to today, we may not have data for today yet.
        logger.info(f'No historical options data found for {symbol}')
        return

    df = parse_historical_options(response['data'])

    # If we weren't given a date, fetch it from the API response. The date will be the same in all entries because it's
    # the last date the market was open, so we can just grab the first one.
    if not close_price:
        date = df.index[0]
        close_price = select_all_by_symbol(CORE_STOCK_TABLE_NAME, symbols={symbol}, start_date=date)['close'].values[0]

    # Get the nearby strike prices
    strikes = get_nearby_strikes(df['strike'].unique(), close_price, num_strikes_on_each_side)

    # Filter down to just data for the nearby strikes
    df = df[df['strike'].isin(strikes)]

    # Filter down to the first num_expirations expirations
    expirations = df['expiration'].unique()[:num_expirations]
    df = df[df['expiration'].isin(expirations)]

    if drop_existing_data:
        try:
            df = drop_existing_rows(df, HISTORICAL_OPTIONS_TABLE_NAME, DATE_COL, symbol)
        # Catch a SQL exception for when the table doesn't exist yet. That just means we don't need to drop anything.
        except ProgrammingError as e:
            logger.error(f'Not dropping historical options data because table doesn\'t exist: {e}')
    store_data(df, table_name=HISTORICAL_OPTIONS_TABLE_NAME, write_option=get_table_write_option(True), include_index=True)


def get_nearby_strikes(strike_prices, stock_price, num_strikes_on_each_side):
    """
    Returns a sorted list of strike prices with given number either side of the current stock price,
    including the closest strike to the stock price in the middle.
    """
    # Sort the strike prices
    sorted_strikes = sorted(strike_prices)

    # Find the index of the closest strike price
    closest_index = min(range(len(sorted_strikes)), key=lambda i: abs(sorted_strikes[i] - stock_price))

    # Get the range of strikes around the closest strike
    start = max(0, closest_index - num_strikes_on_each_side)
    end = min(len(sorted_strikes), closest_index + num_strikes_on_each_side + 1)

    return sorted_strikes[start:end]
