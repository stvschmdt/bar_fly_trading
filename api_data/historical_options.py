import os
import pandas as pd
from datetime import datetime, timedelta
import time

from sqlalchemy import create_engine
from api_data.collector import AlphaVantageClient
from api_data.storage import store_data
from api_data.util import drop_existing_rows, get_last_updated_date, get_table_write_option
import logging

logger = logging.getLogger(__name__)

HISTORICAL_OPTIONS_TABLE_NAME = 'historical_options'
HISTORICAL_OPTIONS_COLUMNS = ['contractID', 'date', 'expiration', 'call_bid', 'call_ask', 'put_bid', 'put_ask', 'strike']
DATE_COL = 'date'

# Define the database connection parameters to query the core_stock table for adjusted_close price
username = 'root'
password = os.getenv('MYSQL_PASSWORD', None)
host = '127.0.0.1'
port = 3306
dbname = 'bar_fly_trading'

# Create the connection string
connection_string = f'mysql+pymysql://{username}:{password}@{host}:{port}/{dbname}'

# Create the SQLAlchemy engine
engine = create_engine(connection_string)


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


def get_close_price(symbol, date):
    query = f"""
        SELECT core.adjusted_close
        FROM core_stock as core
        WHERE core.symbol = %s AND core.date = %s
    """
    df = pd.read_sql_query(query, engine, params=(symbol, date))
    price = df['adjusted_close'].values[0]
    return price


def filter_contracts_by_strike_and_expiration(df, given_date, given_price, lookahead_days=365, nn=10):
    """
    Filter options contracts by strike price and expiration date
    :param df: contracts returned from get_option_data
    :param given_date: contract date
    :param given_price: close price of the stock for the day
    :param lookahead_days: window of time for contracts to expire
    :param nn: number of contracts to return with strikes above and below the close price
    :return:
    """
    # Step 1: Convert given_date to datetime and find the one-year-ahead date
    given_date = pd.to_datetime(given_date)
    one_year_later = given_date + timedelta(days=lookahead_days)

    # Step 2: Filter for expiration dates within the next year
    df.loc[:, 'expiration'] = pd.to_datetime(df['expiration'])
    df_filtered = df[(df['expiration'] >= given_date) & (df['expiration'] <= one_year_later)]

    # Step 3: For each expiration date, find the 10 nearest neighbors below and above the given price
    result = pd.DataFrame()  # Empty dataframe to hold the results
    for exp_date in df_filtered['expiration'].unique():
        # Get the subset of the dataframe for this expiration date
        df_exp = df_filtered[df_filtered['expiration'] == exp_date]

        # Step 4: Split into two groups - below and above the given price
        below_price = df_exp[df_exp['strike'] < given_price].sort_values(by='strike', ascending=False).head(nn)
        above_price = df_exp[df_exp['strike'] > given_price].sort_values(by='strike', ascending=True).head(nn)

        # Combine the two
        combined = pd.concat([below_price, above_price])

        # Add to the result dataframe
        result = pd.concat([result, combined])

    # Step 5: Sort by expiration_date and strike_price for clarity
    result = result.sort_values(by=['expiration', 'strike'])

    return result


def update_all_historical_options(api_client: AlphaVantageClient, symbol: str, start_date: str, incremental: bool = True):
    if incremental:
        try:  # Try/Except to handle if table doesn't exist
            last_updated = get_last_updated_date(HISTORICAL_OPTIONS_TABLE_NAME, DATE_COL, symbol)
            if last_updated is not None:  # If the table exists and the symbol is present, update from the last date
                start_date = last_updated.strftime('%Y-%m-%d')
        except Exception as e:
            pass

    timer = 0  # Counter for API hits
    for date in pd.date_range(start_date, datetime.today()).strftime('%Y-%m-%d'):
        start_time = time.time()
        timer += 1
        print(f"Proc Hist Opts: {symbol} - {date}")
        response = fetch_historical_options_data(api_client, symbol, date)

        if not response['data']:  # Data will be missing if it's not a trading day, so move on to next day
            continue

        df_parsed = parse_historical_options_data(response['data'])
        close_price = get_close_price(symbol, date)
        df_filtered = filter_contracts_by_strike_and_expiration(df_parsed, date, close_price, lookahead_days=365)

        if df_filtered is not None:
            df_filtered['symbol'] = symbol
            if 'historical_options_df' not in locals():
                historical_options_df = df_filtered
            else:
                historical_options_df = pd.concat([historical_options_df, df_filtered], ignore_index=True)

        print("--- %s seconds ---" % (time.time() - start_time))

        # After 20 API hits, sleep for 60 seconds to avoid hitting the API limit
        if timer >= 20:
            time.sleep(60)
            timer = 0

    if incremental:
        historical_options_df = drop_existing_rows(historical_options_df, HISTORICAL_OPTIONS_TABLE_NAME, DATE_COL, symbol)

    logger.info(f'{symbol} historical options')
    store_data(historical_options_df, table_name=HISTORICAL_OPTIONS_TABLE_NAME, write_option=get_table_write_option(incremental), include_index=False)


# Storage requirements:
# Assume 200 characters per row (across 9 columns)
# 300 rows per day, per symbol
# 200 * 300 = 60,000 = 60KB per day, per symbol
# 60KB * 252 = 15MB per year, per symbol
# 5 symbols = 75MB per year
# 10 years = 750MB

# Execution time:
# Assume 1 second for each iteration in for loop (update_all_historical_options)
# Every 20 calls has 60 second sleep
# 5 symbols, 252 trading days, 10 years = 12,600 iterations
# 12,600 / 20 = 630 * 60 = 37,800 seconds = 10.5 hours
# 12,600 seconds = 3.5 hours
# Total = 14 hours
