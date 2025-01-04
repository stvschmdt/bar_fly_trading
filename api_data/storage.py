import logging
import os
import sys
from enum import Enum

import pandas as pd
from sqlalchemy import create_engine, text
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logging_config import setup_logging
from visualizations.screener import StockScreener

setup_logging()
logger = logging.getLogger(__name__)

username = 'root'
password = os.getenv('MYSQL_PASSWORD', None)
host = '127.0.0.1'
port = 3306
dbname = 'bar_fly_trading'


TABLE_CREATES = {
    'core_stock': 'CREATE TABLE core_stock(date DATETIME, open DOUBLE, high DOUBLE, low DOUBLE, adjusted_close DOUBLE, volume BIGINT, symbol VARCHAR(5), PRIMARY KEY (date, symbol));',
    'company_overview': 'CREATE TABLE company_overview(exchange VARCHAR(10), country VARCHAR(20), sector VARCHAR(30), industry VARCHAR(60), market_capitalization bigint, book_value DOUBLE, dividend_yield DOUBLE, eps DOUBLE, price_to_book_ratio DOUBLE, beta DOUBLE, 52_week_high DOUBLE, shares_outstanding BIGINT, 52_week_low DOUBLE, analyst_rating_strong_buy INT, analyst_rating_buy INT, analyst_rating_hold INT, analyst_rating_sell INT, analyst_rating_strong_sell INT, forward_pe DOUBLE, symbol VARCHAR(5), PRIMARY KEY (symbol));',
    'quarterly_earnings': 'CREATE TABLE quarterly_earnings(fiscal_date_ending DATETIME, reported_eps DOUBLE, estimated_eps DOUBLE, ttm_eps DOUBLE, latest_trading_day DATETIME, surprise DOUBLE, surprise_percentage DOUBLE, symbol VARCHAR(5), PRIMARY KEY (fiscal_date_ending, symbol));',
    'economic_indicators': 'CREATE TABLE economic_indicators(date DATETIME, treasury_yield_2year DOUBLE, treasury_yield_10year DOUBLE, ffer DOUBLE, cpi DOUBLE, inflation DOUBLE, retail_sales DOUBLE, durables DOUBLE, unemployment DOUBLE, nonfarm_payroll DOUBLE, PRIMARY KEY (date));',
    'technical_indicators': 'CREATE TABLE technical_indicators(date DATETIME, sma_20 DOUBLE, sma_50 DOUBLE, sma_200 DOUBLE, ema_20 DOUBLE, ema_50 DOUBLE, ema_200 DOUBLE, macd DOUBLE, rsi_14 DOUBLE, adx_14 DOUBLE, atr_14 DOUBLE, bbands_upper_20 DOUBLE, bbands_middle_20 DOUBLE, bbands_lower_20 DOUBLE, symbol VARCHAR(5), PRIMARY KEY (date, symbol));',
    'stock_splits': 'CREATE TABLE stock_splits(symbol VARCHAR(5), effective_date DATETIME, split_factor DOUBLE, PRIMARY KEY (symbol, effective_date));',
}


# Methods for writing to a table. These methods apply at a table level, not a row level.
# i.e. APPEND will add new rows to the table, REPLACE will replace the entire table with the new data.
class TableWriteOption(Enum):
    APPEND = 'append'
    REPLACE = 'replace'


if not password:
    raise Exception('Must set MYSQL_PASSWORD environment variable')


def create_database():
    return create_engine(connection_string)


def store_data(df, table_name, write_option: TableWriteOption, include_index=True):
    if df.empty:
        print(f'No data to store in {table_name}.')
        return

    if write_option == TableWriteOption.REPLACE:
        drop_table(table_name)
        # We create tables manually, rather than using df.to_sql, so we can set primary keys.
        create_table(table_name)

    # If we wanted to REPLACE, we've already dropped and recreated the table. That means we can always do our write as
    # an APPEND. This ensures our tables keep their exact types (e.g. VARCHAR vs. text) and primary keys.
    df.to_sql(table_name, engine, if_exists=TableWriteOption.APPEND.value, index=include_index)


def select_all_from_table(table_name: str, order_by: str, limit: int = 10):
    query = f"""
    SELECT * from {table_name} {f'ORDER BY {order_by} desc' if order_by else ''} LIMIT {limit};
    """
    df = pd.read_sql_query(query, engine)
    print(f'First {limit} rows from {table_name}:')
    pd.set_option('display.max_columns', None)
#    print(df)


def select_all_by_symbol(table_name: str, symbols: set[str], order_by: str = None, start_date: str = None, end_date: str = None):
    symbols_str = ', '.join([f"'{symbol}'" for symbol in symbols])

    query = f"""
    SELECT * from {table_name} WHERE symbol IN ({symbols_str})
    """

    # Add optional date filters
    if start_date:
        query += f" AND date >= '{start_date}'"
    if end_date:
        query += f" AND date <= '{end_date}'"

    # Add optional ordering
    if order_by:
        query += f" ORDER BY {order_by}"

    query += ";"

    return pd.read_sql_query(query, engine)


def get_last_updated_date(table_name: str, date_col: str, symbol: str):
    # Get the most recent date in the table with an optional symbol filter (not all tables have symbols)
    query = text(f"SELECT MAX({date_col}) FROM {table_name}{' WHERE symbol = :symbol' if symbol else ''};")
    with engine.connect() as connection:
        result = connection.execute(query, {"symbol": symbol})
        last_updated_date = pd.to_datetime(result.fetchone()[0])
    return last_updated_date


def drop_table(table_name: str):
    query = text(f'DROP TABLE IF EXISTS {table_name};')
    with engine.connect() as connection:
        connection.execute(query)


def create_table(table_name: str):
    if table_name not in TABLE_CREATES:
        raise ValueError(f'No CREATE TABLE statement found for table: {table_name}')

    query = text(TABLE_CREATES[table_name])
    with engine.connect() as connection:
        connection.execute(query)


def delete_company_overview_row(symbol: str):
    query = text("DELETE FROM company_overview WHERE symbol = :symbol;")
    with engine.connect() as connection:
        # For some reason, deleting rows seems to require a transaction
        with connection.begin() as transaction:
            connection.execute(query, {'symbol': symbol})
            transaction.commit()


def get_stock_splits():
    query = "SELECT symbol, effective_date, split_factor FROM stock_splits;"
    df = pd.read_sql_query(query, engine)

    # Sort splits by effective_date
    df = df.sort_values(by=['symbol', 'effective_date'])
    return df


def adjust_for_stock_splits(df):
    # Adjust open, high, low  for stock splits
    df_splits = get_stock_splits()

    df = df.copy()  # Ensure we're working on a copy of the DataFrame to avoid chaining issues
    df.loc[:, 'adjusted_open'] = df['open']
    df.loc[:, 'adjusted_high'] = df['high']
    df.loc[:, 'adjusted_low'] = df['low']

    # Apply split factors
    for _, split_row in df_splits.iterrows():
        symbol = split_row['symbol']
        effective_date = split_row['effective_date']
        split_factor = split_row['split_factor']

        # Adjust open values for the given symbol and dates less than or equal to the effective_date
        # Note: fractional pennies included in adjusted prices
        mask = (df['symbol'] == symbol) & (df['date'] <= effective_date)
        df.loc[mask, 'adjusted_open'] = df.loc[mask, 'adjusted_open'] / split_factor
        df.loc[mask, 'adjusted_high'] = df.loc[mask, 'adjusted_high'] / split_factor
        df.loc[mask, 'adjusted_low'] = df.loc[mask, 'adjusted_low'] / split_factor

    return df


def process_gold_table_in_batches(symbols: list[str], earliest_date: str = '2016-01-01', symbols_per_batch: int = 15):
    symbol_batches = [symbols[i:i + symbols_per_batch] for i in range(0, len(symbols), symbols_per_batch)]
    for i, symbol_batch in enumerate(symbol_batches):
        logger.info(f'Processing batch {i + 1} of {len(symbol_batches)}: {symbol_batch}')
        gold_table_processing(symbol_batch, i, earliest_date)


def gold_table_processing(symbols: list[str], batch_num: int, earliest_date: str = '2016-01-01', limit: int = 5000000):
    # This query joins all tables together on the date column. It's a way to see all the data we have in one place.
    # core_stock, company_overview, economic_indicators, technical_indicators, quarterly_earnings are the tables and
    # where core_stock date > '01-01-2016' is a filter to reduce the number of rows returned. This is useful for testing
    query = f"""
        SELECT core.date, 
            core.symbol, 
            core.open, 
            core.high, 
            core.low,
            core.adjusted_close, 
            core.volume, 
            tech.sma_20, 
            tech.sma_50, 
            tech.sma_200, 
            tech.ema_20,
            tech.ema_50,
            tech.ema_200,
            tech.macd,
            tech.rsi_14,
            tech.adx_14,
            tech.atr_14,
            tech.bbands_upper_20,
            tech.bbands_middle_20,
            tech.bbands_lower_20,
            econ.treasury_yield_2year,
            econ.treasury_yield_10year,
            econ.ffer,
            econ.cpi,
            econ.inflation,
            econ.retail_sales,
            econ.durables,
            econ.unemployment,
            econ.nonfarm_payroll,
            quart.fiscal_date_ending,
            quart.reported_eps,
            quart.estimated_eps,
            quart.ttm_eps,
            quart.surprise,
            quart.surprise_percentage,
            comp.exchange,
            comp.country,
            comp.sector,
            comp.industry,
            comp.market_capitalization,
            comp.book_value,
            comp.dividend_yield,
            comp.eps,
            comp.price_to_book_ratio,
            comp.beta,
            comp.shares_outstanding,
            comp.52_week_high,
            comp.52_week_low,
            comp.forward_pe,
            comp.analyst_rating_strong_buy,
            comp.analyst_rating_buy,
            comp.analyst_rating_hold,
            comp.analyst_rating_sell,
            comp.analyst_rating_strong_sell
        FROM core_stock as core 
        LEFT JOIN technical_indicators as tech
        ON core.date = tech.date
        AND core.symbol = tech.symbol
        LEFT JOIN economic_indicators as econ
        ON core.date = econ.date
        LEFT JOIN quarterly_earnings as quart
        ON core.date = quart.latest_trading_day
        AND core.symbol = quart.symbol
        LEFT JOIN company_overview as comp
        ON core.symbol = comp.symbol
        WHERE core.date > '{earliest_date}'
        AND core.symbol in ({', '.join([f"'{symbol}'" for symbol in symbols])})
        LIMIT {limit};
    """
    df = pd.read_sql_query(query, engine)

    # Derive adjusted_close_pct from adjusted_close in core_stock
    df['adjusted_close_pct'] = df.groupby('symbol')['adjusted_close'].pct_change(1)
    # Derive volume_pct from volume in core_stock
    df['volume_pct'] = df.groupby('symbol')['volume'].pct_change(1)
    # Derive open_pct from open in core_stock
    df['open_pct'] = df.groupby('symbol')['open'].pct_change(1)
    # Derive high_pct from high in core_stock
    df['high_pct'] = df.groupby('symbol')['high'].pct_change(1)
    # Derive low_pct from low in core_stock
    df['low_pct'] = df.groupby('symbol')['low'].pct_change(1)

    # quarterly_earnings filldown
    df['fiscal_date_ending'] = df.groupby('symbol')['fiscal_date_ending'].ffill()
    df['reported_eps'] = df.groupby('symbol')['reported_eps'].ffill()
    df['estimated_eps'] = df.groupby('symbol')['estimated_eps'].ffill()
    df['surprise'] = df.groupby('symbol')['surprise'].ffill()
    df['surprise_percentage'] = df.groupby('symbol')['surprise_percentage'].ffill()
    df['ttm_eps'] = df.groupby('symbol')['ttm_eps'].ffill()
    # calculate the 9 day exponential moving average of the macd for each symbol
    df["macd_9_ema"] = df.groupby("symbol")["macd"].transform(lambda x: x.ewm(span=9, adjust=False).mean())

    df['pe_ratio'] = df['adjusted_close'] / df['ttm_eps']

    # economic_indicators filldown
    df['treasury_yield_2year'] = df['treasury_yield_2year'].ffill()
    df['treasury_yield_10year'] = df['treasury_yield_10year'].ffill()
    df['ffer'] = df['ffer'].ffill()
    df['cpi'] = df['cpi'].ffill()
    df['retail_sales'] = df['retail_sales'].ffill()
    df['durables'] = df['durables'].ffill()
    df['unemployment'] = df['unemployment'].ffill()
    df['nonfarm_payroll'] = df['nonfarm_payroll'].ffill()

    # add day of week number column based on date
    df['day_of_week_num'] = df['date'].dt.dayofweek
    # add day of week name column based on date
    df['day_of_week_name'] = df['date'].dt.day_name()
    # add month number column based on date
    df['month'] = df['date'].dt.month
    # add day of year column based on date
    df['day_of_year'] = df['date'].dt.dayofyear
    # add year column based on date
    df['year'] = df['date'].dt.year
    # add columns for the percent gain n+3, n+10, n+30 days from current date 
    df["future_3_day_pct"] = (
        df.groupby("symbol")["adjusted_close"]
        .apply(lambda x: (x.shift(-3) - x) / x * 100)
        .reset_index(level=0, drop=True)
    )
    df["future_10_day_pct"] = (
        df.groupby("symbol")["adjusted_close"]
        .apply(lambda x: (x.shift(-10) - x) / x * 100)
        .reset_index(level=0, drop=True)
    )
    df["future_30_day_pct"] = (
        df.groupby("symbol")["adjusted_close"]
        .apply(lambda x: (x.shift(-30) - x) / x * 100)
        .reset_index(level=0, drop=True)
    )
    # Adjust open, high, low for stock splits
    df = adjust_for_stock_splits(df)
    # get unique list of symbols
    symbols = df["symbol"].unique()
    # get the latest date in the df
    latest_date = df["date"].max()
    # change latest_date to a string
    latest_date = latest_date.strftime("%Y-%m-%d")
    # create a StockScreener object to use _check functions
    stock_screener = StockScreener(symbols, latest_date, df.head())
    # for each row in the df, use each of the _check functions to create a new column for each
    # the _check functions will build a list of -1,0,1 for each _check function
    # append this list to the df as new columns
    # the check functions are _check macd, adx, atr, pe_ratio, bollinger_bands, rsi, sma_cross
    # the check functions take the df row as input and three lists, we will dummy the first two
    # the finsl list 'signals' will be appended to for each row 
    full_cols = []
    for index, row in df.iterrows():
        signals = []
        # get the df for the row symbol and date
        screen_df = df[(df.symbol == row['symbol']) & (df.date == row['date'])]
        stock_screener._check_macd(screen_df, [], [], signals)
        stock_screener._check_adx(screen_df, [], [], signals)
        stock_screener._check_atr(screen_df, [], [], signals)
        stock_screener._check_pe_ratio(screen_df, [], [], signals)
        stock_screener._check_bollinger_band(screen_df, [], [], signals)
        stock_screener._check_rsi(screen_df, [], [], signals)
        stock_screener._check_sma_cross(screen_df, [], [], signals)
        # append the row of signals, to the full_cols list
        full_cols.append(signals)

    # create a dataframe from the full_cols list with the signals as columns
    signals_df = pd.DataFrame(full_cols, columns=["macd_signal", "adx_signal", "atr_signal", "pe_ratio_signal", "bollinger_bands_signal", "rsi_signal", "sma_cross_signal"])
    # concat the signals_df as new columns to the df
    df = pd.concat([df, signals_df], axis=1)
    
    # create a new column for the sum of all the signals
    df["bull_bear_delta"] = df["macd_signal"] + df["adx_signal"] + df["atr_signal"] + df["pe_ratio_signal"] + df["bollinger_bands_signal"] + df["rsi_signal"] + df["sma_cross_signal"]


    df.to_csv(f'all_data_{batch_num}.csv')


connection_string = f'mysql+pymysql://{username}:{password}@{host}:{port}/{dbname}'
engine = create_database()
