import os
from enum import Enum

from sqlalchemy import create_engine, text

import pandas as pd

username = 'root'
password = os.getenv('MYSQL_PASSWORD', None)
host = '127.0.0.1'
port = 3306
dbname = 'bar_fly_trading'


TABLE_CREATES = {
    'core_stock': 'CREATE TABLE core_stock(date DATETIME, open DOUBLE, high DOUBLE, low DOUBLE, adjusted_close DOUBLE, volume BIGINT, symbol VARCHAR(5), PRIMARY KEY (date, symbol));',
    'company_overview': 'CREATE TABLE company_overview(exchange VARCHAR(10), country VARCHAR(20), sector VARCHAR(30), industry VARCHAR(50), market_capitalization bigint, book_value DOUBLE, dividend_yield DOUBLE, eps DOUBLE, price_to_book_ratio DOUBLE, beta DOUBLE, 52_week_high DOUBLE, 52_week_low DOUBLE, forward_pe DOUBLE, symbol VARCHAR(5), PRIMARY KEY (symbol));',
    'quarterly_earnings': 'CREATE TABLE quarterly_earnings(fiscal_date_ending DATETIME, reported_eps DOUBLE, estimated_eps DOUBLE, surprise DOUBLE, surprise_percentage DOUBLE, symbol VARCHAR(5), PRIMARY KEY (fiscal_date_ending, symbol));',
    'economic_indicators': 'CREATE TABLE economic_indicators(date DATETIME, treasury_yield_2year DOUBLE, treasury_yield_10year DOUBLE, ffer DOUBLE, cpi DOUBLE, inflation DOUBLE, retail_sales DOUBLE, durables DOUBLE, unemployment DOUBLE, nonfarm_payroll DOUBLE, PRIMARY KEY (date));',
    'technical_indicators': 'CREATE TABLE technical_indicators(date DATETIME, sma_20 DOUBLE, sma_50 DOUBLE, sma_200 DOUBLE, ema_20 DOUBLE, ema_50 DOUBLE, ema_200 DOUBLE, macd DOUBLE, rsi_14 DOUBLE, bbands_upper_20 DOUBLE, bbands_middle_20 DOUBLE, bbands_lower_20 DOUBLE, symbol VARCHAR(5), PRIMARY KEY (date, symbol));',
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


def select_all_by_symbol(table_name: str, symbol: str, order_by: str = None):
    query = f"""
    SELECT * from {table_name} WHERE symbol = '{symbol}' {f'ORDER BY {order_by} desc' if order_by else ''};
    """
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


def write_all_table_joins(limit: int = 50000):
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
            tech.bbands_upper_20,
            tech.bbands_middle_20,
            tech.bbands_lower_20,
            econ.treasury_yield_2year,
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
            comp.52_week_high,
            comp.52_week_low,
            comp.forward_pe
        FROM core_stock as core 
        LEFT JOIN technical_indicators as tech
        ON core.date = tech.date
        AND core.symbol = tech.symbol
        LEFT JOIN economic_indicators as econ
        ON core.date = econ.date
        LEFT JOIN quarterly_earnings as quart
        ON core.date = quart.fiscal_date_ending
        AND core.symbol = quart.symbol
        LEFT JOIN company_overview as comp
        ON core.symbol = comp.symbol
        WHERE core.date > '2016-01-01' 
        LIMIT {limit};
    """
    df = pd.read_sql_query(query, engine)
    df.to_csv('all_data.csv')


connection_string = f'mysql+pymysql://{username}:{password}@{host}:{port}/{dbname}'
engine = create_database()
