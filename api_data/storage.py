import logging
import os
import sys
from enum import Enum

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

DB_CREDENTIALS = {
    'local': {
        'username': 'root',
        'password': os.getenv('MYSQL_PASSWORD', None),
        'port': 3306
    },
    'remote': {
        'username': 'readonly_user',
        'password': os.getenv('MYSQL_READONLY_PASSWORD', None),
        'port': 3307
    }
}

HOST = '127.0.0.1'
DB_NAME = 'bar_fly_trading'

TABLE_CREATES = {
    'core_stock': 'CREATE TABLE core_stock(date DATETIME, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, adjusted_close DOUBLE, volume BIGINT, symbol VARCHAR(5), PRIMARY KEY (date, symbol));',
    'company_overview': 'CREATE TABLE company_overview(exchange VARCHAR(10), country VARCHAR(20), sector VARCHAR(50), industry VARCHAR(100), market_capitalization bigint, book_value DOUBLE, dividend_yield DOUBLE, eps DOUBLE, price_to_book_ratio DOUBLE, beta DOUBLE, 52_week_high DOUBLE, shares_outstanding BIGINT, 52_week_low DOUBLE, analyst_rating_strong_buy INT, analyst_rating_buy INT, analyst_rating_hold INT, analyst_rating_sell INT, analyst_rating_strong_sell INT, forward_pe DOUBLE, symbol VARCHAR(5), PRIMARY KEY (symbol));',
    'quarterly_earnings': 'CREATE TABLE quarterly_earnings(fiscal_date_ending DATETIME, reported_eps DOUBLE, estimated_eps DOUBLE, ttm_eps DOUBLE, latest_trading_day DATETIME, surprise DOUBLE, surprise_percentage DOUBLE, symbol VARCHAR(5), PRIMARY KEY (fiscal_date_ending, symbol));',
    'economic_indicators': 'CREATE TABLE economic_indicators(date DATETIME, treasury_yield_2year DOUBLE, treasury_yield_10year DOUBLE, ffer DOUBLE, cpi DOUBLE, inflation DOUBLE, retail_sales DOUBLE, durables DOUBLE, unemployment DOUBLE, nonfarm_payroll DOUBLE, PRIMARY KEY (date));',
    'technical_indicators': 'CREATE TABLE technical_indicators(date DATETIME, sma_20 DOUBLE, sma_50 DOUBLE, sma_200 DOUBLE, ema_20 DOUBLE, ema_50 DOUBLE, ema_200 DOUBLE, macd DOUBLE, rsi_14 DOUBLE, adx_14 DOUBLE, atr_14 DOUBLE, cci_14 DOUBLE, bbands_upper_20 DOUBLE, bbands_middle_20 DOUBLE, bbands_lower_20 DOUBLE, symbol VARCHAR(5), PRIMARY KEY (date, symbol));',
    'stock_splits': 'CREATE TABLE stock_splits(symbol VARCHAR(5), effective_date DATETIME, split_factor DOUBLE, PRIMARY KEY (symbol, effective_date));',
    'historical_options': 'CREATE TABLE historical_options (contract_id VARCHAR(30) NOT NULL, date DATETIME NOT NULL, symbol VARCHAR(8) NOT NULL, type ENUM(\'call\', \'put\'), expiration DATE, strike FLOAT, last FLOAT, mark FLOAT, bid FLOAT, ask FLOAT, volume INT, implied_volatility FLOAT, delta FLOAT, gamma FLOAT, theta FLOAT, vega FLOAT, rho FLOAT, PRIMARY KEY (contract_id, date, symbol)) PARTITION BY KEY (symbol) PARTITIONS 16;'
}

TABLE_COLS = {
    'historical_options': {'contract_id': str, 'date': str, 'symbol': str, 'type': str, 'expiration': str, 'strike': float, 'last': float, 'mark': float, 'bid': float, 'ask': float, 'volume': int, 'implied_volatility': float, 'delta': float, 'gamma': float, 'theta': float, 'vega': float, 'rho': float},
}

engine = None


# Methods for writing to a table. These methods apply at a table level, not a row level.
# i.e. APPEND will add new rows to the table, REPLACE will replace the entire table with the new data.
class TableWriteOption(Enum):
    APPEND = 'append'
    REPLACE = 'replace'


def connect_database(db_location: str = 'local'):
    global engine
    if engine:
        return

    username = DB_CREDENTIALS[db_location]['username']
    password = DB_CREDENTIALS[db_location]['password']
    port = DB_CREDENTIALS[db_location]['port']
    if not password:
        raise Exception('Must set MYSQL_PASSWORD environment variable for local connection or MYSQL_READONLY_PASSWORD for remote.')

    connection_string = f'mysql+pymysql://{username}:{password}@{HOST}:{port}/{DB_NAME}'
    engine = create_engine(connection_string)


def get_engine():
    global engine
    if not engine:
        raise Exception('Database connection not yet established. Call connect_database() first.')
    return engine


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
    df.to_sql(table_name, get_engine(), if_exists=TableWriteOption.APPEND.value, index=include_index)


def insert_ignore_data(df, table_name):
    """
    Insert data that doesn't already exist into the given table. For non-numerical column types,
    wrap their values in single quotes.
    """
    columns = TABLE_COLS[table_name]
    if df.empty:
        print(f'No data to store in {table_name}.')
        return

    rows = []
    for _, row in df.iterrows():
        cols = ', '.join([f"'{row[col]}'" if col_type not in {int, float} else str(row[col]) for col, col_type in columns.items()])
        rows.append(f'({cols})')
    rows = ', '.join(rows)
    with get_engine().connect() as connection:
        query = text(f'INSERT IGNORE INTO {table_name} VALUES {rows};')
        connection.execute(query)
        connection.commit()


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

    return pd.read_sql_query(query, get_engine())


def get_last_updated_date(table_name: str, date_col: str, symbol: str):
    # Get the most recent date in the table with an optional symbol filter (not all tables have symbols)
    query = text(f"SELECT MAX({date_col}) FROM {table_name}{' WHERE symbol = :symbol' if symbol else ''};")
    with get_engine().connect() as connection:
        result = connection.execute(query, {"symbol": symbol})
        last_updated_date = pd.to_datetime(result.fetchone()[0])
    return last_updated_date


# Get the dates that already exist in a table for a given symbol. start_date and end_date are inclusive.
def get_dates_for_symbol(table_name: str, symbol: str, date_col: str, start_date: str = None, end_date: str = None) -> list[str]:
    and_clause = ''
    if start_date:
        and_clause += f" AND {date_col} >= '{start_date}'"
    if end_date:
        and_clause += f" AND {date_col} <= '{end_date}'"
    query = text(f"SELECT DISTINCT({date_col}) FROM {table_name} WHERE symbol = '{symbol}'{and_clause} ORDER BY {date_col} ASC;")

    with get_engine().connect() as connection:
        result = connection.execute(query)
        dates = [row[0].strftime('%Y-%m-%d') for row in result.fetchall()]
    return dates


def drop_table(table_name: str):
    query = text(f'DROP TABLE IF EXISTS {table_name};')
    with get_engine().connect() as connection:
        connection.execute(query)


def create_table(table_name: str):
    if table_name not in TABLE_CREATES:
        raise ValueError(f'No CREATE TABLE statement found for table: {table_name}')

    query = text(TABLE_CREATES[table_name])
    with get_engine().connect() as connection:
        connection.execute(query)


def delete_company_overview_row(symbol: str):
    query = text("DELETE FROM company_overview WHERE symbol = :symbol;")
    with get_engine().connect() as connection:
        # For some reason, deleting rows seems to require a transaction
        with connection.begin() as transaction:
            connection.execute(query, {'symbol': symbol})
            transaction.commit()


def get_stock_splits():
    query = "SELECT symbol, effective_date, split_factor FROM stock_splits;"
    df = pd.read_sql_query(query, get_engine())

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
        mask = (df['symbol'] == symbol) & (df['date'] < effective_date)
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
            tech.cci_14,
            tech.bbands_upper_20,
            tech.bbands_middle_20,
            tech.bbands_lower_20,
            options_agg.call_volume,
            options_agg.put_volume,
            options_agg.total_volume,
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
        LEFT JOIN
        (
        SELECT
            date AS trade_date,
            symbol,
            SUM(CASE WHEN type = 'call' THEN volume ELSE 0 END) AS call_volume,
            SUM(CASE WHEN type = 'put' THEN volume ELSE 0 END) AS put_volume,
            SUM(volume) AS total_volume
        FROM
            historical_options
        WHERE date > '{earliest_date}'
        AND symbol in ({', '.join([f"'{symbol}'" for symbol in symbols])})
        GROUP BY
            trade_date,
            symbol
        ) AS options_agg
        ON core.date = options_agg.trade_date
        AND core.symbol = options_agg.symbol
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
    df = pd.read_sql_query(query, get_engine())

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

    # Options mean, std volume
    df[['options_14_mean', 'options_14_std']] = df.groupby('symbol')['total_volume'].rolling(window=14, min_periods=1).agg(['mean', 'std']).reset_index(level=0, drop=True)
    df['pcr'] = (df['put_volume'] / df['call_volume']).replace([np.inf, -np.inf], np.nan).round(2)
    df['pcr_14_mean'] = df.groupby('symbol')['pcr'].rolling(window=14, min_periods=1).agg(['mean']).reset_index(level=0, drop=True)


    # quarterly_earnings filldown
    df['fiscal_date_ending'] = df.groupby('symbol')['fiscal_date_ending'].ffill()
    df['reported_eps'] = df.groupby('symbol')['reported_eps'].ffill()
    df['estimated_eps'] = df.groupby('symbol')['estimated_eps'].ffill()
    df['surprise'] = df.groupby('symbol')['surprise'].ffill()
    df['surprise_percentage'] = df.groupby('symbol')['surprise_percentage'].ffill()
    df['ttm_eps'] = df.groupby('symbol')['ttm_eps'].ffill()
    df['book_value'] = df.groupby('symbol')['book_value'].ffill()
    # calculate the 9 day exponential moving average of the macd for each symbol
    df["macd_9_ema"] = df.groupby("symbol")["macd"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    # 52 week high and low filldown
    df['52_week_high'] = df.groupby('symbol')['52_week_high'].ffill()
    df['52_week_low'] = df.groupby('symbol')['52_week_low'].ffill()

    df['pe_ratio'] = (df['adjusted_close'] / df['ttm_eps']).replace([np.inf, -np.inf], np.nan)
    df['price_to_book_ratio'] = (df['adjusted_close'] / df['book_value']).replace([np.inf, -np.inf], np.nan)

    # economic_indicators filldown
    df['treasury_yield_2year'] = df['treasury_yield_2year'].ffill()
    df['treasury_yield_10year'] = df['treasury_yield_10year'].ffill()
    df['ffer'] = df['ffer'].ffill()
    df['cpi'] = df['cpi'].ffill()
    df['retail_sales'] = df['retail_sales'].ffill()
    df['durables'] = df['durables'].ffill()
    df['unemployment'] = df['unemployment'].ffill()
    df['nonfarm_payroll'] = df['nonfarm_payroll'].ffill()
    df['inflation'] = df['inflation'].ffill()

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

    # add columns for percent above or below sma_20, sma_50, sma_200, 52_week_high, 52_week_low rounded to 2 decimal places
    df["sma_20_pct"] = ((df["adjusted_close"] - df["sma_20"]) / df["sma_20"] * 100).round(2)
    df["sma_50_pct"] = ((df["adjusted_close"] - df["sma_50"]) / df["sma_50"] * 100).round(2)
    df["sma_200_pct"] = ((df["adjusted_close"] - df["sma_200"]) / df["sma_200"] * 100).round(2)
    df["52_week_high_pct"] = ((df["adjusted_close"] - df["52_week_high"]) / df["52_week_high"] * 100).round(2)
    df["52_week_low_pct"] = ((df["adjusted_close"] - df["52_week_low"]) / df["52_week_low"] * 100).round(2)

    # Adjust open, high, low for stock splits
    df = adjust_for_stock_splits(df)
    # Vectorized screener signals — replaces O(N²) row-by-row loop
    # Each signal: 1 = bullish, -1 = bearish, 0 = neutral
    # Logic matches StockScreenerV2._check_* methods exactly

    # MACD signal: macd > macd_9_ema → bullish, macd < macd_9_ema → bearish
    df["macd_signal"] = np.where(df["macd"] > df["macd_9_ema"], 1,
                        np.where(df["macd"] < df["macd_9_ema"], -1, 0))

    # MACD zero: macd > 0 → bullish, macd < 0 → bearish
    df["macd_zero_signal"] = np.where(df["macd"] > 0, 1,
                             np.where(df["macd"] < 0, -1, 0))

    # ADX: > 25 → bullish (strong trend), < 20 → bearish (weak trend)
    df["adx_signal"] = np.where(df["adx_14"] > 25, 1,
                       np.where(df["adx_14"] < 20, -1, 0))

    # ATR: close < atr*2 → bullish, close > atr*2 → bearish
    df["atr_signal"] = np.where(df["adjusted_close"] < df["atr_14"] * 2, 1,
                       np.where(df["adjusted_close"] > df["atr_14"] * 2, -1, 0))

    # PE ratio: 0 < pe < 15 → bullish (value), pe > 35 → bearish (expensive)
    df["pe_ratio_signal"] = np.where((df["pe_ratio"] < 15) & (df["pe_ratio"] > 0), 1,
                            np.where(df["pe_ratio"] > 35, -1, 0))

    # Bollinger bands: close > upper → bearish, close < lower → bullish
    df["bollinger_bands_signal"] = np.where(df["adjusted_close"] > df["bbands_upper_20"], -1,
                                   np.where(df["adjusted_close"] < df["bbands_lower_20"], 1, 0))

    # RSI: > 70 → bearish (overbought), < 30 → bullish (oversold)
    df["rsi_signal"] = np.where(df["rsi_14"] > 70, -1,
                       np.where(df["rsi_14"] < 30, 1, 0))

    # SMA cross: sma_20 > sma_50 → bullish, sma_20 < sma_50 → bearish
    df["sma_cross_signal"] = np.where(df["sma_20"] > df["sma_50"], 1,
                             np.where(df["sma_20"] < df["sma_50"], -1, 0))

    # CCI: >= 100 → bearish (overbought), <= -100 → bullish (oversold)
    df["cci_signal"] = np.where(df["cci_14"] >= 100, -1,
                       np.where(df["cci_14"] <= -100, 1, 0))

    # PCR: > 0.7 → bearish, <= 0.5 → bullish, NaN → 0
    pcr_filled = df["pcr"].fillna(0)
    df["pcr_signal"] = np.where(df["pcr"].isna(), 0,
                       np.where(pcr_filled > 0.7, -1,
                       np.where(pcr_filled <= 0.5, 1, 0)))

    # Bull-bear delta: sum of all signals
    signal_cols = ["macd_signal", "macd_zero_signal", "adx_signal", "atr_signal",
                   "pe_ratio_signal", "bollinger_bands_signal", "rsi_signal",
                   "sma_cross_signal", "cci_signal", "pcr_signal"]
    df["bull_bear_delta"] = df[signal_cols].sum(axis=1)

    df.to_csv(f'all_data_{batch_num}.csv')

# for testing queries
if __name__ == '__main__':
    print('running main')
    connect_database()
    import time
    start = time.time()
    gold_table_processing(["TJX", "NVDA", "AMD"], batch_num=1, earliest_date='2025-08-01', limit=50000)
    end = time.time()
    execution_time = end-start
    print(f"Execution time: {execution_time:.4f} seconds")
