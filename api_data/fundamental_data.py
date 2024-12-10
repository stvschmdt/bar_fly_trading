import logging
from enum import Enum

import inflection
import numpy as np
import pandas as pd

from api_data.collector import AlphaVantageClient
from api_data.storage import delete_company_overview_row, store_data
from api_data.util import drop_existing_rows, get_table_write_option, graceful_df_to_numeric
from logging_config import setup_logging

import ipdb

setup_logging()
logger = logging.getLogger(__name__)


class FundamentalDataType(Enum):
    OVERVIEW = 'OVERVIEW'
    EARNINGS = 'EARNINGS'
    SPLITS = 'SPLITS'


DATA_TYPE_TABLES = {
    FundamentalDataType.OVERVIEW: {
        'table_name': 'company_overview',
        'columns': ['exchange', 'country', 'sector', 'industry', 'market_capitalization', 'book_value', 'dividend_yield', 'eps', 'beta', '52_week_high', '52_week_low', 'forward_pe', 'shares_outstanding', 'price_to_book_ratio', 'analyst_rating_strong_buy', 'analyst_rating_buy', 'analyst_rating_hold', 'analyst_rating_sell', 'analyst_rating_strong_sell'],
        'include_index': False,
    },
    FundamentalDataType.EARNINGS: {
        'table_name': 'quarterly_earnings',
        'columns': ['reported_eps', 'estimated_eps', 'surprise', 'surprise_percentage', 'ttm_eps', 'latest_trading_day', 'symbol'],
        'include_index': True,
        'date_col': 'fiscal_date_ending'
    },
    FundamentalDataType.SPLITS: {
        'table_name': 'stock_splits',
        'columns': ['symbol', 'effective_date', 'split_factor'],
        'include_index': False,
        'date_col': 'effective_date'
    },
}


def fetch_fundamental_data(api_client: AlphaVantageClient, symbol: str, data_type: FundamentalDataType):
    params = {
        'function': data_type.value,
        'symbol': symbol,
    }
    return api_client.fetch(**params)


def parse_overview(data: dict, symbol: str):
    pd.set_option('future.no_silent_downcasting', True)
    data = {key: [value] for key, value in data.items()}
    df = pd.DataFrame(data)
    df.columns = [inflection.underscore(col) for col in df.columns]
    df = df.filter(items=DATA_TYPE_TABLES[FundamentalDataType.OVERVIEW]['columns'])
    df['dividend_yield'] = df['dividend_yield'].replace('None', 0)
    df['price_to_book_ratio'] = df['price_to_book_ratio'].replace('None', 0)
    df['price_to_book_ratio'] = df['price_to_book_ratio'].replace('-', np.nan)
    df['book_value'] = df['book_value'].replace('None', 0)
    df['eps'] = df['eps'].replace('None', 0)
    df['beta'] = df['beta'].replace('None', 0)
    df['analyst_rating_strong_buy'] = df['analyst_rating_strong_buy'].replace('-', 0)
    df['analyst_rating_buy'] = df['analyst_rating_buy'].replace('-', 0)
    df['analyst_rating_hold'] = df['analyst_rating_hold'].replace('-', 0)
    df['analyst_rating_sell'] = df['analyst_rating_sell'].replace('-', 0)
    df['analyst_rating_strong_sell'] = df['analyst_rating_strong_sell'].replace('-', 0)
    df['forward_pe'] = df['forward_pe'].replace('-', np.nan)
    df = graceful_df_to_numeric(df)
    # We add in the symbol after converting to numeric because it's not a numeric column.
    df['symbol'] = symbol
    return df


def parse_earnings(data: dict, symbol: str):
    key = 'quarterlyEarnings'
    if key not in data:
        raise ValueError(f"Unexpected response format: '{key}' key not found")

    df = pd.DataFrame(data[key])
    df.set_index('fiscalDateEnding', inplace=True)
    df.index.name = DATA_TYPE_TABLES[FundamentalDataType.EARNINGS]['date_col']
    df.index = pd.to_datetime(df.index)

    df.columns = [inflection.underscore(col) for col in df.columns]
    df = df.filter(items=DATA_TYPE_TABLES[FundamentalDataType.EARNINGS]['columns'])

    # All this data is numeric, but it's possible to have Nones. We don't need to apply numeric gracefully because
    # if it fails, due to the data being None, it will convert to NaN, which is fine because it becomes NULL in MySQL.
    # create a column for ttm_eps which is the last 4 quarters of reported_eps, unless it is NULL then estimated_eps is used
    # this needs to be from the most recent quarter to the least recent quarter and the df is not guaranteed to be in that order
    # if reported_eps is NULL, use estimated_eps
    df['ttm_eps'] = df['reported_eps']
    df['ttm_eps'] = np.where(df['ttm_eps'].isnull(), df['estimated_eps'], df['ttm_eps'])
    # sort the df by date in descending order
    df = df.sort_index(ascending=True)
    # calculate the ttm_eps
    df['ttm_eps'] = df['ttm_eps'].rolling(4).sum()
    # round ttm_eps to 2 decimal places
    df['ttm_eps'] = df['ttm_eps'].round(2)

    df = df.apply(pd.to_numeric, errors='coerce')
    df['latest_trading_day'] = df.index
    df['latest_trading_day'] = np.where(df['latest_trading_day'].dt.dayofweek == 5, df['latest_trading_day'] - pd.Timedelta(days=1), df['latest_trading_day'])
    df['latest_trading_day'] = np.where(df['latest_trading_day'].dt.dayofweek == 6, df['latest_trading_day'] - pd.Timedelta(days=2), df['latest_trading_day'])
    # convert the latest_trading_day to sql DATETIME format
    df['latest_trading_day'] = pd.to_datetime(df['latest_trading_day'], unit='ns', errors='coerce')

    # print the data type of index and latest_trading_day
    # Format for MySQL
    df['latest_trading_day'] = df['latest_trading_day'].dt.strftime('%Y-%m-%d')

    # We add in the symbol after converting to numeric because it's not a numeric column.
    df['symbol'] = symbol

    return df


def parse_splits(data: dict, symbol: str):
    key = 'data'
    if key not in data:
        raise ValueError(f"Unexpected response format: '{key}' key not found")

    df = pd.DataFrame(data[key])
    df.columns = [inflection.underscore(col) for col in df.columns]
    df = df.filter(items=DATA_TYPE_TABLES[FundamentalDataType.SPLITS]['columns'])
    df = graceful_df_to_numeric(df)
    df['symbol'] = symbol
    if not df.empty:
        df = df.set_index(df[DATA_TYPE_TABLES[FundamentalDataType.SPLITS]['date_col']], drop=False)
        df.index = pd.to_datetime(df.index)
    return df


def update_all_fundamental_data(api_client: AlphaVantageClient, symbol: str, incremental: bool = True):
    for data_type in FundamentalDataType:
        response = fetch_fundamental_data(api_client, symbol, data_type)

        # An empty response indicates that the symbol isn't a company (e.g. index, ETF, etc.)
        if not response:
            logger.info(f"No fundamental data found for {symbol}")
            return

        df = DATA_TYPE_PARSERS[data_type](response, symbol)

        write_option = get_table_write_option(incremental)
        if incremental:
            if data_type != FundamentalDataType.OVERVIEW:
                df = drop_existing_rows(df, DATA_TYPE_TABLES[data_type]['table_name'], DATA_TYPE_TABLES[data_type]['date_col'], symbol)
            else:
                # We need to delete the company_overview row for this symbol
                # before inserting a new row because the symbol is the PK.
                delete_company_overview_row(symbol)

        store_data(
            df,
            table_name=DATA_TYPE_TABLES[data_type]['table_name'],
            write_option=write_option,
            include_index=DATA_TYPE_TABLES[data_type].get('include_index', True),
        )


# Functions to parse fundamental various types of data responses into dataframes
DATA_TYPE_PARSERS = {
    FundamentalDataType.OVERVIEW: parse_overview,
    FundamentalDataType.EARNINGS: parse_earnings,
    FundamentalDataType.SPLITS: parse_splits
}
