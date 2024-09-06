import pandas as pd
import numpy as np
import inflection
import json

from api_data.collector import AlphaVantageClient
from api_data.storage import delete_company_overview_row, store_data, TableWriteOption
from api_data.util import drop_existing_rows, get_table_write_option, graceful_df_to_numeric

from enum import Enum


class FundamentalDataType(Enum):
    OVERVIEW = 'OVERVIEW'
    EARNINGS = 'EARNINGS'
    SPLITS = 'SPLITS'


DATA_TYPE_TABLES = {
    FundamentalDataType.OVERVIEW: {
        'table_name': 'company_overview',
        'columns': ['exchange', 'country', 'sector', 'industry', 'market_capitalization', 'book_value', 'dividend_yield', 'eps', 'price_to_book_ratio', 'beta', '52_week_high', '52_week_low', 'forward_pe'],
        'include_index': False,
    },
    FundamentalDataType.EARNINGS: {
        'table_name': 'quarterly_earnings',
        'columns': ['reported_eps', 'estimated_eps', 'surprise', 'surprise_percentage'],
        'include_index': True,
    },
    FundamentalDataType.SPLITS: {
        'table_name': 'stock_splits',
        'columns': ['symbol', 'effective_date', 'split_factor'],
        'include_index': False,
    },
}
EARNINGS_DATE_COL = 'fiscal_date_ending'


def fetch_fundamental_data(api_client: AlphaVantageClient, symbol: str, data_type: FundamentalDataType):
    params = {
        'function': data_type.value,
        'symbol': symbol,
    }
    return api_client.fetch(**params)


def parse_overview(data: dict, symbol: str):
    data = {key: [value] for key, value in data.items()}
    df = pd.DataFrame(data)
    df.columns = [inflection.underscore(col) for col in df.columns]
    df = df.filter(items=DATA_TYPE_TABLES[FundamentalDataType.OVERVIEW]['columns'])
    # replace df['dividend_yield'] values with 0 if it is None
    df['dividend_yield'] = df['dividend_yield'].replace('None', 0)
    # replace price_to_book_ratio values with 0 if it is None
    df['price_to_book_ratio'] = df['price_to_book_ratio'].replace('None', 0)
    df['price_to_book_ratio'] = df['price_to_book_ratio'].replace('-', np.nan)
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
    df.index.name = EARNINGS_DATE_COL
    df.index = pd.to_datetime(df.index)

    df.columns = [inflection.underscore(col) for col in df.columns]
    df = df.filter(items=DATA_TYPE_TABLES[FundamentalDataType.EARNINGS]['columns'])

    # All this data is numeric, but it's possible to have Nones. We don't need to apply numeric gracefully because
    # if it fails, due to the data being None, it will convert to NaN, which is fine because it becomes NULL in MySQL.
    df = df.apply(pd.to_numeric, errors='coerce')

    # We add in the symbol after converting to numeric because it's not a numeric column.
    df['symbol'] = symbol

    #print(df.head())
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
    return df


def update_all_fundamental_data(api_client: AlphaVantageClient, symbol: str, incremental: bool = True):
    for data_type in FundamentalDataType:
        response = fetch_fundamental_data(api_client, symbol, data_type)
        df = DATA_TYPE_PARSERS[data_type](response, symbol)

        write_option = get_table_write_option(incremental)
        if incremental:
            if data_type == FundamentalDataType.EARNINGS:
                df = drop_existing_rows(df, DATA_TYPE_TABLES[data_type]['table_name'], EARNINGS_DATE_COL, symbol)
            else:
                # We need to delete the company_overview row for this symbol
                # before inserting a new row because the symbol is the PK.
                delete_company_overview_row(symbol)

        #print(f'{data_type.value} data')
        #print(df.head())

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
