import pandas as pd
import inflection

from collector import AlphaVantageClient
from storage import store_data, TableWriteOption
from util import drop_existing_rows, get_table_write_option, graceful_df_to_numeric

from enum import Enum


class FundamentalDataType(Enum):
    OVERVIEW = 'OVERVIEW'
    EARNINGS = 'EARNINGS'


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

    print(df.head())
    return df


def update_all_fundamental_data(api_client: AlphaVantageClient, symbol: str, incremental: bool = True):
    for data_type in FundamentalDataType:
        response = fetch_fundamental_data(api_client, symbol, data_type)
        df = DATA_TYPE_PARSERS[data_type](response, symbol)

        # Currently, company_overview is just a snapshot, so it doesn't support incremental updates.
        write_option = get_table_write_option(incremental) if data_type == FundamentalDataType.EARNINGS else TableWriteOption.REPLACE
        if write_option == TableWriteOption.APPEND:
            df = drop_existing_rows(df, DATA_TYPE_TABLES[data_type]['table_name'], EARNINGS_DATE_COL, symbol)

        print(f'{data_type.value} data')
        print(df.head())

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
}
