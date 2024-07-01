import pandas as pd
import inflection

from collector import AlphaVantageClient
from storage import store_data

from enum import Enum


class FundamentalDataType(Enum):
    OVERVIEW = 'OVERVIEW'
    EARNINGS = 'EARNINGS'


DATA_TYPE_TABLES = {
    FundamentalDataType.OVERVIEW: {
        'table_name': 'company_overview',
        'columns': ['exchange', 'country', 'sector', 'industry', 'market_capitalization', 'book_value', 'dividend_yield', 'eps', 'price_to_book_ratio', 'beta', '52_week_high', '52_week_low'],
        'include_index': False,
    },
    FundamentalDataType.EARNINGS: {
        'table_name': 'quarterly_earnings',
        'columns': ['reported_eps', 'estimated_eps', 'surprise', 'surprise_percentage'],
        'include_index': True,
    },
}


def fetch_fundamental_data(api_client: AlphaVantageClient, symbol: str, data_type: FundamentalDataType):
    params = {
        'function': data_type.value,
        'symbol': symbol,
    }
    return api_client.fetch(**params)


def parse_overview(data: dict):
    data = {key: [value] for key, value in data.items()}
    df = pd.DataFrame(data)
    df.columns = [inflection.underscore(col) for col in df.columns]
    df = df.filter(items=DATA_TYPE_TABLES[FundamentalDataType.OVERVIEW]['columns'])
    return df


def parse_earnings(data: dict):
    key = 'quarterlyEarnings'
    if key not in data:
        raise ValueError(f"Unexpected response format: '{key}' key not found")

    df = pd.DataFrame(data[key])
    df.set_index('fiscalDateEnding', inplace=True)
    df.index.name = 'fiscal_date_ending'
    df.index = pd.to_datetime(df.index)

    df.columns = [inflection.underscore(col) for col in df.columns]
    df = df.filter(items=DATA_TYPE_TABLES[FundamentalDataType.EARNINGS]['columns'])
    print(df.head())
    return df


def update_all_fundamental_data(api_client: AlphaVantageClient, symbol: str):
    for data_type in FundamentalDataType:
        response = fetch_fundamental_data(api_client, symbol, data_type)
        df = DATA_TYPE_PARSERS[data_type](response)
        df['symbol'] = symbol
        print(f'{data_type.value} data')
        print(df.head())
        store_data(df, table_name=DATA_TYPE_TABLES[data_type]['table_name'], index=DATA_TYPE_TABLES[data_type]['include_index'])


# Functions to parse fundamental various types of data responses into dataframes
DATA_TYPE_PARSERS = {
    FundamentalDataType.OVERVIEW: parse_overview,
    FundamentalDataType.EARNINGS: parse_earnings,
}
