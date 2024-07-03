from enum import Enum
from functools import reduce
import inflection

from collector import AlphaVantageClient
from storage import store_data

import numpy as np
import pandas as pd


class EconomicIndicatorType(Enum):
    TREASURY_YIELD = 'TREASURY_YIELD'
    FEDERAL_FUNDS_RATE = 'FEDERAL_FUNDS_RATE'
    CPI = 'CPI'
    INFLATION = 'INFLATION'
    RETAIL_SALES = 'RETAIL_SALES'
    DURABLES = 'DURABLES'
    UNEMPLOYMENT = 'UNEMPLOYMENT'
    NONFARM_PAYROLL = 'NONFARM_PAYROLL'


# This map holds intervals for the types that need them, and column name overrides.
# If a column name isn't specified, it'll default to the value of the enum in snake_case.
TYPE_OVERRIDES = {
    EconomicIndicatorType.TREASURY_YIELD: {
        'column_name': 'treasury_yield_{}',
        'interval': 'daily',
        'maturities': ['2year', '10year']
    },
    EconomicIndicatorType.FEDERAL_FUNDS_RATE: {
        'column_name': 'ffer',
        'interval': 'daily',
    },
    EconomicIndicatorType.CPI: {
        'interval': 'monthly',
    },
}
ECONOMIC_INDICATOR_TABLE_NAME = 'economic_indicators'


def fetch_economic_data(api_client: AlphaVantageClient, indicator_type: EconomicIndicatorType, **kwargs):
    params = {
        'function': indicator_type.value,
    }
    params.update(kwargs)

    return api_client.fetch(**params)


def parse_economic_data(data: dict, indicator_type: EconomicIndicatorType):
    key = 'data'
    if key not in data:
        raise ValueError(f"Unexpected response format: '{key}' key not found")

    df = pd.DataFrame(data[key])
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index)

    column_name = TYPE_OVERRIDES.get(indicator_type, {}).get('column_name', inflection.underscore(indicator_type.value))
    df = df.rename(columns={'value': column_name})
    df[column_name] = df[column_name].replace('.', np.nan)
    print(f'{indicator_type.value} data')
    print(df.head())
    return df


def get_treasury_yields(api_client: AlphaVantageClient):
    dfs = []
    for maturity in TYPE_OVERRIDES[EconomicIndicatorType.TREASURY_YIELD]['maturities']:
        response = fetch_economic_data(
            api_client, EconomicIndicatorType.TREASURY_YIELD,
            maturity=maturity,
            interval=TYPE_OVERRIDES[EconomicIndicatorType.TREASURY_YIELD]['interval']
        )
        df = parse_economic_data(response, EconomicIndicatorType.TREASURY_YIELD)
        # Format the maturity into the column name
        df.rename(columns={df.columns[0]: df.columns[0].format(maturity)}, inplace=True)
        dfs.append(df)

    df = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), dfs)
    return df


def update_all_economic_indicators(api_client: AlphaVantageClient, incremental: bool = True):
    dfs = []
    for data_type in EconomicIndicatorType:
        # Treasuries get handled a little differently because they have multiple maturities
        if data_type == EconomicIndicatorType.TREASURY_YIELD:
            df = get_treasury_yields(api_client)
            dfs.append(df)
            continue

        interval = TYPE_OVERRIDES.get(data_type, {}).get('interval')
        params = {'interval': interval} if interval else {}
        response = fetch_economic_data(api_client, data_type, **params)
        df = parse_economic_data(response, data_type)
        dfs.append(df)

    df = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), dfs)
    print('economic_indicator data')
    pd.set_option('display.max_columns', None)
    print(df.head())
    store_data(df, table_name=ECONOMIC_INDICATOR_TABLE_NAME)
