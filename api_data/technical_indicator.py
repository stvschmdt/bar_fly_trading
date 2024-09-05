from enum import Enum
from functools import reduce

from api_data.collector import AlphaVantageClient
from api_data.storage import store_data
from api_data.util import drop_existing_rows, get_table_write_option

import pandas as pd


class TechnicalIndicatorType(Enum):
    SMA = 'SMA'
    EMA = 'EMA'
    MACD = 'MACD'
    RSI = 'RSI'
    ADX = 'ADX'
    ATR = 'ATR'
    BBANDS = 'BBANDS'


TYPE_TIME_PERIODS = {
    TechnicalIndicatorType.SMA: [20, 50, 200],
    TechnicalIndicatorType.EMA: [20, 50, 200],
    TechnicalIndicatorType.MACD: [None],
    TechnicalIndicatorType.RSI: [14],
    TechnicalIndicatorType.ADX: [14],
    TechnicalIndicatorType.ATR: [14],
    TechnicalIndicatorType.BBANDS: [20],
}

TECHNICAL_INDICATORS_TABLE_NAME = 'technical_indicators'
DATE_COL = 'date'


def fetch_technical_data(api_client: AlphaVantageClient, symbol: str, indicator_type: TechnicalIndicatorType, time_period: int, **kwargs):
    params = {
        'function': indicator_type.value,
        'symbol': symbol,
        'interval': kwargs.get('interval', 'daily'),
        'series_type': kwargs.get('series_type', 'close'),
    }
    if time_period:
        params['time_period'] = time_period

    return api_client.fetch(**params)


def parse_technical_data(data: dict, indicator_type: TechnicalIndicatorType):
    key = f'Technical Analysis: {indicator_type.value}'
    if key not in data:
        raise ValueError(f"Unexpected response format: '{key}' key not found")

    df = pd.DataFrame.from_dict(data[key], orient='index')
    df.index = pd.to_datetime(df.index)
    df.index.name = DATE_COL
    # All data in the df is numeric at this stage (no symbol yet), so we can just blindly apply numeric to the whole df.
    df = df.apply(pd.to_numeric)

    return df


# We handle bbands separately from the other technical indicator types because they're represented by 3 numbers,
# while others are just a single number. Therefore, we need to format them slightly differently.
def get_bbands(api_client: AlphaVantageClient, symbol: str, time_period: int):
    response = fetch_technical_data(api_client, symbol, TechnicalIndicatorType.BBANDS, time_period)
    df = parse_technical_data(response, TechnicalIndicatorType.BBANDS)
    df.reset_index()
    df = df.rename(columns={
        'Real Upper Band': f'{TechnicalIndicatorType.BBANDS.value.lower()}_upper_{time_period}',
        'Real Middle Band': f'{TechnicalIndicatorType.BBANDS.value.lower()}_middle_{time_period}',
        'Real Lower Band': f'{TechnicalIndicatorType.BBANDS.value.lower()}_lower_{time_period}',
    })
    df.index.name = DATE_COL
    #print(f'{TechnicalIndicatorType.BBANDS} {time_period}')
    #print(df.head())
    return df


def get_all_technical_indicators(api_client: AlphaVantageClient, symbol: str):
    dfs = []
    for indicator_type, time_periods in TYPE_TIME_PERIODS.items():
        if indicator_type == TechnicalIndicatorType.BBANDS:
            # We handle BBANDS separately because they're represented by 3 numbers instead of 1, like the rest of the types.
            continue

        for time_period in time_periods:
            # Fetch and parse data
            response = fetch_technical_data(api_client, symbol, indicator_type, time_period)
            df = parse_technical_data(response, indicator_type)

            # Filter out any fields we don't need (just grabbing indicator type, like SMA, EMA, etc.)
            df = df.filter(items=[indicator_type.value])
            #print(f'{indicator_type} {time_period}')
            #print(df.head())
            df.reset_index()
            # Add time period into column name
            df.columns = [f'{indicator_type.value.lower()}_{time_period}' if time_period else indicator_type.value.lower()]
            dfs.append(df)

    for time_period in TYPE_TIME_PERIODS[TechnicalIndicatorType.BBANDS]:
        dfs.append(get_bbands(api_client, symbol, time_period))

    # Merge all dfs on date column
    merged_df = reduce(lambda left, right: pd.merge(left, right, on=DATE_COL, how='outer'), dfs)

    # Add symbol to every row
    merged_df['symbol'] = symbol
    return merged_df


def update_all_technical_indicators(api_client: AlphaVantageClient, symbol: str, incremental: bool = True):
    df = get_all_technical_indicators(api_client, symbol)
    if incremental:
        df = drop_existing_rows(df, TECHNICAL_INDICATORS_TABLE_NAME, DATE_COL, symbol)

    #print('technical_indicator data')
    #print(df.head())
    store_data(df, table_name=TECHNICAL_INDICATORS_TABLE_NAME, write_option=get_table_write_option(incremental))
