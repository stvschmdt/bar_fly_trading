from logger import logger
import pandas as pd
import numpy as np
from storage import select_all_by_symbol
from fundamental_data import FundamentalDataType, fetch_fundamental_data, DATA_TYPE_TABLES
from collector import alpha_client


class ColumnValidator:
    def __init__(self, api_field_name: str, db_col_name: str = None, conversions=None):
        self.api_field_name = api_field_name
        self.db_col_name = db_col_name if db_col_name else api_field_name
        # Mapping of values we change from the API to the DB. e.g. 'None' -> 0
        self.conversions = conversions if conversions else {}

    def validate(self, api_val, db_val) -> bool:
        api_val = self.conversions.get(api_val, api_val)
        if api_val == 'None' and np.isnan(db_val):
            # Pandas -> MySQL does this conversion for us
            return True

        equal = pd.to_numeric(api_val, errors='ignore') == db_val
        if not equal:
            logger.error(f"Column {self.db_col_name} does not match for {self.api_field_name}: "
                         f"API value: {api_val}, DB value: {db_val}")
        return equal


DATA_POINTS = {
    FundamentalDataType.OVERVIEW: [
        ColumnValidator('Exchange', 'exchange'), ColumnValidator('Country', 'country'), ColumnValidator('Sector', 'sector'), ColumnValidator('Industry', 'industry'),
        ColumnValidator('MarketCapitalization', 'market_capitalization'), ColumnValidator('BookValue', 'book_value'), ColumnValidator('DividendYield', 'dividend_yield', {'None': 0}),
        ColumnValidator('EPS', 'eps'), ColumnValidator('PriceToBookRatio', 'price_to_book_ratio', {'None': 0, '-': None}), ColumnValidator('Beta', 'beta'),
        ColumnValidator('52WeekHigh', '52_week_high'), ColumnValidator('52WeekLow', '52_week_low'), ColumnValidator('ForwardPE', 'forward_pe'),
    ],
    FundamentalDataType.EARNINGS: [
        ColumnValidator('reportedEPS', 'reported_eps'), ColumnValidator('estimatedEPS', 'estimated_eps'), ColumnValidator('surprise', 'surprise'), ColumnValidator('surprisePercentage', 'surprise_percentage'),
    ]
}


def log_result(symbol, data_type, all_match):
    if all_match:
        logger.info(f'{symbol} {data_type} data matches')
    else:
        logger.error(f'{symbol} {data_type} data does not match')


def validate_company_overview(symbol):
    logger.info(f'Validating {symbol} company overview data')
    data_type = FundamentalDataType.OVERVIEW
    api_data = fetch_fundamental_data(alpha_client, symbol, data_type)
    table_name = DATA_TYPE_TABLES[data_type]['table_name']
    df = select_all_by_symbol(table_name, symbol)
    # There can only be one row because symbol is the primary key
    row = df.iloc[0]
    all_match = True

    for col in DATA_POINTS[data_type]:
        all_match = all_match and col.validate(api_data[col.api_field_name], row[col.db_col_name])

    log_result(symbol, data_type, all_match)


def validate_quarterly_earnings(symbol):
    logger.info(f'Validating {symbol} quarterly earnings data')
    data_type = FundamentalDataType.EARNINGS
    api_data = fetch_fundamental_data(alpha_client, symbol, data_type)['quarterlyEarnings']
    table_name = DATA_TYPE_TABLES[data_type]['table_name']
    df = select_all_by_symbol(table_name, symbol)
    df = df.sort_values(by='fiscal_date_ending', ascending=False)
    df.reset_index(drop=True, inplace=True)
    all_match = True

    for i, row in df.iterrows():
        api_row = api_data[i]
        if api_row is None:
            logger.error(f"API data does not contain fiscal_date_ending: {row['fiscal_date_ending']}")
            all_match = False
            continue

        for col in DATA_POINTS[data_type]:
            if not col.validate(api_row[col.api_field_name], row[col.db_col_name]):
                all_match = False
                logger.error(f"Data doesn't match on {row['fiscal_date_ending']} API data: {api_row[col.api_field_name]}, DB data: {row[col.db_col_name]}")

    log_result(symbol, data_type, all_match)


if __name__ == '__main__':
    symbols = ['AAPL']
    for symbol in symbols:
        validate_company_overview(symbol)
        validate_quarterly_earnings(symbol)
