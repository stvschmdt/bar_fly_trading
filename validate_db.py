import logging
import pandas as pd
import numpy as np
from storage import select_all_by_symbol
from fundamental_data import FundamentalDataType, fetch_fundamental_data, DATA_TYPE_TABLES
from logging_config import setup_logging
from technical_indicator import TechnicalIndicatorType, TYPE_TIME_PERIODS, fetch_technical_data, TECHNICAL_INDICATORS_TABLE_NAME
from collector import alpha_client

setup_logging()
logger = logging.getLogger(__name__)


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


def create_technical_datapoints():
    result = {}
    for indicator in TechnicalIndicatorType:
        for time_period in TYPE_TIME_PERIODS[indicator]:
            if indicator.value not in result:
                result[indicator.value] = {}

            if indicator == TechnicalIndicatorType.BBANDS:
                result[indicator.value][time_period] = [
                    ColumnValidator('Real Upper Band', f'bbands_upper_{time_period}'),
                    ColumnValidator('Real Middle Band', f'bbands_middle_{time_period}'),
                    ColumnValidator('Real Lower Band', f'bbands_lower_{time_period}'),
                ]
            else:
                db_col = f'{indicator.value.lower()}_{time_period}' if time_period else f'{indicator.value.lower()}'
                result[indicator.value][time_period] = [ColumnValidator(indicator.value, db_col)]
    return result


DATA_POINTS = {
    FundamentalDataType.OVERVIEW: [
        ColumnValidator('Exchange', 'exchange'), ColumnValidator('Country', 'country'), ColumnValidator('Sector', 'sector'), ColumnValidator('Industry', 'industry'),
        ColumnValidator('MarketCapitalization', 'market_capitalization'), ColumnValidator('BookValue', 'book_value'), ColumnValidator('DividendYield', 'dividend_yield', {'None': 0}),
        ColumnValidator('EPS', 'eps'), ColumnValidator('PriceToBookRatio', 'price_to_book_ratio', {'None': 0, '-': None}), ColumnValidator('Beta', 'beta'),
        ColumnValidator('52WeekHigh', '52_week_high'), ColumnValidator('52WeekLow', '52_week_low'), ColumnValidator('ForwardPE', 'forward_pe'),
    ],
    FundamentalDataType.EARNINGS: [
        ColumnValidator('reportedEPS', 'reported_eps'), ColumnValidator('estimatedEPS', 'estimated_eps'), ColumnValidator('surprise', 'surprise'), ColumnValidator('surprisePercentage', 'surprise_percentage'),
    ],
    "TechnicalIndicators": create_technical_datapoints()
}


def log_result(symbol, data_type, all_match):
    if all_match:
        logger.info(f'{symbol} {data_type} data matches')
    else:
        logger.info(f'{symbol} {data_type} data does not match')


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


def validate_technical_indicators(symbol):
    logger.info(f'Validating {symbol} technical indicators data')
    data_type = 'TechnicalIndicators'
    df = select_all_by_symbol(TECHNICAL_INDICATORS_TABLE_NAME, symbol)
    df = df.sort_values(by='date', ascending=False)
    df.reset_index(drop=True, inplace=True)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    all_match = True

    for indicator, time_periods in TYPE_TIME_PERIODS.items():
        for time_period in time_periods:
            api_data = fetch_technical_data(alpha_client, symbol, indicator, time_period)[f'Technical Analysis: {indicator.value}']
            dates = 0
            for _, row in df.iterrows():
                api_date_data = api_data.get(row['date'], None)
                if api_date_data:
                    dates += 1
                    for col in DATA_POINTS[data_type][indicator.value][time_period]:
                        if not col.validate(api_date_data[col.api_field_name], row[col.db_col_name]):
                            all_match = False
                            logging.error(f"Data doesn't match on {row['date']} API data: {api_date_data[col.api_field_name]}, DB data: {row[col.db_col_name]}")

    log_result(symbol, data_type, all_match)


if __name__ == '__main__':
    symbols = ['AAPL']
    for symbol in symbols:
        validate_company_overview(symbol)
        validate_quarterly_earnings(symbol)
        validate_technical_indicators(symbol)
