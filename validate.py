import pandas as pd
from logger import Logging
import argparse


logger = Logging()


def check_duplicate_rows(data):
    ohlc_cols = ['open', 'high', 'low', 'adjusted_close', 'volume']
    dups = data[ohlc_cols].duplicated()
    duplicates = dups.sum()
    if duplicates == 0:
        logger.info('No duplicate rows found')
    else:
        logger.error(f'{duplicates} duplicate rows found, {duplicates/data.shape[0]*100} percent of data, please check the data.')
    return duplicates, duplicates/data.shape[0]*100

def check_missing_daily_values(data):
    cols = ['open', 'high', 'low', 'adjusted_close', 'volume', 'sma_20','sma_50', 'sma_200', 'ema_20', 'ema_50', 'ema_200', 'macd', 'rsi_14',
       'bbands_upper_20', 'bbands_middle_20', 'bbands_lower_20']
    missing_values = data[cols].isnull().sum().sum()
    if missing_values == 0:
        logger.info('No missing values found')
    else:
        logger.error(f'{missing_values} missing values found, {missing_values/data.shape[0]*100} percent of data, please check the data.')
    return missing_values, missing_values/data.shape[0]*100

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--csv", help="flat file to validate data", type=str, default='all_data.csv')
    args = parser.parse_args()
    data = pd.read_csv(args.csv)
    check_duplicate_rows(data)
    check_missing_daily_values(data)

