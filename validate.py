import pandas as pd
from logger import Logging

logger = Logging()


def check_duplicate_rows(filename='all_data.csv'):
    data = pd.read_csv('all_data.csv')
    ohlc_cols = ['open', 'high', 'low', 'adjusted_close', 'volume']
    dups = data[ohlc_cols].duplicated()
    duplicates = dups.sum()
    if duplicates == 0:
        logger.info('No duplicate rows found')
    else:
        logger.error(f'{duplicates} duplicate rows found, {duplicates/data.shape[0]*100} percent of data, please check the data.')
    return duplicates, duplicates/data.shape[0]*100


if __name__ == '__main__':
    check_duplicate_rows()
