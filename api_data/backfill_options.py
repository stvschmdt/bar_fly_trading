import argparse
import logging
from concurrent import futures
from functools import partial

import pandas as pd

from api_data import historical_options
from api_data.collector import alpha_client
from api_data.storage import connect_database
from constants import WATCHLIST_PATH, SP500_PATH
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# This script takes either a list of symbols or a watchlist file and calls
# historical_options.update_historical_options() on each symbol between a given start_date and end_date.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backfill historical options data for a list of symbols')
    parser.add_argument('-s', '--symbols', type=str, nargs='+',
                        help='A list of symbols to backfill historical options data for')
    parser.add_argument('-w', '--watchlist',
                        help='file containing symbols to fetch and store data in db. Use `all` for S&P 500 + watchlist.',
                        type=str, default='api_data/watchlist.csv')
    parser.add_argument('--start_date', type=str, help='The start date to backfill from')
    parser.add_argument('--end_date', type=str, help='The end date to backfill to')
    parser.add_argument("-d", "--db", help="database to use, either 'local' or 'remote' - defaults to local.", type=str,
                        default='local')
    args = parser.parse_args()

    if args.symbols:
        symbols = args.symbols
    elif args.watchlist == 'all':
        watchlist = pd.read_csv(WATCHLIST_PATH)['Symbol']
        sp500 = pd.read_csv(SP500_PATH)['Symbol']
        symbols = pd.concat([watchlist, sp500]).drop_duplicates().reset_index(drop=True).tolist()
    else:
        # read in csv file of watch list
        symbols = pd.read_csv(args.watchlist)['Symbol'].tolist()
        symbols = [symbol.upper() for symbol in symbols][0:args.test]

    connect_database(args.db)

    # Process 3 symbols at a time. Any more and we'll hit rate limits too quickly.
    with futures.ThreadPoolExecutor(max_workers=3) as executor:
        update_options = partial(historical_options.update_historical_options, alpha_client, start_date=args.start_date, end_date=args.end_date)
        list(executor.map(update_options, symbols))
