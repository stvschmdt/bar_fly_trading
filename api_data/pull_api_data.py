import argparse
import logging
import os
import sys

import pandas as pd

from constants import WATCHLIST_PATH, SP500_PATH, SECTORS_PATH

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from api_data.collector import alpha_client
from api_data.core_stock import update_core_stock_data
from api_data.economic_indicator import update_all_economic_indicators
from api_data.fundamental_data import update_all_fundamental_data
from api_data.storage import process_gold_table_in_batches
from api_data.technical_indicator import update_all_technical_indicators
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
incremental = True


def main():
    global incremental

    if not args.gold_table_only:
        # Fetch and parse economic indicators. This data is not symbol-specific, so we only need to do this once.
        try:
            update_all_economic_indicators(alpha_client, incremental=incremental)
        except Exception as e:
            print(f"Error updating economic indicators: {e}")

        for symbol in symbols:
            # Update core stock data
            try:
                update_core_stock_data(alpha_client, symbol, incremental=incremental)
            except Exception as e:
                print(f"Error fetching historical data: {e}")

            # Update technical indicators
            try:
                update_all_technical_indicators(alpha_client, symbol, incremental=incremental)
            except Exception as e:
                print(f"Error updating technical indicator data: {e}")

            # Update all fundamental data
            try:
                update_all_fundamental_data(alpha_client, symbol, incremental=incremental, is_etf=symbol in etfs)
            except Exception as e:
                print(f"Error fetching fundamental data: {e}")

            # If incremental is False, that means we want to drop the existing tables and re-insert all the data.
            # To avoid dropping the tables on every iteration, we set incremental to True after the first iteration.
            if not incremental:
                incremental = True

    process_gold_table_in_batches(symbols)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--watchlist", help="file containing symbols to fetch and store data in db. Use `all` for S&P 500 + watchlist.", type=str, default='api_data/watchlist.csv')
    parser.add_argument("-t", "--test", help="number of symbols for testing functionality", type=int, default=600)
    parser.add_argument("-s", "--symbols", help="list of symbols to fetch data for", nargs='+', type=str, default=[])
    parser.add_argument("--gold-table-only", help="only update the gold table", action="store_true")
    args = parser.parse_args()
    logger.info(f"Watchlist file: {args.watchlist}")
    logger.info(f"Test Size: {args.test}")
    try:
        if args.symbols:
            symbols = args.symbols
        elif args.watchlist == 'all':
            watchlist = pd.read_csv(WATCHLIST_PATH)
            sp500 = pd.read_csv(SP500_PATH)['Symbol']
            symbols = pd.concat([watchlist, sp500]).drop_duplicates().reset_index(drop=True).tolist()
        else:
            # read in csv file of watch list
            symbols = pd.read_csv(args.watchlist)['Symbol'].tolist()
            symbols = [symbol.upper() for symbol in symbols][0:args.test]
        logger.info(f"Symbols: {symbols}")
        sectors = set(pd.read_csv(SECTORS_PATH)['Symbol'])
        etfs = set(pd.read_csv(WATCHLIST_PATH).query('`Is ETF` == 1')['Symbol'])
        etfs.update(sectors)
        main()
    except Exception as e:
        print(f"Error: {e}")

