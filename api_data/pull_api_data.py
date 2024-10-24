import argparse
import logging
import os
import sys
import time

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from api_data.storage import gold_table_processing
from api_data.collector import alpha_client
from api_data.core_stock import update_core_stock_data
from api_data.economic_indicator import update_all_economic_indicators
from api_data.fundamental_data import update_all_fundamental_data
from api_data.technical_indicator import update_all_technical_indicators
from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
incremental = True


def main():
    global incremental

    # Fetch and parse economic indicators. This data is not symbol-specific, so we only need to do this once.
    try:
        update_all_economic_indicators(alpha_client, incremental=incremental)
    except Exception as e:
        print(f"Error updating economic indicators: {e}")
    timer = 0
    for symbol in SYMBOLS:
        timer += 1
        # Update core stock data
        try:
            update_core_stock_data(alpha_client, symbol, incremental=incremental)
        except Exception as e:
            print(f"Error fetching historical data: {e}")

        # # Update technical indicators
        try:
            update_all_technical_indicators(alpha_client, symbol, incremental=incremental)
        except Exception as e:
            print(f"Error updating technical indicator data: {e}")

        # Update all fundamental data
        try:
            update_all_fundamental_data(alpha_client, symbol, incremental=incremental)
        except Exception as e:
            print(f"Error fetching fundamental data: {e}")

        # If incremental is False, that means we want to drop the existing tables and re-insert all the data.
        # To avoid dropping the tables on every iteration, we set incremental to True after the first iteration.
        if not incremental:
            incremental = True
        # add a counter for api hits, sleep for 1 minute to reset the counter
        if timer >= 9:
            time.sleep(60)
            timer = 0
    gold_table_processing()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--watchlist", help="file containing symbols to fetch and store data in db", type=str, default='watchlist.csv')
    parser.add_argument("-t", "--test", help="number of symbols for testing functionality", type=int, default=500)
    parser.add_argument("-s", "--symbols", help="list of symbols to fetch data for", nargs='+', type=str, default=[])
    args = parser.parse_args()
    logger.info(f"Watchlist file: {args.watchlist}")
    logger.info(f"Test Size: {args.test}")
    try:
        if args.symbols:
            SYMBOLS = args.symbols
        else:
            # read in csv file of watch list
            SYMBOLS = pd.read_csv(args.watchlist)['Symbol'].tolist()
            # ticker symbol is first token after comma separation
            SYMBOLS = [symbol.split(',')[0] for symbol in SYMBOLS]
            SYMBOLS = [symbol.upper() for symbol in SYMBOLS][0:args.test]
        logger.info(f"Symbols: {SYMBOLS}")
        main()
    except Exception as e:
        print(f"Error: {e}")

