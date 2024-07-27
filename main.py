from collector import alpha_client
from core_stock import update_core_stock_data
from economic_indicator import update_all_economic_indicators
from fundamental_data import update_all_fundamental_data
from technical_indicator import update_all_technical_indicators


incremental = False
SYMBOLS = ['NVDA', 'AAPL']


def main():
    global incremental

    for symbol in SYMBOLS:
        # Update core stock data
        try:
            update_core_stock_data(alpha_client, symbol, incremental=incremental)
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return

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
            return

        # Fetch and parse economic indicators
        try:
            update_all_economic_indicators(alpha_client, incremental=incremental)
        except Exception as e:
            print(f"Error updating economic indicators: {e}")

        # If incremental is False, that means we want to drop the existing tables and re-insert all the data.
        # To avoid dropping the tables on every iteration, we set incremental to True after the first iteration.
        if not incremental:
            incremental = True


if __name__ == "__main__":
    main()

