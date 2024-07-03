import storage
from collector import alpha_client
from core_stock import update_core_stock_data
from economic_indicator import update_all_economic_indicators
from fundamental_data import update_all_fundamental_data
from technical_indicator import update_all_technical_indicators


def main():
    # Fetch data for NVDA
    symbol = 'NVDA'

    # Update core stock data
    try:
        # TODO: Add incremental flag to only fetch new data
        update_core_stock_data(alpha_client, symbol, incremental=False)
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return

    # Update technical indicators
    try:
        update_all_technical_indicators(alpha_client, symbol)
    except Exception as e:
        print(f"Error updating technical indicator data: {e}")

    # Update all fundamental data
    try:
        update_all_fundamental_data(alpha_client, symbol)
    except Exception as e:
        print(f"Error fetching fundamental data: {e}")
        return

    # Fetch and parse economic indicators
    try:
        # TODO: Add incremental flag to only fetch new data
        update_all_economic_indicators(alpha_client, incremental=False)
    except Exception as e:
        print(f"Error updating economic indicators: {e}")

    for table_name in ['core_stock', 'technical_indicators', 'company_overview', 'quarterly_earnings', 'economic_indicators']:
        order_by_overrides = {
            'company_overview': '',
            'quarterly_earnings': 'fiscal_date_ending',
        }
        storage.select_all_from_table(table_name, order_by_overrides.get(table_name, 'date'))


if __name__ == "__main__":
    main()

