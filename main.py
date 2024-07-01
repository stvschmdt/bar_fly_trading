from collector import alpha_client
from core_stock import update_core_stock_data
from fundamental_data import update_all_fundamental_data
from parser import parse_historical_data, parse_treasury_yield
from storage import store_data
from technical_indicator import update_all_technical_indicators


def main():
    # Fetch data for NVDA
    symbol = 'NVDA'

    # Update core stock data
    try:
        # TODO: Add incremental flag to only fetch new data
        update_core_stock_data(alpha_client, symbol, incremental=False)
    except ValueError as e:
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
    except ValueError as e:
        print(f"Error fetching fundamental data: {e}")
        return
    
    # Fetch and parse treasury yield data
    try:
        treasury_2year_data = alpha_client.fetch_treasury_yield(maturity='2year')
        df_treasury_2year = parse_treasury_yield(treasury_2year_data, '2yr')
    except ValueError as e:
        print(f"Error fetching 2-year treasury yield data: {e}")
        df_treasury_2year = None
    
    try:
        treasury_10year_data = alpha_client.fetch_treasury_yield(maturity='10year')
        df_treasury_10year = parse_treasury_yield(treasury_10year_data, '10yr')
    except ValueError as e:
        print(f"Error fetching 10-year treasury yield data: {e}")
        df_treasury_10year = None
    
    # Merge historical data with treasury yields
    if df_treasury_2year is not None:
        store_data(df_treasury_2year, table_name='treasury_2year')
    if df_treasury_10year is not None:
        store_data(df_treasury_10year, table_name='treasury_10year')
    
    if df_treasury_2year is not None:
        print("\n2-Year Treasury Yield Data Columns:")
        print(df_treasury_2year.columns)
        print("\n2-Year Treasury Yield Data First Few Rows:")
        print(df_treasury_2year.head())
    
    if df_treasury_10year is not None:
        print("\n10-Year Treasury Yield Data Columns:")
        print(df_treasury_10year.columns)
        print("\n10-Year Treasury Yield Data First Few Rows:")
        print(df_treasury_10year.head())


if __name__ == "__main__":
    main()

