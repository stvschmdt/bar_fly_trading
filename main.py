# main.py

from collector import alpha_client
from parser import parse_historical_data, parse_technical_indicator, parse_treasury_yield
from storage import create_database, store_data, test_connection
import pandas as pd

def main():
    # Fetch data for NVDA
    symbol = 'NVDA'
    try:
        historical_data = alpha_client.fetch_historical_data(symbol)
        df_historical = parse_historical_data(historical_data)
    except ValueError as e:
        print(f"Error fetching historical data: {e}")
        return
    
    # Fetch and parse technical indicators
    try:
        sma_data = alpha_client.fetch_technical_indicator(symbol, 'SMA', time_period=20, series_type='open')
        df_sma = parse_technical_indicator(sma_data, 'SMA')
    except ValueError as e:
        print(f"Error fetching SMA data: {e}")
        df_sma = None
    
    try:
        macd_data = alpha_client.fetch_technical_indicator(symbol, 'MACD', series_type='open')
        df_macd = parse_technical_indicator(macd_data, 'MACD')
    except ValueError as e:
        print(f"Error fetching MACD data: {e}")
        df_macd = None
    
    try:
        rsi_data = alpha_client.fetch_technical_indicator(symbol, 'RSI', time_period=14, series_type='open')
        df_rsi = parse_technical_indicator(rsi_data, 'RSI')
    except ValueError as e:
        print(f"Error fetching RSI data: {e}")
        df_rsi = None
    
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
    df_merged = df_historical.copy()
    if df_treasury_2year is not None:
        df_merged = df_merged.join(df_treasury_2year, how='left')
    if df_treasury_10year is not None:
        df_merged = df_merged.join(df_treasury_10year, how='left')
    
    store_data(df_merged, table_name=f'{symbol}_historical')
    if df_sma is not None:
        store_data(df_sma, table_name=f'{symbol}_sma')
    if df_macd is not None:
        store_data(df_macd, table_name=f'{symbol}_macd')
    if df_rsi is not None:
        store_data(df_rsi, table_name=f'{symbol}_rsi')
    if df_treasury_2year is not None:
        store_data(df_treasury_2year, table_name='treasury_2year')
    if df_treasury_10year is not None:
        store_data(df_treasury_10year, table_name='treasury_10year')
    
    # Print the first few rows and columns of the dataframe
    print("Historical Data Columns:")
    print(df_merged.columns)
    print("\nHistorical Data First Few Rows:")
    print(df_merged.head())
    
    if df_sma is not None:
        print("\nSMA Data Columns:")
        print(df_sma.columns)
        print("\nSMA Data First Few Rows:")
        print(df_sma.head())
    
    if df_macd is not None:
        print("\nMACD Data Columns:")
        print(df_macd.columns)
        print("\nMACD Data First Few Rows:")
        print(df_macd.head())
    
    if df_rsi is not None:
        print("\nRSI Data Columns:")
        print(df_rsi.columns)
        print("\nRSI Data First Few Rows:")
        print(df_rsi.head())
    
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

