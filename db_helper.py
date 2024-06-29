import pandas as pd
from sqlalchemy import create_engine, text

def query_db(db_name, table):
    engine = create_engine(f'sqlite:///{db_name}')
    query = f"SELECT * FROM {table}"
    df = pd.read_sql(query, con=engine)
    return df

def query_stocks(db_name='stock_data.db'):
    return query_db(db_name, table='daily_data')

def query_sma_indicators(db_name='stock_data.db'):
    return query_db(db_name, table='sma_indicators')

def query_rsi_indicators(db_name='stock_data.db'):
    return query_db(db_name, table='rsi_indicators')

def query_macd_indicators(db_name='stock_data.db'):
    return query_db(db_name, table='macd_indicators')

def query_treasury_yield_indicators(db_name='stock_data.db'):
    return query_db(db_name, table='treasury_yield_indicators')

def query_cpi_indicators(db_name='stock_data.db'):
    return query_db(db_name, table='cpi_indicators')

def query_unemployment_indicators(db_name='stock_data.db'):
    return query_db(db_name, table='unemployment_indicators')

def query_company_overview(db_name='stock_data.db'):
    return query_db(db_name, table='fundamental_overview')

def query_earnings(db_name='stock_data.db'):
    return query_db(db_name, table='fundamental_earnings')

def show_tables(db_name):
    engine = create_engine(f'sqlite:///{db_name}')
    query = text("SELECT name FROM sqlite_master WHERE type='table';")
    with engine.connect() as connection:
        result = connection.execute(query)
        tables = result.fetchall()
    return [table[0] for table in tables]

def complex_query(db_name, symbol='AAPL', start_date='2024-01-01', end_date='2024-12-31'):
    engine = create_engine(f'sqlite:///{db_name}')
    # ensure all dates are in the YYYY-MM-DD format before joining

    
    query = f"""
        SELECT 
            s.date, s.symbol, s.open, s.high, s.low, s.close, s.adjusted_close, s.volume, s.dividend_amount, s.split_coefficient,
            sma.sma,
            rsi.rsi,
            macd.macd, macd.macd_signal, macd.macd_hist,
            ty.treasury_yield, 
            cpi.cpi, 
            ue.unemployment_rate,
            co.marketCapitalization, co.peRatio, co.dividendYield,
            e.reportedEPS, e.estimatedEPS, e.surprise, e.surprisePercentage
        FROM daily_data s
        LEFT JOIN sma_indicators sma ON s.date = sma.date AND s.symbol = sma.symbol
        LEFT JOIN rsi_indicators rsi ON s.date = rsi.date AND s.symbol = rsi.symbol
        LEFT JOIN macd_indicators macd ON s.date = macd.date AND s.symbol = macd.symbol
        LEFT JOIN treasury_yield_indicators ty ON s.date = ty.date
        LEFT JOIN cpi_indicators cpi ON s.date = cpi.date
        LEFT JOIN unemployment_indicators ue ON s.date = ue.date
        LEFT JOIN (
            SELECT symbol, marketCapitalization, peRatio, dividendYield, date 
            FROM fundamental_overview
        ) co ON s.symbol = co.symbol
        LEFT JOIN (
            SELECT symbol, date, reportedEPS, estimatedEPS, surprise, surprisePercentage
            FROM fundamental_earnings
        ) e ON s.symbol = e.symbol AND s.date = e.date
        WHERE s.symbol = '{symbol}' AND s.date BETWEEN '{start_date}' AND '{end_date}'
        GROUP BY s.date, s.symbol
        ORDER BY s.date
    """
    
    df = pd.read_sql(query, con=engine)
    return df

def main():
    db_name = 'stock_data.db'
    
    print("Tables in stock_data.db:")
    stock_tables = show_tables(db_name)
    print(stock_tables)
    
    df_stocks = query_stocks(db_name)
    print("\nStocks DataFrame Head:")
    print(df_stocks.head())
    
    df_sma = query_sma_indicators(db_name)
    print("\nSMA Indicators DataFrame Head:")
    print(df_sma.head())
    
    df_rsi = query_rsi_indicators(db_name)
    print("\nRSI Indicators DataFrame Head:")
    print(df_rsi.head())
    
    df_macd = query_macd_indicators(db_name)
    print("\nMACD Indicators DataFrame Head:")
    print(df_macd.head())
    
    df_treasury_yield = query_treasury_yield_indicators(db_name)
    print("\nTreasury Yield Indicators DataFrame Head:")
    print(df_treasury_yield.head())
    
    df_cpi = query_cpi_indicators(db_name)
    print("\nCPI Indicators DataFrame Head:")
    print(df_cpi.head())
    
    df_unemployment = query_unemployment_indicators(db_name)
    print("\nUnemployment Rate Indicators DataFrame Head:")
    print(df_unemployment.head())
    
    df_company_overview = query_company_overview(db_name)
    print("\nCompany Overview DataFrame Head:")
    print(df_company_overview.head())
    
    df_earnings = query_earnings(db_name)
    print("\nEarnings DataFrame Head:")
    print(df_earnings.head())
    
    df_complex = complex_query(db_name, symbol='AAPL', start_date='2024-01-01', end_date='2024-12-31')
    print("\nComplex Query DataFrame Head:")
    print(df_complex.head())

    print("\nComplex Query DataFrame Columns:")
    print(df_complex.columns)
    
    # NaN analysis on complex query result
    if df_complex.isnull().values.any():
        print("\nNaN values found in Complex Query DataFrame:")
        nan_counts = df_complex.isnull().sum()
        print(nan_counts)
        
        # Analyze why there might be NaNs
        print("\nAnalysis of NaN values:")
        for column, count in nan_counts.items():
            if count > 0:
                if column in ['treasury_yield', 'cpi', 'unemployment_rate']:
                    print(f"Column '{column}' might have NaNs because economic data is often released at different intervals (e.g., monthly, quarterly), leading to gaps when joined with daily stock data.")
                else:
                    print(f"Column '{column}' might have NaNs due to missing technical indicator values for some dates.")
    else:
        print("\nNo NaN values found in Complex Query DataFrame.")
    
    # Additional analysis of NaN values
    print("\nNaN Analysis in Complex Query DataFrame:")
    nan_analysis = df_complex.isnull().sum()
    print(nan_analysis)

if __name__ == "__main__":
    main()

