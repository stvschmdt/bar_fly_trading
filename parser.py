# data_parsing.py

import pandas as pd

def parse_historical_data(data):
    if 'Time Series (Daily)' not in data:
        raise ValueError("Unexpected response format: 'Time Series (Daily)' key not found")
    
    time_series = data['Time Series (Daily)']
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df.index = pd.to_datetime(df.index)
    df.columns = [col.split(' ')[1] for col in df.columns]
    df = df.apply(pd.to_numeric)
    return df

def parse_treasury_yield(data, maturity):
    if 'data' not in data:
        raise ValueError("Unexpected response format: 'data' key not found")
    
    key = 'data'
    df = pd.DataFrame(data[key])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df[['value']].rename(columns={'value': f'{maturity}_yield'})
    return df

