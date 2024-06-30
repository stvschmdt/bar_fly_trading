import requests
import os


class AlphaVantageClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def fetch(self, **kwargs):
        kwargs['apikey'] = self.api_key
        response = requests.get(self.base_url, params=kwargs)
        response.raise_for_status()
        return response.json()

    def fetch_historical_data(self, symbol, outputsize='compact'):
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'outputsize': outputsize,
            'apikey': self.api_key
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        return response.json()

    def fetch_treasury_yield(self, interval='daily', maturity='2year'):
        params = {
            'function': 'TREASURY_YIELD',
            'interval': interval,
            'maturity': maturity,
            'apikey': self.api_key
        }
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        return response.json()


api_key = os.getenv('ALPHAVANTAGE_API_KEY', None)
if not api_key:
    raise Exception('Must set ALPHAVANTAGE_API_KEY environment variable')
alpha_client = AlphaVantageClient(api_key)

