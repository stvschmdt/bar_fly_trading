import logging
import os
from datetime import datetime
from time import sleep

import requests

from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# The actual rate-limit is 150, but we want to be able to make a few ad-hoc requests while things are running.
DEFAULT_MAX_REQUESTS_PER_MIN = 140
RATE_LIMIT_STRING = 'Thank you for using Alpha Vantage! Please contact premium@alphavantage.co if you are targeting a higher API call volume.'


class AlphaVantageClient:
    def __init__(self, api_key, max_requests_per_min=DEFAULT_MAX_REQUESTS_PER_MIN):
        self.api_key = api_key
        self.max_requests_per_min = max_requests_per_min
        self.requests_in_window = 0
        self.window_start_time = datetime.now()
        self.base_url = "https://www.alphavantage.co/query"

    def fetch(self, **kwargs):
        if self.requests_in_window >= self.max_requests_per_min:
            # Use timedelta to get the elapsed time in seconds since self.window_start_time
            elapsed_time = (datetime.now() - self.window_start_time).seconds
            if elapsed_time < 60:
                # Sleep for the remaining time in the minute plus a little buffer
                sleep_time = 60 - elapsed_time + 5
                logger.info(f"Internal rate limit reached. Sleeping for {sleep_time} seconds.")
                sleep(sleep_time)
                self.requests_in_window = 0
                self.window_start_time = datetime.now()

        self.requests_in_window += 1
        kwargs['apikey'] = self.api_key
        response = requests.get(self.base_url, params=kwargs)
        response.raise_for_status()
        response = response.json()
        if response.get('Information', '') == RATE_LIMIT_STRING:
            logger.info(f"AlphaVantage rate limit reached (requests_in_window={self.requests_in_window}). Sleeping for 60 seconds.")
            sleep(60)
            self.requests_in_window = 0
            self.window_start_time = datetime.now()

            # Retry if we haven't already
            if kwargs.get('depth', 0) == 0:
                logger.info(f"Retrying request with depth=1")
                kwargs['depth'] = 1
                return self.fetch(**kwargs)
        return response


api_key = os.getenv('ALPHAVANTAGE_API_KEY', None)
if not api_key:
    raise Exception('Must set ALPHAVANTAGE_API_KEY environment variable')
alpha_client = AlphaVantageClient(api_key)

