import logging
import os
import threading
from datetime import datetime
from time import sleep

import requests

from logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# The actual rate-limit is 150, but we want to be able to make a few ad-hoc requests while things are running.
DEFAULT_MAX_REQUESTS_PER_MIN = 140
RATE_LIMIT_STRING = 'Thank you for using Alpha Vantage! Please contact premium@alphavantage.co if you are targeting a higher API call volume.'

# This lock is used to ensure that the rate-limiting logic is thread-safe.
lock = threading.Lock()


class AlphaVantageClient:
    def __init__(self, api_key, max_requests_per_min=DEFAULT_MAX_REQUESTS_PER_MIN):
        self.api_key = api_key
        self.max_requests_per_min = max_requests_per_min
        self.requests_in_window = 0
        self.window_start_time = datetime.now()
        self.base_url = "https://www.alphavantage.co/query"
        self.max_attempts = 5

    def fetch(self, num_attempt=1, **kwargs):
        if self.requests_in_window >= self.max_requests_per_min:
            # Use timedelta to get the elapsed time in seconds since self.window_start_time
            elapsed_time = (datetime.now() - self.window_start_time).seconds
            if elapsed_time < 60:
                # Sleep for the remaining time in the minute plus a little buffer
                sleep_time = 60 - elapsed_time + 5
                logger.info(f"Internal rate limit reached. Sleeping for {sleep_time} seconds.")
                sleep(sleep_time)

            with lock:
                # We check the requests_in_window again because it could have been updated by another thread.
                if self.requests_in_window >= self.max_requests_per_min:
                    self.requests_in_window = 0
            self.window_start_time = datetime.now()

        with lock:
            self.requests_in_window += 1
        kwargs['apikey'] = self.api_key

        try:
            response = requests.get(self.base_url, params=kwargs)
            response.raise_for_status()
            response = response.json()

            if response.get('Information', '') == RATE_LIMIT_STRING:
                logger.info(
                    f"AlphaVantage rate limit reached (requests_in_window={self.requests_in_window}). Sleeping for 60 seconds.")
                sleep(60)
                with lock:
                    # We check the requests_in_window again because it could have been updated by another thread.
                    if self.requests_in_window >= self.max_requests_per_min:
                        self.requests_in_window = 0
                self.window_start_time = datetime.now()
                return self.retry(num_attempt, **kwargs)
        except Exception as e:
            logger.error(f"Error fetching data from AlphaVantage - url:{self.base_url}, kwargs:{get_kwargs_without_api_key(kwargs)}, error={e}")
            sleep(2)
            return self.retry(num_attempt, **kwargs)

        return response

    def retry(self, num_attempt, **kwargs):
        # Retry if we have attempts left
        if num_attempt < self.max_attempts:
            next_attempt = num_attempt + 1
            logger.info(f"Retrying request num_attempt={next_attempt}")
            return self.fetch(next_attempt, **kwargs)
        raise Exception(f'all AlphaVantage retries failed, url={self.base_url}, kwargs={get_kwargs_without_api_key(kwargs)}')


def get_kwargs_without_api_key(kwargs):
    return {k: v for k, v in kwargs.items() if k != 'apikey'}


api_key = os.getenv('ALPHAVANTAGE_API_KEY', None)
if not api_key:
    raise Exception('Must set ALPHAVANTAGE_API_KEY environment variable')
alpha_client = AlphaVantageClient(api_key)

