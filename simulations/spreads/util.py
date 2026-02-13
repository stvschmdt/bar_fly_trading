from __future__ import annotations

import bisect
import pandas as pd


def find_nearest_strike(strike_prices: list[float], current_price: float, desired_percent_deviation: float):
    if desired_percent_deviation > 0:
        idx = bisect.bisect_right(strike_prices, current_price)
    else:
        idx = bisect.bisect_left(strike_prices, current_price) - 1

    percent_difference = 0
    adjustment = 1 if desired_percent_deviation > 0 else -1
    while 0 <= idx < len(strike_prices):
        new_percent_difference = (strike_prices[idx] - current_price) / current_price
        if abs(new_percent_difference - desired_percent_deviation) > abs(percent_difference - desired_percent_deviation):
            return strike_prices[idx - adjustment]

        percent_difference = new_percent_difference
        idx += adjustment

    return ValueError(f"No strike price with desired_percent_deviation {desired_percent_deviation} of current_price "
                      f"{current_price}. Strike prices between [{strike_prices[0]}, {strike_prices[-1]}].")


def get_expiration_date(expiration_dates: list[str], start_date: str, days_to_expiration: int):
    for date in expiration_dates:
        # If the date is at least days_to_expiration days after start_date, return it
        if (pd.Timestamp(date) - pd.Timestamp(start_date)).days >= days_to_expiration:
            return date
    return ValueError(f"No expiration date found {days_to_expiration} days after start_date {start_date}.")


class MarketClosedError(Exception):
    def __init__(self, message="Market is closed on this date - probably weekend or holiday"):
        self.message = message
        super().__init__(self.message)
