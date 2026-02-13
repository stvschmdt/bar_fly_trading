from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from account.account import Account
from order import Order


class BaseStrategy(ABC):
    def __init__(self, account: Account, symbols: set[str]):
        self.account = account
        self.symbols = symbols

    @abstractmethod
    def evaluate(self, date: datetime.date, current_prices: pd.DataFrame, options_data: pd.DataFrame) -> list[Order]:
        pass
