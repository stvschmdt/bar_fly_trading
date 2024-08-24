from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from account import Account
from position import Position


class BaseStrategy(ABC):
    def __init__(self, account: Account, symbols: set[str]):
        self.account = account
        self.symbols = symbols

    @abstractmethod
    def evaluate(self, date: datetime.date, current_prices: pd.DataFrame) -> list[Position]:
        pass
