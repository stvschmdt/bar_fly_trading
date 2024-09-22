from copy import deepcopy
from enum import Enum

from account.account_values import AccountValues
from order import Order
from order_history import OrderHistory
from abc import ABC, abstractmethod


class PositionType(Enum):
    ALL = 1
    STOCK = 2
    OPTION = 3


class Account(ABC):
    def __init__(self, account_id: str, owner_name: str, account_values: AccountValues, order_history: OrderHistory = None,
                 held_symbols: set[str] = None, stock_positions: dict[str, int] = None, option_positions: dict[str, int] = None):
        self.account_id = account_id
        self.owner_name = owner_name
        self.account_values = account_values
        self.order_history = order_history if order_history else OrderHistory()
        self.held_symbols = held_symbols if held_symbols else set()
        # Open stock positions {symbol: quantity}
        self.stock_positions = {} if not stock_positions else stock_positions
        # Open option positions {contract_id: quantity}
        self.option_positions = {} if not option_positions else option_positions

        if not account_values:
            raise ValueError("AccountValues must be provided to create an account.")

        # Used to track P/L, should never be updated after instantiation
        self._initial_cash_balance = deepcopy(account_values.cash_balance)
        # {timestamp: account_value}
        self.account_value_history = self._construct_account_value_history()

    @property
    def initial_cash_balance(self):
        # This makes initial_cash_balance immutable from outside this class, ensuring no one changes it.
        # We should never have a set_initial_cash_balance method.
        return self._initial_cash_balance

    @abstractmethod
    def execute_order(self, order: Order, current_stock_price: float):
        pass

    @abstractmethod
    def get_num_shares_of_symbol(self, symbol: str):
        pass

    @abstractmethod
    def get_positions(self, symbol = None, position_type: PositionType = PositionType.ALL):
        pass

    def get_max_buyable_shares(self, price: float, percent_of_cash: float = 1.0):
        return (self.account_values.cash_balance * percent_of_cash) // price

    def close_position(self, symbol: str, current_price: float):
        pass

    def _construct_account_value_history(self):
        # A real account will go out to the broker API to get the account value history, backtest accounts will calculate it.
        pass
