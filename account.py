from datetime import datetime, timedelta
from copy import deepcopy

import pandas as pd

from account_values import AccountValues
from order import Order, OrderOperation, StockOrder, OptionOrder, MultiLegOrder
from order_history import OrderHistory
from abc import ABC, abstractmethod


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

    def get_max_buyable_shares(self, price: float, percent_of_cash: float = 1.0):
        return (self.account_values.cash_balance * percent_of_cash) // price

    def close_position(self, symbol: str, current_price: float):
        pass

    def _construct_account_value_history(self):
        # A real account will go out to the broker API to get the account value history, backtest accounts will calculate it.
        pass


# TODO: Create a real account type for interacting with a broker API


class BacktestAccount(Account):
    def __init__(self, account_id: str, owner_name: str, account_values: AccountValues, order_history: dict[str, Order] = None,
                 held_symbols: set[str] = None, stock_positions: dict[str, int] = None, option_positions: dict[str, int] = None):
        super().__init__(account_id, owner_name, account_values, order_history, held_symbols, stock_positions, option_positions)

    def _construct_account_value_history(self):
        # TODO: Implement this
        # Since we know the current cash balance and the order history, we can calculate the account values at each date.
        # This is meant to be called on instantiation of a backtest account that already has orders executed.
        return {}

    @classmethod
    def backtest_account_from_file(cls, file_path: str):
        # TODO: Implement this method to read account details from a file and return a BacktestAccount
        pass

    def execute_order(self, order: Order, current_stock_price: float):
        position_value = order.calculate_current_value(current_stock_price, order.order_date)
        if order.order_operation == OrderOperation.BUY:
            if self.account_values.cash_balance < position_value:
                raise ValueError(f"Insufficient funds to buy {order.symbol}")
            self.account_values.cash_balance -= position_value
            self.held_symbols.add(order.symbol)
        else:
            self.account_values.cash_balance += position_value

        if isinstance(order, MultiLegOrder):
            for leg in order.orders:
                self._add_to_open_positions(leg)
        else:
            self._add_to_open_positions(order)

        self.order_history.add_order(order)

    def _add_to_open_positions(self, order: Order):
        if isinstance(order, MultiLegOrder):
            return ValueError("_add_to_open_positions must be called with a single-leg order")

        multiplier = 1 if order.order_operation == OrderOperation.BUY else -1
        if isinstance(order, StockOrder):
            self.stock_positions[order.symbol] = self.stock_positions.get(order.symbol, 0) + order.quantity * multiplier
        elif isinstance(order, OptionOrder):
            self.option_positions[order.order_id] = self.option_positions.get(order.order_id, 0) + order.quantity * multiplier

    def get_num_shares_of_symbol(self, symbol: str):
        return self.stock_positions.get(symbol, 0)

    def close_stock_position(self, symbol: str, current_price: float):
        if symbol not in self.stock_positions:
            raise ValueError(f"Position with ID {symbol} not found in account")

        quantity = self.stock_positions.pop(symbol)
        self._remove_held_symbol(symbol)
        # TODO: execute a buy/sell order to close the position

    def _remove_held_symbol(self, symbol: str):
        # Only remove the symbol from held_symbols if it's not in any open positions
        if symbol in self.stock_positions:
            return

        # Check if we have any open option orders with this symbol
        for contract_id in self.option_positions.keys():
            if self.order_history.get_order(contract_id).symbol == symbol:
                return

        self.held_symbols.remove(symbol)

    def get_account_values(self, current_prices: dict[str, float], option_data: dict[str, pd.DataFrame]) -> AccountValues:
        missing_symbols = self.held_symbols - set(current_prices.keys())
        if missing_symbols:
            raise ValueError(f"Current prices are missing for some held symbols: {missing_symbols}")

        stock_value = 0
        for symbol, quantity in self.stock_positions.items():
            stock_value += quantity * current_prices[symbol]

        options_value = 0
        for contract_id, quantity in self.option_positions.items():
            symbol = self.order_history.get_order(contract_id).symbol
            df = option_data[symbol]
            options_value += df.loc[df['contractID'] == contract_id, ['mark']] * quantity

        return AccountValues(round(self.account_values.cash_balance, 2), round(stock_value, 2), round(options_value, 2))

    def update_account_values(self, timestamp: datetime, current_prices: dict[str, float], option_data: dict[str, pd.DataFrame]):
        """
        Update the account values at the given timestamp. If the timestamp is already in account_value_history, we add
        one second, so we don't overwrite it. THIS SHOULD BE CALLED AFTER EVERY EXECUTE_ORDER CALL!
        Args:
            timestamp: time that the account values are being updated
            current_prices: current stock prices of all held symbols
            option_data: current options prices of all held symbols

        Returns:
            Updated AccountValues object
        """
        self.account_values = self.get_account_values(current_prices, option_data)
        # If the timestamp is already in account_value_history, add one second, so we don't overwrite it
        if timestamp in self.account_value_history:
            timestamp += timedelta(seconds=1)
        self.account_value_history[timestamp] = self.account_values
        return self.account_values
