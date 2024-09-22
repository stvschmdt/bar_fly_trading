from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd

from account.account import Account, PositionType
from account.account_values import AccountValues
from order import Order, OrderOperation, MultiLegOrder, StockOrder, OptionOrder
from order_history import OrderHistory


class BacktestAccount(Account):
    def __init__(self, account_id: str, owner_name: str, account_values: AccountValues, order_history: OrderHistory = None,
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

    def get_positions(self, symbol=None, position_type: PositionType = PositionType.ALL) -> dict:
        result = {}
        if position_type == PositionType.STOCK:
            result['stock'] = self._get_stock_positions(symbol)
        elif position_type == PositionType.OPTION:
            result['options'] = self._get_option_positions(symbol)
        else:
            result['stock'] = self._get_stock_positions(symbol)
            result['options'] = self._get_option_positions(symbol)
        return result

    def _get_stock_positions(self, symbol=None) -> dict:
        return {symbol: self.stock_positions.get(symbol, 0)} if symbol else self.stock_positions

    def _get_option_positions(self, symbol=None) -> dict:
        result = defaultdict(dict)
        for contract_id, quantity in self.option_positions.items():
            contract_symbol = self.order_history.get_order(contract_id).symbol
            if not symbol or contract_symbol == symbol:
                result[contract_symbol][contract_id] = quantity

        if symbol and symbol not in result:
            result[symbol] = {}
        return result

    def get_position_values(self, symbol=None, position_type: PositionType = PositionType.ALL, current_prices: dict[str, float] = None, option_data: dict[str, pd.DataFrame] = None) -> dict:
        result = {}
        if position_type == PositionType.STOCK:
            result['stock'] = self._get_stock_position_values(symbol, current_prices)
        elif position_type == PositionType.OPTION:
            result['options'] = self._get_option_position_values(symbol, option_data)
        else:
            result['stock'] = self._get_stock_position_values(symbol, current_prices)
            result['options'] = self._get_option_position_values(symbol, option_data)
        return result

    def _get_stock_position_values(self, symbol=None, current_prices: dict[str, float] = None) -> dict:
        if symbol:
            if symbol not in current_prices:
                raise ValueError(f"Current price missing for given symbol {symbol}")
            return {symbol: self.stock_positions.get(symbol, 0) * current_prices[symbol]}

        stock_values = {}
        for owned_symbol, quantity in self.stock_positions.items():
            if owned_symbol not in current_prices:
                raise ValueError(f"Current price missing for owned symbol {owned_symbol}")
            stock_values[owned_symbol] = quantity * current_prices[owned_symbol]
        return stock_values

    def _get_option_position_values(self, symbol=None, option_data: dict[str, pd.DataFrame] = None) -> dict:
        result = defaultdict(lambda: defaultdict(float))
        for contract_id, quantity in self.option_positions.items():
            order = self.order_history.get_order(contract_id)
            if symbol and symbol not in option_data:
                raise ValueError(f"Option data missing for symbol {symbol}")
            if not symbol or order.symbol == symbol:
                # We don't need a multiplier here because if we've sold more of these options than we bought, the quantity will be negative.
                option_price = float(option_data[order.symbol].loc[option_data[order.symbol]['contractID'] == contract_id, ['mark']].iloc[0])
                result[order.symbol][contract_id] += quantity * option_price

        if symbol and symbol not in result:
            result[symbol] = defaultdict(float)
        return result
