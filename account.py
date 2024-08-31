import pandas as pd

from order import Order, OrderOperation, StockOrder, OptionOrder, MultiLegOrder
from abc import ABC, abstractmethod


class Account(ABC):
    def __init__(self, account_id: str, owner_name: str, cash_balance: float, orders: dict[str, Order] = None,
                 held_symbols: set[str] = None, stock_positions: dict[str, int] = None, option_positions: dict[str, int] = None):
        self.account_id = account_id
        self.owner_name = owner_name
        self.cash_balance = cash_balance
        # Append-only order history {order_id: Order} - may want to make this a separate class in the future
        self.orders = orders if orders else {}
        self.held_symbols = held_symbols if held_symbols else set()
        # Open stock positions {symbol: quantity}
        self.stock_positions = {} if not stock_positions else stock_positions
        # Open option positions {contract_id: quantity}
        self.option_positions = {} if not option_positions else option_positions

    @abstractmethod
    def execute_order(self, order: Order, current_stock_price: float):
        pass

    @abstractmethod
    def get_num_shares_of_symbol(self, symbol: str):
        pass

    def get_max_buyable_shares(self, price: float, percent_of_cash: float = 1.0):
        return (self.cash_balance * percent_of_cash) // price

    def close_position(self, symbol: str, current_price: float):
        pass


# TODO: Create a real account type for interacting with a broker API


class BacktestAccount(Account):
    def __init__(self, account_id: str, owner_name: str, cash_balance: float, orders: dict[str, Order] = None,
                 held_symbols: set[str] = None, stock_positions: dict[str, int] = None, option_positions: dict[str, int] = None):
        super().__init__(account_id, owner_name, cash_balance, orders, held_symbols, stock_positions, option_positions)

    @classmethod
    def backtest_account_from_file(cls, file_path: str):
        # TODO: Implement this method to read account details from a file and return a BacktestAccount
        pass

    def execute_order(self, order: Order, current_stock_price: float):
        position_value = order.calculate_current_value(current_stock_price, order.order_date)
        if order.order_operation == OrderOperation.BUY:
            if self.cash_balance < position_value:
                raise ValueError(f"Insufficient funds to buy {order.symbol}")
            self.cash_balance -= position_value
        else:
            self.cash_balance += position_value

        if isinstance(order, MultiLegOrder):
            for leg in order.orders:
                self._add_to_open_positions(leg)
        else:
            self._add_to_open_positions(order)

        self.held_symbols.add(order.symbol)
        self.orders[order.order_id] = order

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
            if self.orders[contract_id].symbol == symbol:
                return

        self.held_symbols.remove(symbol)

    def get_account_values(self, current_prices: dict[str, float], option_data: dict[str, pd.DataFrame]):
        missing_symbols = self.held_symbols - set(current_prices.keys())
        if missing_symbols:
            raise ValueError(f"Current prices are missing for some held symbols: {missing_symbols}")

        equity_value = 0
        for symbol, quantity in self.stock_positions.items():
            equity_value += quantity * current_prices[symbol]

        for contract_id, quantity in self.option_positions.items():
            symbol = self.orders[contract_id].symbol
            df = option_data[symbol]
            equity_value += df.loc[df['contractID'] == contract_id, ['mark']] * quantity

        return round(self.cash_balance, 2), round(equity_value, 2)
