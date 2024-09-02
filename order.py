from abc import ABC, abstractmethod
from enum import Enum

from api_data.historical_option_data import get_option_data
from api_data.collector import alpha_client


class OrderOperation(Enum):
    BUY = 1
    SELL = 2


class OptionType(Enum):
    CALL = 1
    PUT = 2


# TODO: Add OrderExecutionMethod (Market, Limit, Stop, etc.)


class Order(ABC):
    def __init__(self, order_id: str, symbol: str, order_operation: OrderOperation, quantity: int, entry_price: float, order_date: str):
        self.order_id = order_id
        self.symbol = symbol
        self.order_operation = order_operation
        self.quantity = quantity
        self.entry_price = entry_price
        self.order_date = order_date

    def __str__(self):
        return f"{self.order_operation.name} {self.quantity} {self.symbol} at ${self.entry_price:.2f} on {self.order_date}"

    @abstractmethod
    def calculate_current_value(self, current_price: float, date: str) -> float:
        pass


class StockOrder(Order):
    def __init__(self, symbol: str, order_operation: OrderOperation, quantity: int, entry_price: float, order_date: str):
        order_id = f'{symbol}_{order_operation.name}_{quantity}_{entry_price}'
        super().__init__(order_id, symbol, order_operation, quantity, entry_price, order_date)
        self.order_operation = order_operation

    def calculate_current_value(self, current_price: float, date: str) -> float:
        return self.quantity * current_price


class OptionOrder(Order):
    def __init__(self, symbol: str, order_operation: OrderOperation, contract_id: str, option_type: OptionType, quantity: int,
                 entry_price: float, strike_price: float, expiration_date: str, order_date: str):
        super().__init__(contract_id, symbol, order_operation, quantity, entry_price, order_date)
        self.order_operation = order_operation
        self.option_type = option_type
        self.strike_price = strike_price
        self.expiration_date = expiration_date

    def calculate_current_value(self, current_price: float, date: str) -> float:
        df = get_option_data(alpha_client, self.symbol, date)
        return df.loc[df['contractID'] == self.order_id, ['mark']] * self.quantity


class MultiLegOrder(Order):
    def __init__(self, orders: list[Order], order_date: str):
        self.order_id = f"{'_'.join([order.order_id for order in orders])}"
        self.order_date = order_date

        self.validate_multileg_order(orders)
        # Assume all spreads have the same symbol, at least to begin with.
        self.symbol = orders[0].symbol
        self.orders = orders

    def calculate_current_value(self, current_price: float, date: str) -> float:
        return sum([order.calculate_current_value(current_price, date) for order in self.orders])

    @staticmethod
    def validate_multileg_order(orders: list[Order]):
        if len(orders) < 2:
            raise ValueError("A multi-leg order must contain at least two orders")

        at_least_one_option = False
        for i in range(1, len(orders)):
            if orders[i].symbol != orders[0].symbol:
                raise ValueError("All orders in a spread must have the same symbol")
            if isinstance(orders[i], OptionOrder):
                at_least_one_option = True

        if not at_least_one_option:
            raise ValueError("A multi-leg order must contain at least one option order")
