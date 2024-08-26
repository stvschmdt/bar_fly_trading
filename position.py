from abc import ABC, abstractmethod
import uuid
from enum import Enum


class OrderOperation(Enum):
    BUY = 1
    SELL = 2


class OptionType(Enum):
    CALL = 1
    PUT = 2


class Position(ABC):
    def __init__(self, symbol: str, order_operation: OrderOperation, quantity: int, entry_price: float):
        self._position_id = uuid.uuid4()
        self.symbol = symbol
        self.order_operation = order_operation
        self.quantity = quantity
        self.entry_price = entry_price

    def get_position_id(self) -> uuid.UUID:
        return self._position_id

    @abstractmethod
    def calculate_current_value(self, current_price: float) -> float:
        pass


class StockPosition(Position):
    def __init__(self, symbol: str, order_operation: OrderOperation, quantity: int, entry_price: float):
        super().__init__(symbol, order_operation, quantity, entry_price)
        self.order_operation = order_operation

    def calculate_current_value(self, current_price: float) -> float:
        return self.quantity * current_price


class OptionPosition(Position):
    def __init__(self, symbol: str, order_operation: OrderOperation, option_type: OptionType, quantity: int,
                 entry_price: float, strike_price: float, expiration_date: str):
        super().__init__(symbol, order_operation, quantity, entry_price)
        self.order_operation = order_operation
        self.option_type = option_type
        self.strike_price = strike_price
        self.expiration_date = expiration_date

    def calculate_current_value(self, current_price: float) -> float:
        # TODO: Use historical option data to calculate value, but for now, just act as if it's regular stock
        return self.quantity * current_price


class OptionSpreadPosition(Position):
    def __init__(self, options: list[Position]):
        self.validate_option_spread(options)

        self._position = uuid.uuid4()
        # Assume all spreads have the same symbol, at least to begin with.
        self.symbol = options[0].symbol
        self.options = options

    def calculate_current_value(self, current_price: float) -> float:
        total_value = 0
        for option in self.options:
            total_value += option.calculate_current_value(current_price)
        return total_value

    @staticmethod
    def validate_option_spread(options: list[Position]):
        if len(options) < 2:
            raise ValueError("An option spread position must contain at least two options")

        at_least_one_option = False
        for i in range(1, len(options)):
            if options[i].symbol != options[0].symbol:
                raise ValueError("All options in a spread must have the same symbol")
            if isinstance(options[i], OptionPosition):
                at_least_one_option = True

        if not at_least_one_option:
            raise ValueError("An option spread must contain at least one option position")
