from position import Position, OrderOperation
import uuid
from abc import ABC, abstractmethod


class Account(ABC):
    def __init__(self, account_id: str, owner_name: str, cash_balance: float, positions: dict[uuid.UUID, Position] = None,
                 held_symbols: set[str] = None):
        self.account_id = account_id
        self.owner_name = owner_name
        self.cash_balance = cash_balance
        # TODO: Change to orders {orderID: Order}
        self.positions = positions if positions else {}
        self.held_symbols = held_symbols if held_symbols else set()
        # TODO: Take as input: {symbol: quantity}
        self.stock_positions = {}
        # TODO: Take as input: {contractID: quantity}
        self.option_positions = {}

    @abstractmethod
    def add_position(self, position: Position, current_stock_price: float):
        pass

    @abstractmethod
    def get_num_shares_of_symbol(self, symbol: str):
        pass

    def get_max_buyable_shares(self, price: float):
        return self.cash_balance // price

    def close_position(self, position_id: uuid.UUID, current_price: float):
        pass

    def get_account_values(self, current_prices: dict[str, float]):
        pass


# TODO: Create a real account type for interacting with a broker API


class BacktestAccount(Account):
    def __init__(self, account_id: str, owner_name: str, cash_balance: float, positions: dict[uuid.UUID, Position] = None,
                 held_symbols: set[str] = None):
        super().__init__(account_id, owner_name, cash_balance, positions, held_symbols)

    def add_position(self, position: Position, current_stock_price: float):
        position_value = position.calculate_current_value(current_stock_price)
        if position.order_operation == OrderOperation.BUY:
            if self.cash_balance < position_value:
                raise ValueError(f"Insufficient funds to buy {position.symbol}")
            self.cash_balance -= position_value
        else:
            self.cash_balance += position_value

        self.held_symbols.add(position.symbol)
        self.positions[position.get_position_id()] = position

    def get_num_shares_of_symbol(self, symbol: str):
        num_shares = 0
        for position in self.positions.values():
            if position.symbol == symbol:
                num_shares += position.quantity
        return num_shares

    def close_position(self, position_id: uuid.UUID, current_price: float):
        if position_id not in self.positions:
            raise ValueError(f"Position with ID {position_id} not found in account")

        self._remove_position(position_id)
        # Talk to executor to actually close the position and get new cash balance

    def _remove_position(self, position_id: uuid.UUID):
        position = self.positions.pop(position_id)
        found_symbol = False

        # If this was the last position for this symbol, remove it from the held symbols set
        for p in self.positions.values():
            if p.symbol == position.symbol:
                found_symbol = True

        if not found_symbol:
            self.held_symbols.remove(position.symbol)

    def get_account_values(self, current_prices: dict[str, float]):
        missing_symbols = self.held_symbols - set(current_prices.keys())
        if missing_symbols:
            raise ValueError(f"Current prices are missing for some held symbols: {missing_symbols}")

        equity_value = 0
        for position in self.positions.values():
            equity_value += position.calculate_current_value(current_prices[position.symbol])
        return self.cash_balance, equity_value
