from position import Position
import uuid
from abc import ABC, abstractmethod


class Account(ABC):
    def __init__(self, account_id: str, owner_name: str, cash_balance: float, positions: dict[uuid.UUID, Position],
                 held_symbols: set[str]):
        self.account_id = account_id
        self.owner_name = owner_name
        self.cash_balance = cash_balance
        self.positions = positions
        self.held_symbols = held_symbols

    @abstractmethod
    def add_position(self, position: Position):
        pass

    def close_position(self, position_id: uuid.UUID, current_price: float):
        pass

    def get_account_values(self, current_prices: dict[str, float]):
        pass


class BacktestAccount(Account):
    def __init__(self, account_id: str, owner_name: str, cash_balance: float, positions: dict[uuid.UUID, Position],
                 held_symbols: set[str]):
        super().__init__(account_id, owner_name, cash_balance, positions, held_symbols)

    def add_position(self, position: Position):
        self.held_symbols.add(position.symbol)
        self.positions[position.get_position_id()] = position

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
