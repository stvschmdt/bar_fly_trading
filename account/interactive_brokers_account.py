from account.account import Account, PositionType
from order import Order


class InteractiveBrokersAccount(Account):
    def execute_order(self, order: Order, current_stock_price: float):
        pass

    def get_num_shares_of_symbol(self, symbol: str):
        pass

    def get_positions(self, symbol=None, position_type: PositionType = PositionType.ALL):
        pass
