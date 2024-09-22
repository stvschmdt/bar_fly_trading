class AccountValues:
    def __init__(self, cash_balance: float, stock_positions: float, option_positions: float):
        self.cash_balance = cash_balance
        self.stock_positions = stock_positions
        self.option_positions = option_positions

    def __str__(self):
        return f"Cash: ${self.cash_balance:,.2f}, Stocks: ${self.stock_positions:,.2f}, Options: ${self.option_positions:,.2f}, Total: ${self.cash_balance + self.stock_positions + self.option_positions:,.2f}"

    def __eq__(self, other):
        return self.cash_balance == other.cash_balance \
               and self.stock_positions == other.stock_positions \
               and self.option_positions == other.option_positions

    def get_cash_percentage(self):
        return self.cash_balance / (self.cash_balance + self.stock_positions + self.option_positions)

    def get_stock_percentage(self):
        return self.stock_positions / (self.cash_balance + self.stock_positions + self.option_positions)

    def get_option_percentage(self):
        return self.option_positions / (self.cash_balance + self.stock_positions + self.option_positions)

    def get_total_value(self):
        return self.cash_balance + self.stock_positions + self.option_positions
