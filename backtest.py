# This is a backtesting script that takes a strategy name, start_date, and end_date from the command line and runs
# the strategy on historical data from the DB. The script will output the account value at the end of the backtest,
# along with charts to show when positions were opened/closed, and the account value over time.


import argparse
import pandas as pd

from account import Account, BacktestAccount
from core_stock import CORE_STOCK_TABLE_NAME
from strategy.base_strategy import BaseStrategy
from strategy.test_strategy import TestStrategy
from storage import select_all_by_symbol


def backtest(strategy: BaseStrategy, symbols: set[str], start_date: str, end_date: str) -> float:
    # Iterate through every day between start_date and end_date, and call the strategy's evaluate method
    # on each day. The strategy will return positions traded on that day.
    df = select_all_by_symbol(CORE_STOCK_TABLE_NAME, symbols, start_date=start_date, end_date=end_date)

    daily_account_values = []
    daily_positions = []  # List of tuples (date, position)
    for date in pd.date_range(start_date, end_date):
        # Get the 'symbol' and 'open' columns for every row where the 'date' column matches the current date
        current_prices = df.loc[df['date'] == date, ['symbol', 'open']]

        # Skip weekends/holidays
        if current_prices.shape[0] == 0:
            continue

        positions = strategy.evaluate(date, current_prices)
        for position in positions:
            current_price = float(current_prices.loc[current_prices['symbol'] == position.symbol, 'open'].iloc[0])
            strategy.account.add_position(position, current_price)
            daily_positions.append((date, position))

        # Convert df to dict for current_prices
        current_prices = current_prices.set_index('symbol').to_dict()['open']
        daily_account_values.append(strategy.account.get_account_values(current_prices))
        print(f"{date}: {daily_account_values[-1]}")

    # TODO: plot the daily account values and daily positions
    return daily_account_values[-1]


def get_account(account_id: str, start_value: float) -> Account:
    if account_id:
        # Create a new BacktestAccount, using the details from the real account that we fetch from a broker API
        pass
    return BacktestAccount(account_id, "Backtest Account", start_value)


def get_strategy(strategy_name: str, account: Account, symbols: set[str]) -> BaseStrategy:
    if strategy_name == "TestStrategy":
        return TestStrategy(account, symbols)

    raise ValueError(f"Unknown strategy_name: {strategy_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest a trading strategy")
    parser.add_argument("--strategy_name", type=str, required=True, help="Name of the strategy to backtest")
    parser.add_argument("--start_date", type=str, required=True, help="Start date for the backtest in YYYY-MM-DD format")
    parser.add_argument("--end_date", type=str, required=True, help="End date for the backtest in YYYY-MM-DD format")
    parser.add_argument("--symbols", type=str, nargs="+", help="List of symbols to backtest. Leave empty to backtest on entire watchlist.")

    # Args for testing with a real account
    parser.add_argument("--account_id", type=str, help="Account ID to backtest with a real account. Leave empty for dummy account")

    # Args for testing with a dummy account
    # Can add a list of open positions, past trades, etc. in the dummy account later
    parser.add_argument("--start_cash_balance", type=float, help="Initial cash balance in the dummy account")
    args = parser.parse_args()

    account = get_account(args.account_id, args.start_cash_balance)
    strategy = get_strategy(args.strategy_name, account, set(args.symbols))

    account_value = backtest(strategy, set(args.symbols), args.start_date, args.end_date)
    print(f"Final account value: {account_value}")
