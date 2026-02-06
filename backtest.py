# This is a backtesting script that takes a strategy name, start_date, and end_date from the command line and runs
# the strategy on historical data from the DB. The script will output the account value at the end of the backtest,
# along with charts to show when positions were opened/closed, and the account value over time.


import argparse
import os
import sys

import pandas as pd

# Add strategies dir to path for base_strategy and strategy classes
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'strategies'))

from account.account import Account
from account.account_values import AccountValues
from account.backtest_account import BacktestAccount
from order import OptionOrder, OrderOperation
from base_strategy import BaseStrategy


def backtest(strategy: BaseStrategy, symbols: set[str], start_date: str, end_date: str,
             data: pd.DataFrame = None) -> AccountValues:
    # Iterate through every day between start_date and end_date, and call the strategy's evaluate method
    # on each day. The strategy will return positions traded on that day.
    if data is not None:
        # CSV-based: use provided DataFrame instead of database
        core_stock_df = data[data['symbol'].isin(symbols)].copy()
        core_stock_df['date'] = pd.to_datetime(core_stock_df['date'])
        core_stock_df = core_stock_df[
            (core_stock_df['date'] >= pd.to_datetime(start_date)) &
            (core_stock_df['date'] <= pd.to_datetime(end_date))
        ]
        # Ensure 'open' column exists (some CSVs only have adjusted_close)
        if 'open' not in core_stock_df.columns and 'adjusted_close' in core_stock_df.columns:
            core_stock_df['open'] = core_stock_df['adjusted_close']
        historical_options_df = pd.DataFrame()
    else:
        # Legacy SQL path — only import DB modules when actually needed
        from api_data.core_stock import CORE_STOCK_TABLE_NAME
        from api_data.historical_options import HISTORICAL_OPTIONS_TABLE_NAME
        from api_data.storage import select_all_by_symbol
        core_stock_df = select_all_by_symbol(CORE_STOCK_TABLE_NAME, symbols, start_date=start_date, end_date=end_date)
        historical_options_df = select_all_by_symbol(HISTORICAL_OPTIONS_TABLE_NAME, symbols, start_date=start_date, end_date=end_date)

    daily_positions = []  # List of tuples (date, position)
    for date in pd.date_range(start_date, end_date):
        # Get the 'symbol' and 'open' columns for every row where the 'date' column matches the current date
        current_prices = core_stock_df.loc[core_stock_df['date'] == date, ['symbol', 'open', 'adjusted_close', 'high', 'low']]
        current_options = historical_options_df[historical_options_df['date'] == date] if not historical_options_df.empty else pd.DataFrame()

        # Skip weekends/holidays
        if current_prices.shape[0] == 0:
            continue
        # Skip days with no options data (only when options data is available)
        if not historical_options_df.empty and current_options.shape[0] == 0:
            continue

        # Convert current_prices df to dict
        symbol_price_map = current_prices.set_index('symbol').to_dict()['open']

        orders = strategy.evaluate(date, current_prices, current_options)
        new_account_values = None
        for order in orders:
            if isinstance(order, OptionOrder):
                # Get today's option data for this contract
                option_row = current_options[current_options['contract_id'] == order.contract_id]
                bid_ask = 'ask' if order.order_operation == OrderOperation.BUY else 'bid'
                current_price = float(option_row[bid_ask].iloc[0])
            else:
                current_price = float(current_prices.loc[current_prices['symbol'] == order.symbol, 'open'].iloc[0])
            
            strategy.account.execute_order(order, current_price)
            new_account_values = strategy.account.update_account_values(pd.to_datetime(date), symbol_price_map, historical_options_df)
            daily_positions.append((date, order))
            print(f"{date}: {order}")
            print(f"New account values: {new_account_values}")

        # If there were orders, we will have already updated the account values.
        # If not, make sure we update them to account for any price changes of held positions.
        if not new_account_values:
            new_account_values = strategy.account.update_account_values(pd.to_datetime(date), symbol_price_map, historical_options_df)
        print(f"{date} Account values: {new_account_values}")

    # TODO: write account details to some file, then plot the daily account values and daily positions
    return strategy.account.account_values


def get_account(account_id: str, start_value: float, start_date: str) -> BacktestAccount:
    if account_id:
        # Create a new BacktestAccount, using the details from the real account that we fetch from a broker API
        pass
    start_date = pd.to_datetime(start_date)
    return BacktestAccount(account_id, "Backtest Account", AccountValues(start_value, 0, 0), start_date)


def get_strategy(strategy_name: str, account: Account, symbols: set[str]) -> BaseStrategy:
    # Import strategies on demand — only needed when backtest.py is run directly
    if strategy_name == "MLPredictionStrategy":
        from ml_prediction_strategy import MLPredictionStrategy
        return MLPredictionStrategy(account, symbols)
    elif strategy_name == "RegressionMomentumStrategy":
        from regression_momentum_strategy import RegressionMomentumStrategy
        return RegressionMomentumStrategy(account, symbols)

    raise ValueError(f"Unknown strategy_name: {strategy_name}. "
                     f"Use run_template.py, run_backtest.py, or run_regression_momentum.py instead.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest a trading strategy")
    parser.add_argument("--strategy_name", type=str, required=True, help="Name of the strategy to backtest")
    parser.add_argument("--start_date", type=str, required=True, help="Start date for the backtest in YYYY-MM-DD format")
    parser.add_argument("--end_date", type=str, required=True, help="End date for the backtest in YYYY-MM-DD format")
    parser.add_argument("--symbols", type=str, nargs="+", help="List of symbols to backtest. Leave empty to backtest on entire watchlist.")
    parser.add_argument("--db", help="database to use, either 'local' or 'remote' - defaults to local.", type=str, default='local')

    # Args for testing with a real account
    parser.add_argument("--account_id", type=str, help="Account ID to backtest with a real account. Leave empty for dummy account")

    # Args for testing with a dummy account
    # Can add a list of open positions, past trades, etc. in the dummy account later
    parser.add_argument("--start_cash_balance", type=float, help="Initial cash balance in the dummy account")
    args = parser.parse_args()

    account = get_account(args.account_id, args.start_cash_balance, args.start_date)
    strategy = get_strategy(args.strategy_name, account, set(args.symbols))
    # Legacy direct-run CLI uses SQL — all strategy runners use CSV via data= param instead
    from api_data.storage import connect_database
    connect_database(args.db)

    account_values = backtest(strategy, set(args.symbols), args.start_date, args.end_date)
    print(f"Final account values: {account_values}")
