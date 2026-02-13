from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, date

import pandas as pd

from account.account import Account, PositionType
from account.account_values import AccountValues
from api_data.collector import alpha_client
from api_data.historical_options import fetch_historical_options, parse_historical_options
from order import Order, OrderOperation, MultiLegOrder, StockOrder, OptionOrder
from order_history import OrderHistory
from util import extract_symbol_from_contract_id


class BacktestAccount(Account):
    def __init__(self, account_id: str, owner_name: str, account_values: AccountValues, start_date: datetime, order_history: OrderHistory = None,
                 held_symbols: set[str] = None, stock_positions: dict[str, int] = None, option_positions: dict[str, int] = None):
        super().__init__(account_id, owner_name, account_values, order_history, held_symbols, stock_positions, option_positions)
        self.account_value_history[start_date] = account_values

    def _construct_account_value_history(self) -> dict[date, AccountValues]:
        # TODO: Implement this
        # Since we know the current cash balance and the order history, we can calculate the account values at each date.
        # This is meant to be called on instantiation of a backtest account that already has orders executed.
        return {}

    @classmethod
    def backtest_account_from_file(cls, file_path: str):
        # TODO: Implement this method to read account details from a file and return a BacktestAccount
        pass

    def execute_order(self, order: Order, current_price: float):
        # TODO: Add commission to price of options orders: https://www.interactivebrokers.com/en/pricing/commissions-options.php?re=amer
        position_value = order.calculate_current_value(current_price, order.order_date)
        if order.order_operation == OrderOperation.BUY:
            if self.account_values.cash_balance < position_value:
                raise ValueError(f"Insufficient funds to buy {order.symbol}")
            self.held_symbols.add(order.symbol)

        if isinstance(order, MultiLegOrder):
            pass
            # TODO: Implement this. Might need to use a different method for calculating cost of executing multi-leg orders.
            # TODO: Also need to do above operations in a loop for each leg of the order.
            # for leg in order.orders:
            #     self._apply_order_to_open_positions(leg)
        else:
            self._apply_order_to_open_positions(order, position_value)

        # The order of operations is important, that's why we do this down here. We must remove the symbol from
        # held_symbols AFTER calling _apply_order_to_open_positions, so the symbol/contract is removed from the
        # respective map prior to calling _remove_held_symbol.
        if order.order_operation == OrderOperation.SELL:
            self._remove_held_symbol(order.symbol)
        self.order_history.add_order(order)

    def _apply_order_to_open_positions(self, order: Order, total_price: float):
        if isinstance(order, MultiLegOrder):
            raise ValueError("_apply_order_to_open_positions must be called with a single-leg order")

        multiplier = 1 if order.order_operation == OrderOperation.BUY else -1
        self.account_values.cash_balance -= total_price * multiplier

        if isinstance(order, StockOrder):
            self.stock_positions[order.symbol] = self.stock_positions.get(order.symbol, 0) + order.quantity * multiplier
            self.account_values.stock_positions += total_price * multiplier
            if self.stock_positions[order.symbol] == 0:
                self.stock_positions.pop(order.symbol)
        elif isinstance(order, OptionOrder):
            self.option_positions[order.contract_id] = self.option_positions.get(order.contract_id, 0) + order.quantity * multiplier
            self.account_values.option_positions += total_price * multiplier
            if self.option_positions[order.contract_id] == 0:
                self.option_positions.pop(order.contract_id)

    def get_num_shares_of_symbol(self, symbol: str):
        return self.stock_positions.get(symbol, 0)

    def close_stock_position(self, symbol: str, current_price: float):
        if symbol not in self.stock_positions:
            raise ValueError(f"Position with ID {symbol} not found in account")

        quantity = self.stock_positions.pop(symbol)
        self._remove_held_symbol(symbol)
        # TODO: execute a buy/sell order to close the position

    def _remove_held_symbol(self, symbol: str):
        # Can't remove it if it's not in held_symbols
        if symbol not in self.held_symbols:
            return

        # Only remove the symbol from held_symbols if it's not in any open positions
        if symbol in self.stock_positions:
            return

        # Check if any open option positions are for the given symbol. If so, we can't remove it from held_symbols yet
        # since we still have an active position in that symbol (via options)
        for contract_id in self.option_positions.keys():
            if self.order_history.get_order(contract_id).symbol == symbol:
                return

        self.held_symbols.remove(symbol)

    def get_account_values(self, current_prices: dict[str, float], option_data: pd.DataFrame) -> AccountValues:
        """
        Get account values for the current date.
        Args:
            current_prices: <symbol, price> mapping for all held symbols
            option_data: df of options data for held symbols on current date.

        Returns:
            AccountValues object with cash balance, stock value, and options value
        """
        missing_symbols = self.stock_positions.keys() - current_prices.keys()
        if missing_symbols:
            raise ValueError(f"Current prices are missing for some held stock symbols: {missing_symbols}")

        stock_value = 0
        for symbol, quantity in self.stock_positions.items():
            stock_value += quantity * current_prices[symbol]

        options_value = 0
        # Keep track of fetched option data to avoid duplicate API calls
        fetched_option_data = pd.DataFrame()
        
        for contract_id, quantity in self.option_positions.items():
            # Try to find contract in provided option data first
            contract_data = option_data[option_data['contract_id'] == contract_id]
            
            if not contract_data.empty:
                mark = float(contract_data['mark'].iloc[0])
            else:
                symbol = extract_symbol_from_contract_id(contract_id)
                date = option_data['date'].iloc[0].strftime('%Y-%m-%d')
                
                # Only fetch if we haven't already fetched data for this symbol/date
                if fetched_option_data.empty or not ((fetched_option_data['symbol'] == symbol) & (fetched_option_data['date'] == date)).any():
                    api_data = parse_historical_options(fetch_historical_options(alpha_client, symbol, date)['data'])
                    fetched_option_data = pd.concat([fetched_option_data, api_data], ignore_index=True)
                
                # Find the matching contract in the historical data
                contract_data = fetched_option_data[fetched_option_data['contract_id'] == contract_id]
                if contract_data.empty:
                    raise ValueError(f"Could not find historical data for contract {contract_id}")
                mark = float(contract_data['mark'].iloc[0])
                
            options_value += mark * quantity * 100
        return AccountValues(round(self.account_values.cash_balance, 2), round(stock_value, 2), round(options_value, 2))

    def update_account_values(self, timestamp: datetime, current_prices: dict[str, float], option_data: pd.DataFrame):
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
        # Get a string of today's date with time 00:00:00, so we can filter the options data.
        # If we start getting intraday options data, we'll want to match on closest time instead.
        date_without_time = datetime(timestamp.year, timestamp.month, timestamp.day).strftime('%Y-%m-%d %H:%M:%S')
        filtered_options = option_data.loc[option_data['date'] == date_without_time] if not option_data.empty else option_data
        self.account_values = self.get_account_values(current_prices, filtered_options)
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
                option_price = float(option_data[order.symbol].loc[option_data[order.symbol]['contractID'] == contract_id, 'mark'].iloc[0])
                result[order.symbol][contract_id] += quantity * option_price

        if symbol and symbol not in result:
            result[symbol] = defaultdict(float)
        return result

    def get_daily_account_value_percentage_change(self, start_date: str = None, end_date: str = None, change_from_start=True) -> pd.DataFrame:
        """
        Get the daily percentage change in account value from start_date to end_date. If change_from_start is True, we
        compare each day's account value to the given start_date. If False, we compare each day's account value to the
        previous day's account value. We assume there is an entry for every day in the account_value_history because
        the backtester should be updating it each day.
        Args:
            start_date: Date to start comparison. If empty, we start from the account's start date.
            end_date: Date to end comparison. If empty, we end at the account's last date.
            change_from_start: If true, compare each day's account value to the start_date. Else, compare to the previous day.

        Returns:
            df mapping date to daily percentage changes in account value
        """
        earliest_account_date = min(self.account_value_history.keys()).date()
        latest_account_date = max(self.account_value_history.keys()).date()

        if start_date:
            start_date = pd.to_datetime(start_date).date()
            if start_date < earliest_account_date:
                raise ValueError("start_date cannot be before the account's start date")
        else:
            start_date = earliest_account_date
        if end_date:
            end_date = pd.to_datetime(end_date).date()
            if end_date > latest_account_date:
                raise ValueError("end_date cannot be after the account's last date")
        else:
            end_date = latest_account_date
        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")

        df = pd.DataFrame(columns=['date', 'value', 'perc_change'])
        start_value = self.account_value_history[datetime.combine(start_date, datetime.min.time())].get_total_value()
        last_value = start_value

        # Since account values are added each day in chronological order, and Python 3.7+ guarantees dict insertion order,
        # we can iterate through the account_value_history and calculate the daily percentage change. It's possible
        # that we have multiple account values for the same date, so we take the last one datetime from each day.
        keys = list(self.account_value_history.keys())
        for i, dt in enumerate(keys):
            if start_date <= dt.date() <= end_date:
                # There can be multiple account values for the same date, but we only want the last one
                if i < len(keys) - 1 and dt.date() == keys[i + 1].date():
                    continue

                account_value = self.account_value_history[dt].get_total_value()
                if change_from_start:
                    change = (account_value - start_value) / start_value * 100
                else:
                    change = (account_value - last_value) / last_value * 100
                    last_value = account_value
                df.loc[len(df)] = [dt.date().strftime('%Y-%m-%d'), account_value, change]

        return df
