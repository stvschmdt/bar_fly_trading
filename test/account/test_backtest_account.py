import copy
import datetime
from collections import defaultdict
from unittest.mock import patch

import pandas as pd
import pytest

from account.account import PositionType
from account.account_values import AccountValues
from account.backtest_account import BacktestAccount
from order import OptionOrder, OrderOperation, OptionType, StockOrder, Order, MultiLegOrder
from order_history import OrderHistory

sample_order_history = OrderHistory(
    orders={
        'contract1': {
            '2024-01-01': OptionOrder('AAPL', OrderOperation.BUY, 'contract1', OptionType.CALL, 5, 1, 100, '2024-07-01',
                                      '2024-01-01'),
            '2024-01-02': OptionOrder('AAPL', OrderOperation.BUY, 'contract1', OptionType.CALL, 4, 1.5, 100,
                                      '2024-07-01', '2024-01-02')},
        'contract2': {
            '2024-01-01': OptionOrder('AAPL', OrderOperation.BUY, 'contract2', OptionType.PUT, 3, 2, 150, '2024-07-01',
                                      '2024-01-01')},
        'contract3': {
            '2024-01-01': OptionOrder('NVDA', OrderOperation.BUY, 'contract3', OptionType.CALL, 2, 3, 200, '2024-07-01',
                                      '2024-01-01')},
        'contract4': {
            '2024-01-01': OptionOrder('MSFT', OrderOperation.BUY, 'contract4', OptionType.CALL, 1, 4, 250, '2024-07-01',
                                      '2024-01-01')},
    },
)
sample_option_positions = {contract_id: sum(order.quantity for order in orders_by_date.values())
                           for contract_id, orders_by_date in sample_order_history.orders.items()}
sample_options_data = {
    'AAPL': pd.DataFrame({'contractID': ['contract1', 'contract2'], 'mark': [1, 2]}),
    'NVDA': pd.DataFrame({'contractID': ['contract3'], 'mark': [3]}),
    'MSFT': pd.DataFrame({'contractID': ['contract4'], 'mark': [4]}),
}


def test_execute_order__insufficient_funds():
    account = BacktestAccount('test', 'test', AccountValues(100, 0, 0), datetime.datetime(2023, 12, 31))
    with pytest.raises(ValueError):
        account.execute_order(StockOrder('AAPL', OrderOperation.BUY, 1, 101, '2024-01-01'), 101)


def test_execute_order__open_stock_position():
    account = BacktestAccount('test', 'test', AccountValues(100, 0, 0), datetime.datetime(2023, 12, 31))
    order = StockOrder('AAPL', OrderOperation.BUY, 1, 100, '2024-01-01')
    account.execute_order(order, 100)
    assert account.stock_positions == {'AAPL': 1}
    assert account.held_symbols == {'AAPL'}
    assert account.account_values == AccountValues(0, 100, 0)
    expected_orders = defaultdict(lambda: defaultdict(Order))
    expected_orders[order.order_id]['2024-01-01'] = order
    assert account.order_history.orders == expected_orders


def test_execute_order__stock_buy():
    account = BacktestAccount('test', 'test', AccountValues(100, 0, 0), datetime.datetime(2023, 12, 31), stock_positions={'AAPL': 5},
                              held_symbols={'AAPL'})
    order = StockOrder('AAPL', OrderOperation.BUY, 1, 100, '2024-01-01')
    account.execute_order(order, 100)
    assert account.stock_positions == {'AAPL': 6}
    assert account.held_symbols == {'AAPL'}
    assert account.account_values == AccountValues(0, 100, 0)
    expected_orders = defaultdict(lambda: defaultdict(Order))
    expected_orders[order.order_id]['2024-01-01'] = order
    assert account.order_history.orders == expected_orders


def test_execute_order__stock_sell():
    account = BacktestAccount('test', 'test', AccountValues(100, 500, 0), datetime.datetime(2023, 12, 31), stock_positions={'AAPL': 5},
                              held_symbols={'AAPL'})
    order = StockOrder('AAPL', OrderOperation.SELL, 1, 100, '2024-01-01')
    account.execute_order(order, 100)
    assert account.stock_positions == {'AAPL': 4}
    assert account.held_symbols == {'AAPL'}
    assert account.account_values == AccountValues(200, 400, 0)
    expected_orders = defaultdict(lambda: defaultdict(Order))
    expected_orders[order.order_id]['2024-01-01'] = order
    assert account.order_history.orders == expected_orders


def test_execute_order__close_stock_position():
    account = BacktestAccount('test', 'test', AccountValues(100, 500, 0), datetime.datetime(2023, 12, 31), stock_positions={'AAPL': 5},
                              held_symbols={'AAPL'})
    order = StockOrder('AAPL', OrderOperation.SELL, 5, 100, '2024-01-01')
    account.execute_order(order, 100)
    assert account.stock_positions == {}
    assert account.held_symbols == set()
    assert account.account_values == AccountValues(600, 0, 0)
    expected_orders = defaultdict(lambda: defaultdict(Order))
    expected_orders[order.order_id]['2024-01-01'] = order
    assert account.order_history.orders == expected_orders


@patch('order.OptionOrder.calculate_current_value', return_value=10)
def test_execute_order__open_option_position(calculate_current_value_mock):
    account = BacktestAccount('test', 'test', AccountValues(100, 0, 0), datetime.datetime(2023, 12, 31))
    order = OptionOrder('AAPL', OrderOperation.BUY, 'contract1', OptionType.CALL, 1, 10, 100, '2024-07-01',
                        '2024-01-01')
    account.execute_order(order, 100)
    assert account.option_positions == {'contract1': 1}
    assert account.held_symbols == {'AAPL'}
    assert account.account_values == AccountValues(90, 0, 10)
    expected_orders = defaultdict(lambda: defaultdict(Order))
    expected_orders[order.order_id]['2024-01-01'] = order
    assert account.order_history.orders == expected_orders


@patch('order.OptionOrder.calculate_current_value', return_value=10)
def test_execute_order__option_buy(calculate_current_value_mock):
    account = BacktestAccount('test', 'test', AccountValues(100, 0, 50), datetime.datetime(2023, 12, 31), option_positions={'contract1': 5})
    order = OptionOrder('AAPL', OrderOperation.BUY, 'contract1', OptionType.CALL, 1, 10, 100, '2024-07-01',
                        '2024-01-01')
    account.execute_order(order, 100)
    assert account.option_positions == {'contract1': 6}
    assert account.held_symbols == {'AAPL'}
    assert account.account_values == AccountValues(90, 0, 60)
    expected_orders = defaultdict(lambda: defaultdict(Order))
    expected_orders[order.order_id]['2024-01-01'] = order
    assert account.order_history.orders == expected_orders


@patch('order.OptionOrder.calculate_current_value', return_value=10)
def test_execute_order__option_sell(calculate_current_value_mock):
    account = BacktestAccount('test', 'test', AccountValues(100, 0, 50), datetime.datetime(2023, 12, 31), option_positions={'contract1': 5},
                              held_symbols={'AAPL'}, order_history=sample_order_history)
    order = OptionOrder('AAPL', OrderOperation.SELL, 'contract1', OptionType.CALL, 1, 10, 100, '2024-07-01',
                        '2024-01-03')
    account.execute_order(order, 100)
    assert account.option_positions == {'contract1': 4}
    assert account.held_symbols == {'AAPL'}
    assert account.account_values == AccountValues(110, 0, 40)
    expected_order_history = copy.deepcopy(sample_order_history)
    expected_order_history.orders['contract1']['2024-01-03'] = order
    assert account.order_history.orders == expected_order_history.orders


@patch('order.OptionOrder.calculate_current_value', return_value=50)
def test_execute_order__close_option_position(calculate_current_value_mock):
    account = BacktestAccount('test', 'test', AccountValues(100, 0, 50), datetime.datetime(2023, 12, 31), option_positions={'contract1': 5},
                              held_symbols={'AAPL'}, order_history=sample_order_history)
    order = OptionOrder('AAPL', OrderOperation.SELL, 'contract1', OptionType.CALL, 5, 10, 100, '2024-07-01',
                        '2024-01-03')
    account.execute_order(order, 100)
    assert account.option_positions == {}
    assert account.held_symbols == set()
    assert account.account_values == AccountValues(150, 0, 0)
    expected_order_history = copy.deepcopy(sample_order_history)
    expected_order_history.orders['contract1']['2024-01-03'] = order
    assert account.order_history.orders == expected_order_history.orders


def test_get_num_shares_of_symbol__symbol_in_stock_positions():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31), stock_positions={'AAPL': 2})
    assert account.get_num_shares_of_symbol('AAPL') == 2


def test_get_num_shares_of_symbol__symbol_not_in_stock_positions():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31), stock_positions={'AAPL': 2})
    assert account.get_num_shares_of_symbol('NVDA') == 0


def test_apply_order_to_open_positions__multi_leg_order():
    account = BacktestAccount('test', 'test', AccountValues(100, 0, 0), datetime.datetime(2023, 12, 31))
    with pytest.raises(ValueError):
        account._apply_order_to_open_positions(MultiLegOrder(
            [
                StockOrder('AAPL', OrderOperation.BUY, 1, 100, '2024-01-01'),
                OptionOrder('AAPL', OrderOperation.BUY, 'contract1', OptionType.CALL, 1, 100, 100, '2024-07-01',
                            '2024-01-01')
            ], '2024-01-01'),
            100)


def test_apply_order_to_open_positions__open_stock_position():
    account = BacktestAccount('test', 'test', AccountValues(100, 0, 0), datetime.datetime(2023, 12, 31))
    account._apply_order_to_open_positions(StockOrder('AAPL', OrderOperation.BUY, 1, 100, '2024-01-01'), 100)
    assert account.stock_positions == {'AAPL': 1}
    assert account.account_values == AccountValues(0, 100, 0)


def test_apply_order_to_open_positions__stock_buy():
    account = BacktestAccount('test', 'test', AccountValues(100, 500, 0), datetime.datetime(2023, 12, 31), stock_positions={'AAPL': 5})
    account._apply_order_to_open_positions(StockOrder('AAPL', OrderOperation.BUY, 1, 100, '2024-01-01'), 100)
    assert account.stock_positions == {'AAPL': 6}
    assert account.account_values == AccountValues(0, 600, 0)


def test_apply_order_to_open_positions__stock_sell():
    account = BacktestAccount('test', 'test', AccountValues(0, 500, 0), datetime.datetime(2023, 12, 31), stock_positions={'AAPL': 5})
    account._apply_order_to_open_positions(StockOrder('AAPL', OrderOperation.SELL, 1, 100, '2024-01-01'), 100)
    assert account.stock_positions == {'AAPL': 4}
    assert account.account_values == AccountValues(100, 400, 0)


def test_apply_order_to_open_positions__close_stock_position():
    account = BacktestAccount('test', 'test', AccountValues(0, 500, 0), datetime.datetime(2023, 12, 31), stock_positions={'AAPL': 5})
    account._apply_order_to_open_positions(StockOrder('AAPL', OrderOperation.SELL, 5, 100, '2024-01-01'), 500)
    assert account.stock_positions == {}
    assert account.account_values == AccountValues(500, 0, 0)


def test_apply_order_to_open_positions__open_option_position():
    account = BacktestAccount('test', 'test', AccountValues(100, 0, 0), datetime.datetime(2023, 12, 31))
    account._apply_order_to_open_positions(OptionOrder('AAPL', OrderOperation.BUY, 'contract1', OptionType.CALL, 1, 100, 100, '2024-07-01', '2024-01-01'), 10)
    assert account.option_positions == {'contract1': 1}
    assert account.account_values == AccountValues(90, 0, 10)


def test_apply_order_to_open_positions__option_buy():
    account = BacktestAccount('test', 'test', AccountValues(90, 0, 10), datetime.datetime(2023, 12, 31), option_positions={'contract1': 1})
    account._apply_order_to_open_positions(OptionOrder('AAPL', OrderOperation.BUY, 'contract1', OptionType.CALL, 1, 100, 100, '2024-07-01', '2024-01-01'), 10)
    assert account.option_positions == {'contract1': 2}
    assert account.account_values == AccountValues(80, 0, 20)


def test_apply_order_to_open_positions__option_sell():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 20), datetime.datetime(2023, 12, 31), option_positions={'contract1': 2})
    account._apply_order_to_open_positions(OptionOrder('AAPL', OrderOperation.SELL, 'contract1', OptionType.CALL, 1, 100, 100, '2024-07-01', '2024-01-01'), 10)
    assert account.option_positions == {'contract1': 1}
    assert account.account_values == AccountValues(10, 0, 10)


def test_apply_order_to_open_positions__close_option_position():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 10), datetime.datetime(2023, 12, 31), option_positions={'contract1': 1})
    account._apply_order_to_open_positions(OptionOrder('AAPL', OrderOperation.SELL, 'contract1', OptionType.CALL, 1, 100, 100, '2024-07-01', '2024-01-01'), 10)
    assert account.option_positions == {}
    assert account.account_values == AccountValues(10, 0, 0)


def test_remove_held_symbol__symbol_in_held_positions():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31), stock_positions={'NVDA': 2},
                              held_symbols={'AAPL', 'NVDA'})
    account._remove_held_symbol('AAPL')
    assert account.held_symbols == {'NVDA'}


def test_remove_held_symbol__symbol_not_in_held_positions():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31), stock_positions={'AAPL': 1, 'NVDA': 2},
                              held_symbols={'AAPL', 'NVDA'})
    account._remove_held_symbol('AMZN')
    assert account.held_symbols == {'AAPL', 'NVDA'}


def test_remove_held_symbol__one_held_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31), held_symbols={'AAPL'})
    account._remove_held_symbol('AAPL')
    assert account.held_symbols == set()


def test_get_account_values__missing_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31), stock_positions={'AAPL': 1, 'NVDA': 2},
                              held_symbols={'AAPL', 'NVDA'})
    current_prices = {'NVDA': 200, 'MSFT': 300}
    with pytest.raises(ValueError):
        account.get_account_values(current_prices, {})


def test_get_account_values__stock_only():
    account = BacktestAccount('test', 'test', AccountValues(100, 0, 0), datetime.datetime(2023, 12, 31),
                              stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    current_prices = {'AAPL': 100, 'NVDA': 200, 'MSFT': 300}
    assert account.get_account_values(current_prices, {}) == AccountValues(100, 8500, 0)


def test_get_account_values__options_only():
    account = BacktestAccount('test', 'test', AccountValues(100, 0, 0), datetime.datetime(2023, 12, 31),
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    assert account.get_account_values({}, sample_options_data) == AccountValues(100, 0, 25)


def test_get_account_values__stock_and_options():
    account = BacktestAccount('test', 'test', AccountValues(100, 0, 0), datetime.datetime(2023, 12, 31),
                              stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20},
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    current_prices = {'AAPL': 100, 'NVDA': 200, 'MSFT': 300}
    assert account.get_account_values(current_prices, sample_options_data) == AccountValues(100, 8500, 25)


@patch('account.backtest_account.BacktestAccount.get_account_values', return_value=AccountValues(100, 8500, 25))
def test_update_account_values__timestamp_doesnt_exist(get_account_values_mock):
    account = BacktestAccount('test', 'test', AccountValues(100, 0, 0), datetime.datetime(2023, 12, 31))
    account.account_value_history = {datetime.datetime(2024, 1, 1, 12, 0, 0): AccountValues(95, 0, 0)}
    assert account.update_account_values(datetime.datetime(2024, 1, 1, 13, 0, 0), {}, {}) == AccountValues(100, 8500,
                                                                                                           25)
    assert account.account_value_history == {
        datetime.datetime(2024, 1, 1, 12, 0, 0): AccountValues(95, 0, 0),
        datetime.datetime(2024, 1, 1, 13, 0, 0): AccountValues(100, 8500, 25),
    }


@patch('account.backtest_account.BacktestAccount.get_account_values', return_value=AccountValues(100, 8500, 25))
def test_update_account_values__timestamp_exists(get_account_values_mock):
    account = BacktestAccount('test', 'test', AccountValues(100, 0, 0), datetime.datetime(2023, 12, 31))
    account.account_value_history = {datetime.datetime(2024, 1, 1, 12, 0, 0): AccountValues(95, 0, 0)}
    assert account.update_account_values(datetime.datetime(2024, 1, 1, 12, 0, 0), {}, {}) == AccountValues(100, 8500,
                                                                                                           25)
    assert account.account_value_history == {
        datetime.datetime(2024, 1, 1, 12, 0, 0): AccountValues(95, 0, 0),
        datetime.datetime(2024, 1, 1, 12, 0, 1): AccountValues(100, 8500, 25),
    }


@patch('account.backtest_account.BacktestAccount.get_account_values', return_value=AccountValues(100, 8500, 25))
def test_update_account_values__empty_account_values_history(get_account_values_mock):
    account = BacktestAccount('test', 'test', AccountValues(100, 0, 0), datetime.datetime(2023, 12, 31))
    assert account.update_account_values(datetime.datetime(2024, 1, 1, 12, 0, 0), {}, {}) == AccountValues(100, 8500,
                                                                                                           25)
    assert account.account_value_history == {
        datetime.datetime(2023, 12, 31): AccountValues(100, 0, 0),
        datetime.datetime(2024, 1, 1, 12, 0, 0): AccountValues(100, 8500, 25),
    }


def test_get_positions__stock_all_symbols():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    assert account.get_positions(position_type=PositionType.STOCK) == {'stock': {'AAPL': 5, 'NVDA': 10, 'MSFT': 20}}


def test_get_positions__stock_one_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    assert account.get_positions('AAPL', PositionType.STOCK) == {'stock': {'AAPL': 5}}


def test_get_positions__option_all_symbols():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    assert account.get_positions(position_type=PositionType.OPTION) == {
        'options': {'AAPL': {'contract1': 9, 'contract2': 3},
                    'NVDA': {'contract3': 2},
                    'MSFT': {'contract4': 1}}
    }


def test_get_positions__option_one_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    assert account.get_positions('AAPL', PositionType.OPTION) == {'options': {'AAPL': {'contract1': 9, 'contract2': 3}}}


def test_get_positions__all_types_all_symbols():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20},
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    assert account.get_positions() == {
        'stock': {'AAPL': 5, 'NVDA': 10, 'MSFT': 20},
        'options': {'AAPL': {'contract1': 9, 'contract2': 3},
                    'NVDA': {'contract3': 2},
                    'MSFT': {'contract4': 1}}
    }


def test_get_positions__all_types_one_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20},
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    assert account.get_positions('AAPL') == {
        'stock': {'AAPL': 5},
        'options': {'AAPL': {'contract1': 9, 'contract2': 3}}
    }


def test_get_stock_positions__all():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    assert account._get_stock_positions() == {'AAPL': 5, 'NVDA': 10, 'MSFT': 20}


def test_get_stock_positions__one_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    assert account._get_stock_positions('MSFT') == {'MSFT': 20}


def test_get_stock_positions__no_positions():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31), stock_positions={})
    assert account._get_stock_positions() == {}


def test_get_stock_positions__no_positions_for_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    assert account._get_stock_positions('AMZN') == {'AMZN': 0}


def test_get_option_positions__all():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    assert account._get_option_positions() == {
        'AAPL': {'contract1': 9, 'contract2': 3},
        'NVDA': {'contract3': 2},
        'MSFT': {'contract4': 1},
    }


def test_get_option_positions__one_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    assert account._get_option_positions('AAPL') == {'AAPL': {'contract1': 9, 'contract2': 3}}


def test_get_option_positions__no_positions():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31), option_positions={})
    assert account._get_option_positions() == {}


def test_get_option_positions__no_positions_for_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    assert account._get_option_positions('AMZN') == {'AMZN': {}}


def test_get_position_values__stock_all_symbols():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    current_prices = {'AAPL': 100, 'NVDA': 200, 'MSFT': 300}
    assert account.get_position_values(position_type=PositionType.STOCK, current_prices=current_prices) == {
        'stock': {'AAPL': 500, 'NVDA': 2000, 'MSFT': 6000}
    }


def test_get_position_values__stock_one_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    current_prices = {'AAPL': 100, 'NVDA': 200, 'MSFT': 300}
    assert account.get_position_values('NVDA', PositionType.STOCK, current_prices) == {
        'stock': {'NVDA': 2000}
    }


def test_get_position_values__option_all_symbols():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    assert account.get_position_values(position_type=PositionType.OPTION, option_data=sample_options_data) == {
        'options': {'AAPL': {'contract1': 9, 'contract2': 6},
                    'NVDA': {'contract3': 6},
                    'MSFT': {'contract4': 4}}
    }


def test_get_position_values__option_one_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    assert account.get_position_values('NVDA', PositionType.OPTION, option_data=sample_options_data) == {
        'options': {'NVDA': {'contract3': 6}}
    }


def test_get_position_values__all_types_all_symbols():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20},
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    current_prices = {'AAPL': 100, 'NVDA': 200, 'MSFT': 300}
    assert account.get_position_values(current_prices=current_prices, option_data=sample_options_data) == {
        'stock': {'AAPL': 500, 'NVDA': 2000, 'MSFT': 6000},
        'options': {'AAPL': {'contract1': 9, 'contract2': 6},
                    'NVDA': {'contract3': 6},
                    'MSFT': {'contract4': 4}}
    }


def test_get_position_values__all_types_one_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20},
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    current_prices = {'AAPL': 100, 'NVDA': 200, 'MSFT': 300}
    assert account.get_position_values('AAPL', current_prices=current_prices, option_data=sample_options_data) == {
        'stock': {'AAPL': 500},
        'options': {'AAPL': {'contract1': 9, 'contract2': 6}},
    }


def test_get_stock_position_values__all():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    current_prices = {'AAPL': 100, 'NVDA': 200, 'MSFT': 300}
    assert account._get_stock_position_values(current_prices=current_prices) == {'AAPL': 500, 'NVDA': 2000,
                                                                                 'MSFT': 6000}


def test_get_stock_position_values__one_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    current_prices = {'AAPL': 100, 'NVDA': 200, 'MSFT': 300}
    assert account._get_stock_position_values('NVDA', current_prices) == {'NVDA': 2000}


def test_get_stock_position_values__no_positions():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31), stock_positions={})
    current_prices = {'AAPL': 100, 'NVDA': 200, 'MSFT': 300}
    assert account._get_stock_position_values(current_prices=current_prices) == {}


def test_get_stock_position_values__no_positions_for_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    current_prices = {'AAPL': 100, 'NVDA': 200, 'MSFT': 300, 'AMZN': 400}
    assert account._get_stock_position_values('AMZN', current_prices) == {'AMZN': 0}


def test_get_stock_position_values__no_current_price_for_given_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    with pytest.raises(ValueError):
        account._get_stock_position_values('NVDA', current_prices={'AAPL': 100, 'MSFT': 300})


def test_get_stock_position_values__no_current_price_for_owned_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    with pytest.raises(ValueError):
        account._get_stock_position_values(current_prices={'AAPL': 100, 'MSFT': 300})


def test_get_option_position_values__all():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    assert account._get_option_position_values(option_data=sample_options_data) == {
        'AAPL': {'contract1': 9, 'contract2': 6},
        'NVDA': {'contract3': 6},
        'MSFT': {'contract4': 4},
    }


def test_get_option_position_values__one_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    assert account._get_option_position_values('AAPL', sample_options_data) == {
        'AAPL': {'contract1': 9, 'contract2': 6}}


def test_get_option_position_values__no_positions():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31), option_positions={})
    assert account._get_option_position_values(option_data=sample_options_data) == {}


def test_get_option_position_values__no_positions_for_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    option_data = {
        'AMZN': pd.DataFrame({'contractID': ['contract1', 'contract2'], 'mark': [1, 2]}),
    }
    assert account._get_option_position_values('AMZN', option_data) == {'AMZN': {}}


def test_get_option_position_values__no_option_data_for_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    with pytest.raises(ValueError):
        account._get_option_position_values('AMZN', sample_options_data)


def test_get_option_position_values__one_sold_option():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), datetime.datetime(2023, 12, 31),
                              order_history=OrderHistory(orders={
                                  'contract4': {'2024-01-01': OptionOrder('MSFT', OrderOperation.SELL, 'contract4',
                                                                          OptionType.CALL, 1, 4, 250, '2024-07-01',
                                                                          '2024-01-01')},
                              }),
                              option_positions={'contract4': -1})
    assert account._get_option_position_values(option_data=sample_options_data) == {'MSFT': {'contract4': -4}}


def test_get_daily_account_value_percentage_change__start_date_before_account_start_date():
    account = BacktestAccount('test', 'test', AccountValues(100, 0, 0), datetime.datetime(2023, 12, 31))
    with pytest.raises(ValueError):
        account.get_daily_account_value_percentage_change('2023-12-30')


def test_get_daily_account_value_percentage_change__end_date_after_latest_account_date():
    account = BacktestAccount('test', 'test', AccountValues(100, 0, 0), datetime.datetime(2023, 12, 31))
    with pytest.raises(ValueError):
        account.get_daily_account_value_percentage_change('2024-01-01')


def test_get_daily_account_value_percentage_change__start_date_equals_end_date():
    account = BacktestAccount('test', 'test', AccountValues(100, 0, 0), datetime.datetime(2023, 12, 31))
    with pytest.raises(ValueError):
        account.get_daily_account_value_percentage_change('2023-12-31', '2023-12-31')


def test_get_daily_account_value_percentage_change__start_date_after_end_date():
    account = BacktestAccount('test', 'test', AccountValues(100, 0, 0), datetime.datetime(2023, 12, 31))
    account.account_value_history = {datetime.datetime(2023, 12, 31): AccountValues(100, 0, 0), datetime.datetime(2024, 1, 1): AccountValues(100, 0, 0)}
    with pytest.raises(ValueError):
        account.get_daily_account_value_percentage_change('2024-01-01', '2023-12-31')


def test_get_daily_account_value_percentage_change__from_start():
    account = BacktestAccount('test', 'test', AccountValues(100, 0, 0), datetime.datetime(2023, 12, 31))
    account.account_value_history = {datetime.datetime(2023, 12, 31): AccountValues(100, 0, 0), datetime.datetime(2024, 1, 1): AccountValues(0, 100, 50), datetime.datetime(2024, 1, 2): AccountValues(100, 100, 0)}
    assert account.get_daily_account_value_percentage_change().equals(
        pd.DataFrame(
            {'date': ['2023-12-31', '2024-01-01', '2024-01-02'],
             'value': [100, 150, 200],
             'perc_change': [0.0, 50.0, 100.0]}
        )
    )


def test_get_daily_account_value_percentage_change__incremental():
    account = BacktestAccount('test', 'test', AccountValues(100, 0, 0), datetime.datetime(2023, 12, 31))
    account.account_value_history = {datetime.datetime(2023, 12, 31): AccountValues(100, 0, 0), datetime.datetime(2024, 1, 1): AccountValues(0, 100, 50), datetime.datetime(2024, 1, 2): AccountValues(100, 80, 0)}
    assert account.get_daily_account_value_percentage_change(change_from_start=False).equals(
        pd.DataFrame(
            {'date': ['2023-12-31', '2024-01-01', '2024-01-02'],
             'value': [100, 150, 180],
             'perc_change': [0.0, 50.0, 20.0]}
        )
    )

