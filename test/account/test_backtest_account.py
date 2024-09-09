import pandas as pd
import pytest

from account.account import PositionType
from account.account_values import AccountValues
from account.backtest_account import BacktestAccount
from order import OptionOrder, OrderOperation, OptionType
from order_history import OrderHistory

sample_order_history = OrderHistory(
    orders={
      'contract1': {'2024-01-01': OptionOrder('AAPL', OrderOperation.BUY, 'contract1', OptionType.CALL, 5, 1, 100, '2024-07-01', '2024-01-01'),
                    '2024-01-02': OptionOrder('AAPL', OrderOperation.BUY, 'contract1', OptionType.CALL, 4, 1.5, 100, '2024-07-01', '2024-01-02')},
      'contract2': {'2024-01-01': OptionOrder('AAPL', OrderOperation.BUY, 'contract2', OptionType.PUT, 3, 2, 150, '2024-07-01', '2024-01-01')},
      'contract3': {'2024-01-01': OptionOrder('NVDA', OrderOperation.BUY, 'contract3', OptionType.CALL, 2, 3, 200, '2024-07-01', '2024-01-01')},
      'contract4': {'2024-01-01': OptionOrder('MSFT', OrderOperation.BUY, 'contract4', OptionType.CALL, 1, 4, 250, '2024-07-01', '2024-01-01')},
    },
)
sample_option_positions = {contract_id: sum(order.quantity for order in orders_by_date.values())
                           for contract_id, orders_by_date in sample_order_history.orders.items()}
sample_options_data = {
    'AAPL': pd.DataFrame({'contractID': ['contract1', 'contract2'], 'mark': [1, 2]}),
    'NVDA': pd.DataFrame({'contractID': ['contract3'], 'mark': [3]}),
    'MSFT': pd.DataFrame({'contractID': ['contract4'], 'mark': [4]}),
}


def test_get_positions__stock_all_symbols():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    assert account.get_positions(position_type=PositionType.STOCK) == {'stock': {'AAPL': 5, 'NVDA': 10, 'MSFT': 20}}


def test_get_positions__stock_one_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    assert account.get_positions('AAPL', PositionType.STOCK) == {'stock': {'AAPL': 5}}


def test_get_positions__option_all_symbols():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0),
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    assert account.get_positions(position_type=PositionType.OPTION) == {
        'options': {'AAPL': {'contract1': 9, 'contract2': 3},
                    'NVDA': {'contract3': 2},
                    'MSFT': {'contract4': 1}}
    }


def test_get_positions__option_one_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0),
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    assert account.get_positions('AAPL', PositionType.OPTION) == {'options': {'AAPL': {'contract1': 9, 'contract2': 3}}}


def test_get_positions__all_types_all_symbols():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0),
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
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0),
                              stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20},
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    assert account.get_positions('AAPL') == {
        'stock': {'AAPL': 5},
        'options': {'AAPL': {'contract1': 9, 'contract2': 3}}
    }


def test_get_stock_positions__all():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    assert account._get_stock_positions() == {'AAPL': 5, 'NVDA': 10, 'MSFT': 20}


def test_get_stock_positions__one_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    assert account._get_stock_positions('MSFT') == {'MSFT': 20}


def test_get_stock_positions__no_positions():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), stock_positions={})
    assert account._get_stock_positions() == {}


def test_get_stock_positions__no_positions_for_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    assert account._get_stock_positions('AMZN') == {'AMZN': 0}


def test_get_option_positions__all():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0),
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    assert account._get_option_positions() == {
        'AAPL': {'contract1': 9, 'contract2': 3},
        'NVDA': {'contract3': 2},
        'MSFT': {'contract4': 1},
    }


def test_get_option_positions__one_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0),
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    assert account._get_option_positions('AAPL') == {'AAPL': {'contract1': 9, 'contract2': 3}}


def test_get_option_positions__no_positions():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), option_positions={})
    assert account._get_option_positions() == {}


def test_get_option_positions__no_positions_for_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0),
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    assert account._get_option_positions('AMZN') == {'AMZN': {}}


def test_get_position_values__stock_all_symbols():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    current_prices = {'AAPL': 100, 'NVDA': 200, 'MSFT': 300}
    assert account.get_position_values(position_type=PositionType.STOCK, current_prices=current_prices) == {
        'stock': {'AAPL': 500, 'NVDA': 2000, 'MSFT': 6000}
    }


def test_get_position_values__stock_one_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    current_prices = {'AAPL': 100, 'NVDA': 200, 'MSFT': 300}
    assert account.get_position_values('NVDA', PositionType.STOCK, current_prices) == {
        'stock': {'NVDA': 2000}
    }


def test_get_position_values__option_all_symbols():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0),
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    assert account.get_position_values(position_type=PositionType.OPTION, option_data=sample_options_data) == {
        'options': {'AAPL': {'contract1': 9, 'contract2': 6},
                    'NVDA': {'contract3': 6},
                    'MSFT': {'contract4': 4}}
    }


def test_get_position_values__option_one_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0),
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    assert account.get_position_values('NVDA', PositionType.OPTION, option_data=sample_options_data) == {
        'options': {'NVDA': {'contract3': 6}}
    }


def test_get_position_values__all_types_all_symbols():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0),
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
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0),
                              stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20},
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    current_prices = {'AAPL': 100, 'NVDA': 200, 'MSFT': 300}
    assert account.get_position_values('AAPL', current_prices=current_prices, option_data=sample_options_data) == {
        'stock': {'AAPL': 500},
        'options': {'AAPL': {'contract1': 9, 'contract2': 6}},
    }


def test_get_stock_position_values__all():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    current_prices = {'AAPL': 100, 'NVDA': 200, 'MSFT': 300}
    assert account._get_stock_position_values(current_prices=current_prices) == {'AAPL': 500, 'NVDA': 2000, 'MSFT': 6000}


def test_get_stock_position_values__one_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    current_prices = {'AAPL': 100, 'NVDA': 200, 'MSFT': 300}
    assert account._get_stock_position_values('NVDA', current_prices) == {'NVDA': 2000}


def test_get_stock_position_values__no_positions():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), stock_positions={})
    current_prices = {'AAPL': 100, 'NVDA': 200, 'MSFT': 300}
    assert account._get_stock_position_values(current_prices=current_prices) == {}


def test_get_stock_position_values__no_positions_for_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    current_prices = {'AAPL': 100, 'NVDA': 200, 'MSFT': 300, 'AMZN': 400}
    assert account._get_stock_position_values('AMZN', current_prices) == {'AMZN': 0}


def test_get_stock_position_values__no_current_price_for_given_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    with pytest.raises(ValueError):
        account._get_stock_position_values('NVDA', current_prices={'AAPL': 100, 'MSFT': 300})


def test_get_stock_position_values__no_current_price_for_owned_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), stock_positions={'AAPL': 5, 'NVDA': 10, 'MSFT': 20})
    with pytest.raises(ValueError):
        account._get_stock_position_values(current_prices={'AAPL': 100, 'MSFT': 300})


def test_get_option_position_values__all():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0),
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    assert account._get_option_position_values(option_data=sample_options_data) == {
        'AAPL': {'contract1': 9, 'contract2': 6},
        'NVDA': {'contract3': 6},
        'MSFT': {'contract4': 4},
    }


def test_get_option_position_values__one_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0),
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    assert account._get_option_position_values('AAPL', sample_options_data) == {'AAPL': {'contract1': 9, 'contract2': 6}}


def test_get_option_position_values__no_positions():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0), option_positions={})
    assert account._get_option_position_values(option_data=sample_options_data) == {}


def test_get_option_position_values__no_positions_for_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0),
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    option_data = {
        'AMZN': pd.DataFrame({'contractID': ['contract1', 'contract2'], 'mark': [1, 2]}),
    }
    assert account._get_option_position_values('AMZN', option_data) == {'AMZN': {}}


def test_get_option_position_values__no_option_data_for_symbol():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0),
                              order_history=sample_order_history,
                              option_positions=sample_option_positions)
    with pytest.raises(ValueError):
        account._get_option_position_values('AMZN', sample_options_data)


def test_get_option_position_values__one_sold_option():
    account = BacktestAccount('test', 'test', AccountValues(0, 0, 0),
                              order_history=OrderHistory(orders={
                                  'contract4': {'2024-01-01': OptionOrder('MSFT', OrderOperation.SELL, 'contract4', OptionType.CALL, 1, 4, 250, '2024-07-01', '2024-01-01')},
                              }),
                              option_positions={'contract4': -1})
    assert account._get_option_position_values(option_data=sample_options_data) == {'MSFT': {'contract4': -4}}
