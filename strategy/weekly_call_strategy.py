from datetime import datetime, timedelta

import pandas as pd

from order import Order, OrderOperation, OptionType, OptionOrder
from strategy.base_strategy import BaseStrategy
from util import get_atm_strike_data


class WeeklyCallStrategy(BaseStrategy):
    """
    Buys an at-the-money call option for each symbol on Monday, expiring on Friday.

    This is only meant for testing. Since we're currently only buying on Mondays and selling on Fridays,
    it'll skip weeks with Monday holidays, and will fail if there's a Friday holiday. If you're fine with
    skipping weeks with Monday holidays, you can use this strategy, as long as the test window has no Friday holidays.
    """
    def __init__(self, account, symbols):
        super().__init__(account, symbols)

    def evaluate(self, date: datetime.date, current_prices: pd.DataFrame, options_data: pd.DataFrame) -> list[Order]:
        orders = []

        # Check if the date is a Monday or a Friday
        if date.weekday() == 0:
            # Buy a call option for each symbol
            for symbol in self.symbols:
                current_price = current_prices.loc[current_prices['symbol'] == symbol, 'open'].iloc[0]
                atm_strike_data = get_atm_strike_data(symbol, current_price, options_data)
                if atm_strike_data is None:
                    print(f"No ATM strike data found for {symbol} on {date}, skipping")
                    continue

                # Filter down to just calls
                atm_strike_data = atm_strike_data[atm_strike_data['type'] == 'call']
                # Get the latest date this week. Usually Friday, but this accounts for holidays.
                friday_date = (date + timedelta(days=4)).date()
                this_weeks_options = atm_strike_data[atm_strike_data['expiration'] <= friday_date]

                # Get the row with the latest expiration date among the filtered ones
                if this_weeks_options.empty:
                    print("No expiration dates within the next 4 days.")
                else:
                    latest = this_weeks_options.loc[this_weeks_options['expiration'].idxmax()]
                    orders.append(
                        OptionOrder(symbol, OrderOperation.BUY, latest['contract_id'], OptionType.CALL, 1, latest['ask'],
                                    latest['strike'], latest['expiration'], date.strftime('%Y-%m-%d'))
                    )
        elif date.weekday() == 4:
            # Get all option orders we placed on Monday
            monday_date = (date - timedelta(days=4)).strftime('%Y-%m-%d')
            orders_history = self.account.order_history.get_filtered_order_history(start_date=monday_date, end_date=monday_date, order_types=[OptionOrder], order_operation=OrderOperation.BUY)

            # Sell all option orders we placed on Monday
            for symbol, symbol_orders in orders_history.items():
                for _, order_id_dict in symbol_orders.items():
                    for _, order in order_id_dict.items():
                        if order.option_type != OptionType.CALL:
                            continue

                        # Get the latest bid price for the option and sell it
                        current_price = options_data.loc[options_data['contract_id'] == order.contract_id, 'bid'].iloc[0]
                        orders.append(
                            OptionOrder(order.symbol, OrderOperation.SELL, order.contract_id, order.option_type, order.quantity,
                                        current_price, order.strike_price, order.expiration_date, date.strftime('%Y-%m-%d'))
                        )

        return orders

