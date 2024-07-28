import pandas as pd
from simulations.spreads.util import find_nearest_strike

BUY_CALL_PERCENT_ABOVE_START = 0.1
SELL_CALL_PERCENT_BELOW_START = 0.05


def calculate(options_df: pd.DataFrame, stock_price_df: pd.DataFrame, start_date: str, end_date: str):
    start_price = stock_price_df.loc[start_date]['adjusted_close']
    end_price = stock_price_df.loc[end_date]['adjusted_close']
    buy_strike_price = find_nearest_strike(options_df['strike'].tolist(), start_price, BUY_CALL_PERCENT_ABOVE_START)
    sell_strike_price = find_nearest_strike(options_df['strike'].tolist(), start_price, -SELL_CALL_PERCENT_BELOW_START)
    buy_option_price = options_df[options_df['strike'] == buy_strike_price]['call_ask'].iloc[0]
    sell_option_price = options_df[options_df['strike'] == sell_strike_price]['call_bid'].iloc[0]

    # Calculate the profit/loss
    buy_profit = max(0, end_price - buy_strike_price) - buy_option_price
    sell_profit = sell_option_price - max(0, end_price - sell_strike_price)

    print(f'Buy profit: {buy_profit}')
    print(f'Sell profit: {sell_profit}')
    print(f'Total profit: {buy_profit + sell_profit}')
    return (buy_profit + sell_profit) * 100

