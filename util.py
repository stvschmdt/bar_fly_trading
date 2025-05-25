from datetime import datetime, timedelta

import pandas


def get_closest_trading_date(input_date):
    input_date = datetime.strptime(input_date, '%Y-%m-%d')
    while input_date.weekday() > 4:  # If it's Saturday (5) or Sunday (6), move to Friday
        input_date -= timedelta(days=1)
    # Assuming all weekends are non-trading days, for simplicity
    return input_date.strftime('%Y-%m-%d')


def get_atm_strike_data(symbol: str, current_price: float, options_data: pandas.DataFrame):
    """
    Get the closest strike price to the current price.
    """
    # Get the options data for the symbol
    options_data = options_data[options_data['symbol'] == symbol]

    # Get the unique strike prices for the symbol
    strike_prices = options_data['strike'].unique()

    # Find the closest strike price
    closest_strike = min(strike_prices, key=lambda x: abs(x - current_price))

    # Return the row of the closest strike price
    return options_data[options_data['strike'] == closest_strike]

def extract_symbol_from_contract_id(contract_id: str) -> str:
    """
    Extract the symbol from a contract ID. The symbol is all letters until the first number.
    """
    symbol = ''
    for c in contract_id:
        if c.isdigit():
            break
        symbol += c
    return symbol