from datetime import datetime, timedelta


def get_closest_trading_date(input_date):
    input_date = datetime.strptime(input_date, '%Y-%m-%d')
    while input_date.weekday() > 4:  # If it's Saturday (5) or Sunday (6), move to Friday
        input_date -= timedelta(days=1)
    # Assuming all weekends are non-trading days, for simplicity
    return input_date.strftime('%Y-%m-%d')
