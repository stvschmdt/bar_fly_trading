import pandas as pd
import matplotlib.pyplot as plt

from collector import alpha_client
from simulations.spreads.bull_call_spread import BullCallSpread


# Since options only expire on Fridays, we'll find the nearest Friday to start_date + APPROXIMATE_TIME_TO_EXPIRATION
APPROXIMATE_TIME_TO_EXPIRATION = 30
SYMBOLS = ['AAPL']
START_DATE = '2024-01-01'
END_DATE = '2024-08-01'
STRATEGY = 'bull_call_spread'


def graph_results(symbol: str, results: dict):
    plt.figure(figsize=(10, 6))

    # Plot profits
    plt.plot(results['dates'], results['incremental_profits'], label='Incremental Profit', marker='o', color='blue')

    # Plot cumulative profits
    plt.plot(results['dates'], results['cumulative_profits'], label='Cumulative Profit', marker='x', color='purple')

    # Annotate buy and sell call prices
    for i, (date, buy, sell) in enumerate(zip(results['dates'], results['buys'], results['sells'])):
        plt.annotate(f'Buy: {buy}', (date, results['incremental_profits'][i]), textcoords="offset points", xytext=(-10, 10), ha='center',
                     color='green')
        plt.annotate(f'Sell: {sell}', (date, results['incremental_profits'][i]), textcoords="offset points", xytext=(-10, -15), ha='center',
                     color='red')

    plt.xlabel('Date')
    plt.ylabel('Profit')
    plt.title(f'{symbol} {STRATEGY} Daily and Cumulative Profits')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    for symbol in SYMBOLS:
        bull_call_spread = BullCallSpread(alpha_client, symbol, APPROXIMATE_TIME_TO_EXPIRATION, START_DATE, END_DATE)

        try:
            profit, results = bull_call_spread.calculate()
        except Exception as e:
            print(f"Error calculating bull call spread: {e}")
            return

        print(f"Profit for {symbol} from {START_DATE} to {END_DATE}: ${round(profit, 2)}")
        graph_results(symbol, results)


if __name__ == "__main__":
    main()
