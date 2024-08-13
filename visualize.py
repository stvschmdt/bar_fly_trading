import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
from logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

class VisualizeTechnicals:
    def __init__(self, df, symbol, start_date, end_date):
        self.df = df
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.df_filtered = self.filter_data()
    
    def filter_data(self):
        # Filter the dataframe for the given symbol and date range
        df_filtered = self.df[(self.df['symbol'] == self.symbol) & 
                              (self.df['date'] >= self.start_date) & 
                              (self.df['date'] <= self.end_date)]
        
        if df_filtered.empty:
            print(f"No data found for symbol {self.symbol} in the specified date range.")
            return None
        
        # Set the date as the index
        df_filtered.set_index('date', inplace=True)
        return df_filtered

    def plot_adjusted_close_and_sma(self):
        if self.df_filtered is None:
            return
        
        plt.figure(figsize=(14, 10))
        plt.subplot(4, 1, 1)
        plt.plot(self.df_filtered.index, self.df_filtered['adjusted_close'], label='Adjusted Close', color='blue')
        
        if 'sma_20' in self.df_filtered.columns:
            plt.plot(self.df_filtered.index, self.df_filtered['sma_20'], label='SMA 20', color='orange')
        if 'sma_50' in self.df_filtered.columns:
            plt.plot(self.df_filtered.index, self.df_filtered['sma_50'], label='SMA 50', color='green')
        if 'sma_200' in self.df_filtered.columns:
            plt.plot(self.df_filtered.index, self.df_filtered['sma_200'], label='SMA 200', color='red')
        
        if 'sma_20' in self.df_filtered.columns and 'sma_50' in self.df_filtered.columns:
            crossover_indices = self.df_filtered[(self.df_filtered['sma_20'] > self.df_filtered['sma_50']) & 
                                                 (self.df_filtered['sma_20'].shift(1) <= self.df_filtered['sma_50'].shift(1))].index
            for index in crossover_indices:
                plt.annotate('Bullish Crossover', xy=(index, self.df_filtered.loc[index, 'adjusted_close']),
                             xytext=(index, self.df_filtered.loc[index, 'adjusted_close'] + 1),
                             arrowprops=dict(facecolor='green', shrink=0.05))

            crossover_indices = self.df_filtered[(self.df_filtered['sma_20'] < self.df_filtered['sma_50']) & 
                                                 (self.df_filtered['sma_20'].shift(1) >= self.df_filtered['sma_50'].shift(1))].index
            for index in crossover_indices:
                plt.annotate('Bearish Crossover', xy=(index, self.df_filtered.loc[index, 'adjusted_close']),
                             xytext=(index, self.df_filtered.loc[index, 'adjusted_close'] - 1),
                             arrowprops=dict(facecolor='red', shrink=0.05))

        plt.title(f'{self.symbol} Price and Moving Averages Markers')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=90)

    def plot_volume(self):
        if self.df_filtered is None:
            return

        plt.subplot(4, 1, 2)
        plt.bar(self.df_filtered.index, self.df_filtered['volume'], color='gray')
        
        volume_spike_threshold = self.df_filtered['volume'].mean() * 2
        volume_spike_indices = self.df_filtered[self.df_filtered['volume'] > volume_spike_threshold].index
        for index in volume_spike_indices:
            plt.annotate('Volume Spike', xy=(index, self.df_filtered.loc[index, 'volume']),
                         xytext=(index, self.df_filtered.loc[index, 'volume'] + volume_spike_threshold/2),
                         arrowprops=dict(facecolor='purple', shrink=0.05))

        plt.title(f'{self.symbol} Volume Markers')
        plt.grid(True)
        plt.xticks(rotation=90)

    def plot_rsi(self):
        if self.df_filtered is None or 'rsi_14' not in self.df_filtered.columns:
            return

        plt.subplot(4, 1, 3)
        plt.plot(self.df_filtered.index, self.df_filtered['rsi_14'], label='RSI 14', color='magenta')
        plt.axhline(70, color='red', linestyle='--', label='Overbought')
        plt.axhline(30, color='green', linestyle='--', label='Oversold')
        
        overbought_indices = self.df_filtered[self.df_filtered['rsi_14'] > 70].index
        for index in overbought_indices:
            plt.annotate('Overbought', xy=(index, self.df_filtered.loc[index, 'rsi_14']),
                         xytext=(index, self.df_filtered.loc[index, 'rsi_14'] + 5),
                         arrowprops=dict(facecolor='red', shrink=0.05))
        
        oversold_indices = self.df_filtered[self.df_filtered['rsi_14'] < 30].index
        for index in oversold_indices:
            plt.annotate('Oversold', xy=(index, self.df_filtered.loc[index, 'rsi_14']),
                         xytext=(index, self.df_filtered.loc[index, 'rsi_14'] - 5),
                         arrowprops=dict(facecolor='green', shrink=0.05))

        plt.title(f'{self.symbol} RSI 14 Markers')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=90)

    def plot_macd(self):
        if self.df_filtered is None or 'macd' not in self.df_filtered.columns:
            return

        plt.subplot(4, 1, 4)
        plt.plot(self.df_filtered.index, self.df_filtered['macd'], label='MACD', color='blue')
        plt.axhline(0, color='black', linestyle='--')
        
        macd_crossover_indices = self.df_filtered[(self.df_filtered['macd'] > 0) & 
                                                  (self.df_filtered['macd'].shift(1) <= 0)].index
        for index in macd_crossover_indices:
            plt.annotate('MACD Bullish Crossover', xy=(index, self.df_filtered.loc[index, 'macd']),
                         xytext=(index, self.df_filtered.loc[index, 'macd'] + 0.1),
                         arrowprops=dict(facecolor='green', shrink=0.05))
        
        macd_crossover_indices = self.df_filtered[(self.df_filtered['macd'] < 0) & 
                                                  (self.df_filtered['macd'].shift(1) >= 0)].index
        for index in macd_crossover_indices:
            plt.annotate('MACD Bearish Crossover', xy=(index, self.df_filtered.loc[index, 'macd']),
                         xytext=(index, self.df_filtered.loc[index, 'macd'] - 0.1),
                         arrowprops=dict(facecolor='red', shrink=0.05))

        plt.title(f'{self.symbol} MACD Markers')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=90)

    def visualize(self, charts_to_generate):
        plt.figure(figsize=(14, len(charts_to_generate) * 3.5))
        
        if 'adjusted_close' in charts_to_generate:
            self.plot_adjusted_close_and_sma()
        if 'volume' in charts_to_generate:
            self.plot_volume()
        if 'rsi' in charts_to_generate:
            self.plot_rsi()
        if 'macd' in charts_to_generate:
            self.plot_macd()

        plt.tight_layout()
        plt.savefig(f'{self.symbol}_analysis.png')
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="file containing all core and technical data", type=str, default='all_data.csv')
    parser.add_argument("-s", "--symbol", help="stock symbol to visualize", type=str, default='WMT')
    parser.add_argument("-start", "--start_date", help="start date for visual analysis", type=str, default='2024-07-01')
    parser.add_argument("-end", "--end_date", help="end date for visual analysis", type=str, default='2024-08-13')
    parser.add_argument("-c", "--charts", help="comma separated list of charts to generate", type=str, default='adjusted_close,volume,rsi,macd')
    args = parser.parse_args()
    logger.info(f"Data file: {args.data}")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Start date: {args.start_date}")
    logger.info(f"End date: {args.end_date}")
    # Load the stock data
    df = pd.read_csv(args.data)
# Test case using all 4 of the charts
    visualizer = VisualizeTechnicals(df, symbol=args.symbol, start_date=args.start_date, end_date=args.end_date)
    visualizer.visualize(charts_to_generate=args.charts.split(',') if args.charts else [])

