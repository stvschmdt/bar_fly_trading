import pandas as pd
import numpy as np
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



class SectorAnalysis:
    def __init__(self, df, use_treasury='10year'):
        self.df = df
        sectors = ['XLB', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY', 'XLE', 'XRT']
        # list of industries
        industries = [' SPDR FUND MATERIALS SELECT SECTR ETF',' SELECT STR FINANCIAL SELECT SPDR ETF',' SELECT SECTOR INDUSTRIAL SPDR ETF',
                      ' TECHNOLOGY SELECT SECTOR SPDR ETF',' SPDR FUND CONSUMER STAPLES ETF',' REAL ESTATE SELECT SCTR SPDR ETF',
                      ' SELECT SECTOR UTI SELECT SPDR ETF',' SELECT SECTOR HEALTH CARE SPDR ETF',' SPDR FUND CONSUMER DISCRE SELECT ETF',
                      ' ENERGY SELECT SECTOR SPDR ETF',' SPDR S&P RETAIL ETF']
        # create dictionary of sectors and industries
        sector_industry = dict(zip(sectors, industries))
        # filter df to only rows where symbol is one of the sector symbols
        self.df = self.df[self.df['symbol'].isin(sectors)]
        # fill df sector column with the corresponding industry
        self.df['sector'] = df['symbol'].map(sector_industry)
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df.sort_values(by='date', inplace=True)
        
        # Select the appropriate treasury yield as the risk-free rate
        if use_treasury == '10year':
            self.df['risk_free_rate'] = self.df['treasury_yield_10year'] / 100 / 252  # Convert annual to daily rate
        elif use_treasury == '2year':
            self.df['risk_free_rate'] = self.df['treasury_yield_2year'] / 100 / 252  # Convert annual to daily rate
        else:
            raise ValueError("use_treasury must be either '10year' or '2year'")

    def calculate_cumulative_percent_change(self, start_date, end_date):
        filtered_df = self.df[(self.df['date'] >= start_date) & (self.df['date'] <= end_date)]
        sectors = filtered_df['sector'].unique()
        
        cumulative_changes = {}

        for sector in sectors:
            sector_data = filtered_df[filtered_df['sector'] == sector].copy()
            sector_data = sector_data.groupby('date').mean(numeric_only=True)  # Group by date and take the mean of numeric columns only
            sector_data.sort_index(inplace=True)
            
            if not sector_data.empty:
                initial_value = sector_data['adjusted_close'].iloc[0]
                if initial_value != 0:  # Avoid division by zero
                    sector_data['cumulative_change'] = (sector_data['adjusted_close'] / initial_value - 1) * 100
                    cumulative_changes[sector] = sector_data['cumulative_change']

        return cumulative_changes

    def plot_sector_cumulative_percent_change(self, start_date, end_date):
        cumulative_changes = self.calculate_cumulative_percent_change(start_date, end_date)

        sectors = ['XLB', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY', 'XLE', 'XRT']
        colors = plt.cm.tab20(np.linspace(0, 1, len(sectors)))  # Use a colormap that can handle more than 10 colors
        plt.figure(figsize=(14, 6))
        for color, (sector, changes) in zip(colors, cumulative_changes.items()):
            plt.plot(changes.index, changes, label=sector, linestyle='-', marker='', color=color)

        plt.title(f'Sector Cumulative Percent Change from {start_date} to {end_date}')
        plt.ylabel('Cumulative Percent Change (%)')
        plt.xlabel('Date')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.xticks(rotation=45)
        # save figure
        plt.savefig('sector_cumulative_percent_change.png')
        plt.show()

    def plot_rsi_comparison(self, start_date, end_date):
        filtered_df = self.df[(self.df['date'] >= start_date) & (self.df['date'] <= end_date)]
        sectors = filtered_df['sector'].unique()

        # plot with enough colormap for 11 different sectors
        colors = plt.cm.tab20(np.linspace(0, 1, len(sectors)))  # Use a colormap that can handle more than 10 colors
        plt.figure(figsize=(14, 8))
        for sector, color in zip(sectors, colors):
            sector_data = filtered_df[filtered_df['sector'] == sector]
            sector_data = sector_data.groupby('date').mean(numeric_only=True)  # Group by date and take the mean of numeric columns only
            if not sector_data.empty:
                plt.plot(sector_data.index, sector_data['rsi_14'], label=sector, linestyle='-', marker='', color=color)

        # Adding the overbought and oversold lines without adding them to the legend
        plt.axhline(70, color='red', linestyle='--', linewidth=2)
        plt.axhline(30, color='blue', linestyle='--', linewidth=2)

        plt.title(f'Sector RSI_14 Comparison from {start_date} to {end_date}')
        plt.ylabel('RSI_14')
        plt.xlabel('Date')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.xticks(rotation=45)
        # save figure
        plt.savefig('sector_rsi_comparison.png')
        plt.show()

    def plot_rsi_deviation_from_average(self, start_date, end_date):
        filtered_df = self.df[(self.df['date'] >= start_date) & (self.df['date'] <= end_date)]
        sectors = filtered_df['sector'].unique()

        # plot with enough colormap for 11 different sectors
        colors = plt.cm.tab20(np.linspace(0, 1, len(sectors)))  # Use a colormap that can handle more than 10 colors

        plt.figure(figsize=(14, 8))
        for sector, color in zip(sectors, colors):
            sector_data = filtered_df[filtered_df['sector'] == sector]
            sector_data = sector_data.groupby('date').mean(numeric_only=True)  # Group by date and take the mean of numeric columns only
            if not sector_data.empty:
                avg_rsi = sector_data['rsi_14'].mean()
                sector_data['rsi_deviation'] = sector_data['rsi_14'] - avg_rsi
                plt.plot(sector_data.index, sector_data['rsi_deviation'], label=sector, linestyle='-', marker='', color=color)

        plt.title(f'Sector RSI_14 Deviation from Average from {start_date} to {end_date}')
        plt.ylabel('Deviation from Average RSI_14')
        plt.xlabel('Date')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.xticks(rotation=45)
        # save figure
        plt.savefig('sector_rsi_deviation_from_average.png')
        plt.show()

    def plot_sector_sharpe_ratio(self, start_date, end_date):
        filtered_df = self.df[(self.df['date'] >= start_date) & (self.df['date'] <= end_date)]
        sectors = filtered_df['sector'].unique()

        # plot with enough colormap for 11 different sectors
        colors = plt.cm.tab20(np.linspace(0, 1, len(sectors)))  # Use a colormap that can handle more than 10 colors
        plt.figure(figsize=(14, 8))
        for sector, color in zip(sectors, colors):
            sector_data = filtered_df[filtered_df['sector'] == sector]
            sector_data = sector_data.groupby('date').mean(numeric_only=True)  # Group by date and take the mean of numeric columns only
            if not sector_data.empty:
                # Calculate daily returns
                sector_data['daily_return'] = sector_data['adjusted_close'].pct_change()
                
                # Calculate excess return over the risk-free rate
                excess_return = sector_data['daily_return'] - sector_data['risk_free_rate']
                
                # Calculate rolling Sharpe ratio
                sector_data['sharpe_ratio'] = excess_return.rolling(window=5).mean() / sector_data['daily_return'].rolling(window=5).std()
                
                plt.plot(sector_data.index, sector_data['sharpe_ratio'], label=sector, linestyle='-', marker='', color=color)

        # Adding shaded regions for Sharpe ratios
        plt.axhspan(ymin=-float('inf'), ymax=0, color='darkred', alpha=0.3)  # Dark Red for very bad
        plt.axhspan(ymin=0, ymax=1, color='lightcoral', alpha=0.3)  # Light Red for bad
        plt.axhspan(ymin=1, ymax=2, color='yellow', alpha=0.3)  # Yellow for neutral/good
        plt.axhspan(ymin=2, ymax=3, color='lightgreen', alpha=0.3)  # Light Green for very good
        plt.axhspan(ymin=3, ymax=float('inf'), color='darkgreen', alpha=0.3)  # Dark Green for excellent

        plt.title(f'Sector Daily Sharpe Ratio from {start_date} to {end_date}')
        plt.ylabel('Sharpe Ratio')
        plt.xlabel('Date')
        plt.legend(loc='lower left')  # Legend for sectors only
        plt.grid(True)
        plt.xticks(rotation=45)
        # save figure
        plt.savefig('sector_sharpe_ratio.png')
        plt.show()

    def visualize(self, start_date, end_date):
        self.plot_sector_cumulative_percent_change(start_date, end_date)
        self.plot_rsi_comparison(start_date, end_date)
        self.plot_rsi_deviation_from_average(start_date, end_date)
        self.plot_sector_sharpe_ratio(start_date, end_date)


class MarketPerformanceAnalysis:
    def __init__(self, df):
        # Filter to include only relevant symbols
        relevant_symbols = ['SPY', 'QQQ', 'XLB', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY', 'XLE', 'XRT']
        self.df = df[df['symbol'].isin(relevant_symbols)].copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df.sort_values(by='date', inplace=True)

    def calculate_percentage_change(self, column, period='W'):
        df_reset = self.df.set_index('date')
        return df_reset.groupby('symbol').resample(period)[column].last().pct_change() * 100

    def plot_sector_correlation_over_time(self, start_date, end_date):
        filtered_df = self.df[(self.df['date'] >= start_date) & (self.df['date'] <= end_date)]
        sectors = ['XLB', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY', 'XLE', 'XRT']

        colors = plt.cm.tab20(np.linspace(0, 1, len(sectors)))  # Use a colormap that can handle more than 10 colors

        plt.figure(figsize=(14, 8))
        for sector, color in zip(sectors, colors):
            sector_df = filtered_df[filtered_df['symbol'].isin([sector, 'SPY'])].pivot(index='date', columns='symbol', values='adjusted_close')
            sector_corr = sector_df['SPY'].rolling(window=10).corr(sector_df[sector])
            plt.plot(sector_corr.index, sector_corr, label=f'{sector} vs SPY', linestyle='--', color=color)

        plt.axhline(0, color='black', linestyle='--')
        plt.title('Bi-Weekly Correlation Between Sectors and SPY')
        plt.ylabel('Correlation')
        plt.xlabel('Date')
        plt.legend(loc='upper left', ncol=2)
        plt.grid(True)
        plt.xticks(rotation=45)
        # save the plot
        plt.savefig('sector_correlation.png')
        plt.show()

    def plot_combined_market_sentiment(self, start_date, end_date):
        filtered_df = self.df[(self.df['date'] >= start_date) & (self.df['date'] <= end_date)]
        symbols = ['SPY', 'QQQ']

        sentiment_data = {}
        for symbol in symbols:
            weekly_change = self.calculate_percentage_change('adjusted_close', period='W')
            monthly_change = self.calculate_percentage_change('adjusted_close', period='M')
            quarterly_change = self.calculate_percentage_change('adjusted_close', period='Q')

            # Define quarters as January, April, July, and October
            quarterly_change = quarterly_change[(quarterly_change.index.get_level_values('date').month.isin([1, 4, 7, 10]))]

            weekly_change = weekly_change[weekly_change.index.get_level_values('symbol') == symbol]
            monthly_change = monthly_change[monthly_change.index.get_level_values('symbol') == symbol]
            quarterly_change = quarterly_change[quarterly_change.index.get_level_values('symbol') == symbol]

            weekly_change = weekly_change[(weekly_change.index.get_level_values('date') >= start_date) & (weekly_change.index.get_level_values('date') <= end_date)]
            monthly_change = monthly_change[(monthly_change.index.get_level_values('date') >= start_date) & (monthly_change.index.get_level_values('date') <= end_date)]
            quarterly_change = quarterly_change[(quarterly_change.index.get_level_values('date') >= start_date) & (quarterly_change.index.get_level_values('date') <= end_date)]

            common_index = weekly_change.index.get_level_values('date').union(monthly_change.index.get_level_values('date')).union(quarterly_change.index.get_level_values('date'))

            weekly_change = weekly_change.reset_index(level=0, drop=True).reindex(common_index)
            monthly_change = monthly_change.reset_index(level=0, drop=True).reindex(common_index)
            quarterly_change = quarterly_change.reset_index(level=0, drop=True).reindex(common_index)

            sentiment_data[symbol] = pd.DataFrame({
                'Weekly': weekly_change.values,
                'Monthly': monthly_change.values,
                'Quarterly': quarterly_change.values
            }, index=common_index)

        # Plot SPY and QQQ on the same plot using line plots with different line types and markers
        plt.figure(figsize=(14, 8))
        for symbol, data in sentiment_data.items():
            color = 'blue' if symbol == 'SPY' else 'orange'
            plt.plot(data.index, data['Weekly'], label=f'{symbol} Weekly', color=color, linestyle='--', marker='o')
            plt.plot(data.index, data['Monthly'], label=f'{symbol} Monthly', color=color, linestyle='-', marker='s')
            #plt.plot(data.index, data['Quarterly'], label=f'{symbol} Quarterly', color=color, linestyle='-.', marker='d')

        plt.axhline(3, color='green', linestyle=':', linewidth=1)
        plt.axhline(-3, color='red', linestyle=':', linewidth=1)
        plt.axhline(7, color='darkgreen', linestyle='--', linewidth=1)
        plt.axhline(-7, color='darkred', linestyle='--', linewidth=1)

        plt.title('Market Sentiment - SPY and QQQ')
        plt.ylabel('Percentage Change (%)')
        plt.xlabel('Date')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.xticks(rotation=45)
        # save the plot
        plt.savefig('market_sentiment.png')
        plt.show()

    def visualize(self, start_date, end_date):
        self.plot_sector_correlation_over_time(start_date, end_date)
        self.plot_combined_market_sentiment(start_date, end_date)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="file containing all core and technical data", type=str, default='all_data.csv')
    parser.add_argument("-s", "--symbol", help="list of stock symbols to visualize", type=str)
    parser.add_argument("-start", "--start_date", help="start date for visual analysis", type=str, default='2024-07-01')
    parser.add_argument("-end", "--end_date", help="end date for visual analysis", type=str, default='2024-08-13')
    parser.add_argument("-c", "--charts", help="comma separated list of charts to generate", type=str, default='adjusted_close,volume,rsi,macd')
    parser.add_argument("-a", "--analysis", help="comma separated list of analysis to generate", type=str, default='stock, sector,market')
    args = parser.parse_args()
    logger.info(f"Data file: {args.data}")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Charts: {args.charts}")
    logger.info(f"Analysis: {args.analysis}")
    logger.info(f"Start date: {args.start_date}")
    logger.info(f"End date: {args.end_date}")
    # Load the stock data
    df = pd.read_csv(args.data)
    # Test case using all 4 of the charts
    # create visualizer object for each symbol in arg.symbol
    # check for symbol in args.symbol
    if args.symbol is not None and 'stock' in args.analysis:
        for symbol in args.symbol.split(','):
            # make symbol to upper
            symbol = symbol.upper()
            visualizer = VisualizeTechnicals(df, symbol=symbol, start_date=args.start_date, end_date=args.end_date)
            visualizer.visualize(charts_to_generate=args.charts.split(',') if args.charts else [])
    # Sector Analysis if 'sector' is in args.analysis
    if 'sector' in args.analysis:
        analysis = SectorAnalysis(df)
        analysis.visualize(args.start_date, args.end_date)
    # Market Performance Analysis if 'market' is in args.analysis
    if 'market' in args.analysis:
        analysis = MarketPerformanceAnalysis(df)
        analysis.visualize(args.start_date, args.end_date)
