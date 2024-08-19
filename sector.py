import matplotlib.pyplot as plt
import pandas as pd
import argparse

import logging
from logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

class SectorAnalysis:
    def __init__(self, df, use_treasury='10year'):
        self.df = df
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

        plt.figure(figsize=(14, 6))
        for sector, changes in cumulative_changes.items():
            plt.plot(changes.index, changes, label=sector, linestyle='-', marker='')

        plt.title(f'Sector Cumulative Percent Change from {start_date} to {end_date}')
        plt.ylabel('Cumulative Percent Change (%)')
        plt.xlabel('Date')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.show()

    def plot_rsi_comparison(self, start_date, end_date):
        filtered_df = self.df[(self.df['date'] >= start_date) & (self.df['date'] <= end_date)]
        sectors = filtered_df['sector'].unique()

        plt.figure(figsize=(14, 6))
        for sector in sectors:
            sector_data = filtered_df[filtered_df['sector'] == sector]
            sector_data = sector_data.groupby('date').mean(numeric_only=True)  # Group by date and take the mean of numeric columns only
            if not sector_data.empty:
                plt.plot(sector_data.index, sector_data['rsi_14'], label=sector, linestyle='-', marker='')

        # Adding the overbought and oversold lines without adding them to the legend
        plt.axhline(70, color='red', linestyle='--', linewidth=2)
        plt.axhline(30, color='blue', linestyle='--', linewidth=2)

        plt.title(f'Sector RSI_14 Comparison from {start_date} to {end_date}')
        plt.ylabel('RSI_14')
        plt.xlabel('Date')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.show()

    def plot_rsi_deviation_from_average(self, start_date, end_date):
        filtered_df = self.df[(self.df['date'] >= start_date) & (self.df['date'] <= end_date)]
        sectors = filtered_df['sector'].unique()

        plt.figure(figsize=(14, 6))
        for sector in sectors:
            sector_data = filtered_df[filtered_df['sector'] == sector]
            sector_data = sector_data.groupby('date').mean(numeric_only=True)  # Group by date and take the mean of numeric columns only
            if not sector_data.empty:
                avg_rsi = sector_data['rsi_14'].mean()
                sector_data['rsi_deviation'] = sector_data['rsi_14'] - avg_rsi
                plt.plot(sector_data.index, sector_data['rsi_deviation'], label=sector, linestyle='-', marker='')

        plt.title(f'Sector RSI_14 Deviation from Average from {start_date} to {end_date}')
        plt.ylabel('Deviation from Average RSI_14')
        plt.xlabel('Date')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.show()

    def plot_sector_sharpe_ratio(self, start_date, end_date):
        filtered_df = self.df[(self.df['date'] >= start_date) & (self.df['date'] <= end_date)]
        sectors = filtered_df['sector'].unique()

        plt.figure(figsize=(14, 6))
        for sector in sectors:
            sector_data = filtered_df[filtered_df['sector'] == sector]
            sector_data = sector_data.groupby('date').mean(numeric_only=True)  # Group by date and take the mean of numeric columns only
            if not sector_data.empty:
                # Calculate daily returns
                sector_data['daily_return'] = sector_data['adjusted_close'].pct_change()
                
                # Calculate excess return over the risk-free rate
                excess_return = sector_data['daily_return'] - sector_data['risk_free_rate']
                
                # Calculate rolling Sharpe ratio
                sector_data['sharpe_ratio'] = excess_return.rolling(window=5).mean() / sector_data['daily_return'].rolling(window=5).std()
                
                plt.plot(sector_data.index, sector_data['sharpe_ratio'], label=sector, linestyle='-', marker='')

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
        plt.show()

    def visualize(self, start_date, end_date):
        self.plot_sector_cumulative_percent_change(start_date, end_date)
        self.plot_rsi_comparison(start_date, end_date)
        self.plot_rsi_deviation_from_average(start_date, end_date)
        self.plot_sector_sharpe_ratio(start_date, end_date)


# Example usage
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="file containing all core and technical data", type=str, default='all_data.csv')
    parser.add_argument("-start", "--start_date", help="start date for visual analysis", type=str, default='2024-07-01')
    parser.add_argument("-end", "--end_date", help="end date for visual analysis", type=str, default='2024-08-13')
    args = parser.parse_args()
    logger.info(f"Data file: {args.data}")
    logger.info(f"Start date: {args.start_date}")
    logger.info(f"End date: {args.end_date}")
    # Load the stock data
    df = pd.read_csv(args.data)
    # list of sectors
    sectors = ['XLB', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY', 'XLE', 'XRT']
    # list of industries
    industries = [' SPDR FUND MATERIALS SELECT SECTR ETF',' SELECT STR FINANCIAL SELECT SPDR ETF',' SELECT SECTOR INDUSTRIAL SPDR ETF',
                  ' TECHNOLOGY SELECT SECTOR SPDR ETF',' SPDR FUND CONSUMER STAPLES ETF',' REAL ESTATE SELECT SCTR SPDR ETF',
                  ' SELECT SECTOR UTI SELECT SPDR ETF',' SELECT SECTOR HEALTH CARE SPDR ETF',' SPDR FUND CONSUMER DISCRE SELECT ETF',
                  ' ENERGY SELECT SECTOR SPDR ETF',' SPDR S&P RETAIL ETF']
    # create dictionary of sectors and industries
    sector_industry = dict(zip(sectors, industries))
    # filter df to only rows where symbol is one of the sector symbols
    df = df[df['symbol'].isin(sectors)]
    # fill df sector column with the corresponding industry
    df['sector'] = df['symbol'].map(sector_industry)
    analysis = SectorAnalysis(df)
    analysis.visualize(args.start_date, args.end_date)

