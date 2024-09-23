import logging
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys
from matplotlib.backends.backend_pdf import PdfPages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualize import VisualizeTechnicals, SectorAnalysis, MarketPerformanceAnalysis
from logging_config import setup_logging
setup_logging()
#logger = logging.getLogger(__name__)
# Configure logging
#logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class TechnicalAnalyzer:
    def __init__(self, symbols, data):
        """
        Initialize the TechnicalAnalyzer with a list of symbols and their historical data.
        data: A DataFrame with historical data containing multiple symbols.
        Columns should include: ['date', 'symbol', 'adjusted_close', 'volume', 'sma_20', 'sma_50', 'sma_200', 
                                'ema_20', 'ema_50', 'ema_200', 'macd', 'rsi_14', 'adx_14', 'atr_14', 
                                'bbands_upper_20', 'bbands_middle_20', 'bbands_lower_20']
        """
        self.symbols = symbols
        self.data = data
        self.results = []  # Store results for CSV output

        # Create directory for overnight plots if it doesn't exist
        os.makedirs('overnight_plots', exist_ok=True)

    def detect_trends(self, date, visualize=False, output_csv='technical_analysis_report.csv'):
        """
        Detect bullish, bearish trends, support and resistance levels.
        Log the findings in CSV format and optionally generate a combined visualization.
        """
        # Clear previous results
        self.results = []

        for symbol in self.symbols:
            df_symbol = self.data[self.data['symbol'] == symbol]
            df_symbol['date'] = pd.to_datetime(df_symbol['date'])
            target_row = df_symbol[df_symbol['date'] == pd.to_datetime(date)]
            
            if target_row.empty:
                log.warning(f"No data for {symbol} on {date}")
                continue
            
            row = target_row.iloc[0]
            # Analyze the indicators for bullish/bearish trends and support/resistance
            self.analyze_trends(symbol, row)
            self.analyze_support_resistance(symbol, row)
            
            # Optionally visualize the last 30 days if conditions are met
            #if visualize and len(self.results) > 0:
            #    self.visualize(symbol, df_symbol, row)

        # Save results to CSV
        if self.results:
            self.save_to_csv(output_csv)

        # Combine all the individual plots into a single PDF
        #if visualize:
            #self.combine_plots_into_pdf('overnight_plots_report.pdf')
        if visualize:
            # use the VisualizeTechnicals class to visualize the data
            # use 60 days prior to date for better visualization
            results_df = pd.DataFrame(self.results)
            # get list of unique symbols
            symbols = results_df['symbol'].unique()
            start_date = pd.to_datetime(date) - pd.DateOffset(days=60)
            for symbol in symbols:
                try:
                    visualizer = VisualizeTechnicals(self.data, symbol=symbol, start_date=start_date, end_date=date, plot=False)
                    #visualizer.visualize(charts_to_generate=['adjusted_close','volume','rsi','macd','pe_ratio','sharpe_ratio'])
                    visualizer.visualize(charts_to_generate=['adjusted_close','volume','rsi','macd','sharpe_ratio'])
                except Exception as e:
                    log.error(f"Error visualizing {symbol}: {e}")
            analysis = SectorAnalysis(self.data, plot=False)
            analysis.visualize(start_date, date)
                #analysis = MarketPerformanceAnalysis(df, plot=False)
                #analysis.visualize(start_date, date)

    def analyze_trends(self, symbol, row):
        """
        Analyze the row for bullish or bearish signals based on MACD, RSI, SMA, EMA, etc.
        Only logs trends if there is a bullish or bearish signal.
        """
        technical_reasons = []
        
        # Bullish Signals
        if row['macd'] > 0 and row['rsi_14'] < 70 and row['sma_20'] > row['sma_50'] > row['sma_200']:
            technical_reasons.append('Bullish MACD, RSI, SMA crossover')
            log.info(f"{symbol}, bull, {'Bullish MACD, RSI, SMA crossover'}")

        # Bearish Signals
        if row['macd'] < 0 and row['rsi_14'] > 30 and row['sma_20'] < row['sma_50'] < row['sma_200']:
            technical_reasons.append('Bearish MACD, RSI, SMA crossover')
            log.info(f"{symbol}, bear, {'Bearish MACD, RSI, SMA crossover'}")

        # Log results if bullish or bearish trends are detected
        if technical_reasons:
            self.results.append({
                'symbol': symbol,
                'direction': 'bull' if row['macd'] > 0 else 'bear',
                'technical_reasons': '; '.join(technical_reasons)
            })

    def analyze_support_resistance(self, symbol, row):
        """
        Detect if the price is near support or resistance levels within a 1% threshold.
        Only logs if there is a support or resistance level within 1%.
        """
        tolerance = 0.01  # 1% tolerance
        adjusted_close = row['adjusted_close']

        # Check for support (bollinger bands lower)
        if abs(adjusted_close - row['bbands_lower_20']) / adjusted_close < tolerance:
            self.results.append({
                'symbol': symbol,
                'direction': 'support',
                'technical_reasons': 'Near Bollinger Bands Lower'
            })
            log.info(f"{symbol}, support, {'Near Bollinger Bands Lower'}")

        # Check for resistance (bollinger bands upper)
        if abs(adjusted_close - row['bbands_upper_20']) / adjusted_close < tolerance:
            self.results.append({
                'symbol': symbol,
                'direction': 'resistance',
                'technical_reasons': 'Near Bollinger Bands Upper'
            })
            log.info(f"{symbol}, resistance, {'Near Bollinger Bands Upper'}")

    def visualize(self, symbol, df_symbol, row):
        """
        Visualize the last 30 days of price and mark any bullish/bearish trends, support/resistance.
        Save the individual plot as a PNG file to 'overnight_plots/' directory.
        """
        df_symbol = df_symbol[df_symbol['date'] <= row['date']].tail(30)
        plt.figure(figsize=(10, 6))

        # Plot adjusted_close
        plt.plot(df_symbol['date'], df_symbol['adjusted_close'], label='Adjusted Close', color='blue')

        # Plot support and resistance levels
        plt.plot(df_symbol['date'], df_symbol['bbands_upper_20'], label='Bollinger Upper', color='red', linestyle='--')
        plt.plot(df_symbol['date'], df_symbol['bbands_lower_20'], label='Bollinger Lower', color='green', linestyle='--')

        # Highlight any signals from the previous 30 days
        plt.scatter(df_symbol['date'], df_symbol['adjusted_close'], marker='o', color='orange', label='Key levels')

        # Add labels and legend
        plt.title(f"{symbol} Price and Key Levels")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot to 'overnight_plots/' directory
        plt.savefig(f"overnight_plots/{symbol}_visualization.png")
        plt.close()

    def save_to_csv(self, output_csv):
        """
        Save the results to a CSV file.
        """
        df = pd.DataFrame(self.results)
        df.to_csv(output_csv, index=False)
        log.info(f"Results saved to {output_csv}")

    def combine_plots_into_pdf(self, output_pdf):
        """
        Combine individual symbol visualizations from 'overnight_plots/' into a single PDF file.
        """
        symbols_with_plots = [symbol for symbol in self.symbols if os.path.exists(f"overnight_plots/{symbol}_visualization.png")]

        if not symbols_with_plots:
            log.warning("No individual plots found to combine.")
            return

        # Create a PDF to combine the individual PNGs
        with PdfPages(output_pdf) as pdf:
            for symbol in symbols_with_plots:
                img = plt.imread(f"overnight_plots/{symbol}_visualization.png")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(img)
                ax.axis('off')
                pdf.savefig(fig)
                plt.close(fig)
        
        log.info(f"Combined visualization saved as {output_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="file containing all core and technical data", type=str, default='../api_data/all_data.csv')
    parser.add_argument("-start", "--start_date", help="analysis date, default is latest date in data", type=str, default=None)
    args = parser.parse_args()
    log.info(f"Data file: {args.data}")
    # Load the stock data
    df = pd.read_csv(args.data)
    
    # Ensure the date is properly formatted
    df['date'] = pd.to_datetime(df['date'])
    
    # Get the list of unique symbols from the data
    symbols = df['symbol'].unique()

    # get most recent date
    if not args.start_date:
        latest_date = df['date'].max()
    else:
        latest_date = pd.to_datetime(args.start_date)
    log.info(f"Latest date: {latest_date}")

    # Create an instance of the analyzer
    analyzer = TechnicalAnalyzer(symbols, df)

    # Detect trends for a given date and visualize
    analyzer.detect_trends(latest_date, visualize=True)

    
