import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from datetime import datetime
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)
from datetime import datetime, timedelta

class StockScreener:
    def __init__(self, symbols, date, indicators='all', visualize=True, n_days=30, use_candlesticks=False, data='../api_data/all_data.csv'):
        self.symbols = [symbol.upper() for symbol in symbols]
        self.date = pd.to_datetime(self._get_closest_trading_date(date)).strftime('%Y-%m-%d')
        self.indicators = indicators
        self.visualize = visualize
        self.n_days = n_days
        # not implemented
        self.use_candlesticks = use_candlesticks
        self.data = pd.read_csv(data)
        self.results = []

    def _get_closest_trading_date(self, input_date):
        input_date = datetime.strptime(input_date, '%Y-%m-%d')
        while input_date.weekday() > 4:  # If it's Saturday (5) or Sunday (6), move to Friday
            input_date -= timedelta(days=1)
        # Assuming all weekends are non-trading days, for simplicity
        return input_date.strftime('%Y-%m-%d')

    def find_nearest_two_dates(self, target_date):
        target_date = pd.to_datetime(target_date)
        available_dates = pd.to_datetime(self.data['date']).drop_duplicates().sort_values()

        # Calculate the absolute difference between target date and available dates
        differences = abs(available_dates - target_date)
        sorted_dates = available_dates.iloc[differences.argsort()]

        # Get the nearest two dates
        nearest_dates = sorted_dates[:2].sort_values(ascending=False)
        return nearest_dates.tolist()

    def find_nearest_three_dates(self, target_date):
        target_date = pd.to_datetime(target_date)
        available_dates = pd.to_datetime(self.data['date']).drop_duplicates().sort_values()
        
        # Filter to only include dates before or at the target date
        available_dates = available_dates[available_dates <= target_date]
        
        # Get the three most recent dates before the target date
        nearest_dates = available_dates[-3:].sort_values(ascending=False)
        return nearest_dates.tolist()

    def run_screen(self):
        # Get the two nearest dates to the target date
        nearest_dates = self.find_nearest_three_dates(self.date)
        self.latest_date = nearest_dates[0]
        previous_date = nearest_dates[1]
        day_before_previous_date = nearest_dates[2]
        #print dates
        print(self.latest_date)
        print(previous_date)
        print(day_before_previous_date)
        # Create output directory for plots
        output_dir = f'overnight_{self.date}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir
        logging.info(f'Running stock screener for symbols: {self.symbols} on date: {self.date}')
        sectors = ['XLB', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY', 'XLE', 'XRT', 'SPY', 'QQQ']
        for symbol in self.symbols:
            # skip sector ETFs
            if symbol in sectors:
                continue
            symbol_data = self.data[self.data['symbol'] == symbol]
            symbol_data.loc[:, 'date'] = pd.to_datetime(symbol_data['date'], format='%Y-%m-%d')
            symbol_data = symbol_data.sort_values('date')
            symbol_data = symbol_data[symbol_data['date'] <= pd.to_datetime(self.latest_date)]
            latest_date_data = symbol_data[symbol_data['date'] == pd.to_datetime(self.latest_date)]
            previous_date_data = symbol_data[symbol_data['date'] == pd.to_datetime(previous_date)]
            day_before_previous_date_data = symbol_data[symbol_data['date'] == pd.to_datetime(day_before_previous_date)]
            if latest_date_data.empty or previous_date_data.empty:
                logging.warning(f'No data available for symbol {symbol} on date {self.date}')
                continue
            
            latest_bullish, latest_bearish, latest_signals = self._check_signals(latest_date_data, previous_date_data)
            previous_bullish, previous_bearish, previous_signals = self._check_signals(previous_date_data, day_before_previous_date_data)
            
            # Check if there's a change in the number of bullish or bearish signals
            if len(latest_bullish) != len(previous_bullish) or len(latest_bearish) != len(previous_bearish):
                if len(latest_bullish) != len(previous_bullish) or len(latest_bearish) != len(previous_bearish):
                    self.results.append([symbol, len(latest_bullish), len(latest_bearish), *latest_signals])
            
            
            if self.visualize and (len(latest_bullish) != len(previous_bullish) or len(latest_bearish) != len(previous_bearish)):
                self._visualize(symbol, symbol_data, latest_bullish, latest_bearish)

        # run the sector/market ETFs once only
        self._process_sector_data(self.data)
        # Write results to CSV
        self._write_results()

    def _check_macd(self, selected_date_data, bullish_signals, bearish_signals, signals):
        macd = selected_date_data['macd'].values[0]
        if macd > 0:
            bullish_signals.append('bullish_macd')
            signals.append(1)
        elif macd < 0:
            bearish_signals.append('bearish_macd')
            signals.append(-1)
        else:
            signals.append(0)

    def _check_adx(self, selected_date_data, bullish_signals, bearish_signals, signals):
        adx = selected_date_data['adx_14'].values[0]
        if adx > 25:
            bullish_signals.append('bullish_adx')
            signals.append(1)
        elif adx < 20:
            bearish_signals.append('bearish_adx')
            signals.append(-1)
        else:
            signals.append(0)

    def _check_atr(self, selected_date_data, bullish_signals, bearish_signals, signals):
        atr = selected_date_data['atr_14'].values[0]
        # Define ATR thresholds based on volatility conditions
        if atr > 2:
            bearish_signals.append('bearish_atr')
            signals.append(-1)
        elif atr < 1:
            bullish_signals.append('bullish_atr')
            signals.append(1)
        else:
            signals.append(0)

    def _check_signals(self, symbol_data, selected_date_data):
        bullish_signals = []
        bearish_signals = []
        signals = []

        if self.indicators == 'all' or 'macd' in self.indicators:
            self._check_macd(selected_date_data, bullish_signals, bearish_signals, signals)
        if self.indicators == 'all' or 'adx' in self.indicators:
            self._check_adx(selected_date_data, bullish_signals, bearish_signals, signals)
        if self.indicators == 'all' or 'atr' in self.indicators:
            self._check_atr(selected_date_data, bullish_signals, bearish_signals, signals)
        if self.indicators == 'all' or 'pe_ratio' in self.indicators:
            self._check_pe_ratio(selected_date_data, bullish_signals, bearish_signals, signals)
        if self.indicators == 'all' or 'bollinger_band' in self.indicators:
            self._check_bollinger_band(selected_date_data, bullish_signals, bearish_signals, signals)
        if self.indicators == 'all' or 'rsi' in self.indicators:
            self._check_rsi(selected_date_data, bullish_signals, bearish_signals, signals)
        if self.indicators == 'all' or 'sma_cross' in self.indicators:
            self._check_sma_cross(selected_date_data, bullish_signals, bearish_signals, signals)

        return bullish_signals, bearish_signals, signals

    def _check_pe_ratio(self, selected_date_data, bullish_signals, bearish_signals, signals):
        pe_ratio = selected_date_data['adjusted_pe_ratio'].values[0]
        # Append a signal for PE ratio
        if pe_ratio < 15:
            bullish_signals.append('bullish_pe_ratio')
            signals.append(1)
        elif pe_ratio > 35:
            bearish_signals.append('bearish_pe_ratio')
            signals.append(-1)
        else:
            signals.append(0)

    def _check_bollinger_band(self, selected_date_data, bullish_signals, bearish_signals, signals):
        adj_close = selected_date_data['adjusted_close'].values[0]
        bb_upper = selected_date_data['bbands_upper_20'].values[0]
        bb_lower = selected_date_data['bbands_lower_20'].values[0]
        if adj_close > bb_upper:
            bearish_signals.append('bearish_bollinger_band')
            signals.append(-1)
        elif adj_close < bb_lower:
            bullish_signals.append('bullish_bollinger_band')
            signals.append(1)
        else:
            signals.append(0)

    def _check_rsi(self, selected_date_data, bullish_signals, bearish_signals, signals):
        rsi = selected_date_data['rsi_14'].values[0]
        if rsi > 70:
            bearish_signals.append('bearish_rsi')
            signals.append(-1)
        elif rsi < 30:
            bullish_signals.append('bullish_rsi')
            signals.append(1)
        else:
            signals.append(0)

    def _check_sma_cross(self, selected_date_data, bullish_signals, bearish_signals, signals):
        sma_20 = selected_date_data['sma_20'].values[0]
        sma_50 = selected_date_data['sma_50'].values[0]
        if sma_20 > sma_50:
            bullish_signals.append('bullish_sma_cross')
            signals.append(1)
        elif sma_20 < sma_50:
            bearish_signals.append('bearish_sma_cross')
            signals.append(-1)
        else:
            signals.append(0)

    def _plot_pe_ratio(self, symbol, symbol_data):
        output_path = os.path.join(self.output_dir, f'{symbol}_technical_pe_ratio.png')
        #plt.figure()
        plt.figure(figsize=(14, 10))
        pe_ratio = symbol_data['adjusted_pe_ratio']

        # Plot PE Ratio with different colors based on thresholds
        for i in range(len(symbol_data) - 1):
            if pe_ratio.iloc[i] > 35:
                plt.plot([symbol_data['date'].iloc[i], symbol_data['date'].iloc[i + 1]], [pe_ratio.iloc[i], pe_ratio.iloc[i + 1]], color='red')
            elif pe_ratio.iloc[i] < 15:
                plt.plot([symbol_data['date'].iloc[i], symbol_data['date'].iloc[i + 1]], [pe_ratio.iloc[i], pe_ratio.iloc[i + 1]], color='green')
            else:
                plt.plot([symbol_data['date'].iloc[i], symbol_data['date'].iloc[i + 1]], [pe_ratio.iloc[i], pe_ratio.iloc[i + 1]], color='black')

        # Add annotations for the first bullish and bearish signals
        first_bullish = True
        first_bearish = True
        for i, row in symbol_data.iterrows():
            if row['adjusted_pe_ratio'] < 15 and first_bullish:
                plt.annotate('Bullish (Low PE)', (row['date'], row['adjusted_pe_ratio']), textcoords="offset points", xytext=(0,10), ha='center', color='green')
                first_bullish = False
            elif row['pe_ratio'] > 35 and first_bearish:
                plt.annotate('Bearish (High PE)', (row['date'], row['adjusted_pe_ratio']), textcoords="offset points", xytext=(0,10), ha='center', color='red')
                first_bearish = False

        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.ylabel('P/E Ratio')
        plt.title(f'{symbol} - PE Ratio Analysis')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _plot_macd(self, symbol, symbol_data):
        output_path = os.path.join(self.output_dir, f'{symbol}_technical_macd.png')
        #plt.figure()
        plt.figure(figsize=(14, 10))
        macd = symbol_data['macd']
        max_abs_macd = max(abs(macd.min()), abs(macd.max()))
        plt.ylim(-max_abs_macd, max_abs_macd)
        signal_line = symbol_data['macd']
        plt.plot(symbol_data['date'], macd, label='MACD', color='blue')
        plt.plot(symbol_data['date'], signal_line, label='Signal Line', color='red')
        
        # Add annotations for MACD crossovers
        for i in range(1, len(symbol_data)):
            if macd.iloc[i] > 0 and macd.iloc[i - 1] <= 0:
                plt.annotate('Bullish Zero Crossover', 
                             xy=(symbol_data['date'].iloc[i-1], macd.iloc[i-1]), 
                             xytext=(0, 20),
                             textcoords='offset points',
                             ha='center',
                             color='green',
                             fontsize=8,
                             bbox=dict(boxstyle='round,pad=0.3', edgecolor='green', facecolor='white'),
                             arrowprops=dict(arrowstyle='->', color='green'))
            elif macd.iloc[i] < 0 and macd.iloc[i - 1] >= 0:
                plt.annotate('Bearish Zero Crossover', 
                             xy=(symbol_data['date'].iloc[i-1], macd.iloc[i-1]), 
                             xytext=(0, -20),
                             textcoords='offset points',
                             ha='center',
                             color='red',
                             fontsize=8,
                             bbox=dict(boxstyle='round,pad=0.3', edgecolor='red', facecolor='white'),
                             arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.axhline(0, color='black', linestyle='--')
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.axvline(pd.to_datetime(self.latest_date), color='grey', linestyle='--', label='Analysis Date')
        plt.ylabel('MACD')
        plt.title(f'{symbol} - MACD')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _plot_adx(self, symbol, symbol_data):
        output_path = os.path.join(self.output_dir, f'{symbol}_technical_adx.png')
        #plt.figure()
        plt.figure(figsize=(14, 10))
        adx = symbol_data['adx_14']
        plt.plot(symbol_data['date'], adx, label='ADX', color='black')
        plt.fill_between(symbol_data['date'], adx, where=(adx > 70), color='purple', alpha=0.3, interpolate=True, label='Very Strong Trend (ADX > 70)')
        plt.fill_between(symbol_data['date'], adx, where=(adx > 25) & (adx <= 70), color='green', alpha=0.3, interpolate=True, label='Strong Trend (ADX > 25)')
        plt.fill_between(symbol_data['date'], adx, where=(adx >= 20) & (adx <= 25), color='yellow', alpha=0.3, interpolate=True, label='Weak Trend (20 <= ADX <= 25)')
        plt.fill_between(symbol_data['date'], adx, where=(adx < 20), color='red', alpha=0.3, interpolate=True, label='Very Weak Trend (ADX < 20)')
        
        plt.axhline(70, color='purple', linestyle='--', label='Very Strong Trend Threshold (70)')
        plt.axhline(25, color='green', linestyle='--', label='Strong Trend Threshold (25)')
        plt.axhline(20, color='red', linestyle='--', label='Weak Trend Threshold (20)')
        
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.ylabel('ADX')
        plt.title(f'{symbol} - ADX Indicator')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _plot_atr(self, symbol, symbol_data):
        output_path = os.path.join(self.output_dir, f'{symbol}_technical_atr.png')
        #plt.figure()
        plt.figure(figsize=(14, 10))
        atr = symbol_data['atr_14']
        adjusted_close = symbol_data['adjusted_close']
        
        # Plot adjusted close over the n_day period
        plt.plot(symbol_data['date'], adjusted_close, label='Adjusted Close', color='black')
        # plt.plot(symbol_data['date'], atr, label='ATR', color='blue')
        
        # Calculate take profit and stop loss levels based on the most recent date's adjusted close and ATR
        most_recent_close = adjusted_close.iloc[-1]
        most_recent_atr = atr.iloc[-1]
        take_profit_level = most_recent_close + 2 * most_recent_atr
        stop_loss_level = most_recent_close - 2 * most_recent_atr
        
        # Plot take profit and stop loss levels
        plt.axhline(take_profit_level, color='red', linestyle='--', label='Take Profit Level (Adjusted Close + 2 * ATR)')
        plt.axhline(stop_loss_level, color='green', linestyle='--', label='Stop Loss Level (Adjusted Close - 2 * ATR)')

        # Add annotations for 'Buy Watch' and 'Sell Watch'
        take_profit_one_percent = take_profit_level - (take_profit_level * 0.01)
        stop_loss_one_percent = stop_loss_level + (stop_loss_level * 0.01)
        for i, row in symbol_data.iterrows():
            if row['adjusted_close'] >= take_profit_one_percent and row['adjusted_close'] <= take_profit_level:
                plt.annotate('Profit Cap Watch', (row['date'], row['adjusted_close']), textcoords="offset points", xytext=(0,10), ha='center', color='orange')
            elif row['adjusted_close'] <= stop_loss_one_percent and row['adjusted_close'] >= stop_loss_level:
                plt.annotate('Stop Loss Watch', (row['date'], row['adjusted_close']), textcoords="offset points", xytext=(0,10), ha='center', color='blue')
        
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.ylabel('ATR, Adjusted Close, and Stop Loss/Take Profit Levels')
        plt.title(f'{symbol} - ATR Indicator with Stop Loss and Take Profit Levels')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _plot_price_sma(self, symbol, symbol_data, title='daily_price'):
        output_path = os.path.join(self.output_dir, f'{symbol}_{title}.png')
        #plt.figure()
        plt.figure(figsize=(14, 10))
        plt.plot(symbol_data['date'], symbol_data['adjusted_close'], label='Adjusted Close', color='black')
        plt.plot(symbol_data['date'], symbol_data['sma_20'], label='SMA 20', color='orange', linestyle='--')
        plt.plot(symbol_data['date'], symbol_data['sma_50'], label='SMA 50', color='green', linestyle='--')
        plt.plot(symbol_data['date'], symbol_data['sma_200'], label='SMA 200', color='red', linestyle='--')
       
        # Add annotations for SMA crossovers
        for i in range(1, len(symbol_data)):
            if symbol_data['sma_20'].iloc[i] > symbol_data['sma_50'].iloc[i] and symbol_data['sma_20'].iloc[i - 1] <= symbol_data['sma_50'].iloc[i - 1]:
                plt.annotate('Bullish Cross 20/50', (symbol_data['date'].iloc[i-1], symbol_data['sma_20'].iloc[i-1]),
                             textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='green', arrowprops=dict(arrowstyle='->', color='green'))
            elif symbol_data['sma_20'].iloc[i] < symbol_data['sma_50'].iloc[i] and symbol_data['sma_20'].iloc[i - 1] >= symbol_data['sma_50'].iloc[i - 1]:
                plt.annotate('Bearish Cross 20/50', (symbol_data['date'].iloc[i], symbol_data['sma_20'].iloc[i]),
                             textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red', arrowprops=dict(arrowstyle='->', color='red'))
            if symbol_data['sma_20'].iloc[i] > symbol_data['sma_200'].iloc[i] and symbol_data['sma_20'].iloc[i - 1] <= symbol_data['sma_200'].iloc[i - 1]:
                plt.annotate('Bullish Cross 20/200', (symbol_data['date'].iloc[i-1], symbol_data['sma_200'].iloc[i-1]),
                             textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='green', arrowprops=dict(arrowstyle='->', color='green'))
            elif symbol_data['sma_20'].iloc[i] < symbol_data['sma_200'].iloc[i] and symbol_data['sma_20'].iloc[i - 1] >= symbol_data['sma_200'].iloc[i - 1]:
                plt.annotate('Bearish Cross 20/200', (symbol_data['date'].iloc[i-1], symbol_data['sma_200'].iloc[i-1]),
                             textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red', arrowprops=dict(arrowstyle='->', color='red'))
        
        # ensure the y axis contains the entire range of prices, smas
        # find min and max of any data used in the plot
        min_val = min(symbol_data['adjusted_close'].min(), symbol_data['sma_20'].min(), symbol_data['sma_50'].min(), symbol_data['sma_200'].min())
        max_val = max(symbol_data['adjusted_close'].max(), symbol_data['sma_20'].max(), symbol_data['sma_50'].max(), symbol_data['sma_200'].max())
        plt.ylim(min_val * 0.95, max_val * 1.05)  # Adjust y-limits for better visibility
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.ylabel('Price')
        plt.title(f'{symbol} - Price and SMAs')
        plt.legend()
        plt.tight_layout()
        plt.ylim(min(symbol_data['adjusted_close']) * 0.95, max(symbol_data['adjusted_close']) * 1.05)  # Adjust y-limits for better visibility
        plt.savefig(output_path)
        plt.close()

    
    def _plot_volume(self, symbol, symbol_data):
        output_path = os.path.join(self.output_dir, f'{symbol}_daily_volume.png')
        #plt.figure()
        plt.figure(figsize=(14, 10))
        plt.bar(symbol_data['date'], symbol_data['volume'], alpha=0.3, label='Volume')
        
        # Calculate average volume and mark high volume days
        avg_volume = symbol_data['volume'].mean()
        high_volume = symbol_data['volume'] > (1.5 * avg_volume)
        plt.axhline(avg_volume, color='black', linestyle='--', label='Average Volume')
        for idx, is_high in enumerate(high_volume):
            if is_high:
                plt.annotate('High Volume', (symbol_data['date'].iloc[idx], symbol_data['volume'].iloc[idx]),
                             textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red')
        
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.ylabel('Volume')
        plt.title(f'{symbol} - Daily Volume (+50% Above Average)')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _plot_rsi(self, symbol, symbol_data):
        output_path = os.path.join(self.output_dir, f'{symbol}_technical_rsi.png')
        #plt.figure()
        plt.figure(figsize=(14, 10))
        plt.plot(symbol_data['date'], symbol_data['rsi_14'], label='RSI 14')
        plt.axhline(70, color='red', linestyle='--', label='Overbought Level (70)')
        plt.axhline(30, color='green', linestyle='--', label='Oversold Level (30)')
        
        # Add markers for RSI signals
        first_bearish = True
        first_bullish = True
        for i, row in symbol_data.iterrows():
            if row['rsi_14'] > 70 and first_bearish:
                plt.annotate('Bearish (Overbought)', (row['date'], row['rsi_14']), textcoords="offset points", xytext=(0,10), ha='center', color='red')
                first_bearish = False
            elif row['rsi_14'] < 30 and first_bullish:
                plt.annotate('Bullish (Oversold)', (row['date'], row['rsi_14']), textcoords="offset points", xytext=(0,10), ha='center', color='green')
                first_bullish = False
        
        # Add annotations for 'Buy Watch' and 'Sell Watch'
        for i, row in symbol_data.iterrows():
            if 65 <= row['rsi_14'] <= 70:
                plt.annotate('Sell Watch', (row['date'], row['rsi_14']), textcoords="offset points", xytext=(0,10), ha='center', color='orange')
            elif 30 <= row['rsi_14'] <= 35:
                plt.annotate('Buy Watch', (row['date'], row['rsi_14']), textcoords="offset points", xytext=(0,10), ha='center', color='blue')
        
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.ylabel('RSI')
        plt.title(f'{symbol} - RSI Indicator')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


    def _plot_bollinger_band(self, symbol, symbol_data):
        output_path = os.path.join(self.output_dir, f'{symbol}_technical_bband.png')
        #plt.figure()
        plt.figure(figsize=(14, 10))
        adj_close = symbol_data['adjusted_close']
        bb_upper = symbol_data['bbands_upper_20']
        bb_lower = symbol_data['bbands_lower_20']
        
        first_bullish = True
        first_bearish = True
        # Plot adjusted close with different colors depending on its position relative to Bollinger Bands
        for i in range(len(symbol_data) - 1):
            if adj_close.iloc[i] > bb_upper.iloc[i]:
                plt.plot([symbol_data['date'].iloc[i], symbol_data['date'].iloc[i + 1]], [adj_close.iloc[i], adj_close.iloc[i + 1]], color='red')
                if first_bearish:
                    plt.annotate('Bearish Breakout', (symbol_data['date'].iloc[i], adj_close.iloc[i]), textcoords="offset points", xytext=(0,10), ha='center', color='red')
                    first_bearish = False
            elif adj_close.iloc[i] < bb_lower.iloc[i]:
                plt.plot([symbol_data['date'].iloc[i], symbol_data['date'].iloc[i + 1]], [adj_close.iloc[i], adj_close.iloc[i + 1]], color='green')
                if first_bullish:
                    plt.annotate('Bullish Breakout', (symbol_data['date'].iloc[i], adj_close.iloc[i]), textcoords="offset points", xytext=(0,10), ha='center', color='green')
                    first_bullish = False
            else:
                plt.plot([symbol_data['date'].iloc[i], symbol_data['date'].iloc[i + 1]], [adj_close.iloc[i], adj_close.iloc[i + 1]], color='black')
        
        # Add annotations for 'Buy Watch' and 'Sell Watch'
        for i, row in symbol_data.iterrows():
            if abs(row['adjusted_close'] - row['bbands_upper_20']) / row['bbands_upper_20'] <= 0.01:
                plt.annotate('Sell Watch', (row['date'], row['adjusted_close']), textcoords="offset points", xytext=(0,10), ha='center', color='orange')
            elif abs(row['adjusted_close'] - row['bbands_lower_20']) / row['bbands_lower_20'] <= 0.01:
                plt.annotate('Buy Watch', (row['date'], row['adjusted_close']), textcoords="offset points", xytext=(0,10), ha='center', color='blue')
        
        plt.plot(symbol_data['date'], bb_upper, linestyle='--', label='Bollinger Band Upper', color='blue')
        plt.plot(symbol_data['date'], bb_lower, linestyle='--', label='Bollinger Band Lower', color='blue')
        
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.ylabel('Price')
        plt.title(f'{symbol} - Bollinger Bands')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _visualize(self, symbol, symbol_data, bullish, bearish):
        # Filter the data for the past n days
        symbol_data = symbol_data[-self.n_days:]
        # print out the most recent date in symbol_data
        print(symbol_data['date'].iloc[-1])
        # how many rows
        print(len(symbol_data))

        # Plot individual indicators if there are bullish or bearish signals
        if bullish or bearish:
            logging.info(f'Visualizing data for symbol {symbol}')
            if 'macd' in self.indicators or self.indicators == 'all':
                self._plot_macd(symbol, symbol_data)
            if 'adx' in self.indicators or self.indicators == 'all':
                self._plot_adx(symbol, symbol_data)
            if 'atr' in self.indicators or self.indicators == 'all':
                self._plot_atr(symbol, symbol_data)
            if 'rsi' in self.indicators or self.indicators == 'all':
                self._plot_rsi(symbol, symbol_data)
            if 'bollinger_band' in self.indicators or self.indicators == 'all':
                self._plot_bollinger_band(symbol, symbol_data)
            if 'pe_ratio' in self.indicators or self.indicators == 'all':
                self._plot_pe_ratio(symbol, symbol_data)
            self._plot_price_sma(symbol, symbol_data)
            self._plot_volume(symbol, symbol_data)
            self._plot_symbol_sharpe_ratio(symbol, symbol_data)

        #output_path = os.path.join(self.output_dir, f'{symbol}_chart.png')

        #fig, ax = plt.subplots()
        #plt.plot(symbol_data['date'], symbol_data['adjusted_close'], label='Adjusted Close')
        #plt.plot(symbol_data['date'], symbol_data['sma_20'], label='SMA 20')
        #plt.plot(symbol_data['date'], symbol_data['sma_50'], label='SMA 50')
        #plt.plot(symbol_data['date'], symbol_data['sma_200'], label='SMA 200')
        #plt.bar(symbol_data['date'], symbol_data['volume'], alpha=0.3, label='Volume')
#
        #markers = set()
        #for signal in bullish + bearish:
            #if signal not in markers:
                #if signal in bullish:
                    #plt.scatter(symbol_data['date'].iloc[-1], symbol_data['adjusted_close'].iloc[-1], color='green', label=signal)
                #if signal in bearish:
                    #plt.scatter(symbol_data['date'].iloc[-1], symbol_data['adjusted_close'].iloc[-1], color='red', label=signal)
                #markers.add(signal)
#
        #plt.xlabel('Date')
        #plt.xticks(rotation=45)
        #ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        #plt.ylabel('Price')
        #plt.title(f'{symbol} - {self.date}')
        #plt.legend()
        #plt.tight_layout()
#
        #plt.close()

    def _plot_symbol_sharpe_ratio(self, symbol, symbol_data):
        # Plot Sharpe ratio
        output_path = os.path.join(self.output_dir, f'{symbol}_technical_sharpe_ratio.png')
        plt.figure(figsize=(14, 10))
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator())
        # Make a copy of symbol_data to avoid modifying a slice of the original DataFrame
        symbol_df = symbol_data.copy()
        # Convert the 'date' column to datetime to avoid plotting issues
        symbol_df['date'] = pd.to_datetime(symbol_df['date'], errors='coerce')

        # Calculate daily returns
        symbol_df['daily_return'] = symbol_df['adjusted_close'].pct_change()

        # Replace NaN values in daily returns
        #symbol_df['daily_return'].fillna(0, inplace=True)

        # Calculate excess return over the risk-free rate
        symbol_df['excess_return'] = symbol_df['daily_return'] - symbol_df['ffer']*.01/252.

        # Replace NaN values in excess return
        #symbol_df['excess_return'].fillna(0, inplace=True)

        # Calculate rolling Sharpe ratio (e.g., 5-day rolling window)
        rolling_mean = symbol_df['excess_return'].rolling(window=5).mean()
        rolling_std = symbol_df['daily_return'].rolling(window=5).std()

        # Replace 0 standard deviation with NaN to avoid division by zero
        #rolling_std.replace(0, np.nan, inplace=True)

        # Calculate the Sharpe ratio and replace NaN values
        symbol_df['sharpe_ratio'] = (rolling_mean / rolling_std)#.dropna()

        sharpe_ratio = symbol_df['sharpe_ratio']
        plt.plot(symbol_df['date'], symbol_df['sharpe_ratio'], label='Sharpe Ratio', color='black')

        # Determine colors for shading based on sharpe ratio values
        #colors = []
        #for value in sharpe_ratio:
        #    if value < -1.5:
        #        colors.append('darkred')
        #    elif -1.5 <= value < -0.5:
        #        colors.append('red')
        #    elif -0.5 <= value < 0.5:
        #        colors.append('orange')
        #    elif 0.5 <= value < 1.5:
        #        colors.append('lightgreen')
        #    else:
        #        colors.append('darkgreen')
        # Adding shaded regions for Sharpe ratios
        #plt.axhspan(ymin=-50, ymax=0, color='darkred', alpha=0.3)  # Dark Red for very bad
        plt.axhspan(ymin=0, ymax=1, color='lightcoral', alpha=0.3)  # Light Red for bad
        plt.axhspan(ymin=1, ymax=2, color='yellow', alpha=0.3)  # Yellow for neutral/good
        plt.axhspan(ymin=2, ymax=3, color='lightgreen', alpha=0.3)  # Light Green for very good
        plt.axhspan(ymin=3, ymax=np.inf, color='darkgreen', alpha=0.3)  # Dark Green for excellent

        #plt.bar(symbol_df['date'], sharpe_ratio, color=colors, alpha=0.7)
        plt.axhline(0, color='black', linestyle='--')
        # Format the x-axis to show dates correctly
        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.ylabel('Sharpe Ratio')
        plt.title(f'{symbol} - Daily Sharpe Ratio')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def _process_sector_data(self, symbol_data):
        # list of sector etfs
        sectors = ['XLB', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY', 'XLE', 'XRT', 'SPY', 'QQQ']
        # list of industry labels
        industries = [' SPDR FUND MATERIALS SELECT SECTR ETF',' SELECT STR FINANCIAL SELECT SPDR ETF',' SELECT SECTOR INDUSTRIAL SPDR ETF',
                      ' TECHNOLOGY SELECT SECTOR SPDR ETF',' SPDR FUND CONSUMER STAPLES ETF',' REAL ESTATE SELECT SCTR SPDR ETF',
                      ' SELECT SECTOR UTI SELECT SPDR ETF',' SELECT SECTOR HEALTH CARE SPDR ETF',' SPDR FUND CONSUMER DISCRE SELECT ETF',
                      ' ENERGY SELECT SECTOR SPDR ETF',' SPDR S&P RETAIL ETF', 'S&P 500 ETF TRUST ETF', 'NASDAQ QQQ TRUST ETF']
        # for each sector etf we need to manually calculate sma_20, sma_50, sma_200, bbands_upper_20, bbands_lower_20
        for industry, sector in zip(industries, sectors):

            output_path = os.path.join(self.output_dir, f'{sector}_sector_analysis.png')

            sector_data = self.data[self.data['symbol'] == sector]
            sector_data.loc[:, 'date'] = pd.to_datetime(sector_data['date'], format='%Y-%m-%d')
            sector_data = sector_data.sort_values('date')
            sector_data = sector_data[sector_data['date'] <= pd.to_datetime(self.latest_date)]

            sector_data['sma_20'] = sector_data['adjusted_close'].rolling(window=20).mean()
            sector_data['sma_50'] = sector_data['adjusted_close'].rolling(window=50).mean()
            sector_data['sma_200'] = sector_data['adjusted_close'].rolling(window=200).mean()
            # calculate bollinger bands
            sector_data['bbands_middle_20'] = sector_data['adjusted_close'].rolling(window=20).mean()
            sector_data['bbands_std_20'] = sector_data['adjusted_close'].rolling(window=20).std()
            sector_data['bbands_upper_20'] = sector_data['bbands_middle_20'] + (2 * sector_data['bbands_std_20'])
            sector_data['bbands_lower_20'] = sector_data['bbands_middle_20'] - (2 * sector_data['bbands_std_20'])
            # for each sector, plot the chart with sma_20, sma_50, sma_200, bbands_upper_20, bbands_lower_20
            #sector_data = symbol_data[symbol_data['symbol'] == sector]
            # only get the last n_days
            sector_data = sector_data[-self.n_days:]
            print('lenge of sector data:', len(sector_data))
            # plot a chart with adjusted_close, sma_20, sma_50, sma_200, bbands_upper_20, bbands_lower_20
            # add text annotation when there are bullish or bearish signals
            plt.figure(figsize=(14, 10))
            plt.plot(sector_data['date'], sector_data['adjusted_close'], label='Adjusted Close', color='black', linewidth=2)
            plt.plot(sector_data['date'], sector_data['sma_20'], label='SMA 20', color='orange')
            plt.plot(sector_data['date'], sector_data['sma_50'], label='SMA 50', color='green')
            plt.plot(sector_data['date'], sector_data['sma_200'], label='SMA 200', color='red')
            plt.plot(sector_data['date'], sector_data['bbands_upper_20'], linestyle='--', label='Bollinger Band Upper', color='blue')
            plt.plot(sector_data['date'], sector_data['bbands_lower_20'], linestyle='--', label='Bollinger Band Lower', color='blue')

            # Add annotations for SMA crossovers and bollinger band breakouts
            for i in range(1, len(sector_data)):
                if sector_data['sma_20'].iloc[i] > sector_data['sma_50'].iloc[i] and sector_data['sma_20'].iloc[i - 1] <= sector_data['sma_50'].iloc[i - 1]:
                    plt.annotate('Bullish Cross 20/50', (sector_data['date'].iloc[i-1], sector_data['sma_20'].iloc[i-1]),
                                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='green', arrowprops=dict(arrowstyle='->', color='green'))
                elif sector_data['sma_20'].iloc[i] < sector_data['sma_50'].iloc[i] and sector_data['sma_20'].iloc[i - 1] >= sector_data['sma_50'].iloc[i - 1]:
                    plt.annotate('Bearish Cross 20/50', (sector_data['date'].iloc[i], sector_data['sma_20'].iloc[i]),
                                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red', arrowprops=dict(arrowstyle='->', color='red'))
                if sector_data['sma_20'].iloc[i] > sector_data['sma_200'].iloc[i] and sector_data['sma_20'].iloc[i - 1] <= sector_data['sma_200'].iloc[i - 1]:
                    plt.annotate('Bullish Cross 20/200', (sector_data['date'].iloc[i-1], sector_data['sma_200'].iloc[i-1]),
                                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='green', arrowprops=dict(arrowstyle='->', color='green'))
                elif sector_data['sma_20'].iloc[i] < sector_data['sma_200'].iloc[i] and sector_data['sma_20'].iloc[i - 1] >= sector_data['sma_200'].iloc[i - 1]:
                    plt.annotate('Bearish Cross 20/200', (sector_data['date'].iloc[i-1], sector_data['sma_200'].iloc[i-1]),
                                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red', arrowprops=dict(arrowstyle='->', color='red'))
                if sector_data['adjusted_close'].iloc[i] > sector_data['bbands_upper_20'].iloc[i]:
                    plt.annotate('Bearish Breakout', (sector_data['date'].iloc[i], sector_data['adjusted_close'].iloc[i]), textcoords="offset points", xytext=(0,10), ha='center', color='red')
                elif sector_data['adjusted_close'].iloc[i] < sector_data['bbands_lower_20'].iloc[i]:
                    plt.annotate('Bullish Breakout', (sector_data['date'].iloc[i], sector_data['adjusted_close'].iloc[i]), textcoords="offset points", xytext=(0,10), ha='center', color='green')
                # add Buy Watch and Sell Watch annotations for bolinger bands
                if abs(sector_data['adjusted_close'].iloc[i] - sector_data['bbands_upper_20'].iloc[i]) / sector_data['bbands_upper_20'].iloc[i] <= 0.01:
                    # ensure this is from below not above
                    if sector_data['adjusted_close'].iloc[i-1] < sector_data['bbands_upper_20'].iloc[i-1]:
                        plt.annotate('Sell Watch', (sector_data['date'].iloc[i], sector_data['adjusted_close'].iloc[i]), textcoords="offset points", xytext=(0,10), ha='center', color='orange')
                elif abs(sector_data['adjusted_close'].iloc[i] - sector_data['bbands_lower_20'].iloc[i]) / sector_data['bbands_lower_20'].iloc[i] <= 0.01:
                    # ensure this is from above not below
                    if sector_data['adjusted_close'].iloc[i-1] > sector_data['bbands_lower_20'].iloc[i-1]:
                        plt.annotate('Buy Watch', (sector_data['date'].iloc[i], sector_data['adjusted_close'].iloc[i]), textcoords="offset points", xytext=(0,10), ha='center', color='blue')


            # find min and max of any data used in the plot for y axis
            min_val = min(sector_data['adjusted_close'].min(), sector_data['sma_20'].min(), sector_data['sma_50'].min(), sector_data['sma_200'].min(), sector_data['bbands_upper_20'].min(), sector_data['bbands_lower_20'].min())
            max_val = max(sector_data['adjusted_close'].max(), sector_data['sma_20'].max(), sector_data['sma_50'].max(), sector_data['sma_200'].max(), sector_data['bbands_upper_20'].max(), sector_data['bbands_lower_20'].max())
            plt.ylim(min_val * 0.95, max_val * 1.05)  # Adjust y-limits for better visibility
            plt.xlabel('Date')
            plt.xticks(rotation=45)
            plt.ylabel('Price')
            plt.title(f'{sector} {industry}- Price and SMAs')
            plt.legend()
            plt.tight_layout()
            plt.ylim(min(sector_data['adjusted_close']) * 0.95, max(sector_data['adjusted_close']) * 1.05)  # Adjust y-limits for better visibility
            plt.savefig(output_path)
            plt.close()
        # for all sectors and SPY, QQQ, plot the n_day rolling % change
        # plot the daily % change for all sectors and SPY, QQQ on one chart -> use dotted lines for sectors, bold lines for SPY/QQQ
        output_path = os.path.join(self.output_dir, 'market_returns.png')
        plt.figure(figsize=(14, 10))
        # need more than 10 colors from plt color cycle
        colors = plt.cm.tab20(np.linspace(0, 1, len(sectors) + 2))
        for idx, sector in enumerate(sectors):

            sector_data = self.data[self.data['symbol'] == sector]
            sector_data.loc[:, 'date'] = pd.to_datetime(sector_data['date'], format='%Y-%m-%d')
            sector_data = sector_data.sort_values('date')
            sector_data = sector_data[sector_data['date'] <= pd.to_datetime(self.latest_date)]
            # only get the last n_days
            sector_data = sector_data[-self.n_days:]
            # calcluate the rolling cumulative % change since the beginning of the n_days (-n_days is the base value)
            initial_value = sector_data['adjusted_close'].iloc[0]
            if initial_value != 0:  # Avoid division by zero
                sector_data['cumulative_change'] = (sector_data['adjusted_close'] / initial_value - 1) * 100

            # plot with a different color, if SPY or QQQ use a bold line, otherwise dash
            if sector in ['SPY', 'QQQ']:
                plt.plot(sector_data['date'], sector_data['cumulative_change'], label=sector, color='black', linewidth=2)
                # add a plot for SPY and QQQ averages over this time period
                plt.axhline(sector_data['cumulative_change'].mean(), color='black', linestyle='--', label=f'{sector} Average')
            else:
                plt.plot(sector_data['date'], sector_data['cumulative_change'], label=industries[idx], color=colors[idx], linestyle='--')

        plt.xlabel('Date')
        plt.xticks(rotation=45)
        plt.ylabel('Rolling % Change')
        plt.title('Sector ETFs and Major Indices - Cumulative % Change')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        



    def _write_results(self):
        if not self.results:
            logging.warning('No results to write.')
            return

        rows_to_write = []
        for result in self.results:
            symbol, num_bullish, num_bearish, *signals = result
            if any(signals):  # Only write rows where there is at least one signal (1 or -1)
                rows_to_write.append([symbol, num_bullish, num_bearish, *signals])

        if not rows_to_write:
            logging.warning('No significant results to write.')
            return

        # Use a fixed number of columns since the indicators are known
        columns = ['symbol', 'num_bullish', 'num_bearish', 'macd', 'adx', 'atr', 'pe_ratio', 'bollinger_band', 'rsi', 'sma_cross']
        results_df = pd.DataFrame(rows_to_write, columns=columns)
        results_df.to_csv('screener_results_{}.csv'.format(self.date), index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='+', required=False, help='List of stock symbols to check')
    parser.add_argument('--watchlist', type=str, required=False, default='stocklist.csv', help='Watchlist of stock symbols to check')
    parser.add_argument('--data', type=str, default='../api_data/all_data.csv', help='Path to the CSV data file (default: ../api_data/all_data.csv)')
    parser.add_argument('--date', type=str, default=datetime.now().strftime('%Y-%m-%d'), help="Date to check signals for (default is today's date)")
    parser.add_argument('--indicators', type=str, nargs='+', default='all', help='List of indicators to check (default is all)')
    parser.add_argument('--visualize', action='store_true', default=True, help='Flag to visualize data (default is true)')
    parser.add_argument('--n_days', type=int, default=30, help='Number of past days to visualize (default is 30)')
    parser.add_argument('--use_candlesticks', action='store_true', default=False, help='Use candlestick charts instead of line plots (default is false)')

    args = parser.parse_args()

    try:
        if args.symbols:
            SYMBOLS = args.symbols
        else:
            # read in csv file of watch list
            SYMBOLS = pd.read_csv(args.watchlist)['Symbol'].tolist()
            # ticker symbol is first token after comma separation
            SYMBOLS = [symbol.split(',')[0] for symbol in SYMBOLS]
            SYMBOLS = [symbol.upper() for symbol in SYMBOLS]
        logger.info(f"Symbols: {SYMBOLS}")
    except Exception as e:
        print(f"Error: {e}")

    screener = StockScreener(
        symbols=SYMBOLS,
        date=args.date,
        indicators=args.indicators,
        visualize=args.visualize,
        n_days=args.n_days,
        use_candlesticks=args.use_candlesticks,
        data=args.data)
    screener.run_screen()
