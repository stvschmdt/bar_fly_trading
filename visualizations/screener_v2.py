"""
Screener V2 — Improved overnight visualization suite.

Standalone replacement for screener.py with:
  - Cleaner master technicals (markers instead of text, BB fill)
  - Auto-scaled % off highs (no -100% floor)
  - Green/red volume bars (up/down days)
  - Improved sector cumulative returns
  - Consistent color palette

Usage (from project root):
  python visualizations/screener_v2.py --data api_data/ --date 2026-02-05
  python visualizations/screener_v2.py --data api_data/ --symbols AAPL NVDA --n_days 60
"""

import argparse
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime, timedelta
from util import get_closest_trading_date

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# ── Color palette ──────────────────────────────────────────────────
COLORS = {
    'price': '#1a1a2e',
    'sma20': '#e6a817',
    'sma50': '#e07c24',
    'sma200': '#c0392b',
    'bb_fill': '#3498db',
    'bb_line': '#2980b9',
    'fib': '#8e44ad',
    'vol_up': '#27ae60',
    'vol_down': '#e74c3c',
    'vol_avg': '#7f8c8d',
    'bull_marker': '#27ae60',
    'bear_marker': '#e74c3c',
    'neutral': '#95a5a6',
    'grid': '#ecf0f1',
    'bg': '#fafafa',
    'watch': '#f39c12',
    'take_profit': '#e74c3c',
    'stop_loss': '#2ecc71',
    '52wk_high': '#e74c3c',
    '52wk_low': '#27ae60',
    'off_sma20': '#f1c40f',
    'off_sma50': '#e67e22',
    'off_sma200': '#e84393',
}

SECTORS = ['XLB', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY', 'XLE', 'XRT', 'SPY', 'QQQ']
SECTOR_NAMES = {
    'XLB': 'Materials', 'XLF': 'Financials', 'XLI': 'Industrials',
    'XLK': 'Technology', 'XLP': 'Staples', 'XLRE': 'Real Estate',
    'XLU': 'Utilities', 'XLV': 'Healthcare', 'XLY': 'Discretionary',
    'XLE': 'Energy', 'XRT': 'Retail', 'SPY': 'S&P 500', 'QQQ': 'Nasdaq 100',
}


def _style_ax(ax):
    """Apply consistent styling to an axis."""
    ax.set_facecolor(COLORS['bg'])
    ax.grid(True, alpha=0.3, color=COLORS['grid'], linestyle='-', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)


def _format_dates(ax):
    """Format x-axis dates cleanly."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)


class StockScreenerV2:
    def __init__(self, symbols, date, data, indicators='all', visualize=True,
                 n_days=30, whitelist=None, skip_sectors=False):
        self.symbols = [s.upper() for s in symbols]
        self.date = pd.to_datetime(get_closest_trading_date(date)).strftime('%Y-%m-%d')
        self.data = data
        self.indicators = indicators
        self.visualize = visualize
        self.n_days = n_days
        self.whitelist = whitelist or []
        self.skip_sectors = skip_sectors
        self.results = []

    def find_nearest_three_dates(self, target_date):
        target_date = pd.to_datetime(target_date)
        available_dates = pd.to_datetime(self.data['date']).drop_duplicates().sort_values()
        available_dates = available_dates[available_dates <= target_date]
        nearest_dates = available_dates[-3:].sort_values(ascending=False)
        return nearest_dates.tolist()

    def run_screen(self):
        nearest_dates = self.find_nearest_three_dates(self.date)
        if len(nearest_dates) < 3:
            logger.error('Not enough dates in data for screening.')
            return
        self.latest_date = nearest_dates[0].strftime('%Y-%m-%d')
        previous_date = nearest_dates[1].strftime('%Y-%m-%d')
        day_before = nearest_dates[2].strftime('%Y-%m-%d')

        output_dir = f'overnight_v2_{self.latest_date}'
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        logger.info(f'V2 screener running for {len(self.symbols)} symbols on {self.latest_date}')

        for symbol in self.symbols:
            symbol_data = self.data[self.data['symbol'] == symbol].copy()
            symbol_data['date'] = pd.to_datetime(symbol_data['date'])
            symbol_data = symbol_data.sort_values('date')
            symbol_data = symbol_data[symbol_data['date'] <= pd.to_datetime(self.latest_date)]

            latest = symbol_data[symbol_data['date'] == pd.to_datetime(self.latest_date)]
            previous = symbol_data[symbol_data['date'] == pd.to_datetime(previous_date)]
            day_before_data = symbol_data[symbol_data['date'] == pd.to_datetime(day_before)]

            if latest.empty or previous.empty:
                logger.warning(f'No data for {symbol} on {self.latest_date}')
                continue

            latest_bull, latest_bear, latest_signals = self._check_signals(previous, latest)
            prev_bull, prev_bear, prev_signals = self._check_signals(day_before_data, previous)

            change_signals = [latest_signals[i] if latest_signals[i] != prev_signals[i] else 0
                              for i in range(len(latest_signals))]

            if len([s for s in change_signals if s != 0]) > 1 or symbol in self.whitelist:
                if abs(len(latest_bull) - len(latest_bear)) > 1 or len([s for s in change_signals if s != 0]) > 1:
                    self.results.append([symbol, len(latest_bull), len(latest_bear), *latest_signals])
                    if self.visualize:
                        self._visualize(symbol, symbol_data, latest_bull, latest_bear)

        try:
            if not self.skip_sectors:
                self._process_sector_data()
        except Exception as e:
            logger.error(f'Error processing sector data: {e}')

        self._write_results()

    # ── Signal checks (identical logic to v1 for compatibility) ──────

    def _check_signals(self, symbol_data, selected_date_data):
        bullish, bearish, signals = [], [], []
        if self.indicators == 'all' or 'sma_cross' in self.indicators:
            self._check_sma_cross(selected_date_data, bullish, bearish, signals)
        if self.indicators == 'all' or 'bollinger_band' in self.indicators:
            self._check_bollinger_band(selected_date_data, bullish, bearish, signals)
        if self.indicators == 'all' or 'rsi' in self.indicators:
            self._check_rsi(selected_date_data, bullish, bearish, signals)
        if self.indicators == 'all' or 'macd' in self.indicators:
            self._check_macd(selected_date_data, bullish, bearish, signals)
        if self.indicators == 'all' or 'macd_zero' in self.indicators:
            self._check_macd_zero(selected_date_data, bullish, bearish, signals)
        if self.indicators == 'all' or 'adx' in self.indicators:
            self._check_adx(selected_date_data, bullish, bearish, signals)
        if self.indicators == 'all' or 'cci' in self.indicators:
            self._check_cci(selected_date_data, bullish, bearish, signals)
        if self.indicators == 'all' or 'atr' in self.indicators:
            self._check_atr(selected_date_data, bullish, bearish, signals)
        if self.indicators == 'all' or 'pe_ratio' in self.indicators:
            self._check_pe_ratio(selected_date_data, bullish, bearish, signals)
        if self.indicators == 'all' or 'pcr' in self.indicators:
            self._check_pcr(selected_date_data, bullish, bearish, signals)
        return bullish, bearish, signals

    def _check_sma_cross(self, selected_date_data, bullish_signals, bearish_signals, signals):
        s20, s50 = selected_date_data['sma_20'].values[0], selected_date_data['sma_50'].values[0]
        if s20 > s50: bullish_signals.append('bullish_sma_cross'); signals.append(1)
        elif s20 < s50: bearish_signals.append('bearish_sma_cross'); signals.append(-1)
        else: signals.append(0)

    def _check_bollinger_band(self, selected_date_data, bullish_signals, bearish_signals, signals):
        c = selected_date_data['adjusted_close'].values[0]
        u, l = selected_date_data['bbands_upper_20'].values[0], selected_date_data['bbands_lower_20'].values[0]
        if c > u: bearish_signals.append('bearish_bollinger_band'); signals.append(-1)
        elif c < l: bullish_signals.append('bullish_bollinger_band'); signals.append(1)
        else: signals.append(0)

    def _check_rsi(self, selected_date_data, bullish_signals, bearish_signals, signals):
        r = selected_date_data['rsi_14'].values[0]
        if r > 70: bearish_signals.append('bearish_rsi'); signals.append(-1)
        elif r < 30: bullish_signals.append('bullish_rsi'); signals.append(1)
        else: signals.append(0)

    def _check_macd(self, selected_date_data, bullish_signals, bearish_signals, signals):
        m, e = selected_date_data['macd'].values[0], selected_date_data['macd_9_ema'].values[0]
        if m > e: bullish_signals.append('bullish_macd'); signals.append(1)
        elif m < e: bearish_signals.append('bearish_macd'); signals.append(-1)
        else: signals.append(0)

    def _check_macd_zero(self, selected_date_data, bullish_signals, bearish_signals, signals):
        m = selected_date_data['macd'].values[0]
        if m > 0: bullish_signals.append('bullish_macd_zero'); signals.append(1)
        elif m < 0: bearish_signals.append('bearish_macd_zero'); signals.append(-1)
        else: signals.append(0)

    def _check_adx(self, selected_date_data, bullish_signals, bearish_signals, signals):
        a = selected_date_data['adx_14'].values[0]
        if a > 25: bullish_signals.append('bullish_adx'); signals.append(1)
        elif a < 20: bearish_signals.append('bearish_adx'); signals.append(-1)
        else: signals.append(0)

    def _check_cci(self, selected_date_data, bullish_signals, bearish_signals, signals):
        c = selected_date_data['cci_14'].values[0]
        if c >= 100: bearish_signals.append('bearish_cci'); signals.append(-1)
        elif c <= -100: bullish_signals.append('bullish_cci'); signals.append(1)
        else: signals.append(0)

    def _check_atr(self, selected_date_data, bullish_signals, bearish_signals, signals):
        a, c = selected_date_data['atr_14'].values[0], selected_date_data['adjusted_close'].values[0]
        if c > a * 2: bearish_signals.append('bearish_atr'); signals.append(-1)
        elif c < a * 2: bullish_signals.append('bullish_atr'); signals.append(1)
        else: signals.append(0)

    def _check_pe_ratio(self, selected_date_data, bullish_signals, bearish_signals, signals):
        p = selected_date_data['pe_ratio'].values[0]
        if p < 15 and p > 0: bullish_signals.append('bullish_pe_ratio'); signals.append(1)
        elif p > 35: bearish_signals.append('bearish_pe_ratio'); signals.append(-1)
        else: signals.append(0)

    def _check_pcr(self, selected_date_data, bullish_signals, bearish_signals, signals):
        try:
            p = selected_date_data['pcr'].values[0]
            if pd.isna(p): signals.append(0); return
            if p > .7: bearish_signals.append('bearish_pcr'); signals.append(-1)
            elif p <= .5: bullish_signals.append('bullish_pcr'); signals.append(1)
            else: signals.append(0)
        except (KeyError, IndexError):
            signals.append(0)

    # ── Improved Visualization ───────────────────────────────────────

    def _visualize(self, symbol, symbol_data, bullish, bearish):
        symbol_data = symbol_data.tail(self.n_days).copy()
        if abs(len(bullish) - len(bearish)) > 1:
            logger.info(f'V2 visualizing: {symbol}')
            self._plot_master_technicals(symbol, symbol_data)
            self._plot_off_from_highs(symbol, symbol_data)
            self._plot_volume(symbol, symbol_data)
            if symbol not in SECTORS:
                self._plot_pe_ratio(symbol, symbol_data)
            self._plot_rsi(symbol, symbol_data)
            self._plot_cci(symbol, symbol_data)
            self._plot_macd(symbol, symbol_data)

    def _plot_master_technicals(self, symbol, df, title=None, output_path=None):
        """Master chart: price + SMAs + BB fill + signal markers (no text clutter)."""
        if title is None:
            title = symbol
        if output_path is None:
            output_path = os.path.join(self.output_dir, f'{symbol}_daily_price.jpg')

        fig, ax = plt.subplots(figsize=(14, 8))
        _style_ax(ax)
        dates = df['date']

        # Bollinger Band fill
        ax.fill_between(dates, df['bbands_upper_20'], df['bbands_lower_20'],
                         alpha=0.12, color=COLORS['bb_fill'], label='BB 20')
        ax.plot(dates, df['bbands_upper_20'], color=COLORS['bb_line'], linewidth=0.7, alpha=0.5)
        ax.plot(dates, df['bbands_lower_20'], color=COLORS['bb_line'], linewidth=0.7, alpha=0.5)

        # SMAs
        ax.plot(dates, df['sma_200'], color=COLORS['sma200'], linewidth=1.2,
                label='SMA 200', alpha=0.7)
        ax.plot(dates, df['sma_50'], color=COLORS['sma50'], linewidth=1.2,
                label='SMA 50', alpha=0.8)
        ax.plot(dates, df['sma_20'], color=COLORS['sma20'], linewidth=1.0,
                label='SMA 20', alpha=0.8, linestyle='--')

        # Price line (on top)
        ax.plot(dates, df['adjusted_close'], color=COLORS['price'], linewidth=1.8,
                label='Close', zorder=5)

        # ATR take-profit / stop-loss
        last_close = df['adjusted_close'].iloc[-1]
        last_atr = df['atr_14'].iloc[-1]
        tp = last_close + 2 * last_atr
        sl = last_close - 2 * last_atr
        ax.axhline(tp, color=COLORS['take_profit'], linewidth=0.8, linestyle=':', alpha=0.6, label=f'TP ({tp:.1f})')
        ax.axhline(sl, color=COLORS['stop_loss'], linewidth=0.8, linestyle=':', alpha=0.6, label=f'SL ({sl:.1f})')

        # Fibonacci levels (subtle)
        max_p, min_p = df['adjusted_close'].max(), df['adjusted_close'].min()
        diff = max_p - min_p
        for pct, label in [(0.236, '23.6%'), (0.382, '38.2%'), (0.618, '61.8%')]:
            level = max_p - diff * pct
            ax.axhline(level, color=COLORS['fib'], linewidth=0.6, linestyle='--', alpha=0.3)
            ax.text(dates.iloc[0], level, f' Fib {label}', fontsize=7, color=COLORS['fib'],
                    alpha=0.6, va='bottom')

        # Signal markers: SMA crossovers
        for i in range(1, len(df)):
            s20, s50 = df['sma_20'].iloc[i], df['sma_50'].iloc[i]
            ps20, ps50 = df['sma_20'].iloc[i-1], df['sma_50'].iloc[i-1]
            if s20 > s50 and ps20 <= ps50:
                ax.scatter(dates.iloc[i], df['adjusted_close'].iloc[i],
                           marker='^', color=COLORS['bull_marker'], s=80, zorder=10, edgecolors='white', linewidth=0.5)
            elif s20 < s50 and ps20 >= ps50:
                ax.scatter(dates.iloc[i], df['adjusted_close'].iloc[i],
                           marker='v', color=COLORS['bear_marker'], s=80, zorder=10, edgecolors='white', linewidth=0.5)

        # BB buy/sell/watch arrows
        for i in range(len(df)):
            c = df['adjusted_close'].iloc[i]
            bb_u = df['bbands_upper_20'].iloc[i]
            bb_l = df['bbands_lower_20'].iloc[i]
            bb_range = bb_u - bb_l if bb_u != bb_l else 1

            if c > bb_u:
                # SELL — price above upper band
                ax.annotate('', xy=(dates.iloc[i], c),
                            xytext=(dates.iloc[i], c + bb_range * 0.15),
                            arrowprops=dict(arrowstyle='->', color=COLORS['bear_marker'], lw=2.0),
                            zorder=12)
            elif c < bb_l:
                # BUY — price below lower band
                ax.annotate('', xy=(dates.iloc[i], c),
                            xytext=(dates.iloc[i], c - bb_range * 0.15),
                            arrowprops=dict(arrowstyle='->', color=COLORS['bull_marker'], lw=2.0),
                            zorder=12)
            elif bb_u > 0 and abs(c - bb_u) / bb_u <= 0.01:
                # WATCH — within 1% of upper band
                ax.annotate('', xy=(dates.iloc[i], c),
                            xytext=(dates.iloc[i], c + bb_range * 0.12),
                            arrowprops=dict(arrowstyle='->', color=COLORS['watch'], lw=1.5, linestyle='--'),
                            zorder=11)
            elif bb_l > 0 and abs(c - bb_l) / bb_l <= 0.01:
                # WATCH — within 1% of lower band
                ax.annotate('', xy=(dates.iloc[i], c),
                            xytext=(dates.iloc[i], c - bb_range * 0.12),
                            arrowprops=dict(arrowstyle='->', color=COLORS['watch'], lw=1.5, linestyle='--'),
                            zorder=11)

        # ATR watch arrows — yellow when approaching TP/SL
        for i in range(len(df)):
            c = df['adjusted_close'].iloc[i]
            if tp > 0 and abs(c - tp) / tp <= 0.01:
                ax.annotate('', xy=(dates.iloc[i], c),
                            xytext=(dates.iloc[i], c + (tp - sl) * 0.08),
                            arrowprops=dict(arrowstyle='->', color=COLORS['watch'], lw=1.5, linestyle='--'),
                            zorder=11)
            elif sl > 0 and abs(c - sl) / sl <= 0.01:
                ax.annotate('', xy=(dates.iloc[i], c),
                            xytext=(dates.iloc[i], c - (tp - sl) * 0.08),
                            arrowprops=dict(arrowstyle='->', color=COLORS['watch'], lw=1.5, linestyle='--'),
                            zorder=11)

        # Latest price annotation
        ax.annotate(f'${last_close:.2f}', xy=(dates.iloc[-1], last_close),
                    xytext=(10, 0), textcoords='offset points', fontsize=9,
                    fontweight='bold', color=COLORS['price'],
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=COLORS['price'], alpha=0.8))

        # Invisible scatter points for legend entries
        ax.scatter([], [], marker='^', color=COLORS['bull_marker'], s=60, label='Buy signal')
        ax.scatter([], [], marker='v', color=COLORS['bear_marker'], s=60, label='Sell signal')
        ax.scatter([], [], marker='d', color=COLORS['watch'], s=60, label='Watch')

        _format_dates(ax)
        ax.set_ylabel('Price ($)', fontsize=10)
        ax.set_title(f'{title} — Master Technicals', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='upper left', fontsize=7, ncol=4, framealpha=0.8)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_off_from_highs(self, symbol, df):
        """Auto-scaled % off highs — no -100% floor."""
        output_path = os.path.join(self.output_dir, f'{symbol}_technical_off_from_highs.jpg')
        fig, ax = plt.subplots(figsize=(14, 6))
        _style_ax(ax)
        dates = df['date']

        ax.plot(dates, df['52_week_high_pct'], color=COLORS['52wk_high'],
                linewidth=1.5, label='% off 52wk High')
        ax.plot(dates, df['52_week_low_pct'], color=COLORS['52wk_low'],
                linewidth=1.5, label='% above 52wk Low')
        ax.plot(dates, df['sma_20_pct'], color=COLORS['off_sma20'],
                linewidth=0.9, linestyle='--', alpha=0.7, label='% off SMA 20')
        ax.plot(dates, df['sma_50_pct'], color=COLORS['off_sma50'],
                linewidth=0.9, linestyle='--', alpha=0.7, label='% off SMA 50')
        ax.plot(dates, df['sma_200_pct'], color=COLORS['off_sma200'],
                linewidth=0.9, linestyle='--', alpha=0.7, label='% off SMA 200')

        ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.4)

        # Fill between 0 line and 52wk high pct
        ax.fill_between(dates, 0, df['52_week_high_pct'],
                         where=df['52_week_high_pct'] < 0,
                         alpha=0.1, color=COLORS['52wk_high'])

        # Auto-scale y-axis with 10% padding
        all_vals = pd.concat([df['52_week_high_pct'], df['52_week_low_pct'],
                              df['sma_20_pct'], df['sma_50_pct'], df['sma_200_pct']])
        y_min = all_vals.min() * 1.1 if all_vals.min() < 0 else all_vals.min() * 0.9
        y_max = all_vals.max() * 1.1 if all_vals.max() > 0 else all_vals.max() * 0.9
        ax.set_ylim(y_min, y_max)

        # Annotate latest values
        last_high = df['52_week_high_pct'].iloc[-1]
        last_low = df['52_week_low_pct'].iloc[-1]
        ax.annotate(f'{last_high:+.1f}% from high', xy=(dates.iloc[-1], last_high),
                    xytext=(10, 0), textcoords='offset points', fontsize=8,
                    color=COLORS['52wk_high'], fontweight='bold')
        ax.annotate(f'+{last_low:.1f}% from low', xy=(dates.iloc[-1], last_low),
                    xytext=(10, 0), textcoords='offset points', fontsize=8,
                    color=COLORS['52wk_low'], fontweight='bold')

        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%+.0f%%'))
        _format_dates(ax)
        ax.set_ylabel('Percentage', fontsize=10)
        ax.set_title(f'{symbol} — % Off Highs/Lows', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=7, framealpha=0.8)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_volume(self, symbol, df):
        """Green/red bars for up/down days with average line."""
        output_path = os.path.join(self.output_dir, f'{symbol}_daily_volume.jpg')
        fig, ax = plt.subplots(figsize=(14, 5))
        _style_ax(ax)
        dates = df['date']

        # Color by price direction
        colors = [COLORS['vol_up'] if df['adjusted_close'].iloc[i] >= df['open'].iloc[i]
                  else COLORS['vol_down'] for i in range(len(df))]

        ax.bar(dates, df['volume'], color=colors, alpha=0.7, width=0.8)

        # Average volume line
        avg_vol = df['volume'].mean()
        ax.axhline(avg_vol, color=COLORS['vol_avg'], linewidth=1.2, linestyle='--',
                    label=f'Avg ({avg_vol/1e6:.1f}M)')

        # 1.5x spike threshold
        spike = avg_vol * 1.5
        spike_days = df[df['volume'] > spike]
        if not spike_days.empty:
            ax.scatter(spike_days['date'], spike_days['volume'],
                       marker='*', color='#e74c3c', s=60, zorder=10,
                       label=f'Spike (>{spike/1e6:.1f}M)')

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        _format_dates(ax)
        ax.set_ylabel('Volume', fontsize=10)
        ax.set_title(f'{symbol} — Daily Volume', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='upper left', fontsize=8, framealpha=0.8)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_pe_ratio(self, symbol, df):
        """PE ratio with zone coloring."""
        output_path = os.path.join(self.output_dir, f'{symbol}_technical_ttm_pe_ratio.jpg')
        fig, ax = plt.subplots(figsize=(14, 5))
        _style_ax(ax)
        dates = df['date']
        pe = df['pe_ratio']

        ax.plot(dates, pe, color=COLORS['price'], linewidth=1.5, label='TTM P/E')

        # Zone fills
        ax.axhspan(0, 15, alpha=0.08, color=COLORS['bull_marker'])
        ax.axhspan(35, pe.max() * 1.2 if pe.max() > 35 else 50, alpha=0.08, color=COLORS['bear_marker'])
        ax.axhline(15, color=COLORS['bull_marker'], linewidth=0.8, linestyle='--', alpha=0.5, label='Value (<15)')
        ax.axhline(35, color=COLORS['bear_marker'], linewidth=0.8, linestyle='--', alpha=0.5, label='Expensive (>35)')

        # Forward PE
        if 'forward_pe' in df.columns:
            fwd = df['forward_pe'].iloc[-1]
            if pd.notna(fwd) and fwd > 0:
                ax.axhline(fwd, color=COLORS['bb_fill'], linewidth=1.0, linestyle=':',
                            label=f'Forward PE ({fwd:.1f})')

        _format_dates(ax)
        ax.set_ylabel('P/E Ratio', fontsize=10)
        ax.set_title(f'{symbol} — TTM P/E Ratio', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=8, framealpha=0.8)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_rsi(self, symbol, df):
        """RSI with overbought/oversold fill zones."""
        output_path = os.path.join(self.output_dir, f'{symbol}_technical_rsi.jpg')
        fig, ax = plt.subplots(figsize=(14, 4))
        _style_ax(ax)
        dates = df['date']

        ax.plot(dates, df['rsi_14'], color=COLORS['price'], linewidth=1.5, label='RSI 14')
        ax.axhline(70, color=COLORS['bear_marker'], linewidth=0.8, linestyle='--', alpha=0.5)
        ax.axhline(30, color=COLORS['bull_marker'], linewidth=0.8, linestyle='--', alpha=0.5)
        ax.axhline(50, color=COLORS['neutral'], linewidth=0.5, linestyle='-', alpha=0.3)
        ax.axhspan(70, 100, alpha=0.08, color=COLORS['bear_marker'])
        ax.axhspan(0, 30, alpha=0.08, color=COLORS['bull_marker'])
        ax.set_ylim(0, 100)

        # Buy/sell/watch arrows on RSI
        for i in range(1, len(df)):
            r = df['rsi_14'].iloc[i]
            rp = df['rsi_14'].iloc[i-1]
            if r > 70 and rp <= 70:
                # SELL — crossed into overbought
                ax.annotate('', xy=(dates.iloc[i], r),
                            xytext=(dates.iloc[i], r + 6),
                            arrowprops=dict(arrowstyle='->', color=COLORS['bear_marker'], lw=2.0),
                            zorder=12)
            elif r < 30 and rp >= 30:
                # BUY — crossed into oversold
                ax.annotate('', xy=(dates.iloc[i], r),
                            xytext=(dates.iloc[i], r - 6),
                            arrowprops=dict(arrowstyle='->', color=COLORS['bull_marker'], lw=2.0),
                            zorder=12)
            elif 65 <= r <= 70:
                # WATCH — approaching overbought
                ax.annotate('', xy=(dates.iloc[i], r),
                            xytext=(dates.iloc[i], r + 5),
                            arrowprops=dict(arrowstyle='->', color=COLORS['watch'], lw=1.5, linestyle='--'),
                            zorder=11)
            elif 30 <= r <= 35:
                # WATCH — approaching oversold
                ax.annotate('', xy=(dates.iloc[i], r),
                            xytext=(dates.iloc[i], r - 5),
                            arrowprops=dict(arrowstyle='->', color=COLORS['watch'], lw=1.5, linestyle='--'),
                            zorder=11)

        ax.scatter([], [], marker='^', color=COLORS['bull_marker'], s=40, label='Buy')
        ax.scatter([], [], marker='v', color=COLORS['bear_marker'], s=40, label='Sell')
        ax.scatter([], [], marker='d', color=COLORS['watch'], s=40, label='Watch')
        ax.legend(loc='upper right', fontsize=7, framealpha=0.8)

        _format_dates(ax)
        ax.set_ylabel('RSI', fontsize=10)
        ax.set_title(f'{symbol} — RSI', fontsize=11, fontweight='bold', pad=10)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_cci(self, symbol, df):
        """CCI with overbought/oversold zones and zero-cross arrows."""
        output_path = os.path.join(self.output_dir, f'{symbol}_technical_cci.jpg')
        fig, ax = plt.subplots(figsize=(14, 5))
        _style_ax(ax)
        dates = df['date']
        cci = df['cci_14']

        ax.plot(dates, cci, color=COLORS['price'], linewidth=1.5, label='CCI 14')
        ax.axhline(100, color=COLORS['bear_marker'], linewidth=0.8, linestyle='--', alpha=0.5)
        ax.axhline(-100, color=COLORS['bull_marker'], linewidth=0.8, linestyle='--', alpha=0.5)
        ax.axhline(0, color=COLORS['neutral'], linewidth=0.5, linestyle='-', alpha=0.3)

        # Zone fills
        ax.axhspan(100, max(cci.max() * 1.2, 200), alpha=0.08, color=COLORS['bear_marker'])
        ax.axhspan(min(cci.min() * 1.2, -200), -100, alpha=0.08, color=COLORS['bull_marker'])

        cci_range = max(abs(cci.max()), abs(cci.min()), 1)

        for i in range(1, len(df)):
            c = cci.iloc[i]
            cp = cci.iloc[i-1]

            if c >= 100 and cp < 100:
                # SELL — crossed into overbought
                ax.annotate('', xy=(dates.iloc[i], c),
                            xytext=(dates.iloc[i], c + cci_range * 0.1),
                            arrowprops=dict(arrowstyle='->', color=COLORS['bear_marker'], lw=2.0),
                            zorder=12)
            elif c <= -100 and cp > -100:
                # BUY — crossed into oversold
                ax.annotate('', xy=(dates.iloc[i], c),
                            xytext=(dates.iloc[i], c - cci_range * 0.1),
                            arrowprops=dict(arrowstyle='->', color=COLORS['bull_marker'], lw=2.0),
                            zorder=12)

            # Zero-line crosses (bold — these are important)
            if c > 0 and cp <= 0:
                # Bullish zero cross — green arrow up
                ax.annotate('', xy=(dates.iloc[i], c),
                            xytext=(dates.iloc[i], -cci_range * 0.12),
                            arrowprops=dict(arrowstyle='->', color=COLORS['bull_marker'], lw=2.5),
                            zorder=13)
            elif c < 0 and cp >= 0:
                # Bearish zero cross — red arrow down
                ax.annotate('', xy=(dates.iloc[i], c),
                            xytext=(dates.iloc[i], cci_range * 0.12),
                            arrowprops=dict(arrowstyle='->', color=COLORS['bear_marker'], lw=2.5, linestyle='--'),
                            zorder=13)

        ax.scatter([], [], marker='^', color=COLORS['bull_marker'], s=40, label='Buy (oversold)')
        ax.scatter([], [], marker='v', color=COLORS['bear_marker'], s=40, label='Sell (overbought)')
        ax.scatter([], [], marker='^', color=COLORS['bull_marker'], s=60, label='Zero cross up (bullish)')
        ax.scatter([], [], marker='v', color=COLORS['bear_marker'], s=60, label='Zero cross down (bearish)')
        ax.legend(loc='upper right', fontsize=7, framealpha=0.8)

        _format_dates(ax)
        ax.set_ylabel('CCI', fontsize=10)
        ax.set_title(f'{symbol} — CCI', fontsize=11, fontweight='bold', pad=10)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _plot_macd(self, symbol, df):
        """MACD with histogram and signal crossover markers."""
        output_path = os.path.join(self.output_dir, f'{symbol}_technical_macd.jpg')
        fig, ax = plt.subplots(figsize=(14, 5))
        _style_ax(ax)
        dates = df['date']
        macd = df['macd']
        signal = df['macd_9_ema']
        hist = macd - signal

        # Histogram
        colors_hist = [COLORS['bull_marker'] if h >= 0 else COLORS['bear_marker'] for h in hist]
        ax.bar(dates, hist, color=colors_hist, alpha=0.4, width=0.8, label='Histogram')

        # Lines
        ax.plot(dates, macd, color='#2c3e50', linewidth=1.5, label='MACD')
        ax.plot(dates, signal, color=COLORS['bear_marker'], linewidth=1.0, linestyle='--',
                label='Signal', alpha=0.8)
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)

        # Buy/sell arrows at crossovers + zero-cross watch
        macd_range = max(abs(macd.max()), abs(macd.min()), 0.01)
        for i in range(1, len(df)):
            m, s = macd.iloc[i], signal.iloc[i]
            pm, ps = macd.iloc[i-1], signal.iloc[i-1]
            if m > s and pm <= ps:
                # BUY — bullish crossover
                ax.annotate('', xy=(dates.iloc[i], m),
                            xytext=(dates.iloc[i], m - macd_range * 0.15),
                            arrowprops=dict(arrowstyle='->', color=COLORS['bull_marker'], lw=2.0),
                            zorder=12)
            elif m < s and pm >= ps:
                # SELL — bearish crossover
                ax.annotate('', xy=(dates.iloc[i], m),
                            xytext=(dates.iloc[i], m + macd_range * 0.15),
                            arrowprops=dict(arrowstyle='->', color=COLORS['bear_marker'], lw=2.0),
                            zorder=12)

            # WATCH — zero line cross
            if m > 0 and pm <= 0:
                ax.annotate('', xy=(dates.iloc[i], 0),
                            xytext=(dates.iloc[i], -macd_range * 0.1),
                            arrowprops=dict(arrowstyle='->', color=COLORS['watch'], lw=1.5, linestyle='--'),
                            zorder=11)
            elif m < 0 and pm >= 0:
                ax.annotate('', xy=(dates.iloc[i], 0),
                            xytext=(dates.iloc[i], macd_range * 0.1),
                            arrowprops=dict(arrowstyle='->', color=COLORS['watch'], lw=1.5, linestyle='--'),
                            zorder=11)

        ax.scatter([], [], marker='^', color=COLORS['bull_marker'], s=40, label='Buy')
        ax.scatter([], [], marker='v', color=COLORS['bear_marker'], s=40, label='Sell')
        ax.scatter([], [], marker='d', color=COLORS['watch'], s=40, label='Watch (zero cross)')

        _format_dates(ax)
        ax.set_ylabel('MACD', fontsize=10)
        ax.set_title(f'{symbol} — MACD', fontsize=11, fontweight='bold', pad=10)
        ax.legend(loc='best', fontsize=8, framealpha=0.8)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ── Sector / Market ──────────────────────────────────────────────

    def _process_sector_data(self):
        for sector in SECTORS:
            sector_data = self.data[self.data['symbol'] == sector].copy()
            sector_data['date'] = pd.to_datetime(sector_data['date'])
            sector_data = sector_data.sort_values('date')
            sector_data = sector_data[sector_data['date'] <= pd.to_datetime(self.latest_date)]
            sector_data = sector_data.tail(self.n_days)
            if sector_data.empty:
                continue
            name = SECTOR_NAMES.get(sector, sector)
            output_path = os.path.join(self.output_dir, f'{sector}_sector_{name}_analysis.jpg')
            self._plot_master_technicals(sector, sector_data, title=f'{sector} — {name}',
                                          output_path=output_path)

        # Cumulative returns chart
        self._plot_sector_returns()

    def _plot_sector_returns(self):
        """Sector cumulative returns comparison chart."""
        output_path = os.path.join(self.output_dir, 'market_returns.jpg')
        fig, ax = plt.subplots(figsize=(14, 8))
        _style_ax(ax)

        cmap = plt.cm.Set2(np.linspace(0, 1, len(SECTORS)))

        for idx, sector in enumerate(SECTORS):
            sector_data = self.data[self.data['symbol'] == sector].copy()
            sector_data['date'] = pd.to_datetime(sector_data['date'])
            sector_data = sector_data.sort_values('date')
            sector_data = sector_data[sector_data['date'] <= pd.to_datetime(self.latest_date)]
            sector_data = sector_data.tail(self.n_days)
            if sector_data.empty or sector_data['adjusted_close'].iloc[0] == 0:
                continue

            initial = sector_data['adjusted_close'].iloc[0]
            cum_chg = (sector_data['adjusted_close'] / initial - 1) * 100

            name = SECTOR_NAMES.get(sector, sector)
            if sector in ['SPY', 'QQQ']:
                lw, ls, alpha = 2.5, '-', 1.0
                color = '#1a1a2e' if sector == 'SPY' else '#2980b9'
            else:
                lw, ls, alpha = 1.2, '--', 0.7
                color = cmap[idx]

            ax.plot(sector_data['date'], cum_chg, color=color, linewidth=lw,
                    linestyle=ls, alpha=alpha, label=f'{sector} ({name})')

            # End label
            ax.annotate(f'{sector} {cum_chg.iloc[-1]:+.1f}%',
                        xy=(sector_data['date'].iloc[-1], cum_chg.iloc[-1]),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=7, color=color, fontweight='bold' if sector in ['SPY', 'QQQ'] else 'normal')

        ax.axhline(0, color='black', linewidth=0.8, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%+.1f%%'))
        _format_dates(ax)
        ax.set_ylabel('Cumulative Return (%)', fontsize=10)
        ax.set_title(f'Sector & Index Returns — {self.n_days} Day', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='upper left', fontsize=7, ncol=2, framealpha=0.8)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ── Write results ────────────────────────────────────────────────

    def _write_results(self):
        if not self.results:
            logger.warning('No screener results to write.')
            return
        rows = []
        for r in self.results:
            symbol, nb, nbr, *signals = r
            if abs(nb - nbr) > 1:
                rows.append([symbol, nb, nbr, *signals])
        if not rows:
            logger.warning('No significant results.')
            return
        columns = ['symbol', 'num_bullish', 'num_bearish', 'sma_cross', 'bollinger_band',
                   'rsi', 'macd', 'macd_zero', 'adx', 'cci', 'atr', 'pe_ratio', 'pcr']
        df = pd.DataFrame(rows, columns=columns)
        csv_path = f'screener_results_{self.latest_date}.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f'Screener results written to {csv_path}')


# ── Data loading (same as v1) ────────────────────────────────────────

def combine_csvs(all_data_path: str, n_days: int, date: str):
    output = pd.DataFrame()
    for file in os.listdir(all_data_path):
        if file.startswith('all_data'):
            data_path = os.path.join(all_data_path, file)
            logger.info(f'Reading {data_path}')
            chunk = pd.read_csv(data_path)
            chunk['date'] = pd.to_datetime(chunk['date'], format='%Y-%m-%d')
            chunk = chunk[chunk['date'] >= pd.to_datetime(date) - timedelta(days=n_days)]
            if output.empty:
                output = chunk
            else:
                output = pd.concat([output, chunk])
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Screener V2 — Improved overnight visualizations')
    parser.add_argument('--symbols', nargs='+', help='Stock symbols (default: all in data)')
    parser.add_argument('--watchlist', type=str, default='../api_data/watchlist.csv')
    parser.add_argument('--data', type=str, default='../api_data/', help='Path to all_data CSVs')
    parser.add_argument('--date', type=str, default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--indicators', type=str, nargs='+', default='all')
    parser.add_argument('--visualize', action='store_true', default=True)
    parser.add_argument('--n_days', type=int, default=30)
    parser.add_argument('--whitelist', nargs='+', default=[])
    parser.add_argument('--skip_sectors', action='store_true', default=False)
    parser.add_argument('--skip_pdf', action='store_true', default=False, help='Skip PDF generation')
    args = parser.parse_args()

    csv_data = combine_csvs(args.data, args.n_days, args.date)
    symbols = args.symbols if args.symbols else csv_data['symbol'].drop_duplicates().tolist()
    symbols = [s.upper() for s in symbols]

    screener = StockScreenerV2(
        symbols=symbols, date=args.date, data=csv_data,
        indicators=args.indicators, visualize=args.visualize,
        n_days=args.n_days, whitelist=args.whitelist,
        skip_sectors=args.skip_sectors,
    )
    screener.run_screen()

    if not args.skip_pdf:
        from visualizations.pdf_overnight_v2 import SectionedPDFConverterV2
        converter = SectionedPDFConverterV2(
            directory=screener.output_dir,
            output_pdf=f'{screener.output_dir}.pdf'
        )
        converter.convert()
