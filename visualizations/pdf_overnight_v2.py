"""
PDF Overnight V2 — Improved PDF builder with heatmap screener table.

Replaces the plain black/white PIL table with a color-coded heatmap.
Replaces blank section title pages with clean dividers.

Usage (standalone):
  python visualizations/pdf_overnight_v2.py -d overnight_2026-02-05
"""

import argparse
import logging
import os
import sys

import img2pdf
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logging_config import setup_logging
from util import get_closest_trading_date
setup_logging()
logger = logging.getLogger(__name__)


# ── Color helpers ────────────────────────────────────────────────────

def _signal_color(val):
    """Return (bg, fg) tuple for signal value."""
    try:
        v = int(float(val))
    except (ValueError, TypeError):
        return (255, 255, 255), (80, 80, 80)
    if v == 1:
        return (200, 240, 200), (30, 100, 30)   # green bg, dark green text
    elif v == -1:
        return (240, 200, 200), (140, 30, 30)   # red bg, dark red text
    else:
        return (240, 240, 240), (120, 120, 120)  # gray bg, gray text


def _delta_color(val):
    """Return (bg, fg) for bull_bear_delta column."""
    try:
        v = int(float(val))
    except (ValueError, TypeError):
        return (255, 255, 255), (80, 80, 80)
    if v >= 3:
        return (100, 200, 100), (20, 80, 20)    # strong bullish
    elif v >= 1:
        return (180, 230, 180), (30, 100, 30)   # mild bullish
    elif v <= -3:
        return (200, 100, 100), (120, 20, 20)   # strong bearish
    elif v <= -1:
        return (230, 180, 180), (140, 30, 30)   # mild bearish
    else:
        return (230, 230, 230), (100, 100, 100)  # neutral


def _header_color():
    return (50, 60, 80), (255, 255, 255)


def _symbol_color():
    return (60, 70, 90), (255, 255, 255)


class SectionedPDFConverterV2:
    def __init__(self, directory, output_pdf):
        self.directory = directory
        self.output_pdf = output_pdf
        self.date = self.directory.split('/')[-1].split('_')[-1]

    def convert(self):
        sections = {
            "Stock Analysis": [],
            "Sector Analysis": [],
            "Market Analysis": [],
        }

        if not os.path.isdir(self.directory):
            logger.error(f'Directory not found: {self.directory}')
            return

        jpg_files = sorted(
            [f for f in os.listdir(self.directory) if f.lower().endswith('.jpg')],
            key=lambda x: (x.split('_')[0], os.path.getctime(os.path.join(self.directory, x)))
        )

        for f in jpg_files:
            fp = os.path.join(self.directory, f)
            fl = f.lower()
            if 'sector' in fl:
                sections["Sector Analysis"].append(fp)
            elif 'market' in fl:
                sections["Market Analysis"].append(fp)
            elif 'daily' in fl or 'technical' in fl:
                sections["Stock Analysis"].append(fp)

        images = []

        # Build heatmap table as first page
        csv_path = f'screener_results_{self.date}.csv'
        table_img = 'table_image.jpg'
        if os.path.exists(csv_path):
            self._create_heatmap_table(csv_path, table_img)
            if self._verify_image(table_img):
                images.append(table_img)
        else:
            logger.warning(f'No screener CSV found at {csv_path} — skipping table')

        # Section pages
        for section, files in sections.items():
            if not files:
                continue
            divider = self._create_section_divider(section, len(files))
            if divider:
                images.append(divider)
            for fp in files:
                if self._verify_image(fp):
                    images.append(fp)

        if images:
            with open(self.output_pdf, 'wb') as f:
                f.write(img2pdf.convert(images))
            logger.info(f'V2 PDF created: {self.output_pdf} ({len(images)} pages)')
        else:
            logger.warning('No images found — PDF not created.')

    def _verify_image(self, path):
        try:
            with Image.open(path).convert('RGB') as img:
                img.verify()
            return True
        except Exception as e:
            logger.warning(f'Invalid image: {path} — {e}')
            return False

    def _create_section_divider(self, section_name, chart_count):
        """Create a clean section divider page."""
        w, h = 1400, 300
        img = Image.new('RGB', (w, h), color=(50, 60, 80))
        draw = ImageDraw.Draw(img)

        # Try to get a larger font, fall back to default
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except (OSError, IOError):
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # Section title
        bbox = draw.textbbox((0, 0), section_name, font=font_large)
        tw = bbox[2] - bbox[0]
        draw.text(((w - tw) // 2, 100), section_name, fill='white', font=font_large)

        # Subtitle
        subtitle = f'{chart_count} charts  |  {self.date}'
        bbox2 = draw.textbbox((0, 0), subtitle, font=font_small)
        tw2 = bbox2[2] - bbox2[0]
        draw.text(((w - tw2) // 2, 180), subtitle, fill=(180, 190, 210), font=font_small)

        # Accent line
        draw.line([(w // 4, 160), (3 * w // 4, 160)], fill=(100, 120, 160), width=2)

        path = os.path.join(self.directory, f'_divider_{section_name.replace(" ", "_")}.jpg')
        img.save(path, quality=95)
        return path

    def _create_heatmap_table(self, csv_path, output_path):
        """Create a heatmap-style screener results table."""
        df = pd.read_csv(csv_path)
        df = df.sort_values(by='symbol').reset_index(drop=True)
        df['bull_bear_delta'] = df['num_bullish'] - df['num_bearish']

        # Reorder: symbol, delta, then signal columns
        cols = df.columns.tolist()
        signal_cols = [c for c in cols if c not in ('symbol', 'num_bullish', 'num_bearish', 'bull_bear_delta')]
        ordered_cols = ['symbol', 'bull_bear_delta', 'num_bullish', 'num_bearish'] + signal_cols
        ordered_cols = [c for c in ordered_cols if c in df.columns]
        df = df[ordered_cols]

        # Short column names for display
        col_display = {
            'symbol': 'Symbol', 'bull_bear_delta': 'Delta',
            'num_bullish': 'Bull', 'num_bearish': 'Bear',
            'sma_cross': 'SMA', 'bollinger_band': 'BB',
            'rsi': 'RSI', 'macd': 'MACD', 'macd_zero': 'MACD0',
            'adx': 'ADX', 'cci': 'CCI', 'atr': 'ATR',
            'pe_ratio': 'P/E', 'pcr': 'PCR',
        }

        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
            font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        except (OSError, IOError):
            font = ImageFont.load_default()
            font_bold = font
            font_title = font

        padding = 30
        cell_w = 80
        cell_h = 32
        symbol_w = 100

        num_cols = len(df.columns)
        num_rows = len(df) + 1  # +1 for header
        img_w = symbol_w + cell_w * (num_cols - 1) + padding * 2
        title_h = 60
        img_h = title_h + cell_h * num_rows + padding * 2

        img = Image.new('RGB', (img_w, img_h), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)

        # Title
        title = f'Screener Results — {self.date}'
        tbbox = draw.textbbox((0, 0), title, font=font_title)
        tw = tbbox[2] - tbbox[0]
        draw.text(((img_w - tw) // 2, 15), title, fill=(40, 50, 70), font=font_title)

        y = title_h

        # Header row
        hdr_bg, hdr_fg = _header_color()
        for col_idx, col in enumerate(df.columns):
            x = padding + (symbol_w if col_idx > 0 else 0) + (col_idx - 1) * cell_w if col_idx > 0 else padding
            w = symbol_w if col_idx == 0 else cell_w
            if col_idx == 0:
                x = padding
            else:
                x = padding + symbol_w + (col_idx - 1) * cell_w
            draw.rectangle([x, y, x + w, y + cell_h], fill=hdr_bg)
            label = col_display.get(col, col)
            draw.text((x + 6, y + 8), label, fill=hdr_fg, font=font_bold)

        y += cell_h

        # Data rows
        for row_idx, row in df.iterrows():
            for col_idx, col in enumerate(df.columns):
                val = row[col]
                if col_idx == 0:
                    x = padding
                    w = symbol_w
                else:
                    x = padding + symbol_w + (col_idx - 1) * cell_w
                    w = cell_w

                # Choose colors
                if col == 'symbol':
                    bg, fg = _symbol_color()
                    text = str(val)
                    f_use = font_bold
                elif col == 'bull_bear_delta':
                    bg, fg = _delta_color(val)
                    text = f'{int(val):+d}' if pd.notna(val) else '0'
                    f_use = font_bold
                elif col in ('num_bullish', 'num_bearish'):
                    bg = (255, 255, 255)
                    fg = (60, 60, 60)
                    text = str(int(val)) if pd.notna(val) else '0'
                    f_use = font
                else:
                    bg, fg = _signal_color(val)
                    text = str(int(float(val))) if pd.notna(val) else '0'
                    f_use = font

                draw.rectangle([x, y, x + w, y + cell_h], fill=bg)
                draw.rectangle([x, y, x + w, y + cell_h], outline=(200, 200, 200), width=1)
                draw.text((x + 6, y + 8), text, fill=fg, font=f_use)

            y += cell_h

        # Crop to actual content
        img = img.crop((0, 0, img_w, y + padding))
        img.save(output_path, quality=95)
        logger.info(f'Heatmap table saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PDF Overnight V2 — Heatmap table + improved sections')
    parser.add_argument('-d', '--directory',
                        default='overnight_' + get_closest_trading_date(pd.Timestamp.now().strftime('%Y-%m-%d')),
                        help='Directory containing JPG charts')
    args = parser.parse_args()
    output_date = get_closest_trading_date(args.directory.split('_')[-1])
    converter = SectionedPDFConverterV2(
        directory=args.directory,
        output_pdf=f'{args.directory}.pdf'
    )
    converter.convert()
