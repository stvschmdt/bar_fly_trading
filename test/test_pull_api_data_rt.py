"""
Unit tests for api_data.pull_api_data_rt

Tests all core functions with mocked API responses (no AlphaVantage calls).
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from api_data.pull_api_data_rt import (
    CORE_RT_COLUMNS,
    fetch_realtime_quote,
    fetch_symbol_rt_data,
    update_derived_columns,
    update_csv_file,
    find_csv_files,
    get_symbols_from_csv,
    load_symbol_universe,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

class MockApiClient:
    """Mock AlphaVantage client that returns canned responses."""

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.call_log = []

    def fetch(self, **kwargs):
        self.call_log.append(kwargs)
        function = kwargs.get('function', '')
        symbol = kwargs.get('symbol', '')
        key = (function, symbol)
        if key in self.responses:
            return self.responses[key]
        # Also check function-only key for generic responses
        if function in self.responses:
            return self.responses[function]
        return {}


def make_quote_response(symbol='AAPL', price=150.0, open_=148.0,
                        high=152.0, low=147.0, volume=50000000):
    """Build a GLOBAL_QUOTE API response."""
    return {
        'Global Quote': {
            '01. symbol': symbol,
            '02. open': str(open_),
            '03. high': str(high),
            '04. low': str(low),
            '05. price': str(price),
            '06. volume': str(volume),
            '07. latest trading day': '2026-02-07',
            '08. previous close': '149.00',
            '09. change': str(price - 149.0),
            '10. change percent': f'{(price - 149.0) / 149.0 * 100:.4f}%',
        }
    }


def make_test_csv(tmpdir, batch_num=0, symbols=None, n_rows_per_symbol=5):
    """
    Create a test all_data CSV with realistic columns and data.

    Returns:
        Path to the created CSV file
    """
    if symbols is None:
        symbols = ['AAPL', 'NVDA', 'TSLA']

    rows = []
    base_date = pd.Timestamp('2026-01-20')
    idx = 0
    for sym in symbols:
        for day_offset in range(n_rows_per_symbol):
            date = base_date + pd.Timedelta(days=day_offset)
            # Skip weekends
            while date.weekday() >= 5:
                date += pd.Timedelta(days=1)

            price = 100 + day_offset * 2 + hash(sym) % 50
            rows.append({
                'date': date.strftime('%Y-%m-%d'),
                'symbol': sym,
                'open': price - 1,
                'high': price + 2,
                'low': price - 2,
                'adjusted_close': price,
                'volume': 10000000 + day_offset * 100000,
                'sma_20': price * 0.98,
                'sma_50': price * 0.95,
                'sma_200': price * 0.90,
                'ema_20': price * 0.99,
                'ema_50': price * 0.96,
                'ema_200': price * 0.91,
                'macd': 0.5 + day_offset * 0.1,
                'rsi_14': 55 + day_offset,
                'adx_14': 25 + day_offset,
                'atr_14': 3.0 + day_offset * 0.1,
                'cci_14': 50 + day_offset * 10,
                'bbands_upper_20': price + 5,
                'bbands_middle_20': price,
                'bbands_lower_20': price - 5,
                'sma_20_pct': 2.04,
                'sma_50_pct': 5.26,
                'sma_200_pct': 11.11,
                '52_week_high': price * 1.2,
                '52_week_low': price * 0.7,
                '52_week_high_pct': -16.67,
                '52_week_low_pct': 42.86,
                'pe_ratio': 25.0,
                'ttm_eps': price / 25.0,
                'adjusted_close_pct': 0.02 if day_offset > 0 else np.nan,
                'volume_pct': 0.01 if day_offset > 0 else np.nan,
                'macd_9_ema': 0.45,
                'sector': 'TECHNOLOGY',
                'industry': 'SEMICONDUCTORS',
                'treasury_yield_2year': 4.5,
            })
            idx += 1

    df = pd.DataFrame(rows)
    path = os.path.join(tmpdir, f'all_data_{batch_num}.csv')
    df.to_csv(path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Tests: fetch_realtime_quote
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchRealtimeQuote:
    def test_basic_quote(self):
        client = MockApiClient({
            ('GLOBAL_QUOTE', 'AAPL'): make_quote_response('AAPL', price=155.50)
        })
        result = fetch_realtime_quote(client, 'AAPL')

        assert result['adjusted_close'] == 155.50
        assert result['open'] == 148.0
        assert result['high'] == 152.0
        assert result['low'] == 147.0
        assert result['volume'] == 50000000

    def test_missing_quote_raises(self):
        client = MockApiClient({('GLOBAL_QUOTE', 'BAD'): {}})
        with pytest.raises(ValueError, match="No quote data"):
            fetch_realtime_quote(client, 'BAD')

    def test_empty_global_quote_raises(self):
        client = MockApiClient({
            ('GLOBAL_QUOTE', 'BAD'): {'Global Quote': {}}
        })
        with pytest.raises(ValueError, match="No quote data"):
            fetch_realtime_quote(client, 'BAD')

    def test_returns_all_core_columns(self):
        client = MockApiClient({
            ('GLOBAL_QUOTE', 'AAPL'): make_quote_response('AAPL')
        })
        result = fetch_realtime_quote(client, 'AAPL')
        for col in CORE_RT_COLUMNS:
            assert col in result, f"Missing column: {col}"

    def test_single_api_call(self):
        """OHLCV-only mode should make exactly 1 API call per symbol."""
        client = MockApiClient({
            ('GLOBAL_QUOTE', 'AAPL'): make_quote_response('AAPL')
        })
        fetch_realtime_quote(client, 'AAPL')
        assert len(client.call_log) == 1
        assert client.call_log[0]['function'] == 'GLOBAL_QUOTE'


# ─────────────────────────────────────────────────────────────────────────────
# Tests: fetch_symbol_rt_data
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchSymbolRtData:
    def test_success(self):
        responses = {
            ('GLOBAL_QUOTE', 'AAPL'): make_quote_response('AAPL', price=155.0),
        }
        client = MockApiClient(responses)
        result = fetch_symbol_rt_data(client, 'AAPL')

        assert result is not None
        assert 'ohlcv' in result
        assert result['ohlcv']['adjusted_close'] == 155.0

    def test_quote_failure_returns_none(self):
        client = MockApiClient({})
        result = fetch_symbol_rt_data(client, 'BAD')
        assert result is None

    def test_only_one_api_call(self):
        """fetch_symbol_rt_data should make exactly 1 API call (no technicals)."""
        responses = {
            ('GLOBAL_QUOTE', 'AAPL'): make_quote_response('AAPL', price=155.0),
        }
        client = MockApiClient(responses)
        fetch_symbol_rt_data(client, 'AAPL')
        assert len(client.call_log) == 1

    def test_no_technicals_key(self):
        """Result should not contain a 'technicals' key."""
        responses = {
            ('GLOBAL_QUOTE', 'AAPL'): make_quote_response('AAPL', price=155.0),
        }
        client = MockApiClient(responses)
        result = fetch_symbol_rt_data(client, 'AAPL')
        assert 'technicals' not in result


# ─────────────────────────────────────────────────────────────────────────────
# Tests: update_derived_columns
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateDerivedColumns:
    def test_sma_pct_update(self):
        df = pd.DataFrame([
            {'symbol': 'AAPL', 'adjusted_close': 100.0, 'sma_20': 95.0,
             'sma_50': 90.0, 'sma_200': 80.0,
             'sma_20_pct': 0.0, 'sma_50_pct': 0.0, 'sma_200_pct': 0.0},
        ])
        update_derived_columns(df, 0)

        assert df.loc[0, 'sma_20_pct'] == pytest.approx(5.26, abs=0.01)
        assert df.loc[0, 'sma_50_pct'] == pytest.approx(11.11, abs=0.01)
        assert df.loc[0, 'sma_200_pct'] == pytest.approx(25.0, abs=0.01)

    def test_pe_ratio_update(self):
        df = pd.DataFrame([
            {'symbol': 'AAPL', 'adjusted_close': 150, 'ttm_eps': 6.0,
             'pe_ratio': 0, 'sma_20': 0, 'sma_50': 0, 'sma_200': 0,
             'sma_20_pct': 0, 'sma_50_pct': 0, 'sma_200_pct': 0},
        ])
        update_derived_columns(df, 0)
        assert df.loc[0, 'pe_ratio'] == pytest.approx(25.0)

    def test_52_week_pct(self):
        df = pd.DataFrame([
            {'symbol': 'AAPL', 'adjusted_close': 150,
             '52_week_high': 200, '52_week_low': 100,
             '52_week_high_pct': 0, '52_week_low_pct': 0},
        ])
        update_derived_columns(df, 0)
        assert df.loc[0, '52_week_high_pct'] == -25.0
        assert df.loc[0, '52_week_low_pct'] == 50.0

    def test_pct_change_vs_previous(self):
        df = pd.DataFrame([
            {'symbol': 'AAPL', 'adjusted_close': 100, 'volume': 1000000,
             'adjusted_close_pct': np.nan, 'volume_pct': np.nan},
            {'symbol': 'AAPL', 'adjusted_close': 110, 'volume': 1200000,
             'adjusted_close_pct': np.nan, 'volume_pct': np.nan},
        ])
        update_derived_columns(df, 1)
        assert df.loc[1, 'adjusted_close_pct'] == pytest.approx(0.10)
        assert df.loc[1, 'volume_pct'] == pytest.approx(0.20)

    def test_zero_sma_no_crash(self):
        df = pd.DataFrame([
            {'symbol': 'AAPL', 'adjusted_close': 100, 'sma_20': 0,
             'sma_20_pct': 0, 'sma_50': 50, 'sma_50_pct': 0,
             'sma_200': 0, 'sma_200_pct': 0},
        ])
        update_derived_columns(df, 0)
        assert df.loc[0, 'sma_20_pct'] == 0  # unchanged (div by zero guard)
        assert df.loc[0, 'sma_50_pct'] == 100.0

    def test_nan_price_no_crash(self):
        df = pd.DataFrame([
            {'symbol': 'AAPL', 'adjusted_close': np.nan, 'sma_20': 95,
             'sma_20_pct': 0, 'sma_50': 0, 'sma_50_pct': 0,
             'sma_200': 0, 'sma_200_pct': 0},
        ])
        update_derived_columns(df, 0)
        # Should not crash; values unchanged
        assert df.loc[0, 'sma_20_pct'] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Tests: update_csv_file
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateCsvFile:
    def test_updates_latest_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = make_test_csv(tmpdir, symbols=['AAPL', 'NVDA'])

            rt_data = {
                'AAPL': {
                    'ohlcv': {
                        'open': 200.0, 'high': 210.0, 'low': 195.0,
                        'adjusted_close': 205.0, 'volume': 99999999,
                    },
                },
            }

            n = update_csv_file(csv_path, rt_data)
            assert n == 1

            # Read back and verify
            df = pd.read_csv(csv_path, index_col=0)
            aapl = df[df['symbol'] == 'AAPL']
            latest = aapl.iloc[-1]

            assert latest['adjusted_close'] == 205.0
            assert latest['open'] == 200.0
            assert latest['volume'] == 99999999

    def test_technicals_unchanged(self):
        """OHLCV-only update should not touch technical indicator columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = make_test_csv(tmpdir, symbols=['AAPL'])

            orig = pd.read_csv(csv_path, index_col=0)
            orig_sma = orig[orig['symbol'] == 'AAPL'].iloc[-1]['sma_20']
            orig_rsi = orig[orig['symbol'] == 'AAPL'].iloc[-1]['rsi_14']
            orig_bbands = orig[orig['symbol'] == 'AAPL'].iloc[-1]['bbands_upper_20']

            rt_data = {
                'AAPL': {
                    'ohlcv': {
                        'open': 200.0, 'high': 210.0, 'low': 195.0,
                        'adjusted_close': 205.0, 'volume': 99999999,
                    },
                },
            }

            update_csv_file(csv_path, rt_data)

            updated = pd.read_csv(csv_path, index_col=0)
            latest = updated[updated['symbol'] == 'AAPL'].iloc[-1]

            # Technicals should be untouched
            assert latest['sma_20'] == orig_sma
            assert latest['rsi_14'] == orig_rsi
            assert latest['bbands_upper_20'] == orig_bbands

    def test_does_not_modify_other_symbols(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = make_test_csv(tmpdir, symbols=['AAPL', 'NVDA'])

            # Read NVDA's original last row
            original = pd.read_csv(csv_path, index_col=0)
            nvda_original = original[original['symbol'] == 'NVDA'].iloc[-1].copy()

            rt_data = {
                'AAPL': {
                    'ohlcv': {'adjusted_close': 999.0, 'open': 990.0,
                              'high': 1000.0, 'low': 980.0, 'volume': 1},
                },
            }

            update_csv_file(csv_path, rt_data)

            # Verify NVDA unchanged
            updated = pd.read_csv(csv_path, index_col=0)
            nvda_updated = updated[updated['symbol'] == 'NVDA'].iloc[-1]

            assert nvda_updated['adjusted_close'] == nvda_original['adjusted_close']
            assert nvda_updated['sma_20'] == nvda_original['sma_20']

    def test_dry_run_no_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = make_test_csv(tmpdir, symbols=['AAPL'])

            original = pd.read_csv(csv_path, index_col=0)
            orig_price = original[original['symbol'] == 'AAPL'].iloc[-1]['adjusted_close']

            rt_data = {
                'AAPL': {
                    'ohlcv': {'adjusted_close': 999.0, 'open': 990.0,
                              'high': 1000.0, 'low': 980.0, 'volume': 1},
                },
            }

            n = update_csv_file(csv_path, rt_data, dry_run=True)
            assert n == 1

            # Verify file unchanged
            after = pd.read_csv(csv_path, index_col=0)
            after_price = after[after['symbol'] == 'AAPL'].iloc[-1]['adjusted_close']
            assert after_price == orig_price

    def test_unknown_symbol_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = make_test_csv(tmpdir, symbols=['AAPL'])

            rt_data = {
                'UNKNOWN': {
                    'ohlcv': {'adjusted_close': 50.0, 'open': 49.0,
                              'high': 51.0, 'low': 48.0, 'volume': 100},
                },
            }

            n = update_csv_file(csv_path, rt_data)
            assert n == 0

    def test_none_data_skipped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = make_test_csv(tmpdir, symbols=['AAPL'])

            rt_data = {'AAPL': None}
            n = update_csv_file(csv_path, rt_data)
            assert n == 0

    def test_derived_columns_recomputed(self):
        """sma_*_pct should be recomputed using new price vs existing (nightly) SMA."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = make_test_csv(tmpdir, symbols=['AAPL'], n_rows_per_symbol=3)

            rt_data = {
                'AAPL': {
                    'ohlcv': {'adjusted_close': 200.0, 'open': 195.0,
                              'high': 205.0, 'low': 190.0, 'volume': 20000000},
                },
            }

            update_csv_file(csv_path, rt_data)

            df = pd.read_csv(csv_path, index_col=0)
            latest = df[df['symbol'] == 'AAPL'].iloc[-1]

            # sma_20 was set to price * 0.98 in make_test_csv (original price)
            # After update, sma_20 is UNCHANGED (nightly value), but price is now 200
            # sma_20_pct = (200 - sma_20) / sma_20 * 100
            sma_20 = latest['sma_20']
            expected_pct = round((200 - sma_20) / sma_20 * 100, 2)
            assert latest['sma_20_pct'] == pytest.approx(expected_pct, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: find_csv_files and get_symbols_from_csv
# ─────────────────────────────────────────────────────────────────────────────

class TestFileHelpers:
    def test_find_csv_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            make_test_csv(tmpdir, batch_num=0)
            make_test_csv(tmpdir, batch_num=1)
            # Create a non-matching file
            pd.DataFrame({'x': [1]}).to_csv(os.path.join(tmpdir, 'other.csv'))

            files = find_csv_files(tmpdir)
            assert len(files) == 2
            assert all('all_data_' in f for f in files)

    def test_find_csv_files_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            files = find_csv_files(tmpdir)
            assert files == []

    def test_get_symbols_from_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = make_test_csv(tmpdir, symbols=['AAPL', 'NVDA', 'TSLA'])
            symbols = get_symbols_from_csv(csv_path)
            assert set(symbols) == {'AAPL', 'NVDA', 'TSLA'}


# ─────────────────────────────────────────────────────────────────────────────
# Tests: column constants
# ─────────────────────────────────────────────────────────────────────────────

class TestConstants:
    def test_no_duplicates(self):
        assert len(CORE_RT_COLUMNS) == len(set(CORE_RT_COLUMNS))

    def test_expected_core_columns(self):
        assert 'adjusted_close' in CORE_RT_COLUMNS
        assert 'volume' in CORE_RT_COLUMNS
        assert 'open' in CORE_RT_COLUMNS

    def test_ohlcv_only(self):
        """CORE_RT_COLUMNS should only contain OHLCV — no technicals."""
        assert len(CORE_RT_COLUMNS) == 5
        for col in CORE_RT_COLUMNS:
            assert col in ('open', 'high', 'low', 'adjusted_close', 'volume')


# ─────────────────────────────────────────────────────────────────────────────
# Tests: load_symbol_universe
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadSymbolUniverse:
    def test_loads_from_real_files(self):
        """Should load symbols from sp500.csv + watchlist.csv."""
        symbols = load_symbol_universe()
        assert len(symbols) > 0
        assert isinstance(symbols, set)
        # Known symbols that should be in both
        assert 'AAPL' in symbols
        assert 'NVDA' in symbols

    def test_returns_set(self):
        symbols = load_symbol_universe()
        assert isinstance(symbols, set)

    def test_no_duplicates(self):
        """Union should deduplicate overlapping symbols."""
        symbols = load_symbol_universe()
        # If it returns a set, there are no duplicates by definition
        assert len(symbols) == len(set(symbols))

    def test_combined_count(self):
        """Should have roughly SP500 + watchlist unique count (~543)."""
        symbols = load_symbol_universe()
        # At least SP500 count, at most SP500 + full watchlist
        assert len(symbols) >= 400
        assert len(symbols) <= 700


# ─────────────────────────────────────────────────────────────────────────────
# Tests: date format consistency
# ─────────────────────────────────────────────────────────────────────────────

class TestDateFormat:
    def test_csv_date_format_preserved(self):
        """Date column should remain YYYY-MM-DD after RT update."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = make_test_csv(tmpdir, symbols=['AAPL'], n_rows_per_symbol=3)

            # Read original dates
            orig = pd.read_csv(csv_path, index_col=0)
            orig_dates = orig['date'].tolist()

            rt_data = {
                'AAPL': {
                    'ohlcv': {'adjusted_close': 999.0, 'open': 990.0,
                              'high': 1000.0, 'low': 980.0, 'volume': 1},
                },
            }
            update_csv_file(csv_path, rt_data)

            # Dates should be completely unchanged
            updated = pd.read_csv(csv_path, index_col=0)
            assert updated['date'].tolist() == orig_dates

    def test_date_format_is_yyyy_mm_dd(self):
        """All dates should match YYYY-MM-DD pattern."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = make_test_csv(tmpdir, symbols=['AAPL'])
            df = pd.read_csv(csv_path, index_col=0)
            import re
            for d in df['date']:
                assert re.match(r'^\d{4}-\d{2}-\d{2}$', d), f"Bad date format: {d}"

    def test_api_date_format_matches_csv(self):
        """The GLOBAL_QUOTE API returns dates in YYYY-MM-DD format."""
        response = make_quote_response('AAPL')
        api_date = response['Global Quote']['07. latest trading day']
        import re
        assert re.match(r'^\d{4}-\d{2}-\d{2}$', api_date), f"API date format: {api_date}"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: CSV round-trip integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestCsvRoundTrip:
    def test_non_rt_columns_unchanged(self):
        """Static columns (sector, treasury, earnings) should survive RT update."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = make_test_csv(tmpdir, symbols=['AAPL'], n_rows_per_symbol=3)

            orig = pd.read_csv(csv_path, index_col=0)
            orig_sector = orig.iloc[-1]['sector']
            orig_treasury = orig.iloc[-1]['treasury_yield_2year']

            rt_data = {
                'AAPL': {
                    'ohlcv': {'adjusted_close': 999.0, 'open': 990.0,
                              'high': 1000.0, 'low': 980.0, 'volume': 1},
                },
            }
            update_csv_file(csv_path, rt_data)

            updated = pd.read_csv(csv_path, index_col=0)
            assert updated.iloc[-1]['sector'] == orig_sector
            assert updated.iloc[-1]['treasury_yield_2year'] == orig_treasury

    def test_row_count_preserved(self):
        """Number of rows should not change after RT update."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = make_test_csv(tmpdir, symbols=['AAPL', 'NVDA'])

            orig = pd.read_csv(csv_path, index_col=0)
            orig_rows = len(orig)

            rt_data = {
                'AAPL': {
                    'ohlcv': {'adjusted_close': 999.0, 'open': 990.0,
                              'high': 1000.0, 'low': 980.0, 'volume': 1},
                },
                'NVDA': {
                    'ohlcv': {'adjusted_close': 888.0, 'open': 880.0,
                              'high': 890.0, 'low': 870.0, 'volume': 2},
                },
            }
            update_csv_file(csv_path, rt_data)

            updated = pd.read_csv(csv_path, index_col=0)
            assert len(updated) == orig_rows

    def test_column_count_preserved(self):
        """Number of columns should not change after RT update."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = make_test_csv(tmpdir, symbols=['AAPL'])

            orig = pd.read_csv(csv_path, index_col=0)
            orig_cols = set(orig.columns)

            rt_data = {
                'AAPL': {
                    'ohlcv': {'adjusted_close': 999.0, 'open': 990.0,
                              'high': 1000.0, 'low': 980.0, 'volume': 1},
                },
            }
            update_csv_file(csv_path, rt_data)

            updated = pd.read_csv(csv_path, index_col=0)
            assert set(updated.columns) == orig_cols

    def test_historical_rows_untouched(self):
        """Only the latest row per symbol should change; historical rows stay intact."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = make_test_csv(tmpdir, symbols=['AAPL'], n_rows_per_symbol=5)

            orig = pd.read_csv(csv_path, index_col=0)
            # Save first 4 rows (historical)
            orig_historical = orig.iloc[:-1].copy()

            rt_data = {
                'AAPL': {
                    'ohlcv': {'adjusted_close': 999.0, 'open': 990.0,
                              'high': 1000.0, 'low': 980.0, 'volume': 1},
                },
            }
            update_csv_file(csv_path, rt_data)

            updated = pd.read_csv(csv_path, index_col=0)
            updated_historical = updated.iloc[:-1]

            # Compare all historical rows
            pd.testing.assert_frame_equal(updated_historical, orig_historical)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: integration (multi-file update)
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    def test_multi_file_update(self):
        """Test updating symbols across multiple CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv0 = make_test_csv(tmpdir, batch_num=0, symbols=['AAPL', 'NVDA'])
            csv1 = make_test_csv(tmpdir, batch_num=1, symbols=['TSLA', 'GOOG'])

            rt_data = {
                'AAPL': {
                    'ohlcv': {'adjusted_close': 200.0, 'open': 195.0,
                              'high': 205.0, 'low': 190.0, 'volume': 1},
                },
                'TSLA': {
                    'ohlcv': {'adjusted_close': 300.0, 'open': 295.0,
                              'high': 305.0, 'low': 290.0, 'volume': 2},
                },
            }

            # Update file 0
            syms0 = get_symbols_from_csv(csv0)
            data0 = {s: rt_data[s] for s in syms0 if s in rt_data}
            n0 = update_csv_file(csv0, data0)
            assert n0 == 1  # AAPL

            # Update file 1
            syms1 = get_symbols_from_csv(csv1)
            data1 = {s: rt_data[s] for s in syms1 if s in rt_data}
            n1 = update_csv_file(csv1, data1)
            assert n1 == 1  # TSLA

            # Verify
            df0 = pd.read_csv(csv0, index_col=0)
            aapl = df0[df0['symbol'] == 'AAPL'].iloc[-1]
            assert aapl['adjusted_close'] == 200.0

            df1 = pd.read_csv(csv1, index_col=0)
            tsla = df1[df1['symbol'] == 'TSLA'].iloc[-1]
            assert tsla['adjusted_close'] == 300.0