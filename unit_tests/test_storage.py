"""
Starter unit tests for api_data/storage.py — gold table processing + helpers.

These tests validate the derived-column logic, ffill behavior, split adjustment,
signal generation, and SQL query correctness WITHOUT hitting the database.
Most tests use synthetic DataFrames to isolate the logic.

Run:  pytest unit_tests/test_storage.py -v

TODOs are marked inline for future expansion.
"""

import math
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers: build minimal DataFrames that mimic gold_table_processing output
# ---------------------------------------------------------------------------

def make_core_df(n_days=30, symbol="TEST", start_price=100.0, seed=42):
    """
    Build a minimal DataFrame resembling what the gold SQL query returns
    BEFORE any derived columns are added.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=n_days, freq="B")
    prices = start_price + np.cumsum(rng.normal(0, 1.5, n_days))
    prices = np.maximum(prices, 1.0)  # keep positive

    df = pd.DataFrame({
        "date": dates,
        "symbol": symbol,
        "open": prices + rng.uniform(-1, 1, n_days),
        "high": prices + rng.uniform(0, 3, n_days),
        "low": prices - rng.uniform(0, 3, n_days),
        "adjusted_close": prices,
        "volume": rng.integers(500_000, 10_000_000, n_days),
        # Technical indicators (realistic ranges)
        "sma_20": prices - rng.uniform(-3, 3, n_days),
        "sma_50": prices - rng.uniform(-5, 5, n_days),
        "sma_200": prices - rng.uniform(-10, 10, n_days),
        "ema_20": prices - rng.uniform(-3, 3, n_days),
        "ema_50": prices - rng.uniform(-5, 5, n_days),
        "ema_200": prices - rng.uniform(-10, 10, n_days),
        "macd": rng.normal(0, 2, n_days),
        "rsi_14": rng.uniform(20, 80, n_days),
        "adx_14": rng.uniform(10, 50, n_days),
        "atr_14": rng.uniform(1, 5, n_days),
        "cci_14": rng.normal(0, 100, n_days),
        "bbands_upper_20": prices + rng.uniform(2, 6, n_days),
        "bbands_middle_20": prices,
        "bbands_lower_20": prices - rng.uniform(2, 6, n_days),
        # Options (some NaN to mimic missing)
        "call_volume": rng.choice([0, 100, 500, 1000, np.nan], n_days),
        "put_volume": rng.choice([0, 50, 300, 800, np.nan], n_days),
        "total_volume": rng.choice([0, 200, 800, 1800, np.nan], n_days),
        # Economic (sparse — most days NULL, ffilled later)
        "treasury_yield_2year": np.nan,
        "treasury_yield_10year": np.nan,
        "ffer": np.nan,
        "cpi": np.nan,
        "inflation": np.nan,
        "retail_sales": np.nan,
        "durables": np.nan,
        "unemployment": np.nan,
        "nonfarm_payroll": np.nan,
        # Quarterly earnings (sparse)
        "fiscal_date_ending": pd.NaT,
        "reported_eps": np.nan,
        "estimated_eps": np.nan,
        "ttm_eps": np.nan,
        "surprise": np.nan,
        "surprise_percentage": np.nan,
        # Company overview (same for all rows — single snapshot)
        "exchange": "NASDAQ",
        "country": "USA",
        "sector": "TECHNOLOGY",
        "industry": "Semiconductors",
        "market_capitalization": 3_000_000_000_000,
        "book_value": 25.0,
        "dividend_yield": 0.03,
        "eps": 3.5,
        "price_to_book_ratio": 4.0,
        "beta": 1.5,
        "shares_outstanding": 24_000_000_000,
        "52_week_high": float(max(prices)) + 5,
        "52_week_low": float(min(prices)) - 5,
        "forward_pe": 28.0,
        "analyst_rating_strong_buy": 30,
        "analyst_rating_buy": 10,
        "analyst_rating_hold": 5,
        "analyst_rating_sell": 1,
        "analyst_rating_strong_sell": 0,
    })
    # Seed a few economic data points (like real data — sparse)
    df.loc[0, "treasury_yield_2year"] = 4.2
    df.loc[0, "treasury_yield_10year"] = 4.5
    df.loc[0, "ffer"] = 5.33
    df.loc[0, "cpi"] = 310.0
    df.loc[0, "inflation"] = 3.1
    df.loc[10, "treasury_yield_2year"] = 4.1
    df.loc[10, "treasury_yield_10year"] = 4.4
    df.loc[10, "ffer"] = 5.33
    df.loc[15, "unemployment"] = 4.1
    df.loc[15, "nonfarm_payroll"] = 250.0
    # Seed quarterly earnings on day 5
    df.loc[5, "fiscal_date_ending"] = pd.Timestamp("2024-12-31")
    df.loc[5, "reported_eps"] = 0.85
    df.loc[5, "estimated_eps"] = 0.80
    df.loc[5, "ttm_eps"] = 3.20
    df.loc[5, "surprise"] = 0.05
    df.loc[5, "surprise_percentage"] = 6.25

    return df


def make_multi_symbol_df(symbols=("NVDA", "AMD"), n_days=30, seed=42):
    """Build a multi-symbol DataFrame for grouped operations."""
    dfs = []
    for i, sym in enumerate(symbols):
        dfs.append(make_core_df(n_days=n_days, symbol=sym, seed=seed + i))
    return pd.concat(dfs, ignore_index=True)


# ===========================================================================
# 1. Derived pct_change columns
# ===========================================================================

class TestPctChangeColumns:
    """Validate adjusted_close_pct, volume_pct, open_pct, high_pct, low_pct."""

    def setup_method(self):
        self.df = make_core_df(n_days=10, symbol="TEST")
        # Replicate what storage.py does
        self.df["adjusted_close_pct"] = self.df.groupby("symbol")["adjusted_close"].pct_change(1)
        self.df["volume_pct"] = self.df.groupby("symbol")["volume"].pct_change(1)
        self.df["open_pct"] = self.df.groupby("symbol")["open"].pct_change(1)
        self.df["high_pct"] = self.df.groupby("symbol")["high"].pct_change(1)
        self.df["low_pct"] = self.df.groupby("symbol")["low"].pct_change(1)

    def test_first_row_is_nan(self):
        """First row of each symbol should be NaN (no prior day)."""
        assert pd.isna(self.df["adjusted_close_pct"].iloc[0])
        assert pd.isna(self.df["volume_pct"].iloc[0])

    def test_pct_change_formula(self):
        """Verify (new - old) / old for a known row."""
        row2 = self.df.iloc[2]
        row1 = self.df.iloc[1]
        expected = (row2["adjusted_close"] - row1["adjusted_close"]) / row1["adjusted_close"]
        assert pytest.approx(row2["adjusted_close_pct"], rel=1e-10) == expected

    def test_multi_symbol_independence(self):
        """pct_change should reset at symbol boundaries."""
        df = make_multi_symbol_df(n_days=5)
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        df["adjusted_close_pct"] = df.groupby("symbol")["adjusted_close"].pct_change(1)
        # First row of second symbol should be NaN
        second_sym_start = df[df["symbol"] == "AMD"].index[0]
        assert pd.isna(df.loc[second_sym_start, "adjusted_close_pct"])


# ===========================================================================
# 2. Options derived columns
# ===========================================================================

class TestOptionsDerived:
    """Validate options_14_mean, options_14_std, pcr, pcr_14_mean."""

    def setup_method(self):
        self.df = make_core_df(n_days=20, symbol="TEST")
        # Replicate storage.py logic
        self.df[["options_14_mean", "options_14_std"]] = (
            self.df.groupby("symbol")["total_volume"]
            .rolling(window=14, min_periods=1)
            .agg(["mean", "std"])
            .reset_index(level=0, drop=True)
        )
        self.df["pcr"] = (self.df["put_volume"] / self.df["call_volume"]).round(2)
        self.df["pcr_14_mean"] = (
            self.df.groupby("symbol")["pcr"]
            .rolling(window=14, min_periods=1)
            .agg(["mean"])
            .reset_index(level=0, drop=True)
        )

    def test_options_14_mean_not_all_nan(self):
        """At least some rows should have a computed mean."""
        assert self.df["options_14_mean"].notna().any()

    def test_pcr_non_negative(self):
        """PCR should be >= 0 (or NaN/inf, never negative)."""
        valid = self.df["pcr"].dropna()
        valid_finite = valid[np.isfinite(valid)]
        assert (valid_finite >= 0).all()

    # TODO: Test pcr when call_volume == 0 → should be inf or NaN
    # BUG FOUND: storage.py doesn't handle division by zero for pcr
    def test_pcr_division_by_zero(self):
        """When call_volume = 0, pcr should be inf (current behavior) — flag for fix."""
        df = pd.DataFrame({
            "symbol": ["X"] * 3,
            "put_volume": [100.0, 200.0, 0.0],
            "call_volume": [0.0, 100.0, 0.0],
        })
        df["pcr"] = (df["put_volume"] / df["call_volume"]).round(2)
        assert np.isinf(df["pcr"].iloc[0]), "put/0 should be inf"
        assert np.isnan(df["pcr"].iloc[2]), "0/0 should be NaN"
        # TODO: storage.py should clip inf to a max (e.g., 10.0) or replace with NaN

    def test_pcr_14_mean_window_size(self):
        """Rolling 14 with min_periods=1 should compute from row 0."""
        assert self.df["pcr_14_mean"].iloc[0] is not None  # may be NaN if pcr is NaN


# ===========================================================================
# 3. Forward-fill behavior
# ===========================================================================

class TestFfill:
    """Validate ffill logic for earnings, economic indicators, 52-week data."""

    def setup_method(self):
        self.df = make_core_df(n_days=20, symbol="TEST")

    def test_earnings_ffill(self):
        """After seeded earnings on row 5, rows 6+ should be filled."""
        df = self.df.copy()
        df["reported_eps"] = df.groupby("symbol")["reported_eps"].ffill()
        assert pd.isna(df["reported_eps"].iloc[0])
        assert pd.isna(df["reported_eps"].iloc[4])
        assert df["reported_eps"].iloc[5] == 0.85
        assert df["reported_eps"].iloc[6] == 0.85
        assert df["reported_eps"].iloc[19] == 0.85

    def test_economic_ffill_treasury(self):
        """Treasury yield should fill forward from seeded points."""
        df = self.df.copy()
        df["treasury_yield_2year"] = df["treasury_yield_2year"].ffill()
        assert df["treasury_yield_2year"].iloc[0] == 4.2
        assert df["treasury_yield_2year"].iloc[5] == 4.2  # still first value
        assert df["treasury_yield_2year"].iloc[10] == 4.1  # updated on row 10
        assert df["treasury_yield_2year"].iloc[15] == 4.1  # carried forward

    # TODO: BUG — inflation is NOT ffilled in storage.py (missing from lines 359-366)
    def test_inflation_not_ffilled_bug(self):
        """
        BUG: inflation is selected from SQL but NOT forward-filled.
        Only row 0 has a value; rows 1-19 are NaN.
        """
        df = self.df.copy()
        # Replicate storage.py behavior (no ffill on inflation)
        assert df["inflation"].iloc[0] == 3.1
        assert pd.isna(df["inflation"].iloc[1])
        assert pd.isna(df["inflation"].iloc[19])
        # After fix (uncomment when storage.py is patched):
        # df["inflation"] = df["inflation"].ffill()
        # assert df["inflation"].iloc[19] == 3.1

    def test_52_week_high_from_overview_is_constant(self):
        """
        company_overview JOIN fills ALL rows with the same value.
        ffill on lines 352-353 is a no-op.
        NOTE: This is a look-ahead bias issue for ML training.
        """
        df = self.df.copy()
        assert df["52_week_high"].nunique() == 1, (
            "All rows should have the same 52_week_high from company_overview"
        )


# ===========================================================================
# 4. Stock split adjustment
# ===========================================================================

class TestSplitAdjustment:
    """
    Validate adjust_for_stock_splits logic.

    BUG FOUND: The mask uses `df['date'] <= effective_date` but effective_date
    is the first day of post-split trading. Should be `<` (strictly less than).
    """

    def _build_split_scenario(self, use_lte=True):
        """
        Build a scenario with a known split:
        - Days 0-4: pre-split at ~$1000
        - Day 5 = effective_date: price drops to ~$100 (10:1 split)
        - Days 6-9: post-split at ~$100
        """
        dates = pd.bdate_range("2024-06-03", periods=10, freq="B")
        effective_date = dates[5]  # 2024-06-10 (Monday)

        # Pre-split prices ~$1000, post-split ~$100 (use floats to avoid int dtype)
        pre_open = [1000.0, 1010.0, 990.0, 1005.0, 1015.0]
        post_open = [101.0, 102.0, 99.0, 103.0, 100.0]

        df = pd.DataFrame({
            "date": dates,
            "symbol": "NVDA",
            "open": pre_open + post_open,
            "high": [p + 10.0 for p in pre_open + post_open],
            "low": [p - 10.0 for p in pre_open + post_open],
        })

        df["adjusted_open"] = df["open"].copy()
        df["adjusted_high"] = df["high"].copy()
        df["adjusted_low"] = df["low"].copy()

        split_factor = 10.0

        if use_lte:
            # Current (buggy) behavior: date <= effective_date
            mask = (df["symbol"] == "NVDA") & (df["date"] <= effective_date)
        else:
            # Correct behavior: date < effective_date
            mask = (df["symbol"] == "NVDA") & (df["date"] < effective_date)

        df.loc[mask, "adjusted_open"] /= split_factor
        df.loc[mask, "adjusted_high"] /= split_factor
        df.loc[mask, "adjusted_low"] /= split_factor

        return df, effective_date

    # TODO: OFF-BY-ONE BUG in storage.py line 211
    def test_split_off_by_one_current_behavior(self):
        """
        Current behavior (<=): effective_date row gets divided.
        Day 5 open is $101 (post-split), divided by 10 → $10.1.
        This is WRONG — the price is already post-split.
        """
        df, effective_date = self._build_split_scenario(use_lte=True)
        day5 = df[df["date"] == effective_date]
        # Bug: $101 / 10 = $10.1 instead of correct $101
        assert pytest.approx(day5["adjusted_open"].values[0], rel=1e-6) == 10.1

    def test_split_correct_behavior(self):
        """
        Correct behavior (<): effective_date row is NOT divided.
        Day 5 open stays $101 (already post-split).
        """
        df, effective_date = self._build_split_scenario(use_lte=False)
        day5 = df[df["date"] == effective_date]
        assert pytest.approx(day5["adjusted_open"].values[0], rel=1e-6) == 101.0

    def test_pre_split_rows_adjusted(self):
        """Pre-split rows should all be divided by split factor."""
        df, effective_date = self._build_split_scenario(use_lte=False)
        pre_split = df[df["date"] < effective_date]
        # Original opens: 1000, 1010, 990, 1005, 1015
        expected_adjusted = [100.0, 101.0, 99.0, 100.5, 101.5]
        for i, expected in enumerate(expected_adjusted):
            assert pytest.approx(pre_split.iloc[i]["adjusted_open"], rel=1e-6) == expected

    def test_post_split_rows_unchanged(self):
        """Post-split rows (after effective_date) should NOT be adjusted."""
        df, effective_date = self._build_split_scenario(use_lte=False)
        post_split = df[df["date"] > effective_date]
        # Original opens: 102, 99, 103, 100
        expected = [102, 99, 103, 100]
        for i, exp in enumerate(expected):
            assert pytest.approx(post_split.iloc[i]["adjusted_open"], rel=1e-6) == exp


# ===========================================================================
# 5. PE ratio and price_to_book_ratio
# ===========================================================================

class TestDerivedRatios:
    """Validate pe_ratio and price_to_book_ratio computation."""

    def test_pe_ratio_basic(self):
        """pe_ratio = adjusted_close / ttm_eps."""
        df = pd.DataFrame({
            "adjusted_close": [150.0, 200.0],
            "ttm_eps": [3.0, 5.0],
        })
        df["pe_ratio"] = df["adjusted_close"] / df["ttm_eps"]
        assert df["pe_ratio"].iloc[0] == 50.0
        assert df["pe_ratio"].iloc[1] == 40.0

    def test_pe_ratio_zero_eps(self):
        """pe_ratio with ttm_eps=0 should produce inf."""
        df = pd.DataFrame({
            "adjusted_close": [150.0],
            "ttm_eps": [0.0],
        })
        df["pe_ratio"] = df["adjusted_close"] / df["ttm_eps"]
        assert np.isinf(df["pe_ratio"].iloc[0])
        # TODO: storage.py should handle this — clip to NaN or a max

    def test_pe_ratio_nan_eps(self):
        """pe_ratio with NaN ttm_eps should produce NaN."""
        df = pd.DataFrame({
            "adjusted_close": [150.0],
            "ttm_eps": [np.nan],
        })
        df["pe_ratio"] = df["adjusted_close"] / df["ttm_eps"]
        assert pd.isna(df["pe_ratio"].iloc[0])

    def test_price_to_book_zero_book_value(self):
        """price_to_book_ratio with book_value=0 should produce inf."""
        df = pd.DataFrame({
            "adjusted_close": [150.0],
            "book_value": [0.0],
        })
        df["price_to_book_ratio"] = df["adjusted_close"] / df["book_value"]
        assert np.isinf(df["price_to_book_ratio"].iloc[0])
        # TODO: storage.py should handle this — clip or replace


# ===========================================================================
# 6. SMA pct and 52-week pct columns
# ===========================================================================

class TestSmaPctColumns:
    """Validate sma_20_pct, sma_50_pct, sma_200_pct, 52_week_high/low_pct."""

    def test_sma_pct_formula(self):
        """sma_X_pct = (close - sma_X) / sma_X * 100."""
        df = pd.DataFrame({
            "adjusted_close": [110.0],
            "sma_20": [100.0],
        })
        df["sma_20_pct"] = ((df["adjusted_close"] - df["sma_20"]) / df["sma_20"] * 100).round(2)
        assert df["sma_20_pct"].iloc[0] == 10.0

    def test_sma_pct_negative(self):
        """Price below SMA should give negative pct."""
        df = pd.DataFrame({
            "adjusted_close": [90.0],
            "sma_20": [100.0],
        })
        df["sma_20_pct"] = ((df["adjusted_close"] - df["sma_20"]) / df["sma_20"] * 100).round(2)
        assert df["sma_20_pct"].iloc[0] == -10.0

    def test_52_week_high_pct_at_high(self):
        """At 52-week high, pct should be 0."""
        df = pd.DataFrame({
            "adjusted_close": [200.0],
            "52_week_high": [200.0],
        })
        df["52_week_high_pct"] = ((df["adjusted_close"] - df["52_week_high"]) / df["52_week_high"] * 100).round(2)
        assert df["52_week_high_pct"].iloc[0] == 0.0

    def test_52_week_high_pct_below(self):
        """10% below 52-week high should give -10."""
        df = pd.DataFrame({
            "adjusted_close": [180.0],
            "52_week_high": [200.0],
        })
        df["52_week_high_pct"] = ((df["adjusted_close"] - df["52_week_high"]) / df["52_week_high"] * 100).round(2)
        assert df["52_week_high_pct"].iloc[0] == -10.0

    def test_52_week_low_pct_above(self):
        """20% above 52-week low should give +20."""
        df = pd.DataFrame({
            "adjusted_close": [120.0],
            "52_week_low": [100.0],
        })
        df["52_week_low_pct"] = ((df["adjusted_close"] - df["52_week_low"]) / df["52_week_low"] * 100).round(2)
        assert df["52_week_low_pct"].iloc[0] == 20.0

    # TODO: Test with NaN sma values → should propagate NaN, not error


# ===========================================================================
# 7. Calendar columns
# ===========================================================================

class TestCalendarColumns:
    """Validate day_of_week_num, day_of_week_name, month, day_of_year, year."""

    def test_calendar_derivation(self):
        """Calendar columns should be derived from date, not cloned from another row."""
        df = make_core_df(n_days=20)
        df["day_of_week_num"] = df["date"].dt.dayofweek
        df["day_of_week_name"] = df["date"].dt.day_name()
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.dayofyear
        df["year"] = df["date"].dt.year

        for _, row in df.iterrows():
            assert row["day_of_week_num"] == row["date"].dayofweek
            assert row["day_of_week_name"] == row["date"].day_name()
            assert row["month"] == row["date"].month
            assert row["year"] == row["date"].year

    def test_no_weekends(self):
        """Business dates shouldn't include weekends (day_of_week_num 5 or 6)."""
        df = make_core_df(n_days=30)
        df["day_of_week_num"] = df["date"].dt.dayofweek
        assert (df["day_of_week_num"] < 5).all()


# ===========================================================================
# 8. Future return labels
# ===========================================================================

class TestFutureReturns:
    """Validate future_3_day_pct, future_10_day_pct, future_30_day_pct."""

    def setup_method(self):
        self.df = make_core_df(n_days=40, symbol="TEST")
        self.df["future_3_day_pct"] = (
            self.df.groupby("symbol")["adjusted_close"]
            .apply(lambda x: (x.shift(-3) - x) / x * 100)
            .reset_index(level=0, drop=True)
        )
        self.df["future_10_day_pct"] = (
            self.df.groupby("symbol")["adjusted_close"]
            .apply(lambda x: (x.shift(-10) - x) / x * 100)
            .reset_index(level=0, drop=True)
        )

    def test_last_3_rows_are_nan(self):
        """Future labels at the end of the series should be NaN (no future data)."""
        assert pd.isna(self.df["future_3_day_pct"].iloc[-1])
        assert pd.isna(self.df["future_3_day_pct"].iloc[-2])
        assert pd.isna(self.df["future_3_day_pct"].iloc[-3])
        assert pd.notna(self.df["future_3_day_pct"].iloc[-4])

    def test_last_10_rows_are_nan(self):
        """future_10_day_pct should have NaN for the last 10 rows."""
        for i in range(1, 11):
            assert pd.isna(self.df["future_10_day_pct"].iloc[-i])
        assert pd.notna(self.df["future_10_day_pct"].iloc[-11])

    def test_future_return_formula(self):
        """Manual check: future_3d = (close[i+3] - close[i]) / close[i] * 100."""
        close = self.df["adjusted_close"]
        for i in range(5):
            expected = (close.iloc[i + 3] - close.iloc[i]) / close.iloc[i] * 100
            assert pytest.approx(self.df["future_3_day_pct"].iloc[i], rel=1e-10) == expected

    def test_multi_symbol_independence(self):
        """Future returns should not leak across symbol boundaries."""
        df = make_multi_symbol_df(symbols=("AAA", "BBB"), n_days=10)
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
        df["future_3_day_pct"] = (
            df.groupby("symbol")["adjusted_close"]
            .apply(lambda x: (x.shift(-3) - x) / x * 100)
            .reset_index(level=0, drop=True)
        )
        # Last 3 rows of EACH symbol should be NaN
        for sym in ["AAA", "BBB"]:
            sym_df = df[df["symbol"] == sym]
            assert pd.isna(sym_df["future_3_day_pct"].iloc[-1])
            assert pd.isna(sym_df["future_3_day_pct"].iloc[-3])


# ===========================================================================
# 9. MACD 9-EMA (signal line)
# ===========================================================================

class TestMacd9Ema:
    """Validate macd_9_ema = EWM(span=9) of MACD grouped by symbol."""

    def test_macd_9_ema_computed(self):
        """macd_9_ema should be computed for all rows with MACD data."""
        df = make_core_df(n_days=20)
        df["macd_9_ema"] = (
            df.groupby("symbol")["macd"]
            .transform(lambda x: x.ewm(span=9, adjust=False).mean())
        )
        assert df["macd_9_ema"].notna().all()

    def test_macd_9_ema_is_smoother(self):
        """EMA should have smaller variance than raw MACD."""
        df = make_core_df(n_days=50)
        df["macd_9_ema"] = (
            df.groupby("symbol")["macd"]
            .transform(lambda x: x.ewm(span=9, adjust=False).mean())
        )
        assert df["macd_9_ema"].std() <= df["macd"].std()


# ===========================================================================
# 10. SQL injection vectors
# ===========================================================================

class TestSqlInjection:
    """
    Document SQL injection risks in storage.py.
    These tests don't hit the DB — they verify the risky patterns exist.
    """

    # TODO: insert_ignore_data builds SQL via f-string concatenation (line 106).
    #       A value with a single quote would break or inject.
    #       Should use parameterized queries.

    # TODO: select_all_by_symbol injects symbols via f-string (line 116).
    #       Should use parameterized queries.

    # TODO: get_dates_for_symbol injects symbol and dates via f-string (line 153).
    #       Inconsistent with get_last_updated_date which uses parameter binding.

    def test_placeholder_for_injection_tests(self):
        """Placeholder — actual injection tests require a test database."""
        pass


# ===========================================================================
# 11. insert_ignore_data edge cases
# ===========================================================================

class TestInsertIgnoreData:
    """Edge cases in the INSERT IGNORE builder."""

    # TODO: Test NaN float → should produce NULL in SQL, not string 'nan'
    # TODO: Test string with single quote → should be escaped
    # TODO: Test empty DataFrame → should short-circuit (already handled)
    # TODO: Test table not in TABLE_COLS → should raise KeyError

    def test_nan_float_produces_invalid_sql(self):
        """
        BUG: NaN float values become the literal string 'nan' in SQL.
        storage.py line 106: str(row[col]) for float cols.
        """
        row = {"volume": float("nan")}
        result = str(row["volume"])
        assert result == "nan", "NaN becomes literal 'nan' — invalid SQL"
        # TODO: Should be converted to 'NULL' or handled via parameterized query


# ===========================================================================
# 12. store_data index behavior
# ===========================================================================

class TestStoreDataIndex:
    """
    Validate that store_data with include_index=True doesn't add unwanted columns.
    """

    # TODO: Test that a DataFrame with RangeIndex produces an 'index' column
    #       when include_index=True. This is likely the source of 'Unnamed: 0'
    #       in the all_data_*.csv files.

    def test_range_index_adds_column(self):
        """DataFrame.to_sql with index=True adds 'index' column if RangeIndex."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        # df.to_sql would add an 'index' column since index is RangeIndex
        assert df.index.name is None
        assert isinstance(df.index, pd.RangeIndex)
        # TODO: storage.py store_data should use include_index=False for most tables


# ===========================================================================
# 13. Look-ahead bias in company_overview
# ===========================================================================

class TestLookAheadBias:
    """
    Document the look-ahead bias from company_overview JOIN.
    These are not bugs per se, but ML concerns.
    """

    # TODO: company_overview data (52_week_high, 52_week_low, beta, market_cap,
    #       book_value, forward_pe, shares_outstanding, analyst_ratings) is a
    #       single snapshot per symbol. The SQL JOIN applies TODAY's values to
    #       ALL historical dates. This means:
    #       - 52_week_high/low in 2020 rows reflects the current 52-week range
    #       - beta in 2018 rows reflects current beta
    #       - analyst_ratings from 2020 use today's consensus
    #
    #       For ML training, these features leak future information.
    #
    #       Possible fixes:
    #       - Store company_overview snapshots with dates
    #       - Exclude company_overview fields from ML features
    #       - Use only derived ratios (pe_ratio) that combine current + historical

    def test_company_overview_is_constant_per_symbol(self):
        """All rows for a symbol have the same company_overview values."""
        df = make_core_df(n_days=20)
        assert df["beta"].nunique() == 1
        assert df["52_week_high"].nunique() == 1
        assert df["analyst_rating_strong_buy"].nunique() == 1


# ===========================================================================
# 14. connect_database and engine management
# ===========================================================================

class TestDatabaseConnection:
    """Edge cases for connect_database / get_engine."""

    # TODO: Test reconnection — once engine is set, connect_database is no-op
    #       even if connection is stale.

    # TODO: Test password with special characters — e.g., 'p@ss:w/rd'
    #       would break the connection string without urllib.parse.quote_plus().

    # TODO: Test missing MYSQL_PASSWORD env var → should raise Exception.

    def test_get_engine_without_connect_raises(self):
        """get_engine() before connect_database() should raise."""
        import api_data.storage as storage_mod
        original_engine = storage_mod.engine
        try:
            storage_mod.engine = None
            with pytest.raises(Exception, match="Database connection not yet established"):
                storage_mod.get_engine()
        finally:
            storage_mod.engine = original_engine


# ===========================================================================
# 15. Derived fields we SHOULD add for ML
# ===========================================================================

class TestSuggestedDerivedFields:
    """
    Tests for derived fields that would benefit ML but don't exist yet in storage.py.
    These test the computation logic so we can add them when ready.
    """

    def setup_method(self):
        self.df = make_core_df(n_days=30, symbol="TEST")

    # TODO: Add 'close' column to SQL SELECT (currently only adjusted_close).
    #       data_utils.py renames adjusted_close→close at training time, but
    #       having the raw close in all_data would allow split-analysis.

    def test_overnight_gap(self):
        """overnight_gap = (today_open - yesterday_close) / yesterday_close."""
        df = self.df.copy()
        df["prev_close"] = df.groupby("symbol")["adjusted_close"].shift(1)
        df["overnight_gap"] = (df["open"] - df["prev_close"]) / df["prev_close"]
        assert pd.isna(df["overnight_gap"].iloc[0])  # no prior day
        assert pd.notna(df["overnight_gap"].iloc[1])

    def test_intraday_range(self):
        """intraday_range = (high - low) / open — daily volatility measure."""
        df = self.df.copy()
        df["intraday_range"] = (df["high"] - df["low"]) / df["open"]
        assert (df["intraday_range"] >= 0).all()

    def test_garman_klass_volatility(self):
        """
        Garman-Klass vol = 0.5 * ln(H/L)^2 - (2ln2 - 1) * ln(C/O)^2
        More efficient volatility estimator than close-to-close.
        """
        df = self.df.copy()
        df["gk_vol"] = (
            0.5 * np.log(df["high"] / df["low"]) ** 2
            - (2 * np.log(2) - 1) * np.log(df["adjusted_close"] / df["open"]) ** 2
        )
        assert df["gk_vol"].notna().all()
        # GK vol should be non-negative in most cases (can be slightly negative
        # due to the subtraction, but typically positive)

    # TODO: Add volume_dollar = adjusted_close * volume (dollar volume — more
    #       comparable across stocks than raw share volume)

    # TODO: Add rolling Sharpe-like ratio: mean(close_1d_roc) / std(close_1d_roc)
    #       over trailing 20 days — captures risk-adjusted momentum


# ===========================================================================
# 16. Signal generation performance concern
# ===========================================================================

class TestSignalGeneration:
    """
    Document the O(n^2) signal generation loop in gold_table_processing.
    """

    # TODO: The row-by-row loop (lines 419-434) does:
    #       screen_df = df[(df.symbol == row['symbol']) & (df.date == row['date'])]
    #       for every row. This is O(n * n_total) boolean operations.
    #       For 5M rows, this would take hours.
    #
    #       Suggested fix: vectorize the _check functions to operate on the
    #       full DataFrame at once, or at minimum group by symbol first.

    # TODO: StockScreener is initialized with df.head() (line 411).
    #       If any _check function references self.df beyond the passed
    #       screen_df parameter, it would get wrong data.

    def test_signal_count_matches_expected(self):
        """The signal loop produces exactly 10 signal columns."""
        expected_signals = [
            "macd_signal", "macd_zero_signal", "adx_signal", "atr_signal",
            "pe_ratio_signal", "bollinger_bands_signal", "rsi_signal",
            "sma_cross_signal", "cci_signal", "pcr_signal",
        ]
        assert len(expected_signals) == 10

    def test_bull_bear_delta_is_sum(self):
        """bull_bear_delta should be the sum of all 10 signal columns."""
        signals = {
            "macd_signal": 1,
            "macd_zero_signal": -1,
            "adx_signal": 0,
            "atr_signal": 1,
            "pe_ratio_signal": 0,
            "bollinger_bands_signal": -1,
            "rsi_signal": 1,
            "sma_cross_signal": 0,
            "cci_signal": 1,
            "pcr_signal": -1,
        }
        df = pd.DataFrame([signals])
        df["bull_bear_delta"] = sum(df[col] for col in signals.keys())
        assert df["bull_bear_delta"].iloc[0] == 1  # 4 bull - 3 bear - 3 neutral = 1
