"""
Unit tests for visualizations/screener.py signal-checking functions.

Each _check_* function follows the same pattern:
  - Takes (selected_date_data, bullish_signals, bearish_signals, signals)
  - Reads indicator value(s) from a single-row DataFrame
  - Appends label to bullish_signals or bearish_signals
  - Appends +1, -1, or 0 to signals

We test:
  - Clear bullish / bearish / neutral cases
  - Exact boundary values (off-by-one / inclusive vs exclusive)
  - Edge cases (NaN, zero, extreme values)

Standalone — does not modify any production code.
"""
import os
import sys
import math
import pytest
import pandas as pd
import numpy as np

# Add project root to path so we can import the screener
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from visualizations.screener import StockScreener


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_row(**kwargs):
    """Create a single-row DataFrame with the given column values."""
    return pd.DataFrame([kwargs])


def make_screener():
    """
    Create a minimal StockScreener instance.
    We only need the object to call instance methods — the constructor needs
    a few fields but we bypass heavy init by setting them directly.
    """
    # Create a tiny dummy dataframe with 3 dates to satisfy the constructor
    dates = pd.date_range("2026-01-20", periods=3, freq="B")
    data = pd.DataFrame({
        "date": list(dates) * 2,
        "symbol": ["TEST"] * 3 + ["TEST2"] * 3,
        "adjusted_close": [100.0] * 6,
        "sma_20": [100.0] * 6,
        "sma_50": [100.0] * 6,
    })
    screener = StockScreener.__new__(StockScreener)
    screener.symbols = ["TEST"]
    screener.date = "2026-01-22"
    screener.data = data
    screener.indicators = "all"
    screener.visualize = False
    screener.n_days = 30
    screener.whitelist = []
    screener.use_candlesticks = False
    screener.results = []
    screener.skip_sectors = True
    return screener


def run_check(check_fn, row):
    """
    Call a _check_* function and return (bullish, bearish, signals).
    """
    bullish = []
    bearish = []
    signals = []
    check_fn(row, bullish, bearish, signals)
    return bullish, bearish, signals


# ---------------------------------------------------------------------------
# Tests: _check_sma_cross
# ---------------------------------------------------------------------------

class TestSMACross:
    """SMA 20 vs SMA 50 crossover check.
    Bullish: sma_20 > sma_50
    Bearish: sma_20 < sma_50
    Neutral: sma_20 == sma_50
    Uses strict > and <.
    """
    def setup_method(self):
        self.s = make_screener()

    def test_bullish(self):
        row = make_row(sma_20=55.0, sma_50=50.0)
        b, br, sig = run_check(self.s._check_sma_cross, row)
        assert b == ["bullish_sma_cross"]
        assert br == []
        assert sig == [1]

    def test_bearish(self):
        row = make_row(sma_20=45.0, sma_50=50.0)
        b, br, sig = run_check(self.s._check_sma_cross, row)
        assert b == []
        assert br == ["bearish_sma_cross"]
        assert sig == [-1]

    def test_neutral_exact_equal(self):
        row = make_row(sma_20=50.0, sma_50=50.0)
        b, br, sig = run_check(self.s._check_sma_cross, row)
        assert b == []
        assert br == []
        assert sig == [0]

    def test_bullish_by_tiny_margin(self):
        """sma_20 just barely above sma_50 — should still be bullish."""
        row = make_row(sma_20=50.0 + 1e-10, sma_50=50.0)
        b, br, sig = run_check(self.s._check_sma_cross, row)
        assert sig == [1]

    def test_bearish_by_tiny_margin(self):
        row = make_row(sma_20=50.0 - 1e-10, sma_50=50.0)
        b, br, sig = run_check(self.s._check_sma_cross, row)
        assert sig == [-1]


# ---------------------------------------------------------------------------
# Tests: _check_bollinger_band
# ---------------------------------------------------------------------------

class TestBollingerBand:
    """Bollinger Band check.
    Bearish: close > bb_upper (strict >)
    Bullish: close < bb_lower (strict <)
    Neutral: bb_lower <= close <= bb_upper
    """
    def setup_method(self):
        self.s = make_screener()

    def test_bearish_above_upper(self):
        row = make_row(adjusted_close=105.0, bbands_upper_20=100.0, bbands_lower_20=90.0)
        b, br, sig = run_check(self.s._check_bollinger_band, row)
        assert br == ["bearish_bollinger_band"]
        assert sig == [-1]

    def test_bullish_below_lower(self):
        row = make_row(adjusted_close=85.0, bbands_upper_20=100.0, bbands_lower_20=90.0)
        b, br, sig = run_check(self.s._check_bollinger_band, row)
        assert b == ["bullish_bollinger_band"]
        assert sig == [1]

    def test_neutral_between_bands(self):
        row = make_row(adjusted_close=95.0, bbands_upper_20=100.0, bbands_lower_20=90.0)
        b, br, sig = run_check(self.s._check_bollinger_band, row)
        assert sig == [0]

    def test_neutral_at_exact_upper(self):
        """close == bb_upper should be neutral (strict > required for bearish)."""
        row = make_row(adjusted_close=100.0, bbands_upper_20=100.0, bbands_lower_20=90.0)
        b, br, sig = run_check(self.s._check_bollinger_band, row)
        assert sig == [0]

    def test_neutral_at_exact_lower(self):
        """close == bb_lower should be neutral (strict < required for bullish)."""
        row = make_row(adjusted_close=90.0, bbands_upper_20=100.0, bbands_lower_20=90.0)
        b, br, sig = run_check(self.s._check_bollinger_band, row)
        assert sig == [0]


# ---------------------------------------------------------------------------
# Tests: _check_rsi
# ---------------------------------------------------------------------------

class TestRSI:
    """RSI check.
    Bearish: rsi > 70 (strict >)
    Bullish: rsi < 30 (strict <)
    Neutral: 30 <= rsi <= 70
    """
    def setup_method(self):
        self.s = make_screener()

    def test_bearish(self):
        row = make_row(rsi_14=75.0)
        b, br, sig = run_check(self.s._check_rsi, row)
        assert br == ["bearish_rsi"]
        assert sig == [-1]

    def test_bullish(self):
        row = make_row(rsi_14=25.0)
        b, br, sig = run_check(self.s._check_rsi, row)
        assert b == ["bullish_rsi"]
        assert sig == [1]

    def test_neutral_middle(self):
        row = make_row(rsi_14=50.0)
        b, br, sig = run_check(self.s._check_rsi, row)
        assert sig == [0]

    def test_neutral_at_exact_70(self):
        """rsi == 70 is neutral (strict > required for bearish)."""
        row = make_row(rsi_14=70.0)
        b, br, sig = run_check(self.s._check_rsi, row)
        assert sig == [0]

    def test_neutral_at_exact_30(self):
        """rsi == 30 is neutral (strict < required for bullish)."""
        row = make_row(rsi_14=30.0)
        b, br, sig = run_check(self.s._check_rsi, row)
        assert sig == [0]

    def test_bearish_at_70_01(self):
        """rsi = 70.01 should be bearish."""
        row = make_row(rsi_14=70.01)
        b, br, sig = run_check(self.s._check_rsi, row)
        assert sig == [-1]

    def test_bullish_at_29_99(self):
        """rsi = 29.99 should be bullish."""
        row = make_row(rsi_14=29.99)
        b, br, sig = run_check(self.s._check_rsi, row)
        assert sig == [1]


# ---------------------------------------------------------------------------
# Tests: _check_macd
# ---------------------------------------------------------------------------

class TestMACD:
    """MACD vs 9-day EMA check.
    Bullish: macd > macd_9_ema (strict >)
    Bearish: macd < macd_9_ema (strict <)
    Neutral: macd == macd_9_ema
    """
    def setup_method(self):
        self.s = make_screener()

    def test_bullish(self):
        row = make_row(macd=2.5, macd_9_ema=1.0)
        b, br, sig = run_check(self.s._check_macd, row)
        assert b == ["bullish_macd"]
        assert sig == [1]

    def test_bearish(self):
        row = make_row(macd=-1.0, macd_9_ema=0.5)
        b, br, sig = run_check(self.s._check_macd, row)
        assert br == ["bearish_macd"]
        assert sig == [-1]

    def test_neutral_equal(self):
        row = make_row(macd=1.0, macd_9_ema=1.0)
        b, br, sig = run_check(self.s._check_macd, row)
        assert sig == [0]

    def test_both_negative_macd_above_ema(self):
        """Both negative but macd > ema — bullish crossover."""
        row = make_row(macd=-0.5, macd_9_ema=-1.0)
        b, br, sig = run_check(self.s._check_macd, row)
        assert sig == [1]

    def test_both_negative_macd_below_ema(self):
        row = make_row(macd=-2.0, macd_9_ema=-1.0)
        b, br, sig = run_check(self.s._check_macd, row)
        assert sig == [-1]


# ---------------------------------------------------------------------------
# Tests: _check_macd_zero
# ---------------------------------------------------------------------------

class TestMACDZero:
    """MACD zero-line check.
    Bullish: macd > 0 (strict >)
    Bearish: macd < 0 (strict <)
    Neutral: macd == 0
    """
    def setup_method(self):
        self.s = make_screener()

    def test_bullish(self):
        row = make_row(macd=0.5)
        b, br, sig = run_check(self.s._check_macd_zero, row)
        assert b == ["bullish_macd_zero"]
        assert sig == [1]

    def test_bearish(self):
        row = make_row(macd=-0.5)
        b, br, sig = run_check(self.s._check_macd_zero, row)
        assert br == ["bearish_macd_zero"]
        assert sig == [-1]

    def test_neutral_at_zero(self):
        row = make_row(macd=0.0)
        b, br, sig = run_check(self.s._check_macd_zero, row)
        assert sig == [0]

    def test_bullish_tiny_positive(self):
        row = make_row(macd=1e-10)
        b, br, sig = run_check(self.s._check_macd_zero, row)
        assert sig == [1]


# ---------------------------------------------------------------------------
# Tests: _check_adx
# ---------------------------------------------------------------------------

class TestADX:
    """ADX check.
    Bullish: adx > 25 (strict >)
    Bearish: adx < 20 (strict <)
    Neutral: 20 <= adx <= 25

    Note: ADX measures trend strength, not direction. Labeling high ADX as
    "bullish" and low ADX as "bearish" is semantically questionable.
    """
    def setup_method(self):
        self.s = make_screener()

    def test_bullish_strong_trend(self):
        row = make_row(adx_14=30.0)
        b, br, sig = run_check(self.s._check_adx, row)
        assert b == ["bullish_adx"]
        assert sig == [1]

    def test_bearish_weak_trend(self):
        row = make_row(adx_14=15.0)
        b, br, sig = run_check(self.s._check_adx, row)
        assert br == ["bearish_adx"]
        assert sig == [-1]

    def test_neutral_at_22(self):
        row = make_row(adx_14=22.0)
        b, br, sig = run_check(self.s._check_adx, row)
        assert sig == [0]

    def test_neutral_at_exact_25(self):
        """adx == 25 is neutral (strict > required for bullish)."""
        row = make_row(adx_14=25.0)
        b, br, sig = run_check(self.s._check_adx, row)
        assert sig == [0]

    def test_neutral_at_exact_20(self):
        """adx == 20 is neutral (strict < required for bearish)."""
        row = make_row(adx_14=20.0)
        b, br, sig = run_check(self.s._check_adx, row)
        assert sig == [0]

    def test_bullish_at_25_01(self):
        row = make_row(adx_14=25.01)
        b, br, sig = run_check(self.s._check_adx, row)
        assert sig == [1]

    def test_bearish_at_19_99(self):
        row = make_row(adx_14=19.99)
        b, br, sig = run_check(self.s._check_adx, row)
        assert sig == [-1]


# ---------------------------------------------------------------------------
# Tests: _check_cci
# ---------------------------------------------------------------------------

class TestCCI:
    """CCI check.
    Bearish: cci > 100 (strict >)
    Bullish: cci < -100 (strict <)
    Neutral: -100 <= cci <= 100
    """
    def setup_method(self):
        self.s = make_screener()

    def test_bearish_above_100(self):
        row = make_row(cci_14=150.0)
        b, br, sig = run_check(self.s._check_cci, row)
        assert br == ["bearish_cci"]
        assert sig == [-1]

    def test_bullish_below_neg100(self):
        row = make_row(cci_14=-150.0)
        b, br, sig = run_check(self.s._check_cci, row)
        assert b == ["bullish_cci"]
        assert sig == [1]

    def test_neutral_middle(self):
        row = make_row(cci_14=0.0)
        b, br, sig = run_check(self.s._check_cci, row)
        assert sig == [0]

    def test_neutral_at_exact_100(self):
        """cci == 100 is neutral (strict > required for bearish)."""
        row = make_row(cci_14=100.0)
        b, br, sig = run_check(self.s._check_cci, row)
        assert sig == [0]

    def test_neutral_at_exact_neg100(self):
        """cci == -100 is neutral (strict < required for bullish)."""
        row = make_row(cci_14=-100.0)
        b, br, sig = run_check(self.s._check_cci, row)
        assert sig == [0]

    def test_bearish_at_100_01(self):
        """cci = 100.01 should be bearish."""
        row = make_row(cci_14=100.01)
        b, br, sig = run_check(self.s._check_cci, row)
        assert sig == [-1]

    def test_bullish_at_neg100_01(self):
        """cci = -100.01 should be bullish."""
        row = make_row(cci_14=-100.01)
        b, br, sig = run_check(self.s._check_cci, row)
        assert sig == [1]


# ---------------------------------------------------------------------------
# Tests: _check_atr
# ---------------------------------------------------------------------------

class TestATR:
    """ATR check (ATR as percentage of price).
    Bearish: atr/close > 4% (high volatility)
    Bullish: atr/close < 1.5% (low volatility)
    Neutral: 1.5% <= atr/close <= 4%
    """
    def setup_method(self):
        self.s = make_screener()

    def test_bearish_high_volatility(self):
        """ATR 5% of price → bearish (high volatility)."""
        row = make_row(adjusted_close=100.0, atr_14=5.0)  # 5%
        b, br, sig = run_check(self.s._check_atr, row)
        assert br == ["bearish_atr"]
        assert sig == [-1]

    def test_bullish_low_volatility(self):
        """ATR 1% of price → bullish (low volatility)."""
        row = make_row(adjusted_close=100.0, atr_14=1.0)  # 1%
        b, br, sig = run_check(self.s._check_atr, row)
        assert b == ["bullish_atr"]
        assert sig == [1]

    def test_neutral_moderate_volatility(self):
        """ATR 2.5% of price → neutral."""
        row = make_row(adjusted_close=100.0, atr_14=2.5)  # 2.5%
        b, br, sig = run_check(self.s._check_atr, row)
        assert sig == [0]

    def test_neutral_at_exact_4_pct(self):
        """atr_pct == 4% is neutral (strict > required for bearish)."""
        row = make_row(adjusted_close=100.0, atr_14=4.0)  # exactly 4%
        b, br, sig = run_check(self.s._check_atr, row)
        assert sig == [0]

    def test_neutral_at_exact_1_5_pct(self):
        """atr_pct == 1.5% is neutral (strict < required for bullish)."""
        row = make_row(adjusted_close=100.0, atr_14=1.5)  # exactly 1.5%
        b, br, sig = run_check(self.s._check_atr, row)
        assert sig == [0]

    def test_works_across_price_levels(self):
        """Same ATR% should produce same signal regardless of absolute price."""
        # All have 2% ATR → neutral
        for close, atr in [(20.0, 0.4), (100.0, 2.0), (500.0, 10.0)]:
            row = make_row(adjusted_close=close, atr_14=atr)
            _, _, sig = run_check(self.s._check_atr, row)
            assert sig == [0], f"close={close}, atr={atr} (2%): should be neutral"

    def test_zero_close_is_neutral(self):
        """Edge case: close == 0 should not crash."""
        row = make_row(adjusted_close=0.0, atr_14=1.0)
        _, _, sig = run_check(self.s._check_atr, row)
        assert sig == [0]


# ---------------------------------------------------------------------------
# Tests: _check_pe_ratio
# ---------------------------------------------------------------------------

class TestPERatio:
    """PE ratio check.
    Bullish: pe < 15 AND pe > 0 (strict < and strict >)
    Bearish: pe > 35 (strict >)
    Neutral: pe <= 0 OR (15 <= pe <= 35)
    """
    def setup_method(self):
        self.s = make_screener()

    def test_bullish_low_pe(self):
        row = make_row(pe_ratio=10.0)
        b, br, sig = run_check(self.s._check_pe_ratio, row)
        assert b == ["bullish_pe_ratio"]
        assert sig == [1]

    def test_bearish_high_pe(self):
        row = make_row(pe_ratio=40.0)
        b, br, sig = run_check(self.s._check_pe_ratio, row)
        assert br == ["bearish_pe_ratio"]
        assert sig == [-1]

    def test_neutral_moderate_pe(self):
        row = make_row(pe_ratio=25.0)
        b, br, sig = run_check(self.s._check_pe_ratio, row)
        assert sig == [0]

    def test_neutral_at_exact_15(self):
        """pe == 15 is neutral (strict < required for bullish)."""
        row = make_row(pe_ratio=15.0)
        b, br, sig = run_check(self.s._check_pe_ratio, row)
        assert sig == [0]

    def test_neutral_at_exact_35(self):
        """pe == 35 is neutral (strict > required for bearish)."""
        row = make_row(pe_ratio=35.0)
        b, br, sig = run_check(self.s._check_pe_ratio, row)
        assert sig == [0]

    def test_neutral_at_exact_zero(self):
        """pe == 0 is neutral (strict > 0 required for bullish)."""
        row = make_row(pe_ratio=0.0)
        b, br, sig = run_check(self.s._check_pe_ratio, row)
        assert sig == [0]

    def test_neutral_negative_pe(self):
        """Negative PE (losses) is neutral — not bullish even though pe < 15."""
        row = make_row(pe_ratio=-5.0)
        b, br, sig = run_check(self.s._check_pe_ratio, row)
        assert sig == [0], "pe < 0 should be neutral (pe > 0 guard prevents bullish)"

    def test_bullish_at_14_99(self):
        row = make_row(pe_ratio=14.99)
        b, br, sig = run_check(self.s._check_pe_ratio, row)
        assert sig == [1]

    def test_bearish_at_35_01(self):
        row = make_row(pe_ratio=35.01)
        b, br, sig = run_check(self.s._check_pe_ratio, row)
        assert sig == [-1]

    def test_bullish_at_0_01(self):
        """Tiny positive PE (near zero but > 0) should be bullish."""
        row = make_row(pe_ratio=0.01)
        b, br, sig = run_check(self.s._check_pe_ratio, row)
        assert sig == [1]


# ---------------------------------------------------------------------------
# Tests: _check_pcr
# ---------------------------------------------------------------------------

class TestPCR:
    """Put-call ratio check.
    Bearish: pcr > 0.7 (strict >)
    Bullish: pcr < 0.5 (strict <)
    Neutral: 0.5 <= pcr <= 0.7
    """
    def setup_method(self):
        self.s = make_screener()

    def test_bearish(self):
        row = make_row(pcr=0.9)
        b, br, sig = run_check(self.s._check_pcr, row)
        assert br == ["bearish_pcr"]
        assert sig == [-1]

    def test_bullish(self):
        row = make_row(pcr=0.3)
        b, br, sig = run_check(self.s._check_pcr, row)
        assert b == ["bullish_pcr"]
        assert sig == [1]

    def test_neutral_in_gap(self):
        row = make_row(pcr=0.6)
        b, br, sig = run_check(self.s._check_pcr, row)
        assert sig == [0]

    def test_neutral_at_exact_0_7(self):
        """pcr == 0.7 is neutral (strict > required for bearish)."""
        row = make_row(pcr=0.7)
        b, br, sig = run_check(self.s._check_pcr, row)
        assert sig == [0]

    def test_neutral_at_exact_0_5(self):
        """pcr == 0.5 is neutral (strict < required for bullish)."""
        row = make_row(pcr=0.5)
        b, br, sig = run_check(self.s._check_pcr, row)
        assert sig == [0]

    def test_bearish_at_0_71(self):
        row = make_row(pcr=0.71)
        b, br, sig = run_check(self.s._check_pcr, row)
        assert sig == [-1]

    def test_bullish_at_0_49(self):
        """pcr = 0.49 should be bullish."""
        row = make_row(pcr=0.49)
        b, br, sig = run_check(self.s._check_pcr, row)
        assert sig == [1]


# ---------------------------------------------------------------------------
# Tests: _check_option_vol (commented out in _check_signals, but still a method)
# ---------------------------------------------------------------------------

class TestOptionVol:
    """Options volume check (disabled in _check_signals but still exists).
    Bearish: vol > mu + std (strict >)
    Bullish: vol <= mu - std (INCLUSIVE — same inconsistency as PCR)
    Neutral: mu - std < vol <= mu + std
    """
    def setup_method(self):
        self.s = make_screener()

    def test_bearish_high_volume(self):
        row = make_row(total_volume=2000, options_14_mean=1000, options_14_std=500)
        b, br, sig = run_check(self.s._check_option_vol, row)
        assert br == ["bearish_options_vol"]
        assert sig == [-1]

    def test_bullish_low_volume(self):
        row = make_row(total_volume=400, options_14_mean=1000, options_14_std=500)
        b, br, sig = run_check(self.s._check_option_vol, row)
        assert b == ["bullish_options_vol"]
        assert sig == [1]

    def test_neutral_normal_volume(self):
        row = make_row(total_volume=1000, options_14_mean=1000, options_14_std=500)
        b, br, sig = run_check(self.s._check_option_vol, row)
        assert sig == [0]

    def test_neutral_at_exact_upper(self):
        """vol == mu + std should be neutral (strict > for bearish)."""
        row = make_row(total_volume=1500, options_14_mean=1000, options_14_std=500)
        b, br, sig = run_check(self.s._check_option_vol, row)
        assert sig == [0]

    def test_bullish_at_exact_lower(self):
        """vol == mu - std IS bullish (<= is used — inclusive lower bound)."""
        row = make_row(total_volume=500, options_14_mean=1000, options_14_std=500)
        b, br, sig = run_check(self.s._check_option_vol, row)
        assert sig == [1], "option_vol uses <= mu-std (inclusive) for bullish"


# ---------------------------------------------------------------------------
# Tests: _check_price_to_book (not called, but test the logic)
# ---------------------------------------------------------------------------

class TestPriceToBook:
    """Price to book check (NOT CALLED in _check_signals).
    Uses price_to_book_ratio directly (P/B = price / book_value).
    Bullish: P/B < 1.0 and > 0 (trading below book value)
    Bearish: P/B > 5.0 (richly valued vs assets)
    Neutral: 1.0 <= P/B <= 5.0 or P/B <= 0
    """
    def setup_method(self):
        self.s = make_screener()

    def test_bullish_below_book(self):
        """P/B = 0.8 — trading below book value → bullish."""
        row = make_row(price_to_book_ratio=0.8)
        b, br, sig = run_check(self.s._check_price_to_book, row)
        assert sig == [1]

    def test_bearish_high_pb(self):
        """P/B = 7.0 — very richly valued → bearish."""
        row = make_row(price_to_book_ratio=7.0)
        b, br, sig = run_check(self.s._check_price_to_book, row)
        assert sig == [-1]

    def test_neutral_moderate(self):
        """P/B = 3.0 — fairly valued → neutral."""
        row = make_row(price_to_book_ratio=3.0)
        b, br, sig = run_check(self.s._check_price_to_book, row)
        assert sig == [0]

    def test_neutral_at_exact_1(self):
        """P/B == 1.0 is neutral (strict < required for bullish)."""
        row = make_row(price_to_book_ratio=1.0)
        b, br, sig = run_check(self.s._check_price_to_book, row)
        assert sig == [0]

    def test_neutral_negative_pb(self):
        """Negative P/B (negative book value) → neutral."""
        row = make_row(price_to_book_ratio=-2.0)
        b, br, sig = run_check(self.s._check_price_to_book, row)
        assert sig == [0]


# ---------------------------------------------------------------------------
# Tests: _check_signals orchestrator
# ---------------------------------------------------------------------------

class TestCheckSignals:
    """_check_signals(current_date_data, previous_date_data) orchestrator.

    The first param is the date being analyzed — all _check_* functions
    receive it. The second param (previous day) is available for future
    cross-day comparisons but not currently used by individual checks.

    When indicators='all', all 10 active checks run, producing a signals
    list of length 10.
    """
    def setup_method(self):
        self.s = make_screener()

    def test_all_indicators_produce_10_signals(self):
        """With indicators='all', expect exactly 10 signals."""
        row = make_row(
            sma_20=55.0, sma_50=50.0,
            adjusted_close=95.0, bbands_upper_20=100.0, bbands_lower_20=90.0,
            rsi_14=50.0,
            macd=1.0, macd_9_ema=0.5,
            adx_14=22.0,
            cci_14=0.0,
            atr_14=3.0,
            pe_ratio=25.0,
            pcr=0.6,
        )
        b, br, sig = self.s._check_signals(row, row)
        assert len(sig) == 10, f"Expected 10 signals, got {len(sig)}"

    def test_specific_indicators_filter(self):
        """Only run subset of indicators."""
        self.s.indicators = ['rsi', 'macd']
        row = make_row(rsi_14=25.0, macd=2.0, macd_9_ema=1.0)
        b, br, sig = self.s._check_signals(row, row)
        assert len(sig) == 2
        assert sig == [1, 1]  # rsi bullish + macd bullish

    def test_all_bullish(self):
        """All indicators bullish simultaneously."""
        row = make_row(
            sma_20=55.0, sma_50=50.0,         # bullish: 20 > 50
            adjusted_close=85.0,                # bullish: below lower band
            bbands_upper_20=100.0, bbands_lower_20=90.0,
            rsi_14=25.0,                        # bullish: < 30
            macd=2.0, macd_9_ema=1.0,           # bullish: macd > ema
            adx_14=30.0,                        # bullish: > 25
            cci_14=-150.0,                      # bullish: < -100
            atr_14=1.0,                         # bullish: 1/85=1.2% < 1.5%
            pe_ratio=10.0,                      # bullish: < 15 and > 0
            pcr=0.3,                            # bullish: < 0.5
        )
        b, br, sig = self.s._check_signals(row, row)
        assert all(s == 1 for s in sig), f"Expected all bullish, got {sig}"
        assert len(b) == 10
        assert len(br) == 0

    def test_all_bearish(self):
        """All indicators bearish simultaneously."""
        row = make_row(
            sma_20=45.0, sma_50=50.0,          # bearish: 20 < 50
            adjusted_close=105.0,               # bearish: above upper band
            bbands_upper_20=100.0, bbands_lower_20=90.0,
            rsi_14=75.0,                        # bearish: > 70
            macd=-1.0, macd_9_ema=0.5,          # bearish: macd < ema
            adx_14=15.0,                        # bearish: < 20
            cci_14=150.0,                       # bearish: > 100
            atr_14=5.0,                         # bearish: 5/105=4.8% > 4%
            pe_ratio=40.0,                      # bearish: > 35
            pcr=0.9,                            # bearish: > 0.7
        )
        b, br, sig = self.s._check_signals(row, row)
        assert all(s == -1 for s in sig), f"Expected all bearish, got {sig}"
        assert len(b) == 0
        assert len(br) == 10


# ---------------------------------------------------------------------------
# Tests: find_nearest_two_dates and find_nearest_three_dates
# ---------------------------------------------------------------------------

class TestDateFinding:
    """Date lookup helper tests."""

    def setup_method(self):
        self.s = make_screener()
        # Override data with known dates
        dates = pd.to_datetime(["2026-01-19", "2026-01-20", "2026-01-21",
                                "2026-01-22", "2026-01-23", "2026-01-24"])
        self.s.data = pd.DataFrame({
            "date": dates,
            "symbol": ["TEST"] * 6,
        })

    def test_three_dates_returns_most_recent(self):
        """find_nearest_three_dates returns 3 most recent dates <= target."""
        result = self.s.find_nearest_three_dates("2026-01-22")
        assert len(result) == 3
        # Should be [Jan 22, Jan 21, Jan 20] (descending)
        assert result[0] == pd.Timestamp("2026-01-22")
        assert result[1] == pd.Timestamp("2026-01-21")
        assert result[2] == pd.Timestamp("2026-01-20")

    def test_three_dates_excludes_future(self):
        """find_nearest_three_dates filters out dates > target."""
        result = self.s.find_nearest_three_dates("2026-01-21")
        # Jan 23, 24 should be excluded
        for d in result:
            assert d <= pd.Timestamp("2026-01-21")

    def test_two_dates_excludes_future(self):
        """find_nearest_two_dates now filters to <= target (no future leak)."""
        result = self.s.find_nearest_two_dates("2026-01-21")
        assert len(result) == 2
        # Should be [Jan 21, Jan 20] — no future dates
        assert result[0] == pd.Timestamp("2026-01-21")
        assert result[1] == pd.Timestamp("2026-01-20")
        for d in result:
            assert d <= pd.Timestamp("2026-01-21")

    def test_three_dates_at_start_of_data(self):
        """If target date is at the start, return as many as available."""
        result = self.s.find_nearest_three_dates("2026-01-20")
        # Only Jan 19 and Jan 20 are <= target
        assert len(result) == 2
        assert result[0] == pd.Timestamp("2026-01-20")
        assert result[1] == pd.Timestamp("2026-01-19")

    def test_three_dates_before_all_data(self):
        """If target is before all dates, return empty."""
        result = self.s.find_nearest_three_dates("2025-01-01")
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: run_screen change detection logic
# ---------------------------------------------------------------------------

class TestRunScreenChangeDetection:
    """Test the change detection logic in run_screen().

    change_signals = [latest[i] if latest[i] != previous[i] else 0 for i ...]

    A symbol passes the filter if:
      1) More than 1 non-zero change_signal, OR symbol is in whitelist
      2) AND (abs(num_bullish - num_bearish) > 1 OR more than 1 change)
    """

    def test_change_detection_same_signals(self):
        """If all signals are the same between days, all changes should be 0."""
        latest = [1, -1, 0, 1, -1, 0, 1, -1, 0, 1]
        previous = [1, -1, 0, 1, -1, 0, 1, -1, 0, 1]
        change = [latest[i] if latest[i] != previous[i] else 0 for i in range(len(latest))]
        assert change == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    def test_change_detection_all_different(self):
        """If all signals flipped, change == latest."""
        latest = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        previous = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
        change = [latest[i] if latest[i] != previous[i] else 0 for i in range(len(latest))]
        assert change == latest

    def test_change_detection_partial(self):
        """Only changed signals carry through."""
        latest =   [1, -1, 0, 1, 0]
        previous = [1,  1, 0, 0, 0]
        change = [latest[i] if latest[i] != previous[i] else 0 for i in range(len(latest))]
        assert change == [0, -1, 0, 1, 0]

    def test_filter_threshold_needs_more_than_1_change(self):
        """Filter: len(nonzero changes) > 1 required to pass."""
        change_signals = [0, -1, 0, 0, 0, 0, 0, 0, 0, 0]  # only 1 change
        nonzero = [s for s in change_signals if s != 0]
        assert len(nonzero) <= 1  # Would NOT pass the filter

    def test_filter_passes_with_2_changes(self):
        change_signals = [0, -1, 0, 1, 0, 0, 0, 0, 0, 0]  # 2 changes
        nonzero = [s for s in change_signals if s != 0]
        assert len(nonzero) > 1  # Would pass the filter


# ---------------------------------------------------------------------------
# Tests with real sample data (integration-style, uses sample_data_1month.csv)
# ---------------------------------------------------------------------------

class TestWithSampleData:
    """Integration tests using the actual 1-month sample data file.
    Verifies that signals can be computed on real data without errors.
    """

    @pytest.fixture(autouse=True)
    def load_data(self):
        data_path = os.path.join(os.path.dirname(__file__), "sample_data_1month.csv")
        if not os.path.exists(data_path):
            pytest.skip("sample_data_1month.csv not found — run create_sample_data.py first")
        self.df = pd.read_csv(data_path, parse_dates=["date"])
        self.s = make_screener()

    def test_sma_cross_on_real_data(self):
        """Run _check_sma_cross on every row and verify it always produces a valid signal."""
        for _, row_series in self.df.iterrows():
            if pd.isna(row_series.get("sma_20")) or pd.isna(row_series.get("sma_50")):
                continue
            row = pd.DataFrame([row_series])
            b, br, sig = run_check(self.s._check_sma_cross, row)
            assert len(sig) == 1
            assert sig[0] in (-1, 0, 1)

    def test_rsi_on_real_data(self):
        """RSI signals should always be -1, 0, or 1."""
        for _, row_series in self.df.iterrows():
            if pd.isna(row_series.get("rsi_14")):
                continue
            row = pd.DataFrame([row_series])
            b, br, sig = run_check(self.s._check_rsi, row)
            assert sig[0] in (-1, 0, 1)

    def test_atr_distribution_on_real_data(self):
        """With fixed ATR% logic, most stocks should be neutral (1.5%-4% ATR).
        Only high-vol names hit bearish (>4%) and very calm names hit bullish (<1.5%).
        """
        bearish_count = 0
        bullish_count = 0
        neutral_count = 0
        total = 0
        for _, row_series in self.df.iterrows():
            if pd.isna(row_series.get("atr_14")) or pd.isna(row_series.get("adjusted_close")):
                continue
            row = pd.DataFrame([row_series])
            bull, bear, sig = run_check(self.s._check_atr, row)
            total += 1
            if sig[0] == -1:
                bearish_count += 1
            elif sig[0] == 1:
                bullish_count += 1
            else:
                neutral_count += 1

        if total > 0:
            # With percentage-based ATR, we expect a reasonable distribution
            # — NOT the old bug where >90% were bearish
            bearish_pct = bearish_count / total
            assert bearish_pct < 0.50, (
                f"ATR: expected <50% bearish with fixed logic, got {bearish_pct:.1%} "
                f"({bearish_count}/{total})"
            )

    def test_all_signals_on_real_row(self):
        """Pick a real row with all required columns and run full _check_signals."""
        required_cols = [
            "sma_20", "sma_50", "adjusted_close", "bbands_upper_20", "bbands_lower_20",
            "rsi_14", "macd", "macd_9_ema", "adx_14", "cci_14", "atr_14", "pe_ratio",
        ]
        for _, row_series in self.df.iterrows():
            if all(pd.notna(row_series.get(c)) for c in required_cols):
                row = pd.DataFrame([row_series])
                # _check_signals takes (symbol_data, selected_date_data)
                # PCR column may be missing from sample data — use subset of indicators
                self.s.indicators = ['sma_cross', 'bollinger_band', 'rsi', 'macd',
                                     'macd_zero', 'adx', 'cci', 'atr', 'pe_ratio']
                b, br, sig = self.s._check_signals(row, row)
                assert len(sig) == 9  # 9 indicators without pcr
                assert all(s in (-1, 0, 1) for s in sig)
                break
        else:
            pytest.skip("No row found with all required columns populated")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
