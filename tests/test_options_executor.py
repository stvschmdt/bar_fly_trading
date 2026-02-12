#!/usr/bin/env python
"""
Tests for ibkr/options_executor.py

Tests:
1. select_option_contract — strike selection logic (call OTM, put OTM)
2. validate_option — spread, liquidity, and bid/ask guardrails
3. execute_option_signal — dry-run end-to-end
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ibkr.options_executor import (
    select_option_contract,
    validate_option,
    execute_option_signal,
    MAX_SPREAD_PCT,
    MIN_OPEN_INTEREST,
    MIN_VOLUME,
)


def _make_options_df(current_price=150.0):
    """Create a realistic options DataFrame for testing.

    Bid-ask spread is ~1% of mid to pass the 3% guardrail.
    """
    rows = []
    expiration = '2026-03-20'
    for strike in [145.0, 147.5, 150.0, 152.5, 155.0]:
        # Calls — ITM calls have more value, OTM have less but still reasonable
        itm = strike < current_price
        mid_call = max(current_price - strike, 1.0) + 3.0
        half_spread = round(mid_call * 0.005, 2)  # 0.5% each side = 1% total
        rows.append({
            'strike': strike,
            'type': 'call',
            'bid': round(mid_call - half_spread, 2),
            'ask': round(mid_call + half_spread, 2),
            'mid': round(mid_call, 2),
            'last': round(mid_call, 2),
            'volume': 500,
            'open_interest': 2000,
            'implied_volatility': 0.30,
            'delta': 0.6 if itm else 0.4,
            'gamma': 0.03,
            'theta': -0.05,
            'vega': 0.15,
            'expiration': expiration,
        })
        # Puts — OTM puts have more value when strike > price
        mid_put = max(strike - current_price, 1.0) + 3.0
        half_spread = round(mid_put * 0.005, 2)
        rows.append({
            'strike': strike,
            'type': 'put',
            'bid': round(mid_put - half_spread, 2),
            'ask': round(mid_put + half_spread, 2),
            'mid': round(mid_put, 2),
            'last': round(mid_put, 2),
            'volume': 300,
            'open_interest': 1500,
            'implied_volatility': 0.32,
            'delta': -0.4 if not itm else -0.6,
            'gamma': 0.03,
            'theta': -0.04,
            'vega': 0.14,
            'expiration': expiration,
        })
    return pd.DataFrame(rows)


class TestSelectOptionContract(unittest.TestCase):
    """Test strike selection logic."""

    @patch('ibkr.options_executor.get_options_snapshot')
    def test_buy_signal_selects_call_otm(self, mock_snapshot):
        """BUY signal should pick the first call strike above current price."""
        current_price = 150.0
        quote = {'price': current_price, 'symbol': 'AAPL'}
        df = _make_options_df(current_price)
        # Filter to calls only (as get_options_snapshot would with option_type='call')
        mock_snapshot.return_value = (quote, df[df['type'] == 'call'].reset_index(drop=True))

        option, q, _ = select_option_contract('AAPL', 'BUY', current_price)

        self.assertIsNotNone(option)
        self.assertGreater(option['strike'], current_price)
        self.assertEqual(option['strike'], 152.5)  # First strike above 150

    @patch('ibkr.options_executor.get_options_snapshot')
    def test_sell_signal_selects_put_otm(self, mock_snapshot):
        """SELL signal should pick the first put strike below current price."""
        current_price = 150.0
        quote = {'price': current_price, 'symbol': 'AAPL'}
        df = _make_options_df(current_price)
        mock_snapshot.return_value = (quote, df[df['type'] == 'put'].reset_index(drop=True))

        option, q, _ = select_option_contract('AAPL', 'SELL', current_price)

        self.assertIsNotNone(option)
        self.assertLess(option['strike'], current_price)
        self.assertEqual(option['strike'], 147.5)  # First strike below 150

    @patch('ibkr.options_executor.get_options_snapshot')
    def test_no_otm_strikes_returns_none(self, mock_snapshot):
        """If no OTM strikes exist, should return None."""
        current_price = 200.0  # All strikes are below this
        quote = {'price': current_price, 'symbol': 'AAPL'}
        df = _make_options_df(150.0)
        mock_snapshot.return_value = (quote, df[df['type'] == 'call'].reset_index(drop=True))

        option, q, _ = select_option_contract('AAPL', 'BUY', current_price)

        self.assertIsNone(option)  # No calls above 200

    @patch('ibkr.options_executor.get_options_snapshot')
    def test_empty_chain_returns_none(self, mock_snapshot):
        """Empty options chain should return None."""
        mock_snapshot.return_value = ({'price': 150.0}, pd.DataFrame())

        option, q, _ = select_option_contract('AAPL', 'BUY', 150.0)
        self.assertIsNone(option)

    @patch('ibkr.options_executor.get_options_snapshot')
    def test_api_error_returns_none(self, mock_snapshot):
        """API error should return None gracefully."""
        mock_snapshot.side_effect = ValueError("No options data")

        option, q, _ = select_option_contract('AAPL', 'BUY', 150.0)
        self.assertIsNone(option)


class TestValidateOption(unittest.TestCase):
    """Test options guardrail validation."""

    def _good_option(self):
        return {
            'strike': 152.5,
            'type': 'call',
            'bid': 3.40,
            'ask': 3.50,
            'mid': 3.45,
            'volume': 500,
            'open_interest': 2000,
            'expiration': '2026-03-20',
        }

    def test_valid_option_passes(self):
        """A well-priced, liquid option should pass."""
        result = validate_option(self._good_option(), 'AAPL')
        self.assertIsNone(result)

    def test_wide_spread_rejected(self):
        """Spread > 3% should be rejected."""
        opt = self._good_option()
        opt['bid'] = 3.00
        opt['ask'] = 3.50
        opt['mid'] = 3.25
        # spread = 0.50 / 3.25 = 15.4%
        result = validate_option(opt, 'AAPL')
        self.assertIsNotNone(result)
        self.assertIn('Spread too wide', result)

    def test_borderline_spread_passes(self):
        """Spread just under 3% should pass."""
        opt = self._good_option()
        # Set spread to exactly 2.9%
        opt['mid'] = 10.0
        opt['bid'] = 10.0 - 0.145
        opt['ask'] = 10.0 + 0.145
        result = validate_option(opt, 'AAPL')
        self.assertIsNone(result)

    def test_low_open_interest_rejected(self):
        """Open interest below threshold should be rejected."""
        opt = self._good_option()
        opt['open_interest'] = 5
        result = validate_option(opt, 'AAPL')
        self.assertIsNotNone(result)
        self.assertIn('Open interest too low', result)

    def test_zero_volume_rejected(self):
        """Zero volume should be rejected."""
        opt = self._good_option()
        opt['volume'] = 0
        result = validate_option(opt, 'AAPL')
        self.assertIsNotNone(result)
        self.assertIn('Volume too low', result)

    def test_zero_ask_rejected(self):
        """Zero ask should be rejected."""
        opt = self._good_option()
        opt['ask'] = 0
        result = validate_option(opt, 'AAPL')
        self.assertIsNotNone(result)
        self.assertIn('No valid ask', result)

    def test_zero_mid_computes_from_bid_ask(self):
        """If mid is 0, should compute from bid/ask."""
        opt = self._good_option()
        opt['mid'] = 0
        # bid=3.40, ask=3.50 -> mid=3.45, spread=0.10/3.45=2.9% — passes
        result = validate_option(opt, 'AAPL')
        self.assertIsNone(result)

    def test_none_values_handled(self):
        """None values for bid/ask/volume should be treated as 0."""
        opt = self._good_option()
        opt['bid'] = None
        opt['ask'] = None
        result = validate_option(opt, 'AAPL')
        self.assertIsNotNone(result)  # Should reject (no valid ask)


class TestExecuteOptionSignal(unittest.TestCase):
    """Test the full execute_option_signal function in dry-run mode."""

    @patch('ibkr.options_executor.get_options_snapshot')
    def test_dry_run_buy_signal(self, mock_snapshot):
        """Dry-run BUY should select call and return dry_run status."""
        current_price = 150.0
        quote = {'price': current_price, 'symbol': 'AAPL'}
        df = _make_options_df(current_price)
        mock_snapshot.return_value = (quote, df[df['type'] == 'call'].reset_index(drop=True))

        sig = {
            'action': 'BUY', 'symbol': 'AAPL', 'shares': 1,
            'price': 150.0, 'strategy': 'bollinger', 'reason': 'test'
        }

        result = execute_option_signal(None, sig, dry_run=True)

        self.assertEqual(result['status'], 'dry_run')
        self.assertEqual(result['contract_type'], 'call')
        self.assertEqual(result['strike'], 152.5)
        self.assertIn('2026-03-20', result['expiration'])
        self.assertGreater(result['premium'], 0)

    @patch('ibkr.options_executor.get_options_snapshot')
    def test_dry_run_sell_signal(self, mock_snapshot):
        """Dry-run SELL should select put and return dry_run status."""
        current_price = 150.0
        quote = {'price': current_price, 'symbol': 'AAPL'}
        df = _make_options_df(current_price)
        mock_snapshot.return_value = (quote, df[df['type'] == 'put'].reset_index(drop=True))

        sig = {
            'action': 'SELL', 'symbol': 'AAPL', 'shares': 1,
            'price': 150.0, 'strategy': 'oversold_bounce', 'reason': 'test'
        }

        result = execute_option_signal(None, sig, dry_run=True)

        self.assertEqual(result['status'], 'dry_run')
        self.assertEqual(result['contract_type'], 'put')
        self.assertLess(result['strike'], current_price)

    @patch('ibkr.options_executor.get_options_snapshot')
    def test_illiquid_option_rejected(self, mock_snapshot):
        """Illiquid option should be rejected."""
        current_price = 150.0
        quote = {'price': current_price, 'symbol': 'AAPL'}
        df = _make_options_df(current_price)
        # Make all calls illiquid
        df_calls = df[df['type'] == 'call'].copy()
        df_calls['open_interest'] = 2
        df_calls['volume'] = 0
        mock_snapshot.return_value = (quote, df_calls.reset_index(drop=True))

        sig = {
            'action': 'BUY', 'symbol': 'AAPL', 'shares': 1,
            'price': 150.0, 'strategy': 'test', 'reason': 'test'
        }

        result = execute_option_signal(None, sig, dry_run=True)

        self.assertEqual(result['status'], 'failed')
        self.assertIn('too low', result['error'])

    @patch('ibkr.options_executor.get_options_snapshot')
    def test_wide_spread_rejected(self, mock_snapshot):
        """Wide spread option should be rejected."""
        current_price = 150.0
        quote = {'price': current_price, 'symbol': 'AAPL'}
        df = _make_options_df(current_price)
        df_calls = df[df['type'] == 'call'].copy()
        # Widen spreads to 20%
        df_calls['bid'] = 1.0
        df_calls['ask'] = 3.0
        df_calls['mid'] = 2.0
        mock_snapshot.return_value = (quote, df_calls.reset_index(drop=True))

        sig = {
            'action': 'BUY', 'symbol': 'AAPL', 'shares': 1,
            'price': 150.0, 'strategy': 'test', 'reason': 'test'
        }

        result = execute_option_signal(None, sig, dry_run=True)

        self.assertEqual(result['status'], 'failed')
        self.assertIn('Spread too wide', result['error'])

    @patch('ibkr.options_executor.get_options_snapshot')
    def test_no_options_data_handled(self, mock_snapshot):
        """If no options data, should fail gracefully."""
        mock_snapshot.side_effect = ValueError("No options data returned for XYZ")

        sig = {
            'action': 'BUY', 'symbol': 'XYZ', 'shares': 1,
            'price': 50.0, 'strategy': 'test', 'reason': 'test'
        }

        result = execute_option_signal(None, sig, dry_run=True)

        self.assertEqual(result['status'], 'failed')
        self.assertIn('No suitable', result['error'])

    @patch('ibkr.options_executor.get_options_snapshot')
    def test_result_dict_has_required_fields(self, mock_snapshot):
        """Result dict should be compatible with execute_signals email format."""
        current_price = 150.0
        quote = {'price': current_price, 'symbol': 'AAPL'}
        df = _make_options_df(current_price)
        mock_snapshot.return_value = (quote, df[df['type'] == 'call'].reset_index(drop=True))

        sig = {
            'action': 'BUY', 'symbol': 'AAPL', 'shares': 1,
            'price': 150.0, 'strategy': 'bollinger', 'reason': 'test'
        }

        result = execute_option_signal(None, sig, dry_run=True)

        # Must have all fields expected by _send_execution_summary_email
        required_fields = ['status', 'fill_price', 'filled_shares', 'error',
                          'executed_at', 'action', 'symbol',
                          'contract_type', 'strike', 'expiration', 'premium']
        for field in required_fields:
            self.assertIn(field, result, f"Missing required field: {field}")


if __name__ == '__main__':
    unittest.main()
