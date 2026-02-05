"""
Pytest configuration and shared fixtures for IBKR tests.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton state between tests."""
    yield
    # Cleanup after test


@pytest.fixture
def sample_symbols():
    """Provide sample stock symbols for testing."""
    return {"AAPL", "GOOGL", "MSFT", "NVDA", "TSLA"}


@pytest.fixture
def sample_predictions():
    """Provide sample prediction data."""
    return {
        "AAPL": {
            "pred_reg_3d": 0.015,
            "pred_reg_10d": 0.025,
            "adx_signal": 1,
            "cci_signal": 0.5
        },
        "GOOGL": {
            "pred_reg_3d": -0.005,
            "pred_reg_10d": 0.01,
            "adx_signal": 0,
            "cci_signal": -0.3
        }
    }
