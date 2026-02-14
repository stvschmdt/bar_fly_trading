"""
Observation builder for the config-driven TradingEnv.

Builds (lookback_window, n_features) observation arrays from config-selected
feature columns plus account state features.
"""

import numpy as np
import gymnasium as gym

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from stockformer.config import BASE_FEATURE_COLUMNS, MARKET_FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Feature presets — curated subsets of BASE_FEATURE_COLUMNS
# ---------------------------------------------------------------------------

FEATURE_PRESETS = {
    "minimal": [
        "rsi_14", "adx_14",
        "close_1d_roc", "close_3d_roc",
        "sma_20_pct", "sma_50_pct",
        "bollinger_bands_signal", "macd_signal", "rsi_signal",
        "volume_pct",
        "bull_bear_delta",
        "pe_ratio", "beta",
        "treasury_yield_10year",
        "day_of_week_num",
    ],
    "standard": [
        # Momentum
        "close_1d_roc", "close_3d_roc", "close_10d_roc",
        "close_5d_vol", "vol_3d_mean",
        # Moving averages
        "sma_20_pct", "sma_50_pct", "sma_200_pct",
        # Technical indicators
        "rsi_14", "adx_14", "cci_14",
        # Relative position
        "52_week_high_pct", "52_week_low_pct",
        "high_pct", "low_pct", "open_pct",
        # Signals
        "macd_signal", "rsi_signal", "adx_signal",
        "atr_signal", "cci_signal",
        "bollinger_bands_signal", "sma_cross_signal",
        "pe_ratio_signal", "bull_bear_delta",
        # Fundamentals
        "pe_ratio", "beta", "dividend_yield",
        # Calendar
        "day_of_week_num", "month",
        # Market
        "spy_ret_1d", "spy_ret_5d", "spy_vol_5d",
        "sector_etf_ret_1d", "sector_rel_ret_1d",
    ],
    "full": BASE_FEATURE_COLUMNS,
}

# Account features appended to each row in the observation window
N_ACCOUNT_FEATURES = 5


def resolve_feature_columns(features_config):
    """Resolve feature column list from config preset + custom + exclude."""
    preset = features_config.get("preset", "standard")
    columns = list(FEATURE_PRESETS.get(preset, FEATURE_PRESETS["standard"]))

    # Add custom columns
    for col in features_config.get("custom_columns", []):
        if col not in columns:
            columns.append(col)

    # Remove excluded columns
    for col in features_config.get("exclude_columns", []):
        if col in columns:
            columns.remove(col)

    # Add market context if requested (avoid duplicates)
    if features_config.get("market_context", True):
        for col in MARKET_FEATURE_COLUMNS:
            if col not in columns:
                columns.append(col)

    return columns


class ObservationBuilder:
    """
    Builds observation arrays from config-selected feature columns.

    Observation shape: (lookback_window, n_market_features + N_ACCOUNT_FEATURES)

    Market features: selected columns from all_data CSVs for the lookback window.
    Account features (appended to each row):
        - cash_fraction: current_cash / initial_cash
        - position_fraction: position_value / portfolio_value
        - unrealized_pnl_pct: (price - entry_price) / entry_price
        - days_held_normalized: days_in_position / max_hold_days
        - is_long: 1.0 if holding, 0.0 if flat
    """

    def __init__(self, config):
        self.lookback_window = config["environment"]["lookback_window"]
        self.max_hold_days = config["environment"]["exit_safety"].get("max_hold_days", 20)
        self.feature_columns = resolve_feature_columns(config.get("features", {}))
        self.n_market_features = len(self.feature_columns)
        self.n_total_features = self.n_market_features + N_ACCOUNT_FEATURES

    def get_space(self):
        """Return the observation space."""
        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback_window, self.n_total_features),
            dtype=np.float32,
        )

    def build(self, symbol_data, current_step, account_state):
        """
        Build observation array for current step.

        Args:
            symbol_data: (T, F) numpy array of market features for this symbol
            current_step: Index into symbol_data for "now"
            account_state: dict with keys:
                cash, initial_cash, position_value, portfolio_value,
                entry_price, current_price, days_held, is_long

        Returns:
            (lookback_window, n_total_features) float32 array
        """
        start = max(0, current_step - self.lookback_window + 1)
        window = symbol_data[start:current_step + 1]  # (W, F)

        # Pad if we don't have enough history
        if len(window) < self.lookback_window:
            pad_rows = self.lookback_window - len(window)
            padding = np.zeros((pad_rows, window.shape[1]), dtype=np.float32)
            window = np.vstack([padding, window])

        # Build account features (same for all rows in the window)
        initial_cash = account_state.get("initial_cash", 100000)
        portfolio_value = account_state.get("portfolio_value", initial_cash)
        cash_fraction = account_state.get("cash", initial_cash) / max(initial_cash, 1e-8)
        position_fraction = account_state.get("position_value", 0) / max(portfolio_value, 1e-8)

        entry_price = account_state.get("entry_price", 0)
        current_price = account_state.get("current_price", 0)
        if account_state.get("is_long", False) and entry_price > 0:
            unrealized_pnl_pct = (current_price - entry_price) / entry_price
        else:
            unrealized_pnl_pct = 0.0

        days_held_norm = account_state.get("days_held", 0) / max(self.max_hold_days, 1)
        is_long = 1.0 if account_state.get("is_long", False) else 0.0

        acct_features = np.array(
            [cash_fraction, position_fraction, unrealized_pnl_pct, days_held_norm, is_long],
            dtype=np.float32,
        )

        # Tile account features across all rows in the window
        acct_tiled = np.tile(acct_features, (self.lookback_window, 1))

        obs = np.hstack([window.astype(np.float32), acct_tiled])
        return obs
