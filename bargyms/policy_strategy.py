"""
PolicyStrategy: Wraps a trained SB3 model as a BaseStrategy.

Enables using RL policies in the same signal pipeline as rule-based strategies:
    - Backtest via backtest.py
    - Signal generation via SignalWriter
    - Live execution via ibkr/execute_signals.py

Usage:
    strategy = PolicyStrategy(account, symbols, model_path, config_path)
    orders = strategy.evaluate(date, current_prices, options_data)
"""

import os
import sys
import yaml

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_BARGYMS_DIR = os.path.abspath(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _BARGYMS_DIR not in sys.path:
    sys.path.insert(0, _BARGYMS_DIR)

from stable_baselines3 import PPO, SAC, TD3
from order import StockOrder, OrderOperation
from strategies.base_strategy import BaseStrategy
from bargyms.envs.trading_env import RLDataLoader
from bargyms.envs.observation_builder import ObservationBuilder
from bargyms.envs.action_spaces import ActionSpaceFactory

ALGO_MAP = {"PPO": PPO, "SAC": SAC, "TD3": TD3}


class PolicyStrategy(BaseStrategy):
    """
    Wraps a trained SB3 policy as a BaseStrategy for the signal pipeline.

    The RL model sees the same observation as during training (features from
    all_data CSVs + account state) and produces an action, which is translated
    to BUY/SELL/HOLD orders.
    """

    STRATEGY_NAME = "rl_policy"
    REQUIRED_COLUMNS = ["date", "symbol", "adjusted_close"]

    def __init__(self, account, symbols, model_path=None, config_path=None,
                 data=None, data_path=None, position_size=0.10):
        super().__init__(account, symbols)
        self.position_size = position_size

        # Load config
        if config_path:
            with open(config_path, "r") as f:
                self._config = yaml.safe_load(f)
        else:
            raise ValueError("config_path is required for PolicyStrategy")

        # Load trained model
        algo_name = self._config["algorithm"]["name"].upper()
        algo_cls = ALGO_MAP[algo_name]
        self._model = algo_cls.load(model_path)

        # Set up data loader and observation builder
        self._data_loader = RLDataLoader(self._config)
        self._data_loader.load()
        self._obs_builder = ObservationBuilder(self._config)
        self._action_handler = ActionSpaceFactory.create(self._config)

        # Exit safety params from config
        safety = self._config["environment"].get("exit_safety", {})
        self.STOP_LOSS_PCT = safety.get("stop_loss_pct", -0.08)
        self.TAKE_PROFIT_PCT = safety.get("take_profit_pct", 0.15)
        self.TRAILING_STOP_PCT = safety.get("trailing_stop_pct")
        self.TRAILING_ACTIVATION_PCT = safety.get("trailing_activation_pct", 0.0)
        self.MAX_HOLD_DAYS = safety.get("max_hold_days", 20)

        # Load overnight data if provided
        if data is not None:
            self.load_overnight_data_from_df(data)
        elif data_path:
            self.load_overnight_data(data_path)

    def check_entry(self, row):
        """Not used — evaluate() handles all decisions."""
        return False

    def check_exit(self, row, hold_days, entry_price):
        """Not used — evaluate() handles all decisions."""
        return False, ""

    def entry_reason(self, row):
        return "rl_policy_signal"

    def fetch_realtime(self, symbol):
        """Not needed — uses overnight data."""
        return pd.DataFrame()

    def evaluate(self, date, current_prices, options_data):
        """
        Evaluate trading decisions for all symbols on a given date.

        For each symbol:
        1. Build observation from data_loader (lookback window ending at date)
        2. Feed to model.predict(obs, deterministic=True)
        3. Decode action to BUY/SELL/HOLD
        4. Emit StockOrder if buy or sell
        """
        orders = []
        date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)[:10]
        current_date = pd.to_datetime(date)
        cash_committed = 0.0

        for symbol in self.symbols:
            price_row = current_prices[current_prices['symbol'] == symbol]
            if len(price_row) == 0:
                continue
            current_price = float(price_row['open'].iloc[0])

            has_position = symbol in self.positions and self.positions[symbol]['shares'] > 0

            # Build observation for this symbol
            obs = self._build_observation(symbol, date, current_price, has_position)
            if obs is None:
                continue

            # Get action from policy
            action, _ = self._model.predict(obs, deterministic=True)
            action_type, size_fraction = self._action_handler.decode(action, has_position)

            if has_position:
                entry_price = self.positions[symbol]['entry_price']
                entry_date = self.positions[symbol]['entry_date']
                hold_days = (current_date - entry_date).days

                # Check if policy wants to sell
                should_exit = action_type == "sell"

                # Also check exit safety (non-overridable)
                if not should_exit:
                    should_exit, exit_reason = self.check_exit_safety(
                        symbol, current_price, entry_price
                    )
                else:
                    exit_reason = "rl_policy_sell"

                # Max hold days
                if not should_exit and hold_days >= self.MAX_HOLD_DAYS:
                    should_exit = True
                    exit_reason = "max_hold"

                if should_exit:
                    shares = self.positions[symbol]['shares']
                    orders.append(StockOrder(
                        symbol, OrderOperation.SELL, shares, current_price, date_str
                    ))
                    self.trade_log.append({
                        'symbol': symbol,
                        'entry_date': entry_date.strftime('%Y-%m-%d'),
                        'exit_date': date_str,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'shares': shares,
                        'pnl': (current_price - entry_price) * shares,
                        'return_pct': (current_price - entry_price) / entry_price * 100,
                        'hold_days': hold_days,
                        'exit_reason': exit_reason,
                    })
                    self.record_exit(symbol, current_date)
                    del self.positions[symbol]

            elif action_type == "buy":
                allowed, _ = self.is_reentry_allowed(symbol, current_date)
                if not allowed:
                    continue
                if len(self.positions) >= self.MAX_POSITIONS:
                    continue

                available_cash = self.account.account_values.cash_balance - cash_committed
                invest = available_cash * size_fraction
                shares = int(invest // current_price)
                order_cost = shares * current_price

                if shares > 0 and order_cost <= available_cash:
                    orders.append(StockOrder(
                        symbol, OrderOperation.BUY, shares, current_price, date_str
                    ))
                    cash_committed += order_cost
                    self.positions[symbol] = {
                        'shares': shares,
                        'entry_date': current_date,
                        'entry_price': current_price,
                    }
                    self.record_entry(symbol)

        return orders

    def _build_observation(self, symbol, date, current_price, has_position):
        """Build observation array for the RL model."""
        if symbol not in self._data_loader._symbol_data:
            return None

        symbol_data = self._data_loader.get_symbol_data(symbol)
        dates = self._data_loader.get_dates(symbol)

        # Find the index for this date
        target = pd.to_datetime(date)
        date_idx = None
        for i, d in enumerate(dates):
            if pd.to_datetime(d) >= target:
                date_idx = i
                break

        if date_idx is None or date_idx < self._obs_builder.lookback_window:
            return None

        # Build account state
        if has_position:
            entry_price = self.positions[symbol]['entry_price']
            shares = self.positions[symbol]['shares']
            entry_date = self.positions[symbol]['entry_date']
            position_value = shares * current_price
            days_held = (target - entry_date).days
        else:
            entry_price = 0
            position_value = 0
            days_held = 0

        portfolio_value = self.account.account_values.get_total_value()
        account_state = {
            "cash": self.account.account_values.cash_balance,
            "initial_cash": self.account.initial_cash_balance,
            "position_value": position_value,
            "portfolio_value": portfolio_value,
            "entry_price": entry_price,
            "current_price": current_price,
            "days_held": days_held,
            "is_long": has_position,
        }

        obs = self._obs_builder.build(symbol_data, date_idx, account_state)
        return obs[np.newaxis, ...]  # Add batch dim for model.predict
