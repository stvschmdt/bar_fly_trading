"""
Config-driven Gymnasium environment for RL trading.

Consumes the same all_data_*.csv data as the stockformer pipeline,
applies the same exit safety logic as the strategies framework,
and supports discrete/continuous/multi-discrete action spaces.
"""

import sys
import os

import numpy as np
import pandas as pd
import gymnasium as gym

# Add project root to path for stockformer/strategies imports
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from bargyms.envs.action_spaces import ActionSpaceFactory
from bargyms.envs.reward_functions import RewardRegistry
from bargyms.envs.observation_builder import ObservationBuilder, resolve_feature_columns


class RLDataLoader:
    """
    Loads all_data_*.csv using stockformer.data_utils.load_panel_csvs(),
    then builds per-symbol numpy arrays for fast environment access.
    """

    def __init__(self, config):
        self.config = config
        self.feature_columns = resolve_feature_columns(config.get("features", {}))
        self._symbol_data = {}   # {symbol: np.ndarray (T, F)}
        self._symbol_prices = {} # {symbol: np.ndarray (T,)}
        self._symbol_dates = {}  # {symbol: np.ndarray (T,)}
        self._symbols = []

    def load(self):
        """Load and preprocess all data."""
        from stockformer.data_utils import load_panel_csvs

        data_cfg = self.config.get("data", {})
        path = data_cfg.get("path", "all_data_*.csv")

        # Resolve path relative to project root
        if not os.path.isabs(path) and not path.startswith("/"):
            path = os.path.join(_PROJECT_ROOT, path)

        print(f"RLDataLoader: loading data from {path}")
        df = load_panel_csvs(path)

        # Normalize column names
        if "ticker" in df.columns and "symbol" not in df.columns:
            df = df.rename(columns={"ticker": "symbol"})
        if "close" in df.columns and "adjusted_close" not in df.columns:
            df = df.rename(columns={"close": "adjusted_close"})

        # Ensure date column is datetime
        df["date"] = pd.to_datetime(df["date"])

        # Filter date range
        date_range = data_cfg.get("date_range", {})
        if date_range.get("start"):
            df = df[df["date"] >= pd.to_datetime(date_range["start"])]
        if date_range.get("end"):
            df = df[df["date"] <= pd.to_datetime(date_range["end"])]

        # Filter symbols if specified
        symbols = data_cfg.get("symbols")
        if symbols:
            df = df[df["symbol"].isin(symbols)]

        # Determine which feature columns actually exist
        available = [c for c in self.feature_columns if c in df.columns]
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            print(f"RLDataLoader: {len(missing)} columns not in data, skipping: {missing[:5]}...")
        self.feature_columns = available

        # Build per-symbol arrays
        normalize = self.config.get("features", {}).get("normalize", "zscore")

        for symbol, group in df.groupby("symbol"):
            group = group.sort_values("date").reset_index(drop=True)
            if len(group) < 30:  # Skip symbols with too little data
                continue

            features = group[self.feature_columns].values.astype(np.float32)
            # Replace NaN/inf with 0
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            # Per-ticker normalization
            if normalize == "zscore":
                mean = features.mean(axis=0)
                std = features.std(axis=0) + 1e-8
                features = (features - mean) / std
            elif normalize == "minmax":
                fmin = features.min(axis=0)
                fmax = features.max(axis=0)
                denom = fmax - fmin + 1e-8
                features = (features - fmin) / denom

            self._symbol_data[symbol] = features
            self._symbol_prices[symbol] = group["adjusted_close"].values.astype(np.float32)
            self._symbol_dates[symbol] = group["date"].values

        self._symbols = sorted(self._symbol_data.keys())
        print(f"RLDataLoader: {len(self._symbols)} symbols, "
              f"{len(self.feature_columns)} features, "
              f"normalize={normalize}")

    def get_symbol_data(self, symbol):
        return self._symbol_data[symbol]

    def get_prices(self, symbol):
        return self._symbol_prices[symbol]

    def get_dates(self, symbol):
        return self._symbol_dates[symbol]

    def get_symbols(self):
        return self._symbols

    def train_val_split(self):
        """Return (train_symbols_dates, val_symbols_dates) based on temporal split."""
        val_frac = self.config.get("data", {}).get("val_split", 0.2)
        # Use dates from first symbol as reference (all aligned after load)
        if not self._symbols:
            return [], []
        ref_dates = self._symbol_dates[self._symbols[0]]
        n = len(ref_dates)
        split_idx = int(n * (1 - val_frac))
        return split_idx, n


class TradingEnv(gym.Env):
    """
    Config-driven trading environment.

    Observation: (lookback_window, n_features) float32
    Action: discrete/continuous/multi-discrete (from config)
    Reward: pluggable reward function (from config)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, data_loader, config, mode="train"):
        super().__init__()
        self.data_loader = data_loader
        self.config = config
        self.mode = mode

        env_cfg = config["environment"]
        self.lookback_window = env_cfg["lookback_window"]
        self.episode_length = env_cfg.get("episode_length", 40)
        self.initial_cash = env_cfg.get("initial_cash", 100000)
        self.transaction_cost_bps = env_cfg.get("transaction_cost_bps", 10)
        self.max_positions = env_cfg.get("max_positions", 1)

        # Exit safety config
        safety = env_cfg.get("exit_safety", {})
        self.exit_safety_enabled = safety.get("enabled", True)
        self.stop_loss_pct = safety.get("stop_loss_pct", -0.08)
        self.take_profit_pct = safety.get("take_profit_pct", 0.15)
        self.trailing_stop_pct = safety.get("trailing_stop_pct")
        self.trailing_activation_pct = safety.get("trailing_activation_pct", 0.0)
        self.max_hold_days = safety.get("max_hold_days", 20)

        # Delegates
        self._action_handler = ActionSpaceFactory.create(config)
        self._reward_fn = RewardRegistry.get(config.get("reward", {}))
        self._obs_builder = ObservationBuilder(config)

        # Sync obs builder's feature count with actual loaded features
        actual_n_features = data_loader.get_symbol_data(data_loader.get_symbols()[0]).shape[1]
        self._obs_builder.n_market_features = actual_n_features
        self._obs_builder.n_total_features = actual_n_features + 5  # + N_ACCOUNT_FEATURES

        # Spaces
        self.observation_space = self._obs_builder.get_space()
        self.action_space = self._action_handler.get_space()

        # Train/val split
        self._train_end_idx, self._total_len = data_loader.train_val_split()
        self._symbols = data_loader.get_symbols()

        # Episode state (set in reset)
        self._symbol = None
        self._symbol_data = None
        self._prices = None
        self._start_idx = 0
        self._current_step = 0
        self._cash = self.initial_cash
        self._shares = 0
        self._entry_price = 0.0
        self._entry_step = 0
        self._trailing_high = 0.0
        self._peak_value = self.initial_cash
        self._trade_log = []
        self._step_returns = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reward_fn.reset()

        # Pick symbol and start index
        if options and "symbol" in options:
            self._symbol = options["symbol"]
            self._start_idx = options.get("start_idx", self.lookback_window)
        elif self.mode == "train":
            self._symbol = self.np_random.choice(self._symbols)
            max_start = self._train_end_idx - self.episode_length - 1
            min_start = self.lookback_window
            if max_start <= min_start:
                max_start = min_start + 1
            self._start_idx = self.np_random.integers(min_start, max_start)
        else:  # eval
            self._symbol = self.np_random.choice(self._symbols)
            min_start = max(self._train_end_idx, self.lookback_window)
            data_len = len(self.data_loader.get_prices(self._symbol))
            max_start = data_len - self.episode_length - 1
            if max_start <= min_start:
                max_start = min_start + 1
            self._start_idx = self.np_random.integers(min_start, max(min_start + 1, max_start))

        self._symbol_data = self.data_loader.get_symbol_data(self._symbol)
        self._prices = self.data_loader.get_prices(self._symbol)
        self._current_step = 0
        self._cash = self.initial_cash
        self._shares = 0
        self._entry_price = 0.0
        self._entry_step = 0
        self._trailing_high = 0.0
        self._peak_value = self.initial_cash
        self._trade_log = []
        self._step_returns = []

        obs = self._build_obs()
        info = {"symbol": self._symbol}
        return obs, info

    def step(self, action):
        prev_value = self._portfolio_value()
        data_idx = self._start_idx + self._current_step
        current_price = float(self._prices[data_idx])
        has_position = self._shares > 0

        # 1. Decode action
        action_type, size_fraction = self._action_handler.decode(action, has_position)

        # 2. Check legality
        was_illegal = False
        if action_type == "buy" and has_position:
            was_illegal = True
            action_type = "hold"
        elif action_type == "sell" and not has_position:
            was_illegal = True
            action_type = "hold"

        trade_just_closed = False
        trade_return_pct = 0.0

        # 3. Execute action
        if action_type == "buy" and not has_position:
            invest_amount = self._cash * size_fraction
            cost_bps = self.transaction_cost_bps / 10000.0
            invest_after_cost = invest_amount * (1 - cost_bps)
            self._shares = int(invest_after_cost / current_price)
            if self._shares > 0:
                actual_cost = self._shares * current_price
                self._cash -= actual_cost * (1 + cost_bps)
                self._entry_price = current_price
                self._entry_step = self._current_step
                self._trailing_high = current_price
            else:
                action_type = "hold"

        elif action_type == "sell" and has_position:
            trade_just_closed = True
            trade_return_pct = (current_price - self._entry_price) / self._entry_price
            cost_bps = self.transaction_cost_bps / 10000.0
            proceeds = self._shares * current_price * (1 - cost_bps)
            self._cash += proceeds
            # Log trade
            self._trade_log.append({
                "symbol": self._symbol,
                "entry_step": self._entry_step,
                "exit_step": self._current_step,
                "entry_price": self._entry_price,
                "exit_price": current_price,
                "shares": self._shares,
                "pnl": (current_price - self._entry_price) * self._shares,
                "return_pct": trade_return_pct * 100,
                "hold_days": self._current_step - self._entry_step,
                "exit_reason": "agent_sell",
            })
            self._shares = 0
            self._entry_price = 0.0

        # 4. Check exit safety (may force-close)
        if self.exit_safety_enabled and self._shares > 0:
            forced, reason = self._check_exit_safety(current_price)
            if forced:
                trade_return_pct = (current_price - self._entry_price) / self._entry_price
                cost_bps = self.transaction_cost_bps / 10000.0
                proceeds = self._shares * current_price * (1 - cost_bps)
                self._cash += proceeds
                self._trade_log.append({
                    "symbol": self._symbol,
                    "entry_step": self._entry_step,
                    "exit_step": self._current_step,
                    "entry_price": self._entry_price,
                    "exit_price": current_price,
                    "shares": self._shares,
                    "pnl": (current_price - self._entry_price) * self._shares,
                    "return_pct": trade_return_pct * 100,
                    "hold_days": self._current_step - self._entry_step,
                    "exit_reason": reason,
                })
                self._shares = 0
                self._entry_price = 0.0
                trade_just_closed = True

        # Update trailing high
        if self._shares > 0:
            self._trailing_high = max(self._trailing_high, current_price)

        # 5. Advance step
        self._current_step += 1

        # 6. Compute reward
        new_value = self._portfolio_value()
        self._peak_value = max(self._peak_value, new_value)
        step_return = (new_value - prev_value) / max(prev_value, 1e-8)
        self._step_returns.append(step_return)

        days_in_position = (self._current_step - self._entry_step) if self._shares > 0 else 0

        env_state = {
            "step_return": step_return,
            "cumulative_return": (new_value - self.initial_cash) / self.initial_cash,
            "position_pnl_pct": (current_price - self._entry_price) / self._entry_price if self._entry_price > 0 else 0.0,
            "days_in_position": days_in_position,
            "action_type": action_type,
            "was_illegal": was_illegal,
            "trade_just_closed": trade_just_closed,
            "trade_return_pct": trade_return_pct,
            "portfolio_value": new_value,
            "initial_cash": self.initial_cash,
            "peak_value": self._peak_value,
        }
        reward = self._reward_fn.compute(env_state)

        # 7. Check termination
        terminated = False
        truncated = False

        # Episode length reached
        if self._current_step >= self.episode_length:
            truncated = True
        # Data boundary
        if self._start_idx + self._current_step >= len(self._prices) - 1:
            truncated = True
        # Account blown (lost 50%)
        if new_value < self.initial_cash * 0.5:
            terminated = True

        # Force-close on episode end
        if (terminated or truncated) and self._shares > 0:
            end_price = float(self._prices[min(self._start_idx + self._current_step, len(self._prices) - 1)])
            cost_bps = self.transaction_cost_bps / 10000.0
            proceeds = self._shares * end_price * (1 - cost_bps)
            self._cash += proceeds
            self._trade_log.append({
                "symbol": self._symbol,
                "entry_step": self._entry_step,
                "exit_step": self._current_step,
                "entry_price": self._entry_price,
                "exit_price": end_price,
                "shares": self._shares,
                "pnl": (end_price - self._entry_price) * self._shares,
                "return_pct": (end_price - self._entry_price) / self._entry_price * 100,
                "hold_days": self._current_step - self._entry_step,
                "exit_reason": "episode_end",
            })
            self._shares = 0

        # 8. Build observation
        obs = self._build_obs()

        # 9. Build info
        info = {
            "symbol": self._symbol,
            "portfolio_value": new_value,
            "step_return": step_return,
            "action_type": action_type,
            "was_illegal": was_illegal,
            "n_trades": len(self._trade_log),
        }

        if terminated or truncated:
            final_value = self._portfolio_value()
            info["episode_return_pct"] = (final_value - self.initial_cash) / self.initial_cash * 100
            info["final_value"] = final_value
            info["trade_log"] = self._trade_log
            info["n_trades"] = len(self._trade_log)
            if self._trade_log:
                returns = [t["return_pct"] for t in self._trade_log]
                info["avg_trade_return_pct"] = np.mean(returns)
                info["win_rate"] = sum(1 for r in returns if r > 0) / len(returns) * 100

        return obs, reward, terminated, truncated, info

    def _portfolio_value(self):
        """Current portfolio value (cash + position value)."""
        if self._shares > 0:
            idx = min(self._start_idx + self._current_step, len(self._prices) - 1)
            price = float(self._prices[idx])
            return self._cash + self._shares * price
        return self._cash

    def _build_obs(self):
        """Build observation for current step."""
        data_idx = self._start_idx + self._current_step
        current_price = float(self._prices[min(data_idx, len(self._prices) - 1)])
        portfolio_value = self._portfolio_value()

        account_state = {
            "cash": self._cash,
            "initial_cash": self.initial_cash,
            "position_value": self._shares * current_price if self._shares > 0 else 0,
            "portfolio_value": portfolio_value,
            "entry_price": self._entry_price,
            "current_price": current_price,
            "days_held": (self._current_step - self._entry_step) if self._shares > 0 else 0,
            "is_long": self._shares > 0,
        }

        return self._obs_builder.build(self._symbol_data, data_idx, account_state)

    def _check_exit_safety(self, current_price):
        """
        Exit safety checks — same math as BaseStrategy.check_exit_safety().

        Returns (should_exit, reason) tuple.
        """
        if self._entry_price <= 0:
            return False, ""

        pct_change = (current_price - self._entry_price) / self._entry_price

        # Hard stop-loss
        if self.stop_loss_pct is not None and pct_change <= self.stop_loss_pct:
            return True, "stop_loss"

        # Take-profit
        if self.take_profit_pct is not None and pct_change >= self.take_profit_pct:
            return True, "take_profit"

        # Max hold days
        hold_days = self._current_step - self._entry_step
        if hold_days >= self.max_hold_days:
            return True, "max_hold"

        # Trailing stop with progressive tightening
        if self.trailing_stop_pct is not None and pct_change >= self.trailing_activation_pct:
            # Progressive tightening: trail narrows as price approaches TP
            if self.take_profit_pct and self.take_profit_pct > self.trailing_activation_pct:
                progress = (pct_change - self.trailing_activation_pct) / (
                    self.take_profit_pct - self.trailing_activation_pct
                )
                progress = min(max(progress, 0.0), 1.0)
                effective_trail = self.trailing_stop_pct * (1.0 - 0.5 * progress)
            else:
                effective_trail = self.trailing_stop_pct

            drawdown = (current_price - self._trailing_high) / self._trailing_high
            if drawdown <= effective_trail:
                return True, "trailing_stop"

        return False, ""
