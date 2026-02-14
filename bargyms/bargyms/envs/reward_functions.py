"""
Pluggable reward functions for the config-driven TradingEnv.

All parameters come from the YAML config — zero hardcoded magic numbers.
Each reward function receives an env_state dict and returns a scalar reward.
"""

from abc import ABC, abstractmethod
from collections import deque

import numpy as np


class RewardFunction(ABC):
    """Base class for reward functions."""

    def __init__(self, params):
        self.params = params

    @abstractmethod
    def compute(self, env_state):
        """
        Compute reward for this step.

        Args:
            env_state: dict with keys:
                step_return:      float  — portfolio pct change this step
                cumulative_return: float — total pct return since episode start
                position_pnl_pct: float  — unrealized P&L on current position
                days_in_position: int    — 0 if flat
                action_type:      str    — "buy", "sell", "hold"
                was_illegal:      bool   — action was invalid (buy when holding, etc.)
                trade_just_closed: bool  — a trade was closed this step
                trade_return_pct: float  — return of the just-closed trade (or 0)
                portfolio_value:  float
                initial_cash:     float
                peak_value:       float  — max portfolio value seen

        Returns:
            float: scalar reward
        """

    def reset(self):
        """Called at episode start. Override if function has internal state."""
        pass


class PnLStepReward(RewardFunction):
    """Raw dollar P&L change per step, normalized by initial cash."""

    def compute(self, env_state):
        if env_state["was_illegal"]:
            return self.params.get("illegal_action_penalty", -1.0)
        return env_state["step_return"]

    def reset(self):
        pass


class PnLPctStepReward(RewardFunction):
    """Percent portfolio change per step."""

    def compute(self, env_state):
        if env_state["was_illegal"]:
            return self.params.get("illegal_action_penalty", -1.0)

        reward = env_state["step_return"]

        # Optional holding cost
        if env_state["days_in_position"] > 0:
            reward -= self.params.get("holding_cost", 0.0)

        # Optional no-trade penalty
        if env_state["action_type"] == "hold" and env_state["days_in_position"] == 0:
            reward -= self.params.get("no_trade_penalty", 0.0)

        return reward

    def reset(self):
        pass


class SharpeStepReward(RewardFunction):
    """
    Rolling Sharpe ratio of step returns.

    Until the window is full, falls back to raw step return.
    """

    def __init__(self, params):
        super().__init__(params)
        window_size = params.get("sharpe_window", 20)
        self._returns = deque(maxlen=window_size)

    def compute(self, env_state):
        if env_state["was_illegal"]:
            return self.params.get("illegal_action_penalty", -1.0)

        step_return = env_state["step_return"]
        self._returns.append(step_return)

        if len(self._returns) >= self._returns.maxlen:
            arr = np.array(self._returns)
            mean_r = np.mean(arr)
            std_r = np.std(arr) + 1e-8
            reward = mean_r / std_r
        else:
            reward = step_return

        # Optional penalties
        if env_state["days_in_position"] > 0:
            reward -= self.params.get("holding_cost", 0.0)
        if env_state["action_type"] == "hold" and env_state["days_in_position"] == 0:
            reward -= self.params.get("no_trade_penalty", 0.0)

        return reward

    def reset(self):
        self._returns.clear()


class RiskAdjustedReward(RewardFunction):
    """Step return minus drawdown penalty."""

    def compute(self, env_state):
        if env_state["was_illegal"]:
            return self.params.get("illegal_action_penalty", -1.0)

        reward = env_state["step_return"]

        # Drawdown penalty
        peak = env_state.get("peak_value", env_state["initial_cash"])
        current = env_state["portfolio_value"]
        if peak > 0:
            drawdown = (peak - current) / peak
            dd_penalty = self.params.get("drawdown_penalty", 1.0)
            reward -= drawdown * dd_penalty

        return reward

    def reset(self):
        pass


class TradeCompletionReward(RewardFunction):
    """
    Sparse reward: only fires when a trade is closed.

    Between trades, reward is 0 (plus optional penalties).
    When a trade closes, reward = trade_return_pct * multiplier.
    """

    def compute(self, env_state):
        if env_state["was_illegal"]:
            return self.params.get("illegal_action_penalty", -1.0)

        reward = 0.0

        if env_state.get("trade_just_closed", False):
            multiplier = self.params.get("trade_reward_multiplier", 10.0)
            reward = env_state.get("trade_return_pct", 0.0) * multiplier

        # Optional no-trade penalty to encourage activity
        if env_state["action_type"] == "hold" and env_state["days_in_position"] == 0:
            reward -= self.params.get("no_trade_penalty", 0.0)

        return reward

    def reset(self):
        pass


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class RewardRegistry:
    """Registry for reward functions. Maps config names to classes."""

    _registry = {
        "pnl_step": PnLStepReward,
        "pnl_pct_step": PnLPctStepReward,
        "sharpe_step": SharpeStepReward,
        "risk_adjusted": RiskAdjustedReward,
        "trade_completion": TradeCompletionReward,
    }

    @classmethod
    def get(cls, reward_config):
        """
        Create a reward function from config.

        Args:
            reward_config: dict with "function" (name) and "params" (dict)
        """
        name = reward_config.get("function", "pnl_pct_step")
        params = reward_config.get("params", {})
        reward_cls = cls._registry.get(name)
        if reward_cls is None:
            raise ValueError(
                f"Unknown reward function '{name}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return reward_cls(params)

    @classmethod
    def register(cls, name, reward_cls):
        """Register a custom reward function."""
        cls._registry[name] = reward_cls
