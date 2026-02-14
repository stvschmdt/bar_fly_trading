"""
Action space handlers for the config-driven TradingEnv.

Supports discrete (PPO), continuous (SAC/TD3/PPO), and multi-discrete action spaces.
Each handler translates raw gym actions into (action_type, size_fraction) tuples.
"""

from abc import ABC, abstractmethod

import numpy as np
import gymnasium as gym


class ActionHandler(ABC):
    """Base class for translating gym actions into trading decisions."""

    @abstractmethod
    def get_space(self):
        """Return the gymnasium action space."""

    @abstractmethod
    def decode(self, action, has_position):
        """
        Decode a raw gym action into a trading decision.

        Args:
            action: Raw action from the policy
            has_position: Whether agent currently holds a position

        Returns:
            (action_type, size_fraction) where:
                action_type: "buy" | "sell" | "hold"
                size_fraction: 0.0-1.0 fraction of available cash (buy) or position (sell)
        """


class DiscreteActionHandler(ActionHandler):
    """
    Discrete(3): 0=hold, 1=buy, 2=sell

    Buy uses the configured position_size fraction.
    Sell closes the entire position.
    """

    def __init__(self, config):
        self.position_size = config["environment"].get("position_size", 0.10)

    def get_space(self):
        return gym.spaces.Discrete(3)

    def decode(self, action, has_position):
        if action == 1 and not has_position:
            return "buy", self.position_size
        elif action == 2 and has_position:
            return "sell", 1.0
        return "hold", 0.0


class ContinuousActionHandler(ActionHandler):
    """
    Box(-1, 1, shape=(1,)): continuous position sizing.

    action > +dead_zone: buy, magnitude = fraction of available cash * position_size
    action < -dead_zone: sell, magnitude = fraction of position to close
    |action| <= dead_zone: hold

    Dead zone prevents noisy hold signals from SAC/TD3 exploration.
    """

    def __init__(self, config):
        self.position_size = config["environment"].get("position_size", 0.10)
        self.dead_zone = config["environment"].get("dead_zone", 0.1)

    def get_space(self):
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def decode(self, action, has_position):
        if isinstance(action, np.ndarray):
            action = action.flatten()
            val = float(np.clip(action[0], -1.0, 1.0))
        else:
            val = float(np.clip(action, -1.0, 1.0))

        if val > self.dead_zone and not has_position:
            # Scale action from (dead_zone, 1.0) to (0, position_size)
            frac = (val - self.dead_zone) / (1.0 - self.dead_zone)
            return "buy", frac * self.position_size
        elif val < -self.dead_zone and has_position:
            # Scale action from (-1.0, -dead_zone) to (0, 1.0) sell fraction
            frac = (-val - self.dead_zone) / (1.0 - self.dead_zone)
            return "sell", min(frac, 1.0)
        return "hold", 0.0


class MultiDiscreteActionHandler(ActionHandler):
    """
    MultiDiscrete([3, 5]): [action_type, size_bucket]

    action_type: 0=hold, 1=buy, 2=sell
    size_bucket: 0=20%, 1=40%, 2=60%, 3=80%, 4=100% of position_size
    """

    SIZE_FRACTIONS = [0.2, 0.4, 0.6, 0.8, 1.0]

    def __init__(self, config):
        self.position_size = config["environment"].get("position_size", 0.10)

    def get_space(self):
        return gym.spaces.MultiDiscrete([3, len(self.SIZE_FRACTIONS)])

    def decode(self, action, has_position):
        action_type = int(action[0])
        size_idx = int(action[1])
        frac = self.SIZE_FRACTIONS[min(size_idx, len(self.SIZE_FRACTIONS) - 1)]

        if action_type == 1 and not has_position:
            return "buy", frac * self.position_size
        elif action_type == 2 and has_position:
            return "sell", frac
        return "hold", 0.0


class ActionSpaceFactory:
    """Factory for creating action handlers from config."""

    _handlers = {
        "discrete": DiscreteActionHandler,
        "continuous": ContinuousActionHandler,
        "multi_discrete": MultiDiscreteActionHandler,
    }

    @classmethod
    def create(cls, config):
        action_type = config["environment"].get("action_space", "discrete")
        handler_cls = cls._handlers.get(action_type)
        if handler_cls is None:
            raise ValueError(
                f"Unknown action space '{action_type}'. "
                f"Available: {list(cls._handlers.keys())}"
            )
        return handler_cls(config)
