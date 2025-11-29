"""
Trading environment for Reinforcement Learning.

Gym-compatible environment for training RL agents on trading.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class TradingEnvironment(gym.Env):
    """
    Gymnasium environment for trading.

    All parameters come from config - no hardcoded values.

    Config parameters (rl.environment):
    - observation_features: List of features for observation
    - action_space: Action space configuration
    - reward_function: Reward function type
    - episode_length: Length of each episode
    - initial_balance: Starting balance
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config: Dict[str, Any],
        data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize trading environment.

        Args:
            config: Full configuration dictionary
            data: OHLCV market data
            features: Pre-calculated features (optional)
            render_mode: Rendering mode
        """
        super().__init__()

        self.config = config.get("rl", {}).get("environment", {})
        self.data = data.reset_index(drop=True)
        self.features = features
        self.render_mode = render_mode

        # Environment parameters from config
        self.observation_features = self.config.get(
            "observation_features",
            ["ofi", "tfi", "rsi", "adx", "regime", "position", "pnl", "drawdown"],
        )
        self.episode_length = self.config.get("episode_length", 1440)
        self.initial_balance = self.config.get("initial_balance", 10000)

        # Action space configuration
        action_config = self.config.get("action_space", {})
        self.direction_choices = action_config.get("direction", [-1, 0, 1])
        self.position_size_choices = action_config.get(
            "position_size", [0.0, 0.25, 0.5, 0.75, 1.0]
        )
        self.sl_mult_choices = action_config.get(
            "sl_atr_mult", [0.5, 1.0, 1.5, 2.0, 3.0]
        )
        self.tp_mult_choices = action_config.get(
            "tp_atr_mult", [1.0, 1.5, 2.0, 3.0, 5.0]
        )

        # Reward function type
        self.reward_function = self.config.get("reward_function", "sharpe")

        # Define observation space
        n_features = len(self.observation_features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )

        # Define action space (Dict of Discrete spaces)
        self.action_space = spaces.Dict(
            {
                "direction": spaces.Discrete(len(self.direction_choices)),
                "position_size": spaces.Discrete(len(self.position_size_choices)),
                "sl_mult": spaces.Discrete(len(self.sl_mult_choices)),
                "tp_mult": spaces.Discrete(len(self.tp_mult_choices)),
            }
        )

        # State variables
        self.current_step = 0
        self.start_step = 0
        self.position = 0  # -1, 0, or 1
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.balance = self.initial_balance
        self.equity_curve: List[float] = []
        self.returns: List[float] = []
        self.trades: List[Dict[str, Any]] = []
        self._rng = np.random.default_rng()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Random start point, leaving room for episode
        max_start = max(0, len(self.data) - self.episode_length - 1)
        self.start_step = int(self._rng.integers(0, max(1, max_start)))
        self.current_step = self.start_step

        # Reset state
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.balance = self.initial_balance
        self.equity_curve = [self.balance]
        self.returns = []
        self.trades = []

        return self._get_observation(), {"step": self.current_step}

    def step(
        self, action: Dict[str, int]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action dictionary with direction, position_size, sl_mult, tp_mult

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Decode action
        direction = self.direction_choices[action["direction"]]
        position_size = self.position_size_choices[action["position_size"]]
        sl_mult = self.sl_mult_choices[action["sl_mult"]]
        tp_mult = self.tp_mult_choices[action["tp_mult"]]

        # Get current price data
        current_bar = self.data.iloc[self.current_step]
        current_price = float(current_bar.get("close", current_bar.get("Close", 0)))

        # Calculate ATR if available
        atr = float(current_bar.get("atr", current_bar.get("ATR", current_price * 0.02)))

        # Execute trading logic
        pnl = 0.0

        # Check if we should close existing position
        if self.position != 0:
            should_close, close_reason = self._check_exit(current_price)
            if should_close:
                pnl = self._close_position(current_price, close_reason)

        # Open new position if direction changed
        if self.position == 0 and direction != 0 and position_size > 0:
            self._open_position(
                direction=direction,
                size=position_size,
                price=current_price,
                atr=atr,
                sl_mult=sl_mult,
                tp_mult=tp_mult,
            )

        # Update equity
        unrealized_pnl = self._calculate_unrealized_pnl(current_price)
        current_equity = self.balance + unrealized_pnl
        self.equity_curve.append(current_equity)

        # Calculate return
        if len(self.equity_curve) >= 2:
            period_return = (self.equity_curve[-1] / self.equity_curve[-2]) - 1
            self.returns.append(period_return)
        else:
            self.returns.append(0.0)

        # Calculate reward
        reward = self._calculate_reward(pnl, current_equity)

        # Move to next step
        self.current_step += 1

        # Check termination
        terminated = (
            self.current_step >= self.start_step + self.episode_length
            or self.current_step >= len(self.data) - 1
        )
        truncated = self.balance <= 0

        # Get info
        info = {
            "step": self.current_step,
            "balance": self.balance,
            "equity": current_equity,
            "position": self.position,
            "pnl": pnl,
            "n_trades": len(self.trades),
        }

        return self._get_observation(), float(reward), terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation.

        Returns:
            Observation array
        """
        features = []

        # Get data row
        if self.features is not None and len(self.features) > self.current_step:
            row = self.features.iloc[self.current_step]
        elif self.current_step < len(self.data):
            row = self.data.iloc[self.current_step]
        else:
            row = self.data.iloc[-1]

        for feat_name in self.observation_features:
            if feat_name == "position":
                features.append(float(self.position))
            elif feat_name == "pnl":
                pnl_pct = (self.balance - self.initial_balance) / self.initial_balance
                features.append(float(pnl_pct))
            elif feat_name == "drawdown":
                features.append(float(self._current_drawdown()))
            elif feat_name == "position_size":
                features.append(float(self.position_size))
            else:
                # Get from data/features
                value = row.get(feat_name, 0.0)
                if pd.isna(value):
                    value = 0.0
                features.append(float(value))

        return np.array(features, dtype=np.float32)

    def _open_position(
        self,
        direction: int,
        size: float,
        price: float,
        atr: float,
        sl_mult: float,
        tp_mult: float,
    ) -> None:
        """
        Open a new position.

        Args:
            direction: Trade direction (1 for long, -1 for short)
            size: Position size as fraction of balance
            price: Entry price
            atr: ATR for stop/take profit calculation
            sl_mult: Stop loss ATR multiplier
            tp_mult: Take profit ATR multiplier
        """
        self.position = direction
        self.position_size = size
        self.entry_price = price

        # Calculate stop loss and take profit
        if direction == 1:  # Long
            self.stop_loss = price - atr * sl_mult
            self.take_profit = price + atr * tp_mult
        else:  # Short
            self.stop_loss = price + atr * sl_mult
            self.take_profit = price - atr * tp_mult

    def _check_exit(self, current_price: float) -> Tuple[bool, str]:
        """
        Check if position should be closed.

        Args:
            current_price: Current market price

        Returns:
            Tuple of (should_close, reason)
        """
        if self.position == 0:
            return False, ""

        # Check stop loss
        if self.position == 1 and current_price <= self.stop_loss:
            return True, "stop_loss"
        if self.position == -1 and current_price >= self.stop_loss:
            return True, "stop_loss"

        # Check take profit
        if self.position == 1 and current_price >= self.take_profit:
            return True, "take_profit"
        if self.position == -1 and current_price <= self.take_profit:
            return True, "take_profit"

        return False, ""

    def _close_position(self, price: float, reason: str) -> float:
        """
        Close current position.

        Args:
            price: Exit price
            reason: Reason for closing

        Returns:
            P&L from the trade
        """
        if self.position == 0:
            return 0.0

        # Calculate P&L
        position_value = self.balance * self.position_size
        if self.position == 1:  # Long
            pnl = position_value * (price - self.entry_price) / self.entry_price
        else:  # Short
            pnl = position_value * (self.entry_price - price) / self.entry_price

        # Record trade
        self.trades.append(
            {
                "entry_step": self.current_step,
                "entry_price": self.entry_price,
                "exit_price": price,
                "direction": self.position,
                "size": self.position_size,
                "pnl": pnl,
                "reason": reason,
            }
        )

        # Update balance
        self.balance += pnl

        # Reset position
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0

        return pnl

    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L for open position.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L
        """
        if self.position == 0:
            return 0.0

        position_value = self.balance * self.position_size
        if self.position == 1:  # Long
            return position_value * (current_price - self.entry_price) / self.entry_price
        else:  # Short
            return position_value * (self.entry_price - current_price) / self.entry_price

    def _current_drawdown(self) -> float:
        """
        Calculate current drawdown.

        Returns:
            Current drawdown as percentage
        """
        if len(self.equity_curve) == 0:
            return 0.0

        peak = max(self.equity_curve)
        current = self.equity_curve[-1]

        if peak == 0:
            return 0.0

        return (peak - current) / peak * 100

    def _calculate_reward(self, pnl: float, current_equity: float) -> float:
        """
        Calculate reward based on configured reward function.

        Args:
            pnl: Period P&L
            current_equity: Current equity

        Returns:
            Reward value
        """
        if self.reward_function == "pnl":
            return pnl / self.initial_balance

        elif self.reward_function == "sharpe":
            return self._sharpe_reward()

        elif self.reward_function == "sortino":
            return self._sortino_reward()

        elif self.reward_function == "calmar":
            return self._calmar_reward()

        else:
            return pnl / self.initial_balance

    def _sharpe_reward(self, window: int = 20) -> float:
        """Calculate Sharpe-based reward."""
        if len(self.returns) < window:
            return 0.0

        recent = np.array(self.returns[-window:])
        std = recent.std()
        if std == 0:
            return 0.0

        return float(recent.mean() / std)

    def _sortino_reward(self, window: int = 20) -> float:
        """Calculate Sortino-based reward."""
        if len(self.returns) < window:
            return 0.0

        recent = np.array(self.returns[-window:])
        downside = recent[recent < 0]

        if len(downside) == 0:
            return float(recent.mean() * 100)  # All positive

        downside_std = downside.std()
        if downside_std == 0:
            return 0.0

        return float(recent.mean() / downside_std)

    def _calmar_reward(self) -> float:
        """Calculate Calmar-based reward."""
        if len(self.returns) == 0:
            return 0.0

        total_return = np.sum(self.returns)
        max_dd = self._current_drawdown()

        if max_dd == 0:
            return total_return * 100

        return float(total_return / max_dd)

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "human":
            print(
                f"Step: {self.current_step}, "
                f"Balance: ${self.balance:.2f}, "
                f"Position: {self.position}, "
                f"Trades: {len(self.trades)}"
            )
        return None

    def close(self) -> None:
        """Clean up environment."""
        pass

    def get_episode_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the current episode.

        Returns:
            Dictionary of episode statistics
        """
        if len(self.trades) == 0:
            return {
                "n_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
            }

        trades_df = pd.DataFrame(self.trades)
        winning = trades_df[trades_df["pnl"] > 0]

        returns_arr = np.array(self.returns)
        std = returns_arr.std()
        sharpe = returns_arr.mean() / std * np.sqrt(252) if std > 0 else 0.0

        return {
            "n_trades": len(self.trades),
            "total_pnl": trades_df["pnl"].sum(),
            "win_rate": len(winning) / len(trades_df) * 100 if len(trades_df) > 0 else 0,
            "sharpe": sharpe,
            "max_drawdown": self._current_drawdown(),
        }


# Wrapper for flattened action space (for algorithms that don't support Dict)
class FlatActionWrapper(gym.ActionWrapper):
    """Wrapper to flatten Dict action space to Discrete."""

    def __init__(self, env: TradingEnvironment):
        super().__init__(env)

        # Calculate total combinations
        self.n_direction = len(env.direction_choices)
        self.n_position_size = len(env.position_size_choices)
        self.n_sl_mult = len(env.sl_mult_choices)
        self.n_tp_mult = len(env.tp_mult_choices)

        self.n_actions = (
            self.n_direction * self.n_position_size * self.n_sl_mult * self.n_tp_mult
        )
        self.action_space = spaces.Discrete(self.n_actions)

    def action(self, action: int) -> Dict[str, int]:
        """Convert flat action to dict action."""
        direction = action % self.n_direction
        remainder = action // self.n_direction

        position_size = remainder % self.n_position_size
        remainder = remainder // self.n_position_size

        sl_mult = remainder % self.n_sl_mult
        tp_mult = remainder // self.n_sl_mult

        return {
            "direction": direction,
            "position_size": position_size,
            "sl_mult": sl_mult,
            "tp_mult": tp_mult,
        }
