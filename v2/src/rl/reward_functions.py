"""
Reward functions for Reinforcement Learning.

Provides various reward function implementations for RL training.
"""

from typing import List, Optional

import numpy as np


class RewardFunctions:
    """
    Collection of reward functions for RL trading environments.

    All methods can be used as standalone functions or through the class.
    """

    @staticmethod
    def pnl_reward(pnl: float, balance: float) -> float:
        """
        Simple P&L-based reward.

        Args:
            pnl: Period profit/loss
            balance: Current balance

        Returns:
            Normalized reward
        """
        if balance == 0:
            return 0.0
        return pnl / balance

    @staticmethod
    def sharpe_reward(
        returns: List[float],
        window: int = 20,
        annualization_factor: float = 1.0,
    ) -> float:
        """
        Sharpe ratio-based reward using rolling window.

        Args:
            returns: List of period returns
            window: Lookback window for calculation
            annualization_factor: Factor to annualize (default 1.0 for no annualization)

        Returns:
            Sharpe-based reward
        """
        if len(returns) < window:
            return 0.0

        recent = np.array(returns[-window:])
        std = recent.std()

        if std == 0 or np.isnan(std):
            return 0.0

        return float((recent.mean() / std) * annualization_factor)

    @staticmethod
    def sortino_reward(
        returns: List[float],
        window: int = 20,
        annualization_factor: float = 1.0,
    ) -> float:
        """
        Sortino ratio-based reward (only considers downside volatility).

        Args:
            returns: List of period returns
            window: Lookback window for calculation
            annualization_factor: Factor to annualize

        Returns:
            Sortino-based reward
        """
        if len(returns) < window:
            return 0.0

        recent = np.array(returns[-window:])
        downside = recent[recent < 0]

        if len(downside) == 0:
            # No negative returns - return scaled mean
            return float(recent.mean() * 100)

        downside_std = downside.std()
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0

        return float((recent.mean() / downside_std) * annualization_factor)

    @staticmethod
    def risk_adjusted_reward(
        pnl: float,
        drawdown: float,
        drawdown_penalty: float = 0.5,
    ) -> float:
        """
        Risk-adjusted reward that penalizes drawdown.

        Args:
            pnl: Period profit/loss
            drawdown: Current drawdown percentage
            drawdown_penalty: Penalty weight for drawdown

        Returns:
            Risk-adjusted reward
        """
        return pnl - drawdown_penalty * abs(drawdown)

    @staticmethod
    def asymmetric_reward(
        pnl: float,
        win_multiplier: float = 1.0,
        loss_multiplier: float = 2.0,
    ) -> float:
        """
        Asymmetric reward that penalizes losses more than rewarding gains.

        Args:
            pnl: Period profit/loss
            win_multiplier: Multiplier for positive P&L
            loss_multiplier: Multiplier for negative P&L

        Returns:
            Asymmetrically weighted reward
        """
        if pnl >= 0:
            return pnl * win_multiplier
        else:
            return pnl * loss_multiplier

    @staticmethod
    def differential_sharpe_reward(
        prev_sharpe: float,
        current_sharpe: float,
        scaling: float = 10.0,
    ) -> float:
        """
        Reward based on improvement in Sharpe ratio.

        Args:
            prev_sharpe: Sharpe ratio from previous step
            current_sharpe: Current Sharpe ratio
            scaling: Scaling factor for the difference

        Returns:
            Differential Sharpe reward
        """
        return (current_sharpe - prev_sharpe) * scaling

    @staticmethod
    def calmar_reward(
        returns: List[float],
        equity_curve: List[float],
        window: int = 252,
    ) -> float:
        """
        Calmar ratio-based reward.

        Args:
            returns: List of period returns
            equity_curve: List of equity values
            window: Lookback window

        Returns:
            Calmar-based reward
        """
        if len(returns) < 2 or len(equity_curve) < 2:
            return 0.0

        # Calculate return
        recent_returns = returns[-window:] if len(returns) >= window else returns
        total_return = sum(recent_returns)

        # Calculate max drawdown
        curve = equity_curve[-window:] if len(equity_curve) >= window else equity_curve
        peak = max(curve)
        trough = min(curve)

        if peak == 0:
            return 0.0

        max_dd = (peak - trough) / peak

        if max_dd == 0:
            return total_return * 100

        return total_return / max_dd

    @staticmethod
    def position_holding_reward(
        pnl: float,
        holding_steps: int,
        optimal_holding: int = 60,
        penalty_factor: float = 0.01,
    ) -> float:
        """
        Reward that considers position holding time.

        Args:
            pnl: Trade profit/loss
            holding_steps: Number of steps position was held
            optimal_holding: Optimal holding period
            penalty_factor: Penalty for deviating from optimal

        Returns:
            Holding-adjusted reward
        """
        holding_penalty = abs(holding_steps - optimal_holding) * penalty_factor
        return pnl - holding_penalty

    @staticmethod
    def trade_cost_aware_reward(
        pnl: float,
        n_trades: int,
        trade_cost: float = 0.001,
    ) -> float:
        """
        Reward that accounts for trading costs.

        Args:
            pnl: Period profit/loss
            n_trades: Number of trades in period
            trade_cost: Cost per trade as fraction

        Returns:
            Cost-adjusted reward
        """
        return pnl - n_trades * trade_cost

    @staticmethod
    def regime_aware_reward(
        pnl: float,
        regime: int,
        regime_multipliers: Optional[dict] = None,
    ) -> float:
        """
        Reward that scales based on market regime.

        Args:
            pnl: Period profit/loss
            regime: Current market regime (0=crash, 1=normal, 2=trending)
            regime_multipliers: Dict mapping regime to multiplier

        Returns:
            Regime-adjusted reward
        """
        if regime_multipliers is None:
            regime_multipliers = {
                0: 0.5,  # Crash - reward more for surviving
                1: 1.0,  # Normal
                2: 0.8,  # Trending - slightly less to avoid overtrading
            }

        multiplier = regime_multipliers.get(regime, 1.0)
        return pnl * multiplier

    @staticmethod
    def composite_reward(
        pnl: float,
        returns: List[float],
        drawdown: float,
        weights: Optional[dict] = None,
    ) -> float:
        """
        Composite reward combining multiple factors.

        Args:
            pnl: Period profit/loss
            returns: List of returns for Sharpe calculation
            drawdown: Current drawdown percentage
            weights: Dict of weights for each component

        Returns:
            Composite reward
        """
        if weights is None:
            weights = {
                "pnl": 0.4,
                "sharpe": 0.3,
                "drawdown": 0.3,
            }

        # P&L component
        pnl_component = pnl * weights.get("pnl", 0.4)

        # Sharpe component
        sharpe = RewardFunctions.sharpe_reward(returns) if len(returns) >= 5 else 0.0
        sharpe_component = sharpe * weights.get("sharpe", 0.3) * 0.1  # Scale down

        # Drawdown component (negative is bad)
        dd_component = -abs(drawdown) * weights.get("drawdown", 0.3) * 0.01

        return pnl_component + sharpe_component + dd_component

    @staticmethod
    def sparse_reward(
        pnl: float,
        trade_closed: bool,
        episode_end: bool,
    ) -> float:
        """
        Sparse reward - only given at trade close or episode end.

        Args:
            pnl: Trade profit/loss
            trade_closed: Whether a trade was closed
            episode_end: Whether episode ended

        Returns:
            Sparse reward (0 during episode, actual reward at events)
        """
        if trade_closed:
            return pnl
        elif episode_end:
            return pnl
        return 0.0

    @staticmethod
    def curiosity_reward(
        intrinsic_reward: float,
        extrinsic_reward: float,
        curiosity_weight: float = 0.1,
    ) -> float:
        """
        Reward combining intrinsic curiosity with extrinsic reward.

        Args:
            intrinsic_reward: Curiosity-based reward (e.g., prediction error)
            extrinsic_reward: External reward (e.g., P&L)
            curiosity_weight: Weight for intrinsic component

        Returns:
            Combined reward
        """
        return (
            extrinsic_reward * (1 - curiosity_weight)
            + intrinsic_reward * curiosity_weight
        )

    @staticmethod
    def potential_based_reward(
        current_potential: float,
        previous_potential: float,
        gamma: float = 0.99,
    ) -> float:
        """
        Potential-based reward shaping.

        Args:
            current_potential: Potential function value at current state
            previous_potential: Potential function value at previous state
            gamma: Discount factor

        Returns:
            Shaped reward
        """
        return gamma * current_potential - previous_potential

    @staticmethod
    def clipped_reward(
        reward: float,
        min_reward: float = -1.0,
        max_reward: float = 1.0,
    ) -> float:
        """
        Clip reward to prevent extreme values.

        Args:
            reward: Original reward
            min_reward: Minimum allowed reward
            max_reward: Maximum allowed reward

        Returns:
            Clipped reward
        """
        return float(np.clip(reward, min_reward, max_reward))

    @staticmethod
    def normalized_reward(
        reward: float,
        running_mean: float,
        running_std: float,
        epsilon: float = 1e-8,
    ) -> float:
        """
        Normalize reward using running statistics.

        Args:
            reward: Original reward
            running_mean: Running mean of rewards
            running_std: Running standard deviation
            epsilon: Small value to prevent division by zero

        Returns:
            Normalized reward
        """
        return (reward - running_mean) / (running_std + epsilon)


class RewardScaler:
    """
    Helper class to scale and normalize rewards over time.
    """

    def __init__(
        self,
        window_size: int = 100,
        clip_range: tuple = (-10.0, 10.0),
    ):
        """
        Initialize reward scaler.

        Args:
            window_size: Window for running statistics
            clip_range: Range to clip normalized rewards
        """
        self.window_size = window_size
        self.clip_range = clip_range
        self.rewards: List[float] = []

    def scale(self, reward: float) -> float:
        """
        Scale a reward using running statistics.

        Args:
            reward: Raw reward

        Returns:
            Scaled reward
        """
        self.rewards.append(reward)

        # Trim to window size
        if len(self.rewards) > self.window_size:
            self.rewards = self.rewards[-self.window_size :]

        if len(self.rewards) < 2:
            return reward

        # Calculate running statistics
        mean = np.mean(self.rewards)
        std = np.std(self.rewards)

        # Normalize and clip
        if std > 0:
            normalized = (reward - mean) / std
        else:
            normalized = reward - mean

        return float(np.clip(normalized, self.clip_range[0], self.clip_range[1]))

    def reset(self) -> None:
        """Reset the scaler."""
        self.rewards = []
