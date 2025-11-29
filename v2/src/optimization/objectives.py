"""
Objective functions for optimization.

Provides various performance metrics used as objectives for Optuna optimization.
"""

from typing import Optional

import numpy as np
import pandas as pd


class ObjectiveFunctions:
    """
    Collection of objective functions for portfolio/strategy optimization.

    All methods are static and can be used directly or through the class.
    """

    @staticmethod
    def sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate annualized Sharpe Ratio.

        Args:
            returns: Series of period returns
            risk_free_rate: Annual risk-free rate (default 0)
            periods_per_year: Number of periods per year (252 for daily)

        Returns:
            Annualized Sharpe Ratio
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / periods_per_year

        std = excess_returns.std()
        if std == 0 or np.isnan(std):
            return 0.0

        return float((excess_returns.mean() / std) * np.sqrt(periods_per_year))

    @staticmethod
    def sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate annualized Sortino Ratio.

        Only considers downside volatility (negative returns).

        Args:
            returns: Series of period returns
            risk_free_rate: Annual risk-free rate (default 0)
            periods_per_year: Number of periods per year (252 for daily)

        Returns:
            Annualized Sortino Ratio
        """
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            # No negative returns - return high value
            return float(excess_returns.mean() * periods_per_year * 100)

        downside_std = downside_returns.std()
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0

        return float((excess_returns.mean() / downside_std) * np.sqrt(periods_per_year))

    @staticmethod
    def calmar_ratio(
        returns: pd.Series,
        max_drawdown: Optional[float] = None,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate Calmar Ratio (Annual Return / Max Drawdown).

        Args:
            returns: Series of period returns
            max_drawdown: Max drawdown percentage (positive value).
                         If None, calculated from returns.
            periods_per_year: Number of periods per year

        Returns:
            Calmar Ratio
        """
        if len(returns) == 0:
            return 0.0

        annual_return = returns.mean() * periods_per_year

        if max_drawdown is None:
            # Calculate max drawdown from returns
            cumulative = (1 + returns).cumprod()
            max_drawdown = ObjectiveFunctions.max_drawdown(cumulative)

        if max_drawdown == 0 or np.isnan(max_drawdown):
            return 0.0

        return float(annual_return / abs(max_drawdown) * 100)

    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown percentage.

        Args:
            equity_curve: Series of equity values

        Returns:
            Maximum drawdown as percentage (positive value)
        """
        if len(equity_curve) == 0:
            return 0.0

        # Calculate running maximum
        peak = equity_curve.expanding().max()

        # Calculate drawdown at each point
        drawdown = (equity_curve - peak) / peak

        # Return maximum drawdown (as positive percentage)
        return float(abs(drawdown.min()) * 100)

    @staticmethod
    def profit_factor(trades: pd.DataFrame, pnl_column: str = "pnl") -> float:
        """
        Calculate Profit Factor (Gross Profit / Gross Loss).

        Args:
            trades: DataFrame with trade data
            pnl_column: Name of P&L column

        Returns:
            Profit Factor
        """
        if len(trades) == 0 or pnl_column not in trades.columns:
            return 0.0

        gross_profit = trades[trades[pnl_column] > 0][pnl_column].sum()
        gross_loss = abs(trades[trades[pnl_column] < 0][pnl_column].sum())

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return float(gross_profit / gross_loss)

    @staticmethod
    def win_rate(trades: pd.DataFrame, pnl_column: str = "pnl") -> float:
        """
        Calculate win rate percentage.

        Args:
            trades: DataFrame with trade data
            pnl_column: Name of P&L column

        Returns:
            Win rate as percentage
        """
        if len(trades) == 0 or pnl_column not in trades.columns:
            return 0.0

        winning_trades = len(trades[trades[pnl_column] > 0])
        return float(winning_trades / len(trades) * 100)

    @staticmethod
    def average_win_loss_ratio(
        trades: pd.DataFrame, pnl_column: str = "pnl"
    ) -> float:
        """
        Calculate average win / average loss ratio.

        Args:
            trades: DataFrame with trade data
            pnl_column: Name of P&L column

        Returns:
            Average win/loss ratio
        """
        if len(trades) == 0 or pnl_column not in trades.columns:
            return 0.0

        winning_trades = trades[trades[pnl_column] > 0]
        losing_trades = trades[trades[pnl_column] < 0]

        if len(winning_trades) == 0 or len(losing_trades) == 0:
            return 0.0

        avg_win = winning_trades[pnl_column].mean()
        avg_loss = abs(losing_trades[pnl_column].mean())

        if avg_loss == 0:
            return float("inf") if avg_win > 0 else 0.0

        return float(avg_win / avg_loss)

    @staticmethod
    def expectancy(trades: pd.DataFrame, pnl_column: str = "pnl") -> float:
        """
        Calculate trade expectancy (expected profit per trade).

        Args:
            trades: DataFrame with trade data
            pnl_column: Name of P&L column

        Returns:
            Expectancy per trade
        """
        if len(trades) == 0 or pnl_column not in trades.columns:
            return 0.0

        return float(trades[pnl_column].mean())

    @staticmethod
    def annualized_return(
        returns: pd.Series, periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized return.

        Args:
            returns: Series of period returns
            periods_per_year: Number of periods per year

        Returns:
            Annualized return as percentage
        """
        if len(returns) == 0:
            return 0.0

        # Compound return
        total_return = (1 + returns).prod() - 1
        n_periods = len(returns)

        if n_periods < periods_per_year:
            # Annualize for partial year
            annualized = (1 + total_return) ** (periods_per_year / n_periods) - 1
        else:
            # Already a full year or more
            years = n_periods / periods_per_year
            annualized = (1 + total_return) ** (1 / years) - 1

        return float(annualized * 100)

    @staticmethod
    def volatility(
        returns: pd.Series, periods_per_year: int = 252
    ) -> float:
        """
        Calculate annualized volatility.

        Args:
            returns: Series of period returns
            periods_per_year: Number of periods per year

        Returns:
            Annualized volatility as percentage
        """
        if len(returns) == 0:
            return 0.0

        return float(returns.std() * np.sqrt(periods_per_year) * 100)

    @staticmethod
    def ulcer_index(equity_curve: pd.Series) -> float:
        """
        Calculate Ulcer Index (measure of downside risk).

        Args:
            equity_curve: Series of equity values

        Returns:
            Ulcer Index
        """
        if len(equity_curve) == 0:
            return 0.0

        # Calculate running maximum
        peak = equity_curve.expanding().max()

        # Calculate percentage drawdown
        drawdown_pct = (equity_curve - peak) / peak * 100

        # Ulcer Index is RMS of drawdowns
        return float(np.sqrt((drawdown_pct**2).mean()))

    @staticmethod
    def risk_of_ruin(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        max_risk_per_trade: float = 0.02,
        account_ruin_threshold: float = 0.5,
    ) -> float:
        """
        Estimate risk of ruin probability.

        Args:
            win_rate: Win rate as decimal (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive value)
            max_risk_per_trade: Maximum risk per trade as decimal
            account_ruin_threshold: Account drawdown considered as ruin

        Returns:
            Estimated risk of ruin as percentage
        """
        if win_rate <= 0 or win_rate >= 1:
            return 100.0 if win_rate <= 0 else 0.0

        if avg_loss == 0:
            return 0.0

        # Calculate edge
        edge = win_rate * avg_win - (1 - win_rate) * avg_loss

        if edge <= 0:
            # Negative edge - high risk of ruin
            return 100.0

        # Simplified risk of ruin formula
        # Uses Kelly criterion derivative
        p = win_rate
        q = 1 - win_rate
        b = avg_win / avg_loss

        if b * p <= q:
            # Losing system
            return 100.0

        # Risk of ruin approximation
        a = (q / (p * b)) ** (1 / max_risk_per_trade)
        n = int(account_ruin_threshold / max_risk_per_trade)
        risk = a**n * 100

        return float(min(risk, 100.0))

    @staticmethod
    def information_ratio(
        returns: pd.Series,
        benchmark_returns: pd.Series,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate Information Ratio (active return / tracking error).

        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns
            periods_per_year: Number of periods per year

        Returns:
            Information Ratio
        """
        if len(returns) == 0 or len(benchmark_returns) == 0:
            return 0.0

        # Align series
        aligned = pd.DataFrame(
            {"portfolio": returns, "benchmark": benchmark_returns}
        ).dropna()

        if len(aligned) == 0:
            return 0.0

        # Active return
        active_return = aligned["portfolio"] - aligned["benchmark"]

        tracking_error = active_return.std()
        if tracking_error == 0:
            return 0.0

        return float(
            (active_return.mean() / tracking_error) * np.sqrt(periods_per_year)
        )

    @staticmethod
    def composite_score(
        sharpe: float,
        sortino: float,
        max_dd: float,
        win_rate: float,
        weights: Optional[dict] = None,
    ) -> float:
        """
        Calculate composite optimization score.

        Combines multiple metrics with customizable weights.

        Args:
            sharpe: Sharpe ratio
            sortino: Sortino ratio
            max_dd: Maximum drawdown percentage
            win_rate: Win rate percentage
            weights: Dict of weights for each metric

        Returns:
            Composite score
        """
        if weights is None:
            weights = {
                "sharpe": 0.3,
                "sortino": 0.3,
                "max_dd": 0.2,
                "win_rate": 0.2,
            }

        # Normalize metrics (higher is better for all)
        # Max drawdown is inverted and capped
        normalized_dd = max(0, 100 - max_dd) / 100

        score = (
            weights.get("sharpe", 0.3) * min(sharpe, 5.0) / 5.0
            + weights.get("sortino", 0.3) * min(sortino, 5.0) / 5.0
            + weights.get("max_dd", 0.2) * normalized_dd
            + weights.get("win_rate", 0.2) * min(win_rate, 100) / 100
        )

        return float(score * 100)  # Scale to 0-100
