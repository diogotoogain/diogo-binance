"""
Performance metrics for backtesting.

Provides comprehensive metrics calculation for strategy evaluation.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class PerformanceMetrics:
    """
    Calculate performance metrics for backtesting.

    All calculations use data from config - no hardcoded values.
    """

    def __init__(self, periods_per_year: int = 252):
        """
        Initialize performance metrics calculator.

        Args:
            periods_per_year: Number of trading periods per year
        """
        self.periods_per_year = periods_per_year

    def calculate_all(
        self,
        equity_curve: pd.Series,
        trades: pd.DataFrame,
        initial_capital: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate all performance metrics.

        Args:
            equity_curve: Series of portfolio equity values
            trades: DataFrame of executed trades
            initial_capital: Starting capital (uses first equity value if None)

        Returns:
            Dictionary of all metrics
        """
        if len(equity_curve) == 0:
            return self._empty_metrics()

        initial_capital = initial_capital or equity_curve.iloc[0]

        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        # Core metrics
        metrics = {
            "initial_capital": initial_capital,
            "final_equity": float(equity_curve.iloc[-1]),
            "total_return": self._total_return(equity_curve),
            "total_return_pct": self._total_return_pct(equity_curve),
            "sharpe_ratio": self._sharpe_ratio(returns),
            "sortino_ratio": self._sortino_ratio(returns),
            "calmar_ratio": self._calmar_ratio(returns, equity_curve),
            "max_drawdown": self._max_drawdown(equity_curve),
            "max_drawdown_duration": self._max_drawdown_duration(equity_curve),
            "volatility": self._volatility(returns),
            "downside_volatility": self._downside_volatility(returns),
            "var_95": self._value_at_risk(returns, 0.95),
            "cvar_95": self._conditional_var(returns, 0.95),
            "skewness": self._skewness(returns),
            "kurtosis": self._kurtosis(returns),
        }

        # Trade metrics
        if len(trades) > 0 and "pnl" in trades.columns:
            metrics.update(self._trade_metrics(trades))
        else:
            metrics.update(self._empty_trade_metrics())

        # Time metrics
        metrics.update(self._time_metrics(equity_curve))

        return metrics

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics dictionary."""
        return {
            "initial_capital": 0.0,
            "final_equity": 0.0,
            "total_return": 0.0,
            "total_return_pct": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration": 0,
            "volatility": 0.0,
            "downside_volatility": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            **self._empty_trade_metrics(),
        }

    def _empty_trade_metrics(self) -> Dict[str, Any]:
        """Return empty trade metrics."""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_trade_pnl": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
            "avg_trade_duration": 0.0,
            "expectancy": 0.0,
            "sqn": 0.0,
        }

    def _total_return(self, equity_curve: pd.Series) -> float:
        """Calculate total absolute return."""
        return float(equity_curve.iloc[-1] - equity_curve.iloc[0])

    def _total_return_pct(self, equity_curve: pd.Series) -> float:
        """Calculate total percentage return."""
        initial = equity_curve.iloc[0]
        if initial == 0:
            return 0.0
        return float((equity_curve.iloc[-1] / initial - 1) * 100)

    def _sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.0
    ) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / self.periods_per_year
        std = excess_returns.std()

        if std == 0 or np.isnan(std):
            return 0.0

        return float(
            (excess_returns.mean() / std) * np.sqrt(self.periods_per_year)
        )

    def _sortino_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.0
    ) -> float:
        """Calculate annualized Sortino ratio."""
        if len(returns) == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / self.periods_per_year
        downside = excess_returns[excess_returns < 0]

        if len(downside) == 0:
            return float(excess_returns.mean() * self.periods_per_year * 100)

        downside_std = downside.std()
        if downside_std == 0 or np.isnan(downside_std):
            return 0.0

        return float(
            (excess_returns.mean() / downside_std) * np.sqrt(self.periods_per_year)
        )

    def _calmar_ratio(
        self, returns: pd.Series, equity_curve: pd.Series
    ) -> float:
        """Calculate Calmar ratio."""
        if len(returns) == 0:
            return 0.0

        annual_return = returns.mean() * self.periods_per_year
        max_dd = self._max_drawdown(equity_curve)

        if max_dd == 0:
            return 0.0

        return float(annual_return / max_dd * 100)

    def _max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown percentage."""
        if len(equity_curve) == 0:
            return 0.0

        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak

        return float(abs(drawdown.min()) * 100)

    def _max_drawdown_duration(self, equity_curve: pd.Series) -> int:
        """Calculate maximum drawdown duration in periods."""
        if len(equity_curve) == 0:
            return 0

        peak = equity_curve.expanding().max()
        is_drawdown = equity_curve < peak

        # Find longest consecutive drawdown
        max_duration = 0
        current_duration = 0

        for in_dd in is_drawdown:
            if in_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def _volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        if len(returns) == 0:
            return 0.0
        return float(returns.std() * np.sqrt(self.periods_per_year) * 100)

    def _downside_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized downside volatility."""
        if len(returns) == 0:
            return 0.0

        downside = returns[returns < 0]
        if len(downside) == 0:
            return 0.0

        return float(downside.std() * np.sqrt(self.periods_per_year) * 100)

    def _value_at_risk(
        self, returns: pd.Series, confidence: float = 0.95
    ) -> float:
        """Calculate Value at Risk."""
        if len(returns) == 0:
            return 0.0
        return float(abs(np.percentile(returns, (1 - confidence) * 100)) * 100)

    def _conditional_var(
        self, returns: pd.Series, confidence: float = 0.95
    ) -> float:
        """Calculate Conditional VaR (Expected Shortfall)."""
        if len(returns) == 0:
            return 0.0

        var = np.percentile(returns, (1 - confidence) * 100)
        cvar = returns[returns <= var].mean()

        return float(abs(cvar) * 100) if not np.isnan(cvar) else 0.0

    def _skewness(self, returns: pd.Series) -> float:
        """Calculate returns skewness."""
        if len(returns) < 3:
            return 0.0
        return float(returns.skew())

    def _kurtosis(self, returns: pd.Series) -> float:
        """Calculate returns kurtosis."""
        if len(returns) < 4:
            return 0.0
        return float(returns.kurtosis())

    def _trade_metrics(self, trades: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trade-based metrics."""
        if len(trades) == 0 or "pnl" not in trades.columns:
            return self._empty_trade_metrics()

        pnl = trades["pnl"]
        winning = trades[trades["pnl"] > 0]
        losing = trades[trades["pnl"] < 0]

        # Basic counts
        total = len(trades)
        n_winning = len(winning)
        n_losing = len(losing)

        # Win rate
        win_rate = (n_winning / total * 100) if total > 0 else 0.0

        # Profit factor
        gross_profit = winning["pnl"].sum() if n_winning > 0 else 0.0
        gross_loss = abs(losing["pnl"].sum()) if n_losing > 0 else 0.0
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0
        )

        # Average P&L
        avg_pnl = float(pnl.mean())
        avg_win = float(winning["pnl"].mean()) if n_winning > 0 else 0.0
        avg_loss = float(losing["pnl"].mean()) if n_losing > 0 else 0.0

        # Max win/loss
        max_win = float(pnl.max())
        max_loss = float(pnl.min())

        # Duration
        avg_duration = 0.0
        if "duration" in trades.columns:
            avg_duration = float(trades["duration"].mean())

        # Expectancy
        expectancy = avg_pnl

        # System Quality Number (SQN)
        sqn = 0.0
        if len(pnl) > 0 and pnl.std() > 0:
            sqn = float(pnl.mean() / pnl.std() * np.sqrt(len(pnl)))

        return {
            "total_trades": total,
            "winning_trades": n_winning,
            "losing_trades": n_losing,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_trade_pnl": avg_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "max_win": max_win,
            "max_loss": max_loss,
            "avg_trade_duration": avg_duration,
            "expectancy": expectancy,
            "sqn": sqn,
        }

    def _time_metrics(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """Calculate time-based metrics."""
        if len(equity_curve) == 0:
            return {
                "trading_days": 0,
                "positive_days": 0,
                "negative_days": 0,
                "positive_day_pct": 0.0,
            }

        returns = equity_curve.pct_change().dropna()
        positive = len(returns[returns > 0])
        negative = len(returns[returns < 0])
        total = len(returns)

        return {
            "trading_days": total,
            "positive_days": positive,
            "negative_days": negative,
            "positive_day_pct": (positive / total * 100) if total > 0 else 0.0,
        }

    def calculate_rolling_metrics(
        self,
        equity_curve: pd.Series,
        window: int = 30,
    ) -> pd.DataFrame:
        """
        Calculate rolling metrics over time.

        Args:
            equity_curve: Series of equity values
            window: Rolling window size

        Returns:
            DataFrame with rolling metrics
        """
        returns = equity_curve.pct_change().dropna()

        rolling = pd.DataFrame(index=returns.index)

        # Rolling returns
        rolling["return"] = returns.rolling(window).mean() * self.periods_per_year * 100

        # Rolling volatility
        rolling["volatility"] = (
            returns.rolling(window).std() * np.sqrt(self.periods_per_year) * 100
        )

        # Rolling Sharpe
        rolling["sharpe"] = rolling["return"] / rolling["volatility"]
        rolling["sharpe"] = rolling["sharpe"].replace([np.inf, -np.inf], 0)

        # Rolling max drawdown
        def rolling_max_dd(x):
            if len(x) == 0:
                return 0.0
            eq = (1 + x).cumprod()
            peak = eq.expanding().max()
            dd = (eq - peak) / peak
            return abs(dd.min()) * 100

        rolling["max_drawdown"] = returns.rolling(window).apply(rolling_max_dd)

        return rolling

    def generate_summary(self, metrics: Dict[str, Any]) -> str:
        """
        Generate text summary of metrics.

        Args:
            metrics: Dictionary of calculated metrics

        Returns:
            Formatted summary string
        """
        return f"""
╔═══════════════════════════════════════════════════════════════╗
║                    📊 PERFORMANCE METRICS                      ║
╠═══════════════════════════════════════════════════════════════╣
║  RETURNS                                                       ║
║    Total Return:         {metrics.get('total_return_pct', 0):>10.2f}%                     ║
║    Sharpe Ratio:         {metrics.get('sharpe_ratio', 0):>10.2f}                      ║
║    Sortino Ratio:        {metrics.get('sortino_ratio', 0):>10.2f}                      ║
║    Calmar Ratio:         {metrics.get('calmar_ratio', 0):>10.2f}                      ║
╠═══════════════════════════════════════════════════════════════╣
║  RISK                                                          ║
║    Max Drawdown:         {metrics.get('max_drawdown', 0):>10.2f}%                     ║
║    Volatility:           {metrics.get('volatility', 0):>10.2f}%                     ║
║    VaR (95%):            {metrics.get('var_95', 0):>10.2f}%                     ║
║    CVaR (95%):           {metrics.get('cvar_95', 0):>10.2f}%                     ║
╠═══════════════════════════════════════════════════════════════╣
║  TRADES                                                        ║
║    Total Trades:         {metrics.get('total_trades', 0):>10}                      ║
║    Win Rate:             {metrics.get('win_rate', 0):>10.2f}%                     ║
║    Profit Factor:        {metrics.get('profit_factor', 0):>10.2f}                      ║
║    Avg Trade P&L:        {metrics.get('avg_trade_pnl', 0):>10.2f}                      ║
║    SQN:                  {metrics.get('sqn', 0):>10.2f}                      ║
╚═══════════════════════════════════════════════════════════════╝
"""
