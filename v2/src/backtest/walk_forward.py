"""
Walk-Forward Optimization for robust strategy validation.

Implements rolling window train/test methodology to prevent overfitting.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from .engine import BacktestEngine
from ..optimization.optuna_optimizer import OptunaOptimizer

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """Represents a single walk-forward window."""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    window_number: int


@dataclass
class WalkForwardResult:
    """Result from a single walk-forward window."""

    window: WalkForwardWindow
    best_params: Dict[str, Any]
    train_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    optimization_value: float


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization for robust validation.

    All parameters come from config - no hardcoded values.

    Config parameters (backtest.walk_forward):
    - enabled: Whether walk-forward is enabled
    - train_days: Training window in days
    - test_days: Test window in days
    - step_days: Step size between windows
    """

    def __init__(
        self,
        config: Dict[str, Any],
        optimizer: OptunaOptimizer,
        backtest_engine: BacktestEngine,
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            config: Full configuration dictionary
            optimizer: Optuna optimizer instance
            backtest_engine: Backtest engine instance
        """
        self.config = config.get("backtest", {}).get("walk_forward", {})
        self.full_config = config
        self.optimizer = optimizer
        self.engine = backtest_engine

        # Configuration from YAML
        self.train_days = self.config.get("train_days", 60)
        self.test_days = self.config.get("test_days", 14)
        self.step_days = self.config.get("step_days", 7)

        # Results storage
        self.results: List[WalkForwardResult] = []

    def run(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        strategy_factory: Callable[[Dict[str, Any]], Any],
        n_trials_per_window: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[WalkForwardResult]:
        """
        Execute Walk-Forward Optimization.

        Process:
        1. Divide data into train/test windows
        2. For each window:
           a. Optimize parameters on training data
           b. Test with optimized parameters on test data
        3. Aggregate results

        Args:
            data: OHLCV market data with DatetimeIndex
            features: Feature data
            strategy_factory: Function that creates strategy from params
            n_trials_per_window: Optuna trials per window
            progress_callback: Progress callback

        Returns:
            List of results per window
        """
        logger.info(
            f"Starting Walk-Forward Optimization: "
            f"train={self.train_days}d, test={self.test_days}d, step={self.step_days}d"
        )

        # Generate windows
        windows = self._generate_windows(data)
        logger.info(f"Generated {len(windows)} walk-forward windows")

        self.results = []

        for i, window in enumerate(windows):
            logger.info(
                f"Processing window {i+1}/{len(windows)}: "
                f"train {window.train_start} to {window.train_end}, "
                f"test {window.test_start} to {window.test_end}"
            )

            # Extract train and test data
            train_data = data.iloc[window.train_start_idx : window.train_end_idx]
            train_features = features.iloc[window.train_start_idx : window.train_end_idx]

            test_data = data.iloc[window.test_start_idx : window.test_end_idx]
            test_features = features.iloc[window.test_start_idx : window.test_end_idx]

            # Optimize on training data
            best_params = self._optimize_window(
                train_data,
                train_features,
                strategy_factory,
                n_trials_per_window,
                window.window_number,
            )

            # Get training metrics with best params
            train_metrics = self._backtest_window(
                train_data, train_features, strategy_factory, best_params
            )

            # Test with optimized parameters
            test_metrics = self._backtest_window(
                test_data, test_features, strategy_factory, best_params
            )

            result = WalkForwardResult(
                window=window,
                best_params=best_params,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                optimization_value=test_metrics.get("sharpe_ratio", 0.0),
            )

            self.results.append(result)

            if progress_callback:
                progress_callback(i + 1, len(windows))

        logger.info("Walk-Forward Optimization complete")
        return self.results

    def _generate_windows(self, data: pd.DataFrame) -> List[WalkForwardWindow]:
        """
        Generate walk-forward windows.

        Args:
            data: Market data with datetime index

        Returns:
            List of windows
        """
        windows = []

        # Convert days to approximate bar counts
        # Assuming daily data, 1 day = 1 bar; for minute data, adjust accordingly
        bars_per_day = self._estimate_bars_per_day(data)
        train_bars = self.train_days * bars_per_day
        test_bars = self.test_days * bars_per_day
        step_bars = self.step_days * bars_per_day

        # Generate windows
        start_idx = 0
        window_num = 0

        while start_idx + train_bars + test_bars <= len(data):
            train_end_idx = start_idx + train_bars
            test_end_idx = train_end_idx + test_bars

            # Get dates from data
            train_start_date = self._get_date(data, start_idx)
            train_end_date = self._get_date(data, train_end_idx - 1)
            test_start_date = self._get_date(data, train_end_idx)
            test_end_date = self._get_date(data, test_end_idx - 1)

            window = WalkForwardWindow(
                train_start=train_start_date,
                train_end=train_end_date,
                test_start=test_start_date,
                test_end=test_end_date,
                train_start_idx=start_idx,
                train_end_idx=train_end_idx,
                test_start_idx=train_end_idx,
                test_end_idx=test_end_idx,
                window_number=window_num,
            )

            windows.append(window)

            # Move to next window
            start_idx += step_bars
            window_num += 1

        return windows

    def _estimate_bars_per_day(self, data: pd.DataFrame) -> int:
        """Estimate number of bars per day from data."""
        if len(data) < 2:
            return 1

        # Check if index is datetime
        if isinstance(data.index, pd.DatetimeIndex):
            # Sample time difference
            time_diff = (data.index[-1] - data.index[0]) / len(data)
            if time_diff.total_seconds() < 3600:  # Less than 1 hour
                return 1440  # Assume minute data
            elif time_diff.total_seconds() < 86400:  # Less than 1 day
                return int(86400 / time_diff.total_seconds())
            else:
                return 1  # Daily or longer

        return 1  # Default to daily

    def _get_date(self, data: pd.DataFrame, idx: int) -> datetime:
        """Get datetime from data index."""
        if isinstance(data.index, pd.DatetimeIndex):
            return data.index[idx].to_pydatetime()
        elif "timestamp" in data.columns:
            return pd.Timestamp(data.iloc[idx]["timestamp"]).to_pydatetime()
        else:
            return datetime.now()

    def _optimize_window(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        strategy_factory: Callable,
        n_trials: Optional[int],
        window_num: int,
    ) -> Dict[str, Any]:
        """
        Optimize parameters for a single window.

        Args:
            data: Training data
            features: Training features
            strategy_factory: Factory to create strategy
            n_trials: Number of optimization trials
            window_num: Window number for study naming

        Returns:
            Best parameters
        """
        # Create study for this window
        study_name = f"{self.full_config.get('optimization', {}).get('study_name', 'wf')}_w{window_num}"

        self.optimizer.create_study(
            study_name=study_name,
            load_if_exists=False,  # Fresh study per window
        )

        # Create objective function
        def objective(trial):
            params = self.optimizer.suggest_params(trial)
            strategy = strategy_factory(params)

            # Run backtest
            results = self.engine.run(data, strategy, features)
            return results.get("sharpe_ratio", 0.0)

        # Run optimization
        n_trials = n_trials or self.full_config.get("optimization", {}).get(
            "n_trials", 100
        )

        # Use fewer trials for walk-forward
        n_trials = min(n_trials, 200)

        best_params = self.optimizer.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=False,
        )

        return best_params

    def _backtest_window(
        self,
        data: pd.DataFrame,
        features: pd.DataFrame,
        strategy_factory: Callable,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Run backtest for a window with given parameters.

        Args:
            data: Market data
            features: Features
            strategy_factory: Factory to create strategy
            params: Strategy parameters

        Returns:
            Backtest results
        """
        strategy = strategy_factory(params)
        results = self.engine.run(data, strategy, features)

        # Return only metrics, not large data
        return {
            k: v
            for k, v in results.items()
            if k not in ["equity_curve", "trades"]
        }

    def get_aggregated_results(self) -> Dict[str, Any]:
        """
        Get aggregated results across all windows.

        Returns:
            Aggregated metrics
        """
        if not self.results:
            return {}

        # Collect metrics from all windows
        train_sharpes = [r.train_metrics.get("sharpe_ratio", 0) for r in self.results]
        test_sharpes = [r.test_metrics.get("sharpe_ratio", 0) for r in self.results]
        test_returns = [r.test_metrics.get("total_return_pct", 0) for r in self.results]
        test_drawdowns = [r.test_metrics.get("max_drawdown", 0) for r in self.results]

        import numpy as np

        return {
            "n_windows": len(self.results),
            "train_sharpe_mean": float(np.mean(train_sharpes)),
            "train_sharpe_std": float(np.std(train_sharpes)),
            "test_sharpe_mean": float(np.mean(test_sharpes)),
            "test_sharpe_std": float(np.std(test_sharpes)),
            "test_return_mean": float(np.mean(test_returns)),
            "test_return_std": float(np.std(test_returns)),
            "test_drawdown_mean": float(np.mean(test_drawdowns)),
            "test_drawdown_max": float(max(test_drawdowns)),
            "out_of_sample_efficiency": self._calculate_oos_efficiency(),
            "robustness_score": self._calculate_robustness_score(),
        }

    def _calculate_oos_efficiency(self) -> float:
        """
        Calculate out-of-sample efficiency.

        Compares test performance to training performance.

        Returns:
            OOS efficiency ratio (1.0 = perfect, <1.0 = degradation)
        """
        if not self.results:
            return 0.0

        train_sharpes = [
            r.train_metrics.get("sharpe_ratio", 0) for r in self.results
        ]
        test_sharpes = [
            r.test_metrics.get("sharpe_ratio", 0) for r in self.results
        ]

        import numpy as np

        avg_train = np.mean(train_sharpes)
        avg_test = np.mean(test_sharpes)

        if avg_train <= 0:
            return 0.0

        return float(avg_test / avg_train)

    def _calculate_robustness_score(self) -> float:
        """
        Calculate robustness score.

        Based on consistency of test results across windows.

        Returns:
            Robustness score (0-100)
        """
        if not self.results:
            return 0.0

        import numpy as np

        test_sharpes = [
            r.test_metrics.get("sharpe_ratio", 0) for r in self.results
        ]

        # Calculate metrics
        positive_windows = sum(1 for s in test_sharpes if s > 0)
        consistency = positive_windows / len(test_sharpes)

        # Coefficient of variation (lower = more consistent)
        mean_sharpe = np.mean(test_sharpes)
        std_sharpe = np.std(test_sharpes)

        if mean_sharpe > 0:
            cv = std_sharpe / mean_sharpe
            stability = max(0, 1 - cv)
        else:
            stability = 0

        # Combined score
        score = (consistency * 50 + stability * 50)
        return float(score)

    def generate_report(self) -> str:
        """
        Generate walk-forward report.

        Returns:
            Formatted report string
        """
        if not self.results:
            return "No walk-forward results available"

        aggregated = self.get_aggregated_results()

        report = f"""
╔═══════════════════════════════════════════════════════════════╗
║                WALK-FORWARD OPTIMIZATION REPORT                ║
╠═══════════════════════════════════════════════════════════════╣
║  Configuration                                                 ║
║    Train Window:         {self.train_days:>10} days                      ║
║    Test Window:          {self.test_days:>10} days                      ║
║    Step Size:            {self.step_days:>10} days                      ║
║    Total Windows:        {aggregated['n_windows']:>10}                           ║
╠═══════════════════════════════════════════════════════════════╣
║  Training Performance                                          ║
║    Sharpe (mean):        {aggregated['train_sharpe_mean']:>10.2f}                        ║
║    Sharpe (std):         {aggregated['train_sharpe_std']:>10.2f}                        ║
╠═══════════════════════════════════════════════════════════════╣
║  Test Performance (Out-of-Sample)                             ║
║    Sharpe (mean):        {aggregated['test_sharpe_mean']:>10.2f}                        ║
║    Sharpe (std):         {aggregated['test_sharpe_std']:>10.2f}                        ║
║    Return (mean):        {aggregated['test_return_mean']:>10.2f}%                       ║
║    Max Drawdown (avg):   {aggregated['test_drawdown_mean']:>10.2f}%                       ║
╠═══════════════════════════════════════════════════════════════╣
║  Robustness Metrics                                           ║
║    OOS Efficiency:       {aggregated['out_of_sample_efficiency']:>10.2f}                        ║
║    Robustness Score:     {aggregated['robustness_score']:>10.2f}/100                   ║
╚═══════════════════════════════════════════════════════════════╝
"""

        # Add per-window details
        report += "\nPer-Window Results:\n"
        report += "-" * 60 + "\n"

        for r in self.results:
            report += (
                f"  Window {r.window.window_number + 1}: "
                f"Train Sharpe={r.train_metrics.get('sharpe_ratio', 0):.2f}, "
                f"Test Sharpe={r.test_metrics.get('sharpe_ratio', 0):.2f}, "
                f"Test Return={r.test_metrics.get('total_return_pct', 0):.2f}%\n"
            )

        return report

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to DataFrame.

        Returns:
            DataFrame with all window results
        """
        rows = []
        for r in self.results:
            row = {
                "window": r.window.window_number,
                "train_start": r.window.train_start,
                "train_end": r.window.train_end,
                "test_start": r.window.test_start,
                "test_end": r.window.test_end,
                "train_sharpe": r.train_metrics.get("sharpe_ratio", 0),
                "train_return": r.train_metrics.get("total_return_pct", 0),
                "test_sharpe": r.test_metrics.get("sharpe_ratio", 0),
                "test_return": r.test_metrics.get("total_return_pct", 0),
                "test_drawdown": r.test_metrics.get("max_drawdown", 0),
            }
            rows.append(row)

        return pd.DataFrame(rows)
