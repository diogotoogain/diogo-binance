"""
Parameter Optimizer for Walk-Forward Strategy Optimization.

Optimizes strategy parameters using walk-forward validation
to prevent overfitting.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for parameter optimization."""

    metric: str = "sortino"  # Primary optimization metric
    train_pct: float = 0.8   # Training percentage per window
    walk_forward_windows: int = 12  # Number of walk-forward windows
    min_samples_per_window: int = 1000  # Minimum samples per window


@dataclass
class OptimizationResult:
    """Result from optimization."""

    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    window_results: List[Dict[str, Any]]
    aggregated_metrics: Dict[str, float]


class ParameterOptimizer:
    """
    Optimizes strategy parameters using Walk-Forward Optimization.

    Parameters that can be optimized:
    - ofi_threshold: [0.1, 0.2, 0.3, 0.4, 0.5]
    - tfi_threshold: [0.1, 0.2, 0.3, 0.4, 0.5]
    - holding_seconds: [10, 30, 60, 120, 300]
    - max_adx: [20, 25, 30, 35, 40]
    - sl_atr_mult: [0.5, 1.0, 1.5, 2.0]
    - tp_atr_mult: [1.0, 1.5, 2.0, 3.0]

    Uses Walk-Forward Optimization:
    1. Train on 80% of data in window
    2. Test on 20% following
    3. Move window forward
    4. Repeat
    """

    # Default parameter grid
    DEFAULT_PARAM_GRID = {
        'ofi_threshold': [0.1, 0.2, 0.3, 0.4, 0.5],
        'tfi_threshold': [0.1, 0.2, 0.3, 0.4, 0.5],
        'holding_seconds': [10, 30, 60, 120, 300],
        'max_adx': [20, 25, 30, 35, 40]
    }

    def __init__(self, config: OptimizationConfig):
        """
        Initialize ParameterOptimizer.

        Args:
            config: Optimization configuration
        """
        self.config = config
        self.results_history: List[Dict[str, Any]] = []

    def optimize(
        self,
        data: pd.DataFrame,
        param_grid: Optional[Dict[str, List]] = None,
        evaluator: Optional[Callable] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> OptimizationResult:
        """
        Optimize parameters with walk-forward validation.

        Args:
            data: Historical data for optimization
            param_grid: Dictionary of parameter names to lists of values
            evaluator: Function that evaluates a parameter set on data
            progress_callback: Optional callback for progress updates

        Returns:
            OptimizationResult with best parameters and metrics
        """
        if param_grid is None:
            param_grid = self.DEFAULT_PARAM_GRID

        if evaluator is None:
            evaluator = self._default_evaluator

        logger.info(
            f"Starting Walk-Forward Optimization with "
            f"{self.config.walk_forward_windows} windows"
        )

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(product(*param_values))

        logger.info(f"Testing {len(all_combinations)} parameter combinations")

        # Generate walk-forward windows
        windows = self._generate_windows(data)

        all_results = []
        window_results = []

        total_iterations = len(all_combinations) * len(windows)
        current_iteration = 0

        for params_tuple in all_combinations:
            params = dict(zip(param_names, params_tuple))

            # Test this parameter set across all windows
            window_scores = []

            for window in windows:
                train_data = data.iloc[window['train_start']:window['train_end']]
                test_data = data.iloc[window['test_start']:window['test_end']]

                try:
                    # Evaluate on test data (out-of-sample)
                    score = evaluator(params, train_data, test_data)
                    window_scores.append(score)
                except Exception as e:
                    logger.warning(f"Evaluation failed for {params}: {e}")
                    window_scores.append(float('-inf'))

                current_iteration += 1
                if progress_callback is not None and current_iteration % 10 == 0:
                    progress_callback(current_iteration, total_iterations)

            # Aggregate scores across windows
            avg_score = np.mean(window_scores) if window_scores else 0.0
            std_score = np.std(window_scores) if window_scores else 0.0

            all_results.append({
                'params': params,
                'avg_score': avg_score,
                'std_score': std_score,
                'window_scores': window_scores,
                'min_score': min(window_scores) if window_scores else 0.0,
                'max_score': max(window_scores) if window_scores else 0.0
            })

        # Find best parameters (by average score)
        all_results.sort(key=lambda x: x['avg_score'], reverse=True)
        best_result = all_results[0]

        logger.info(
            f"Best parameters: {best_result['params']}, "
            f"Avg score: {best_result['avg_score']:.4f}"
        )

        # Calculate aggregated metrics
        aggregated = self._calculate_aggregated_metrics(all_results, windows)

        return OptimizationResult(
            best_params=best_result['params'],
            best_score=best_result['avg_score'],
            all_results=all_results,
            window_results=window_results,
            aggregated_metrics=aggregated
        )

    def _generate_windows(self, data: pd.DataFrame) -> List[Dict[str, int]]:
        """
        Generate walk-forward windows.

        Args:
            data: Full dataset

        Returns:
            List of window definitions with train/test indices
        """
        n_samples = len(data)
        n_windows = self.config.walk_forward_windows

        # Calculate window size
        window_size = n_samples // n_windows
        train_size = int(window_size * self.config.train_pct)
        test_size = window_size - train_size

        # Ensure minimum samples
        if train_size < self.config.min_samples_per_window:
            # Reduce number of windows
            train_size = self.config.min_samples_per_window
            test_size = int(train_size * (1 - self.config.train_pct) / self.config.train_pct)
            window_size = train_size + test_size
            n_windows = max(1, n_samples // window_size)

        windows = []
        for i in range(n_windows):
            start = i * window_size

            # Avoid going past the data
            if start + window_size > n_samples:
                break

            windows.append({
                'window_num': i,
                'train_start': start,
                'train_end': start + train_size,
                'test_start': start + train_size,
                'test_end': min(start + window_size, n_samples)
            })

        logger.info(
            f"Generated {len(windows)} windows, "
            f"train_size={train_size}, test_size={test_size}"
        )

        return windows

    def _default_evaluator(
        self,
        params: Dict[str, Any],
        train_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> float:
        """
        Default parameter evaluator using simple momentum strategy.

        Args:
            params: Parameters to evaluate
            train_data: Training data (for parameter fitting)
            test_data: Test data (for evaluation)

        Returns:
            Score (higher is better)
        """
        # Simple momentum-based evaluation
        if len(test_data) < 10:
            return 0.0

        close = test_data['close'].values
        returns = np.diff(close) / close[:-1]

        if len(returns) == 0:
            return 0.0

        # Calculate Sortino-like metric
        mean_return = np.mean(returns)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            # Scale mean return when no downside for comparable magnitudes
            # Using annualization factor for minute data (525600 minutes/year)
            return mean_return * np.sqrt(525600)

        sortino = mean_return / np.std(downside_returns)
        return float(sortino)

    def _calculate_aggregated_metrics(
        self,
        all_results: List[Dict[str, Any]],
        windows: List[Dict[str, int]]
    ) -> Dict[str, float]:
        """Calculate aggregated optimization metrics."""
        if not all_results:
            return {}

        best = all_results[0]
        scores = [r['avg_score'] for r in all_results]

        return {
            'best_score': best['avg_score'],
            'best_score_std': best['std_score'],
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'n_windows': len(windows),
            'n_combinations': len(all_results),
            'robustness': best['avg_score'] - best['std_score'] if best['std_score'] > 0 else best['avg_score']
        }

    def grid_search(
        self,
        data: pd.DataFrame,
        param_grid: Dict[str, List],
        evaluator: Callable,
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        Simple grid search without walk-forward.

        Args:
            data: Dataset
            param_grid: Parameter grid
            evaluator: Evaluation function
            n_splits: Number of cross-validation splits

        Returns:
            Best parameters and scores
        """
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(product(*param_values))

        results = []
        split_size = len(data) // n_splits

        for params_tuple in all_combinations:
            params = dict(zip(param_names, params_tuple))
            scores = []

            for i in range(n_splits - 1):
                train_end = (i + 1) * split_size
                test_end = (i + 2) * split_size
                test_end = min(test_end, len(data))

                train_data = data.iloc[:train_end]
                test_data = data.iloc[train_end:test_end]

                try:
                    score = evaluator(params, train_data, test_data)
                    scores.append(score)
                except Exception:
                    scores.append(0.0)

            avg_score = np.mean(scores) if scores else 0.0
            results.append({'params': params, 'avg_score': avg_score, 'scores': scores})

        results.sort(key=lambda x: x['avg_score'], reverse=True)

        return {
            'best_params': results[0]['params'] if results else {},
            'best_score': results[0]['avg_score'] if results else 0.0,
            'all_results': results
        }

    def generate_report(self, result: OptimizationResult) -> str:
        """
        Generate optimization report.

        Args:
            result: Optimization result

        Returns:
            Formatted report string
        """
        report = f"""
╔═══════════════════════════════════════════════════════════════╗
║              PARAMETER OPTIMIZATION REPORT                     ║
╠═══════════════════════════════════════════════════════════════╣
║  Configuration                                                 ║
║    Metric:               {self.config.metric:>15}                     ║
║    Train Percentage:     {self.config.train_pct * 100:>15.1f}%                    ║
║    Walk-Forward Windows: {self.config.walk_forward_windows:>15}                     ║
╠═══════════════════════════════════════════════════════════════╣
║  Results                                                       ║
║    Best Score:           {result.best_score:>15.4f}                     ║
║    Combinations Tested:  {len(result.all_results):>15}                     ║
╠═══════════════════════════════════════════════════════════════╣
║  Best Parameters                                               ║
"""
        for param, value in result.best_params.items():
            report += f"║    {param:20}: {str(value):>15}                     ║\n"

        report += "╚═══════════════════════════════════════════════════════════════╝\n"

        return report


def create_strategy_evaluator(simulator_factory: Callable) -> Callable:
    """
    Create an evaluator function for strategy optimization.

    Args:
        simulator_factory: Function that creates a simulator with given params

    Returns:
        Evaluator function compatible with ParameterOptimizer
    """
    def evaluator(
        params: Dict[str, Any],
        train_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> float:
        # Create simulator with these parameters
        simulator = simulator_factory(params)

        # Run on test data
        result = simulator.run(test_data)

        # Return optimization metric (sortino by default)
        metrics = result.metrics
        return metrics.get('sortino_ratio', metrics.get('sharpe_ratio', 0.0))

    return evaluator
