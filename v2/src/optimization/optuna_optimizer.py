"""
Optuna-based parameter optimization.

Provides high-level interface for optimizing trading strategy parameters.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import optuna
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import CmaEsSampler, RandomSampler, TPESampler

from .param_space import ParamSpace

logger = logging.getLogger(__name__)


class OptunaOptimizer:
    """
    Optimize parameters using Optuna framework.

    All parameters come from config - no hardcoded values.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Optuna optimizer.

        Args:
            config: Full configuration dictionary
        """
        self.full_config = config
        self.config = config.get("optimization", {})
        self.param_space = ParamSpace(config)
        self.study: Optional[optuna.Study] = None
        self._best_params: Optional[Dict[str, Any]] = None

    def create_study(
        self,
        direction: str = "maximize",
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        load_if_exists: bool = True,
    ) -> optuna.Study:
        """
        Create or load an Optuna study.

        Args:
            direction: Optimization direction ('maximize' or 'minimize')
            study_name: Name for the study (uses config if None)
            storage: Storage URL (uses config if None)
            load_if_exists: Whether to load existing study

        Returns:
            Optuna Study object
        """
        # Get study name from config or parameter
        study_name = study_name or self.config.get("study_name", "btcusdt_v2")

        # Build storage URL
        if storage is None:
            storage_type = self.config.get("storage_type", "sqlite")
            if storage_type == "sqlite":
                storage = f"sqlite:///v2/data/optuna_{study_name}.db"
            else:
                storage = None  # Use in-memory

        # Create pruner if enabled
        pruner = None
        pruning_config = self.config.get("pruning", {})
        if pruning_config.get("enabled", True):
            min_trials = pruning_config.get("min_trials", 50)
            pruner = MedianPruner(
                n_startup_trials=min_trials,
                n_warmup_steps=pruning_config.get("patience", 20),
            )

        # Create sampler
        sampler = self._create_sampler()

        # Create study
        self.study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=load_if_exists,
        )

        logger.info(
            f"Created/loaded study '{study_name}' with {len(self.study.trials)} existing trials"
        )

        return self.study

    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler based on config."""
        sampler_config = self.config.get("sampler", {})
        sampler_type = sampler_config.get("type", "TPE")
        n_startup_trials = sampler_config.get("n_startup_trials", 10)

        if sampler_type == "TPE":
            return TPESampler(
                n_startup_trials=n_startup_trials,
                multivariate=sampler_config.get("multivariate", True),
            )
        elif sampler_type == "CMA-ES":
            return CmaEsSampler(n_startup_trials=n_startup_trials)
        elif sampler_type == "Random":
            return RandomSampler()
        else:
            logger.warning(f"Unknown sampler type '{sampler_type}', using TPE")
            return TPESampler(n_startup_trials=n_startup_trials)

    def optimize(
        self,
        objective_fn: Callable[[optuna.Trial], float],
        n_trials: Optional[int] = None,
        n_jobs: Optional[int] = None,
        timeout: Optional[float] = None,
        show_progress_bar: bool = True,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, Any]:
        """
        Run optimization.

        Args:
            objective_fn: Objective function that takes Trial and returns metric
            n_trials: Number of trials (uses config if None)
            n_jobs: Number of parallel jobs (-1 for all cores, uses config if None)
            timeout: Maximum optimization time in seconds
            show_progress_bar: Whether to show progress bar
            callbacks: List of callback functions

        Returns:
            Best parameters found
        """
        if self.study is None:
            self.create_study()

        n_trials = n_trials or self.config.get("n_trials", 1000)
        n_jobs = n_jobs if n_jobs is not None else self.config.get("n_jobs", -1)

        logger.info(f"Starting optimization with {n_trials} trials, {n_jobs} jobs")

        self.study.optimize(
            objective_fn,
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            show_progress_bar=show_progress_bar,
            callbacks=callbacks,
        )

        self._best_params = self.study.best_params
        logger.info(f"Optimization complete. Best value: {self.study.best_value:.4f}")

        return self._best_params

    def suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest parameters for a trial.

        Uses ParamSpace to generate suggestions based on schema.

        Args:
            trial: Optuna Trial object

        Returns:
            Dictionary of suggested parameters
        """
        return self.param_space.suggest(trial)

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get best parameters from optimization."""
        if self.study is None:
            return None
        return self.study.best_params

    def get_best_value(self) -> Optional[float]:
        """Get best objective value from optimization."""
        if self.study is None:
            return None
        return self.study.best_value

    def get_best_trial(self) -> Optional[optuna.trial.FrozenTrial]:
        """Get best trial from optimization."""
        if self.study is None:
            return None
        return self.study.best_trial

    def get_trials_dataframe(self) -> Optional["pd.DataFrame"]:
        """Get trials as pandas DataFrame."""
        if self.study is None:
            return None
        return self.study.trials_dataframe()

    def get_param_importances(self) -> Dict[str, float]:
        """
        Get parameter importances using fANOVA.

        Returns:
            Dictionary mapping parameter names to importance scores
        """
        if self.study is None or len(self.study.trials) < 10:
            return {}

        try:
            importances = optuna.importance.get_param_importances(self.study)
            return dict(importances)
        except Exception as e:
            logger.warning(f"Could not calculate param importances: {e}")
            return {}

    def save_best_params(self, path: Union[str, Path]) -> None:
        """
        Save best parameters to YAML file.

        Args:
            path: Output file path
        """
        import yaml

        if self.study is None:
            raise ValueError("No study to save params from")

        params = {
            "best_params": self.study.best_params,
            "best_value": self.study.best_value,
            "n_trials": len(self.study.trials),
            "study_name": self.study.study_name,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(params, f, default_flow_style=False)

        logger.info(f"Saved best params to {path}")

    def load_best_params(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load best parameters from YAML file.

        Args:
            path: Input file path

        Returns:
            Best parameters dictionary
        """
        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        self._best_params = data.get("best_params", {})
        return self._best_params

    def apply_best_params_to_config(
        self, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Apply best parameters to configuration.

        Args:
            config: Config to update (uses original if None)

        Returns:
            Updated configuration
        """
        if self._best_params is None:
            if self.study is not None:
                self._best_params = self.study.best_params
            else:
                raise ValueError("No best params available")

        config = config or self.full_config
        return self.param_space.apply_params_to_config(config, self._best_params)

    def visualize_optimization_history(self) -> None:
        """Display optimization history plot."""
        if self.study is None:
            return

        try:
            import optuna.visualization as vis

            fig = vis.plot_optimization_history(self.study)
            fig.show()
        except ImportError:
            logger.warning("plotly not installed for visualization")

    def visualize_param_importances(self) -> None:
        """Display parameter importance plot."""
        if self.study is None or len(self.study.trials) < 10:
            return

        try:
            import optuna.visualization as vis

            fig = vis.plot_param_importances(self.study)
            fig.show()
        except ImportError:
            logger.warning("plotly not installed for visualization")

    def create_objective_wrapper(
        self,
        backtest_fn: Callable,
        metric: str = "sharpe_ratio",
    ) -> Callable[[optuna.Trial], float]:
        """
        Create objective function wrapper for backtesting.

        Args:
            backtest_fn: Function that takes config and returns results dict
            metric: Metric to optimize from results

        Returns:
            Objective function for Optuna
        """

        def objective(trial: optuna.Trial) -> float:
            # Get suggested parameters
            params = self.suggest_params(trial)

            # Apply to config
            trial_config = self.param_space.apply_params_to_config(
                self.full_config, params
            )

            # Run backtest
            try:
                results = backtest_fn(trial_config)
                value = results.get(metric, 0.0)

                # Report intermediate values for pruning
                trial.report(value, step=0)

                if trial.should_prune():
                    raise optuna.TrialPruned()

                return value

            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float("-inf")

        return objective
