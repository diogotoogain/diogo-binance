"""
Parameter space definitions for Optuna optimization.

Reads optimizable parameters from the schema and generates suggestions
for Optuna trials.
"""

import copy
from typing import Any, Dict, List, Optional

import optuna

from ..config.schema import (
    FEATURE_TOGGLES,
    OPTIMIZABLE_PARAMS,
    STRATEGY_TOGGLES,
    OptimizableParam,
    ParamType,
    get_params_dict,
    get_feature_toggles,
    get_strategy_toggles,
)


class ParamSpace:
    """
    Defines parameter space for Optuna optimization.

    Reads parameters marked with OPTIMIZE from config/schema.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        include_feature_toggles: bool = True,
        include_strategy_toggles: bool = True,
    ):
        """
        Initialize parameter space.

        Args:
            config: Configuration dictionary
            include_feature_toggles: Whether to include feature on/off toggles
            include_strategy_toggles: Whether to include strategy on/off toggles
        """
        self.config = config
        self.include_feature_toggles = include_feature_toggles
        self.include_strategy_toggles = include_strategy_toggles
        self.params = self._load_optimizable_params()

    def _load_optimizable_params(self) -> Dict[str, List[OptimizableParam]]:
        """Load all optimizable parameters from schema."""
        return get_params_dict()

    def suggest(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Generate parameter suggestions for an Optuna trial.

        Args:
            trial: Optuna trial object

        Returns:
            Dictionary of suggested parameter values
        """
        suggestions: Dict[str, Any] = {}

        # Suggest values for all optimizable parameters
        for section, params in self.params.items():
            for param in params:
                key = f"{section}.{param.name}"
                suggestions[key] = self._suggest_param(trial, key, param)

        # Suggest feature toggles if enabled
        if self.include_feature_toggles:
            for key, param in get_feature_toggles().items():
                suggestions[key] = self._suggest_param(trial, key, param)

        # Suggest strategy toggles if enabled
        if self.include_strategy_toggles:
            for key, param in get_strategy_toggles().items():
                suggestions[key] = self._suggest_param(trial, key, param)

        return suggestions

    def _suggest_param(
        self, trial: optuna.Trial, key: str, param: OptimizableParam
    ) -> Any:
        """
        Suggest a single parameter value.

        Args:
            trial: Optuna trial object
            key: Parameter key
            param: Parameter definition

        Returns:
            Suggested value
        """
        if param.param_type == ParamType.FLOAT:
            if param.log_scale:
                return trial.suggest_float(key, param.low, param.high, log=True)
            elif param.step:
                return trial.suggest_float(key, param.low, param.high, step=param.step)
            else:
                return trial.suggest_float(key, param.low, param.high)

        elif param.param_type == ParamType.INT:
            if param.step:
                return trial.suggest_int(key, param.low, param.high, step=param.step)
            else:
                return trial.suggest_int(key, param.low, param.high)

        elif param.param_type == ParamType.CATEGORICAL:
            return trial.suggest_categorical(key, param.choices)

        elif param.param_type == ParamType.BOOL:
            return trial.suggest_categorical(key, [True, False])

        else:
            raise ValueError(f"Unknown param type: {param.param_type}")

    def get_default_params(self) -> Dict[str, Any]:
        """
        Get default values for all parameters.

        Returns:
            Dictionary of default parameter values
        """
        defaults: Dict[str, Any] = {}

        for section, params in self.params.items():
            for param in params:
                key = f"{section}.{param.name}"
                defaults[key] = param.default

        if self.include_feature_toggles:
            for key, param in get_feature_toggles().items():
                defaults[key] = param.default

        if self.include_strategy_toggles:
            for key, param in get_strategy_toggles().items():
                defaults[key] = param.default

        return defaults

    def get_param_count(self) -> int:
        """Get total number of optimizable parameters."""
        count = 0
        for params in self.params.values():
            count += len(params)

        if self.include_feature_toggles:
            count += len(get_feature_toggles())

        if self.include_strategy_toggles:
            count += len(get_strategy_toggles())

        return count

    def get_search_space(self) -> Dict[str, Dict[str, Any]]:
        """
        Get search space definition for visualization/logging.

        Returns:
            Dictionary describing the search space
        """
        search_space: Dict[str, Dict[str, Any]] = {}

        for section, params in self.params.items():
            for param in params:
                key = f"{section}.{param.name}"
                search_space[key] = {
                    "type": param.param_type.value,
                    "low": param.low,
                    "high": param.high,
                    "choices": param.choices,
                    "default": param.default,
                    "log_scale": param.log_scale,
                    "description": param.description,
                }

        return search_space

    def apply_params_to_config(
        self, config: Dict[str, Any], params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply suggested parameters to a configuration dictionary.

        Args:
            config: Original configuration
            params: Suggested parameters (dot notation keys)

        Returns:
            Updated configuration dictionary
        """
        updated_config = copy.deepcopy(config)

        for key, value in params.items():
            keys = key.split(".")
            target = updated_config

            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]

            target[keys[-1]] = value

        return updated_config

    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """
        Validate suggested parameters.

        Args:
            params: Suggested parameters

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: List[str] = []

        for section, param_list in self.params.items():
            for param in param_list:
                key = f"{section}.{param.name}"
                if key not in params:
                    continue

                value = params[key]

                if param.param_type in (ParamType.FLOAT, ParamType.INT):
                    if value < param.low or value > param.high:
                        errors.append(
                            f"{key}: value {value} out of range [{param.low}, {param.high}]"
                        )

                elif param.param_type == ParamType.CATEGORICAL:
                    if value not in param.choices:
                        errors.append(
                            f"{key}: value {value} not in choices {param.choices}"
                        )

        return errors
