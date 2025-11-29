"""Optimization module with Optuna integration."""

from .parameter_optimizer import (
    ParameterOptimizer,
    OptimizationConfig,
    OptimizationResult,
    create_strategy_evaluator
)

# Optional imports that depend on optuna
try:
    from .optuna_optimizer import OptunaOptimizer
    from .param_space import ParamSpace
    from .objectives import ObjectiveFunctions
    from .study_manager import StudyManager
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    OptunaOptimizer = None
    ParamSpace = None
    ObjectiveFunctions = None
    StudyManager = None

__all__ = [
    "ParameterOptimizer",
    "OptimizationConfig",
    "OptimizationResult",
    "create_strategy_evaluator",
    "OptunaOptimizer",
    "ParamSpace",
    "ObjectiveFunctions",
    "StudyManager"
]
