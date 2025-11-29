"""
Módulo de configuração do V2 Trading Bot.

Inclui loader de configuração YAML e schema de parâmetros otimizáveis.
"""

from .loader import ConfigLoader, get_config
from .schema import OptimizableParam, ParamType, get_all_optimizable_params

__all__ = [
    'ConfigLoader',
    'get_config',
    'OptimizableParam',
    'ParamType',
    'get_all_optimizable_params'
]
"""Configuration module."""

from .loader import ConfigLoader
from .schema import ParamType, get_all_optimizable_params

__all__ = ["ConfigLoader", "ParamType", "get_all_optimizable_params"]
"""V2 Config Package."""
from .loader import ConfigLoader, KillSwitchDisabledError, ConfigValidationError

__all__ = ['ConfigLoader', 'KillSwitchDisabledError', 'ConfigValidationError']
