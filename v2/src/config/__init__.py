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
