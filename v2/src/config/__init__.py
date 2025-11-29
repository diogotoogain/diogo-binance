"""V2 Config Package."""
from .loader import ConfigLoader, KillSwitchDisabledError, ConfigValidationError

__all__ = ['ConfigLoader', 'KillSwitchDisabledError', 'ConfigValidationError']
