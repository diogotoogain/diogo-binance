"""
Configuration loader for V2 trading system.

Loads configuration from YAML files and provides access to all parameters.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class KillSwitchDisabledError(Exception):
    """Raised when kill switch is disabled in config."""

    pass


class ConfigLoader:
    """
    Load and manage configuration from YAML files.

    Validates critical settings and provides nested access to config values.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        skip_env: bool = False,
    ):
        """
        Initialize config loader.

        Args:
            config_path: Path to config file. If None, uses default.yaml
            skip_env: If True, skip environment variable interpolation
        """
        self.skip_env = skip_env

        if config_path is None:
            # Default config path
            base_path = Path(__file__).parent.parent.parent
            config_path = base_path / "config" / "default.yaml"

        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()
        self._validate_config()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            self._config = yaml.safe_load(f)

        if not self.skip_env:
            self._interpolate_env_vars(self._config)

    def _interpolate_env_vars(self, config: Dict[str, Any]) -> None:
        """Replace ${VAR} placeholders with environment variables."""
        for key, value in config.items():
            if isinstance(value, dict):
                self._interpolate_env_vars(value)
            elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                config[key] = os.environ.get(env_var, value)

    def _validate_config(self) -> None:
        """Validate critical configuration settings."""
        # Kill switch must always be enabled
        kill_switch_enabled = self.get("risk.kill_switch.enabled", True)
        if not kill_switch_enabled:
            raise KillSwitchDisabledError(
                "Kill switch must always be enabled for safety. "
                "Set risk.kill_switch.enabled: true in config."
            )

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Dot-separated path to config value (e.g., 'risk.max_drawdown_pct')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.

        Args:
            section: Section name (e.g., 'risk', 'backtest')

        Returns:
            Configuration section as dict
        """
        return self.get(section, {})

    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration dictionary."""
        return self._config

    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access to config."""
        return self.get(key)

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Args:
            updates: Dictionary of updates (can use dot notation keys)
        """
        for key, value in updates.items():
            keys = key.split(".")
            config = self._config

            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]

            config[keys[-1]] = value

        # Re-validate after updates
        self._validate_config()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.copy()

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """
        Save current configuration to YAML file.

        Args:
            path: Output path. If None, saves to original path.
        """
        output_path = Path(path) if path else self.config_path

        with open(output_path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)
