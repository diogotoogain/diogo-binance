"""
Config Loader - Carrega e valida configuração YAML

Requisitos:
- Carregar YAML com PyYAML
- Validar que kill_switch está SEMPRE ativo
- Override por variáveis de ambiente (.env)
- Método get() com notação de ponto: config.get('risk.max_leverage')
- Singleton pattern
- Validação de ranges (leverage 1-20, risk 0.1-5%)
Configuration loader for V2 trading system.

Loads configuration from YAML files and provides access to all parameters.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from dotenv import load_dotenv


class ConfigValidationError(Exception):
    """Erro de validação de configuração."""
    pass


class KillSwitchDisabledError(ConfigValidationError):
    """Erro quando kill_switch está desativado."""

Config loader for V2 trading bot.

Loads configuration from YAML files and validates critical settings.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class KillSwitchDisabledError(Exception):
    """Raised when kill switch is disabled in config."""

    """Raised when kill switch is disabled - this is NEVER allowed!"""
    pass


class ConfigValidationError(Exception):
    """Raised when config validation fails."""
    pass


class ConfigLoader:
    """
    Carregador de configuração com padrão Singleton.
    
    Carrega configuração YAML e permite override por variáveis de ambiente.
    Valida que configurações críticas (como kill_switch) estão corretas.
    """
    
    _instance: Optional['ConfigLoader'] = None
    _config: Dict[str, Any] = {}
    _config_path: Optional[Path] = None
    
    def __new__(cls, config_path: Optional[str] = None, skip_env: bool = False) -> 'ConfigLoader':
        """Implementa padrão Singleton."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None, skip_env: bool = False):
        """
        Inicializa o ConfigLoader.
        
        Args:
            config_path: Caminho para o arquivo YAML de configuração.
                        Se não fornecido, usa v2/config/default.yaml
            skip_env: Se True, não carrega variáveis de ambiente (útil para testes)
        """
        if self._initialized and config_path is None:
            return
            
        # Carrega variáveis de ambiente (exceto em testes)
        if not skip_env:
            load_dotenv()
        
        # Define caminho padrão
        if config_path is None:
            base_dir = Path(__file__).parent.parent.parent
            config_path = base_dir / "config" / "default.yaml"
        else:
            config_path = Path(config_path)
        
        self._config_path = config_path
        self._skip_env = skip_env
        self._load_config()
        if not skip_env:
            self._apply_env_overrides()
        self._validate_config()
        self._initialized = True
    
    def _load_config(self) -> None:
        """Carrega configuração do arquivo YAML."""
        if not self._config_path or not self._config_path.exists():
            raise FileNotFoundError(
                f"Arquivo de configuração não encontrado: {self._config_path}"
            )
        
        with open(self._config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
    
    def _apply_env_overrides(self) -> None:
        """Aplica overrides de variáveis de ambiente."""
        env_mappings = {
            'BINANCE_API_KEY': ('environment', 'api_key'),
            'BINANCE_SECRET_KEY': ('environment', 'secret_key'),
            'USE_DEMO': ('environment', 'use_demo_header'),
            'SYMBOL': ('market', 'symbol'),
            'RISK_PER_TRADE': ('risk', 'risk_per_trade_pct'),
            'MAX_LEVERAGE': ('risk', 'max_leverage'),
            'MAX_DAILY_LOSS': ('risk', 'max_daily_loss_pct'),
            'DEFAULT_SL_PERCENT': ('position', 'sl', 'default_pct'),
            'DEFAULT_TP_PERCENT': ('position', 'tp', 'default_pct'),
        }
        
        for env_var, path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested(path, self._parse_value(value))
    
    def _parse_value(self, value: str) -> Union[bool, int, float, str]:
        """Converte string para tipo apropriado."""
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # Numeric
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            return value
    
    def _set_nested(self, path: tuple, value: Any) -> None:
        """Define valor em caminho aninhado."""
        current = self._config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[path[-1]] = value
    
    def _validate_config(self) -> None:
        """Valida configuração crítica."""
        # VALIDAÇÃO CRÍTICA: Kill switch DEVE estar ativo!
        kill_switch_enabled = self.get('risk.kill_switch.enabled', False)
        if not kill_switch_enabled:
            raise KillSwitchDisabledError(
                "🚨 ERRO CRÍTICO: Kill switch DEVE estar SEMPRE ativo! "
                "Defina risk.kill_switch.enabled: true no config."
            )
        
        # Valida ranges de risco
        risk_per_trade = self.get('risk.risk_per_trade_pct', 0)
        if not 0.1 <= risk_per_trade <= 5.0:
            raise ConfigValidationError(
                f"risk_per_trade_pct ({risk_per_trade}) deve estar entre 0.1 e 5.0 (percentual)"
            )
        
        # Valida leverage
        max_leverage = self.get('risk.max_leverage', 0)
        if not 1 <= max_leverage <= 20:
            raise ConfigValidationError(
                f"max_leverage ({max_leverage}) deve estar entre 1 e 20"
            )
        
        # Valida buffer do WebSocket
        buffer_size = self.get('websocket.buffer_size', 0)
        if buffer_size < 1000:
            raise ConfigValidationError(
                f"websocket.buffer_size ({buffer_size}) deve ser >= 1000 para evitar overflow"
            )
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtém valor da configuração usando notação de ponto.
        
        Args:
            key: Chave com notação de ponto (ex: 'risk.max_leverage')
            default: Valor padrão se chave não existir
            
        Returns:
            Valor da configuração ou default
            
        Example:
            >>> config = ConfigLoader()
            >>> config.get('risk.max_leverage')
            10
            >>> config.get('risk.kill_switch.enabled')
            True
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Obtém seção completa da configuração.
        
        Args:
            section: Nome da seção (ex: 'risk', 'strategies')
            
        Returns:
            Dicionário com a seção ou {} se não existir
        """
        return self._config.get(section, {})
    
    @property
    def config(self) -> Dict[str, Any]:
        """Retorna configuração completa (read-only)."""
        return self._config.copy()
    
    @property
    def is_demo_mode(self) -> bool:
        """Verifica se está em modo demo."""
        return self.get('environment.mode', 'demo') == 'demo'
    
    @property
    def use_demo_header(self) -> bool:
        """Verifica se deve usar header X-MBX-DEMO."""
        return self.get('environment.use_demo_header', True)
    
    def reload(self) -> None:
        """Recarrega configuração do arquivo."""
        self._load_config()
        self._apply_env_overrides()
        self._validate_config()
    
    @classmethod
    def reset(cls) -> None:
        """Reseta singleton (útil para testes)."""
        cls._instance = None


# Função de conveniência
def get_config(config_path: Optional[str] = None, skip_env: bool = False) -> ConfigLoader:
    """
    Obtém instância do ConfigLoader.
    
    Args:
        config_path: Caminho opcional para arquivo de configuração
        skip_env: Se True, não carrega variáveis de ambiente
        
    Returns:
        Instância do ConfigLoader
    """
    return ConfigLoader(config_path, skip_env=skip_env)
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
    Loads and validates configuration for V2 trading bot.
    
    CRITICAL: Kill switch must ALWAYS be enabled!
    """
    
    DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "default.yaml"
    
    def __init__(self, config_path: Optional[str] = None, skip_env: bool = False):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to config file. Defaults to default.yaml
            skip_env: Skip environment variable overrides (for testing)
        """
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self.skip_env = skip_env
        self._config: Dict[str, Any] = {}
        
    def load(self) -> Dict[str, Any]:
        """
        Load and validate configuration.
        
        Returns:
            Validated configuration dictionary
            
        Raises:
            KillSwitchDisabledError: If kill switch is disabled
            ConfigValidationError: If config is invalid
        """
        self._config = self._load_yaml()
        
        if not self.skip_env:
            self._apply_env_overrides()
            
        self._validate()
        
        return self._config
        
    def _load_yaml(self) -> Dict[str, Any]:
        """Load YAML config file."""
        if not self.config_path.exists():
            raise ConfigValidationError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to config."""
        # Example: RISK_MAX_DAILY_LOSS_PCT=5 overrides risk.max_daily_loss_pct
        env_mappings = {
            'RISK_MAX_DAILY_LOSS_PCT': ('risk', 'max_daily_loss_pct', float),
            'RISK_MAX_DRAWDOWN_PCT': ('risk', 'max_drawdown_pct', float),
            'RISK_KILL_SWITCH_TRIGGER_PCT': ('risk', 'kill_switch', 'trigger_loss_pct', float),
        }
        
        for env_var, path_and_type in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                self._set_nested(path_and_type[:-1], path_and_type[-1](value))
                
    def _set_nested(self, keys: tuple, value: Any) -> None:
        """Set a nested config value."""
        d = self._config
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
        
    def _validate(self) -> None:
        """
        Validate configuration.
        
        CRITICAL: Kill switch validation is ALWAYS performed!
        """
        # CRITICAL: Kill switch MUST be enabled!
        self._validate_kill_switch()
        
        # Validate risk parameters
        self._validate_risk_params()
        
        # Validate bet sizing
        self._validate_bet_sizing()
        
        # Validate position params
        self._validate_position_params()
        
    def _validate_kill_switch(self) -> None:
        """
        Validate kill switch is ALWAYS enabled.
        
        Raises:
            KillSwitchDisabledError: If kill switch is disabled
        """
        kill_switch = self._config.get('risk', {}).get('kill_switch', {})
        
        if not kill_switch.get('enabled', True):
            raise KillSwitchDisabledError(
                "🚨 KILL SWITCH NÃO PODE SER DESATIVADO! "
                "Este é um mecanismo de proteção crítico."
            )
            
    def _validate_risk_params(self) -> None:
        """Validate risk management parameters."""
        risk = self._config.get('risk', {})
        
        required_params = [
            'risk_per_trade_pct',
            'max_position_size_pct',
            'max_daily_loss_pct',
            'max_drawdown_pct',
            'max_concurrent_positions',
        ]
        
        for param in required_params:
            if param not in risk:
                raise ConfigValidationError(f"Missing required risk parameter: {param}")
                
    def _validate_bet_sizing(self) -> None:
        """Validate bet sizing parameters."""
        bet_sizing = self._config.get('bet_sizing', {})
        
        method = bet_sizing.get('method')
        if method not in ['fixed', 'kelly', 'vol_target', 'rl']:
            raise ConfigValidationError(f"Invalid bet sizing method: {method}")
            
    def _validate_position_params(self) -> None:
        """Validate position management parameters."""
        position = self._config.get('position', {})
        
        sl_type = position.get('sl_type')
        if sl_type not in ['fixed_pct', 'atr', 'volatility']:
            raise ConfigValidationError(f"Invalid SL type: {sl_type}")
            
        tp_type = position.get('tp_type')
        if tp_type not in ['fixed_pct', 'atr', 'rr_ratio']:
            raise ConfigValidationError(f"Invalid TP type: {tp_type}")
            
    @property
    def config(self) -> Dict[str, Any]:
        """Get loaded configuration."""
        if not self._config:
            self.load()
        return self._config
