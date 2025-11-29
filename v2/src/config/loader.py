"""
Config Loader - Carrega e valida configuração YAML

Requisitos:
- Carregar YAML com PyYAML
- Validar que kill_switch está SEMPRE ativo
- Override por variáveis de ambiente (.env)
- Método get() com notação de ponto: config.get('risk.max_leverage')
- Singleton pattern
- Validação de ranges (leverage 1-20, risk 0.1-5%)
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
