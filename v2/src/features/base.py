"""
Classe base para todas as features.
Todas as features devem herdar desta classe e implementar os métodos abstratos.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd


class Feature(ABC):
    """
    Classe base para todas as features.
    
    Todas as features devem:
    1. Ler parâmetros do config (NADA hardcoded!)
    2. Ter flag enabled/disabled
    3. Suportar cálculo batch e incremental
    """
    
    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """
        Inicializa a feature.
        
        Args:
            config: Dicionário de configuração com parâmetros
            enabled: Se a feature está habilitada
        """
        self.config = config
        self.enabled = enabled
        self.name = self.__class__.__name__
        
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcula a feature para um DataFrame inteiro (batch).
        
        Args:
            data: DataFrame com dados de mercado (OHLCV, order book, etc)
            
        Returns:
            pd.Series com os valores calculados da feature
        """
        pass
        
    @abstractmethod
    def calculate_incremental(self, new_data: Any, state: Dict) -> float:
        """
        Calcula a feature incrementalmente (para real-time).
        
        Args:
            new_data: Novo dado recebido
            state: Estado anterior da feature (para manter contexto)
            
        Returns:
            Valor atual da feature
        """
        pass
        
    def get_params(self) -> Dict[str, Any]:
        """
        Retorna parâmetros para logging/otimização.
        
        Returns:
            Dicionário com parâmetros da feature
        """
        return self.config.copy()
    
    def get_name(self) -> str:
        """
        Retorna o nome da feature.
        
        Returns:
            Nome da feature
        """
        return self.name
    
    def is_enabled(self) -> bool:
        """
        Verifica se a feature está habilitada.
        
        Returns:
            True se habilitada, False caso contrário
        """
        return self.enabled
    
    def set_enabled(self, enabled: bool) -> None:
        """
        Habilita ou desabilita a feature.
        
        Args:
            enabled: Novo estado da feature
        """
        self.enabled = enabled
