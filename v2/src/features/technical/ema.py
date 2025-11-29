"""
EMA (Exponential Moving Average) Feature.

Médias móveis exponenciais com múltiplos períodos.
"""
from typing import Any, Dict, List
import pandas as pd
import numpy as np

from v2.src.features.base import Feature


class EMA(Feature):
    """
    Exponential Moving Average (EMA).
    
    Calcula EMAs com múltiplos períodos para diferentes horizontes de tempo.
    
    Parâmetros do config:
        periods: Lista de períodos (default: [9, 21, 50, 100, 200])
        
    OPTIMIZE: periods - diferentes combinações de períodos
    """
    
    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """
        Inicializa o EMA.
        
        Args:
            config: Deve conter 'periods' (lista de períodos)
            enabled: Se a feature está habilitada
        """
        super().__init__(config, enabled)
        self.periods = config.get('periods', [9, 21, 50, 100, 200])
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula EMAs para um DataFrame.
        
        Args:
            data: DataFrame com coluna 'close'
            
        Returns:
            pd.DataFrame com colunas 'ema_{period}' para cada período
        """
        if not self.enabled:
            return pd.DataFrame(index=data.index)
            
        if 'close' not in data.columns:
            return pd.DataFrame(index=data.index)
        
        result = pd.DataFrame(index=data.index)
        
        for period in self.periods:
            result[f'ema_{period}'] = data['close'].ewm(span=period, adjust=False).mean()
        
        return result
        
    def calculate_incremental(self, new_data: Any, state: Dict) -> Dict[str, float]:
        """
        Calcula EMAs incrementalmente.
        
        Args:
            new_data: Dict com 'close'
            state: Dict com 'ema_{period}' para cada período
            
        Returns:
            Dict com valores atuais de EMA para cada período
        """
        if not self.enabled:
            return {}
        
        close = new_data.get('close', 0.0)
        result = {}
        
        for period in self.periods:
            key = f'ema_{period}'
            alpha = 2 / (period + 1)
            
            if key not in state:
                # Inicializa com o primeiro preço
                state[key] = close
            else:
                # EMA = alpha * price + (1 - alpha) * prev_ema
                state[key] = alpha * close + (1 - alpha) * state[key]
            
            result[key] = state[key]
        
        return result
    
    def get_crossover(
        self, data: pd.DataFrame, fast_period: int, slow_period: int
    ) -> pd.Series:
        """
        Detecta crossovers entre EMAs.
        
        Args:
            data: DataFrame com coluna 'close'
            fast_period: Período da EMA rápida
            slow_period: Período da EMA lenta
            
        Returns:
            pd.Series com 1 (golden cross), -1 (death cross), 0 (nenhum)
        """
        emas = self.calculate(data)
        
        fast_col = f'ema_{fast_period}'
        slow_col = f'ema_{slow_period}'
        
        if fast_col not in emas.columns or slow_col not in emas.columns:
            return pd.Series(0, index=data.index)
        
        fast = emas[fast_col]
        slow = emas[slow_col]
        
        # Crossover: fast cruza slow de baixo para cima (golden) ou vice-versa (death)
        crossover = pd.Series(0, index=data.index)
        
        # Golden cross: fast estava abaixo e agora está acima
        crossover[(fast > slow) & (fast.shift(1) <= slow.shift(1))] = 1
        
        # Death cross: fast estava acima e agora está abaixo
        crossover[(fast < slow) & (fast.shift(1) >= slow.shift(1))] = -1
        
        return crossover
