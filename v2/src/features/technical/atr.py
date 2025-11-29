"""
ATR (Average True Range) Feature.

Mede a volatilidade do mercado.
"""
from typing import Any, Dict
import pandas as pd
import numpy as np

from v2.src.features.base import Feature


class ATR(Feature):
    """
    Average True Range (ATR).
    
    Mede a volatilidade baseada no True Range.
    Útil para:
    - Definir stop-loss/take-profit
    - Identificar períodos de alta/baixa volatilidade
    - Normalizar indicadores
    
    Parâmetros do config:
        period: Período de cálculo (default: 14)
        
    OPTIMIZE: period em [7, 14, 21, 28]
    """
    
    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """
        Inicializa o ATR.
        
        Args:
            config: Deve conter 'period'
            enabled: Se a feature está habilitada
        """
        super().__init__(config, enabled)
        self.period = config.get('period', 14)
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcula o ATR para um DataFrame.
        
        Args:
            data: DataFrame com colunas 'high', 'low', 'close'
            
        Returns:
            pd.Series com valores de ATR
        """
        if not self.enabled:
            return pd.Series(index=data.index, dtype=float)
            
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            return pd.Series(index=data.index, dtype=float)
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR (média móvel exponencial do True Range)
        atr = tr.ewm(span=self.period, adjust=False).mean()
        
        return atr.fillna(0.0)
        
    def calculate_incremental(self, new_data: Any, state: Dict) -> float:
        """
        Calcula o ATR incrementalmente.
        
        Args:
            new_data: Dict com 'high', 'low', 'close'
            state: Dict com 'prev_close' e 'atr'
            
        Returns:
            Valor atual do ATR
        """
        if not self.enabled:
            return 0.0
        
        high = new_data.get('high', 0.0)
        low = new_data.get('low', 0.0)
        close = new_data.get('close', 0.0)
        
        # Inicializa estado se necessário
        if 'prev_close' not in state:
            state['prev_close'] = close
            state['atr'] = high - low  # TR inicial
            return state['atr']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - state['prev_close'])
        tr3 = abs(low - state['prev_close'])
        tr = max(tr1, tr2, tr3)
        
        # Atualiza estado
        state['prev_close'] = close
        
        # ATR (EMA)
        alpha = 2 / (self.period + 1)
        state['atr'] = alpha * tr + (1 - alpha) * state['atr']
        
        return state['atr']
    
    def get_atr_multiple(self, atr_value: float, close: float, multiplier: float) -> float:
        """
        Calcula múltiplo do ATR (útil para stops).
        
        Args:
            atr_value: Valor atual do ATR
            close: Preço atual
            multiplier: Multiplicador do ATR
            
        Returns:
            Valor absoluto do múltiplo do ATR
        """
        return atr_value * multiplier
    
    def get_stop_loss(
        self, close: float, atr_value: float, multiplier: float, direction: str
    ) -> float:
        """
        Calcula stop-loss baseado no ATR.
        
        Args:
            close: Preço de entrada
            atr_value: Valor atual do ATR
            multiplier: Multiplicador do ATR para distância do stop
            direction: 'long' ou 'short'
            
        Returns:
            Preço do stop-loss
        """
        atr_distance = atr_value * multiplier
        
        if direction.lower() == 'long':
            return close - atr_distance
        else:  # short
            return close + atr_distance
    
    def get_take_profit(
        self, close: float, atr_value: float, multiplier: float, direction: str
    ) -> float:
        """
        Calcula take-profit baseado no ATR.
        
        Args:
            close: Preço de entrada
            atr_value: Valor atual do ATR
            multiplier: Multiplicador do ATR para distância do TP
            direction: 'long' ou 'short'
            
        Returns:
            Preço do take-profit
        """
        atr_distance = atr_value * multiplier
        
        if direction.lower() == 'long':
            return close + atr_distance
        else:  # short
            return close - atr_distance
