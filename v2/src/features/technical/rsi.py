"""
RSI (Relative Strength Index) Feature.

Indicador de momentum que mede velocidade e magnitude de movimentos de preço.
RSI varia de 0 a 100.
"""
from typing import Any, Dict
import pandas as pd
import numpy as np

from v2.src.features.base import Feature


class RSI(Feature):
    """
    Relative Strength Index (RSI).
    
    Mede a velocidade e magnitude dos movimentos de preço.
    - RSI > overbought (70) = Sobrecomprado
    - RSI < oversold (30) = Sobrevendido
    
    Parâmetros do config:
        period: Período de cálculo (default: 14)
        overbought: Limite de sobrecompra (default: 70)
        oversold: Limite de sobrevenda (default: 30)
        
    OPTIMIZE: period em [7, 14, 21, 28]
              overbought em [65, 70, 75, 80]
              oversold em [20, 25, 30, 35]
    """
    
    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """
        Inicializa o RSI.
        
        Args:
            config: Deve conter 'period', 'overbought', 'oversold'
            enabled: Se a feature está habilitada
        """
        super().__init__(config, enabled)
        self.period = config.get('period', 14)
        self.overbought = config.get('overbought', 70)
        self.oversold = config.get('oversold', 30)
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcula o RSI para um DataFrame.
        
        Args:
            data: DataFrame com coluna 'close'
            
        Returns:
            pd.Series com valores de RSI em [0, 100]
        """
        if not self.enabled:
            return pd.Series(index=data.index, dtype=float)
            
        if 'close' not in data.columns:
            return pd.Series(index=data.index, dtype=float)
        
        # Calcula mudanças de preço
        delta = data['close'].diff()
        
        # Separa ganhos e perdas
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        # Média exponencial de ganhos e perdas
        avg_gain = gain.ewm(span=self.period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.period, adjust=False).mean()
        
        # RS = avg_gain / avg_loss
        rs = avg_gain / avg_loss.replace(0, np.nan)
        
        # RSI = 100 - (100 / (1 + RS))
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50.0)  # Neutro quando indeterminado
        
    def calculate_incremental(self, new_data: Any, state: Dict) -> float:
        """
        Calcula o RSI incrementalmente.
        
        Args:
            new_data: Dict com 'close'
            state: Dict com 'prev_close', 'avg_gain', 'avg_loss'
            
        Returns:
            Valor atual do RSI em [0, 100]
        """
        if not self.enabled:
            return 50.0
        
        close = new_data.get('close', 0.0)
        
        # Inicializa estado se necessário
        if 'prev_close' not in state:
            state['prev_close'] = close
            state['avg_gain'] = 0.0
            state['avg_loss'] = 0.0
            return 50.0
        
        # Calcula mudança
        delta = close - state['prev_close']
        state['prev_close'] = close
        
        # Ganho e perda atuais
        current_gain = max(delta, 0.0)
        current_loss = max(-delta, 0.0)
        
        # Atualiza médias exponenciais
        alpha = 1.0 / self.period
        state['avg_gain'] = alpha * current_gain + (1 - alpha) * state['avg_gain']
        state['avg_loss'] = alpha * current_loss + (1 - alpha) * state['avg_loss']
        
        # Calcula RSI
        if state['avg_loss'] == 0:
            if state['avg_gain'] == 0:
                return 50.0
            return 100.0
        
        rs = state['avg_gain'] / state['avg_loss']
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def is_overbought(self, rsi_value: float) -> bool:
        """Verifica se o RSI está em zona de sobrecompra."""
        return rsi_value >= self.overbought
    
    def is_oversold(self, rsi_value: float) -> bool:
        """Verifica se o RSI está em zona de sobrevenda."""
        return rsi_value <= self.oversold
    
    def get_signal(self, rsi_value: float) -> int:
        """
        Retorna sinal baseado no RSI.
        
        Returns:
            1 = Sobrevendido (possível compra)
            -1 = Sobrecomprado (possível venda)
            0 = Neutro
        """
        if self.is_oversold(rsi_value):
            return 1
        elif self.is_overbought(rsi_value):
            return -1
        return 0
