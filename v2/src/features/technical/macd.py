"""
MACD (Moving Average Convergence Divergence) Feature.

Indicador de tendência e momentum baseado em EMAs.
"""
from typing import Any, Dict, Tuple
import pandas as pd
import numpy as np

from v2.src.features.base import Feature


class MACD(Feature):
    """
    Moving Average Convergence Divergence (MACD).
    
    Calcula:
    - MACD Line: EMA(fast) - EMA(slow)
    - Signal Line: EMA(MACD Line, signal)
    - Histogram: MACD Line - Signal Line
    
    Parâmetros do config:
        fast: Período da EMA rápida (default: 12)
        slow: Período da EMA lenta (default: 26)
        signal: Período da linha de sinal (default: 9)
        
    OPTIMIZE: fast em [8, 12, 16]
              slow em [20, 26, 30]
              signal em [6, 9, 12]
    """
    
    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """
        Inicializa o MACD.
        
        Args:
            config: Deve conter 'fast', 'slow', 'signal'
            enabled: Se a feature está habilitada
        """
        super().__init__(config, enabled)
        self.fast = config.get('fast', 12)
        self.slow = config.get('slow', 26)
        self.signal_period = config.get('signal', 9)
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula o MACD para um DataFrame.
        
        Args:
            data: DataFrame com coluna 'close'
            
        Returns:
            pd.DataFrame com colunas 'macd', 'macd_signal', 'macd_hist'
        """
        if not self.enabled:
            return pd.DataFrame(index=data.index)
            
        if 'close' not in data.columns:
            return pd.DataFrame(index=data.index)
        
        # EMAs
        ema_fast = data['close'].ewm(span=self.fast, adjust=False).mean()
        ema_slow = data['close'].ewm(span=self.slow, adjust=False).mean()
        
        # MACD Line
        macd_line = ema_fast - ema_slow
        
        # Signal Line
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        result = pd.DataFrame({
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_hist': histogram
        }, index=data.index)
        
        return result
        
    def calculate_incremental(self, new_data: Any, state: Dict) -> Dict[str, float]:
        """
        Calcula o MACD incrementalmente.
        
        Args:
            new_data: Dict com 'close'
            state: Dict com 'ema_fast', 'ema_slow', 'ema_signal'
            
        Returns:
            Dict com 'macd', 'macd_signal', 'macd_hist'
        """
        if not self.enabled:
            return {'macd': 0.0, 'macd_signal': 0.0, 'macd_hist': 0.0}
        
        close = new_data.get('close', 0.0)
        
        # Inicializa estado se necessário
        if 'ema_fast' not in state:
            state['ema_fast'] = close
            state['ema_slow'] = close
            state['ema_signal'] = 0.0
        
        # Atualiza EMAs
        alpha_fast = 2 / (self.fast + 1)
        alpha_slow = 2 / (self.slow + 1)
        alpha_signal = 2 / (self.signal_period + 1)
        
        state['ema_fast'] = alpha_fast * close + (1 - alpha_fast) * state['ema_fast']
        state['ema_slow'] = alpha_slow * close + (1 - alpha_slow) * state['ema_slow']
        
        # MACD Line
        macd_line = state['ema_fast'] - state['ema_slow']
        
        # Signal Line
        state['ema_signal'] = (
            alpha_signal * macd_line + (1 - alpha_signal) * state['ema_signal']
        )
        
        # Histogram
        histogram = macd_line - state['ema_signal']
        
        return {
            'macd': macd_line,
            'macd_signal': state['ema_signal'],
            'macd_hist': histogram
        }
    
    def get_signal(self, macd_data: Dict[str, float]) -> int:
        """
        Retorna sinal baseado no MACD.
        
        Args:
            macd_data: Dict com 'macd', 'macd_signal', 'macd_hist'
            
        Returns:
            1 = MACD cruzou acima do signal (bullish)
            -1 = MACD cruzou abaixo do signal (bearish)
            0 = Neutro
        """
        histogram = macd_data.get('macd_hist', 0.0)
        
        # Histograma positivo = bullish, negativo = bearish
        if histogram > 0:
            return 1
        elif histogram < 0:
            return -1
        return 0
