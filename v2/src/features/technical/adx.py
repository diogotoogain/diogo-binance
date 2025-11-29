"""
ADX (Average Directional Index) Feature.

Mede a força de uma tendência, independente da direção.
"""
from typing import Any, Dict
import pandas as pd
import numpy as np

from v2.src.features.base import Feature


class ADX(Feature):
    """
    Average Directional Index (ADX).
    
    Mede a força da tendência (não a direção):
    - ADX > trending_threshold (25) = Mercado em tendência
    - ADX < ranging_threshold (20) = Mercado lateralizado
    
    Também calcula +DI e -DI para direção.
    
    Parâmetros do config:
        period: Período de cálculo (default: 14)
        trending_threshold: Limite para considerar tendência (default: 25)
        ranging_threshold: Limite para considerar lateralização (default: 20)
        
    OPTIMIZE: period em [7, 14, 21, 28]
              trending_threshold em [20, 25, 30, 35]
              ranging_threshold em [15, 18, 20]
    """
    
    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """
        Inicializa o ADX.
        
        Args:
            config: Deve conter 'period', 'trending_threshold', 'ranging_threshold'
            enabled: Se a feature está habilitada
        """
        super().__init__(config, enabled)
        self.period = config.get('period', 14)
        self.trending_threshold = config.get('trending_threshold', 25)
        self.ranging_threshold = config.get('ranging_threshold', 20)
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula o ADX para um DataFrame.
        
        Args:
            data: DataFrame com colunas 'high', 'low', 'close'
            
        Returns:
            pd.DataFrame com colunas 'adx', 'plus_di', 'minus_di'
        """
        if not self.enabled:
            return pd.DataFrame(index=data.index)
            
        required_cols = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_cols):
            return pd.DataFrame(index=data.index)
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # +DM e -DM
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        
        # Smoothed TR, +DM, -DM
        atr = tr.ewm(span=self.period, adjust=False).mean()
        smooth_plus_dm = plus_dm.ewm(span=self.period, adjust=False).mean()
        smooth_minus_dm = minus_dm.ewm(span=self.period, adjust=False).mean()
        
        # +DI e -DI
        plus_di = 100 * smooth_plus_dm / atr.replace(0, np.nan)
        minus_di = 100 * smooth_minus_dm / atr.replace(0, np.nan)
        
        # DX
        di_sum = plus_di + minus_di
        dx = 100 * abs(plus_di - minus_di) / di_sum.replace(0, np.nan)
        
        # ADX (smoothed DX)
        adx = dx.ewm(span=self.period, adjust=False).mean()
        
        result = pd.DataFrame({
            'adx': adx.fillna(0.0),
            'plus_di': plus_di.fillna(0.0),
            'minus_di': minus_di.fillna(0.0)
        }, index=data.index)
        
        return result
        
    def calculate_incremental(self, new_data: Any, state: Dict) -> Dict[str, float]:
        """
        Calcula o ADX incrementalmente.
        
        Args:
            new_data: Dict com 'high', 'low', 'close'
            state: Dict com estados das EMAs
            
        Returns:
            Dict com 'adx', 'plus_di', 'minus_di'
        """
        if not self.enabled:
            return {'adx': 0.0, 'plus_di': 0.0, 'minus_di': 0.0}
        
        high = new_data.get('high', 0.0)
        low = new_data.get('low', 0.0)
        close = new_data.get('close', 0.0)
        
        # Inicializa estado se necessário
        if 'prev_high' not in state:
            state['prev_high'] = high
            state['prev_low'] = low
            state['prev_close'] = close
            state['smooth_tr'] = 0.0
            state['smooth_plus_dm'] = 0.0
            state['smooth_minus_dm'] = 0.0
            state['smooth_dx'] = 0.0
            return {'adx': 0.0, 'plus_di': 0.0, 'minus_di': 0.0}
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - state['prev_close'])
        tr3 = abs(low - state['prev_close'])
        tr = max(tr1, tr2, tr3)
        
        # +DM e -DM
        up_move = high - state['prev_high']
        down_move = state['prev_low'] - low
        
        plus_dm = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm = down_move if (down_move > up_move and down_move > 0) else 0.0
        
        # Atualiza estado anterior
        state['prev_high'] = high
        state['prev_low'] = low
        state['prev_close'] = close
        
        # Smoothing (EMA)
        alpha = 2 / (self.period + 1)
        state['smooth_tr'] = alpha * tr + (1 - alpha) * state['smooth_tr']
        state['smooth_plus_dm'] = alpha * plus_dm + (1 - alpha) * state['smooth_plus_dm']
        state['smooth_minus_dm'] = (
            alpha * minus_dm + (1 - alpha) * state['smooth_minus_dm']
        )
        
        # +DI e -DI
        if state['smooth_tr'] == 0:
            plus_di = 0.0
            minus_di = 0.0
        else:
            plus_di = 100 * state['smooth_plus_dm'] / state['smooth_tr']
            minus_di = 100 * state['smooth_minus_dm'] / state['smooth_tr']
        
        # DX
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx = 0.0
        else:
            dx = 100 * abs(plus_di - minus_di) / di_sum
        
        # ADX (smoothed DX)
        state['smooth_dx'] = alpha * dx + (1 - alpha) * state['smooth_dx']
        
        return {
            'adx': state['smooth_dx'],
            'plus_di': plus_di,
            'minus_di': minus_di
        }
    
    def is_trending(self, adx_value: float) -> bool:
        """Verifica se o mercado está em tendência."""
        return adx_value >= self.trending_threshold
    
    def is_ranging(self, adx_value: float) -> bool:
        """Verifica se o mercado está lateralizado."""
        return adx_value <= self.ranging_threshold
    
    def get_regime(self, adx_value: float) -> str:
        """
        Retorna o regime do mercado.
        
        Returns:
            'trending', 'ranging' ou 'neutral'
        """
        if self.is_trending(adx_value):
            return 'trending'
        elif self.is_ranging(adx_value):
            return 'ranging'
        return 'neutral'
