"""
Bollinger Bands Feature.

Bandas de volatilidade baseadas em desvio padrão.
"""
from typing import Any, Dict
import pandas as pd
import numpy as np

from v2.src.features.base import Feature


class BollingerBands(Feature):
    """
    Bollinger Bands.
    
    Bandas de volatilidade:
    - Upper Band: SMA + (std_dev * StdDev)
    - Middle Band: SMA
    - Lower Band: SMA - (std_dev * StdDev)
    
    %B indica posição relativa do preço nas bandas:
    - %B > 1: Acima da banda superior
    - %B < 0: Abaixo da banda inferior
    - %B = 0.5: No meio
    
    Parâmetros do config:
        period: Período da média móvel (default: 20)
        std_dev: Multiplicador do desvio padrão (default: 2.0)
        
    OPTIMIZE: period em [10, 20, 30]
              std_dev em [1.5, 2.0, 2.5, 3.0]
    """
    
    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """
        Inicializa o Bollinger Bands.
        
        Args:
            config: Deve conter 'period' e 'std_dev'
            enabled: Se a feature está habilitada
        """
        super().__init__(config, enabled)
        self.period = config.get('period', 20)
        self.std_dev = config.get('std_dev', 2.0)
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula Bollinger Bands para um DataFrame.
        
        Args:
            data: DataFrame com coluna 'close'
            
        Returns:
            pd.DataFrame com colunas 'bb_upper', 'bb_middle', 'bb_lower', 
            'bb_width', 'bb_percent_b'
        """
        if not self.enabled:
            return pd.DataFrame(index=data.index)
            
        if 'close' not in data.columns:
            return pd.DataFrame(index=data.index)
        
        close = data['close']
        
        # Média móvel simples
        middle = close.rolling(window=self.period).mean()
        
        # Desvio padrão
        std = close.rolling(window=self.period).std()
        
        # Bandas
        upper = middle + (self.std_dev * std)
        lower = middle - (self.std_dev * std)
        
        # Width (largura normalizada)
        width = (upper - lower) / middle.replace(0, np.nan)
        
        # %B (posição relativa)
        band_range = upper - lower
        percent_b = (close - lower) / band_range.replace(0, np.nan)
        
        result = pd.DataFrame({
            'bb_upper': upper,
            'bb_middle': middle,
            'bb_lower': lower,
            'bb_width': width.fillna(0.0),
            'bb_percent_b': percent_b.fillna(0.5)
        }, index=data.index)
        
        return result
        
    def calculate_incremental(self, new_data: Any, state: Dict) -> Dict[str, float]:
        """
        Calcula Bollinger Bands incrementalmente.
        
        Args:
            new_data: Dict com 'close'
            state: Dict com 'prices' (lista de preços recentes)
            
        Returns:
            Dict com 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_percent_b'
        """
        if not self.enabled:
            return {
                'bb_upper': 0.0, 'bb_middle': 0.0, 'bb_lower': 0.0,
                'bb_width': 0.0, 'bb_percent_b': 0.5
            }
        
        close = new_data.get('close', 0.0)
        
        # Inicializa estado se necessário
        if 'prices' not in state:
            state['prices'] = []
        
        state['prices'].append(close)
        
        # Mantém apenas os últimos 'period' preços
        if len(state['prices']) > self.period:
            state['prices'] = state['prices'][-self.period:]
        
        # Precisa de dados suficientes
        if len(state['prices']) < self.period:
            return {
                'bb_upper': close, 'bb_middle': close, 'bb_lower': close,
                'bb_width': 0.0, 'bb_percent_b': 0.5
            }
        
        prices = np.array(state['prices'])
        
        # Cálculos
        middle = np.mean(prices)
        std = np.std(prices)
        
        upper = middle + (self.std_dev * std)
        lower = middle - (self.std_dev * std)
        
        # Width
        width = (upper - lower) / middle if middle != 0 else 0.0
        
        # %B
        band_range = upper - lower
        percent_b = (close - lower) / band_range if band_range != 0 else 0.5
        
        return {
            'bb_upper': upper,
            'bb_middle': middle,
            'bb_lower': lower,
            'bb_width': width,
            'bb_percent_b': percent_b
        }
    
    def is_squeeze(self, bb_data: Dict[str, float], threshold_percentile: float = 20) -> bool:
        """
        Verifica se as bandas estão em squeeze (volatilidade baixa).
        
        Args:
            bb_data: Dict com 'bb_width'
            threshold_percentile: Percentil abaixo do qual considera squeeze
            
        Returns:
            True se em squeeze
        """
        # Simplificado: considera squeeze se width < 0.02 (2%)
        return bb_data.get('bb_width', 0.0) < 0.02
    
    def get_signal(self, bb_data: Dict[str, float], close: float) -> int:
        """
        Retorna sinal baseado nas Bollinger Bands.
        
        Args:
            bb_data: Dict com dados das bandas
            close: Preço de fechamento atual
            
        Returns:
            1 = Preço abaixo da banda inferior (possível compra)
            -1 = Preço acima da banda superior (possível venda)
            0 = Dentro das bandas
        """
        percent_b = bb_data.get('bb_percent_b', 0.5)
        
        if percent_b < 0:
            return 1  # Oversold
        elif percent_b > 1:
            return -1  # Overbought
        return 0
