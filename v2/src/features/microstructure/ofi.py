"""
Order Flow Imbalance (OFI) Feature.

Mede o desequilíbrio entre fluxo de compra e venda.
OFI = sum(sign(price_change) * volume) / window
"""
from typing import Any, Dict
import pandas as pd
import numpy as np

from v2.src.features.base import Feature


class OFI(Feature):
    """
    Order Flow Imbalance (OFI).
    
    Mede a pressão de compra/venda baseada em mudanças de preço e volume.
    
    Parâmetros do config:
        window: Janela de cálculo (default: 20)
        
    OPTIMIZE: window em [5, 10, 20, 50, 100]
    """
    
    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """
        Inicializa o OFI.
        
        Args:
            config: Deve conter 'window' (tamanho da janela)
            enabled: Se a feature está habilitada
        """
        super().__init__(config, enabled)
        self.window = config.get('window', 20)
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcula o OFI para um DataFrame.
        
        Args:
            data: DataFrame com colunas 'close' e 'volume'
            
        Returns:
            pd.Series com valores de OFI
        """
        if not self.enabled:
            return pd.Series(index=data.index, dtype=float)
            
        if 'close' not in data.columns or 'volume' not in data.columns:
            return pd.Series(index=data.index, dtype=float)
        
        # Calcula mudança de preço
        price_change = data['close'].diff()
        
        # Sinal da mudança de preço (-1, 0, +1)
        sign = np.sign(price_change)
        
        # OFI = sign(price_change) * volume
        ofi_raw = sign * data['volume']
        
        # Soma sobre a janela e normaliza
        ofi_sum = ofi_raw.rolling(window=self.window).sum()
        
        # Normaliza pelo volume total da janela para obter [-1, +1]
        volume_sum = data['volume'].rolling(window=self.window).sum()
        ofi_normalized = ofi_sum / volume_sum.replace(0, np.nan)
        
        return ofi_normalized.fillna(0.0)
        
    def calculate_incremental(self, new_data: Any, state: Dict) -> float:
        """
        Calcula o OFI incrementalmente.
        
        Args:
            new_data: Dict com 'close', 'volume' e 'prev_close'
            state: Dict com 'ofi_values' (lista) e 'volume_values' (lista)
            
        Returns:
            Valor atual do OFI
        """
        if not self.enabled:
            return 0.0
            
        # Inicializa estado se necessário
        if 'ofi_values' not in state:
            state['ofi_values'] = []
            state['volume_values'] = []
            state['prev_close'] = None
            
        close = new_data.get('close', 0.0)
        volume = new_data.get('volume', 0.0)
        prev_close = state.get('prev_close')
        
        # Inicializa prev_close no primeiro tick
        if prev_close is None:
            prev_close = close
        
        # Calcula OFI para este ponto
        price_change = close - prev_close
        sign = 1 if price_change > 0 else (-1 if price_change < 0 else 0)
        ofi_value = sign * volume
        
        # Atualiza estado
        state['ofi_values'].append(ofi_value)
        state['volume_values'].append(volume)
        state['prev_close'] = close
        
        # Mantém apenas os últimos 'window' valores
        if len(state['ofi_values']) > self.window:
            state['ofi_values'] = state['ofi_values'][-self.window:]
            state['volume_values'] = state['volume_values'][-self.window:]
        
        # Calcula OFI normalizado
        ofi_sum = sum(state['ofi_values'])
        volume_sum = sum(state['volume_values'])
        
        if volume_sum == 0:
            return 0.0
            
        return ofi_sum / volume_sum
