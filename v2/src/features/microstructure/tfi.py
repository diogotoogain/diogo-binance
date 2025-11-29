"""
Trade Flow Imbalance (TFI) Feature.

Mede o desequilíbrio entre volume de compra e venda.
TFI = (buy_volume - sell_volume) / total_volume
"""
from typing import Any, Dict
import pandas as pd
import numpy as np

from v2.src.features.base import Feature


class TFI(Feature):
    """
    Trade Flow Imbalance (TFI).
    
    Mede a pressão de compra/venda baseada em volume classificado.
    
    Parâmetros do config:
        window: Janela de cálculo (default: 20)
        
    OPTIMIZE: window em [5, 10, 20, 50, 100]
    """
    
    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """
        Inicializa o TFI.
        
        Args:
            config: Deve conter 'window' (tamanho da janela)
            enabled: Se a feature está habilitada
        """
        super().__init__(config, enabled)
        self.window = config.get('window', 20)
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcula o TFI para um DataFrame.
        
        Args:
            data: DataFrame com colunas 'buy_volume' e 'sell_volume'
                  OU 'volume' e 'close' para estimativa
            
        Returns:
            pd.Series com valores de TFI em [-1, +1]
        """
        if not self.enabled:
            return pd.Series(index=data.index, dtype=float)
        
        # Se temos volumes de compra/venda explícitos
        if 'buy_volume' in data.columns and 'sell_volume' in data.columns:
            buy_vol = data['buy_volume'].rolling(window=self.window).sum()
            sell_vol = data['sell_volume'].rolling(window=self.window).sum()
        else:
            # Estima usando close e volume
            # Trades em alta = compra, trades em baixa = venda
            if 'close' not in data.columns or 'volume' not in data.columns:
                return pd.Series(index=data.index, dtype=float)
                
            price_change = data['close'].diff()
            # Volume up = buy volume, Volume down = sell volume
            buy_vol = (data['volume'] * (price_change > 0).astype(float)).rolling(
                window=self.window
            ).sum()
            sell_vol = (data['volume'] * (price_change < 0).astype(float)).rolling(
                window=self.window
            ).sum()
        
        total_vol = buy_vol + sell_vol
        
        # TFI = (buy - sell) / total
        tfi = (buy_vol - sell_vol) / total_vol.replace(0, np.nan)
        
        return tfi.fillna(0.0)
        
    def calculate_incremental(self, new_data: Any, state: Dict) -> float:
        """
        Calcula o TFI incrementalmente.
        
        Args:
            new_data: Dict com 'buy_volume' e 'sell_volume'
                      OU 'volume', 'close' e 'prev_close'
            state: Dict com 'buy_volumes' e 'sell_volumes' (listas)
            
        Returns:
            Valor atual do TFI em [-1, +1]
        """
        if not self.enabled:
            return 0.0
            
        # Inicializa estado se necessário
        if 'buy_volumes' not in state:
            state['buy_volumes'] = []
            state['sell_volumes'] = []
            state['prev_close'] = None
        
        # Obtém volumes de compra/venda
        if 'buy_volume' in new_data and 'sell_volume' in new_data:
            buy_vol = new_data['buy_volume']
            sell_vol = new_data['sell_volume']
        else:
            # Estima baseado em mudança de preço
            close = new_data.get('close', 0.0)
            volume = new_data.get('volume', 0.0)
            prev_close = state.get('prev_close', close)
            
            price_change = close - prev_close
            if price_change > 0:
                buy_vol = volume
                sell_vol = 0.0
            elif price_change < 0:
                buy_vol = 0.0
                sell_vol = volume
            else:
                buy_vol = volume / 2
                sell_vol = volume / 2
                
            state['prev_close'] = close
        
        # Atualiza estado
        state['buy_volumes'].append(buy_vol)
        state['sell_volumes'].append(sell_vol)
        
        # Mantém apenas os últimos 'window' valores
        if len(state['buy_volumes']) > self.window:
            state['buy_volumes'] = state['buy_volumes'][-self.window:]
            state['sell_volumes'] = state['sell_volumes'][-self.window:]
        
        # Calcula TFI
        total_buy = sum(state['buy_volumes'])
        total_sell = sum(state['sell_volumes'])
        total_vol = total_buy + total_sell
        
        if total_vol == 0:
            return 0.0
            
        return (total_buy - total_sell) / total_vol
