"""
Micro Price Feature.

Preço ajustado pelo desequilíbrio do order book.
micro_price = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
"""
from typing import Any, Dict
import pandas as pd
import numpy as np

from v2.src.features.base import Feature


class MicroPrice(Feature):
    """
    Micro Price.
    
    Preço que leva em conta a pressão de compra/venda do order book.
    Mais preciso que o mid-price para prever movimento de curto prazo.
    
    Fórmula:
        micro_price = (bid * ask_size + ask * bid_size) / (bid_size + ask_size)
    """
    
    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """
        Inicializa o Micro Price.
        
        Args:
            config: Configuração (não usa parâmetros específicos)
            enabled: Se a feature está habilitada
        """
        super().__init__(config, enabled)
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcula o Micro Price para um DataFrame.
        
        Args:
            data: DataFrame com colunas 'bid', 'ask', 'bid_size', 'ask_size'
            
        Returns:
            pd.Series com valores de Micro Price
        """
        if not self.enabled:
            return pd.Series(index=data.index, dtype=float)
            
        required_cols = ['bid', 'ask', 'bid_size', 'ask_size']
        if not all(col in data.columns for col in required_cols):
            # Fallback para mid-price se não temos order book
            if 'close' in data.columns:
                return data['close']
            return pd.Series(index=data.index, dtype=float)
        
        bid = data['bid']
        ask = data['ask']
        bid_size = data['bid_size']
        ask_size = data['ask_size']
        
        # Micro price: ponderado pelo tamanho oposto do book
        total_size = bid_size + ask_size
        micro_price = (bid * ask_size + ask * bid_size) / total_size.replace(0, np.nan)
        
        return micro_price.fillna((bid + ask) / 2)
        
    def calculate_incremental(self, new_data: Any, state: Dict) -> float:
        """
        Calcula o Micro Price incrementalmente.
        
        Args:
            new_data: Dict com 'bid', 'ask', 'bid_size', 'ask_size'
            state: Não utilizado (sem estado)
            
        Returns:
            Valor atual do Micro Price
        """
        if not self.enabled:
            return 0.0
        
        bid = new_data.get('bid', 0.0)
        ask = new_data.get('ask', 0.0)
        bid_size = new_data.get('bid_size', 0.0)
        ask_size = new_data.get('ask_size', 0.0)
        
        total_size = bid_size + ask_size
        
        if total_size == 0:
            # Fallback para mid-price
            if bid > 0 and ask > 0:
                return (bid + ask) / 2
            return new_data.get('close', 0.0)
        
        return (bid * ask_size + ask * bid_size) / total_size
