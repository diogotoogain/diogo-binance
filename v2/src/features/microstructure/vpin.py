"""
VPIN (Volume-Synchronized Probability of Informed Trading) Feature.

Mede a probabilidade de traders informados no mercado.
VPIN = Σ|V_buy - V_sell| / Σ(V_buy + V_sell)
"""
from typing import Any, Dict, List
from collections import deque
import pandas as pd
import numpy as np

from v2.src.features.base import Feature


class VPIN(Feature):
    """
    VPIN (Volume-Synchronized Probability of Informed Trading).
    
    Detecta fluxo tóxico de traders informados.
    - VPIN alto (> 0.7) = Smart money agindo = Sinal forte
    - VPIN baixo (< 0.3) = Mercado calmo = Não operar
    
    Parâmetros do config:
        n_buckets: Número de buckets na janela (default: 50)
        bucket_size_usd: Tamanho do bucket em USD (default: 100000)
        
    OPTIMIZE: n_buckets em [20, 50, 100]
    """
    
    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """
        Inicializa o VPIN.
        
        Args:
            config: Deve conter 'n_buckets' e opcionalmente 'bucket_size_usd'
            enabled: Se a feature está habilitada
        """
        super().__init__(config, enabled)
        self.n_buckets = config.get('n_buckets', 50)
        self.bucket_size_usd = config.get('bucket_size_usd', 100000)
        
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        Calcula o VPIN para um DataFrame.
        
        Args:
            data: DataFrame com 'buy_volume' e 'sell_volume'
                  OU 'price', 'quantity' e 'is_buyer_maker'
            
        Returns:
            pd.Series com valores de VPIN em [0, 1]
        """
        if not self.enabled:
            return pd.Series(index=data.index, dtype=float)
        
        # Determina volumes de compra/venda
        if 'buy_volume' in data.columns and 'sell_volume' in data.columns:
            buy_vol = data['buy_volume']
            sell_vol = data['sell_volume']
        elif 'volume' in data.columns and 'close' in data.columns:
            # Estima usando mudança de preço
            price_change = data['close'].diff()
            buy_vol = data['volume'] * (price_change > 0).astype(float)
            sell_vol = data['volume'] * (price_change <= 0).astype(float)
        else:
            return pd.Series(index=data.index, dtype=float)
        
        # VPIN rolling: |buy - sell| / (buy + sell)
        imbalance = (buy_vol - sell_vol).abs().rolling(window=self.n_buckets).sum()
        total_vol = (buy_vol + sell_vol).rolling(window=self.n_buckets).sum()
        
        vpin = imbalance / total_vol.replace(0, np.nan)
        
        # Limita ao range [0, 1]
        vpin = vpin.clip(0, 1)
        
        return vpin.fillna(0.0)
        
    def calculate_incremental(self, new_data: Any, state: Dict) -> float:
        """
        Calcula o VPIN incrementalmente (bucket-by-bucket).
        
        Args:
            new_data: Dict com 'price', 'quantity', 'is_buyer_maker'
                      OU 'buy_volume', 'sell_volume'
            state: Dict com 'buckets' (deque de dicts com buy/sell volumes)
                   e 'current_bucket' (dict em construção)
            
        Returns:
            Valor atual do VPIN em [0, 1]
        """
        if not self.enabled:
            return 0.0
            
        # Inicializa estado se necessário
        if 'buckets' not in state:
            state['buckets'] = deque(maxlen=self.n_buckets)
            state['current_bucket'] = {
                'buy_volume': 0.0,
                'sell_volume': 0.0,
                'total_volume': 0.0
            }
        
        # Processa novo trade
        if 'buy_volume' in new_data and 'sell_volume' in new_data:
            buy_vol = new_data['buy_volume']
            sell_vol = new_data['sell_volume']
        else:
            price = new_data.get('price', 0.0)
            quantity = new_data.get('quantity', 0.0)
            is_buyer_maker = new_data.get('is_buyer_maker', True)
            
            volume_usd = price * quantity
            if is_buyer_maker:
                # Vendedor agrediu = volume de venda
                buy_vol = 0.0
                sell_vol = volume_usd
            else:
                # Comprador agrediu = volume de compra
                buy_vol = volume_usd
                sell_vol = 0.0
        
        # Atualiza bucket atual
        current = state['current_bucket']
        current['buy_volume'] += buy_vol
        current['sell_volume'] += sell_vol
        current['total_volume'] += buy_vol + sell_vol
        
        # Verifica se bucket está cheio
        if current['total_volume'] >= self.bucket_size_usd:
            # Completa bucket
            state['buckets'].append({
                'buy_volume': current['buy_volume'],
                'sell_volume': current['sell_volume']
            })
            
            # Reset bucket atual
            state['current_bucket'] = {
                'buy_volume': 0.0,
                'sell_volume': 0.0,
                'total_volume': 0.0
            }
        
        # Calcula VPIN
        if len(state['buckets']) < 10:  # Mínimo de buckets para cálculo
            return 0.0
            
        total_imbalance = 0.0
        total_volume = 0.0
        
        for bucket in state['buckets']:
            buy = bucket['buy_volume']
            sell = bucket['sell_volume']
            total_imbalance += abs(buy - sell)
            total_volume += buy + sell
        
        if total_volume == 0:
            return 0.0
            
        vpin = total_imbalance / total_volume
        return min(vpin, 1.0)
