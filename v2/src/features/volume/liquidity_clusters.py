"""
Liquidity Clusters Feature.

Identifica níveis de preço com alta concentração de volume.
"""
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np

from v2.src.features.base import Feature


class LiquidityClusters(Feature):
    """
    Liquidity Clusters.
    
    Identifica níveis de preço com alta concentração de volume,
    que podem atuar como suportes e resistências.
    
    Parâmetros do config:
        levels: Número de níveis a identificar (default: 10)
        threshold_percentile: Percentil mínimo de volume para ser cluster (default: 80)
        
    OPTIMIZE: levels em [5, 10, 15, 20]
              threshold_percentile em [70, 80, 90]
    """
    
    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """
        Inicializa o Liquidity Clusters.
        
        Args:
            config: Deve conter 'levels' e 'threshold_percentile'
            enabled: Se a feature está habilitada
        """
        super().__init__(config, enabled)
        self.levels = config.get('levels', 10)
        self.threshold_percentile = config.get('threshold_percentile', 80)
        self.recalc_frequency = config.get('recalc_frequency', 100)
        self.cluster_threshold_pct = config.get('cluster_threshold_pct', 0.1)
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula Liquidity Clusters para um DataFrame.
        
        Args:
            data: DataFrame com colunas 'close' (ou 'price') e 'volume'
            
        Returns:
            pd.DataFrame com 'cluster_distance' (distância ao cluster mais próximo)
            e 'in_cluster' (se está em zona de cluster)
        """
        if not self.enabled:
            return pd.DataFrame(index=data.index)
        
        if 'volume' not in data.columns:
            return pd.DataFrame(index=data.index)
        
        price_col = 'close' if 'close' in data.columns else 'price'
        if price_col not in data.columns:
            return pd.DataFrame(index=data.index)
        
        prices = data[price_col]
        volumes = data['volume']
        
        # Cria perfil de volume (volume @ price)
        n_bins = min(100, len(data) // 5)
        if n_bins < 5:
            n_bins = max(5, len(data))
        
        # Discretiza preços em bins
        price_min = prices.min()
        price_max = prices.max()
        
        if price_min == price_max:
            return pd.DataFrame({
                'cluster_distance': pd.Series(0.0, index=data.index),
                'in_cluster': pd.Series(0, index=data.index)
            })
        
        bin_size = (price_max - price_min) / n_bins
        
        # Cria histograma de volume por preço
        volume_profile = {}
        for price, volume in zip(prices, volumes):
            if pd.isna(price) or pd.isna(volume):
                continue
            bin_idx = int((price - price_min) / bin_size)
            bin_idx = min(bin_idx, n_bins - 1)  # Garante que não ultrapassa
            bin_price = price_min + (bin_idx + 0.5) * bin_size
            
            if bin_price not in volume_profile:
                volume_profile[bin_price] = 0
            volume_profile[bin_price] += volume
        
        if not volume_profile:
            return pd.DataFrame({
                'cluster_distance': pd.Series(0.0, index=data.index),
                'in_cluster': pd.Series(0, index=data.index)
            })
        
        # Identifica clusters (top níveis por volume)
        volume_threshold = np.percentile(
            list(volume_profile.values()), 
            self.threshold_percentile
        )
        
        cluster_levels = [
            price for price, vol in volume_profile.items() 
            if vol >= volume_threshold
        ]
        
        if not cluster_levels:
            cluster_levels = list(volume_profile.keys())[:self.levels]
        
        # Limita ao número de níveis configurado
        if len(cluster_levels) > self.levels:
            # Pega os com maior volume
            sorted_levels = sorted(
                [(p, volume_profile[p]) for p in cluster_levels],
                key=lambda x: x[1],
                reverse=True
            )
            cluster_levels = [p for p, _ in sorted_levels[:self.levels]]
        
        # Calcula distância ao cluster mais próximo para cada preço
        def distance_to_nearest_cluster(price: float) -> float:
            if pd.isna(price) or not cluster_levels:
                return 0.0
            return min(abs(price - level) for level in cluster_levels)
        
        cluster_distances = prices.apply(distance_to_nearest_cluster)
        
        # Verifica se está em zona de cluster (dentro de 0.1% de um cluster)
        cluster_threshold = (price_max - price_min) * 0.001  # 0.1%
        in_cluster = (cluster_distances < cluster_threshold).astype(int)
        
        result = pd.DataFrame({
            'cluster_distance': cluster_distances,
            'in_cluster': in_cluster
        }, index=data.index)
        
        return result
        
    def calculate_incremental(self, new_data: Any, state: Dict) -> Dict[str, float]:
        """
        Calcula Liquidity Clusters incrementalmente.
        
        Args:
            new_data: Dict com 'price' (ou 'close') e 'volume'
            state: Dict com 'volume_profile' e 'cluster_levels'
            
        Returns:
            Dict com 'cluster_distance' e 'in_cluster'
        """
        if not self.enabled:
            return {'cluster_distance': 0.0, 'in_cluster': 0}
        
        price = new_data.get('close', new_data.get('price', 0.0))
        volume = new_data.get('volume', 0.0)
        
        # Inicializa estado se necessário
        if 'volume_profile' not in state:
            state['volume_profile'] = {}
            state['cluster_levels'] = []
            state['prices'] = []
            state['update_counter'] = 0
        
        # Atualiza perfil de volume
        # Usa bins de preço com precisão de 0.1%
        if price > 0:
            # Normaliza para bin
            bin_size = price * 0.001  # 0.1% bins
            bin_price = round(price / bin_size) * bin_size
            
            if bin_price not in state['volume_profile']:
                state['volume_profile'][bin_price] = 0
            state['volume_profile'][bin_price] += volume
        
        state['prices'].append(price)
        state['update_counter'] += 1
        
        # Recalcula clusters periodicamente (configurável)
        if state['update_counter'] >= self.recalc_frequency:
            state['update_counter'] = 0
            self._update_cluster_levels(state)
        
        # Calcula distância ao cluster mais próximo
        cluster_levels = state['cluster_levels']
        
        if not cluster_levels or price <= 0:
            return {'cluster_distance': 0.0, 'in_cluster': 0}
        
        cluster_distance = min(abs(price - level) for level in cluster_levels)
        
        # Verifica se está em zona de cluster (configurável via cluster_threshold_pct)
        cluster_threshold = price * self.cluster_threshold_pct / 100  # Convertido para porcentagem
        in_cluster = 1 if cluster_distance < cluster_threshold else 0
        
        return {
            'cluster_distance': cluster_distance,
            'in_cluster': in_cluster
        }
    
    def _update_cluster_levels(self, state: Dict) -> None:
        """Atualiza níveis de cluster baseado no volume profile atual."""
        volume_profile = state['volume_profile']
        
        if not volume_profile:
            state['cluster_levels'] = []
            return
        
        # Threshold de volume
        volumes = list(volume_profile.values())
        volume_threshold = np.percentile(volumes, self.threshold_percentile)
        
        # Identifica clusters
        cluster_levels = [
            price for price, vol in volume_profile.items()
            if vol >= volume_threshold
        ]
        
        # Limita ao número de níveis
        if len(cluster_levels) > self.levels:
            sorted_levels = sorted(
                [(p, volume_profile[p]) for p in cluster_levels],
                key=lambda x: x[1],
                reverse=True
            )
            cluster_levels = [p for p, _ in sorted_levels[:self.levels]]
        
        state['cluster_levels'] = cluster_levels
    
    def get_support_resistance_levels(
        self, data: pd.DataFrame, current_price: float
    ) -> Tuple[List[float], List[float]]:
        """
        Retorna níveis de suporte e resistência.
        
        Args:
            data: DataFrame com histórico de preços e volumes
            current_price: Preço atual
            
        Returns:
            Tupla (supports, resistances) com listas de níveis
        """
        cluster_data = self.calculate(data)
        
        if cluster_data.empty:
            return [], []
        
        # Recalcula clusters para obter níveis
        prices = data['close'] if 'close' in data.columns else data['price']
        volumes = data['volume']
        
        # Volume profile simplificado
        volume_profile = {}
        price_min = prices.min()
        price_max = prices.max()
        n_bins = 50
        bin_size = (price_max - price_min) / n_bins if price_min != price_max else 1
        
        for price, volume in zip(prices, volumes):
            if pd.isna(price):
                continue
            bin_idx = int((price - price_min) / bin_size) if bin_size > 0 else 0
            bin_idx = min(bin_idx, n_bins - 1)
            bin_price = price_min + (bin_idx + 0.5) * bin_size
            
            if bin_price not in volume_profile:
                volume_profile[bin_price] = 0
            volume_profile[bin_price] += volume
        
        # Identifica níveis de alto volume
        if not volume_profile:
            return [], []
        
        volume_threshold = np.percentile(
            list(volume_profile.values()),
            self.threshold_percentile
        )
        
        cluster_levels = [
            price for price, vol in volume_profile.items()
            if vol >= volume_threshold
        ]
        
        # Separa em suportes e resistências
        supports = sorted([l for l in cluster_levels if l < current_price], reverse=True)
        resistances = sorted([l for l in cluster_levels if l > current_price])
        
        return supports[:self.levels // 2], resistances[:self.levels // 2]
