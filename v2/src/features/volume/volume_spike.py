"""
Volume Spike Detection Feature.

Detecta picos de volume que podem indicar movimentos significativos.
"""
from typing import Any, Dict
import pandas as pd
import numpy as np

from v2.src.features.base import Feature


class VolumeSpike(Feature):
    """
    Volume Spike Detection.
    
    Detecta quando o volume atual está significativamente acima da média.
    Picos de volume geralmente precedem ou acompanham movimentos importantes.
    
    Fórmula:
        volume_spike = current_volume / rolling_mean(volume, lookback)
        is_spike = volume_spike > threshold_multiplier
    
    Parâmetros do config:
        lookback: Janela para cálculo da média (default: 20)
        threshold_multiplier: Multiplicador para detectar spike (default: 2.0)
        
    OPTIMIZE: lookback em [10, 20, 30, 50]
              threshold_multiplier em [1.5, 2.0, 2.5, 3.0]
    """
    
    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """
        Inicializa o Volume Spike.
        
        Args:
            config: Deve conter 'lookback' e 'threshold_multiplier'
            enabled: Se a feature está habilitada
        """
        super().__init__(config, enabled)
        self.lookback = config.get('lookback', 20)
        self.threshold_multiplier = config.get('threshold_multiplier', 2.0)
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula Volume Spike para um DataFrame.
        
        Args:
            data: DataFrame com coluna 'volume'
            
        Returns:
            pd.DataFrame com colunas 'volume_spike_ratio' e 'is_spike'
        """
        if not self.enabled:
            return pd.DataFrame(index=data.index)
            
        if 'volume' not in data.columns:
            return pd.DataFrame(index=data.index)
        
        volume = data['volume']
        
        # Média móvel do volume
        avg_volume = volume.rolling(window=self.lookback).mean()
        
        # Ratio: volume atual / média
        spike_ratio = volume / avg_volume.replace(0, np.nan)
        
        # Detecta spikes
        is_spike = (spike_ratio > self.threshold_multiplier).astype(int)
        
        result = pd.DataFrame({
            'volume_spike_ratio': spike_ratio.fillna(1.0),
            'is_spike': is_spike
        }, index=data.index)
        
        return result
        
    def calculate_incremental(self, new_data: Any, state: Dict) -> Dict[str, float]:
        """
        Calcula Volume Spike incrementalmente.
        
        Args:
            new_data: Dict com 'volume'
            state: Dict com 'volumes' (lista de volumes recentes)
            
        Returns:
            Dict com 'volume_spike_ratio' e 'is_spike'
        """
        if not self.enabled:
            return {'volume_spike_ratio': 1.0, 'is_spike': 0}
        
        volume = new_data.get('volume', 0.0)
        
        # Inicializa estado se necessário
        if 'volumes' not in state:
            state['volumes'] = []
        
        # Adiciona volume atual
        state['volumes'].append(volume)
        
        # Mantém apenas os últimos 'lookback' volumes
        if len(state['volumes']) > self.lookback:
            state['volumes'] = state['volumes'][-self.lookback:]
        
        # Calcula média
        if len(state['volumes']) == 0:
            return {'volume_spike_ratio': 1.0, 'is_spike': 0}
        
        avg_volume = sum(state['volumes']) / len(state['volumes'])
        
        # Spike ratio
        if avg_volume == 0:
            spike_ratio = 1.0
        else:
            spike_ratio = volume / avg_volume
        
        # Detecta spike
        is_spike = 1 if spike_ratio > self.threshold_multiplier else 0
        
        return {
            'volume_spike_ratio': spike_ratio,
            'is_spike': is_spike
        }
    
    def get_spike_direction(
        self, data: pd.DataFrame, spike_data: pd.DataFrame
    ) -> pd.Series:
        """
        Determina a direção dos spikes de volume.
        
        Args:
            data: DataFrame com 'close'
            spike_data: DataFrame com 'is_spike'
            
        Returns:
            pd.Series com 1 (spike up), -1 (spike down), 0 (no spike)
        """
        if 'close' not in data.columns or 'is_spike' not in spike_data.columns:
            return pd.Series(0, index=data.index)
        
        price_change = data['close'].diff()
        is_spike = spike_data['is_spike']
        
        direction = pd.Series(0, index=data.index)
        direction[(is_spike == 1) & (price_change > 0)] = 1
        direction[(is_spike == 1) & (price_change < 0)] = -1
        
        return direction
