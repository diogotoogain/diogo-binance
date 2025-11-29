"""
Feature Pipeline.

Orquestra o cálculo de todas as features configuradas.
"""
from typing import Any, Dict, List, Optional
import pandas as pd
import logging

from v2.src.features.base import Feature
from v2.src.features.microstructure import OFI, TFI, MicroPrice, ShannonEntropy, VPIN
from v2.src.features.technical import EMA, RSI, MACD, ADX, BollingerBands, ATR
from v2.src.features.volume import VolumeSpike, LiquidityClusters
from v2.src.features.temporal import DayOfWeekFeatures


logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Orquestra cálculo de todas as features.
    
    Inicializa features baseado na configuração (enabled/disabled)
    e coordena cálculos batch e incrementais.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o pipeline de features.
        
        Args:
            config: Dicionário de configuração com seção 'features'
        """
        self.config = config
        self.features: List[Feature] = []
        self.feature_states: Dict[str, Dict] = {}  # Estados para cálculo incremental
        self._initialize_features()
        
    def _initialize_features(self) -> None:
        """Inicializa features baseado no config (enabled/disabled)."""
        features_config = self.config.get('features', {})
        
        # Microstructure features
        micro_config = features_config.get('microstructure', {})
        if micro_config.get('enabled', True):
            self._init_microstructure_features(micro_config)
        
        # Technical indicators
        tech_config = features_config.get('technical', {})
        if tech_config.get('enabled', True):
            self._init_technical_features(tech_config)
        
        # Volume analysis
        vol_config = features_config.get('volume_analysis', {})
        if vol_config.get('enabled', True):
            self._init_volume_features(vol_config)
        
        # Temporal features
        temporal_config = features_config.get('temporal', {})
        if temporal_config.get('enabled', True):
            self._init_temporal_features(temporal_config)
        
        logger.info(f"Pipeline inicializado com {len(self.features)} features")
        
    def _init_microstructure_features(self, config: Dict[str, Any]) -> None:
        """Inicializa features de microestrutura."""
        # OFI
        ofi_config = config.get('ofi', {})
        if ofi_config.get('enabled', True):
            self.features.append(OFI(ofi_config, enabled=True))
            
        # TFI
        tfi_config = config.get('tfi', {})
        if tfi_config.get('enabled', True):
            self.features.append(TFI(tfi_config, enabled=True))
            
        # Micro Price
        mp_config = config.get('micro_price', {})
        if mp_config.get('enabled', True):
            self.features.append(MicroPrice(mp_config, enabled=True))
            
        # Shannon Entropy
        entropy_config = config.get('entropy', {})
        if entropy_config.get('enabled', True):
            self.features.append(ShannonEntropy(entropy_config, enabled=True))
            
        # VPIN
        vpin_config = config.get('vpin', {})
        if vpin_config.get('enabled', True):
            self.features.append(VPIN(vpin_config, enabled=True))
            
    def _init_technical_features(self, config: Dict[str, Any]) -> None:
        """Inicializa indicadores técnicos."""
        # EMA
        ema_config = config.get('ema', {})
        if ema_config.get('enabled', True):
            self.features.append(EMA(ema_config, enabled=True))
            
        # RSI
        rsi_config = config.get('rsi', {})
        if rsi_config.get('enabled', True):
            self.features.append(RSI(rsi_config, enabled=True))
            
        # MACD
        macd_config = config.get('macd', {})
        if macd_config.get('enabled', True):
            self.features.append(MACD(macd_config, enabled=True))
            
        # ADX
        adx_config = config.get('adx', {})
        if adx_config.get('enabled', True):
            self.features.append(ADX(adx_config, enabled=True))
            
        # Bollinger Bands
        bb_config = config.get('bollinger', {})
        if bb_config.get('enabled', True):
            self.features.append(BollingerBands(bb_config, enabled=True))
            
        # ATR
        atr_config = config.get('atr', {})
        if atr_config.get('enabled', True):
            self.features.append(ATR(atr_config, enabled=True))
            
    def _init_volume_features(self, config: Dict[str, Any]) -> None:
        """Inicializa features de volume."""
        # Volume Spike
        spike_config = config.get('volume_spike', {})
        if spike_config.get('enabled', True):
            self.features.append(VolumeSpike(spike_config, enabled=True))
            
        # Liquidity Clusters
        clusters_config = config.get('liquidity_clusters', {})
        if clusters_config.get('enabled', True):
            self.features.append(LiquidityClusters(clusters_config, enabled=True))
            
    def _init_temporal_features(self, config: Dict[str, Any]) -> None:
        """Inicializa features temporais."""
        # Day of Week
        dow_config = config.get('day_of_week', {})
        if dow_config.get('enabled', True):
            self.features.append(DayOfWeekFeatures(dow_config, enabled=True))
            
    def calculate_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula todas as features habilitadas.
        
        Args:
            data: DataFrame com dados de mercado
            
        Returns:
            DataFrame com todas as features calculadas
        """
        result = pd.DataFrame(index=data.index)
        
        for feature in self.features:
            if not feature.is_enabled():
                continue
                
            try:
                calculated = feature.calculate(data)
                
                if isinstance(calculated, pd.Series):
                    result[feature.get_name()] = calculated
                elif isinstance(calculated, pd.DataFrame):
                    # Múltiplas colunas (ex: MACD, EMA múltiplos)
                    for col in calculated.columns:
                        result[col] = calculated[col]
                        
            except Exception as e:
                logger.error(f"Erro calculando {feature.get_name()}: {e}")
                
        return result
        
    def calculate_incremental(
        self, new_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calcula incrementalmente para real-time.
        
        Args:
            new_data: Dict com dados do novo tick/bar
            
        Returns:
            Dict com valores atuais de todas as features
        """
        result = {}
        
        for feature in self.features:
            if not feature.is_enabled():
                continue
                
            feature_name = feature.get_name()
            
            # Obtém ou cria estado para esta feature
            if feature_name not in self.feature_states:
                self.feature_states[feature_name] = {}
            
            state = self.feature_states[feature_name]
            
            try:
                calculated = feature.calculate_incremental(new_data, state)
                
                if isinstance(calculated, dict):
                    # Múltiplos valores (ex: MACD)
                    for key, value in calculated.items():
                        result[key] = value
                else:
                    result[feature_name] = calculated
                    
            except Exception as e:
                logger.error(f"Erro calculando incremental {feature_name}: {e}")
                
        return result
        
    def get_feature_names(self) -> List[str]:
        """
        Retorna nomes das features ativas.
        
        Returns:
            Lista de nomes das features habilitadas
        """
        return [f.get_name() for f in self.features if f.is_enabled()]
    
    def get_feature(self, name: str) -> Optional[Feature]:
        """
        Obtém uma feature específica pelo nome.
        
        Args:
            name: Nome da feature
            
        Returns:
            Feature ou None se não encontrada
        """
        for feature in self.features:
            if feature.get_name() == name:
                return feature
        return None
    
    def enable_feature(self, name: str) -> bool:
        """
        Habilita uma feature.
        
        Args:
            name: Nome da feature
            
        Returns:
            True se encontrou e habilitou, False caso contrário
        """
        feature = self.get_feature(name)
        if feature:
            feature.set_enabled(True)
            return True
        return False
    
    def disable_feature(self, name: str) -> bool:
        """
        Desabilita uma feature.
        
        Args:
            name: Nome da feature
            
        Returns:
            True se encontrou e desabilitou, False caso contrário
        """
        feature = self.get_feature(name)
        if feature:
            feature.set_enabled(False)
            return True
        return False
    
    def reset_states(self) -> None:
        """Reseta todos os estados incrementais."""
        self.feature_states.clear()
        
    def get_all_params(self) -> Dict[str, Dict[str, Any]]:
        """
        Retorna parâmetros de todas as features.
        
        Returns:
            Dict com nome da feature -> parâmetros
        """
        return {f.get_name(): f.get_params() for f in self.features}
