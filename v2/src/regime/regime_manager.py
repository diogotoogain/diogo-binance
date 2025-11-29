"""
Regime Manager Module.

Orchestrates multiple regime detectors to provide unified regime information.
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np

from v2.src.regime.hmm_detector import HMMRegimeDetector
from v2.src.regime.adx_regime import ADXRegimeDetector
from v2.src.regime.volatility_regime import VolatilityRegimeDetector


class RegimeManager:
    """
    Orquestra múltiplos detectores de regime.
    
    Combina sinais de HMM, ADX e Volatility para decisão final.
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o gerenciador de regime.
        
        Args:
            config: Dicionário de configuração
        """
        self.config = config
        
        # Inicializa detectores se habilitados
        hmm_enabled = config.get('regime', {}).get('hmm', {}).get('enabled', True)
        adx_enabled = config.get('regime', {}).get('adx_based', {}).get('enabled', True)
        vol_enabled = config.get('regime', {}).get('volatility', {}).get('enabled', True)
        
        self.hmm = HMMRegimeDetector(config) if hmm_enabled else None
        self.adx = ADXRegimeDetector(config) if adx_enabled else None
        self.vol = VolatilityRegimeDetector(config) if vol_enabled else None
        
        # Multiplicadores de risco por regime combinado
        risk_adjustment = config.get('risk_adjustment', {})
        self.risk_multipliers = {
            'favorable': risk_adjustment.get('favorable_multiplier', 1.0),
            'neutral': risk_adjustment.get('neutral_multiplier', 0.7),
            'unfavorable': risk_adjustment.get('unfavorable_multiplier', 0.3)
        }
        
        # Pesos de estratégia por regime ADX
        self.strategy_weights = config.get('strategy_weights', {
            'trending': {'momentum': 0.6, 'mean_reversion': 0.1, 'volatility_breakout': 0.3},
            'ranging': {'momentum': 0.1, 'mean_reversion': 0.6, 'volatility_breakout': 0.3},
            'transition': {'momentum': 0.33, 'mean_reversion': 0.33, 'volatility_breakout': 0.34}
        })
        
    def fit(self, data: pd.DataFrame) -> 'RegimeManager':
        """
        Treina todos os detectores que precisam de treinamento.
        
        Args:
            data: DataFrame com dados históricos
            
        Returns:
            self para encadeamento
        """
        # Treina HMM se habilitado
        if self.hmm is not None:
            self.hmm.fit(data)
            
        # Treina detector de volatilidade se habilitado
        if self.vol is not None:
            if 'volatility' in data.columns:
                self.vol.fit(data['volatility'])
            elif 'close' in data.columns:
                # Calcula volatilidade rolling
                volatility = data['close'].pct_change().rolling(20).std()
                self.vol.fit(volatility.dropna())
                
        return self
        
    def get_current_regime(self, data: pd.DataFrame) -> Dict:
        """
        Retorna regime atual de todos os detectores.
        
        Args:
            data: DataFrame com dados recentes (inclui close, adx, volatility)
            
        Returns:
            Dicionário com regimes de cada detector e regime combinado:
            {
                "hmm": 0/1/2,
                "adx": "trending"/"ranging"/"transition",
                "volatility": "low"/"normal"/"high"/"extreme",
                "combined": "favorable"/"neutral"/"unfavorable"
            }
        """
        result = {
            'hmm': None,
            'adx': None,
            'volatility': None,
            'combined': 'neutral'
        }
        
        # Regime HMM
        if self.hmm is not None and self.hmm.is_fitted:
            result['hmm'] = self.hmm.predict(data)
            
        # Regime ADX
        if self.adx is not None:
            adx_value = self._get_adx_value(data)
            if adx_value is not None:
                result['adx'] = self.adx.detect(adx_value)
                
        # Regime Volatilidade
        if self.vol is not None and self.vol.is_fitted:
            vol_value = self._get_volatility_value(data)
            if vol_value is not None:
                result['volatility'] = self.vol.detect(vol_value)
                
        # Combina regimes para decisão final
        result['combined'] = self._combine_regimes(result)
        
        return result
        
    def _get_adx_value(self, data: pd.DataFrame) -> Optional[float]:
        """Extrai valor do ADX dos dados."""
        if 'adx' in data.columns:
            value = data['adx'].iloc[-1]
            return float(value) if not pd.isna(value) else None
        return None
        
    def _get_volatility_value(self, data: pd.DataFrame) -> Optional[float]:
        """Extrai valor de volatilidade dos dados."""
        if 'volatility' in data.columns:
            value = data['volatility'].iloc[-1]
            return float(value) if not pd.isna(value) else None
        elif 'close' in data.columns:
            # Calcula volatilidade rolling
            vol = data['close'].pct_change().rolling(20).std().iloc[-1]
            return float(vol) if not pd.isna(vol) else None
        return None
        
    def _combine_regimes(self, regimes: Dict) -> str:
        """
        Combina diferentes sinais de regime em uma decisão.
        
        Lógica:
        - Favorável: trending + baixa/normal vol
        - Desfavorável: ranging + alta vol OU extreme_vol
        - Neutro: outros casos
        
        Args:
            regimes: Dicionário com regimes de cada detector
            
        Returns:
            "favorable", "neutral", ou "unfavorable"
        """
        adx_regime = regimes.get('adx')
        vol_regime = regimes.get('volatility')
        
        # Casos desfavoráveis
        if vol_regime == 'extreme_vol':
            return 'unfavorable'
        if adx_regime == 'ranging' and vol_regime in ['high_vol', 'extreme_vol']:
            return 'unfavorable'
            
        # Casos favoráveis
        if adx_regime == 'trending' and vol_regime in ['low_vol', 'normal_vol']:
            return 'favorable'
        if adx_regime == 'trending' and vol_regime is None:
            return 'favorable'
            
        return 'neutral'
        
    def get_risk_multiplier(self, regime: Dict) -> float:
        """
        Retorna multiplicador de risco baseado no regime.
        
        Em regimes desfavoráveis, reduz exposição.
        
        Args:
            regime: Dicionário com regime combinado
            
        Returns:
            Multiplicador entre 0 e 1
        """
        combined = regime.get('combined', 'neutral')
        return self.risk_multipliers.get(combined, 0.7)
        
    def get_strategy_weights(self, regime: Dict) -> Dict[str, float]:
        """
        Retorna pesos para cada estratégia baseado no regime.
        
        Args:
            regime: Dicionário com regime (deve ter 'adx')
            
        Returns:
            Dicionário com pesos por estratégia
        """
        adx_regime = regime.get('adx', 'transition')
        
        if adx_regime not in self.strategy_weights:
            adx_regime = 'transition'
            
        base_weights = self.strategy_weights[adx_regime].copy()
        
        # Ajusta pesos baseado na volatilidade
        vol_regime = regime.get('volatility')
        if vol_regime in ['high_vol', 'extreme_vol']:
            # Reduz peso de estratégias agressivas
            for strategy in base_weights:
                if strategy == 'momentum':
                    base_weights[strategy] *= 0.5
                    
        return base_weights
        
    def get_full_regime_info(self, data: pd.DataFrame) -> Dict:
        """
        Retorna informações completas sobre o regime atual.
        
        Args:
            data: DataFrame com dados recentes
            
        Returns:
            Dicionário completo com informações de regime
        """
        regimes = self.get_current_regime(data)
        
        info = {
            'regimes': regimes,
            'risk_multiplier': self.get_risk_multiplier(regimes),
            'strategy_weights': self.get_strategy_weights(regimes),
            'detectors': {
                'hmm': {'enabled': self.hmm is not None, 'fitted': self.hmm.is_fitted if self.hmm else False},
                'adx': {'enabled': self.adx is not None},
                'volatility': {'enabled': self.vol is not None, 'fitted': self.vol.is_fitted if self.vol else False}
            }
        }
        
        # Adiciona estatísticas do HMM se disponível
        if self.hmm is not None and self.hmm.is_fitted:
            info['hmm_stats'] = self.hmm.get_regime_stats()
            
        # Adiciona thresholds do ADX
        if self.adx is not None:
            info['adx_thresholds'] = {
                'trending': self.adx.trending_threshold,
                'ranging': self.adx.ranging_threshold
            }
            
        # Adiciona thresholds de volatilidade
        if self.vol is not None and self.vol.is_fitted:
            info['volatility_thresholds'] = self.vol.get_thresholds()
            
        return info
        
    def should_reduce_exposure(self, data: pd.DataFrame) -> bool:
        """
        Verifica se deve reduzir exposição baseado no regime.
        
        Args:
            data: DataFrame com dados recentes
            
        Returns:
            True se deve reduzir exposição
        """
        regime = self.get_current_regime(data)
        return regime.get('combined') == 'unfavorable'
        
    def is_favorable_for_strategy(self, data: pd.DataFrame, strategy: str) -> bool:
        """
        Verifica se o regime é favorável para uma estratégia específica.
        
        Args:
            data: DataFrame com dados recentes
            strategy: Nome da estratégia
            
        Returns:
            True se regime é favorável para a estratégia
        """
        regime = self.get_current_regime(data)
        weights = self.get_strategy_weights(regime)
        
        # Considera favorável se peso > 0.3
        return weights.get(strategy, 0) > 0.3
