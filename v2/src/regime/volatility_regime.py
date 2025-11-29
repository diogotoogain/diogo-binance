"""
Volatility-based Regime Detection.

Detects market regimes based on volatility percentiles.
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np


class VolatilityRegimeDetector:
    """
    Detecção de regime baseada em volatilidade.
    
    Usa percentis históricos de volatilidade para classificar.
    
    Regimes:
    - "low_vol": Volatilidade abaixo do percentil baixo
    - "normal_vol": Volatilidade entre percentis baixo e alto
    - "high_vol": Volatilidade entre percentil alto e extremo
    - "extreme_vol": Volatilidade acima do percentil extremo
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o detector de volatilidade com parâmetros do config.
        
        Args:
            config: Dicionário de configuração
        """
        vol_config = config.get('regime', {}).get('volatility', {})
        self.lookback = vol_config.get('lookback', 100)
        self.low_percentile = vol_config.get('low_percentile', 25)
        self.high_percentile = vol_config.get('high_percentile', 75)
        self.extreme_percentile = vol_config.get('extreme_percentile', 95)
        
        # Valores calculados após fit
        self._low_threshold: Optional[float] = None
        self._high_threshold: Optional[float] = None
        self._extreme_threshold: Optional[float] = None
        self._is_fitted = False
        
    def fit(self, volatility_series: pd.Series) -> 'VolatilityRegimeDetector':
        """
        Calcula percentis históricos de volatilidade.
        
        Args:
            volatility_series: Série com valores de volatilidade histórica
            
        Returns:
            self para encadeamento
        """
        if volatility_series.empty:
            raise ValueError("Volatility series cannot be empty")
            
        # Remove NaN
        vol_clean = volatility_series.dropna()
        
        if len(vol_clean) < self.lookback:
            # Usa todos os dados disponíveis se menor que lookback
            vol_data = vol_clean
        else:
            vol_data = vol_clean.tail(self.lookback)
            
        if len(vol_data) == 0:
            raise ValueError("No valid volatility data after removing NaN")
            
        self._low_threshold = float(np.percentile(vol_data, self.low_percentile))
        self._high_threshold = float(np.percentile(vol_data, self.high_percentile))
        self._extreme_threshold = float(np.percentile(vol_data, self.extreme_percentile))
        self._is_fitted = True
        
        return self
        
    def detect(self, current_vol: float) -> str:
        """
        Detecta regime de volatilidade.
        
        Args:
            current_vol: Volatilidade atual
            
        Returns:
            "low_vol", "normal_vol", "high_vol", ou "extreme_vol"
        """
        if not self._is_fitted:
            raise ValueError("Detector must be fitted before detecting regime")
            
        if current_vol is None:
            return "normal_vol"
            
        if current_vol <= self._low_threshold:
            return "low_vol"
        elif current_vol <= self._high_threshold:
            return "normal_vol"
        elif current_vol <= self._extreme_threshold:
            return "high_vol"
        else:
            return "extreme_vol"
            
    def detect_from_returns(
        self, 
        returns: pd.Series, 
        window: int = 20
    ) -> str:
        """
        Detecta regime calculando volatilidade a partir dos retornos.
        
        Args:
            returns: Série de retornos
            window: Janela para cálculo da volatilidade rolling
            
        Returns:
            Regime de volatilidade
        """
        if returns.empty:
            return "normal_vol"
            
        # Calcula volatilidade rolling e usa o último valor
        volatility = returns.rolling(window).std()
        current_vol = volatility.iloc[-1]
        
        if pd.isna(current_vol):
            return "normal_vol"
            
        return self.detect(float(current_vol))
        
    def get_regime_multiplier(self, regime: str) -> float:
        """
        Retorna multiplicador de posição baseado no regime.
        
        - low_vol: Pode aumentar posição
        - normal_vol: Tamanho normal
        - high_vol: Reduz posição
        - extreme_vol: Posição mínima
        
        Args:
            regime: Regime de volatilidade
            
        Returns:
            Multiplicador entre 0 e 1.5
        """
        multipliers = {
            'low_vol': 1.2,      # Pode ser mais agressivo
            'normal_vol': 1.0,   # Padrão
            'high_vol': 0.6,     # Reduz risco
            'extreme_vol': 0.3   # Risco mínimo
        }
        return multipliers.get(regime, 1.0)
        
    def get_thresholds(self) -> Dict[str, Optional[float]]:
        """
        Retorna os thresholds calculados.
        
        Returns:
            Dicionário com thresholds
        """
        return {
            'low': self._low_threshold,
            'high': self._high_threshold,
            'extreme': self._extreme_threshold
        }
        
    def get_regime_info(self, current_vol: float) -> Dict:
        """
        Retorna informações completas sobre o regime de volatilidade.
        
        Args:
            current_vol: Volatilidade atual
            
        Returns:
            Dicionário com informações do regime
        """
        if not self._is_fitted:
            return {
                'regime': 'unknown',
                'current_vol': current_vol,
                'is_fitted': False
            }
            
        regime = self.detect(current_vol)
        multiplier = self.get_regime_multiplier(regime)
        
        return {
            'regime': regime,
            'current_vol': current_vol,
            'multiplier': multiplier,
            'thresholds': self.get_thresholds(),
            'percentiles': {
                'low': self.low_percentile,
                'high': self.high_percentile,
                'extreme': self.extreme_percentile
            },
            'is_fitted': True
        }
        
    @property
    def is_fitted(self) -> bool:
        """Retorna se o detector foi ajustado."""
        return self._is_fitted
        
    def is_low_volatility(self, current_vol: float) -> bool:
        """Verifica se está em regime de baixa volatilidade."""
        return self.detect(current_vol) == "low_vol"
        
    def is_high_volatility(self, current_vol: float) -> bool:
        """Verifica se está em regime de alta volatilidade."""
        return self.detect(current_vol) in ["high_vol", "extreme_vol"]
