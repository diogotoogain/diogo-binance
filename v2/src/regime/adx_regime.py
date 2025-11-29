"""
ADX-based Regime Detection.

Detects market regimes based on ADX (Average Directional Index).
"""

from typing import Dict


class ADXRegimeDetector:
    """
    Detecção de regime baseada em ADX.
    
    Parâmetros do config:
    - regime.adx_based.trending_threshold: 25   # OPTIMIZE: [20, 25, 30]
    - regime.adx_based.ranging_threshold: 20    # OPTIMIZE: [15, 18, 20]
    
    Regimes:
    - "trending": ADX > trending_threshold
    - "ranging": ADX < ranging_threshold
    - "transition": entre os dois
    """
    
    # Mapeamento de regimes para estratégias
    REGIME_STRATEGY_MAP = {
        'trending': {
            'momentum': True,
            'mean_reversion': False,
            'volatility_breakout': True,
            'scalping': False
        },
        'ranging': {
            'momentum': False,
            'mean_reversion': True,
            'volatility_breakout': False,
            'scalping': True
        },
        'transition': {
            'momentum': True,
            'mean_reversion': True,
            'volatility_breakout': True,
            'scalping': True
        }
    }
    
    def __init__(self, config: Dict):
        """
        Inicializa o detector ADX com parâmetros do config.
        
        Args:
            config: Dicionário de configuração
        """
        adx_config = config.get('regime', {}).get('adx_based', {})
        self.trending_threshold = adx_config.get('trending_threshold', 25)
        self.ranging_threshold = adx_config.get('ranging_threshold', 20)
        
        # Valida thresholds
        if self.ranging_threshold >= self.trending_threshold:
            raise ValueError(
                f"ranging_threshold ({self.ranging_threshold}) must be less than "
                f"trending_threshold ({self.trending_threshold})"
            )
        
    def detect(self, adx_value: float) -> str:
        """
        Detecta regime atual baseado no valor do ADX.
        
        Args:
            adx_value: Valor atual do ADX
            
        Returns:
            Regime: "trending", "ranging", ou "transition"
        """
        if adx_value is None:
            return "transition"
            
        if adx_value >= self.trending_threshold:
            return "trending"
        elif adx_value <= self.ranging_threshold:
            return "ranging"
        else:
            return "transition"
            
    def get_strategy_filter(self, regime: str) -> Dict[str, bool]:
        """
        Retorna quais estratégias devem estar ativas para cada regime.
        
        Args:
            regime: Regime atual ("trending", "ranging", "transition")
            
        Returns:
            Dicionário {"momentum": True/False, "mean_reversion": True/False, ...}
        """
        if regime not in self.REGIME_STRATEGY_MAP:
            return self.REGIME_STRATEGY_MAP['transition']
            
        return self.REGIME_STRATEGY_MAP[regime].copy()
        
    def get_regime_strength(self, adx_value: float) -> float:
        """
        Retorna força do regime atual (0 a 1).
        
        - Para trending: quão forte é a tendência
        - Para ranging: quão lateral está o mercado
        - Para transition: valor intermediário
        
        Args:
            adx_value: Valor atual do ADX
            
        Returns:
            Float entre 0 e 1
        """
        if adx_value is None:
            return 0.5
            
        regime = self.detect(adx_value)
        
        if regime == "trending":
            # Quanto mais alto o ADX, mais forte a tendência
            # Escala de trending_threshold até 50 (ADX raramente passa de 50)
            max_adx = 50
            return min(1.0, (adx_value - self.trending_threshold) / 
                      (max_adx - self.trending_threshold))
                      
        elif regime == "ranging":
            # Quanto mais baixo o ADX, mais lateral
            # Escala de 0 até ranging_threshold
            return 1.0 - (adx_value / self.ranging_threshold)
            
        else:  # transition
            # Valor intermediário
            range_size = self.trending_threshold - self.ranging_threshold
            position = adx_value - self.ranging_threshold
            return position / range_size
            
    def get_regime_info(self, adx_value: float) -> Dict:
        """
        Retorna informações completas sobre o regime atual.
        
        Args:
            adx_value: Valor atual do ADX
            
        Returns:
            Dicionário com informações do regime
        """
        regime = self.detect(adx_value)
        strength = self.get_regime_strength(adx_value)
        strategy_filter = self.get_strategy_filter(regime)
        
        return {
            'regime': regime,
            'adx_value': adx_value,
            'strength': strength,
            'thresholds': {
                'trending': self.trending_threshold,
                'ranging': self.ranging_threshold
            },
            'strategy_filter': strategy_filter
        }
        
    def is_trending(self, adx_value: float) -> bool:
        """Verifica se mercado está em tendência."""
        return self.detect(adx_value) == "trending"
        
    def is_ranging(self, adx_value: float) -> bool:
        """Verifica se mercado está lateral."""
        return self.detect(adx_value) == "ranging"
