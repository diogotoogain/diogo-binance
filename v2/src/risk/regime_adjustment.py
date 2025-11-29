"""
Regime Adjustment - Ajusta risco baseado no regime de mercado.

Reduz exposição em regimes perigosos como crashes e alta volatilidade.
"""
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class RegimeAdjustment:
    """
    Ajusta risco baseado no regime de mercado.
    
    Parâmetros do config (risk.regime_adjustment):
    - enabled: true
    - crash_multiplier: 0.3           # OPTIMIZE: [0.1, 0.2, 0.3, 0.5]
    - high_vol_multiplier: 0.5        # OPTIMIZE: [0.3, 0.5, 0.7]
    """
    
    def __init__(self, config: Dict):
        """
        Initialize regime adjustment.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config['risk']['regime_adjustment']
        self.enabled = self.config['enabled']
        
    def get_multiplier(self, regime: Dict) -> float:
        """
        Retorna multiplicador de risco baseado no regime.
        
        Em regimes perigosos, reduz exposição significativamente.
        
        Args:
            regime: Dictionary com informações do regime atual
                - volatility: 'low', 'normal', 'high', 'extreme'
                - hmm: 0, 1, 2 (regime from HMM model)
                - trend: 'bullish', 'bearish', 'neutral'
                
        Returns:
            Multiplicador de risco [0, 1]. 1.0 = risco normal.
        """
        if not self.enabled:
            return 1.0
            
        volatility_regime = regime.get('volatility', 'normal')
        hmm_regime = regime.get('hmm', 1)
        
        # Crash/extreme volatility - máxima redução
        if volatility_regime == 'extreme' or hmm_regime == 2:
            mult = self.config['crash_multiplier']
            logger.info(f"⚠️ Regime extremo detectado! Multiplicador: {mult}")
            return mult
            
        # High volatility - redução moderada
        if volatility_regime == 'high':
            mult = self.config['high_vol_multiplier']
            logger.debug(f"📊 Alta volatilidade detectada. Multiplicador: {mult}")
            return mult
            
        # Low volatility - pode aumentar ligeiramente (opcional)
        if volatility_regime == 'low':
            # Opcionalmente poderia aumentar o multiplier em low vol
            # Por segurança, mantemos 1.0
            return 1.0
            
        # Normal
        return 1.0
        
    def get_regime_risk_profile(self, regime: Dict) -> Dict:
        """
        Retorna perfil de risco completo para o regime.
        
        Args:
            regime: Dictionary com informações do regime
            
        Returns:
            Dictionary com ajustes de risco recomendados
        """
        multiplier = self.get_multiplier(regime)
        volatility = regime.get('volatility', 'normal')
        hmm = regime.get('hmm', 1)
        
        return {
            'multiplier': multiplier,
            'volatility_regime': volatility,
            'hmm_regime': hmm,
            'recommendation': self._get_recommendation(volatility, hmm),
            'allow_new_positions': multiplier > 0.0,
            'reduce_existing': multiplier < 0.5,
        }
        
    def _get_recommendation(self, volatility: str, hmm: int) -> str:
        """
        Get trading recommendation based on regime.
        
        Args:
            volatility: Volatility regime
            hmm: HMM regime state
            
        Returns:
            Trading recommendation string
        """
        if volatility == 'extreme' or hmm == 2:
            return "🚨 REGIME PERIGOSO - Reduzir exposição drasticamente"
        elif volatility == 'high':
            return "⚠️ Alta volatilidade - Reduzir tamanho de posições"
        elif volatility == 'low':
            return "📉 Baixa volatilidade - Cuidado com breakouts"
        else:
            return "✅ Regime normal - Trading regular"
            
    def should_reduce_position(self, regime: Dict, current_exposure: float,
                                max_exposure: float) -> float:
        """
        Calcula quanto reduzir a exposição atual.
        
        Args:
            regime: Dictionary com informações do regime
            current_exposure: Exposição atual em USD
            max_exposure: Exposição máxima permitida
            
        Returns:
            Nova exposição recomendada em USD
        """
        multiplier = self.get_multiplier(regime)
        
        # Calcula nova exposição máxima ajustada
        adjusted_max = max_exposure * multiplier
        
        # Se exposição atual excede o ajustado, recomendar redução
        if current_exposure > adjusted_max:
            logger.warning(
                f"📉 Exposição atual ({current_exposure:.2f}) excede "
                f"limite ajustado ({adjusted_max:.2f}). Reduzir!"
            )
            return adjusted_max
            
        return current_exposure
        
    def get_adjusted_parameters(self, base_params: Dict, regime: Dict) -> Dict:
        """
        Ajusta parâmetros de risco baseado no regime.
        
        Args:
            base_params: Parâmetros base de risco
            regime: Dictionary com informações do regime
            
        Returns:
            Parâmetros ajustados
        """
        multiplier = self.get_multiplier(regime)
        
        adjusted = base_params.copy()
        
        # Ajusta parâmetros que devem diminuir em regimes perigosos
        if 'position_size' in adjusted:
            adjusted['position_size'] *= multiplier
        if 'max_exposure' in adjusted:
            adjusted['max_exposure'] *= multiplier
        if 'risk_per_trade' in adjusted:
            adjusted['risk_per_trade'] *= multiplier
            
        # Ajusta parâmetros que devem aumentar em regimes perigosos
        # (stops mais apertados)
        if multiplier < 1.0:
            if 'stop_loss_pct' in adjusted:
                # Aumenta SL em regimes voláteis para não ser stopado por ruído
                adjusted['stop_loss_pct'] *= (1 / multiplier) ** 0.5
                
        return adjusted
