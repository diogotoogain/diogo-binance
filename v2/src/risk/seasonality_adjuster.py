"""
Seasonality Position Size Adjuster.

Ajusta position size baseado em padrões sazonais históricos.
"""
import logging
from datetime import datetime
from typing import Any, Dict

from v2.src.features.temporal.seasonality import SeasonalityFeatures

logger = logging.getLogger(__name__)


class SeasonalityAdjuster:
    """
    Ajusta risco baseado em sazonalidade.
    
    Funcionalidades:
    - Reduz exposição em meses historicamente ruins
    - Aumenta exposição em meses historicamente bons
    - Considera período do mês (início/meio/fim)
    - Considera quarter
    
    Parâmetros do config (seasonality):
    - enabled: true                      # OPTIMIZE: [true, false]
    - monthly_adjustment.enabled: true   # OPTIMIZE: [true, false]
    - monthly_adjustment.january: 1.0    # OPTIMIZE: range(0.5, 1.5)
    - ... (todos os meses)
    - month_period.enabled: true         # OPTIMIZE: [true, false]
    - month_period.early_month_multiplier: 1.0   # OPTIMIZE: range(0.7, 1.3)
    - month_period.mid_month_multiplier: 1.0     # OPTIMIZE: range(0.7, 1.3)
    - month_period.late_month_multiplier: 0.9    # OPTIMIZE: range(0.7, 1.3)
    - quarter_adjustment.enabled: true   # OPTIMIZE: [true, false]
    - quarter_adjustment.q1_multiplier: 1.0      # OPTIMIZE: range(0.7, 1.3)
    - quarter_adjustment.q2_multiplier: 0.9      # OPTIMIZE: range(0.7, 1.3)
    - quarter_adjustment.q3_multiplier: 0.8      # OPTIMIZE: range(0.7, 1.3)
    - quarter_adjustment.q4_multiplier: 1.2      # OPTIMIZE: range(0.7, 1.3)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SeasonalityAdjuster.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config.get('seasonality', {})
        self.enabled = self.config.get('enabled', True)
        self.seasonality_features = SeasonalityFeatures(config)
        
        logger.info(
            f"🗓️ SeasonalityAdjuster initialized: enabled={self.enabled}"
        )
    
    def is_enabled(self) -> bool:
        """Check if seasonality adjustment is enabled."""
        return self.enabled
    
    def get_multiplier(self, timestamp: datetime) -> float:
        """
        Get the combined seasonality multiplier for position sizing.
        
        Args:
            timestamp: Current datetime
            
        Returns:
            Multiplier to apply to position size
        """
        if not self.enabled:
            return 1.0
        
        multiplier = self.seasonality_features.get_size_multiplier(timestamp)
        
        logger.debug(
            f"Seasonality multiplier for {timestamp.strftime('%Y-%m-%d')}: {multiplier:.3f}"
        )
        
        return multiplier
    
    def adjust_position_size(
        self,
        base_size: float,
        timestamp: datetime
    ) -> float:
        """
        Ajusta position size baseado em sazonalidade.
        
        Args:
            base_size: Base position size before adjustment
            timestamp: Current datetime
            
        Returns:
            Adjusted position size
        """
        if not self.enabled:
            return base_size
        
        multiplier = self.get_multiplier(timestamp)
        adjusted_size = base_size * multiplier
        
        logger.debug(
            f"Position size adjusted: {base_size:.6f} -> {adjusted_size:.6f} "
            f"(multiplier: {multiplier:.3f})"
        )
        
        return adjusted_size
    
    def get_seasonality_info(self, timestamp: datetime) -> Dict[str, Any]:
        """
        Get detailed seasonality information for the given timestamp.
        
        Args:
            timestamp: Current datetime
            
        Returns:
            Dictionary with seasonality details
        """
        features = self.seasonality_features.calculate(timestamp)
        multiplier = self.get_multiplier(timestamp)
        
        return {
            'enabled': self.enabled,
            'timestamp': timestamp.isoformat(),
            'month': int(features['month']),
            'quarter': int(features['quarter']),
            'month_period': features['month_period'],
            'is_q4': bool(features['is_q4']),
            'is_summer': bool(features['is_summer']),
            'combined_multiplier': multiplier,
            'monthly_multiplier': self.seasonality_features.get_monthly_multiplier(
                int(features['month'])
            ),
            'period_multiplier': self.seasonality_features.get_period_multiplier(
                int(features['day_of_month'])
            ),
            'quarter_multiplier': self.seasonality_features.get_quarter_multiplier(
                int(features['month'])
            ),
        }
