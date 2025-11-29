"""
Seasonality Feature Calculator.

Adiciona features de sazonalidade (mês, quarter, período do mês).
"""
import calendar
from datetime import datetime
from typing import Any, Dict


class SeasonalityFeatures:
    """
    Calcula features de sazonalidade.
    
    Features geradas:
    - month: 1-12
    - quarter: 1-4
    - month_period: 'early', 'mid', 'late'
    - is_q4: bool (historicamente mais forte)
    - is_summer: bool (Jun-Ago, historicamente mais fraco)
    - days_to_month_end: int
    - days_from_month_start: int
    """
    
    # Month names for config lookup
    MONTH_NAMES = [
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december'
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SeasonalityFeatures.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config.get('seasonality', {})
        self.enabled = self.config.get('enabled', True)
        
        # Monthly adjustment config
        self.monthly_config = self.config.get('monthly_adjustment', {})
        self.monthly_enabled = self.monthly_config.get('enabled', True)
        
        # Month period config
        self.period_config = self.config.get('month_period', {})
        self.period_enabled = self.period_config.get('enabled', True)
        
        # Quarter config
        self.quarter_config = self.config.get('quarter_adjustment', {})
        self.quarter_enabled = self.quarter_config.get('enabled', True)
    
    def calculate(self, timestamp: datetime) -> Dict[str, float]:
        """
        Calcula features de sazonalidade.
        
        Args:
            timestamp: Datetime object for the analysis
            
        Returns:
            Dictionary with seasonality features
        """
        month = timestamp.month
        day = timestamp.day
        quarter = (month - 1) // 3 + 1
        
        # Get days in current month
        _, days_in_month = calendar.monthrange(timestamp.year, month)
        days_from_start = day
        days_to_end = days_in_month - day
        
        # Determine month period
        if day <= 10:
            month_period = 'early'
            month_period_numeric = 0.0
        elif day <= 20:
            month_period = 'mid'
            month_period_numeric = 0.5
        else:
            month_period = 'late'
            month_period_numeric = 1.0
        
        # Boolean flags
        is_q4 = quarter == 4  # Oct, Nov, Dec
        is_summer = month in [6, 7, 8]  # Jun, Jul, Aug
        is_q1 = quarter == 1
        is_q2 = quarter == 2
        is_q3 = quarter == 3
        
        return {
            'month': float(month),
            'quarter': float(quarter),
            'month_period': month_period,
            'month_period_numeric': month_period_numeric,
            'is_q4': 1.0 if is_q4 else 0.0,
            'is_summer': 1.0 if is_summer else 0.0,
            'is_q1': 1.0 if is_q1 else 0.0,
            'is_q2': 1.0 if is_q2 else 0.0,
            'is_q3': 1.0 if is_q3 else 0.0,
            'days_to_month_end': float(days_to_end),
            'days_from_month_start': float(days_from_start),
            'day_of_month': float(day),
        }
    
    def get_monthly_multiplier(self, month: int) -> float:
        """
        Get monthly position size multiplier.
        
        Args:
            month: Month number (1-12)
            
        Returns:
            Multiplier for position size
        """
        if not self.enabled or not self.monthly_enabled:
            return 1.0
        
        if month < 1 or month > 12:
            return 1.0
        
        month_name = self.MONTH_NAMES[month - 1]
        return self.monthly_config.get(month_name, 1.0)
    
    def get_period_multiplier(self, day: int) -> float:
        """
        Get month period position size multiplier.
        
        Args:
            day: Day of the month (1-31)
            
        Returns:
            Multiplier for position size
        """
        if not self.enabled or not self.period_enabled:
            return 1.0
        
        if day <= 10:
            return self.period_config.get('early_month_multiplier', 1.0)
        elif day <= 20:
            return self.period_config.get('mid_month_multiplier', 1.0)
        else:
            return self.period_config.get('late_month_multiplier', 0.9)
    
    def get_quarter_multiplier(self, month: int) -> float:
        """
        Get quarter position size multiplier.
        
        Args:
            month: Month number (1-12)
            
        Returns:
            Multiplier for position size
        """
        if not self.enabled or not self.quarter_enabled:
            return 1.0
        
        quarter = (month - 1) // 3 + 1
        
        quarter_keys = {
            1: 'q1_multiplier',
            2: 'q2_multiplier',
            3: 'q3_multiplier',
            4: 'q4_multiplier',
        }
        
        key = quarter_keys.get(quarter, 'q1_multiplier')
        return self.quarter_config.get(key, 1.0)
    
    def get_size_multiplier(self, timestamp: datetime) -> float:
        """
        Retorna o multiplier de position size baseado na sazonalidade.
        
        Combina:
        - Monthly multiplier
        - Month period multiplier
        - Quarter multiplier
        
        Args:
            timestamp: Datetime object for the analysis
            
        Returns:
            Combined multiplier for position size (product of all multipliers)
        """
        if not self.enabled:
            return 1.0
        
        month = timestamp.month
        day = timestamp.day
        
        monthly_mult = self.get_monthly_multiplier(month)
        period_mult = self.get_period_multiplier(day)
        quarter_mult = self.get_quarter_multiplier(month)
        
        # Combine multipliers (product)
        return monthly_mult * period_mult * quarter_mult
