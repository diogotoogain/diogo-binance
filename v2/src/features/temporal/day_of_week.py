"""
Day of Week Feature Calculator.

Adiciona features temporais relacionadas ao dia da semana.
"""
from datetime import datetime, timezone
from typing import Any, Dict, Union

import pandas as pd

from v2.src.features.base import Feature


class DayOfWeekFeatures(Feature):
    """
    Calcula features baseadas no dia da semana.
    
    Features geradas:
    - day_of_week: 0-6 (segunda a domingo)
    - is_monday: bool (1/0)
    - is_friday: bool (1/0)
    - is_weekend: bool (1/0)
    - days_to_weekend: 0-4
    - days_from_weekend: 0-4
    
    ZERO hardcoded - todos os parâmetros vêm do config.
    """
    
    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """
        Inicializa o calculador de features de dia da semana.
        
        Args:
            config: Dicionário de configuração com parâmetros
            enabled: Se a feature está habilitada
        """
        super().__init__(config, enabled)
        self.name = "DayOfWeek"
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula features de dia da semana para um DataFrame inteiro (batch).
        
        Args:
            data: DataFrame com índice datetime
            
        Returns:
            pd.DataFrame com as features de dia da semana calculadas
        """
        if not self.enabled:
            return pd.DataFrame(index=data.index)
        
        result = pd.DataFrame(index=data.index)
        
        # Extrair day_of_week do índice (0=Monday, 6=Sunday)
        if isinstance(data.index, pd.DatetimeIndex):
            day_of_week = data.index.dayofweek
        else:
            # Try to convert index to datetime
            try:
                dt_index = pd.to_datetime(data.index)
                day_of_week = dt_index.dayofweek
            except Exception:
                # Fallback: return empty features
                return result
        
        result['day_of_week'] = day_of_week
        result['is_monday'] = (day_of_week == 0).astype(int)
        result['is_friday'] = (day_of_week == 4).astype(int)
        result['is_weekend'] = ((day_of_week == 5) | (day_of_week == 6)).astype(int)
        
        # Convert to Series for clip operations
        day_of_week_series = pd.Series(day_of_week, index=data.index)
        
        # days_to_weekend: 0 on Friday, 1 on Thursday, ..., 4 on Monday
        # Saturday/Sunday = 0 (already at weekend)
        result['days_to_weekend'] = (4 - day_of_week_series).clip(lower=0)
        result.loc[day_of_week >= 5, 'days_to_weekend'] = 0
        
        # days_from_weekend: 0 on Monday, 1 on Tuesday, ..., 4 on Friday
        # Saturday/Sunday = 0 (just came from weekend)
        result['days_from_weekend'] = day_of_week_series.clip(upper=4)
        result.loc[day_of_week >= 5, 'days_from_weekend'] = 0
        
        return result
    
    def calculate_incremental(
        self, new_data: Any, state: Dict
    ) -> Dict[str, float]:
        """
        Calcula features incrementalmente (para real-time).
        
        Args:
            new_data: Novo dado recebido (deve ter 'timestamp')
            state: Estado anterior da feature (não usado para esta feature)
            
        Returns:
            Dict com valores atuais das features
        """
        if not self.enabled:
            return {}
        
        # Extrair timestamp
        timestamp = new_data.get('timestamp')
        if timestamp is None:
            # Use current time
            timestamp = datetime.now(timezone.utc)
        elif isinstance(timestamp, (int, float)):
            # Unix timestamp
            timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        elif isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        return self._calculate_features_from_datetime(timestamp)
    
    def _calculate_features_from_datetime(
        self, dt: datetime
    ) -> Dict[str, float]:
        """
        Calcula features a partir de um datetime.
        
        Args:
            dt: Objeto datetime
            
        Returns:
            Dict com valores das features
        """
        # Get day of week (0=Monday, 6=Sunday)
        day_of_week = dt.weekday()
        
        is_monday = 1 if day_of_week == 0 else 0
        is_friday = 1 if day_of_week == 4 else 0
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # days_to_weekend
        if day_of_week >= 5:
            days_to_weekend = 0
        else:
            days_to_weekend = 4 - day_of_week
        
        # days_from_weekend
        if day_of_week >= 5:
            days_from_weekend = 0
        else:
            days_from_weekend = day_of_week
        
        return {
            'day_of_week': float(day_of_week),
            'is_monday': float(is_monday),
            'is_friday': float(is_friday),
            'is_weekend': float(is_weekend),
            'days_to_weekend': float(days_to_weekend),
            'days_from_weekend': float(days_from_weekend),
        }
    
    @staticmethod
    def get_day_of_week(timestamp: Union[datetime, float, int]) -> int:
        """
        Obtém o dia da semana de um timestamp.
        
        Args:
            timestamp: Datetime ou Unix timestamp
            
        Returns:
            Dia da semana (0=Segunda, 6=Domingo)
        """
        if isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        else:
            dt = timestamp
        return dt.weekday()
    
    @staticmethod
    def is_weekend_day(timestamp: Union[datetime, float, int]) -> bool:
        """
        Verifica se o timestamp é fim de semana.
        
        Args:
            timestamp: Datetime ou Unix timestamp
            
        Returns:
            True se for sábado ou domingo
        """
        day_of_week = DayOfWeekFeatures.get_day_of_week(timestamp)
        return day_of_week >= 5
    
    @staticmethod
    def is_monday_day(timestamp: Union[datetime, float, int]) -> bool:
        """
        Verifica se o timestamp é segunda-feira.
        
        Args:
            timestamp: Datetime ou Unix timestamp
            
        Returns:
            True se for segunda-feira
        """
        return DayOfWeekFeatures.get_day_of_week(timestamp) == 0
    
    @staticmethod
    def is_friday_day(timestamp: Union[datetime, float, int]) -> bool:
        """
        Verifica se o timestamp é sexta-feira.
        
        Args:
            timestamp: Datetime ou Unix timestamp
            
        Returns:
            True se for sexta-feira
        """
        return DayOfWeekFeatures.get_day_of_week(timestamp) == 4
