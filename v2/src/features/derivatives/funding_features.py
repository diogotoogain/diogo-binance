"""
Funding Rate Feature Calculator.

Features baseadas em Funding Rate da Binance Futures.
"""
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from v2.src.features.base import Feature


class FundingRateFeatures(Feature):
    """
    Features baseadas em Funding Rate.
    
    Features geradas:
    - funding_rate_current: Taxa atual
    - funding_rate_avg_24h: Média 24h (3 períodos)
    - funding_rate_trend: Tendência (subindo/descendo)
    - funding_rate_extreme: bool (está em extremo)
    - funding_rate_zscore: Z-score vs histórico
    - time_to_funding: Minutos até próximo funding
    
    Todos os parâmetros vêm do config - ZERO hardcoded!
    """
    
    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """
        Inicializa o calculador de features de Funding Rate.
        
        Args:
            config: Configuração com parâmetros
            enabled: Se a feature está habilitada
        """
        super().__init__(config, enabled)
        
        # Parâmetros do config
        self.extreme_positive = config.get('extreme_positive', 0.001)
        self.extreme_negative = config.get('extreme_negative', -0.001)
        self.lookback_periods = config.get('lookback_periods', 8)
        self.include_in_features = config.get('include_in_features', True)
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula features de Funding Rate para DataFrame.
        
        Args:
            data: DataFrame com coluna 'funding_rate' e opcionalmente 
                  'funding_time', 'next_funding_time'
                  
        Returns:
            DataFrame com features calculadas
        """
        if not self.enabled:
            return pd.DataFrame(index=data.index)
        
        result = pd.DataFrame(index=data.index)
        
        # Verifica se há dados de funding rate
        if 'funding_rate' not in data.columns:
            # Retorna DataFrame vazio se não houver dados
            return result
        
        funding_rate = data['funding_rate'].astype(float)
        
        # Feature 1: Taxa atual
        result['funding_rate_current'] = funding_rate
        
        # Feature 2: Média móvel (3 períodos = 24h com funding a cada 8h)
        result['funding_rate_avg_24h'] = funding_rate.rolling(
            window=3,
            min_periods=1
        ).mean()
        
        # Feature 3: Média móvel baseada em lookback_periods
        result['funding_rate_avg'] = funding_rate.rolling(
            window=self.lookback_periods,
            min_periods=1
        ).mean()
        
        # Feature 4: Tendência (diferença entre atual e média)
        result['funding_rate_trend'] = funding_rate - result['funding_rate_avg']
        
        # Feature 5: Extremo (bool)
        result['funding_rate_extreme'] = (
            (funding_rate >= self.extreme_positive) |
            (funding_rate <= self.extreme_negative)
        ).astype(int)
        
        # Feature 6: Direção do extremo (-1 = muito short, 0 = neutro, 1 = muito long)
        result['funding_rate_extreme_direction'] = np.where(
            funding_rate >= self.extreme_positive, 1,
            np.where(funding_rate <= self.extreme_negative, -1, 0)
        )
        
        # Feature 7: Z-score vs histórico
        rolling_mean = funding_rate.rolling(
            window=self.lookback_periods,
            min_periods=2
        ).mean()
        rolling_std = funding_rate.rolling(
            window=self.lookback_periods,
            min_periods=2
        ).std()
        
        # Evita divisão por zero
        result['funding_rate_zscore'] = np.where(
            rolling_std > 0,
            (funding_rate - rolling_mean) / rolling_std,
            0.0
        )
        
        # Feature 8: Volatilidade do funding rate
        result['funding_rate_volatility'] = funding_rate.rolling(
            window=self.lookback_periods,
            min_periods=2
        ).std()
        
        # Feature 9: Mudança percentual
        result['funding_rate_change'] = funding_rate.diff()
        
        # Feature 10: Acumulado últimas 24h (3 períodos de 8h)
        result['funding_rate_cumsum_24h'] = funding_rate.rolling(
            window=3,
            min_periods=1
        ).sum()
        
        return result
    
    def calculate_incremental(
        self,
        new_data: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calcula features incrementalmente para real-time.
        
        Args:
            new_data: Dict com funding_rate atual
            state: Estado anterior
            
        Returns:
            Dict com valores atuais de todas as features
        """
        if not self.enabled:
            return {}
        
        # Inicializa estado se necessário
        if 'history' not in state:
            state['history'] = []
        
        funding_rate = float(new_data.get('funding_rate', 0.0))
        
        # Adiciona ao histórico
        state['history'].append(funding_rate)
        
        # Mantém apenas lookback_periods + 1 valores
        max_history = self.lookback_periods + 1
        if len(state['history']) > max_history:
            state['history'] = state['history'][-max_history:]
        
        history = state['history']
        
        # Calcula features
        result = {
            'funding_rate_current': funding_rate
        }
        
        # Média 24h (3 períodos)
        result['funding_rate_avg_24h'] = np.mean(history[-3:]) if history else 0.0
        
        # Média lookback
        result['funding_rate_avg'] = np.mean(history) if history else 0.0
        
        # Tendência
        result['funding_rate_trend'] = funding_rate - result['funding_rate_avg']
        
        # Extremo
        is_extreme = (
            funding_rate >= self.extreme_positive or
            funding_rate <= self.extreme_negative
        )
        result['funding_rate_extreme'] = 1 if is_extreme else 0
        
        # Direção do extremo
        if funding_rate >= self.extreme_positive:
            result['funding_rate_extreme_direction'] = 1
        elif funding_rate <= self.extreme_negative:
            result['funding_rate_extreme_direction'] = -1
        else:
            result['funding_rate_extreme_direction'] = 0
        
        # Z-score
        if len(history) >= 2:
            mean_val = np.mean(history)
            std_val = np.std(history, ddof=1)
            if std_val > 0:
                result['funding_rate_zscore'] = (funding_rate - mean_val) / std_val
            else:
                result['funding_rate_zscore'] = 0.0
        else:
            result['funding_rate_zscore'] = 0.0
        
        # Volatilidade
        if len(history) >= 2:
            result['funding_rate_volatility'] = float(np.std(history, ddof=1))
        else:
            result['funding_rate_volatility'] = 0.0
        
        # Mudança
        if len(history) >= 2:
            result['funding_rate_change'] = history[-1] - history[-2]
        else:
            result['funding_rate_change'] = 0.0
        
        # Acumulado 24h
        result['funding_rate_cumsum_24h'] = sum(history[-3:])
        
        return result
    
    def is_extreme_positive(self, funding_rate: float) -> bool:
        """
        Verifica se o funding rate está em extremo positivo (muito bullish).
        
        Args:
            funding_rate: Taxa de funding atual
            
        Returns:
            True se está em extremo positivo
        """
        return funding_rate >= self.extreme_positive
    
    def is_extreme_negative(self, funding_rate: float) -> bool:
        """
        Verifica se o funding rate está em extremo negativo (muito bearish).
        
        Args:
            funding_rate: Taxa de funding atual
            
        Returns:
            True se está em extremo negativo
        """
        return funding_rate <= self.extreme_negative
    
    def is_extreme(self, funding_rate: float) -> bool:
        """
        Verifica se o funding rate está em qualquer extremo.
        
        Args:
            funding_rate: Taxa de funding atual
            
        Returns:
            True se está em extremo
        """
        return self.is_extreme_positive(funding_rate) or self.is_extreme_negative(funding_rate)
    
    def get_signal(self, funding_rate: float) -> int:
        """
        Retorna sinal baseado no funding rate.
        
        Lógica:
        - Funding muito positivo = mercado muito long = sinal short (-1)
        - Funding muito negativo = mercado muito short = sinal long (1)
        - Funding neutro = sem sinal (0)
        
        Args:
            funding_rate: Taxa de funding atual
            
        Returns:
            -1 (short), 0 (neutro), 1 (long)
        """
        if self.is_extreme_positive(funding_rate):
            return -1  # Muito longs, pode haver correção
        elif self.is_extreme_negative(funding_rate):
            return 1  # Muito shorts, pode haver squeeze
        return 0
    
    def get_feature_names(self) -> List[str]:
        """
        Retorna nomes de todas as features geradas.
        
        Returns:
            Lista de nomes das features
        """
        return [
            'funding_rate_current',
            'funding_rate_avg_24h',
            'funding_rate_avg',
            'funding_rate_trend',
            'funding_rate_extreme',
            'funding_rate_extreme_direction',
            'funding_rate_zscore',
            'funding_rate_volatility',
            'funding_rate_change',
            'funding_rate_cumsum_24h'
        ]
