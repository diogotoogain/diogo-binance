"""
Open Interest Feature Calculator.

Features baseadas em Open Interest da Binance Futures.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from v2.src.features.base import Feature


class OpenInterestFeatures(Feature):
    """
    Features baseadas em Open Interest.
    
    Features geradas:
    - oi_current: OI atual
    - oi_change_pct: Mudança % vs período anterior
    - oi_trend: Tendência de OI
    - oi_price_divergence: Divergência OI vs Preço (-1, 0, 1)
    - oi_zscore: Z-score vs histórico
    
    Todos os parâmetros vêm do config - ZERO hardcoded!
    """
    
    def __init__(self, config: Dict[str, Any], enabled: bool = True):
        """
        Inicializa o calculador de features de Open Interest.
        
        Args:
            config: Configuração com parâmetros
            enabled: Se a feature está habilitada
        """
        super().__init__(config, enabled)
        
        # Parâmetros do config
        self.significant_change_pct = config.get('significant_change_pct', 5.0)
        self.include_in_features = config.get('include_in_features', True)
        
        # Parâmetros de divergência
        divergence_config = config.get('divergence_detection', {})
        self.divergence_enabled = divergence_config.get('enabled', True)
        self.divergence_lookback = divergence_config.get('lookback_bars', 20)
        self.price_change_threshold = divergence_config.get('price_change_threshold', 0.02)
        self.oi_change_threshold = divergence_config.get('oi_change_threshold', 0.03)
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula features de Open Interest para DataFrame.
        
        Args:
            data: DataFrame com colunas 'open_interest' e 'close'
                  
        Returns:
            DataFrame com features calculadas
        """
        if not self.enabled:
            return pd.DataFrame(index=data.index)
        
        result = pd.DataFrame(index=data.index)
        
        # Verifica se há dados de open interest
        if 'open_interest' not in data.columns:
            return result
        
        oi = data['open_interest'].astype(float)
        
        # Feature 1: OI atual
        result['oi_current'] = oi
        
        # Feature 2: Mudança percentual
        result['oi_change_pct'] = oi.pct_change() * 100
        
        # Feature 3: Mudança significativa (bool)
        result['oi_significant_change'] = (
            np.abs(result['oi_change_pct']) >= self.significant_change_pct
        ).astype(int)
        
        # Feature 4: Média móvel
        result['oi_sma'] = oi.rolling(
            window=self.divergence_lookback,
            min_periods=1
        ).mean()
        
        # Feature 5: Tendência (OI acima ou abaixo da média)
        result['oi_trend'] = np.where(
            oi > result['oi_sma'], 1,
            np.where(oi < result['oi_sma'], -1, 0)
        )
        
        # Feature 6: Z-score
        rolling_mean = oi.rolling(
            window=self.divergence_lookback,
            min_periods=2
        ).mean()
        rolling_std = oi.rolling(
            window=self.divergence_lookback,
            min_periods=2
        ).std()
        
        result['oi_zscore'] = np.where(
            rolling_std > 0,
            (oi - rolling_mean) / rolling_std,
            0.0
        )
        
        # Feature 7: Volatilidade de OI
        result['oi_volatility'] = oi.pct_change().rolling(
            window=self.divergence_lookback,
            min_periods=2
        ).std() * 100
        
        # Feature 8: Taxa de crescimento
        result['oi_growth_rate'] = oi.pct_change(periods=self.divergence_lookback) * 100
        
        # Feature 9: Divergência OI vs Preço
        if self.divergence_enabled and 'close' in data.columns:
            result['oi_price_divergence'] = self._calculate_divergence(
                oi,
                data['close'].astype(float)
            )
        else:
            result['oi_price_divergence'] = 0
        
        # Feature 10: Normalizado (0-1 baseado em histórico)
        oi_min = oi.rolling(window=self.divergence_lookback, min_periods=1).min()
        oi_max = oi.rolling(window=self.divergence_lookback, min_periods=1).max()
        oi_range = oi_max - oi_min
        
        result['oi_normalized'] = np.where(
            oi_range > 0,
            (oi - oi_min) / oi_range,
            0.5
        )
        
        return result
    
    def _calculate_divergence(
        self,
        oi: pd.Series,
        price: pd.Series
    ) -> pd.Series:
        """
        Calcula divergência entre OI e preço.
        
        Lógica:
        - Preço sobe + OI cai = rally falso (bearish) = -1
        - Preço cai + OI sobe = capitulação/acumulação = 1
        - Preço e OI movem juntos = confirmação = 0
        
        Args:
            oi: Série de Open Interest
            price: Série de preços
            
        Returns:
            Série com valores de divergência (-1, 0, 1)
        """
        # Calcula mudanças percentuais
        price_change = price.pct_change(periods=self.divergence_lookback)
        oi_change = oi.pct_change(periods=self.divergence_lookback)
        
        # Detecta divergências
        divergence = pd.Series(0, index=oi.index)
        
        # Preço sobe + OI cai = bearish divergence
        bearish_div = (
            (price_change > self.price_change_threshold) &
            (oi_change < -self.oi_change_threshold)
        )
        divergence = divergence.where(~bearish_div, -1)
        
        # Preço cai + OI sobe = bullish divergence
        bullish_div = (
            (price_change < -self.price_change_threshold) &
            (oi_change > self.oi_change_threshold)
        )
        divergence = divergence.where(~bullish_div, 1)
        
        return divergence
    
    def calculate_incremental(
        self,
        new_data: Dict[str, Any],
        state: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calcula features incrementalmente para real-time.
        
        Args:
            new_data: Dict com open_interest e close atuais
            state: Estado anterior
            
        Returns:
            Dict com valores atuais de todas as features
        """
        if not self.enabled:
            return {}
        
        # Inicializa estado se necessário
        if 'oi_history' not in state:
            state['oi_history'] = []
        if 'price_history' not in state:
            state['price_history'] = []
        
        oi = float(new_data.get('open_interest', 0.0))
        price = float(new_data.get('close', new_data.get('price', 0.0)))
        
        # Adiciona ao histórico
        state['oi_history'].append(oi)
        state['price_history'].append(price)
        
        # Mantém apenas lookback + 1 valores
        max_history = self.divergence_lookback + 1
        if len(state['oi_history']) > max_history:
            state['oi_history'] = state['oi_history'][-max_history:]
            state['price_history'] = state['price_history'][-max_history:]
        
        oi_history = state['oi_history']
        price_history = state['price_history']
        
        # Calcula features
        result = {
            'oi_current': oi
        }
        
        # Mudança percentual
        if len(oi_history) >= 2 and oi_history[-2] != 0:
            result['oi_change_pct'] = (
                (oi - oi_history[-2]) / oi_history[-2] * 100
            )
        else:
            result['oi_change_pct'] = 0.0
        
        # Mudança significativa
        result['oi_significant_change'] = (
            1 if abs(result['oi_change_pct']) >= self.significant_change_pct else 0
        )
        
        # Média móvel
        result['oi_sma'] = float(np.mean(oi_history)) if oi_history else oi
        
        # Tendência
        if oi > result['oi_sma']:
            result['oi_trend'] = 1
        elif oi < result['oi_sma']:
            result['oi_trend'] = -1
        else:
            result['oi_trend'] = 0
        
        # Z-score
        if len(oi_history) >= 2:
            mean_oi = np.mean(oi_history)
            std_oi = np.std(oi_history, ddof=1)
            if std_oi > 0:
                result['oi_zscore'] = (oi - mean_oi) / std_oi
            else:
                result['oi_zscore'] = 0.0
        else:
            result['oi_zscore'] = 0.0
        
        # Volatilidade
        if len(oi_history) >= 3:
            changes = [
                (oi_history[i] - oi_history[i-1]) / oi_history[i-1] * 100
                for i in range(1, len(oi_history))
                if oi_history[i-1] != 0
            ]
            result['oi_volatility'] = float(np.std(changes, ddof=1)) if changes else 0.0
        else:
            result['oi_volatility'] = 0.0
        
        # Taxa de crescimento
        if len(oi_history) >= self.divergence_lookback and oi_history[0] != 0:
            result['oi_growth_rate'] = (
                (oi - oi_history[0]) / oi_history[0] * 100
            )
        else:
            result['oi_growth_rate'] = 0.0
        
        # Divergência
        result['oi_price_divergence'] = self._calculate_divergence_incremental(
            oi_history,
            price_history
        )
        
        # Normalizado
        if len(oi_history) >= 2:
            min_oi = min(oi_history)
            max_oi = max(oi_history)
            oi_range = max_oi - min_oi
            if oi_range > 0:
                result['oi_normalized'] = (oi - min_oi) / oi_range
            else:
                result['oi_normalized'] = 0.5
        else:
            result['oi_normalized'] = 0.5
        
        return result
    
    def _calculate_divergence_incremental(
        self,
        oi_history: List[float],
        price_history: List[float]
    ) -> int:
        """
        Calcula divergência incrementalmente.
        
        Args:
            oi_history: Histórico de OI
            price_history: Histórico de preços
            
        Returns:
            -1 (bearish), 0 (neutro), 1 (bullish)
        """
        if not self.divergence_enabled:
            return 0
        
        if len(oi_history) < self.divergence_lookback or len(price_history) < self.divergence_lookback:
            return 0
        
        # Pega valores inicial e final do lookback
        oi_start = oi_history[-self.divergence_lookback]
        oi_end = oi_history[-1]
        price_start = price_history[-self.divergence_lookback]
        price_end = price_history[-1]
        
        # Calcula mudanças
        if oi_start == 0 or price_start == 0:
            return 0
        
        oi_change = (oi_end - oi_start) / oi_start
        price_change = (price_end - price_start) / price_start
        
        # Detecta divergências
        # Preço sobe + OI cai = bearish
        if (price_change > self.price_change_threshold and
            oi_change < -self.oi_change_threshold):
            return -1
        
        # Preço cai + OI sobe = bullish
        if (price_change < -self.price_change_threshold and
            oi_change > self.oi_change_threshold):
            return 1
        
        return 0
    
    def detect_divergence(
        self,
        price_change_pct: float,
        oi_change_pct: float
    ) -> Tuple[bool, int, str]:
        """
        Detecta divergência entre preço e OI.
        
        Args:
            price_change_pct: Mudança percentual do preço
            oi_change_pct: Mudança percentual do OI
            
        Returns:
            Tuple (has_divergence, direction, description)
        """
        price_threshold = self.price_change_threshold * 100  # Converte para percentual
        oi_threshold = self.oi_change_threshold * 100
        
        # Preço sobe + OI cai = rally falso (bearish)
        if price_change_pct > price_threshold and oi_change_pct < -oi_threshold:
            return (True, -1, "Bearish divergence: Price up but OI down - potential false rally")
        
        # Preço cai + OI sobe = acumulação (bullish)
        if price_change_pct < -price_threshold and oi_change_pct > oi_threshold:
            return (True, 1, "Bullish divergence: Price down but OI up - potential accumulation")
        
        return (False, 0, "No divergence detected")
    
    def get_signal(
        self,
        oi_change_pct: float,
        price_change_pct: Optional[float] = None
    ) -> int:
        """
        Retorna sinal baseado em OI e opcionalmente preço.
        
        Args:
            oi_change_pct: Mudança percentual do OI
            price_change_pct: Mudança percentual do preço (opcional)
            
        Returns:
            -1 (bearish), 0 (neutro), 1 (bullish)
        """
        # Se temos mudança de preço, verifica divergência
        if price_change_pct is not None:
            has_div, direction, _ = self.detect_divergence(
                price_change_pct,
                oi_change_pct
            )
            if has_div:
                return direction
        
        # Se OI está crescendo significativamente, pode indicar força
        if oi_change_pct > self.significant_change_pct:
            return 1  # Força compradora/vendedora entrando
        elif oi_change_pct < -self.significant_change_pct:
            return -1  # Posições sendo fechadas, possível exaustão
        
        return 0
    
    def get_feature_names(self) -> List[str]:
        """
        Retorna nomes de todas as features geradas.
        
        Returns:
            Lista de nomes das features
        """
        return [
            'oi_current',
            'oi_change_pct',
            'oi_significant_change',
            'oi_sma',
            'oi_trend',
            'oi_zscore',
            'oi_volatility',
            'oi_growth_rate',
            'oi_price_divergence',
            'oi_normalized'
        ]
