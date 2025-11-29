"""
Hidden Markov Model for Regime Detection.

Detects market regimes using HMM.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class HMMRegimeDetector:
    """
    Hidden Markov Model para detecção de regime.
    
    Parâmetros do config:
    - regime.hmm.n_regimes: 3  # OPTIMIZE: [2, 3, 4]
    - regime.hmm.features: ["returns", "volatility", "volume"]
    
    Regimes típicos:
    - 0: Low volatility / Ranging
    - 1: Trending up
    - 2: Trending down / High volatility
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o detector HMM com parâmetros do config.
        
        Args:
            config: Dicionário de configuração
        """
        hmm_config = config.get('regime', {}).get('hmm', {})
        self.n_regimes = hmm_config.get('n_regimes', 3)
        self.features = hmm_config.get('features', ['returns', 'volatility', 'volume'])
        self.covariance_type = hmm_config.get('covariance_type', 'full')
        self.n_iter = hmm_config.get('n_iter', 100)
        self.random_state = hmm_config.get('random_state', 42)
        self.model = None
        self._is_fitted = False
        self._regime_stats = {}
        
    def _create_model(self):
        """
        Cria instância do modelo HMM.
        
        Returns:
            GaussianHMM model
        """
        try:
            from hmmlearn.hmm import GaussianHMM
            return GaussianHMM(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state
            )
        except ImportError:
            raise ImportError("hmmlearn not installed. Run: pip install hmmlearn")
            
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepara features para o HMM.
        
        Args:
            data: DataFrame com dados de preço e volume
            
        Returns:
            Array de features normalizado
        """
        features = []
        
        for feature in self.features:
            if feature == 'returns':
                if 'close' in data.columns:
                    ret = data['close'].pct_change().fillna(0)
                    features.append(ret.values)
                elif 'returns' in data.columns:
                    features.append(data['returns'].fillna(0).values)
                    
            elif feature == 'volatility':
                if 'volatility' in data.columns:
                    features.append(data['volatility'].fillna(0).values)
                elif 'close' in data.columns:
                    # Calcula volatilidade rolling
                    vol = data['close'].pct_change().rolling(20).std().fillna(0)
                    features.append(vol.values)
                    
            elif feature == 'volume':
                if 'volume' in data.columns:
                    # Normaliza volume pelo seu desvio padrão
                    vol = data['volume'].fillna(0)
                    vol_normalized = (vol - vol.mean()) / (vol.std() + 1e-8)
                    features.append(vol_normalized.values)
                    
        if not features:
            raise ValueError(f"No valid features found. Required: {self.features}")
            
        return np.column_stack(features)
        
    def fit(self, data: pd.DataFrame) -> 'HMMRegimeDetector':
        """
        Treina HMM nos dados históricos.
        
        Args:
            data: DataFrame com dados de mercado
            
        Returns:
            self para encadeamento
        """
        if data.empty:
            raise ValueError("Data cannot be empty")
            
        X = self._prepare_features(data)
        
        # Remove NaN e infinitos
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
        X = X[valid_mask]
        
        if len(X) < self.n_regimes * 10:
            raise ValueError(f"Not enough valid samples. Need at least {self.n_regimes * 10}")
            
        self.model = self._create_model()
        self.model.fit(X)
        self._is_fitted = True
        
        # Calcula estatísticas por regime
        regimes = self.model.predict(X)
        self._compute_regime_stats(data.iloc[valid_mask], regimes)
        
        return self
        
    def _compute_regime_stats(self, data: pd.DataFrame, regimes: np.ndarray) -> None:
        """
        Calcula estatísticas para cada regime.
        
        Args:
            data: Dados originais
            regimes: Array de regimes previstos
        """
        self._regime_stats = {}
        
        if 'close' in data.columns:
            returns = data['close'].pct_change().values
        else:
            returns = np.zeros(len(data))
            
        for regime in range(self.n_regimes):
            mask = regimes == regime
            regime_returns = returns[mask]
            
            self._regime_stats[regime] = {
                'count': int(mask.sum()),
                'mean_return': float(np.nanmean(regime_returns)),
                'volatility': float(np.nanstd(regime_returns)),
                'avg_duration': self._compute_avg_duration(regimes, regime)
            }
            
    def _compute_avg_duration(self, regimes: np.ndarray, regime: int) -> float:
        """
        Calcula duração média de um regime.
        
        Args:
            regimes: Array de regimes
            regime: Regime específico
            
        Returns:
            Duração média em número de períodos
        """
        durations = []
        current_duration = 0
        
        for r in regimes:
            if r == regime:
                current_duration += 1
            elif current_duration > 0:
                durations.append(current_duration)
                current_duration = 0
                
        if current_duration > 0:
            durations.append(current_duration)
            
        return float(np.mean(durations)) if durations else 0.0
        
    def predict(self, data: pd.DataFrame) -> int:
        """
        Prediz regime atual.
        
        Args:
            data: DataFrame com dados recentes
            
        Returns:
            Regime predito (0 até n_regimes-1)
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before predicting")
            
        if data.empty:
            return 0
            
        X = self._prepare_features(data)
        
        # Remove NaN
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
        X = X[valid_mask]
        
        if len(X) == 0:
            return 0
            
        # Retorna o último regime
        regimes = self.model.predict(X)
        return int(regimes[-1])
        
    def predict_sequence(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prediz sequência de regimes.
        
        Args:
            data: DataFrame com dados
            
        Returns:
            Array de regimes
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before predicting")
            
        if data.empty:
            return np.array([])
            
        X = self._prepare_features(data)
        
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
        X = X[valid_mask]
        
        if len(X) == 0:
            return np.array([])
            
        return self.model.predict(X)
        
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """
        Retorna probabilidades de cada regime.
        
        Args:
            data: DataFrame com dados recentes
            
        Returns:
            Array de probabilidades [n_samples, n_regimes]
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before predicting")
            
        if data.empty:
            return np.array([[1.0 / self.n_regimes] * self.n_regimes])
            
        X = self._prepare_features(data)
        
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
        X = X[valid_mask]
        
        if len(X) == 0:
            return np.array([[1.0 / self.n_regimes] * self.n_regimes])
            
        _, posteriors = self.model.score_samples(X)
        return posteriors
        
    def get_current_proba(self, data: pd.DataFrame) -> Dict[int, float]:
        """
        Retorna probabilidades do regime atual.
        
        Args:
            data: DataFrame com dados recentes
            
        Returns:
            Dicionário {regime: probabilidade}
        """
        proba = self.predict_proba(data)
        return {i: float(proba[-1, i]) for i in range(self.n_regimes)}
        
    def get_regime_stats(self) -> Dict:
        """
        Retorna estatísticas de cada regime.
        
        Returns:
            Dicionário com estatísticas por regime:
            {regime: {count, mean_return, volatility, avg_duration}}
        """
        return self._regime_stats.copy()
        
    @property
    def is_fitted(self) -> bool:
        """Retorna se o modelo foi treinado."""
        return self._is_fitted
