"""
Meta-Labeling Module.

Meta-Labeling: secondary model that learns WHEN to trust the primary model.
"""

from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np


class MetaLabeler:
    """
    Meta-Labeling: modelo secundário que aprende QUANDO confiar no modelo primário.
    
    Parâmetros do config:
    - labeling.meta_labeling.model_type: "xgboost"  # TOGGLE: [xgboost, lightgbm, random_forest]
    - labeling.meta_labeling.confidence_threshold: 0.6  # OPTIMIZE: [0.5, 0.55, 0.6, 0.65, 0.7]
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o meta-labeler com parâmetros do config.
        
        Args:
            config: Dicionário de configuração
        """
        meta_config = config.get('labeling', {}).get('meta_labeling', {})
        self.model_type = meta_config.get('model_type', 'xgboost')
        self.threshold = meta_config.get('confidence_threshold', 0.6)
        self.model = None
        self._is_fitted = False
        
    def _create_model(self):
        """
        Cria o modelo baseado no tipo configurado.
        
        Returns:
            Instância do modelo
        """
        if self.model_type == 'xgboost':
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    use_label_encoder=False,
                    eval_metric='logloss',
                    random_state=42
                )
            except ImportError:
                raise ImportError("xgboost not installed. Run: pip install xgboost")
                
        elif self.model_type == 'lightgbm':
            try:
                from lightgbm import LGBMClassifier
                return LGBMClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                )
            except ImportError:
                raise ImportError("lightgbm not installed. Run: pip install lightgbm")
                
        elif self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. "
                           f"Supported types: xgboost, lightgbm, random_forest")
        
    def fit(
        self, 
        X: pd.DataFrame, 
        primary_predictions: pd.Series, 
        actual_labels: pd.Series
    ) -> 'MetaLabeler':
        """
        Treina meta-modelo.
        
        O meta-modelo aprende quando o modelo primário acerta ou erra.
        
        Args:
            X: Features de treino
            primary_predictions: Predições do modelo primário (-1, 0, 1)
            actual_labels: Labels reais (-1, 0, 1)
            
        Returns:
            self para encadeamento
        """
        if X.empty:
            raise ValueError("X cannot be empty")
            
        if len(X) != len(primary_predictions) or len(X) != len(actual_labels):
            raise ValueError("X, primary_predictions, and actual_labels must have same length")
            
        # Meta-label: 1 se modelo primário acertou a direção, 0 caso contrário
        # Considera acerto quando:
        # - primary == actual (mesma direção)
        # - Ou quando primary != 0 e actual tem mesmo sinal
        meta_labels = (
            (primary_predictions == actual_labels) |
            ((primary_predictions != 0) & (np.sign(primary_predictions) == np.sign(actual_labels)))
        ).astype(int)
        
        # Adiciona predição primária como feature
        X_meta = X.copy()
        X_meta['primary_prediction'] = primary_predictions.values
        
        # Remove linhas com NaN
        valid_mask = ~(X_meta.isna().any(axis=1) | meta_labels.isna())
        X_meta = X_meta[valid_mask]
        meta_labels = meta_labels[valid_mask]
        
        if len(X_meta) == 0:
            raise ValueError("No valid samples after removing NaN values")
            
        self.model = self._create_model()
        self.model.fit(X_meta, meta_labels)
        self._is_fitted = True
        self._feature_names = list(X_meta.columns)
        
        return self
        
    def predict_confidence(
        self, 
        X: pd.DataFrame, 
        primary_prediction: int
    ) -> float:
        """
        Retorna confiança na predição do modelo primário.
        
        Args:
            X: Features para predição (single row)
            primary_prediction: Predição do modelo primário
            
        Returns:
            float entre 0 e 1 indicando confiança
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before predicting")
            
        if X.empty:
            return 0.0
            
        # Prepara features
        X_meta = X.copy()
        if isinstance(X_meta, pd.Series):
            X_meta = X_meta.to_frame().T
        X_meta['primary_prediction'] = primary_prediction
        
        # Garante ordem das features
        X_meta = X_meta.reindex(columns=self._feature_names, fill_value=0)
        
        # Obtém probabilidade da classe positiva (modelo primário acerta)
        proba = self.model.predict_proba(X_meta)
        
        # Retorna probabilidade de acerto (classe 1)
        if proba.shape[1] == 2:
            return float(proba[0, 1])
        else:
            return float(proba[0, 0])
            
    def predict_confidence_batch(
        self, 
        X: pd.DataFrame, 
        primary_predictions: pd.Series
    ) -> pd.Series:
        """
        Retorna confiança para múltiplas predições.
        
        Args:
            X: DataFrame com features
            primary_predictions: Series com predições do modelo primário
            
        Returns:
            Series com confiança para cada predição
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before predicting")
            
        if X.empty:
            return pd.Series(dtype=float)
            
        X_meta = X.copy()
        X_meta['primary_prediction'] = primary_predictions.values
        X_meta = X_meta.reindex(columns=self._feature_names, fill_value=0)
        
        proba = self.model.predict_proba(X_meta)
        
        if proba.shape[1] == 2:
            return pd.Series(proba[:, 1], index=X.index)
        else:
            return pd.Series(proba[:, 0], index=X.index)
        
    def should_trade(self, confidence: float) -> bool:
        """
        Decide se deve executar o trade baseado na confiança.
        
        Args:
            confidence: Nível de confiança (0-1)
            
        Returns:
            True se confiança >= threshold, False caso contrário
        """
        return confidence >= self.threshold
        
    def get_threshold(self) -> float:
        """Retorna threshold atual."""
        return self.threshold
        
    def set_threshold(self, threshold: float) -> None:
        """
        Define novo threshold.
        
        Args:
            threshold: Novo valor (deve estar entre 0 e 1)
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        self.threshold = threshold
        
    @property
    def is_fitted(self) -> bool:
        """Retorna se o modelo foi treinado."""
        return self._is_fitted
