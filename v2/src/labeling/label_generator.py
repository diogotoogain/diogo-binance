"""
Label Generator Module.

Orchestrates the generation of labels for training using Triple Barrier and Meta-Labeling.
"""

from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from v2.src.labeling.triple_barrier import TripleBarrierLabeler
from v2.src.labeling.meta_labeling import MetaLabeler


class LabelGenerator:
    """
    Orquestra geração de labels para treinamento.
    """
    
    def __init__(self, config: Dict):
        """
        Inicializa o gerador de labels.
        
        Args:
            config: Dicionário de configuração
        """
        self.config = config
        self.triple_barrier = TripleBarrierLabeler(config)
        
        meta_config = config.get('labeling', {}).get('meta_labeling', {})
        self.meta_labeler = MetaLabeler(config) if meta_config.get('enabled', True) else None
        
    def generate_labels(
        self, 
        data: pd.DataFrame, 
        events: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Gera labels completos usando Triple Barrier Method.
        
        Args:
            data: DataFrame com dados de preço (close, high, low, opcionalmente atr)
            events: DataFrame com índice de timestamps dos eventos
            
        Returns:
            DataFrame com colunas: [timestamp, label, tp, sl, max_bars]
        """
        if data.empty or events.empty:
            return pd.DataFrame(columns=['timestamp', 'label', 'tp', 'sl', 'max_bars'])
            
        # Gera labels usando triple barrier
        labels = self.triple_barrier.label(data, events)
        
        # Prepara DataFrame de resultado
        result = pd.DataFrame(index=events.index)
        result['timestamp'] = events.index
        result['label'] = labels
        
        # Adiciona informações das barreiras
        barrier_info = []
        for idx in events.index:
            if idx not in data.index:
                barrier_info.append({
                    'tp': np.nan, 
                    'sl': np.nan, 
                    'max_bars': self.triple_barrier.max_bars
                })
                continue
                
            entry_price = data.loc[idx, 'close']
            
            # Obtém ATR
            if 'atr' in data.columns:
                atr = data.loc[idx, 'atr']
            else:
                idx_loc = data.index.get_loc(idx)
                lookback = min(14, idx_loc)
                if lookback > 0:
                    atr = (data['high'].iloc[idx_loc-lookback:idx_loc+1] - 
                           data['low'].iloc[idx_loc-lookback:idx_loc+1]).mean()
                else:
                    atr = entry_price * 0.01
                    
            tp, sl, max_bars = self.triple_barrier.get_barriers(entry_price, atr)
            barrier_info.append({'tp': tp, 'sl': sl, 'max_bars': max_bars})
            
        barrier_df = pd.DataFrame(barrier_info, index=events.index)
        result = pd.concat([result, barrier_df], axis=1)
        
        return result
        
    def add_meta_labels(
        self, 
        labeled_data: pd.DataFrame, 
        features: pd.DataFrame,
        primary_predictions: pd.Series
    ) -> pd.DataFrame:
        """
        Adiciona meta-labels aos dados labelados.
        
        Args:
            labeled_data: DataFrame com labels do triple barrier
            features: Features usadas para treino
            primary_predictions: Predições do modelo primário
            
        Returns:
            DataFrame com coluna adicional 'meta_confidence'
        """
        if self.meta_labeler is None:
            labeled_data['meta_confidence'] = 1.0  # Sem meta-labeling
            return labeled_data
            
        if not self.meta_labeler.is_fitted:
            raise ValueError("Meta-labeler must be fitted before adding meta-labels")
            
        # Gera confidências
        confidences = self.meta_labeler.predict_confidence_batch(
            features, 
            primary_predictions
        )
        
        labeled_data['meta_confidence'] = confidences
        labeled_data['should_trade'] = confidences >= self.meta_labeler.threshold
        
        return labeled_data
        
    def fit_meta_labeler(
        self,
        features: pd.DataFrame,
        primary_predictions: pd.Series,
        actual_labels: pd.Series
    ) -> None:
        """
        Treina o meta-labeler.
        
        Args:
            features: Features de treino
            primary_predictions: Predições do modelo primário
            actual_labels: Labels reais do triple barrier
        """
        if self.meta_labeler is None:
            return
            
        self.meta_labeler.fit(features, primary_predictions, actual_labels)
        
    def get_training_data(
        self, 
        features: pd.DataFrame, 
        labels: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepara dados para treinamento.
        
        Args:
            features: DataFrame com features
            labels: DataFrame com labels (deve ter coluna 'label')
            test_size: Proporção para teste (padrão 0.2)
            random_state: Seed para reprodutibilidade
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        if features.empty or labels.empty:
            raise ValueError("Features and labels cannot be empty")
            
        # Alinha índices
        common_idx = features.index.intersection(labels.index)
        X = features.loc[common_idx]
        y = labels.loc[common_idx, 'label']
        
        # Remove NaN
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 10:
            raise ValueError("Not enough valid samples for training (minimum 10)")
            
        # Split temporal (sem shuffle para dados de séries temporais)
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
        
    def get_label_statistics(self, labels: pd.Series) -> Dict:
        """
        Retorna estatísticas sobre os labels gerados.
        
        Args:
            labels: Series com labels (-1, 0, 1)
            
        Returns:
            Dicionário com estatísticas
        """
        if labels.empty:
            return {
                'total': 0,
                'profitable': 0,
                'loss': 0,
                'timeout': 0,
                'profitable_pct': 0,
                'loss_pct': 0,
                'timeout_pct': 0
            }
            
        total = len(labels)
        profitable = (labels == 1).sum()
        loss = (labels == -1).sum()
        timeout = (labels == 0).sum()
        
        return {
            'total': total,
            'profitable': profitable,
            'loss': loss,
            'timeout': timeout,
            'profitable_pct': profitable / total * 100,
            'loss_pct': loss / total * 100,
            'timeout_pct': timeout / total * 100
        }
