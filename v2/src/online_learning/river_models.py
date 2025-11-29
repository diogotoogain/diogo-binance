"""
Online Learning Models using River library.

This module implements incremental learning for continuous adaptation
to market changes without retraining from scratch.
"""

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from river import tree, metrics, drift
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False


class OnlineLearner:
    """
    Online Learning model using Hoeffding Adaptive Tree.
    
    Features:
    - Incremental learning with learn_one() and learn_batch()
    - Drift detection with ADWIN
    - Persistence with pickle
    - Performance metrics tracking
    
    Example:
        >>> learner = OnlineLearner()
        >>> # Learn from individual samples
        >>> learner.learn_one({'price_change': 0.01, 'volume_change': 0.5}, 1)
        >>> # Predict
        >>> pred = learner.predict_one({'price_change': 0.02, 'volume_change': 0.3})
        >>> # Save and load
        >>> learner.save('model.pkl')
        >>> learner = OnlineLearner.load('model.pkl')
    """
    
    def __init__(
        self,
        grace_period: int = 200,
        delta: float = 1e-7,
        drift_detector_delta: float = 0.002,
        seed: int = 42
    ):
        """
        Initialize OnlineLearner.
        
        Args:
            grace_period: Number of samples before considering splits
            delta: Confidence level for splits (lower = more conservative)
            drift_detector_delta: Sensitivity for drift detection (lower = more sensitive)
            seed: Random seed for reproducibility
        """
        if not RIVER_AVAILABLE:
            raise ImportError(
                "River library is required for OnlineLearner. "
                "Install with: pip install river"
            )
        
        self.grace_period = grace_period
        self.delta = delta
        self.drift_detector_delta = drift_detector_delta
        self.seed = seed
        
        # Initialize Hoeffding Adaptive Tree classifier
        self.model = tree.HoeffdingAdaptiveTreeClassifier(
            grace_period=grace_period,
            delta=delta,
            seed=seed,
            drift_detector=drift.ADWIN(delta=drift_detector_delta)
        )
        
        # Metrics tracking
        self.accuracy_metric = metrics.Accuracy()
        self.f1_metric = metrics.F1()
        
        # Training stats
        self.n_samples_seen = 0
        self.n_drifts_detected = 0
        self.last_update_time: Optional[datetime] = None
        self.training_history: List[Dict[str, Any]] = []
        
    def learn_one(self, features: Dict[str, float], label: int) -> bool:
        """
        Learn from a single sample.
        
        Args:
            features: Dictionary of feature name -> value
            label: Target label (0, 1, or -1 for classification)
            
        Returns:
            True if drift was detected during this update
        """
        # Make prediction before learning (for metrics)
        if self.n_samples_seen > 0:
            pred = self.model.predict_one(features)
            if pred is not None:
                self.accuracy_metric.update(label, pred)
                self.f1_metric.update(label, pred)
        
        # Learn from sample
        self.model.learn_one(features, label)
        
        self.n_samples_seen += 1
        self.last_update_time = datetime.now(timezone.utc)
        
        # Check for drift (approximation based on model state)
        drift_detected = self._check_drift()
        if drift_detected:
            self.n_drifts_detected += 1
            
        return drift_detected
    
    def learn_batch(
        self,
        features_list: List[Dict[str, float]],
        labels: List[int]
    ) -> Dict[str, Any]:
        """
        Learn from a batch of samples.
        
        Args:
            features_list: List of feature dictionaries
            labels: List of target labels
            
        Returns:
            Dictionary with batch learning statistics
        """
        if len(features_list) != len(labels):
            raise ValueError("Features and labels must have same length")
            
        n_drifts = 0
        n_correct = 0
        
        for features, label in zip(features_list, labels):
            # Predict before learning
            if self.n_samples_seen > 0:
                pred = self.predict_one(features)
                if pred == label:
                    n_correct += 1
            
            # Learn
            drift_detected = self.learn_one(features, label)
            if drift_detected:
                n_drifts += 1
        
        batch_stats = {
            'n_samples': len(features_list),
            'n_drifts': n_drifts,
            'batch_accuracy': n_correct / len(features_list) if features_list else 0,
            'total_samples': self.n_samples_seen,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        self.training_history.append(batch_stats)
        
        return batch_stats
    
    def predict_one(self, features: Dict[str, float]) -> Optional[int]:
        """
        Predict label for a single sample.
        
        Args:
            features: Dictionary of feature name -> value
            
        Returns:
            Predicted label or None if model not trained
        """
        if self.n_samples_seen == 0:
            return None
            
        return self.model.predict_one(features)
    
    def predict_proba_one(self, features: Dict[str, float]) -> Dict[int, float]:
        """
        Predict class probabilities for a single sample.
        
        Args:
            features: Dictionary of feature name -> value
            
        Returns:
            Dictionary of label -> probability
        """
        if self.n_samples_seen == 0:
            return {}
            
        return self.model.predict_proba_one(features)
    
    def _check_drift(self) -> bool:
        """
        Check if drift was detected.
        
        Note: The HoeffdingAdaptiveTree handles drift internally via ADWIN.
        This method provides a way to track drift events externally.
        
        Returns:
            True if drift was likely detected
        """
        # The ADWIN detector is integrated into the tree
        # We track drift through accuracy changes
        if len(self.training_history) >= 2:
            recent_acc = self.training_history[-1].get('batch_accuracy', 0)
            prev_acc = self.training_history[-2].get('batch_accuracy', 0)
            # Significant accuracy drop might indicate drift
            if prev_acc - recent_acc > 0.1:
                return True
        return False
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary with accuracy, f1, etc.
        """
        return {
            'accuracy': self.accuracy_metric.get(),
            'f1': self.f1_metric.get(),
            'n_samples_seen': self.n_samples_seen,
            'n_drifts_detected': self.n_drifts_detected
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive model statistics.
        
        Returns:
            Dictionary with all model stats
        """
        metrics = self.get_metrics()
        return {
            **metrics,
            'grace_period': self.grace_period,
            'delta': self.delta,
            'drift_detector_delta': self.drift_detector_delta,
            'last_update_time': self.last_update_time.isoformat() if self.last_update_time else None,
            'training_history': self.training_history[-10:]  # Last 10 batches
        }
    
    def save(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save the model
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'model': self.model,
            'accuracy_metric': self.accuracy_metric,
            'f1_metric': self.f1_metric,
            'n_samples_seen': self.n_samples_seen,
            'n_drifts_detected': self.n_drifts_detected,
            'last_update_time': self.last_update_time,
            'training_history': self.training_history,
            'grace_period': self.grace_period,
            'delta': self.delta,
            'drift_detector_delta': self.drift_detector_delta,
            'seed': self.seed
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'OnlineLearner':
        """
        Load model from file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            OnlineLearner instance
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Create instance with saved parameters
        instance = cls(
            grace_period=state['grace_period'],
            delta=state['delta'],
            drift_detector_delta=state['drift_detector_delta'],
            seed=state['seed']
        )
        
        # Restore state
        instance.model = state['model']
        instance.accuracy_metric = state['accuracy_metric']
        instance.f1_metric = state['f1_metric']
        instance.n_samples_seen = state['n_samples_seen']
        instance.n_drifts_detected = state['n_drifts_detected']
        instance.last_update_time = state['last_update_time']
        instance.training_history = state['training_history']
        
        return instance
    
    def reset_metrics(self) -> None:
        """Reset performance metrics (but keep model)."""
        self.accuracy_metric = metrics.Accuracy()
        self.f1_metric = metrics.F1()


class OnlineRegressor:
    """
    Online Learning model for regression tasks.
    
    Uses Hoeffding Adaptive Tree Regressor for predicting
    continuous values like price changes or returns.
    """
    
    def __init__(
        self,
        grace_period: int = 200,
        delta: float = 1e-7,
        drift_detector_delta: float = 0.002,
        seed: int = 42
    ):
        """
        Initialize OnlineRegressor.
        
        Args:
            grace_period: Number of samples before considering splits
            delta: Confidence level for splits
            drift_detector_delta: Sensitivity for drift detection
            seed: Random seed for reproducibility
        """
        if not RIVER_AVAILABLE:
            raise ImportError(
                "River library is required for OnlineRegressor. "
                "Install with: pip install river"
            )
        
        self.grace_period = grace_period
        self.delta = delta
        self.drift_detector_delta = drift_detector_delta
        self.seed = seed
        
        # Initialize Hoeffding Adaptive Tree regressor
        self.model = tree.HoeffdingAdaptiveTreeRegressor(
            grace_period=grace_period,
            delta=delta,
            seed=seed,
            drift_detector=drift.ADWIN(delta=drift_detector_delta)
        )
        
        # Metrics tracking
        self.mae_metric = metrics.MAE()
        self.rmse_metric = metrics.RMSE()
        
        # Training stats
        self.n_samples_seen = 0
        self.last_update_time: Optional[datetime] = None
        
    def learn_one(self, features: Dict[str, float], target: float) -> None:
        """
        Learn from a single sample.
        
        Args:
            features: Dictionary of feature name -> value
            target: Target value
        """
        # Make prediction before learning (for metrics)
        if self.n_samples_seen > 0:
            pred = self.model.predict_one(features)
            if pred is not None:
                self.mae_metric.update(target, pred)
                self.rmse_metric.update(target, pred)
        
        # Learn from sample
        self.model.learn_one(features, target)
        
        self.n_samples_seen += 1
        self.last_update_time = datetime.now(timezone.utc)
    
    def predict_one(self, features: Dict[str, float]) -> Optional[float]:
        """
        Predict target for a single sample.
        
        Args:
            features: Dictionary of feature name -> value
            
        Returns:
            Predicted value or None if model not trained
        """
        if self.n_samples_seen == 0:
            return None
            
        return self.model.predict_one(features)
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary with MAE, RMSE, etc.
        """
        return {
            'mae': self.mae_metric.get(),
            'rmse': self.rmse_metric.get(),
            'n_samples_seen': self.n_samples_seen
        }
    
    def save(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save the model
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'model': self.model,
            'mae_metric': self.mae_metric,
            'rmse_metric': self.rmse_metric,
            'n_samples_seen': self.n_samples_seen,
            'last_update_time': self.last_update_time,
            'grace_period': self.grace_period,
            'delta': self.delta,
            'drift_detector_delta': self.drift_detector_delta,
            'seed': self.seed
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'OnlineRegressor':
        """
        Load model from file.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            OnlineRegressor instance
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        instance = cls(
            grace_period=state['grace_period'],
            delta=state['delta'],
            drift_detector_delta=state['drift_detector_delta'],
            seed=state['seed']
        )
        
        instance.model = state['model']
        instance.mae_metric = state['mae_metric']
        instance.rmse_metric = state['rmse_metric']
        instance.n_samples_seen = state['n_samples_seen']
        instance.last_update_time = state['last_update_time']
        
        return instance
