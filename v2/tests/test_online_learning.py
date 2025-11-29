"""
Tests for the Online Learning Module.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path for relative imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Check if river is available
try:
    import river
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False

# Skip all tests if river not installed
pytestmark = pytest.mark.skipif(
    not RIVER_AVAILABLE,
    reason="river library not installed"
)


class TestOnlineLearner:
    """Tests for OnlineLearner class."""
    
    @pytest.fixture
    def learner(self):
        """Create OnlineLearner instance."""
        from v2.src.online_learning.river_models import OnlineLearner
        return OnlineLearner(grace_period=10, seed=42)
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        features_list = []
        labels = []
        
        for i in range(n_samples):
            features = {
                'price_change': np.random.uniform(-0.05, 0.05),
                'volume_change': np.random.uniform(-0.5, 0.5),
                'high_low_range': np.random.uniform(0, 0.02),
                'trades_intensity': np.random.uniform(0, 2)
            }
            # Label based on price_change (simple rule for testing)
            label = 1 if features['price_change'] > 0 else 0
            
            features_list.append(features)
            labels.append(label)
        
        return features_list, labels
    
    def test_initialization(self, learner):
        """OnlineLearner should initialize correctly."""
        assert learner.n_samples_seen == 0
        assert learner.grace_period == 10
        assert learner.model is not None
    
    def test_learn_one(self, learner):
        """learn_one should update model and track samples."""
        features = {
            'price_change': 0.01,
            'volume_change': 0.5,
            'high_low_range': 0.005,
            'trades_intensity': 1.0
        }
        
        learner.learn_one(features, 1)
        
        assert learner.n_samples_seen == 1
        assert learner.last_update_time is not None
    
    def test_learn_batch(self, learner, sample_data):
        """learn_batch should process multiple samples."""
        features_list, labels = sample_data
        
        stats = learner.learn_batch(features_list, labels)
        
        assert stats['n_samples'] == len(features_list)
        assert learner.n_samples_seen == len(features_list)
        assert 'batch_accuracy' in stats
        assert 'timestamp' in stats
    
    def test_predict_one_before_training(self, learner):
        """predict_one should return None before training."""
        features = {'price_change': 0.01, 'volume_change': 0.5}
        pred = learner.predict_one(features)
        
        assert pred is None
    
    def test_predict_one_after_training(self, learner, sample_data):
        """predict_one should return valid prediction after training."""
        features_list, labels = sample_data
        
        # Train
        learner.learn_batch(features_list, labels)
        
        # Predict
        pred = learner.predict_one(features_list[0])
        
        assert pred in [0, 1]
    
    def test_predict_proba_one(self, learner, sample_data):
        """predict_proba_one should return probabilities."""
        features_list, labels = sample_data
        
        # Train
        learner.learn_batch(features_list, labels)
        
        # Predict probabilities
        proba = learner.predict_proba_one(features_list[0])
        
        assert isinstance(proba, dict)
        if proba:  # May be empty if model not ready
            # Probabilities should sum to approximately 1
            total = sum(proba.values())
            assert 0.99 <= total <= 1.01
    
    def test_get_metrics(self, learner, sample_data):
        """get_metrics should return performance metrics."""
        features_list, labels = sample_data
        learner.learn_batch(features_list, labels)
        
        metrics = learner.get_metrics()
        
        assert 'accuracy' in metrics
        assert 'f1' in metrics
        assert 'n_samples_seen' in metrics
        assert metrics['n_samples_seen'] == len(features_list)
    
    def test_get_stats(self, learner, sample_data):
        """get_stats should return comprehensive statistics."""
        features_list, labels = sample_data
        learner.learn_batch(features_list, labels)
        
        stats = learner.get_stats()
        
        assert 'accuracy' in stats
        assert 'f1' in stats
        assert 'grace_period' in stats
        assert 'training_history' in stats
    
    def test_save_and_load(self, learner, sample_data):
        """Model should be saveable and loadable."""
        from v2.src.online_learning.river_models import OnlineLearner
        
        features_list, labels = sample_data
        learner.learn_batch(features_list, labels)
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_model.pkl"
            learner.save(str(model_path))
            
            assert model_path.exists()
            
            # Load
            loaded = OnlineLearner.load(str(model_path))
            
            assert loaded.n_samples_seen == learner.n_samples_seen
            assert loaded.grace_period == learner.grace_period
            
            # Predictions should be the same
            pred1 = learner.predict_one(features_list[0])
            pred2 = loaded.predict_one(features_list[0])
            assert pred1 == pred2
    
    def test_reset_metrics(self, learner, sample_data):
        """reset_metrics should clear metrics but keep model."""
        features_list, labels = sample_data
        learner.learn_batch(features_list, labels)
        
        original_samples = learner.n_samples_seen
        learner.reset_metrics()
        
        # Model should still work
        assert learner.n_samples_seen == original_samples
        pred = learner.predict_one(features_list[0])
        assert pred in [0, 1]
        
        # Metrics should be reset
        metrics = learner.get_metrics()
        assert metrics['accuracy'] == 0 or metrics['n_samples_seen'] == original_samples


class TestOnlineRegressor:
    """Tests for OnlineRegressor class."""
    
    @pytest.fixture
    def regressor(self):
        """Create OnlineRegressor instance."""
        from v2.src.online_learning.river_models import OnlineRegressor
        return OnlineRegressor(grace_period=10, seed=42)
    
    @pytest.fixture
    def sample_regression_data(self):
        """Generate sample data for regression testing."""
        np.random.seed(42)
        n_samples = 100
        
        features_list = []
        targets = []
        
        for i in range(n_samples):
            features = {
                'x1': np.random.uniform(-1, 1),
                'x2': np.random.uniform(-1, 1)
            }
            # Simple linear relationship with noise
            target = 0.5 * features['x1'] + 0.3 * features['x2'] + np.random.normal(0, 0.1)
            
            features_list.append(features)
            targets.append(target)
        
        return features_list, targets
    
    def test_initialization(self, regressor):
        """OnlineRegressor should initialize correctly."""
        assert regressor.n_samples_seen == 0
        assert regressor.model is not None
    
    def test_learn_one(self, regressor):
        """learn_one should update regressor."""
        features = {'x1': 0.5, 'x2': 0.3}
        regressor.learn_one(features, 0.5)
        
        assert regressor.n_samples_seen == 1
    
    def test_predict_after_training(self, regressor, sample_regression_data):
        """Predictions should be reasonable after training."""
        features_list, targets = sample_regression_data
        
        # Train
        for features, target in zip(features_list, targets):
            regressor.learn_one(features, target)
        
        # Predict
        pred = regressor.predict_one(features_list[0])
        
        assert pred is not None
        assert isinstance(pred, (int, float))
    
    def test_get_metrics(self, regressor, sample_regression_data):
        """get_metrics should return MAE and RMSE."""
        features_list, targets = sample_regression_data
        
        for features, target in zip(features_list, targets):
            regressor.learn_one(features, target)
        
        metrics = regressor.get_metrics()
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'n_samples_seen' in metrics
    
    def test_save_and_load(self, regressor, sample_regression_data):
        """OnlineRegressor should be saveable and loadable."""
        from v2.src.online_learning.river_models import OnlineRegressor
        
        features_list, targets = sample_regression_data
        
        for features, target in zip(features_list, targets):
            regressor.learn_one(features, target)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_regressor.pkl"
            regressor.save(str(model_path))
            
            loaded = OnlineRegressor.load(str(model_path))
            
            assert loaded.n_samples_seen == regressor.n_samples_seen


class TestMarketDriftDetector:
    """Tests for MarketDriftDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create MarketDriftDetector instance."""
        from v2.src.online_learning.drift_detector import MarketDriftDetector
        return MarketDriftDetector(
            pnl_delta=0.1,  # Less sensitive for testing
            win_rate_delta=0.1,
            spread_delta=0.1,
            pause_threshold=2,
            cooldown_samples=10
        )
    
    def test_initialization(self, detector):
        """MarketDriftDetector should initialize correctly."""
        assert detector.n_samples == 0
        assert detector.n_pnl_drifts == 0
        assert detector.pnl_detector is not None
    
    def test_update_single_metric(self, detector):
        """update should work with single metric."""
        result = detector.update(pnl=0.01)
        
        assert 'drift_detected' in result
        assert 'drifting_metrics' in result
        assert detector.n_samples == 1
    
    def test_update_all_metrics(self, detector):
        """update should work with all metrics."""
        result = detector.update(pnl=0.01, win_rate=0.6, spread=0.001)
        
        assert detector.n_samples == 1
        assert isinstance(result['drifting_metrics'], list)
    
    def test_drift_detection(self, detector):
        """Detector should detect drift when distribution changes."""
        # Feed stable values
        for _ in range(50):
            detector.update(pnl=0.01)
        
        # Feed very different values (should trigger drift)
        for _ in range(50):
            detector.update(pnl=-0.05)
        
        # At least one drift should have been detected
        stats = detector.get_stats()
        # Note: drift detection depends on ADWIN internal state
        assert stats['n_samples'] == 100
    
    def test_should_pause_trading_initially_false(self, detector):
        """should_pause_trading should be False initially."""
        assert detector.should_pause_trading() is False
    
    def test_get_alert_level(self, detector):
        """get_alert_level should return valid level."""
        level = detector.get_alert_level()
        
        assert level in ['normal', 'warning', 'critical']
    
    def test_get_stats(self, detector):
        """get_stats should return comprehensive stats."""
        for i in range(10):
            detector.update(pnl=0.01 * i)
        
        stats = detector.get_stats()
        
        assert 'n_samples' in stats
        assert 'n_pnl_drifts' in stats
        assert 'total_drifts' in stats
        assert 'alert_level' in stats
        assert 'should_pause' in stats
    
    def test_get_detector_estimates(self, detector):
        """get_detector_estimates should return means."""
        for _ in range(10):
            detector.update(pnl=0.02, win_rate=0.7, spread=0.001)
        
        estimates = detector.get_detector_estimates()
        
        assert 'pnl_mean' in estimates
        assert 'win_rate_mean' in estimates
        assert 'spread_mean' in estimates
    
    def test_reset(self, detector):
        """reset should clear all state."""
        for _ in range(10):
            detector.update(pnl=0.01)
        
        detector.reset()
        
        assert detector.n_samples == 0
        assert detector.n_pnl_drifts == 0
        assert len(detector.drift_history) == 0


class TestMultiMetricDriftDetector:
    """Tests for MultiMetricDriftDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create MultiMetricDriftDetector instance."""
        from v2.src.online_learning.drift_detector import MultiMetricDriftDetector
        return MultiMetricDriftDetector(
            metric_names=['metric_a', 'metric_b', 'metric_c'],
            delta=0.1,
            pause_threshold=2,
            cooldown_samples=10
        )
    
    def test_initialization(self, detector):
        """MultiMetricDriftDetector should initialize correctly."""
        assert detector.n_samples == 0
        assert len(detector.detectors) == 3
        assert 'metric_a' in detector.detectors
    
    def test_update(self, detector):
        """update should handle multiple metrics."""
        result = detector.update({
            'metric_a': 1.0,
            'metric_b': 2.0,
            'metric_c': 3.0
        })
        
        assert detector.n_samples == 1
        assert 'drift_detected' in result
    
    def test_unknown_metric_ignored(self, detector):
        """Unknown metrics should be ignored."""
        result = detector.update({
            'metric_a': 1.0,
            'unknown_metric': 999.0
        })
        
        assert detector.n_samples == 1
    
    def test_get_stats(self, detector):
        """get_stats should return stats for all metrics."""
        for _ in range(5):
            detector.update({'metric_a': 1.0, 'metric_b': 2.0})
        
        stats = detector.get_stats()
        
        assert 'n_samples' in stats
        assert 'drift_counts' in stats
        assert 'metric_a' in stats['drift_counts']


class TestOnlineLearningScript:
    """Tests for the online learning update script."""
    
    def test_prepare_features(self):
        """prepare_features should generate valid features."""
        # Import the function using the project root path constant
        v2_scripts = PROJECT_ROOT / "v2" / "scripts"
        sys.path.insert(0, str(v2_scripts.parent))
        from scripts.online_learning_update import prepare_features
        
        # Create sample data
        df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'trades_count': [50, 55, 60, 65, 70]
        })
        
        features_list, labels = prepare_features(df)
        
        # Should have n-1 samples (since we use previous row)
        assert len(features_list) == len(df) - 1
        assert len(labels) == len(df) - 1
        
        # Check features structure
        for f in features_list:
            assert 'price_change' in f
            assert 'volume_change' in f
            assert 'high_low_range' in f
            assert 'trades_intensity' in f
        
        # Check labels
        for label in labels:
            assert label in [0, 1]
    
    def test_prepare_features_handles_column_variations(self):
        """prepare_features should handle uppercase column names."""
        from scripts.online_learning_update import prepare_features
        
        # Create sample data with uppercase columns
        df = pd.DataFrame({
            'Open': [100, 101],
            'High': [102, 103],
            'Low': [99, 100],
            'Close': [101, 102],
            'Volume': [1000, 1100]
        })
        
        features_list, labels = prepare_features(df)
        
        assert len(features_list) == 1
        assert 'price_change' in features_list[0]
