"""
Tests for Ensemble Manager
"""
import pytest
import sys
import time

sys.path.insert(0, '.')

from v2.src.strategies.ensemble import EnsembleManager
from v2.src.strategies.base import Strategy, Signal, SignalDirection


class MockStrategy(Strategy):
    """Mock strategy for testing."""
    
    def __init__(self, name, returns=None, win_rate=0.5):
        super().__init__(name, {}, enabled=True)
        self._mock_returns = returns or []
        self._returns = self._mock_returns.copy()
        self._mock_win_rate = win_rate
    
    def generate_signal(self, market_data):
        direction = market_data.get('direction')
        if direction:
            return Signal(
                direction=SignalDirection[direction],
                strategy_name=self.name,
                confidence=market_data.get('confidence', 0.8)
            )
        return None
    
    def get_win_rate(self):
        return self._mock_win_rate


class TestEnsembleManager:
    """Tests for EnsembleManager."""
    
    def setup_method(self):
        self.config = {
            'weighting_method': 'equal',
            'lookback_days': 30,
            'min_confidence': 0.5,
            'rebalance_frequency_hours': 24,
            'strategy_constraints': {
                'max_weight_per_strategy': 0.6,
                'min_weight_per_strategy': 0.0
            },
            'disable_strategy_if': {
                'max_drawdown_pct': 15.0,
                'min_sharpe': 0.5
            }
        }
        self.ensemble = EnsembleManager(self.config)
    
    def test_initialization(self):
        assert self.ensemble.weighting_method == 'equal'
        assert self.ensemble.min_confidence == 0.5
        assert self.ensemble.max_drawdown_pct == 15.0
        assert self.ensemble.min_sharpe == 0.5
    
    def test_register_strategy(self):
        strategy = MockStrategy("TestStrategy1")
        self.ensemble.register_strategy(strategy)
        
        assert "TestStrategy1" in self.ensemble._strategies
        assert "TestStrategy1" in self.ensemble._weights
    
    def test_remove_strategy(self):
        strategy = MockStrategy("TestStrategy1")
        self.ensemble.register_strategy(strategy)
        self.ensemble.remove_strategy("TestStrategy1")
        
        assert "TestStrategy1" not in self.ensemble._strategies
        assert "TestStrategy1" not in self.ensemble._weights
    
    def test_equal_weights(self):
        strategy1 = MockStrategy("Strategy1")
        strategy2 = MockStrategy("Strategy2")
        
        self.ensemble.register_strategy(strategy1)
        self.ensemble.register_strategy(strategy2)
        
        weights = self.ensemble.get_weights()
        assert weights["Strategy1"] == 0.5
        assert weights["Strategy2"] == 0.5
    
    def test_process_signals_unanimous_buy(self):
        strategy1 = MockStrategy("Strategy1")
        strategy2 = MockStrategy("Strategy2")
        
        self.ensemble.register_strategy(strategy1)
        self.ensemble.register_strategy(strategy2)
        
        signals = [
            Signal(SignalDirection.BUY, "Strategy1", 0.8),
            Signal(SignalDirection.BUY, "Strategy2", 0.7)
        ]
        
        result = self.ensemble.process_signals(signals)
        
        assert result is not None
        assert result.direction == SignalDirection.BUY
        assert result.strategy_name == "Ensemble"
    
    def test_process_signals_unanimous_sell(self):
        strategy1 = MockStrategy("Strategy1")
        strategy2 = MockStrategy("Strategy2")
        
        self.ensemble.register_strategy(strategy1)
        self.ensemble.register_strategy(strategy2)
        
        signals = [
            Signal(SignalDirection.SELL, "Strategy1", 0.8),
            Signal(SignalDirection.SELL, "Strategy2", 0.7)
        ]
        
        result = self.ensemble.process_signals(signals)
        
        assert result is not None
        assert result.direction == SignalDirection.SELL
    
    def test_process_signals_conflicting(self):
        strategy1 = MockStrategy("Strategy1")
        strategy2 = MockStrategy("Strategy2")
        
        self.ensemble.register_strategy(strategy1)
        self.ensemble.register_strategy(strategy2)
        
        # Equal weights, equal confidence - should have higher score for one
        signals = [
            Signal(SignalDirection.BUY, "Strategy1", 0.9),  # Higher confidence
            Signal(SignalDirection.SELL, "Strategy2", 0.7)
        ]
        
        result = self.ensemble.process_signals(signals)
        
        # BUY should win due to higher confidence
        assert result is not None
        assert result.direction == SignalDirection.BUY
    
    def test_min_confidence_filter(self):
        strategy1 = MockStrategy("Strategy1")
        self.ensemble.register_strategy(strategy1)
        
        signals = [
            Signal(SignalDirection.BUY, "Strategy1", 0.3)  # Below min_confidence
        ]
        
        result = self.ensemble.process_signals(signals)
        assert result is None
    
    def test_disabled_strategy_ignored(self):
        strategy1 = MockStrategy("Strategy1")
        strategy2 = MockStrategy("Strategy2")
        
        self.ensemble.register_strategy(strategy1)
        self.ensemble.register_strategy(strategy2)
        
        # Simulate high drawdown to keep strategy disabled
        strategy1._current_drawdown = 0.20  # 20% > max_drawdown_pct (15%)
        
        # Check health (will disable Strategy1 due to high drawdown)
        self.ensemble._check_strategy_health()
        
        signals = [
            Signal(SignalDirection.BUY, "Strategy1", 0.9),
            Signal(SignalDirection.SELL, "Strategy2", 0.8)
        ]
        
        result = self.ensemble.process_signals(signals)
        
        # Only Strategy2's signal should count
        assert result is not None
        assert result.direction == SignalDirection.SELL
    
    def test_hold_signals_ignored(self):
        strategy1 = MockStrategy("Strategy1")
        self.ensemble.register_strategy(strategy1)
        
        signals = [
            Signal(SignalDirection.HOLD, "Strategy1", 0.8)
        ]
        
        result = self.ensemble.process_signals(signals)
        assert result is None
    
    def test_empty_signals(self):
        result = self.ensemble.process_signals([])
        assert result is None
    
    def test_generate_ensemble_signal(self):
        strategy1 = MockStrategy("Strategy1")
        self.ensemble.register_strategy(strategy1)
        
        market_data = {'direction': 'BUY', 'confidence': 0.8}
        result = self.ensemble.generate_ensemble_signal(market_data)
        
        assert result is not None
        assert result.direction == SignalDirection.BUY
    
    def test_get_status(self):
        strategy1 = MockStrategy("Strategy1")
        self.ensemble.register_strategy(strategy1)
        
        status = self.ensemble.get_status()
        
        assert status['weighting_method'] == 'equal'
        assert status['min_confidence'] == 0.5
        assert status['total_strategies'] == 1
        assert 'weights' in status
        assert 'strategies' in status
    
    def test_combined_metadata(self):
        strategy1 = MockStrategy("Strategy1")
        strategy2 = MockStrategy("Strategy2")
        
        self.ensemble.register_strategy(strategy1)
        self.ensemble.register_strategy(strategy2)
        
        signals = [
            Signal(SignalDirection.BUY, "Strategy1", 0.8),
            Signal(SignalDirection.BUY, "Strategy2", 0.7)
        ]
        
        result = self.ensemble.process_signals(signals)
        
        assert 'ensemble_score' in result.metadata
        assert 'contributing_strategies' in result.metadata
        assert 'buy_score' in result.metadata
        assert 'sell_score' in result.metadata


class TestEnsembleWeighting:
    """Tests for different weighting methods."""
    
    def test_sharpe_weighting(self):
        config = {
            'weighting_method': 'sharpe',
            'min_confidence': 0.5,
            'strategy_constraints': {'max_weight_per_strategy': 0.9, 'min_weight_per_strategy': 0.0},
            'disable_strategy_if': {'max_drawdown_pct': 99, 'min_sharpe': -99}
        }
        ensemble = EnsembleManager(config)
        
        # Strategy with positive returns
        strategy1 = MockStrategy("Strategy1", returns=[0.01] * 100)
        # Strategy with no returns (should have 0 Sharpe)
        strategy2 = MockStrategy("Strategy2", returns=[])
        
        ensemble.register_strategy(strategy1)
        ensemble.register_strategy(strategy2)
        
        weights = ensemble.get_weights()
        
        # Strategy1 should have more weight (better Sharpe)
        assert weights["Strategy1"] >= weights["Strategy2"]
    
    def test_equal_weighting(self):
        config = {
            'weighting_method': 'equal',
            'min_confidence': 0.5,
            'strategy_constraints': {'max_weight_per_strategy': 0.9, 'min_weight_per_strategy': 0.0},
            'disable_strategy_if': {'max_drawdown_pct': 99, 'min_sharpe': -99}
        }
        ensemble = EnsembleManager(config)
        
        strategy1 = MockStrategy("Strategy1")
        strategy2 = MockStrategy("Strategy2")
        strategy3 = MockStrategy("Strategy3")
        
        ensemble.register_strategy(strategy1)
        ensemble.register_strategy(strategy2)
        ensemble.register_strategy(strategy3)
        
        weights = ensemble.get_weights()
        
        # Should be approximately 1/3 each
        assert abs(weights["Strategy1"] - 1/3) < 0.01
        assert abs(weights["Strategy2"] - 1/3) < 0.01
        assert abs(weights["Strategy3"] - 1/3) < 0.01
    
    def test_kelly_weighting(self):
        config = {
            'weighting_method': 'kelly',
            'min_confidence': 0.5,
            'strategy_constraints': {'max_weight_per_strategy': 0.9, 'min_weight_per_strategy': 0.0},
            'disable_strategy_if': {'max_drawdown_pct': 99, 'min_sharpe': -99}
        }
        ensemble = EnsembleManager(config)
        
        # Higher win rate should have more weight
        strategy1 = MockStrategy("Strategy1", win_rate=0.7)
        strategy2 = MockStrategy("Strategy2", win_rate=0.5)
        
        ensemble.register_strategy(strategy1)
        ensemble.register_strategy(strategy2)
        
        weights = ensemble.get_weights()
        
        # Strategy1 should have more weight (better Kelly)
        assert weights["Strategy1"] > weights["Strategy2"]


class TestEnsembleAutoDisable:
    """Tests for automatic strategy disabling."""
    
    def test_disable_on_high_drawdown(self):
        config = {
            'weighting_method': 'equal',
            'min_confidence': 0.5,
            'strategy_constraints': {'max_weight_per_strategy': 0.9, 'min_weight_per_strategy': 0.0},
            'disable_strategy_if': {'max_drawdown_pct': 15.0, 'min_sharpe': -99}
        }
        ensemble = EnsembleManager(config)
        
        strategy = MockStrategy("HighDrawdown")
        ensemble.register_strategy(strategy)
        
        # Simulate high drawdown
        strategy._peak_pnl = 100
        strategy._current_pnl = 80
        strategy._current_drawdown = 0.20  # 20% drawdown
        
        # This should trigger health check
        ensemble._check_strategy_health()
        
        assert "HighDrawdown" in ensemble.get_disabled_strategies()
