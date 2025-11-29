"""
Tests for V2 Strategies Base Classes
"""
import pytest
import sys
import time

sys.path.insert(0, '.')

from v2.src.strategies.base import Strategy, Signal, SignalDirection


class ConcreteStrategy(Strategy):
    """Implementação concreta para testes."""
    
    def generate_signal(self, market_data):
        if market_data.get('trigger_buy'):
            return Signal(
                direction=SignalDirection.BUY,
                strategy_name=self.name,
                confidence=0.8,
                reason="Test buy signal"
            )
        elif market_data.get('trigger_sell'):
            return Signal(
                direction=SignalDirection.SELL,
                strategy_name=self.name,
                confidence=0.7,
                reason="Test sell signal"
            )
        return None


class TestSignal:
    """Tests for Signal dataclass."""
    
    def test_signal_creation(self):
        signal = Signal(
            direction=SignalDirection.BUY,
            strategy_name="TestStrategy",
            confidence=0.9,
            reason="Test reason"
        )
        
        assert signal.direction == SignalDirection.BUY
        assert signal.strategy_name == "TestStrategy"
        assert signal.confidence == 0.9
        assert signal.reason == "Test reason"
        assert signal.timestamp > 0
        assert signal.metadata == {}
    
    def test_signal_to_dict(self):
        signal = Signal(
            direction=SignalDirection.SELL,
            strategy_name="Test",
            confidence=0.5,
            reason="Reason",
            metadata={'key': 'value'}
        )
        
        d = signal.to_dict()
        assert d['direction'] == 'SELL'
        assert d['strategy_name'] == 'Test'
        assert d['confidence'] == 0.5
        assert d['reason'] == 'Reason'
        assert d['metadata'] == {'key': 'value'}


class TestStrategy:
    """Tests for Strategy base class."""
    
    def setup_method(self):
        self.config = {'param1': 10, 'param2': 'test'}
        self.strategy = ConcreteStrategy("TestStrategy", self.config)
    
    def test_initialization(self):
        assert self.strategy.name == "TestStrategy"
        assert self.strategy.config == self.config
        assert self.strategy.enabled is True
        assert self.strategy._total_signals == 0
        assert self.strategy._profitable_signals == 0
    
    def test_initialization_disabled(self):
        strategy = ConcreteStrategy("Disabled", {}, enabled=False)
        assert strategy.enabled is False
    
    def test_generate_signal_buy(self):
        signal = self.strategy.generate_signal({'trigger_buy': True})
        
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.confidence == 0.8
    
    def test_generate_signal_sell(self):
        signal = self.strategy.generate_signal({'trigger_sell': True})
        
        assert signal is not None
        assert signal.direction == SignalDirection.SELL
        assert signal.confidence == 0.7
    
    def test_generate_signal_none(self):
        signal = self.strategy.generate_signal({})
        assert signal is None
    
    def test_update_performance_profit(self):
        self.strategy.update_performance(100.0)
        
        assert self.strategy._current_pnl == 100.0
        assert self.strategy._peak_pnl == 100.0
        assert self.strategy._current_drawdown == 0.0
        assert len(self.strategy._returns) == 1
    
    def test_update_performance_drawdown(self):
        self.strategy.update_performance(100.0)  # Peak
        self.strategy.update_performance(-50.0)   # Loss
        
        assert self.strategy._current_pnl == 50.0
        assert self.strategy._peak_pnl == 100.0
        assert self.strategy._current_drawdown == 0.5  # 50% drawdown
    
    def test_get_win_rate(self):
        self.strategy._total_signals = 10
        self.strategy._profitable_signals = 6
        
        assert self.strategy.get_win_rate() == 0.6
    
    def test_get_win_rate_no_signals(self):
        assert self.strategy.get_win_rate() == 0.0
    
    def test_get_sharpe_ratio_insufficient_data(self):
        assert self.strategy.get_sharpe_ratio() == 0.0
    
    def test_get_sharpe_ratio(self):
        # Add some returns
        for i in range(100):
            self.strategy._returns.append(0.01)  # 1% daily return
        
        sharpe = self.strategy.get_sharpe_ratio()
        # Should be positive with consistent positive returns
        assert sharpe > 0
    
    def test_get_sortino_ratio_insufficient_data(self):
        assert self.strategy.get_sortino_ratio() == 0.0
    
    def test_get_sortino_ratio_no_negative_returns(self):
        for i in range(10):
            self.strategy._returns.append(0.01)
        
        sortino = self.strategy.get_sortino_ratio()
        # Should be infinity when no negative returns
        assert sortino == float('inf')
    
    def test_get_status(self):
        status = self.strategy.get_status()
        
        assert status['name'] == 'TestStrategy'
        assert status['enabled'] is True
        assert status['total_signals'] == 0
        assert status['win_rate'] == 0.0
        assert 'sharpe_ratio' in status
        assert 'sortino_ratio' in status
        assert 'current_drawdown' in status
    
    def test_reset_performance(self):
        self.strategy.update_performance(100.0)
        self.strategy._total_signals = 5
        self.strategy._profitable_signals = 3
        
        self.strategy.reset_performance()
        
        assert self.strategy._total_signals == 0
        assert self.strategy._profitable_signals == 0
        assert self.strategy._current_pnl == 0.0
        assert self.strategy._peak_pnl == 0.0
        assert len(self.strategy._returns) == 0


class TestSignalDirection:
    """Tests for SignalDirection enum."""
    
    def test_values(self):
        assert SignalDirection.BUY.value == "BUY"
        assert SignalDirection.SELL.value == "SELL"
        assert SignalDirection.HOLD.value == "HOLD"
