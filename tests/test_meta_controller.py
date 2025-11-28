import pytest
import time
import sys
sys.path.insert(0, '.')

from src.core.meta_controller import MetaController


class TestMetaController:
    def setup_method(self):
        self.mc = MetaController(
            min_strategies_agree=2,
            signal_timeout_seconds=5.0,
            debounce_seconds=0.1
        )

    def test_initialization(self):
        assert self.mc.min_strategies_agree == 2
        assert self.mc.total_decisions == 0

    def test_single_signal_no_action(self):
        signal = {'action': 'BUY', 'reason': 'test'}
        result = self.mc.receive_signal('FluxoBrabo', signal)
        assert result is None

    def test_two_signals_same_direction(self):
        self.mc.receive_signal('FluxoBrabo', {'action': 'BUY'})
        time.sleep(0.15)
        result = self.mc.receive_signal('VPINDetector', {'action': 'BUY'})
        assert result is not None
        assert result['action'] == 'BUY'
        assert result['confidence'] == 'MEDIUM'

    def test_three_signals_high_confidence(self):
        self.mc.receive_signal('FluxoBrabo', {'action': 'BUY'})
        time.sleep(0.15)
        self.mc.receive_signal('VPINDetector', {'action': 'BUY'})
        time.sleep(0.15)
        result = self.mc.receive_signal('OBI', {'action': 'BUY'})
        assert result is not None
        assert result['confidence'] == 'HIGH'

    def test_conflicting_signals(self):
        self.mc.receive_signal('FluxoBrabo', {'action': 'BUY'})
        time.sleep(0.15)
        result = self.mc.receive_signal('VPINDetector', {'action': 'SELL'})
        assert result is None
