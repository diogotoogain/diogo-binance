"""
Tests for Volatility Breakout Strategy
"""
import pytest
import sys

sys.path.insert(0, '.')

from v2.src.strategies.volatility_breakout import VolatilityBreakout
from v2.src.strategies.base import SignalDirection


class TestVolatilityBreakout:
    """Tests for Volatility Breakout strategy."""
    
    def setup_method(self):
        self.config = {
            'enabled': True,
            'params': {
                'squeeze_bb_width_percentile': 20,
                'breakout_atr_mult': 1.5,
                'volume_confirm_mult': 1.5,
                'holding_minutes': 120
            },
            'filters': {
                'adx_filter': {
                    'enabled': True,
                    'min_adx_after_breakout': 20
                }
            },
            'throttling': {
                'enabled': True,
                'max_trades_per_day': 3
            }
        }
        self.strategy = VolatilityBreakout(self.config)
    
    def test_initialization(self):
        assert self.strategy.name == "VolatilityBreakout"
        assert self.strategy.enabled is True
        assert self.strategy.squeeze_bb_width_percentile == 20
        assert self.strategy.breakout_atr_mult == 1.5
        assert self.strategy.volume_confirm_mult == 1.5
        assert self.strategy.holding_minutes == 120
    
    def test_squeeze_detection(self):
        # Enter squeeze
        market_data = {
            'price': 50000,
            'high': 50000,
            'low': 50000,
            'prev_high': 50000,
            'prev_low': 50000,
            'atr': 100,
            'adx': 25,
            'bb_width_percentile': 15,  # Below threshold = squeeze
            'volume_ratio': 1.0
        }
        
        signal = self.strategy.generate_signal(market_data)
        
        # No signal yet (in squeeze, no breakout)
        assert signal is None
        # But squeeze should be detected
        assert self.strategy._in_squeeze is True
    
    def test_bullish_breakout_signal(self):
        # First, enter squeeze
        market_data_squeeze = {
            'price': 50000,
            'high': 50000,
            'low': 49900,
            'prev_high': 50000,
            'prev_low': 49900,
            'atr': 100,
            'adx': 25,
            'bb_width_percentile': 15,  # Squeeze
            'volume_ratio': 1.0
        }
        self.strategy.generate_signal(market_data_squeeze)
        assert self.strategy._in_squeeze is True
        
        # Then, breakout up
        market_data_breakout = {
            'price': 50200,
            'high': 50300,  # Breaks above prev_high + ATR*mult
            'low': 50100,
            'prev_high': 50000,
            'prev_low': 49900,
            'atr': 100,  # threshold = 100 * 1.5 = 150
            'adx': 25,
            'bb_width_percentile': 15,
            'volume_ratio': 2.0  # High volume confirms
        }
        signal = self.strategy.generate_signal(market_data_breakout)
        
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert 'Vol_Breakout_Up' in signal.reason
    
    def test_bearish_breakout_signal(self):
        # Enter squeeze
        market_data_squeeze = {
            'price': 50000,
            'high': 50100,
            'low': 50000,
            'prev_high': 50100,
            'prev_low': 50000,
            'atr': 100,
            'adx': 25,
            'bb_width_percentile': 15,
            'volume_ratio': 1.0
        }
        self.strategy.generate_signal(market_data_squeeze)
        
        # Breakout down
        market_data_breakout = {
            'price': 49800,
            'high': 49900,
            'low': 49700,  # Breaks below prev_low - ATR*mult
            'prev_high': 50100,
            'prev_low': 50000,
            'atr': 100,
            'adx': 25,
            'bb_width_percentile': 15,
            'volume_ratio': 2.0
        }
        signal = self.strategy.generate_signal(market_data_breakout)
        
        assert signal is not None
        assert signal.direction == SignalDirection.SELL
        assert 'Vol_Breakout_Down' in signal.reason
    
    def test_no_breakout_without_squeeze(self):
        # High volatility (no squeeze)
        market_data = {
            'price': 50200,
            'high': 50300,
            'low': 50100,
            'prev_high': 50000,
            'prev_low': 49900,
            'atr': 100,
            'adx': 25,
            'bb_width_percentile': 50,  # Not in squeeze
            'volume_ratio': 2.0
        }
        signal = self.strategy.generate_signal(market_data)
        
        # Should not signal without squeeze
        assert signal is None
    
    def test_volume_confirmation_required(self):
        # Enter squeeze
        market_data_squeeze = {
            'price': 50000,
            'high': 50000,
            'low': 49900,
            'prev_high': 50000,
            'prev_low': 49900,
            'atr': 100,
            'adx': 25,
            'bb_width_percentile': 15,
            'volume_ratio': 1.0
        }
        self.strategy.generate_signal(market_data_squeeze)
        
        # Breakout without volume
        market_data_breakout = {
            'price': 50200,
            'high': 50300,
            'low': 50100,
            'prev_high': 50000,
            'prev_low': 49900,
            'atr': 100,
            'adx': 25,
            'bb_width_percentile': 15,
            'volume_ratio': 1.0  # Below volume_confirm_mult
        }
        signal = self.strategy.generate_signal(market_data_breakout)
        
        assert signal is None
    
    def test_disabled_strategy(self):
        config = {'enabled': False, 'params': {}, 'filters': {}, 'throttling': {}}
        strategy = VolatilityBreakout(config)
        
        market_data = {
            'price': 50200,
            'high': 50300,
            'low': 50100,
            'prev_high': 50000,
            'prev_low': 49900,
            'atr': 100,
            'adx': 25,
            'bb_width_percentile': 15,
            'volume_ratio': 2.0
        }
        signal = strategy.generate_signal(market_data)
        assert signal is None
    
    def test_get_status(self):
        status = self.strategy.get_status()
        
        assert status['squeeze_bb_width_percentile'] == 20
        assert status['breakout_atr_mult'] == 1.5
        assert status['volume_confirm_mult'] == 1.5
        assert status['holding_minutes'] == 120
        assert 'in_squeeze' in status
        assert 'throttler' in status
    
    def test_confidence_calculation(self):
        # Enter squeeze with deep squeeze (low percentile)
        market_data_squeeze = {
            'price': 50000,
            'high': 50000,
            'low': 49900,
            'prev_high': 50000,
            'prev_low': 49900,
            'atr': 100,
            'adx': 25,
            'bb_width_percentile': 5,  # Very deep squeeze
            'volume_ratio': 1.0
        }
        self.strategy.generate_signal(market_data_squeeze)
        
        # Breakout with all confirmations
        market_data_breakout = {
            'price': 50200,
            'high': 50300,
            'low': 50100,
            'prev_high': 50000,
            'prev_low': 49900,
            'atr': 100,
            'adx': 25,  # ADX confirms
            'bb_width_percentile': 5,
            'volume_ratio': 2.0  # Volume confirms
        }
        signal = self.strategy.generate_signal(market_data_breakout)
        
        # Should have high confidence with all confirmations
        assert signal.confidence >= 0.9
    
    def test_squeeze_reset_after_signal(self):
        # Enter squeeze and generate signal
        market_data_squeeze = {
            'price': 50000,
            'high': 50000,
            'low': 49900,
            'prev_high': 50000,
            'prev_low': 49900,
            'atr': 100,
            'adx': 25,
            'bb_width_percentile': 15,
            'volume_ratio': 1.0
        }
        self.strategy.generate_signal(market_data_squeeze)
        
        market_data_breakout = {
            'price': 50200,
            'high': 50300,
            'low': 50100,
            'prev_high': 50000,
            'prev_low': 49900,
            'atr': 100,
            'adx': 25,
            'bb_width_percentile': 15,
            'volume_ratio': 2.0
        }
        signal = self.strategy.generate_signal(market_data_breakout)
        
        assert signal is not None
        # Squeeze should be reset after signal
        assert self.strategy._in_squeeze is False
        assert self.strategy._squeeze_start_price is None


class TestVolatilityBreakoutDefaultConfig:
    """Test with default config values."""
    
    def test_default_values(self):
        config = {'enabled': True, 'params': {}, 'filters': {}, 'throttling': {}}
        strategy = VolatilityBreakout(config)
        
        assert strategy.squeeze_bb_width_percentile == 20
        assert strategy.breakout_atr_mult == 1.5
        assert strategy.volume_confirm_mult == 1.5
        assert strategy.holding_minutes == 120
