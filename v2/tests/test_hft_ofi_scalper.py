"""
Tests for HFT OFI Scalper Strategy
"""
import pytest
import sys
import time

sys.path.insert(0, '.')

from v2.src.strategies.hft_ofi_scalper import HFTOFIScalper
from v2.src.strategies.base import SignalDirection


class TestHFTOFIScalper:
    """Tests for HFT OFI Scalper strategy."""
    
    def setup_method(self):
        self.config = {
            'enabled': True,
            'params': {
                'ofi_threshold': 0.3,
                'tfi_threshold': 0.3,
                'min_spread_bps': 1.0,
                'holding_seconds': 30
            },
            'filters': {
                'adx_filter': {
                    'enabled': True,
                    'max_adx': 30
                },
                'volume_spike_filter': {
                    'enabled': True,
                    'min_spike_multiplier': 1.5
                }
            },
            'throttling': {
                'enabled': True,
                'max_trades_per_minute': 5
            }
        }
        self.strategy = HFTOFIScalper(self.config)
    
    def test_initialization(self):
        assert self.strategy.name == "HFTOFIScalper"
        assert self.strategy.enabled is True
        assert self.strategy.ofi_threshold == 0.3
        assert self.strategy.tfi_threshold == 0.3
        assert self.strategy.holding_seconds == 30
        assert self.strategy.max_adx == 30
    
    def test_initialization_disabled(self):
        config = {'enabled': False, 'params': {}, 'throttling': {}}
        strategy = HFTOFIScalper(config)
        
        assert strategy.enabled is False
    
    def test_generate_signal_buy(self):
        market_data = {
            'ofi': 0.5,
            'tfi': 0.4,
            'adx': 20,
            'volume_spike': 2.0,
            'spread_bps': 1.5,
            'price': 50000
        }
        
        signal = self.strategy.generate_signal(market_data)
        
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert signal.confidence > 0
        assert 'OFI_High_Buy' in signal.reason
    
    def test_generate_signal_sell(self):
        market_data = {
            'ofi': -0.5,
            'tfi': -0.4,
            'adx': 20,
            'volume_spike': 2.0,
            'spread_bps': 1.5,
            'price': 50000
        }
        
        signal = self.strategy.generate_signal(market_data)
        
        assert signal is not None
        assert signal.direction == SignalDirection.SELL
        assert 'OFI_High_Sell' in signal.reason
    
    def test_generate_signal_below_threshold(self):
        market_data = {
            'ofi': 0.2,  # Below threshold
            'tfi': 0.2,
            'adx': 20,
            'volume_spike': 2.0,
            'spread_bps': 1.5,
            'price': 50000
        }
        
        signal = self.strategy.generate_signal(market_data)
        assert signal is None
    
    def test_adx_filter_blocks(self):
        market_data = {
            'ofi': 0.5,
            'tfi': 0.5,
            'adx': 40,  # Above max_adx
            'volume_spike': 2.0,
            'spread_bps': 1.5,
            'price': 50000
        }
        
        signal = self.strategy.generate_signal(market_data)
        assert signal is None
    
    def test_volume_spike_filter_blocks(self):
        market_data = {
            'ofi': 0.5,
            'tfi': 0.5,
            'adx': 20,
            'volume_spike': 1.0,  # Below min
            'spread_bps': 1.5,
            'price': 50000
        }
        
        signal = self.strategy.generate_signal(market_data)
        assert signal is None
    
    def test_spread_filter_blocks(self):
        market_data = {
            'ofi': 0.5,
            'tfi': 0.5,
            'adx': 20,
            'volume_spike': 2.0,
            'spread_bps': 0.5,  # Below min
            'price': 50000
        }
        
        signal = self.strategy.generate_signal(market_data)
        assert signal is None
    
    def test_disabled_strategy(self):
        config = {'enabled': False, 'params': {}, 'throttling': {}}
        strategy = HFTOFIScalper(config)
        
        market_data = {
            'ofi': 0.5,
            'tfi': 0.5,
            'adx': 20,
            'volume_spike': 2.0,
            'spread_bps': 1.5,
            'price': 50000
        }
        
        signal = strategy.generate_signal(market_data)
        assert signal is None
    
    def test_throttling_blocks_after_limit(self):
        market_data = {
            'ofi': 0.5,
            'tfi': 0.5,
            'adx': 20,
            'volume_spike': 2.0,
            'spread_bps': 1.5,
            'price': 50000
        }
        
        # Generate 5 signals (max per minute)
        for _ in range(5):
            signal = self.strategy.generate_signal(market_data)
            assert signal is not None
        
        # 6th should be blocked
        signal = self.strategy.generate_signal(market_data)
        assert signal is None
    
    def test_get_status(self):
        status = self.strategy.get_status()
        
        assert status['name'] == 'HFTOFIScalper'
        assert status['ofi_threshold'] == 0.3
        assert status['tfi_threshold'] == 0.3
        assert status['holding_seconds'] == 30
        assert 'throttler' in status
    
    def test_signal_metadata(self):
        market_data = {
            'ofi': 0.5,
            'tfi': 0.4,
            'adx': 20,
            'volume_spike': 2.0,
            'spread_bps': 1.5,
            'price': 50000
        }
        
        signal = self.strategy.generate_signal(market_data)
        
        assert signal.metadata['ofi'] == 0.5
        assert signal.metadata['tfi'] == 0.4
        assert signal.metadata['price'] == 50000
        assert signal.metadata['holding_seconds'] == 30


class TestHFTOFIScalperDefaultConfig:
    """Test with default config values."""
    
    def test_default_values(self):
        config = {'enabled': True, 'params': {}, 'filters': {}, 'throttling': {}}
        strategy = HFTOFIScalper(config)
        
        # Should use defaults
        assert strategy.ofi_threshold == 0.3
        assert strategy.tfi_threshold == 0.3
        assert strategy.min_spread_bps == 1.0
        assert strategy.holding_seconds == 30
