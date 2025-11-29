"""
Tests for Momentum Intraday Strategy
"""
import pytest
import sys

sys.path.insert(0, '.')

from v2.src.strategies.momentum_intraday import MomentumIntraday
from v2.src.strategies.base import SignalDirection


class TestMomentumIntraday:
    """Tests for Momentum Intraday strategy."""
    
    def setup_method(self):
        self.config = {
            'enabled': True,
            'params': {
                'ema_fast': 9,
                'ema_slow': 21,
                'rsi_entry_threshold': 50,
                'holding_minutes': 60
            },
            'filters': {
                'adx_filter': {
                    'enabled': True,
                    'min_adx': 25
                },
                'volume_confirmation': {
                    'enabled': True,
                    'min_volume_percentile': 60
                }
            },
            'throttling': {
                'enabled': True,
                'max_trades_per_hour': 4
            }
        }
        self.strategy = MomentumIntraday(self.config)
    
    def test_initialization(self):
        assert self.strategy.name == "MomentumIntraday"
        assert self.strategy.enabled is True
        assert self.strategy.ema_fast == 9
        assert self.strategy.ema_slow == 21
        assert self.strategy.min_adx == 25
        assert self.strategy.holding_minutes == 60
    
    def test_bullish_crossover_signal(self):
        # First tick - initialize state (EMA fast below slow)
        market_data_1 = {
            'ema_fast': 50,
            'ema_slow': 51,
            'rsi': 55,
            'adx': 30,
            'volume_percentile': 70,
            'price': 50000
        }
        signal = self.strategy.generate_signal(market_data_1)
        assert signal is None  # First tick, no crossover yet
        
        # Second tick - bullish crossover (EMA fast above slow)
        market_data_2 = {
            'ema_fast': 52,
            'ema_slow': 51,
            'rsi': 55,
            'adx': 30,
            'volume_percentile': 70,
            'price': 50100
        }
        signal = self.strategy.generate_signal(market_data_2)
        
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert 'Bullish_EMA_Cross' in signal.reason
    
    def test_bearish_crossover_signal(self):
        # First tick - EMA fast above slow
        market_data_1 = {
            'ema_fast': 52,
            'ema_slow': 51,
            'rsi': 45,
            'adx': 30,
            'volume_percentile': 70,
            'price': 50000
        }
        self.strategy.generate_signal(market_data_1)
        
        # Second tick - bearish crossover
        market_data_2 = {
            'ema_fast': 50,
            'ema_slow': 51,
            'rsi': 45,
            'adx': 30,
            'volume_percentile': 70,
            'price': 49900
        }
        signal = self.strategy.generate_signal(market_data_2)
        
        assert signal is not None
        assert signal.direction == SignalDirection.SELL
        assert 'Bearish_EMA_Cross' in signal.reason
    
    def test_adx_filter_blocks(self):
        # First tick
        market_data_1 = {
            'ema_fast': 50,
            'ema_slow': 51,
            'rsi': 55,
            'adx': 20,  # Below min_adx
            'volume_percentile': 70,
            'price': 50000
        }
        self.strategy.generate_signal(market_data_1)
        
        # Second tick - would be crossover but ADX too low
        market_data_2 = {
            'ema_fast': 52,
            'ema_slow': 51,
            'rsi': 55,
            'adx': 20,
            'volume_percentile': 70,
            'price': 50100
        }
        signal = self.strategy.generate_signal(market_data_2)
        assert signal is None
    
    def test_volume_filter_blocks(self):
        market_data_1 = {
            'ema_fast': 50,
            'ema_slow': 51,
            'rsi': 55,
            'adx': 30,
            'volume_percentile': 50,  # Below min
            'price': 50000
        }
        self.strategy.generate_signal(market_data_1)
        
        market_data_2 = {
            'ema_fast': 52,
            'ema_slow': 51,
            'rsi': 55,
            'adx': 30,
            'volume_percentile': 50,
            'price': 50100
        }
        signal = self.strategy.generate_signal(market_data_2)
        assert signal is None
    
    def test_rsi_blocks_buy_when_too_low(self):
        # Setup for bullish crossover
        market_data_1 = {
            'ema_fast': 50,
            'ema_slow': 51,
            'rsi': 40,  # Below threshold
            'adx': 30,
            'volume_percentile': 70,
            'price': 50000
        }
        self.strategy.generate_signal(market_data_1)
        
        market_data_2 = {
            'ema_fast': 52,
            'ema_slow': 51,
            'rsi': 40,  # Still below
            'adx': 30,
            'volume_percentile': 70,
            'price': 50100
        }
        signal = self.strategy.generate_signal(market_data_2)
        assert signal is None
    
    def test_no_crossover_no_signal(self):
        # Continuous uptrend, no crossover
        market_data_1 = {
            'ema_fast': 52,
            'ema_slow': 50,
            'rsi': 55,
            'adx': 30,
            'volume_percentile': 70,
            'price': 50000
        }
        self.strategy.generate_signal(market_data_1)
        
        market_data_2 = {
            'ema_fast': 53,
            'ema_slow': 50,  # Still above
            'rsi': 55,
            'adx': 30,
            'volume_percentile': 70,
            'price': 50100
        }
        signal = self.strategy.generate_signal(market_data_2)
        assert signal is None
    
    def test_disabled_strategy(self):
        config = {'enabled': False, 'params': {}, 'filters': {}, 'throttling': {}}
        strategy = MomentumIntraday(config)
        
        market_data = {
            'ema_fast': 52,
            'ema_slow': 51,
            'rsi': 55,
            'adx': 30,
            'volume_percentile': 70,
            'price': 50000
        }
        signal = strategy.generate_signal(market_data)
        assert signal is None
    
    def test_get_status(self):
        status = self.strategy.get_status()
        
        assert status['ema_fast_period'] == 9
        assert status['ema_slow_period'] == 21
        assert status['min_adx'] == 25
        assert status['holding_minutes'] == 60
        assert 'throttler' in status
    
    def test_confidence_based_on_adx(self):
        market_data_1 = {
            'ema_fast': 50,
            'ema_slow': 51,
            'rsi': 55,
            'adx': 50,  # High ADX
            'volume_percentile': 70,
            'price': 50000
        }
        self.strategy.generate_signal(market_data_1)
        
        market_data_2 = {
            'ema_fast': 52,
            'ema_slow': 51,
            'rsi': 55,
            'adx': 50,
            'volume_percentile': 70,
            'price': 50100
        }
        signal = self.strategy.generate_signal(market_data_2)
        
        # ADX 50 should give max confidence
        assert signal.confidence == 1.0


class TestMomentumIntradayDefaultConfig:
    """Test with default config values."""
    
    def test_default_values(self):
        config = {'enabled': True, 'params': {}, 'filters': {}, 'throttling': {}}
        strategy = MomentumIntraday(config)
        
        assert strategy.ema_fast == 9
        assert strategy.ema_slow == 21
        assert strategy.rsi_entry_threshold == 50
        assert strategy.holding_minutes == 60
