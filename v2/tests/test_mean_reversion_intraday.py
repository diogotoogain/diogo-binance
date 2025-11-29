"""
Tests for Mean Reversion Intraday Strategy
"""
import pytest
import sys

sys.path.insert(0, '.')

from v2.src.strategies.mean_reversion_intraday import MeanReversionIntraday
from v2.src.strategies.base import SignalDirection


class TestMeanReversionIntraday:
    """Tests for Mean Reversion Intraday strategy."""
    
    def setup_method(self):
        self.config = {
            'enabled': True,
            'params': {
                'zscore_entry': 2.0,
                'zscore_exit': 0.5,
                'bollinger_entry_std': 2.0,
                'holding_minutes': 30
            },
            'filters': {
                'adx_filter': {
                    'enabled': True,
                    'max_adx': 20
                },
                'liquidity_cluster_filter': {
                    'enabled': True,
                    'proximity_threshold_pct': 0.5
                }
            },
            'throttling': {
                'enabled': True,
                'max_trades_per_hour': 6
            }
        }
        self.strategy = MeanReversionIntraday(self.config)
    
    def test_initialization(self):
        assert self.strategy.name == "MeanReversionIntraday"
        assert self.strategy.enabled is True
        assert self.strategy.zscore_entry == 2.0
        assert self.strategy.zscore_exit == 0.5
        assert self.strategy.max_adx == 20
        assert self.strategy.holding_minutes == 30
    
    def test_buy_signal_negative_zscore(self):
        market_data = {
            'zscore': -2.5,  # Below -zscore_entry
            'bb_upper': 51000,
            'bb_lower': 49000,
            'bb_middle': 50000,
            'adx': 15,
            'liquidity_proximity_pct': 0.3,
            'price': 48900  # Below BB lower
        }
        
        signal = self.strategy.generate_signal(market_data)
        
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert 'MeanRev_Buy' in signal.reason
    
    def test_sell_signal_positive_zscore(self):
        market_data = {
            'zscore': 2.5,  # Above zscore_entry
            'bb_upper': 51000,
            'bb_lower': 49000,
            'bb_middle': 50000,
            'adx': 15,
            'liquidity_proximity_pct': 0.3,
            'price': 51100  # Above BB upper
        }
        
        signal = self.strategy.generate_signal(market_data)
        
        assert signal is not None
        assert signal.direction == SignalDirection.SELL
        assert 'MeanRev_Sell' in signal.reason
    
    def test_no_signal_zscore_within_range(self):
        market_data = {
            'zscore': 1.0,  # Within range
            'bb_upper': 51000,
            'bb_lower': 49000,
            'bb_middle': 50000,
            'adx': 15,
            'price': 50000
        }
        
        signal = self.strategy.generate_signal(market_data)
        assert signal is None
    
    def test_adx_filter_blocks(self):
        market_data = {
            'zscore': -2.5,
            'bb_upper': 51000,
            'bb_lower': 49000,
            'bb_middle': 50000,
            'adx': 25,  # Above max_adx (trending market)
            'price': 48900
        }
        
        signal = self.strategy.generate_signal(market_data)
        assert signal is None
    
    def test_bollinger_confirmation_buy(self):
        # Price below BB lower = BB confirms buy
        market_data = {
            'zscore': -2.5,
            'bb_upper': 51000,
            'bb_lower': 49000,
            'bb_middle': 50000,
            'adx': 15,
            'liquidity_proximity_pct': 0.3,
            'price': 48900  # Below BB lower
        }
        
        signal = self.strategy.generate_signal(market_data)
        
        assert signal is not None
        assert signal.metadata['bb_confirmed'] is True
    
    def test_bollinger_no_confirmation(self):
        # Price not at BB extremes
        market_data = {
            'zscore': -2.5,
            'bb_upper': 51000,
            'bb_lower': 49000,
            'bb_middle': 50000,
            'adx': 15,
            'liquidity_proximity_pct': 0.3,
            'price': 49500  # Not below BB lower
        }
        
        signal = self.strategy.generate_signal(market_data)
        
        assert signal is not None
        assert signal.metadata['bb_confirmed'] is False
    
    def test_liquidity_support_increases_confidence(self):
        # With liquidity support
        market_data_with = {
            'zscore': -2.5,
            'bb_upper': 51000,
            'bb_lower': 49000,
            'bb_middle': 50000,
            'adx': 15,
            'liquidity_proximity_pct': 0.3,  # Within threshold
            'price': 48900
        }
        signal_with = self.strategy.generate_signal(market_data_with)
        
        # Reset throttler
        self.strategy.throttler.reset()
        
        # Without liquidity support
        market_data_without = {
            'zscore': -2.5,
            'bb_upper': 51000,
            'bb_lower': 49000,
            'bb_middle': 50000,
            'adx': 15,
            'liquidity_proximity_pct': 2.0,  # Far from support
            'price': 48900
        }
        signal_without = self.strategy.generate_signal(market_data_without)
        
        # Signal with support should have higher confidence
        assert signal_with.confidence >= signal_without.confidence
    
    def test_disabled_strategy(self):
        config = {'enabled': False, 'params': {}, 'filters': {}, 'throttling': {}}
        strategy = MeanReversionIntraday(config)
        
        market_data = {
            'zscore': -2.5,
            'bb_upper': 51000,
            'bb_lower': 49000,
            'bb_middle': 50000,
            'adx': 15,
            'price': 48900
        }
        signal = strategy.generate_signal(market_data)
        assert signal is None
    
    def test_get_status(self):
        status = self.strategy.get_status()
        
        assert status['zscore_entry'] == 2.0
        assert status['zscore_exit'] == 0.5
        assert status['max_adx'] == 20
        assert status['holding_minutes'] == 30
        assert 'throttler' in status
    
    def test_signal_metadata(self):
        market_data = {
            'zscore': -2.5,
            'bb_upper': 51000,
            'bb_lower': 49000,
            'bb_middle': 50000,
            'adx': 15,
            'liquidity_proximity_pct': 0.3,
            'price': 48900
        }
        
        signal = self.strategy.generate_signal(market_data)
        
        assert 'zscore' in signal.metadata
        assert 'bb_upper' in signal.metadata
        assert 'bb_lower' in signal.metadata
        assert 'holding_minutes' in signal.metadata


class TestMeanReversionIntradayDefaultConfig:
    """Test with default config values."""
    
    def test_default_values(self):
        config = {'enabled': True, 'params': {}, 'filters': {}, 'throttling': {}}
        strategy = MeanReversionIntraday(config)
        
        assert strategy.zscore_entry == 2.0
        assert strategy.zscore_exit == 0.5
        assert strategy.bollinger_entry_std == 2.0
        assert strategy.holding_minutes == 30
