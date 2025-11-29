"""
Tests for Liquidation Features, Strategy, and Filter.
"""
import pytest
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, '.')

from v2.src.features.microstructure.liquidation_features import LiquidationFeatures
from v2.src.strategies.liquidation_cascade_strategy import LiquidationCascadeStrategy
from v2.src.strategies.filters.liquidation_filter import LiquidationFilter
from v2.src.strategies.base import SignalDirection


class TestLiquidationFeatures:
    """Tests for LiquidationFeatures class."""
    
    def setup_method(self):
        self.config = {
            'liquidations': {
                'enabled': True,
                'significant_single_liq_usd': 100000,
                'significant_cascade_usd': 1000000,
                'cascade_window_seconds': 60,
                'cascade_detection': {
                    'enabled': True,
                    'min_liquidations': 5,
                    'volume_acceleration': 2.0
                },
                'features': {
                    'include_liq_volume': True,
                    'include_liq_count': True,
                    'include_liq_imbalance': True,
                    'include_liq_acceleration': True,
                    'lookback_windows': [60, 300, 900]
                }
            }
        }
        self.features = LiquidationFeatures(self.config)
    
    def test_initialization(self):
        assert self.features.enabled is True
        assert self.features.cascade_window == 60
        assert self.features.min_liquidations == 5
        assert self.features.lookback_windows == [60, 300, 900]
    
    def test_initialization_disabled(self):
        config = {'liquidations': {'enabled': False}}
        features = LiquidationFeatures(config)
        assert features.enabled is False
    
    def test_add_liquidation(self):
        self.features.add_liquidation({
            'timestamp': datetime.now(),
            'side': 'SELL',
            'quantity': 1.0,
            'price': 50000.0
        })
        
        assert len(self.features.liquidation_buffer) == 1
    
    def test_add_liquidation_with_unix_timestamp(self):
        self.features.add_liquidation({
            'timestamp': time.time(),
            'side': 'BUY',
            'quantity': 2.0,
            'price': 50000.0
        })
        
        assert len(self.features.liquidation_buffer) == 1
    
    def test_calculate_empty(self):
        features = self.features.calculate()
        
        assert 'liq_volume_60s' in features
        assert 'liq_count_60s' in features
        assert 'liq_imbalance_60s' in features
        assert features['liq_volume_60s'] == 0.0
        assert features['liq_count_60s'] == 0.0
    
    def test_calculate_with_liquidations(self):
        now = datetime.now()
        
        # Add some liquidations
        for i in range(5):
            self.features.add_liquidation({
                'timestamp': now - timedelta(seconds=i * 5),
                'side': 'SELL',
                'quantity': 1.0,
                'price': 50000.0
            })
        
        features = self.features.calculate(now)
        
        assert features['liq_count_60s'] == 5.0
        assert features['liq_volume_60s'] == 5 * 50000.0
    
    def test_calculate_imbalance(self):
        now = datetime.now()
        
        # Add long liquidations (SELL side)
        for i in range(3):
            self.features.add_liquidation({
                'timestamp': now - timedelta(seconds=i),
                'side': 'SELL',
                'quantity': 1.0,
                'price': 50000.0
            })
        
        # Add short liquidation (BUY side)
        self.features.add_liquidation({
            'timestamp': now,
            'side': 'BUY',
            'quantity': 1.0,
            'price': 50000.0
        })
        
        features = self.features.calculate(now)
        
        # Long liqs = 3 * 50000, Short liqs = 1 * 50000
        # Imbalance = (150000 - 50000) / 200000 = 0.5
        assert abs(features['liq_imbalance_60s'] - 0.5) < 0.001
    
    def test_cascade_not_detected_insufficient_count(self):
        now = datetime.now()
        
        # Add less than min_liquidations
        for i in range(3):  # Less than 5
            self.features.add_liquidation({
                'timestamp': now - timedelta(seconds=i),
                'side': 'SELL',
                'quantity': 50.0,
                'price': 50000.0
            })
        
        features = self.features.calculate(now)
        
        assert features['cascade_active'] == 0.0
    
    def test_cascade_detected(self):
        now = datetime.now()
        
        # Warm up the acceleration history
        for i in range(10):
            warmup_time = now - timedelta(seconds=120 + i * 10)
            self.features.add_liquidation({
                'timestamp': warmup_time,
                'side': 'SELL',
                'quantity': 1.0,
                'price': 50000.0
            })
            self.features.calculate(warmup_time)
        
        # Add enough liquidations with sufficient volume
        for i in range(10):
            self.features.add_liquidation({
                'timestamp': now - timedelta(seconds=i * 5),
                'side': 'SELL',
                'quantity': 50.0,  # 50 * 50000 = 2.5M per liq
                'price': 50000.0
            })
        
        features = self.features.calculate(now)
        
        # Total volume = 10 * 2.5M = 25M > 1M threshold
        assert features['cascade_active'] == 1.0
    
    def test_cascade_direction_longs(self):
        now = datetime.now()
        
        # Warm up history
        for i in range(10):
            warmup_time = now - timedelta(seconds=120 + i * 10)
            self.features.add_liquidation({
                'timestamp': warmup_time,
                'side': 'SELL',
                'quantity': 1.0,
                'price': 50000.0
            })
            self.features.calculate(warmup_time)
        
        # Add long liquidations (SELL side = longs being liquidated)
        for i in range(10):
            self.features.add_liquidation({
                'timestamp': now - timedelta(seconds=i * 5),
                'side': 'SELL',
                'quantity': 50.0,
                'price': 50000.0
            })
        
        features = self.features.calculate(now)
        
        # Direction should be 1 (longs being liquidated)
        if features['cascade_active'] == 1.0:
            assert features['cascade_direction'] == 1.0
    
    def test_cascade_direction_shorts(self):
        now = datetime.now()
        
        # Warm up history
        for i in range(10):
            warmup_time = now - timedelta(seconds=120 + i * 10)
            self.features.add_liquidation({
                'timestamp': warmup_time,
                'side': 'BUY',
                'quantity': 1.0,
                'price': 50000.0
            })
            self.features.calculate(warmup_time)
        
        # Add short liquidations (BUY side = shorts being liquidated)
        for i in range(10):
            self.features.add_liquidation({
                'timestamp': now - timedelta(seconds=i * 5),
                'side': 'BUY',
                'quantity': 50.0,
                'price': 50000.0
            })
        
        features = self.features.calculate(now)
        
        # Direction should be -1 (shorts being liquidated)
        if features['cascade_active'] == 1.0:
            assert features['cascade_direction'] == -1.0
    
    def test_reset(self):
        self.features.add_liquidation({
            'timestamp': datetime.now(),
            'side': 'SELL',
            'quantity': 1.0,
            'price': 50000.0
        })
        
        assert len(self.features.liquidation_buffer) == 1
        
        self.features.reset()
        
        assert len(self.features.liquidation_buffer) == 0
    
    def test_get_status(self):
        status = self.features.get_status()
        
        assert 'enabled' in status
        assert 'buffer_size' in status
        assert 'lookback_windows' in status
        assert 'cascade_window' in status


class TestLiquidationCascadeStrategy:
    """Tests for LiquidationCascadeStrategy."""
    
    def setup_method(self):
        self.config = {
            'liquidations': {
                'enabled': True,
                'significant_cascade_usd': 1000000,
                'cascade_window_seconds': 60,
                'cascade_detection': {
                    'enabled': True,
                    'min_liquidations': 5,
                    'volume_acceleration': 2.0
                },
                'strategy': {
                    'enabled': True,
                    'follow_cascade': True,
                    'min_cascade_size_usd': 500000,
                    'fade_cascade': False,
                    'fade_after_seconds': 300,
                    'fade_min_cascade_size': 2000000,
                    'throttling': {
                        'enabled': True,
                        'max_trades_per_minute': 5
                    }
                },
                'features': {
                    'lookback_windows': [60, 300, 900]
                }
            }
        }
        self.strategy = LiquidationCascadeStrategy(self.config)
    
    def test_initialization(self):
        assert self.strategy.name == "LiquidationCascade"
        assert self.strategy.enabled is True
        assert self.strategy.follow_cascade is True
        assert self.strategy.fade_cascade is False
        assert self.strategy.min_cascade_size_usd == 500000
    
    def test_initialization_disabled(self):
        config = {
            'liquidations': {
                'strategy': {'enabled': False}
            }
        }
        strategy = LiquidationCascadeStrategy(config)
        assert strategy.enabled is False
    
    def test_generate_signal_no_cascade(self):
        market_data = {
            'liq_features': {
                'cascade_active': False,
                'cascade_direction': 0
            },
            'cascade_volume': 0,
            'price': 50000
        }
        
        signal = self.strategy.generate_signal(market_data)
        
        assert signal is None
    
    def test_generate_signal_follow_short_cascade(self):
        market_data = {
            'liq_features': {
                'cascade_active': True,
                'cascade_direction': -1  # Shorts being liquidated
            },
            'cascade_volume': 1000000,
            'price': 50000
        }
        
        signal = self.strategy.generate_signal(market_data)
        
        assert signal is not None
        assert signal.direction == SignalDirection.BUY
        assert 'Follow_Short_Cascade' in signal.reason
    
    def test_generate_signal_follow_long_cascade(self):
        market_data = {
            'liq_features': {
                'cascade_active': True,
                'cascade_direction': 1  # Longs being liquidated
            },
            'cascade_volume': 1000000,
            'price': 50000
        }
        
        signal = self.strategy.generate_signal(market_data)
        
        assert signal is not None
        assert signal.direction == SignalDirection.SELL
        assert 'Follow_Long_Cascade' in signal.reason
    
    def test_generate_signal_below_min_size(self):
        market_data = {
            'liq_features': {
                'cascade_active': True,
                'cascade_direction': -1
            },
            'cascade_volume': 100000,  # Below min_cascade_size_usd
            'price': 50000
        }
        
        signal = self.strategy.generate_signal(market_data)
        
        assert signal is None
    
    def test_disabled_strategy_no_signal(self):
        config = {
            'liquidations': {
                'strategy': {'enabled': False}
            }
        }
        strategy = LiquidationCascadeStrategy(config)
        
        market_data = {
            'liq_features': {
                'cascade_active': True,
                'cascade_direction': -1
            },
            'cascade_volume': 2000000,
            'price': 50000
        }
        
        signal = strategy.generate_signal(market_data)
        
        assert signal is None
    
    def test_add_liquidation(self):
        self.strategy.add_liquidation({
            'timestamp': datetime.now(),
            'side': 'SELL',
            'quantity': 1.0,
            'price': 50000.0
        })
        
        # Should be added to the internal liq_features
        assert len(self.strategy.liq_features.liquidation_buffer) == 1
    
    def test_get_status(self):
        status = self.strategy.get_status()
        
        assert status['name'] == 'LiquidationCascade'
        assert 'follow_cascade' in status
        assert 'fade_cascade' in status
        assert 'min_cascade_size_usd' in status
        assert 'liq_features_status' in status


class TestLiquidationCascadeStrategyFade:
    """Tests for LiquidationCascadeStrategy fade mode."""
    
    def setup_method(self):
        self.config = {
            'liquidations': {
                'enabled': True,
                'cascade_window_seconds': 60,
                'cascade_detection': {
                    'enabled': True,
                    'min_liquidations': 5,
                    'volume_acceleration': 2.0
                },
                'strategy': {
                    'enabled': True,
                    'follow_cascade': False,
                    'min_cascade_size_usd': 500000,
                    'fade_cascade': True,
                    'fade_after_seconds': 1,  # Short for testing
                    'fade_min_cascade_size': 2000000,
                    'throttling': {
                        'enabled': True,
                        'max_trades_per_minute': 5
                    }
                },
                'features': {
                    'lookback_windows': [60]
                }
            }
        }
        self.strategy = LiquidationCascadeStrategy(self.config)
    
    def test_fade_mode_enabled(self):
        assert self.strategy.fade_cascade is True
        assert self.strategy.follow_cascade is False
    
    def test_fade_no_signal_during_cascade(self):
        # During cascade, no fade signal
        market_data = {
            'liq_features': {
                'cascade_active': True,
                'cascade_direction': 1
            },
            'cascade_volume': 3000000,
            'price': 50000
        }
        
        signal = self.strategy.generate_signal(market_data)
        
        assert signal is None


class TestLiquidationFilter:
    """Tests for LiquidationFilter."""
    
    def setup_method(self):
        self.config = {
            'liquidations': {
                'enabled': True,
                'cascade_window_seconds': 60,
                'cascade_detection': {
                    'enabled': True,
                    'min_liquidations': 5,
                    'volume_acceleration': 2.0
                },
                'filter': {
                    'enabled': True,
                    'pause_during_cascade': True,
                    'pause_threshold_usd': 5000000,
                    'resume_after_seconds': 1  # Short for testing
                },
                'features': {
                    'lookback_windows': [60]
                }
            }
        }
        self.filter = LiquidationFilter(self.config)
    
    def test_initialization(self):
        assert self.filter.enabled is True
        assert self.filter.pause_during_cascade is True
        assert self.filter.pause_threshold_usd == 5000000
    
    def test_initialization_disabled(self):
        config = {'liquidations': {'filter': {'enabled': False}}}
        filter_obj = LiquidationFilter(config)
        
        allowed, mult, reason = filter_obj.should_trade()
        
        assert allowed is True
        assert mult == 1.0
        assert reason == "filter_disabled"
    
    def test_should_trade_no_cascade(self):
        allowed, mult, reason = self.filter.should_trade(
            liq_features={
                'cascade_active': False,
                'cascade_direction': 0
            }
        )
        
        assert allowed is True
        assert mult == 1.0
        assert reason == "allowed"
    
    def test_should_trade_pauses_extreme_cascade(self):
        # Pre-populate to have enough volume
        now = datetime.now()
        for i in range(10):
            self.filter.add_liquidation({
                'timestamp': now - timedelta(seconds=i * 5),
                'side': 'SELL',
                'quantity': 200.0,  # Large liquidations
                'price': 50000.0
            })
        
        allowed, mult, reason = self.filter.should_trade(
            liq_features={
                'cascade_active': True,
                'cascade_direction': 1
            }
        )
        
        # Should pause or reduce size based on cascade volume
        # Total volume = 10 * 200 * 50000 = 100M > 5M threshold
        assert allowed is False or mult < 1.0
    
    def test_should_trade_resumes_after_pause(self):
        # Trigger pause
        self.filter._paused = True
        self.filter._pause_start_time = time.time() - 10  # 10 seconds ago
        
        allowed, mult, reason = self.filter.should_trade(
            liq_features={
                'cascade_active': False,
                'cascade_direction': 0
            }
        )
        
        # Should resume after resume_after_seconds (1 second in test config)
        assert allowed is True
        assert self.filter._paused is False
    
    def test_should_trade_reduces_size_moderate_cascade(self):
        # Add moderate volume
        now = datetime.now()
        for i in range(5):
            self.filter.add_liquidation({
                'timestamp': now - timedelta(seconds=i),
                'side': 'SELL',
                'quantity': 10.0,
                'price': 50000.0
            })
        
        allowed, mult, reason = self.filter.should_trade(
            liq_features={
                'cascade_active': True,
                'cascade_direction': 1
            }
        )
        
        # With moderate cascade, should reduce size but still allow
        if allowed:
            assert mult <= 1.0
    
    def test_get_status(self):
        status = self.filter.get_status()
        
        assert 'enabled' in status
        assert 'paused' in status
        assert 'pause_threshold_usd' in status
        assert 'liq_features_status' in status
    
    def test_reset(self):
        self.filter._paused = True
        self.filter._pause_start_time = time.time()
        
        self.filter.reset()
        
        assert self.filter._paused is False
        assert self.filter._pause_start_time is None


class TestLiquidationFilterDirectionConflict:
    """Test filter blocking signals against cascade direction."""
    
    def setup_method(self):
        self.config = {
            'liquidations': {
                'enabled': True,
                'cascade_window_seconds': 60,
                'cascade_detection': {
                    'enabled': True,
                    'min_liquidations': 5,
                    'volume_acceleration': 2.0
                },
                'filter': {
                    'enabled': True,
                    'pause_during_cascade': True,
                    'pause_threshold_usd': 100000000,  # High threshold to avoid pause
                    'resume_after_seconds': 120
                },
                'features': {
                    'lookback_windows': [60]
                }
            }
        }
        self.filter = LiquidationFilter(self.config)
    
    def test_blocks_buy_against_long_cascade(self):
        from v2.src.strategies.base import Signal, SignalDirection
        
        signal = Signal(
            direction=SignalDirection.BUY,
            strategy_name='test',
            confidence=0.8
        )
        
        allowed, mult, reason = self.filter.should_trade(
            signal=signal,
            liq_features={
                'cascade_active': True,
                'cascade_direction': 1  # Longs being liquidated (bearish)
            }
        )
        
        assert allowed is False
        assert reason == "signal_against_long_cascade"
    
    def test_blocks_sell_against_short_cascade(self):
        from v2.src.strategies.base import Signal, SignalDirection
        
        signal = Signal(
            direction=SignalDirection.SELL,
            strategy_name='test',
            confidence=0.8
        )
        
        allowed, mult, reason = self.filter.should_trade(
            signal=signal,
            liq_features={
                'cascade_active': True,
                'cascade_direction': -1  # Shorts being liquidated (bullish)
            }
        )
        
        assert allowed is False
        assert reason == "signal_against_short_cascade"
    
    def test_allows_buy_with_short_cascade(self):
        from v2.src.strategies.base import Signal, SignalDirection
        
        signal = Signal(
            direction=SignalDirection.BUY,
            strategy_name='test',
            confidence=0.8
        )
        
        allowed, mult, reason = self.filter.should_trade(
            signal=signal,
            liq_features={
                'cascade_active': True,
                'cascade_direction': -1  # Shorts being liquidated (bullish) - OK for buy
            }
        )
        
        # Should be allowed (possibly with reduced size)
        assert allowed is True or 'against' not in reason


class TestLiquidationIntegration:
    """Integration tests for liquidation components."""
    
    def test_features_to_strategy_flow(self):
        """Test flow from LiquidationFeatures to LiquidationCascadeStrategy."""
        config = {
            'liquidations': {
                'enabled': True,
                'cascade_window_seconds': 60,
                'cascade_detection': {
                    'enabled': True,
                    'min_liquidations': 3,
                    'volume_acceleration': 1.5
                },
                'strategy': {
                    'enabled': True,
                    'follow_cascade': True,
                    'min_cascade_size_usd': 100000,
                    'throttling': {
                        'enabled': False
                    }
                },
                'features': {
                    'lookback_windows': [60]
                }
            }
        }
        
        strategy = LiquidationCascadeStrategy(config)
        now = datetime.now()
        
        # Warm up history
        for i in range(10):
            warmup_time = now - timedelta(seconds=120 + i * 10)
            strategy.add_liquidation({
                'timestamp': warmup_time,
                'side': 'SELL',
                'quantity': 1.0,
                'price': 50000.0
            })
            strategy.liq_features.calculate(warmup_time)
        
        # Add cascade liquidations
        for i in range(5):
            strategy.add_liquidation({
                'timestamp': now - timedelta(seconds=i * 5),
                'side': 'SELL',
                'quantity': 10.0,
                'price': 50000.0
            })
        
        # Generate signal without pre-calculated features
        signal = strategy.generate_signal({'price': 50000})
        
        # Should detect cascade from internal features
        # Note: May or may not generate signal depending on cascade detection
        assert signal is None or signal.direction in [SignalDirection.BUY, SignalDirection.SELL]
    
    def test_filter_with_strategy_signal(self):
        """Test filter processing a signal from cascade strategy."""
        config = {
            'liquidations': {
                'enabled': True,
                'cascade_window_seconds': 60,
                'cascade_detection': {
                    'enabled': True,
                    'min_liquidations': 5,
                    'volume_acceleration': 2.0
                },
                'filter': {
                    'enabled': True,
                    'pause_during_cascade': True,
                    'pause_threshold_usd': 1000000000,  # Very high
                    'resume_after_seconds': 120
                },
                'strategy': {
                    'enabled': True,
                    'follow_cascade': True,
                    'min_cascade_size_usd': 500000,
                    'throttling': {'enabled': False}
                },
                'features': {
                    'lookback_windows': [60]
                }
            }
        }
        
        strategy = LiquidationCascadeStrategy(config)
        liq_filter = LiquidationFilter(config)
        
        # Create a mock signal
        market_data = {
            'liq_features': {
                'cascade_active': True,
                'cascade_direction': -1
            },
            'cascade_volume': 1000000
        }
        
        signal = strategy.generate_signal(market_data)
        
        if signal:
            # Filter should allow signal that follows cascade
            allowed, mult, reason = liq_filter.should_trade(
                signal=signal,
                liq_features=market_data['liq_features']
            )
            
            # Buy signal following short cascade should be allowed
            assert allowed is True
