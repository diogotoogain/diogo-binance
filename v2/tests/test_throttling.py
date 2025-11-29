"""
Tests for Throttling Module
"""
import pytest
import sys
import time

sys.path.insert(0, '.')

from v2.src.strategies.throttling import Throttler


class TestThrottler:
    """Tests for Throttler class."""
    
    def test_initialization_with_config(self):
        config = {
            'enabled': True,
            'max_trades_per_minute': 5,
            'max_trades_per_hour': 10,
            'max_trades_per_day': 50,
            'cooldown_after_loss_seconds': 30
        }
        throttler = Throttler(config)
        
        assert throttler.enabled is True
        assert throttler.max_per_minute == 5
        assert throttler.max_per_hour == 10
        assert throttler.max_per_day == 50
        assert throttler.cooldown_after_loss == 30
    
    def test_initialization_empty_config(self):
        throttler = Throttler({})
        
        assert throttler.enabled is True  # Default
        assert throttler.max_per_minute is None
        assert throttler.max_per_hour is None
        assert throttler.max_per_day is None
    
    def test_can_trade_disabled(self):
        throttler = Throttler({'enabled': False})
        
        # Should always return True when disabled
        for _ in range(100):
            throttler.record_trade()
        
        assert throttler.can_trade() is True
    
    def test_can_trade_per_minute_limit(self):
        config = {'enabled': True, 'max_trades_per_minute': 2}
        throttler = Throttler(config)
        
        assert throttler.can_trade() is True
        throttler.record_trade()
        
        assert throttler.can_trade() is True
        throttler.record_trade()
        
        # Third trade should be blocked
        assert throttler.can_trade() is False
    
    def test_can_trade_per_hour_limit(self):
        config = {'enabled': True, 'max_trades_per_hour': 2}
        throttler = Throttler(config)
        
        assert throttler.can_trade() is True
        throttler.record_trade()
        throttler.record_trade()
        
        assert throttler.can_trade() is False
    
    def test_can_trade_per_day_limit(self):
        config = {'enabled': True, 'max_trades_per_day': 2}
        throttler = Throttler(config)
        
        throttler.record_trade()
        throttler.record_trade()
        
        assert throttler.can_trade() is False
    
    def test_cooldown_after_loss(self):
        config = {'enabled': True, 'cooldown_after_loss_seconds': 60}
        throttler = Throttler(config)
        
        assert throttler.can_trade() is True
        
        # Record a loss
        throttler.record_trade(is_loss=True)
        
        # Should be in cooldown
        assert throttler.can_trade() is False
        assert throttler.get_cooldown_remaining() > 0
    
    def test_cooldown_remaining(self):
        config = {'enabled': True, 'cooldown_after_loss_seconds': 60}
        throttler = Throttler(config)
        
        # No cooldown initially
        assert throttler.get_cooldown_remaining() == 0.0
        
        # After loss
        throttler.record_trade(is_loss=True)
        remaining = throttler.get_cooldown_remaining()
        assert remaining > 0
        assert remaining <= 60
    
    def test_get_remaining_trades(self):
        config = {
            'enabled': True,
            'max_trades_per_minute': 5,
            'max_trades_per_hour': 10,
            'max_trades_per_day': 20
        }
        throttler = Throttler(config)
        
        remaining = throttler.get_remaining_trades()
        assert remaining['minute'] == 5
        assert remaining['hour'] == 10
        assert remaining['day'] == 20
        
        # After one trade
        throttler.record_trade()
        remaining = throttler.get_remaining_trades()
        assert remaining['minute'] == 4
        assert remaining['hour'] == 9
        assert remaining['day'] == 19
    
    def test_reset(self):
        config = {
            'enabled': True,
            'max_trades_per_minute': 2,
            'cooldown_after_loss_seconds': 60
        }
        throttler = Throttler(config)
        
        throttler.record_trade()
        throttler.record_trade(is_loss=True)
        
        assert throttler.can_trade() is False
        
        throttler.reset()
        
        assert throttler.can_trade() is True
        assert len(throttler._trades) == 0
        assert throttler._last_loss_time is None
    
    def test_get_status(self):
        config = {
            'enabled': True,
            'max_trades_per_minute': 5
        }
        throttler = Throttler(config)
        throttler.record_trade()
        
        status = throttler.get_status()
        
        assert status['enabled'] is True
        assert 'remaining' in status
        assert status['total_trades_today'] == 1
        assert 'cooldown_remaining' in status


class TestThrottlerEdgeCases:
    """Edge case tests for Throttler."""
    
    def test_multiple_limits_combined(self):
        """Test that all limits are checked together."""
        config = {
            'enabled': True,
            'max_trades_per_minute': 100,
            'max_trades_per_hour': 2,  # This is the limiting factor
            'max_trades_per_day': 100
        }
        throttler = Throttler(config)
        
        throttler.record_trade()
        throttler.record_trade()
        
        # Should be blocked by hour limit
        assert throttler.can_trade() is False
    
    def test_cooldown_takes_priority(self):
        """Cooldown should block even if trade limits allow."""
        config = {
            'enabled': True,
            'max_trades_per_minute': 100,
            'cooldown_after_loss_seconds': 3600  # 1 hour
        }
        throttler = Throttler(config)
        
        throttler.record_trade(is_loss=True)
        
        # Even though minute limit allows, cooldown blocks
        assert throttler.can_trade() is False
