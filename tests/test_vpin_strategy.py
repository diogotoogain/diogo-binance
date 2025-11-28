import pytest
import sys
import time
sys.path.insert(0, '.')

from unittest.mock import patch, MagicMock
from src.strategies.vpin_strategy import VPINStrategy


class TestVPINStrategy:
    def setup_method(self):
        self.strategy = VPINStrategy(
            bucket_size_usd=100_000,
            n_buckets=50,
            vpin_threshold_high=0.7,
            vpin_threshold_low=0.3,
            signal_cooldown_seconds=30
        )

    def test_initialization(self):
        assert self.strategy.bucket_size_usd == 100_000
        assert self.strategy.n_buckets == 50
        assert self.strategy.vpin_threshold_high == 0.7
        assert self.strategy.current_vpin == 0.0

    def test_can_signal_respects_cooldown(self):
        """Test that _can_signal returns False during cooldown."""
        assert self.strategy._can_signal() is True
        
        # Simulate a signal was just sent
        self.strategy.last_signal_time = time.time()
        
        assert self.strategy._can_signal() is False
        
        # Simulate cooldown has passed
        self.strategy.last_signal_time = time.time() - 31
        
        assert self.strategy._can_signal() is True

    def test_complete_bucket_only_logs_vpin_when_can_signal(self):
        """Test that VPIN ALTO is only logged when signal can be generated (not during cooldown)."""
        # Fill enough buckets to calculate VPIN with high imbalance (90/10 = 0.8 VPIN)
        for i in range(15):
            self.strategy.buckets.append({
                'buy_volume': 90000,
                'sell_volume': 10000,
            })
        
        # Set up a high VPIN scenario in current bucket
        self.strategy.current_bucket_buy_volume = 90000
        self.strategy.current_bucket_sell_volume = 10000
        self.strategy.current_bucket_total_volume = 100000
        
        with patch.object(self.strategy, 'log') as mock_log:
            # During cooldown - should NOT log VPIN ALTO
            self.strategy.last_signal_time = time.time()  # Just signaled
            self.strategy._complete_bucket()
            
            # Check that VPIN ALTO was NOT logged (only bucket complete log)
            vpin_alto_calls = [call for call in mock_log.call_args_list 
                             if '🚨 VPIN ALTO' in str(call)]
            assert len(vpin_alto_calls) == 0

    def test_complete_bucket_logs_vpin_when_not_in_cooldown(self):
        """Test that VPIN ALTO is logged when not in cooldown period."""
        # Fill enough buckets to calculate VPIN with high imbalance (90/10 = 0.8 VPIN)
        for i in range(15):
            self.strategy.buckets.append({
                'buy_volume': 90000,
                'sell_volume': 10000,
            })
        
        # Set up a high VPIN scenario in current bucket
        self.strategy.current_bucket_buy_volume = 90000
        self.strategy.current_bucket_sell_volume = 10000
        self.strategy.current_bucket_total_volume = 100000
        
        with patch.object(self.strategy, 'log') as mock_log:
            # NOT in cooldown
            self.strategy.last_signal_time = time.time() - 35  # Signal was 35 seconds ago
            self.strategy._complete_bucket()
            
            # Check that VPIN ALTO WAS logged
            vpin_alto_calls = [call for call in mock_log.call_args_list 
                             if '🚨 VPIN ALTO' in str(call)]
            assert len(vpin_alto_calls) == 1

    def test_vpin_calculation(self):
        """Test that VPIN is calculated correctly."""
        # Add buckets with known imbalance
        for i in range(10):
            self.strategy.buckets.append({
                'buy_volume': 70000,
                'sell_volume': 30000,
            })
        
        vpin = self.strategy._calculate_vpin()
        
        # VPIN = Σ|V_buy - V_sell| / Σ(V_buy + V_sell)
        # = 10 * |70000 - 30000| / (10 * (70000 + 30000))
        # = 400000 / 1000000 = 0.4
        assert abs(vpin - 0.4) < 0.001

    def test_detect_direction_buy(self):
        """Test direction detection for BUY."""
        # Add 5 buckets where buy > sell
        for i in range(5):
            self.strategy.buckets.append({
                'buy_volume': 60000,
                'sell_volume': 40000,
            })
        
        direction = self.strategy._detect_direction()
        assert direction == "BUY"

    def test_detect_direction_sell(self):
        """Test direction detection for SELL."""
        # Add 5 buckets where sell > buy
        for i in range(5):
            self.strategy.buckets.append({
                'buy_volume': 40000,
                'sell_volume': 60000,
            })
        
        direction = self.strategy._detect_direction()
        assert direction == "SELL"

    def test_detect_direction_indefinido(self):
        """Test direction detection when mixed."""
        # Add mixed buckets
        self.strategy.buckets.append({'buy_volume': 60000, 'sell_volume': 40000})
        self.strategy.buckets.append({'buy_volume': 40000, 'sell_volume': 60000})
        self.strategy.buckets.append({'buy_volume': 60000, 'sell_volume': 40000})
        self.strategy.buckets.append({'buy_volume': 40000, 'sell_volume': 60000})
        self.strategy.buckets.append({'buy_volume': 50000, 'sell_volume': 50000})  # Equal
        
        direction = self.strategy._detect_direction()
        assert direction == "INDEFINIDO"
