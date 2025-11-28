import pytest
import sys
sys.path.insert(0, '.')

from src.execution.position_manager import PositionManager


class TestPositionManager:
    def setup_method(self):
        self.pm = PositionManager(
            default_sl_percent=0.01,
            default_tp_percent=0.02,
            use_trailing_stop=False
        )

    def test_initialization(self):
        assert self.pm.default_sl_percent == 0.01
        assert self.pm.default_tp_percent == 0.02
        assert self.pm.has_position() is False

    def test_calculate_sl_tp_long(self):
        sl, tp = self.pm.calculate_sl_tp(50000, 'BUY')
        assert sl == 49500  # -1%
        assert tp == 51000  # +2%

    def test_calculate_sl_tp_short(self):
        sl, tp = self.pm.calculate_sl_tp(50000, 'SELL')
        assert sl == 50500  # +1%
        assert tp == 49000  # -2%

    def test_open_position(self):
        pos = self.pm.open_position('BTCUSDT', 'BUY', 50000, 0.1)
        assert pos is not None
        assert self.pm.has_position() is True
        assert pos.side == 'LONG'

    def test_cannot_open_duplicate(self):
        self.pm.open_position('BTCUSDT', 'BUY', 50000, 0.1)
        pos2 = self.pm.open_position('BTCUSDT', 'BUY', 51000, 0.1)
        assert pos2 is None

    def test_close_position_profit(self):
        self.pm.open_position('BTCUSDT', 'BUY', 50000, 0.1)
        pnl = self.pm.close_position(51000)
        assert pnl == 100  # (51000-50000) * 0.1
        assert self.pm.has_position() is False

    def test_close_position_loss(self):
        self.pm.open_position('BTCUSDT', 'BUY', 50000, 0.1)
        pnl = self.pm.close_position(49000)
        assert pnl == -100

    def test_should_close_stop_loss(self):
        self.pm.open_position('BTCUSDT', 'BUY', 50000, 0.1, 49500, 51000)
        result = self.pm.should_close(49400)
        assert result == 'STOP_LOSS'

    def test_should_close_take_profit(self):
        self.pm.open_position('BTCUSDT', 'BUY', 50000, 0.1, 49500, 51000)
        result = self.pm.should_close(51100)
        assert result == 'TAKE_PROFIT'

    def test_close_position_no_division_by_zero(self):
        """Test that closing position with zero entry price returns None (rejected)."""
        # Manually set a position with zero entry price (edge case)
        from src.execution.position_manager import Position
        from datetime import datetime
        self.pm.current_position = Position(
            symbol='BTCUSDT',
            side='LONG',
            entry_price=0.0,  # Invalid entry price
            quantity=0.1,
            stop_loss=0.0,
            take_profit=0.0,
            entry_time=datetime.now()
        )
        # Should return None because entry price is invalid
        pnl = self.pm.close_position(50000)
        assert pnl is None  # Rejected due to invalid entry price
        assert self.pm.has_position() is True  # Position still exists because close was rejected

    def test_close_position_zero_quantity_no_crash(self):
        """Test that closing position with zero quantity does not crash."""
        from src.execution.position_manager import Position
        from datetime import datetime
        self.pm.current_position = Position(
            symbol='BTCUSDT',
            side='LONG',
            entry_price=50000,
            quantity=0.0,  # Invalid but should not crash
            stop_loss=49500,
            take_profit=51000,
            entry_time=datetime.now()
        )
        # Should not raise an exception
        pnl = self.pm.close_position(51000)
        assert pnl == 0  # No quantity means no PNL
        assert self.pm.has_position() is False

    def test_sync_position(self):
        """Test the sync_position method."""
        self.pm.sync_position(
            symbol='BTCUSDT',
            side='LONG',
            entry_price=50000,
            quantity=0.1,
            unrealized_pnl=100
        )
        assert self.pm.has_position() is True
        assert self.pm.current_position.side == 'LONG'
        assert self.pm.current_position.entry_price == 50000
        assert self.pm.current_position.quantity == 0.1

    def test_clear_position(self):
        """Test the clear_position method."""
        self.pm.open_position('BTCUSDT', 'BUY', 50000, 0.1)
        assert self.pm.has_position() is True
        self.pm.clear_position()
        assert self.pm.has_position() is False

    # New tests for price validation (BUG 1 fixes)
    
    def test_close_position_with_none_price(self):
        """Test that closing position with None price is rejected."""
        self.pm.open_position('BTCUSDT', 'BUY', 50000, 0.1)
        pnl = self.pm.close_position(None)
        assert pnl is None  # Rejected
        assert self.pm.has_position() is True  # Position still exists

    def test_close_position_with_zero_price(self):
        """Test that closing position with zero price is rejected."""
        self.pm.open_position('BTCUSDT', 'BUY', 50000, 0.1)
        pnl = self.pm.close_position(0)
        assert pnl is None  # Rejected
        assert self.pm.has_position() is True  # Position still exists

    def test_close_position_with_negative_price(self):
        """Test that closing position with negative price is rejected."""
        self.pm.open_position('BTCUSDT', 'BUY', 50000, 0.1)
        pnl = self.pm.close_position(-100)
        assert pnl is None  # Rejected
        assert self.pm.has_position() is True  # Position still exists

    def test_close_position_with_suspicious_price(self):
        """Test that closing position with suspicious price (>50% deviation) is rejected."""
        self.pm.open_position('BTCUSDT', 'BUY', 50000, 0.1)
        # Set last valid price
        self.pm.last_valid_price = 50000
        # Try to close with a price that is 60% different
        pnl = self.pm.close_position(20000)  # 60% lower
        assert pnl is None  # Rejected
        assert self.pm.has_position() is True  # Position still exists

    def test_should_close_with_invalid_price(self):
        """Test that should_close ignores invalid prices."""
        self.pm.open_position('BTCUSDT', 'BUY', 50000, 0.1, 49500, 51000)
        
        # None price should be ignored
        assert self.pm.should_close(None) is None
        
        # Zero price should be ignored
        assert self.pm.should_close(0) is None
        
        # Negative price should be ignored
        assert self.pm.should_close(-100) is None

    def test_is_valid_price(self):
        """Test the is_valid_price helper method."""
        assert self.pm.is_valid_price(50000) is True
        assert self.pm.is_valid_price(0.01) is True
        assert self.pm.is_valid_price(0) is False
        assert self.pm.is_valid_price(-100) is False
        assert self.pm.is_valid_price(None) is False

    def test_update_last_valid_price(self):
        """Test that last_valid_price is updated correctly."""
        assert self.pm.last_valid_price is None
        
        # Valid price updates the cache
        self.pm.update_last_valid_price(50000)
        assert self.pm.last_valid_price == 50000
        
        # Invalid prices don't update
        self.pm.update_last_valid_price(0)
        assert self.pm.last_valid_price == 50000
        
        self.pm.update_last_valid_price(None)
        assert self.pm.last_valid_price == 50000

    def test_should_close_updates_last_valid_price(self):
        """Test that should_close updates last_valid_price."""
        self.pm.open_position('BTCUSDT', 'BUY', 50000, 0.1, 49500, 51000)
        
        # should_close with valid price should update last_valid_price
        self.pm.should_close(50500)
        assert self.pm.last_valid_price == 50500
