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

    def test_close_position_zero_entry_price(self):
        """Test that division by zero is avoided when entry_price is 0."""
        # Manually set a position with entry_price = 0 to simulate the bug
        from src.execution.position_manager import Position
        from datetime import datetime
        self.pm.current_position = Position(
            symbol='BTCUSDT',
            side='LONG',
            entry_price=0.0,  # Bug scenario: zero entry price
            quantity=0.1,
            stop_loss=49500,
            take_profit=51000,
            entry_time=datetime.now()
        )
        # Should not raise ZeroDivisionError
        pnl = self.pm.close_position(50000)
        assert pnl == 50000 * 0.1  # pnl calculation still works
        assert self.pm.has_position() is False

    def test_close_position_zero_quantity(self):
        """Test that division by zero is avoided when quantity is 0."""
        from src.execution.position_manager import Position
        from datetime import datetime
        self.pm.current_position = Position(
            symbol='BTCUSDT',
            side='LONG',
            entry_price=50000,
            quantity=0.0,  # Bug scenario: zero quantity
            stop_loss=49500,
            take_profit=51000,
            entry_time=datetime.now()
        )
        # Should not raise ZeroDivisionError
        pnl = self.pm.close_position(51000)
        assert pnl == 0.0  # pnl is 0 because quantity is 0
        assert self.pm.has_position() is False
