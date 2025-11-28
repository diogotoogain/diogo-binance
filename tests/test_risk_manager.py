import pytest
import sys
sys.path.insert(0, '.')

from src.execution.risk_manager import RiskManager


class TestRiskManager:
    def setup_method(self):
        self.rm = RiskManager(
            risk_per_trade=0.01,
            max_position_size=0.05,
            max_daily_loss=0.03,
            max_concurrent_positions=1
        )

    def test_initialization(self):
        assert self.rm.risk_per_trade == 0.01
        assert self.rm.max_position_size == 0.05
        assert self.rm.max_daily_loss == 0.03

    def test_set_initial_balance(self):
        self.rm.set_initial_balance(10000)
        assert self.rm.initial_balance == 10000

    def test_calculate_position_size(self):
        size = self.rm.calculate_position_size(
            balance=10000,
            entry_price=50000,
            stop_loss_price=49500
        )
        assert size > 0
        assert size <= 10000 * 0.05 / 50000

    def test_can_open_position_ok(self):
        self.rm.set_initial_balance(10000)
        can_open, msg = self.rm.can_open_position(0)
        assert can_open == True
        assert msg == "OK"

    def test_can_open_position_max_reached(self):
        can_open, msg = self.rm.can_open_position(1)
        assert can_open == False

    def test_daily_stop_loss(self):
        self.rm.set_initial_balance(10000)
        self.rm.update_daily_pnl(-350)  # -3.5% > 3% limit
        can_open, msg = self.rm.can_open_position(0)
        assert can_open == False
        assert "STOP" in msg

    def test_reset_daily(self):
        self.rm.update_daily_pnl(-100)
        self.rm.reset_daily()
        assert self.rm.daily_pnl == 0
        assert self.rm.trades_today == 0
