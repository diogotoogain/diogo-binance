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
        assert can_open is True
        assert msg == "OK"

    def test_can_open_position_max_reached(self):
        can_open, msg = self.rm.can_open_position(1)
        assert can_open is False

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

    def test_position_size_limited_to_5_percent_of_balance(self):
        """Test that position size is limited to 5% of balance in USD."""
        # With balance of $10,000 and BTC at $50,000
        # Max USD value = $10,000 * 0.05 = $500
        # Max BTC from USD = $500 / $50,000 = 0.01 BTC
        size = self.rm.calculate_position_size(
            balance=10000,
            entry_price=50000,
            stop_loss_price=49000  # 2% stop loss, would normally allow larger size
        )
        # Maximum should be 0.01 BTC (5% of balance / price)
        max_expected = (10000 * 0.05) / 50000  # 0.01 BTC
        assert size <= max_expected
        assert size == 0.01  # Should be exactly 5% limit

    def test_position_size_zero_entry_price(self):
        """Test that zero entry price doesn't cause division by zero."""
        # Should return 0 when entry_price is 0
        size = self.rm.calculate_position_size(
            balance=10000,
            entry_price=0,  # Edge case
            stop_loss_price=0
        )
        # With zero price, max_btc_from_usd would be 0
        assert size == 0

    # New tests for PnL validation (BUG 1 fixes)

    def test_update_daily_pnl_with_none(self):
        """Test that update_daily_pnl rejects None values."""
        self.rm.set_initial_balance(10000)
        initial_pnl = self.rm.daily_pnl
        initial_trades = self.rm.trades_today
        
        self.rm.update_daily_pnl(None)
        
        # Should not update pnl or trades
        assert self.rm.daily_pnl == initial_pnl
        assert self.rm.trades_today == initial_trades

    def test_update_daily_pnl_with_suspicious_value(self):
        """Test that update_daily_pnl rejects suspicious values (> 100% of balance)."""
        self.rm.set_initial_balance(10000)
        initial_pnl = self.rm.daily_pnl
        initial_trades = self.rm.trades_today
        
        # Try to update with suspicious PnL (larger than entire balance)
        self.rm.update_daily_pnl(-15000)  # 150% loss, impossible
        
        # Should not update pnl or trades
        assert self.rm.daily_pnl == initial_pnl
        assert self.rm.trades_today == initial_trades

    def test_update_daily_pnl_valid_values(self):
        """Test that update_daily_pnl accepts valid values."""
        self.rm.set_initial_balance(10000)
        
        # Valid positive PnL
        self.rm.update_daily_pnl(100)
        assert self.rm.daily_pnl == 100
        assert self.rm.trades_today == 1
        
        # Valid negative PnL
        self.rm.update_daily_pnl(-50)
        assert self.rm.daily_pnl == 50
        assert self.rm.trades_today == 2
