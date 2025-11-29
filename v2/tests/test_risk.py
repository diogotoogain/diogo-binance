"""
Tests for V2 Risk Management module.

Tests cover:
- Kill Switch (CRITICAL)
- Daily Limits
- Position Sizer
- Stop Loss Calculator
- Take Profit Calculator
- Trailing Stop
- Regime Adjustment
- Risk Manager
"""
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import sys
from pathlib import Path

# Add v2/src to path
v2_src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(v2_src))

from risk.kill_switch import KillSwitch
from risk.daily_limits import DailyLimits
from risk.position_sizer import PositionSizer
from risk.stop_loss import StopLossCalculator
from risk.take_profit import TakeProfitCalculator
from risk.trailing_stop import TrailingStop
from risk.regime_adjustment import RegimeAdjustment
from risk.risk_manager import RiskManager


# ═══════════════════════════════════════════════════════════════
# TEST FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def valid_config():
    """Return a valid configuration for testing."""
    return {
        'risk': {
            'risk_per_trade_pct': 1.0,
            'max_position_size_pct': 5.0,
            'max_daily_loss_pct': 3.0,
            'max_daily_trades': 50,
            'max_weekly_loss_pct': 7.0,
            'max_monthly_loss_pct': 15.0,
            'max_drawdown_pct': 20.0,
            'max_concurrent_positions': 1,
            'max_leverage': 10,
            'kill_switch': {
                'enabled': True,
                'trigger_loss_pct': 5.0,
                'pause_hours': 24,
            },
            'regime_adjustment': {
                'enabled': True,
                'crash_multiplier': 0.3,
                'high_vol_multiplier': 0.5,
            },
        },
        'bet_sizing': {
            'method': 'kelly',
            'fixed': {
                'size_pct': 2.0,
            },
            'kelly': {
                'fraction': 0.25,
                'max_size_pct': 10.0,
            },
            'vol_target': {
                'annual_vol': 0.15,
                'base_size_pct': 2.0,
                'max_size_pct': 10.0,
            },
        },
        'position': {
            'sl_type': 'atr',
            'sl_fixed_pct': 1.0,
            'sl_atr_multiplier': 1.5,
            'tp_type': 'atr',
            'tp_fixed_pct': 2.0,
            'tp_atr_multiplier': 2.5,
            'tp_rr_ratio': 2.0,
            'trailing_stop': {
                'enabled': True,
                'type': 'fixed_pct',
                'activation_pct': 1.0,
                'trail_pct': 0.5,
                'trail_atr_mult': 1.0,
            },
        },
    }


@pytest.fixture
def disabled_kill_switch_config(valid_config):
    """Return config with disabled kill switch (should fail!)."""
    config = valid_config.copy()
    config['risk'] = valid_config['risk'].copy()
    config['risk']['kill_switch'] = {
        'enabled': False,
        'trigger_loss_pct': 5.0,
        'pause_hours': 24,
    }
    return config


# ═══════════════════════════════════════════════════════════════
# KILL SWITCH TESTS (CRITICAL!)
# ═══════════════════════════════════════════════════════════════

class TestKillSwitch:
    """Tests for KillSwitch - THE MOST CRITICAL COMPONENT!"""
    
    def test_kill_switch_cannot_be_disabled(self, disabled_kill_switch_config):
        """Kill switch must ALWAYS be enabled - CRITICAL test!"""
        with pytest.raises(ValueError, match="KILL SWITCH"):
            KillSwitch(disabled_kill_switch_config)
            
    def test_kill_switch_initialization(self, valid_config):
        """Kill switch deve inicializar corretamente quando habilitado."""
        ks = KillSwitch(valid_config)
        assert ks.triggered is False
        assert ks.trigger_loss_pct == 5.0
        assert ks.pause_hours == 24
        
    def test_kill_switch_triggers_on_loss(self, valid_config):
        """Kill switch deve ativar quando perda > threshold."""
        ks = KillSwitch(valid_config)
        balance = 10000
        
        # Loss of 6% (> 5% threshold)
        daily_pnl = -600
        result = ks.check(daily_pnl, balance)
        
        assert result is True
        assert ks.is_triggered() is True
        
    def test_kill_switch_not_triggered_on_small_loss(self, valid_config):
        """Kill switch NÃO deve ativar com perdas menores que threshold."""
        ks = KillSwitch(valid_config)
        balance = 10000
        
        # Loss of 4% (< 5% threshold)
        daily_pnl = -400
        result = ks.check(daily_pnl, balance)
        
        assert result is False
        assert ks.is_triggered() is False
        
    def test_kill_switch_not_triggered_on_profit(self, valid_config):
        """Kill switch NÃO deve ativar com lucro."""
        ks = KillSwitch(valid_config)
        balance = 10000
        
        daily_pnl = 500  # Profit
        result = ks.check(daily_pnl, balance)
        
        assert result is False
        assert ks.is_triggered() is False
        
    def test_kill_switch_force_trigger(self, valid_config):
        """Force trigger deve ativar kill switch."""
        ks = KillSwitch(valid_config)
        ks.force_trigger("Emergency stop")
        
        assert ks.is_triggered() is True
        assert "Emergency stop" in ks.trigger_reason
        
    def test_kill_switch_auto_reset_after_pause(self, valid_config):
        """Kill switch deve desativar automaticamente após pause_hours."""
        ks = KillSwitch(valid_config)
        ks.force_trigger("Test")
        
        # Simulate passage of time (25 hours > 24 hours pause)
        ks.trigger_time = datetime.now(timezone.utc) - timedelta(hours=25)
        
        assert ks.is_triggered() is False
        
    def test_kill_switch_stays_active_during_pause(self, valid_config):
        """Kill switch deve permanecer ativo durante pause period."""
        ks = KillSwitch(valid_config)
        ks.force_trigger("Test")
        
        # Simulate passage of time (10 hours < 24 hours pause)
        ks.trigger_time = datetime.now(timezone.utc) - timedelta(hours=10)
        
        assert ks.is_triggered() is True


# ═══════════════════════════════════════════════════════════════
# DAILY LIMITS TESTS
# ═══════════════════════════════════════════════════════════════

class TestDailyLimits:
    """Tests for DailyLimits."""
    
    def test_initialization(self, valid_config):
        """Daily limits deve inicializar corretamente."""
        dl = DailyLimits(valid_config)
        assert dl.max_daily_loss_pct == 3.0
        assert dl.max_daily_trades == 50
        
    def test_daily_loss_limit_exceeded(self, valid_config):
        """Deve detectar quando limite diário de perda é excedido."""
        dl = DailyLimits(valid_config)
        dl.set_initial_balance(10000)
        
        # Add loss that exceeds 3% limit
        dl.update_pnl(-350)  # 3.5% loss
        
        assert dl.is_exceeded() is True
        
    def test_daily_loss_limit_not_exceeded(self, valid_config):
        """Não deve bloquear com perdas menores que limite."""
        dl = DailyLimits(valid_config)
        dl.set_initial_balance(10000)
        
        # Add loss that's under 3% limit
        dl.update_pnl(-200)  # 2% loss
        
        assert dl.is_exceeded() is False
        
    def test_daily_trades_limit(self, valid_config):
        """Deve bloquear após exceder limite de trades diários."""
        dl = DailyLimits(valid_config)
        dl.set_initial_balance(10000)
        
        # Simulate 50 trades
        for _ in range(50):
            dl.update_pnl(10)  # Small profit each trade
            
        assert dl.is_exceeded() is True
        
    def test_weekly_loss_limit(self, valid_config):
        """Should detect when weekly limit is exceeded."""
        # Increase daily limit so weekly limit is hit first
        config = valid_config.copy()
        config['risk'] = valid_config['risk'].copy()
        config['risk']['max_daily_loss_pct'] = 10.0  # Higher daily limit
        
        dl = DailyLimits(config)
        dl.set_initial_balance(10000)
        
        # Loss that exceeds 7% weekly limit but not 10% daily
        dl.update_pnl(-800)  # 8% loss
        
        exceeded, reason = dl.check_all_limits()
        assert exceeded is True
        assert "weekly" in reason.lower()
        
    def test_reset_daily(self, valid_config):
        """Reset diário deve zerar contadores."""
        dl = DailyLimits(valid_config)
        dl.set_initial_balance(10000)
        dl.update_pnl(-100)
        
        dl.reset_daily()
        
        assert dl.daily_pnl == 0.0
        assert dl.daily_trades == 0


# ═══════════════════════════════════════════════════════════════
# POSITION SIZER TESTS
# ═══════════════════════════════════════════════════════════════

class TestPositionSizer:
    """Tests for PositionSizer."""
    
    def test_fixed_size(self, valid_config):
        """Fixed size deve calcular corretamente."""
        config = valid_config.copy()
        config['bet_sizing']['method'] = 'fixed'
        
        ps = PositionSizer(config)
        size = ps.calculate(
            balance=10000,
            current_price=50000,
            atr=1000,
            signal_confidence=0.8
        )
        
        # 2% of $10,000 = $200, at $50,000/BTC = 0.004 BTC
        expected = (10000 * 0.02) / 50000
        assert abs(size - expected) < 0.0001
        
    def test_kelly_size(self, valid_config):
        """Kelly size deve calcular baseado em confiança."""
        ps = PositionSizer(valid_config)
        
        # High confidence should give larger size
        size_high = ps.calculate(
            balance=10000,
            current_price=50000,
            atr=1000,
            signal_confidence=0.8
        )
        
        # Low confidence should give smaller size
        size_low = ps.calculate(
            balance=10000,
            current_price=50000,
            atr=1000,
            signal_confidence=0.55
        )
        
        assert size_high > size_low
        
    def test_kelly_negative_edge(self, valid_config):
        """Kelly com edge negativo deve retornar 0 ou muito pequeno."""
        ps = PositionSizer(valid_config)
        
        # Low confidence (< 50%) gives negative Kelly
        size = ps.calculate(
            balance=10000,
            current_price=50000,
            atr=1000,
            signal_confidence=0.4  # Below 50%
        )
        
        # Should be 0 (no bet with negative edge)
        assert size == 0.0
        
    def test_vol_target_size(self, valid_config):
        """Vol target deve ajustar baseado em volatilidade."""
        config = valid_config.copy()
        config['bet_sizing']['method'] = 'vol_target'
        
        ps = PositionSizer(config)
        
        # Low ATR = low volatility = larger position
        size_low_vol = ps.calculate(
            balance=10000,
            current_price=50000,
            atr=500,
            signal_confidence=0.8
        )
        
        # High ATR = high volatility = smaller position
        size_high_vol = ps.calculate(
            balance=10000,
            current_price=50000,
            atr=2000,
            signal_confidence=0.8
        )
        
        assert size_low_vol > size_high_vol
        
    def test_position_size_respects_max(self, valid_config):
        """Position sizer não deve exceder max_position_size_pct."""
        ps = PositionSizer(valid_config)
        
        # Very high confidence shouldn't exceed max
        size = ps.calculate(
            balance=10000,
            current_price=50000,
            atr=1000,
            signal_confidence=0.99
        )
        
        # Max is 10% (kelly.max_size_pct)
        max_size = (10000 * 0.10) / 50000
        assert size <= max_size
        
    def test_zero_balance_returns_zero(self, valid_config):
        """Zero balance deve retornar 0."""
        ps = PositionSizer(valid_config)
        
        size = ps.calculate(
            balance=0,
            current_price=50000,
            atr=1000,
            signal_confidence=0.8
        )
        
        assert size == 0.0
        
    def test_zero_price_returns_zero(self, valid_config):
        """Zero price deve retornar 0."""
        ps = PositionSizer(valid_config)
        
        size = ps.calculate(
            balance=10000,
            current_price=0,
            atr=1000,
            signal_confidence=0.8
        )
        
        assert size == 0.0


# ═══════════════════════════════════════════════════════════════
# STOP LOSS TESTS
# ═══════════════════════════════════════════════════════════════

class TestStopLossCalculator:
    """Tests for StopLossCalculator."""
    
    def test_fixed_pct_long(self, valid_config):
        """Fixed % SL para LONG deve estar abaixo do preço."""
        config = valid_config.copy()
        config['position']['sl_type'] = 'fixed_pct'
        
        sl = StopLossCalculator(config)
        price = sl.calculate(entry_price=50000, side="LONG")
        
        # 1% SL = $49,500
        expected = 50000 * 0.99
        assert abs(price - expected) < 1
        assert price < 50000
        
    def test_fixed_pct_short(self, valid_config):
        """Fixed % SL para SHORT deve estar acima do preço."""
        config = valid_config.copy()
        config['position']['sl_type'] = 'fixed_pct'
        
        sl = StopLossCalculator(config)
        price = sl.calculate(entry_price=50000, side="SHORT")
        
        # 1% SL = $50,500
        expected = 50000 * 1.01
        assert abs(price - expected) < 1
        assert price > 50000
        
    def test_atr_based_long(self, valid_config):
        """ATR SL para LONG deve usar multiplicador."""
        sl = StopLossCalculator(valid_config)
        
        entry = 50000
        atr = 1000  # $1000 ATR
        
        price = sl.calculate(entry_price=entry, side="LONG", atr=atr)
        
        # SL distance = ATR * 1.5 = $1500
        expected = 50000 - 1500
        assert abs(price - expected) < 1
        
    def test_atr_based_short(self, valid_config):
        """ATR SL para SHORT deve usar multiplicador."""
        sl = StopLossCalculator(valid_config)
        
        entry = 50000
        atr = 1000
        
        price = sl.calculate(entry_price=entry, side="SHORT", atr=atr)
        
        expected = 50000 + 1500
        assert abs(price - expected) < 1
        
    def test_validate_stop_loss_long(self, valid_config):
        """Validação de SL para LONG."""
        sl = StopLossCalculator(valid_config)
        
        assert sl.validate_stop_loss(50000, 49000, "LONG") is True
        assert sl.validate_stop_loss(50000, 51000, "LONG") is False
        
    def test_validate_stop_loss_short(self, valid_config):
        """Validação de SL para SHORT."""
        sl = StopLossCalculator(valid_config)
        
        assert sl.validate_stop_loss(50000, 51000, "SHORT") is True
        assert sl.validate_stop_loss(50000, 49000, "SHORT") is False


# ═══════════════════════════════════════════════════════════════
# TAKE PROFIT TESTS
# ═══════════════════════════════════════════════════════════════

class TestTakeProfitCalculator:
    """Tests for TakeProfitCalculator."""
    
    def test_fixed_pct_long(self, valid_config):
        """Fixed % TP para LONG deve estar acima do preço."""
        config = valid_config.copy()
        config['position']['tp_type'] = 'fixed_pct'
        
        tp = TakeProfitCalculator(config)
        price = tp.calculate(entry_price=50000, side="LONG")
        
        # 2% TP = $51,000
        expected = 50000 * 1.02
        assert abs(price - expected) < 1
        assert price > 50000
        
    def test_fixed_pct_short(self, valid_config):
        """Fixed % TP para SHORT deve estar abaixo do preço."""
        config = valid_config.copy()
        config['position']['tp_type'] = 'fixed_pct'
        
        tp = TakeProfitCalculator(config)
        price = tp.calculate(entry_price=50000, side="SHORT")
        
        # 2% TP = $49,000
        expected = 50000 * 0.98
        assert abs(price - expected) < 1
        assert price < 50000
        
    def test_rr_ratio_long(self, valid_config):
        """R:R ratio TP para LONG."""
        config = valid_config.copy()
        config['position']['tp_type'] = 'rr_ratio'
        
        tp = TakeProfitCalculator(config)
        
        entry = 50000
        sl = 49000  # $1000 risk
        
        price = tp.calculate(entry_price=entry, side="LONG", stop_loss=sl)
        
        # 2:1 R:R = $2000 reward = $52,000 TP
        expected = 52000
        assert abs(price - expected) < 1
        
    def test_reward_risk_ratio(self, valid_config):
        """Calcula R:R ratio corretamente."""
        tp = TakeProfitCalculator(valid_config)
        
        ratio = tp.calculate_reward_risk_ratio(
            entry_price=50000,
            stop_loss_price=49000,
            take_profit_price=52000
        )
        
        # Risk = $1000, Reward = $2000, Ratio = 2.0
        assert abs(ratio - 2.0) < 0.01


# ═══════════════════════════════════════════════════════════════
# TRAILING STOP TESTS
# ═══════════════════════════════════════════════════════════════

class TestTrailingStop:
    """Tests for TrailingStop."""
    
    def test_not_active_when_disabled(self, valid_config):
        """Trailing stop desabilitado não deve fazer nada."""
        config = valid_config.copy()
        config['position']['trailing_stop']['enabled'] = False
        
        ts = TrailingStop(config)
        result = ts.update(
            current_price=51000,
            entry_price=50000,
            side="LONG"
        )
        
        assert result is None
        assert ts.is_active is False
        
    def test_activation_threshold(self, valid_config):
        """Trailing stop só ativa após threshold."""
        ts = TrailingStop(valid_config)
        
        entry = 50000
        # Price up 0.5% (< 1% threshold)
        result = ts.update(current_price=50250, entry_price=entry, side="LONG")
        
        assert result is None
        assert ts.activated is False
        
        # Price up 1.5% (> 1% threshold)
        result = ts.update(current_price=50750, entry_price=entry, side="LONG")
        
        assert ts.activated is True
        assert result is not None
        
    def test_trailing_long_moves_up(self, valid_config):
        """Trailing SL para LONG move para cima com preço."""
        ts = TrailingStop(valid_config)
        
        entry = 50000
        
        # Activate trailing
        ts.update(current_price=51000, entry_price=entry, side="LONG")
        first_sl = ts.current_trailing_sl
        
        # Price goes higher
        ts.update(current_price=52000, entry_price=entry, side="LONG")
        second_sl = ts.current_trailing_sl
        
        assert second_sl > first_sl
        
    def test_trailing_short_moves_down(self, valid_config):
        """Trailing SL para SHORT move para baixo com preço."""
        ts = TrailingStop(valid_config)
        
        entry = 50000
        
        # Activate trailing (price goes down for SHORT profit)
        ts.update(current_price=49000, entry_price=entry, side="SHORT")
        first_sl = ts.current_trailing_sl
        
        # Price goes lower
        ts.update(current_price=48000, entry_price=entry, side="SHORT")
        second_sl = ts.current_trailing_sl
        
        assert second_sl < first_sl
        
    def test_should_trigger(self, valid_config):
        """Deve detectar quando trailing stop é atingido."""
        ts = TrailingStop(valid_config)
        
        entry = 50000
        
        # Activate and set trailing
        ts.update(current_price=51000, entry_price=entry, side="LONG")
        
        # Get current trailing SL
        trailing_sl = ts.current_trailing_sl
        
        # Price at or below trailing SL should trigger
        assert ts.should_trigger(trailing_sl - 1, "LONG") is True
        assert ts.should_trigger(trailing_sl + 100, "LONG") is False


# ═══════════════════════════════════════════════════════════════
# REGIME ADJUSTMENT TESTS
# ═══════════════════════════════════════════════════════════════

class TestRegimeAdjustment:
    """Tests for RegimeAdjustment."""
    
    def test_normal_regime(self, valid_config):
        """Regime normal deve retornar multiplier 1.0."""
        ra = RegimeAdjustment(valid_config)
        
        mult = ra.get_multiplier({'volatility': 'normal', 'hmm': 1})
        assert mult == 1.0
        
    def test_extreme_volatility_reduces_risk(self, valid_config):
        """Volatilidade extrema deve reduzir risco."""
        ra = RegimeAdjustment(valid_config)
        
        mult = ra.get_multiplier({'volatility': 'extreme', 'hmm': 1})
        
        assert mult == 0.3  # crash_multiplier
        assert mult < 1.0
        
    def test_high_volatility_reduces_risk(self, valid_config):
        """Alta volatilidade deve reduzir risco moderadamente."""
        ra = RegimeAdjustment(valid_config)
        
        mult = ra.get_multiplier({'volatility': 'high', 'hmm': 1})
        
        assert mult == 0.5  # high_vol_multiplier
        assert mult < 1.0
        
    def test_hmm_regime_2_is_dangerous(self, valid_config):
        """HMM regime 2 é perigoso."""
        ra = RegimeAdjustment(valid_config)
        
        mult = ra.get_multiplier({'volatility': 'normal', 'hmm': 2})
        
        assert mult == 0.3  # crash_multiplier
        
    def test_disabled_returns_1(self, valid_config):
        """Ajuste desabilitado deve retornar 1.0."""
        config = valid_config.copy()
        config['risk']['regime_adjustment']['enabled'] = False
        
        ra = RegimeAdjustment(config)
        mult = ra.get_multiplier({'volatility': 'extreme', 'hmm': 2})
        
        assert mult == 1.0


# ═══════════════════════════════════════════════════════════════
# RISK MANAGER INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════

class TestRiskManager:
    """Integration tests for RiskManager."""
    
    def test_initialization(self, valid_config):
        """Risk manager deve inicializar todos os componentes."""
        rm = RiskManager(valid_config)
        
        assert rm.kill_switch is not None
        assert rm.daily_limits is not None
        assert rm.position_sizer is not None
        assert rm.stop_loss_calculator is not None
        assert rm.take_profit_calculator is not None
        assert rm.trailing_stop is not None
        assert rm.regime_adjuster is not None
        
    def test_can_open_position_ok(self, valid_config):
        """Deve permitir abrir posição quando tudo OK."""
        rm = RiskManager(valid_config)
        rm.set_initial_balance(10000)
        
        can_open, msg = rm.can_open_position(balance=10000)
        
        assert can_open is True
        assert "OK" in msg
        
    def test_cannot_open_when_kill_switch_active(self, valid_config):
        """Não deve permitir abrir quando kill switch ativo."""
        rm = RiskManager(valid_config)
        rm.set_initial_balance(10000)
        rm.force_kill_switch("Test")
        
        can_open, msg = rm.can_open_position(balance=10000)
        
        assert can_open is False
        assert "KILL" in msg.upper() or "Test" in msg
        
    def test_cannot_open_max_positions(self, valid_config):
        """Não deve permitir abrir quando max posições atingido."""
        rm = RiskManager(valid_config)
        rm.set_initial_balance(10000)
        
        # Open 1 position (max is 1)
        rm.on_position_opened()
        
        can_open, msg = rm.can_open_position(balance=10000)
        
        assert can_open is False
        assert "posições" in msg.lower() or "position" in msg.lower()
        
    def test_cannot_open_max_drawdown(self, valid_config):
        """Não deve permitir abrir quando drawdown máximo atingido."""
        rm = RiskManager(valid_config)
        rm.set_initial_balance(10000)
        
        # Simulate 25% drawdown (> 20% limit)
        rm._peak_balance = 10000
        rm._current_drawdown = 25.0
        
        can_open, msg = rm.can_open_position(balance=7500)
        
        assert can_open is False
        assert "drawdown" in msg.lower()
        
    def test_position_size_calculation(self, valid_config):
        """Position size deve calcular corretamente."""
        rm = RiskManager(valid_config)
        
        size = rm.calculate_position_size(
            balance=10000,
            current_price=50000,
            atr=1000,
            signal_confidence=0.8
        )
        
        assert size > 0
        # Max 5% of balance / price
        max_size = (10000 * 0.05) / 50000
        assert size <= max_size
        
    def test_position_size_regime_adjusted(self, valid_config):
        """Position size deve ser ajustado pelo regime."""
        rm = RiskManager(valid_config)
        
        size_normal = rm.calculate_position_size(
            balance=10000,
            current_price=50000,
            atr=1000,
            signal_confidence=0.8,
            regime={'volatility': 'normal', 'hmm': 1}
        )
        
        size_extreme = rm.calculate_position_size(
            balance=10000,
            current_price=50000,
            atr=1000,
            signal_confidence=0.8,
            regime={'volatility': 'extreme', 'hmm': 2}
        )
        
        # Extreme regime should have smaller size
        assert size_extreme < size_normal
        
    def test_pnl_updates(self, valid_config):
        """P&L deve atualizar corretamente."""
        rm = RiskManager(valid_config)
        rm.set_initial_balance(10000)
        
        rm.update_pnl(100, balance=10100)
        assert rm.daily_pnl == 100
        
        rm.update_pnl(-50, balance=10050)
        assert rm.daily_pnl == 50
        
    def test_suspicious_pnl_rejected(self, valid_config):
        """P&L suspeito deve ser rejeitado."""
        rm = RiskManager(valid_config)
        rm.set_initial_balance(10000)
        
        # Try to update with None
        rm.update_pnl(None, balance=10000)
        assert rm.daily_pnl == 0
        
        # Try to update with value > balance
        rm.update_pnl(-15000, balance=10000)  # 150% loss
        assert rm.daily_pnl == 0
        
    def test_risk_summary(self, valid_config):
        """Risk summary deve retornar todas as métricas."""
        rm = RiskManager(valid_config)
        rm.set_initial_balance(10000)
        rm.update_pnl(100, balance=10100)
        
        summary = rm.get_risk_summary()
        
        assert 'daily_pnl' in summary
        assert 'weekly_pnl' in summary
        assert 'monthly_pnl' in summary
        assert 'current_drawdown_pct' in summary
        assert 'kill_switch_active' in summary
        assert 'open_positions' in summary
        
    def test_stop_loss_calculation(self, valid_config):
        """Stop loss deve ser calculado corretamente."""
        rm = RiskManager(valid_config)
        
        sl = rm.calculate_stop_loss(
            entry_price=50000,
            side="LONG",
            atr=1000
        )
        
        assert sl < 50000  # SL below entry for LONG
        
    def test_take_profit_calculation(self, valid_config):
        """Take profit deve ser calculado corretamente."""
        rm = RiskManager(valid_config)
        
        tp = rm.calculate_take_profit(
            entry_price=50000,
            side="LONG",
            atr=1000
        )
        
        assert tp > 50000  # TP above entry for LONG


# ═══════════════════════════════════════════════════════════════
# RUN TESTS
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
