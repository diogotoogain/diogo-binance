"""
Tests for V2 Seasonality module.

Tests cover:
- SeasonalityFeatures (temporal features calculation)
- SeasonalityAdjuster (position size adjustment)
- Integration with RiskManager
- Optuna parameter optimization compatibility
"""
import pytest
from datetime import datetime

import sys
from pathlib import Path

# Add v2/src to path
v2_src = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(v2_src))

from features.temporal.seasonality import SeasonalityFeatures
from risk.seasonality_adjuster import SeasonalityAdjuster
from risk.risk_manager import RiskManager


# ═══════════════════════════════════════════════════════════════
# TEST FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def seasonality_config():
    """Return a configuration with seasonality enabled."""
    return {
        'seasonality': {
            'enabled': True,
            'monthly_adjustment': {
                'enabled': True,
                'january': 1.0,
                'february': 1.0,
                'march': 1.0,
                'april': 1.0,
                'may': 0.8,
                'june': 0.7,
                'july': 0.7,
                'august': 0.7,
                'september': 0.8,
                'october': 1.2,
                'november': 1.3,
                'december': 1.2,
            },
            'month_period': {
                'enabled': True,
                'early_month_multiplier': 1.0,
                'mid_month_multiplier': 1.0,
                'late_month_multiplier': 0.9,
            },
            'quarter_adjustment': {
                'enabled': True,
                'q1_multiplier': 1.0,
                'q2_multiplier': 0.9,
                'q3_multiplier': 0.8,
                'q4_multiplier': 1.2,
            },
        },
    }


@pytest.fixture
def disabled_seasonality_config():
    """Return a configuration with seasonality disabled."""
    return {
        'seasonality': {
            'enabled': False,
            'monthly_adjustment': {
                'enabled': True,
                'october': 1.2,
            },
            'month_period': {
                'enabled': True,
                'late_month_multiplier': 0.9,
            },
            'quarter_adjustment': {
                'enabled': True,
                'q4_multiplier': 1.2,
            },
        },
    }


@pytest.fixture
def full_config(seasonality_config):
    """Return full config including risk and seasonality."""
    return {
        **seasonality_config,
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


# ═══════════════════════════════════════════════════════════════
# SEASONALITY FEATURES TESTS
# ═══════════════════════════════════════════════════════════════

class TestSeasonalityFeatures:
    """Tests for SeasonalityFeatures."""
    
    def test_initialization(self, seasonality_config):
        """SeasonalityFeatures should initialize correctly."""
        sf = SeasonalityFeatures(seasonality_config)
        assert sf.enabled is True
        assert sf.monthly_enabled is True
        assert sf.period_enabled is True
        assert sf.quarter_enabled is True
        
    def test_calculate_features_january(self, seasonality_config):
        """Calculates features for a January date correctly."""
        sf = SeasonalityFeatures(seasonality_config)
        
        # January 15th
        ts = datetime(2024, 1, 15)
        features = sf.calculate(ts)
        
        assert features['month'] == 1.0
        assert features['quarter'] == 1.0
        assert features['month_period'] == 'mid'
        assert features['is_q4'] == 0.0
        assert features['is_summer'] == 0.0
        assert features['is_q1'] == 1.0
        
    def test_calculate_features_october_q4(self, seasonality_config):
        """Calculates Q4 features correctly for October."""
        sf = SeasonalityFeatures(seasonality_config)
        
        # October 25th (late month, Q4)
        ts = datetime(2024, 10, 25)
        features = sf.calculate(ts)
        
        assert features['month'] == 10.0
        assert features['quarter'] == 4.0
        assert features['month_period'] == 'late'
        assert features['is_q4'] == 1.0
        assert features['is_summer'] == 0.0
        
    def test_calculate_features_july_summer(self, seasonality_config):
        """Calculates summer features correctly for July."""
        sf = SeasonalityFeatures(seasonality_config)
        
        # July 5th (early month, summer)
        ts = datetime(2024, 7, 5)
        features = sf.calculate(ts)
        
        assert features['month'] == 7.0
        assert features['quarter'] == 3.0
        assert features['month_period'] == 'early'
        assert features['is_q4'] == 0.0
        assert features['is_summer'] == 1.0
        assert features['is_q3'] == 1.0
        
    def test_get_monthly_multiplier_october(self, seasonality_config):
        """October should have higher multiplier (Uptober)."""
        sf = SeasonalityFeatures(seasonality_config)
        
        mult = sf.get_monthly_multiplier(10)
        assert mult == 1.2
        
    def test_get_monthly_multiplier_june_summer(self, seasonality_config):
        """June should have lower multiplier (summer)."""
        sf = SeasonalityFeatures(seasonality_config)
        
        mult = sf.get_monthly_multiplier(6)
        assert mult == 0.7
        
    def test_get_monthly_multiplier_may_sell(self, seasonality_config):
        """May should have lower multiplier (Sell in May)."""
        sf = SeasonalityFeatures(seasonality_config)
        
        mult = sf.get_monthly_multiplier(5)
        assert mult == 0.8
        
    def test_get_monthly_multiplier_november(self, seasonality_config):
        """November should have highest multiplier."""
        sf = SeasonalityFeatures(seasonality_config)
        
        mult = sf.get_monthly_multiplier(11)
        assert mult == 1.3
        
    def test_get_period_multiplier_early(self, seasonality_config):
        """Early month (days 1-10) should have normal multiplier."""
        sf = SeasonalityFeatures(seasonality_config)
        
        mult = sf.get_period_multiplier(5)
        assert mult == 1.0
        
    def test_get_period_multiplier_late(self, seasonality_config):
        """Late month (days 21-31) should have reduced multiplier."""
        sf = SeasonalityFeatures(seasonality_config)
        
        mult = sf.get_period_multiplier(25)
        assert mult == 0.9
        
    def test_get_quarter_multiplier_q4(self, seasonality_config):
        """Q4 should have higher multiplier."""
        sf = SeasonalityFeatures(seasonality_config)
        
        # October is Q4
        mult = sf.get_quarter_multiplier(10)
        assert mult == 1.2
        
    def test_get_quarter_multiplier_q3(self, seasonality_config):
        """Q3 should have lower multiplier."""
        sf = SeasonalityFeatures(seasonality_config)
        
        # July is Q3
        mult = sf.get_quarter_multiplier(7)
        assert mult == 0.8
        
    def test_get_size_multiplier_combines_all(self, seasonality_config):
        """Combined multiplier should multiply all factors."""
        sf = SeasonalityFeatures(seasonality_config)
        
        # October 25th: month=1.2, period=0.9, quarter=1.2
        ts = datetime(2024, 10, 25)
        combined = sf.get_size_multiplier(ts)
        
        expected = 1.2 * 0.9 * 1.2
        assert abs(combined - expected) < 0.01
        
    def test_get_size_multiplier_summer_combination(self, seasonality_config):
        """Summer months should have significantly reduced multiplier."""
        sf = SeasonalityFeatures(seasonality_config)
        
        # July 15th: month=0.7, period=1.0, quarter=0.8
        ts = datetime(2024, 7, 15)
        combined = sf.get_size_multiplier(ts)
        
        expected = 0.7 * 1.0 * 0.8
        assert abs(combined - expected) < 0.01
        
    def test_disabled_returns_1(self, disabled_seasonality_config):
        """Disabled seasonality should return 1.0."""
        sf = SeasonalityFeatures(disabled_seasonality_config)
        
        ts = datetime(2024, 10, 25)
        mult = sf.get_size_multiplier(ts)
        
        assert mult == 1.0


# ═══════════════════════════════════════════════════════════════
# SEASONALITY ADJUSTER TESTS
# ═══════════════════════════════════════════════════════════════

class TestSeasonalityAdjuster:
    """Tests for SeasonalityAdjuster."""
    
    def test_initialization(self, seasonality_config):
        """SeasonalityAdjuster should initialize correctly."""
        sa = SeasonalityAdjuster(seasonality_config)
        assert sa.enabled is True
        assert sa.seasonality_features is not None
        
    def test_is_enabled(self, seasonality_config):
        """Should correctly report enabled status."""
        sa = SeasonalityAdjuster(seasonality_config)
        assert sa.is_enabled() is True
        
    def test_is_disabled(self, disabled_seasonality_config):
        """Should correctly report disabled status."""
        sa = SeasonalityAdjuster(disabled_seasonality_config)
        assert sa.is_enabled() is False
        
    def test_get_multiplier_q4(self, seasonality_config):
        """Should return correct multiplier for Q4."""
        sa = SeasonalityAdjuster(seasonality_config)
        
        # November 15th in Q4
        ts = datetime(2024, 11, 15)
        mult = sa.get_multiplier(ts)
        
        # November: 1.3 * period: 1.0 * Q4: 1.2 = 1.56
        expected = 1.3 * 1.0 * 1.2
        assert abs(mult - expected) < 0.01
        
    def test_get_multiplier_summer(self, seasonality_config):
        """Should return reduced multiplier for summer."""
        sa = SeasonalityAdjuster(seasonality_config)
        
        # August 10th in summer
        ts = datetime(2024, 8, 10)
        mult = sa.get_multiplier(ts)
        
        # August: 0.7 * period: 1.0 * Q3: 0.8 = 0.56
        expected = 0.7 * 1.0 * 0.8
        assert abs(mult - expected) < 0.01
        
    def test_adjust_position_size_reduces(self, seasonality_config):
        """Should reduce position size in weak months."""
        sa = SeasonalityAdjuster(seasonality_config)
        
        base_size = 1.0
        # July late month
        ts = datetime(2024, 7, 25)
        
        adjusted = sa.adjust_position_size(base_size, ts)
        
        # Should be significantly less than 1.0
        assert adjusted < base_size
        assert adjusted > 0
        
    def test_adjust_position_size_increases(self, seasonality_config):
        """Should increase position size in strong months."""
        sa = SeasonalityAdjuster(seasonality_config)
        
        base_size = 1.0
        # November early month (strong)
        ts = datetime(2024, 11, 5)
        
        adjusted = sa.adjust_position_size(base_size, ts)
        
        # Should be more than 1.0
        assert adjusted > base_size
        
    def test_adjust_position_size_disabled_returns_same(self, disabled_seasonality_config):
        """Disabled adjuster should return same size."""
        sa = SeasonalityAdjuster(disabled_seasonality_config)
        
        base_size = 1.5
        ts = datetime(2024, 10, 15)
        
        adjusted = sa.adjust_position_size(base_size, ts)
        
        assert adjusted == base_size
        
    def test_get_seasonality_info(self, seasonality_config):
        """Should return complete seasonality information."""
        sa = SeasonalityAdjuster(seasonality_config)
        
        ts = datetime(2024, 10, 25)
        info = sa.get_seasonality_info(ts)
        
        assert info['enabled'] is True
        assert info['month'] == 10
        assert info['quarter'] == 4
        assert info['month_period'] == 'late'
        assert info['is_q4'] is True
        assert info['is_summer'] is False
        assert 'combined_multiplier' in info
        assert 'monthly_multiplier' in info
        assert 'period_multiplier' in info
        assert 'quarter_multiplier' in info


# ═══════════════════════════════════════════════════════════════
# RISK MANAGER INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════

class TestRiskManagerSeasonalityIntegration:
    """Integration tests for RiskManager with SeasonalityAdjuster."""
    
    def test_risk_manager_has_seasonality_adjuster(self, full_config):
        """RiskManager should initialize seasonality adjuster."""
        rm = RiskManager(full_config)
        assert rm.seasonality_adjuster is not None
        
    def test_position_size_adjusted_by_seasonality(self, full_config):
        """Position size should be adjusted by seasonality."""
        # Increase max position size to allow for seasonality differences
        config = full_config.copy()
        config['risk'] = full_config['risk'].copy()
        config['risk']['max_position_size_pct'] = 20.0  # Higher limit
        
        rm = RiskManager(config)
        
        # Calculate size in strong month (November)
        size_strong = rm.calculate_position_size(
            balance=10000,
            current_price=50000,
            atr=1000,
            signal_confidence=0.8,
            timestamp=datetime(2024, 11, 10)  # November
        )
        
        # Calculate size in weak month (July)
        size_weak = rm.calculate_position_size(
            balance=10000,
            current_price=50000,
            atr=1000,
            signal_confidence=0.8,
            timestamp=datetime(2024, 7, 10)  # July
        )
        
        # Strong month should have larger size
        assert size_strong > size_weak
        
    def test_position_size_with_regime_and_seasonality(self, full_config):
        """Position size should combine regime and seasonality adjustments."""
        rm = RiskManager(full_config)
        
        # With extreme regime
        size_extreme = rm.calculate_position_size(
            balance=10000,
            current_price=50000,
            atr=1000,
            signal_confidence=0.8,
            regime={'volatility': 'extreme', 'hmm': 2},
            timestamp=datetime(2024, 10, 15)
        )
        
        # Normal regime
        size_normal = rm.calculate_position_size(
            balance=10000,
            current_price=50000,
            atr=1000,
            signal_confidence=0.8,
            regime={'volatility': 'normal', 'hmm': 1},
            timestamp=datetime(2024, 10, 15)
        )
        
        # Extreme regime should still produce smaller size
        assert size_extreme < size_normal
        
    def test_risk_summary_includes_seasonality(self, full_config):
        """Risk summary should include seasonality information."""
        rm = RiskManager(full_config)
        rm.set_initial_balance(10000)
        
        ts = datetime(2024, 10, 25)
        summary = rm.get_risk_summary(timestamp=ts)
        
        assert 'seasonality' in summary
        assert summary['seasonality']['month'] == 10
        assert summary['seasonality']['is_q4'] is True
        
    def test_get_seasonality_info_method(self, full_config):
        """RiskManager should expose seasonality info method."""
        rm = RiskManager(full_config)
        
        ts = datetime(2024, 11, 15)
        info = rm.get_seasonality_info(ts)
        
        assert info['month'] == 11
        assert info['quarter'] == 4
        assert 'combined_multiplier' in info


# ═══════════════════════════════════════════════════════════════
# EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════

class TestSeasonalityEdgeCases:
    """Tests for edge cases in seasonality calculations."""
    
    def test_february_leap_year(self, seasonality_config):
        """Should handle February in leap year correctly."""
        sf = SeasonalityFeatures(seasonality_config)
        
        # Feb 29 in leap year
        ts = datetime(2024, 2, 29)
        features = sf.calculate(ts)
        
        assert features['month'] == 2.0
        assert features['month_period'] == 'late'
        
    def test_december_31(self, seasonality_config):
        """Should handle last day of year correctly."""
        sf = SeasonalityFeatures(seasonality_config)
        
        ts = datetime(2024, 12, 31)
        features = sf.calculate(ts)
        
        assert features['month'] == 12.0
        assert features['quarter'] == 4.0
        assert features['month_period'] == 'late'
        assert features['is_q4'] == 1.0
        
    def test_january_1(self, seasonality_config):
        """Should handle first day of year correctly."""
        sf = SeasonalityFeatures(seasonality_config)
        
        ts = datetime(2024, 1, 1)
        features = sf.calculate(ts)
        
        assert features['month'] == 1.0
        assert features['quarter'] == 1.0
        assert features['month_period'] == 'early'
        assert features['is_q1'] == 1.0
        
    def test_invalid_month_returns_default(self, seasonality_config):
        """Invalid month should return default multiplier."""
        sf = SeasonalityFeatures(seasonality_config)
        
        # Month 0 is invalid
        mult = sf.get_monthly_multiplier(0)
        assert mult == 1.0
        
        # Month 13 is invalid
        mult = sf.get_monthly_multiplier(13)
        assert mult == 1.0
        
    def test_empty_config(self):
        """Should handle empty config gracefully."""
        sf = SeasonalityFeatures({})
        
        ts = datetime(2024, 10, 15)
        mult = sf.get_size_multiplier(ts)
        
        # With empty config, all defaults should be 1.0
        assert mult == 1.0


# ═══════════════════════════════════════════════════════════════
# OPTUNA OPTIMIZATION TESTS
# ═══════════════════════════════════════════════════════════════

class TestSeasonalityOptimization:
    """Tests that Optuna can optimize seasonality parameters."""
    
    def test_params_are_in_schema(self):
        """Seasonality params should be in optimizable params."""
        from config.schema import get_all_optimizable_params
        
        params = get_all_optimizable_params()
        param_names = [p.name for p in params]
        
        # Check key seasonality params are present
        assert 'seasonality.enabled' in param_names
        assert 'seasonality.monthly_adjustment.enabled' in param_names
        assert 'seasonality.monthly_adjustment.october' in param_names
        assert 'seasonality.month_period.late_month_multiplier' in param_names
        assert 'seasonality.quarter_adjustment.q4_multiplier' in param_names
        
    def test_monthly_params_have_correct_range(self):
        """Monthly params should have optimizable range 0.5-1.5."""
        from config.schema import get_all_optimizable_params
        
        params = get_all_optimizable_params()
        
        october_param = next(
            (p for p in params if p.name == 'seasonality.monthly_adjustment.october'),
            None
        )
        
        assert october_param is not None
        assert october_param.low == 0.5
        assert october_param.high == 1.5
        assert october_param.default == 1.2
        
    def test_quarter_params_have_correct_range(self):
        """Quarter params should have optimizable range 0.7-1.3."""
        from config.schema import get_all_optimizable_params
        
        params = get_all_optimizable_params()
        
        q4_param = next(
            (p for p in params if p.name == 'seasonality.quarter_adjustment.q4_multiplier'),
            None
        )
        
        assert q4_param is not None
        assert q4_param.low == 0.7
        assert q4_param.high == 1.3
        assert q4_param.default == 1.2


# ═══════════════════════════════════════════════════════════════
# RUN TESTS
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
