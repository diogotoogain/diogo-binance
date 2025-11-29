"""
Testes para o filtro de dia da semana.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from v2.src.features.temporal.day_of_week import DayOfWeekFeatures
from v2.src.strategies.base import Strategy, Signal, SignalDirection


# Mock strategy for testing
class MockStrategy(Strategy):
    """Estratégia mock para testar filtro de dia da semana."""
    
    def __init__(self, config):
        super().__init__("MockStrategy", config, enabled=True)
    
    def generate_signal(self, market_data):
        """Generate a mock signal for testing."""
        return Signal(
            direction=SignalDirection.BUY,
            strategy_name=self.name,
            confidence=0.8,
            reason="Test signal"
        )


class TestDayOfWeekFeatures:
    """Testes para DayOfWeekFeatures."""
    
    @pytest.fixture
    def feature(self):
        """Cria instância de DayOfWeekFeatures."""
        return DayOfWeekFeatures({}, enabled=True)
    
    @pytest.fixture
    def sample_data_with_dates(self):
        """Gera dados de teste com datas variadas."""
        # Create a week of data starting from a Monday
        dates = pd.date_range(start='2024-01-01', periods=7, freq='D')  # 2024-01-01 is Monday
        data = pd.DataFrame({
            'close': np.random.rand(7) * 1000 + 50000,
            'volume': np.random.rand(7) * 1000000
        }, index=dates)
        return data
    
    def test_monday_is_identified_correctly(self, feature):
        """Testa que segunda-feira é identificada corretamente."""
        # 2024-01-01 is a Monday
        monday = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        result = feature._calculate_features_from_datetime(monday)
        
        assert result['day_of_week'] == 0.0
        assert result['is_monday'] == 1.0
        assert result['is_friday'] == 0.0
        assert result['is_weekend'] == 0.0
        assert result['days_from_weekend'] == 0.0
        assert result['days_to_weekend'] == 4.0
    
    def test_friday_is_identified_correctly(self, feature):
        """Testa que sexta-feira é identificada corretamente."""
        # 2024-01-05 is a Friday
        friday = datetime(2024, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
        
        result = feature._calculate_features_from_datetime(friday)
        
        assert result['day_of_week'] == 4.0
        assert result['is_monday'] == 0.0
        assert result['is_friday'] == 1.0
        assert result['is_weekend'] == 0.0
        assert result['days_from_weekend'] == 4.0
        assert result['days_to_weekend'] == 0.0
    
    def test_weekend_is_identified_correctly(self, feature):
        """Testa que fim de semana é identificado corretamente."""
        # 2024-01-06 is a Saturday
        saturday = datetime(2024, 1, 6, 12, 0, 0, tzinfo=timezone.utc)
        
        result = feature._calculate_features_from_datetime(saturday)
        
        assert result['day_of_week'] == 5.0
        assert result['is_monday'] == 0.0
        assert result['is_friday'] == 0.0
        assert result['is_weekend'] == 1.0
        assert result['days_from_weekend'] == 0.0
        assert result['days_to_weekend'] == 0.0
        
        # 2024-01-07 is a Sunday
        sunday = datetime(2024, 1, 7, 12, 0, 0, tzinfo=timezone.utc)
        
        result = feature._calculate_features_from_datetime(sunday)
        
        assert result['day_of_week'] == 6.0
        assert result['is_weekend'] == 1.0
    
    def test_static_methods(self, feature):
        """Testa métodos estáticos."""
        monday = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        friday = datetime(2024, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
        saturday = datetime(2024, 1, 6, 12, 0, 0, tzinfo=timezone.utc)
        
        assert DayOfWeekFeatures.is_monday_day(monday) is True
        assert DayOfWeekFeatures.is_monday_day(friday) is False
        
        assert DayOfWeekFeatures.is_friday_day(friday) is True
        assert DayOfWeekFeatures.is_friday_day(monday) is False
        
        assert DayOfWeekFeatures.is_weekend_day(saturday) is True
        assert DayOfWeekFeatures.is_weekend_day(monday) is False
        assert DayOfWeekFeatures.is_weekend_day(friday) is False
        
        assert DayOfWeekFeatures.get_day_of_week(monday) == 0
        assert DayOfWeekFeatures.get_day_of_week(friday) == 4
        assert DayOfWeekFeatures.get_day_of_week(saturday) == 5
    
    def test_batch_calculation(self, feature, sample_data_with_dates):
        """Testa cálculo batch de features."""
        result = feature.calculate(sample_data_with_dates)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 7
        assert 'day_of_week' in result.columns
        assert 'is_monday' in result.columns
        assert 'is_friday' in result.columns
        assert 'is_weekend' in result.columns
        assert 'days_to_weekend' in result.columns
        assert 'days_from_weekend' in result.columns
        
        # Check day_of_week values
        assert result['day_of_week'].iloc[0] == 0  # Monday
        assert result['day_of_week'].iloc[4] == 4  # Friday
        assert result['day_of_week'].iloc[5] == 5  # Saturday
        assert result['day_of_week'].iloc[6] == 6  # Sunday
    
    def test_incremental_calculation(self, feature):
        """Testa cálculo incremental."""
        monday_ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        
        result = feature.calculate_incremental({'timestamp': monday_ts}, {})
        
        assert result['day_of_week'] == 0.0
        assert result['is_monday'] == 1.0
    
    def test_incremental_with_unix_timestamp(self, feature):
        """Testa cálculo incremental com Unix timestamp."""
        # Use a timestamp for Friday Jan 5, 2024
        friday_ts = datetime(2024, 1, 5, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        
        result = feature.calculate_incremental({'timestamp': friday_ts}, {})
        
        assert result['day_of_week'] == 4.0
        assert result['is_friday'] == 1.0
    
    def test_disabled_returns_empty(self):
        """Feature desabilitada retorna dict vazio."""
        feature = DayOfWeekFeatures({}, enabled=False)
        
        result = feature.calculate_incremental({'timestamp': 1704110400}, {})
        assert result == {}
        
        dates = pd.date_range(start='2024-01-01', periods=7, freq='D')
        data = pd.DataFrame({
            'close': np.random.rand(7) * 1000
        }, index=dates)
        
        batch_result = feature.calculate(data)
        assert isinstance(batch_result, pd.DataFrame)
        assert len(batch_result.columns) == 0


class TestDayOfWeekFilterInStrategy:
    """Testes para filtro de dia da semana nas estratégias."""
    
    @pytest.fixture
    def strategy_with_filter_enabled(self):
        """Cria estratégia com filtro habilitado."""
        config = {
            'filters': {
                'day_of_week_filter': {
                    'enabled': True,
                    'allowed_days': [0, 1, 2, 3, 4],
                    'monday_multiplier': 0.7,
                    'friday_multiplier': 0.8,
                    'weekend_allowed': False
                }
            }
        }
        return MockStrategy(config)
    
    @pytest.fixture
    def strategy_with_filter_disabled(self):
        """Cria estratégia com filtro desabilitado."""
        config = {
            'filters': {
                'day_of_week_filter': {
                    'enabled': False
                }
            }
        }
        return MockStrategy(config)
    
    @pytest.fixture
    def strategy_weekend_allowed(self):
        """Cria estratégia que permite fim de semana."""
        config = {
            'filters': {
                'day_of_week_filter': {
                    'enabled': True,
                    'allowed_days': [0, 1, 2, 3, 4, 5, 6],
                    'monday_multiplier': 0.7,
                    'friday_multiplier': 0.8,
                    'weekend_allowed': True
                }
            }
        }
        return MockStrategy(config)
    
    def test_filter_blocks_weekend(self, strategy_with_filter_enabled):
        """Testa que filtro bloqueia fins de semana."""
        # Saturday Jan 6, 2024
        saturday = datetime(2024, 1, 6, 12, 0, 0, tzinfo=timezone.utc)
        
        allowed, multiplier = strategy_with_filter_enabled._check_day_of_week_filter(saturday)
        
        assert allowed is False
        assert multiplier == 0.0
    
    def test_filter_allows_weekday(self, strategy_with_filter_enabled):
        """Testa que filtro permite dias da semana."""
        # Wednesday Jan 3, 2024
        wednesday = datetime(2024, 1, 3, 12, 0, 0, tzinfo=timezone.utc)
        
        allowed, multiplier = strategy_with_filter_enabled._check_day_of_week_filter(wednesday)
        
        assert allowed is True
        assert multiplier == 1.0  # Wednesday has no special multiplier
    
    def test_monday_multiplier_is_applied(self, strategy_with_filter_enabled):
        """Testa que multiplicador de segunda é aplicado corretamente."""
        # Monday Jan 1, 2024
        monday = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        
        allowed, multiplier = strategy_with_filter_enabled._check_day_of_week_filter(monday)
        
        assert allowed is True
        assert multiplier == 0.7
    
    def test_friday_multiplier_is_applied(self, strategy_with_filter_enabled):
        """Testa que multiplicador de sexta é aplicado corretamente."""
        # Friday Jan 5, 2024
        friday = datetime(2024, 1, 5, 12, 0, 0, tzinfo=timezone.utc)
        
        allowed, multiplier = strategy_with_filter_enabled._check_day_of_week_filter(friday)
        
        assert allowed is True
        assert multiplier == 0.8
    
    def test_disabled_filter_allows_all(self, strategy_with_filter_disabled):
        """Testa que filtro desabilitado permite todos os dias."""
        saturday = datetime(2024, 1, 6, 12, 0, 0, tzinfo=timezone.utc)
        
        allowed, multiplier = strategy_with_filter_disabled._check_day_of_week_filter(saturday)
        
        assert allowed is True
        assert multiplier == 1.0
    
    def test_weekend_allowed_flag(self, strategy_weekend_allowed):
        """Testa que flag weekend_allowed funciona."""
        saturday = datetime(2024, 1, 6, 12, 0, 0, tzinfo=timezone.utc)
        
        allowed, multiplier = strategy_weekend_allowed._check_day_of_week_filter(saturday)
        
        assert allowed is True
        assert multiplier == 1.0
    
    def test_uses_current_time_when_none(self, strategy_with_filter_disabled):
        """Testa que usa hora atual quando timestamp é None."""
        allowed, multiplier = strategy_with_filter_disabled._check_day_of_week_filter(None)
        
        # Filter is disabled, should always allow
        assert allowed is True
        assert multiplier == 1.0
    
    def test_handles_unix_timestamp(self, strategy_with_filter_enabled):
        """Testa que aceita Unix timestamp."""
        # Monday Jan 1, 2024 as Unix timestamp
        monday_ts = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc).timestamp()
        
        allowed, multiplier = strategy_with_filter_enabled._check_day_of_week_filter(monday_ts)
        
        assert allowed is True
        assert multiplier == 0.7
    
    def test_blocked_day_not_in_allowed_list(self):
        """Testa que dias não na lista são bloqueados."""
        config = {
            'filters': {
                'day_of_week_filter': {
                    'enabled': True,
                    'allowed_days': [1, 2, 3],  # Only Tuesday, Wednesday, Thursday
                    'monday_multiplier': 0.7,
                    'friday_multiplier': 0.8,
                    'weekend_allowed': False
                }
            }
        }
        strategy = MockStrategy(config)
        
        # Monday should be blocked
        monday = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        allowed, multiplier = strategy._check_day_of_week_filter(monday)
        
        assert allowed is False
        assert multiplier == 0.0
        
        # Tuesday should be allowed
        tuesday = datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
        allowed, multiplier = strategy._check_day_of_week_filter(tuesday)
        
        assert allowed is True


class TestDayOfWeekFilterIntegration:
    """Testes de integração do filtro de dia da semana."""
    
    def test_strategy_loads_config_correctly(self):
        """Testa que estratégia carrega config corretamente."""
        config = {
            'filters': {
                'day_of_week_filter': {
                    'enabled': True,
                    'allowed_days': [0, 1, 2, 3, 4],
                    'monday_multiplier': 0.65,
                    'friday_multiplier': 0.75,
                    'weekend_allowed': False
                }
            }
        }
        strategy = MockStrategy(config)
        
        assert strategy._day_of_week_filter_config['enabled'] is True
        assert strategy._day_of_week_filter_config['allowed_days'] == [0, 1, 2, 3, 4]
        assert strategy._day_of_week_filter_config['monday_multiplier'] == 0.65
        assert strategy._day_of_week_filter_config['friday_multiplier'] == 0.75
        assert strategy._day_of_week_filter_config['weekend_allowed'] is False
    
    def test_default_config_values(self):
        """Testa valores padrão quando config não é fornecida."""
        config = {}  # Empty config
        strategy = MockStrategy(config)
        
        # Should use defaults
        assert strategy._day_of_week_filter_config['enabled'] is False
        assert strategy._day_of_week_filter_config['allowed_days'] == [0, 1, 2, 3, 4]
        assert strategy._day_of_week_filter_config['monday_multiplier'] == 1.0
        assert strategy._day_of_week_filter_config['friday_multiplier'] == 1.0
        assert strategy._day_of_week_filter_config['weekend_allowed'] is False
    
    def test_feature_pipeline_includes_temporal(self):
        """Testa que pipeline inclui features temporais."""
        from v2.src.features.pipeline import FeaturePipeline
        
        config = {
            'features': {
                'microstructure': {'enabled': False},
                'technical': {'enabled': False},
                'volume_analysis': {'enabled': False},
                'temporal': {
                    'enabled': True,
                    'day_of_week': {'enabled': True}
                }
            }
        }
        
        pipeline = FeaturePipeline(config)
        feature_names = pipeline.get_feature_names()
        
        assert 'DayOfWeek' in feature_names


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
