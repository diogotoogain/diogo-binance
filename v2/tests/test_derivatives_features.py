"""
Tests for derivatives features (Funding Rate, Open Interest).
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from v2.src.connectors.binance_derivatives import BinanceDerivativesConnector
from v2.src.features.derivatives.funding_features import FundingRateFeatures
from v2.src.features.derivatives.open_interest_features import OpenInterestFeatures
from v2.src.strategies.filters.derivatives_filter import (
    DerivativesFilter,
    FilterResult,
    ExtremeAction
)


# ═══════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_funding_data():
    """Gera dados de funding rate de teste."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='8h')
    np.random.seed(42)
    
    # Funding rates entre -0.002 e 0.002
    funding_rate = np.random.uniform(-0.002, 0.002, 100)
    
    data = pd.DataFrame({
        'funding_rate': funding_rate,
        'funding_time': dates.astype(np.int64) // 10**6,
        'next_funding_time': (dates + pd.Timedelta(hours=8)).astype(np.int64) // 10**6
    }, index=dates)
    
    return data


@pytest.fixture
def sample_oi_data():
    """Gera dados de open interest de teste."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    np.random.seed(42)
    
    # Open interest base com tendência e ruído
    base_oi = 500000
    trend = np.linspace(0, 50000, 100)
    noise = np.random.randn(100) * 10000
    oi = base_oi + trend + noise
    
    # Preços correlacionados mas com divergências ocasionais
    base_price = 50000
    price_trend = np.linspace(0, 1000, 100)
    price_noise = np.random.randn(100) * 200
    close = base_price + price_trend + price_noise
    
    data = pd.DataFrame({
        'open_interest': oi,
        'close': close,
        'volume': np.random.rand(100) * 1000000 + 100000
    }, index=dates)
    
    return data


@pytest.fixture
def funding_config():
    """Configuração para features de Funding Rate."""
    return {
        'enabled': True,
        'extreme_positive': 0.001,
        'extreme_negative': -0.001,
        'extreme_action': 'reduce_size',
        'extreme_size_multiplier': 0.5,
        'include_in_features': True,
        'lookback_periods': 8
    }


@pytest.fixture
def oi_config():
    """Configuração para features de Open Interest."""
    return {
        'enabled': True,
        'significant_change_pct': 5.0,
        'include_in_features': True,
        'divergence_detection': {
            'enabled': True,
            'lookback_bars': 20,
            'price_change_threshold': 0.02,
            'oi_change_threshold': 0.03
        }
    }


@pytest.fixture
def derivatives_config():
    """Configuração completa de derivativos."""
    return {
        'enabled': True,
        'funding_rate': {
            'enabled': True,
            'extreme_positive': 0.001,
            'extreme_negative': -0.001,
            'extreme_action': 'reduce_size',
            'extreme_size_multiplier': 0.5,
            'include_in_features': True,
            'lookback_periods': 8
        },
        'open_interest': {
            'enabled': True,
            'significant_change_pct': 5.0,
            'include_in_features': True,
            'divergence_detection': {
                'enabled': True,
                'lookback_bars': 20,
                'price_change_threshold': 0.02,
                'oi_change_threshold': 0.03
            }
        },
        'long_short_ratio': {
            'enabled': True,
            'extreme_long': 2.0,
            'extreme_short': 0.5
        }
    }


# ═══════════════════════════════════════════════════════════════════════════
# BINANCE DERIVATIVES CONNECTOR TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestBinanceDerivativesConnector:
    """Testes para o conector de derivativos da Binance."""
    
    def test_init_default_url(self):
        """Testa inicialização com URL padrão."""
        connector = BinanceDerivativesConnector()
        assert connector.base_url == "https://fapi.binance.com"
    
    def test_init_custom_url(self):
        """Testa inicialização com URL customizada."""
        custom_url = "https://testnet.binance.com"
        connector = BinanceDerivativesConnector(base_url=custom_url)
        assert connector.base_url == custom_url
    
    @pytest.mark.asyncio
    async def test_get_funding_rate_history_mock(self):
        """Testa coleta de histórico de funding rate."""
        connector = BinanceDerivativesConnector()
        
        mock_response = [
            {'symbol': 'BTCUSDT', 'fundingRate': '0.0001', 'fundingTime': 1704067200000},
            {'symbol': 'BTCUSDT', 'fundingRate': '0.0002', 'fundingTime': 1704038400000}
        ]
        
        with patch.object(connector, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await connector.get_funding_rate_history("BTCUSDT", limit=2)
            
            assert len(result) == 2
            assert result[0]['fundingRate'] == '0.0001'
            mock_request.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_current_funding_rate_mock(self):
        """Testa obtenção de funding rate atual."""
        connector = BinanceDerivativesConnector()
        
        mock_response = [
            {'symbol': 'BTCUSDT', 'fundingRate': '0.00015', 'fundingTime': 1704067200000}
        ]
        
        with patch.object(connector, 'get_funding_rate_history', new_callable=AsyncMock) as mock_history:
            mock_history.return_value = mock_response
            
            result = await connector.get_current_funding_rate("BTCUSDT")
            
            assert result == 0.00015
    
    @pytest.mark.asyncio
    async def test_get_open_interest_mock(self):
        """Testa obtenção de open interest."""
        connector = BinanceDerivativesConnector()
        
        mock_response = {'symbol': 'BTCUSDT', 'openInterest': '50000.5'}
        
        with patch.object(connector, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await connector.get_open_interest("BTCUSDT")
            
            assert result == 50000.5
    
    @pytest.mark.asyncio
    async def test_get_long_short_ratio_mock(self):
        """Testa obtenção de long/short ratio."""
        connector = BinanceDerivativesConnector()
        
        mock_response = [
            {'symbol': 'BTCUSDT', 'longShortRatio': '1.5', 'timestamp': 1704067200000}
        ]
        
        with patch.object(connector, '_request', new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_response
            
            result = await connector.get_long_short_ratio("BTCUSDT", limit=1)
            
            assert len(result) == 1
            assert result[0]['longShortRatio'] == '1.5'


# ═══════════════════════════════════════════════════════════════════════════
# FUNDING RATE FEATURES TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestFundingRateFeatures:
    """Testes para features de Funding Rate."""
    
    def test_init_params(self, funding_config):
        """Testa inicialização com parâmetros do config."""
        fr = FundingRateFeatures(funding_config, enabled=True)
        
        assert fr.extreme_positive == 0.001
        assert fr.extreme_negative == -0.001
        assert fr.lookback_periods == 8
        assert fr.include_in_features is True
    
    def test_calculate_batch(self, funding_config, sample_funding_data):
        """Testa cálculo batch de features."""
        fr = FundingRateFeatures(funding_config, enabled=True)
        result = fr.calculate(sample_funding_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_funding_data)
        
        # Verifica colunas geradas
        expected_columns = [
            'funding_rate_current',
            'funding_rate_avg_24h',
            'funding_rate_avg',
            'funding_rate_trend',
            'funding_rate_extreme',
            'funding_rate_extreme_direction',
            'funding_rate_zscore'
        ]
        for col in expected_columns:
            assert col in result.columns
    
    def test_calculate_incremental(self, funding_config):
        """Testa cálculo incremental de features."""
        fr = FundingRateFeatures(funding_config, enabled=True)
        state = {}
        
        # Simula 10 períodos de funding
        for i in range(10):
            funding_rate = 0.0001 * (i - 5)  # Varia de -0.0005 a 0.0004
            result = fr.calculate_incremental({'funding_rate': funding_rate}, state)
            
            assert 'funding_rate_current' in result
            assert 'funding_rate_extreme' in result
    
    def test_is_extreme_positive(self, funding_config):
        """Testa detecção de funding rate extremo positivo."""
        fr = FundingRateFeatures(funding_config, enabled=True)
        
        assert fr.is_extreme_positive(0.0015) is True
        assert fr.is_extreme_positive(0.001) is True
        assert fr.is_extreme_positive(0.0005) is False
    
    def test_is_extreme_negative(self, funding_config):
        """Testa detecção de funding rate extremo negativo."""
        fr = FundingRateFeatures(funding_config, enabled=True)
        
        assert fr.is_extreme_negative(-0.0015) is True
        assert fr.is_extreme_negative(-0.001) is True
        assert fr.is_extreme_negative(-0.0005) is False
    
    def test_get_signal(self, funding_config):
        """Testa geração de sinais baseados em funding rate."""
        fr = FundingRateFeatures(funding_config, enabled=True)
        
        # Funding muito positivo = sinal short
        assert fr.get_signal(0.002) == -1
        
        # Funding muito negativo = sinal long
        assert fr.get_signal(-0.002) == 1
        
        # Funding neutro = sem sinal
        assert fr.get_signal(0.0005) == 0
    
    def test_disabled_returns_empty(self, sample_funding_data):
        """Feature desabilitada deve retornar DataFrame vazio."""
        fr = FundingRateFeatures({'enabled': False}, enabled=False)
        result = fr.calculate(sample_funding_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 0


# ═══════════════════════════════════════════════════════════════════════════
# OPEN INTEREST FEATURES TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestOpenInterestFeatures:
    """Testes para features de Open Interest."""
    
    def test_init_params(self, oi_config):
        """Testa inicialização com parâmetros do config."""
        oi = OpenInterestFeatures(oi_config, enabled=True)
        
        assert oi.significant_change_pct == 5.0
        assert oi.divergence_enabled is True
        assert oi.divergence_lookback == 20
        assert oi.price_change_threshold == 0.02
        assert oi.oi_change_threshold == 0.03
    
    def test_calculate_batch(self, oi_config, sample_oi_data):
        """Testa cálculo batch de features."""
        oi = OpenInterestFeatures(oi_config, enabled=True)
        result = oi.calculate(sample_oi_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_oi_data)
        
        # Verifica colunas geradas
        expected_columns = [
            'oi_current',
            'oi_change_pct',
            'oi_significant_change',
            'oi_sma',
            'oi_trend',
            'oi_zscore',
            'oi_price_divergence'
        ]
        for col in expected_columns:
            assert col in result.columns
    
    def test_calculate_incremental(self, oi_config):
        """Testa cálculo incremental de features."""
        oi = OpenInterestFeatures(oi_config, enabled=True)
        state = {}
        
        # Simula 30 períodos
        for i in range(30):
            open_interest = 500000 + i * 1000
            close = 50000 + i * 10
            
            result = oi.calculate_incremental({
                'open_interest': open_interest,
                'close': close
            }, state)
            
            assert 'oi_current' in result
            assert 'oi_change_pct' in result
    
    def test_detect_divergence_bearish(self, oi_config):
        """Testa detecção de divergência bearish (preço sobe, OI cai)."""
        oi = OpenInterestFeatures(oi_config, enabled=True)
        
        # Preço sobe 5%, OI cai 5% - divergência bearish
        has_div, direction, desc = oi.detect_divergence(5.0, -5.0)
        
        assert has_div is True
        assert direction == -1
        assert "Bearish" in desc
    
    def test_detect_divergence_bullish(self, oi_config):
        """Testa detecção de divergência bullish (preço cai, OI sobe)."""
        oi = OpenInterestFeatures(oi_config, enabled=True)
        
        # Preço cai 5%, OI sobe 5% - divergência bullish
        has_div, direction, desc = oi.detect_divergence(-5.0, 5.0)
        
        assert has_div is True
        assert direction == 1
        assert "Bullish" in desc
    
    def test_no_divergence(self, oi_config):
        """Testa quando não há divergência."""
        oi = OpenInterestFeatures(oi_config, enabled=True)
        
        # Preço e OI sobem juntos - sem divergência
        has_div, direction, desc = oi.detect_divergence(3.0, 3.0)
        
        assert has_div is False
        assert direction == 0
    
    def test_get_signal(self, oi_config):
        """Testa geração de sinais baseados em OI."""
        oi = OpenInterestFeatures(oi_config, enabled=True)
        
        # Divergência bearish
        assert oi.get_signal(oi_change_pct=-5.0, price_change_pct=5.0) == -1
        
        # Divergência bullish
        assert oi.get_signal(oi_change_pct=5.0, price_change_pct=-5.0) == 1
        
        # OI crescendo muito (força)
        assert oi.get_signal(oi_change_pct=10.0) == 1
        
        # OI caindo muito (exaustão)
        assert oi.get_signal(oi_change_pct=-10.0) == -1
    
    def test_disabled_returns_empty(self, sample_oi_data):
        """Feature desabilitada deve retornar DataFrame vazio."""
        oi = OpenInterestFeatures({'enabled': False}, enabled=False)
        result = oi.calculate(sample_oi_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 0


# ═══════════════════════════════════════════════════════════════════════════
# DERIVATIVES FILTER TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestDerivativesFilter:
    """Testes para o filtro de derivativos."""
    
    def test_init_params(self, derivatives_config):
        """Testa inicialização com parâmetros do config."""
        filter = DerivativesFilter(derivatives_config)
        
        assert filter.enabled is True
        assert filter.fr_extreme_positive == 0.001
        assert filter.fr_extreme_negative == -0.001
        assert filter.ls_extreme_long == 2.0
        assert filter.ls_extreme_short == 0.5
    
    def test_should_trade_normal_conditions(self, derivatives_config):
        """Testa decisão de trade em condições normais."""
        filter = DerivativesFilter(derivatives_config)
        
        derivatives_data = {
            'funding_rate': 0.0005,  # Normal
            'open_interest': 500000,
            'long_short_ratio': 1.0,
            'price_change_pct': 1.0,
            'oi_change_pct': 1.0
        }
        
        allowed, multiplier, reason = filter.should_trade("BUY", derivatives_data)
        
        assert allowed is True
        assert multiplier == 1.0
    
    def test_should_trade_extreme_positive_funding_long(self, derivatives_config):
        """Testa rejeição de long quando funding muito positivo."""
        filter = DerivativesFilter(derivatives_config)
        
        derivatives_data = {
            'funding_rate': 0.002,  # Muito positivo
            'long_short_ratio': 1.0,
            'price_change_pct': 0.0,
            'oi_change_pct': 0.0
        }
        
        allowed, multiplier, reason = filter.should_trade("BUY", derivatives_data)
        
        # Deve reduzir o tamanho do long
        assert allowed is True
        assert multiplier < 1.0
        assert "FR=" in reason
    
    def test_should_trade_extreme_positive_funding_short(self, derivatives_config):
        """Testa favorecimento de short quando funding muito positivo."""
        filter = DerivativesFilter(derivatives_config)
        
        derivatives_data = {
            'funding_rate': 0.002,  # Muito positivo - favorece shorts
            'long_short_ratio': 1.0,
            'price_change_pct': 0.0,
            'oi_change_pct': 0.0
        }
        
        allowed, multiplier, reason = filter.should_trade("SELL", derivatives_data)
        
        assert allowed is True
        assert multiplier >= 1.0  # Shorts favorecidos
    
    def test_should_trade_extreme_negative_funding(self, derivatives_config):
        """Testa comportamento com funding muito negativo."""
        filter = DerivativesFilter(derivatives_config)
        
        derivatives_data = {
            'funding_rate': -0.002,  # Muito negativo - favorece longs
            'long_short_ratio': 1.0,
            'price_change_pct': 0.0,
            'oi_change_pct': 0.0
        }
        
        # Long favorecido
        allowed_long, mult_long, _ = filter.should_trade("BUY", derivatives_data)
        assert allowed_long is True
        assert mult_long >= 1.0
        
        # Short reduzido
        allowed_short, mult_short, _ = filter.should_trade("SELL", derivatives_data)
        assert allowed_short is True
        assert mult_short < 1.0
    
    def test_should_trade_bearish_divergence(self, derivatives_config):
        """Testa comportamento com divergência bearish."""
        filter = DerivativesFilter(derivatives_config)
        
        derivatives_data = {
            'funding_rate': 0.0005,
            'long_short_ratio': 1.0,
            'price_change_pct': 5.0,  # Preço subindo
            'oi_change_pct': -5.0     # OI caindo
        }
        
        # Long deve ser reduzido
        allowed, multiplier, reason = filter.should_trade("BUY", derivatives_data)
        assert allowed is True
        assert multiplier < 1.0
    
    def test_should_trade_extreme_long_short_ratio(self, derivatives_config):
        """Testa comportamento com L/S ratio extremo."""
        filter = DerivativesFilter(derivatives_config)
        
        # Muitos longs
        derivatives_data = {
            'funding_rate': 0.0005,
            'long_short_ratio': 2.5,  # Muito mais longs
            'price_change_pct': 0.0,
            'oi_change_pct': 0.0
        }
        
        allowed_long, mult_long, _ = filter.should_trade("BUY", derivatives_data)
        assert mult_long < 1.0  # Reduz longs
        
        allowed_short, mult_short, _ = filter.should_trade("SELL", derivatives_data)
        # Note: Filter uses min() for all multipliers so boosts are capped at 1.0
        # The L/S ratio result is stored in metadata for potential use
        assert allowed_short is True
        assert mult_short >= 1.0  # Shorts allowed (not reduced)
    
    def test_get_market_bias(self, derivatives_config):
        """Testa cálculo de viés de mercado."""
        filter = DerivativesFilter(derivatives_config)
        
        # Mercado muito bullish (deve gerar bias bearish)
        derivatives_data = {
            'funding_rate': 0.002,  # Muito longs
            'long_short_ratio': 2.5,  # Muitos longs
            'price_change_pct': 0.0,
            'oi_change_pct': 0.0
        }
        
        bias, explanation = filter.get_market_bias(derivatives_data)
        assert bias < 0  # Bearish bias
        
        # Mercado muito bearish (deve gerar bias bullish)
        derivatives_data_bearish = {
            'funding_rate': -0.002,  # Muito shorts
            'long_short_ratio': 0.4,  # Muitos shorts
            'price_change_pct': 0.0,
            'oi_change_pct': 0.0
        }
        
        bias_bearish, _ = filter.get_market_bias(derivatives_data_bearish)
        assert bias_bearish > 0  # Bullish bias
    
    def test_filter_disabled(self, derivatives_config):
        """Testa filtro desabilitado."""
        derivatives_config['enabled'] = False
        filter = DerivativesFilter(derivatives_config)
        
        derivatives_data = {
            'funding_rate': 0.002,  # Seria bloqueado se habilitado
            'long_short_ratio': 3.0
        }
        
        allowed, multiplier, reason = filter.should_trade("BUY", derivatives_data)
        
        assert allowed is True
        assert multiplier == 1.0
        assert "disabled" in reason.lower()
    
    def test_pause_action(self):
        """Testa ação de pausa quando funding extremo."""
        config = {
            'enabled': True,
            'funding_rate': {
                'enabled': True,
                'extreme_positive': 0.001,
                'extreme_negative': -0.001,
                'extreme_action': 'pause',  # Pausa em vez de reduzir
                'extreme_size_multiplier': 0.5
            },
            'open_interest': {'enabled': False},
            'long_short_ratio': {'enabled': False}
        }
        
        filter = DerivativesFilter(config)
        
        derivatives_data = {
            'funding_rate': 0.002,  # Muito positivo
        }
        
        allowed, multiplier, reason = filter.should_trade("BUY", derivatives_data)
        
        assert allowed is False
        assert multiplier == 0.0
        assert "pausing" in reason.lower()


# ═══════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestDerivativesIntegration:
    """Testes de integração entre features e filtros."""
    
    def test_funding_features_with_filter(self, funding_config, derivatives_config):
        """Testa uso de features de funding com filtro."""
        fr = FundingRateFeatures(funding_config, enabled=True)
        filter = DerivativesFilter(derivatives_config)
        
        # Simula cenário com funding extremo
        state = {}
        for i in range(10):
            result = fr.calculate_incremental({'funding_rate': 0.002}, state)
        
        # Usa o sinal da feature para decidir
        signal = fr.get_signal(0.002)
        assert signal == -1  # Deve ser short
        
        # Filtro deve reduzir longs
        allowed, mult, _ = filter.should_trade("BUY", {'funding_rate': 0.002, 'long_short_ratio': 1.0})
        assert mult < 1.0
    
    def test_oi_features_with_filter(self, oi_config, derivatives_config):
        """Testa uso de features de OI com filtro."""
        oi = OpenInterestFeatures(oi_config, enabled=True)
        filter = DerivativesFilter(derivatives_config)
        
        # Simula divergência bearish
        has_div, direction, _ = oi.detect_divergence(5.0, -5.0)
        assert has_div is True
        assert direction == -1
        
        # Filtro deve reduzir longs
        derivatives_data = {
            'funding_rate': 0.0005,
            'long_short_ratio': 1.0,
            'price_change_pct': 5.0,
            'oi_change_pct': -5.0
        }
        
        allowed, mult, _ = filter.should_trade("BUY", derivatives_data)
        assert mult < 1.0


# ═══════════════════════════════════════════════════════════════════════════
# CONFIG PARAMS TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestConfigParams:
    """Testa que todos os parâmetros vêm do config - ZERO hardcoded."""
    
    def test_funding_uses_config_thresholds(self):
        """Funding Rate usa thresholds do config."""
        config1 = {'extreme_positive': 0.002, 'extreme_negative': -0.002}
        fr1 = FundingRateFeatures(config1, enabled=True)
        
        config2 = {'extreme_positive': 0.0005, 'extreme_negative': -0.0005}
        fr2 = FundingRateFeatures(config2, enabled=True)
        
        # Com threshold maior, 0.001 não é extremo
        assert fr1.is_extreme_positive(0.001) is False
        
        # Com threshold menor, 0.001 é extremo
        assert fr2.is_extreme_positive(0.001) is True
    
    def test_oi_uses_config_lookback(self):
        """Open Interest usa lookback do config."""
        config1 = {
            'divergence_detection': {
                'enabled': True,
                'lookback_bars': 10,
                'price_change_threshold': 0.02,
                'oi_change_threshold': 0.03
            }
        }
        oi1 = OpenInterestFeatures(config1, enabled=True)
        
        config2 = {
            'divergence_detection': {
                'enabled': True,
                'lookback_bars': 50,
                'price_change_threshold': 0.02,
                'oi_change_threshold': 0.03
            }
        }
        oi2 = OpenInterestFeatures(config2, enabled=True)
        
        assert oi1.divergence_lookback == 10
        assert oi2.divergence_lookback == 50
    
    def test_filter_uses_config_all_params(self):
        """Filtro usa todos os parâmetros do config."""
        config = {
            'enabled': True,
            'funding_rate': {
                'enabled': True,
                'extreme_positive': 0.005,  # Diferente do padrão
                'extreme_negative': -0.005,
                'extreme_action': 'pause',  # Diferente do padrão
                'extreme_size_multiplier': 0.3
            },
            'open_interest': {
                'enabled': True,
                'significant_change_pct': 10.0,
                'divergence_detection': {
                    'enabled': True,
                    'price_change_threshold': 0.05,
                    'oi_change_threshold': 0.05
                }
            },
            'long_short_ratio': {
                'enabled': True,
                'extreme_long': 3.0,
                'extreme_short': 0.3
            }
        }
        
        filter = DerivativesFilter(config)
        
        assert filter.fr_extreme_positive == 0.005
        assert filter.fr_extreme_negative == -0.005
        assert filter.fr_extreme_action == ExtremeAction.PAUSE
        assert filter.fr_extreme_size_multiplier == 0.3
        assert filter.ls_extreme_long == 3.0
        assert filter.ls_extreme_short == 0.3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
