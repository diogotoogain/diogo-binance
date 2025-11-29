"""
Tests for the Regime Detection Module.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
# Add parent directory to path for relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v2.src.regime.hmm_detector import HMMRegimeDetector
from v2.src.regime.adx_regime import ADXRegimeDetector
from v2.src.regime.volatility_regime import VolatilityRegimeDetector
from v2.src.regime.regime_manager import RegimeManager
from v2.src.labeling.triple_barrier import TripleBarrierLabeler


class TestHMMRegimeDetector:
    """Tests for HMMRegimeDetector."""
    
    @pytest.fixture
    def config(self):
        return {
            'regime': {
                'hmm': {
                    'enabled': True,
                    'n_regimes': 3,
                    'features': ['returns', 'volatility'],
                    'covariance_type': 'diag',  # More stable for testing
                    'n_iter': 50,
                    'random_state': 42
                }
            }
        }
        
    @pytest.fixture
    def detector(self, config):
        return HMMRegimeDetector(config)
        
    def test_initialization_from_config(self, detector):
        """HMM deve ler parâmetros do config."""
        assert detector.n_regimes == 3
        assert detector.features == ['returns', 'volatility']
        assert detector.covariance_type == 'diag'
        
    def test_n_regimes_configurable(self):
        """Deve suportar diferentes números de regimes."""
        for n in [2, 3, 4]:
            config = {
                'regime': {
                    'hmm': {
                        'n_regimes': n,
                        'features': ['returns', 'volatility'],
                        'random_state': 42
                    }
                }
            }
            detector = HMMRegimeDetector(config)
            assert detector.n_regimes == n
            
    def test_predict_returns_valid_regime(self, detector):
        """HMM deve retornar regime válido (0 a n_regimes-1)."""
        # Cria dados de treino com mais variação
        np.random.seed(42)
        n_samples = 500
        
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='1h')
        # Cria dados com diferentes regimes (trending, ranging)
        returns = np.concatenate([
            np.random.randn(200) * 0.001 + 0.001,  # Trending up
            np.random.randn(100) * 0.002,           # High vol
            np.random.randn(200) * 0.0005           # Low vol
        ])
        prices = 50000 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'close': prices,
            'volume': np.abs(np.random.randn(n_samples) * 1000) + 500
        }, index=dates)
        
        detector.fit(data)
        regime = detector.predict(data)
        
        assert 0 <= regime < detector.n_regimes
        
    def test_predict_proba_sums_to_one(self, detector):
        """Probabilidades devem somar 1."""
        np.random.seed(42)
        n_samples = 500
        
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='1h')
        # Cria dados com diferentes regimes
        returns = np.concatenate([
            np.random.randn(200) * 0.001 + 0.001,
            np.random.randn(100) * 0.002,
            np.random.randn(200) * 0.0005
        ])
        prices = 50000 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'close': prices,
            'volume': np.abs(np.random.randn(n_samples) * 1000) + 500
        }, index=dates)
        
        detector.fit(data)
        proba = detector.predict_proba(data)
        
        # Última linha de probabilidades deve somar ~1
        assert abs(proba[-1].sum() - 1.0) < 0.01
        
    def test_regime_stats(self, detector):
        """get_regime_stats deve retornar estatísticas válidas."""
        np.random.seed(42)
        n_samples = 500
        
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='1h')
        # Cria dados com diferentes regimes
        returns = np.concatenate([
            np.random.randn(200) * 0.001 + 0.001,
            np.random.randn(100) * 0.002,
            np.random.randn(200) * 0.0005
        ])
        prices = 50000 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'close': prices,
            'volume': np.abs(np.random.randn(n_samples) * 1000) + 500
        }, index=dates)
        
        detector.fit(data)
        stats = detector.get_regime_stats()
        
        assert len(stats) == detector.n_regimes
        for regime in range(detector.n_regimes):
            assert regime in stats
            assert 'count' in stats[regime]
            assert 'mean_return' in stats[regime]
            assert 'volatility' in stats[regime]
            
    def test_predict_without_fit_raises(self, detector):
        """Deve levantar erro se predizer sem treinar."""
        data = pd.DataFrame({'close': [50000], 'volume': [1000]})
        
        with pytest.raises(ValueError):
            detector.predict(data)


class TestADXRegimeDetector:
    """Tests for ADXRegimeDetector."""
    
    @pytest.fixture
    def config(self):
        return {
            'regime': {
                'adx_based': {
                    'trending_threshold': 25,
                    'ranging_threshold': 20
                }
            }
        }
        
    @pytest.fixture
    def detector(self, config):
        return ADXRegimeDetector(config)
        
    def test_initialization_from_config(self, detector):
        """ADX deve ler thresholds do config."""
        assert detector.trending_threshold == 25
        assert detector.ranging_threshold == 20
        
    def test_detect_trending(self, detector):
        """ADX >= trending_threshold deve retornar 'trending'."""
        assert detector.detect(25) == "trending"
        assert detector.detect(30) == "trending"
        assert detector.detect(50) == "trending"
        
    def test_detect_ranging(self, detector):
        """ADX <= ranging_threshold deve retornar 'ranging'."""
        assert detector.detect(20) == "ranging"
        assert detector.detect(15) == "ranging"
        assert detector.detect(10) == "ranging"
        
    def test_detect_transition(self, detector):
        """ADX entre thresholds deve retornar 'transition'."""
        assert detector.detect(21) == "transition"
        assert detector.detect(22) == "transition"
        assert detector.detect(24) == "transition"
        
    def test_detect_none_returns_transition(self, detector):
        """ADX None deve retornar 'transition'."""
        assert detector.detect(None) == "transition"
        
    def test_thresholds_validation(self):
        """ranging_threshold deve ser menor que trending_threshold."""
        config = {
            'regime': {
                'adx_based': {
                    'trending_threshold': 20,
                    'ranging_threshold': 25  # Inválido!
                }
            }
        }
        
        with pytest.raises(ValueError):
            ADXRegimeDetector(config)
            
    def test_strategy_filter(self, detector):
        """get_strategy_filter deve retornar filtros corretos."""
        # Em trending, momentum deve estar ativo
        trending_filter = detector.get_strategy_filter('trending')
        assert trending_filter['momentum'] == True
        assert trending_filter['mean_reversion'] == False
        
        # Em ranging, mean_reversion deve estar ativo
        ranging_filter = detector.get_strategy_filter('ranging')
        assert ranging_filter['momentum'] == False
        assert ranging_filter['mean_reversion'] == True
        
    def test_regime_strength(self, detector):
        """get_regime_strength deve retornar valor entre 0 e 1."""
        # Trending
        strength = detector.get_regime_strength(30)
        assert 0 <= strength <= 1
        
        # Ranging
        strength = detector.get_regime_strength(10)
        assert 0 <= strength <= 1
        
        # Transition
        strength = detector.get_regime_strength(22)
        assert 0 <= strength <= 1
        
    def test_is_trending_is_ranging(self, detector):
        """Helpers devem funcionar corretamente."""
        assert detector.is_trending(30) == True
        assert detector.is_trending(15) == False
        
        assert detector.is_ranging(15) == True
        assert detector.is_ranging(30) == False


class TestVolatilityRegimeDetector:
    """Tests for VolatilityRegimeDetector."""
    
    @pytest.fixture
    def config(self):
        return {
            'regime': {
                'volatility': {
                    'lookback': 100,
                    'low_percentile': 25,
                    'high_percentile': 75,
                    'extreme_percentile': 95
                }
            }
        }
        
    @pytest.fixture
    def detector(self, config):
        return VolatilityRegimeDetector(config)
        
    def test_initialization_from_config(self, detector):
        """Volatility detector deve ler parâmetros do config."""
        assert detector.lookback == 100
        assert detector.low_percentile == 25
        assert detector.high_percentile == 75
        
    def test_detect_regimes(self, detector):
        """Deve detectar regimes corretamente após fit."""
        np.random.seed(42)
        
        # Cria série de volatilidade com distribuição conhecida
        volatility = pd.Series(np.random.uniform(0, 1, 200))
        
        detector.fit(volatility)
        
        # Testa detecção
        thresholds = detector.get_thresholds()
        
        assert detector.detect(thresholds['low'] - 0.01) == "low_vol"
        assert detector.detect(thresholds['low'] + 0.01) == "normal_vol"
        assert detector.detect(thresholds['high'] + 0.01) == "high_vol"
        assert detector.detect(thresholds['extreme'] + 0.01) == "extreme_vol"
        
    def test_detect_without_fit_raises(self, detector):
        """Deve levantar erro se detectar sem fit."""
        with pytest.raises(ValueError):
            detector.detect(0.5)
            
    def test_regime_multipliers(self, detector):
        """get_regime_multiplier deve retornar valores corretos."""
        assert detector.get_regime_multiplier("low_vol") == 1.2
        assert detector.get_regime_multiplier("normal_vol") == 1.0
        assert detector.get_regime_multiplier("high_vol") == 0.6
        assert detector.get_regime_multiplier("extreme_vol") == 0.3
        
    def test_is_fitted_property(self, detector):
        """is_fitted deve refletir estado."""
        assert detector.is_fitted == False
        
        volatility = pd.Series(np.random.uniform(0, 1, 100))
        detector.fit(volatility)
        
        assert detector.is_fitted == True


class TestRegimeManager:
    """Tests for RegimeManager."""
    
    @pytest.fixture
    def config(self):
        return {
            'regime': {
                'hmm': {
                    'enabled': True,
                    'n_regimes': 3,
                    'features': ['returns', 'volatility'],
                    'covariance_type': 'diag',  # More stable for testing
                    'random_state': 42
                },
                'adx_based': {
                    'enabled': True,
                    'trending_threshold': 25,
                    'ranging_threshold': 20
                },
                'volatility': {
                    'enabled': True,
                    'lookback': 100
                }
            },
            'risk_adjustment': {
                'favorable_multiplier': 1.0,
                'neutral_multiplier': 0.7,
                'unfavorable_multiplier': 0.3
            },
            'strategy_weights': {
                'trending': {'momentum': 0.6, 'mean_reversion': 0.1},
                'ranging': {'momentum': 0.1, 'mean_reversion': 0.6},
                'transition': {'momentum': 0.35, 'mean_reversion': 0.35}
            }
        }
        
    @pytest.fixture
    def manager(self, config):
        return RegimeManager(config)
        
    def test_initialization(self, manager):
        """Manager deve inicializar todos os detectores."""
        assert manager.hmm is not None
        assert manager.adx is not None
        assert manager.vol is not None
        
    def test_disabled_detectors(self):
        """Detectores desabilitados devem ser None."""
        config = {
            'regime': {
                'hmm': {'enabled': False},
                'adx_based': {'enabled': False},
                'volatility': {'enabled': False}
            }
        }
        manager = RegimeManager(config)
        
        assert manager.hmm is None
        assert manager.adx is None
        assert manager.vol is None
        
    def test_get_current_regime_combines_all(self, manager):
        """get_current_regime deve combinar todos os detectores."""
        np.random.seed(42)
        n_samples = 500
        
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='1h')
        # Cria dados com diferentes regimes
        returns = np.concatenate([
            np.random.randn(200) * 0.001 + 0.001,
            np.random.randn(100) * 0.002,
            np.random.randn(200) * 0.0005
        ])
        prices = 50000 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'close': prices,
            'high': prices + 50,
            'low': prices - 50,
            'volume': np.abs(np.random.randn(n_samples) * 1000) + 500,
            'adx': np.random.uniform(15, 35, n_samples)
        }, index=dates)
        
        # Fit manager
        manager.fit(data)
        
        # Get regime
        regime = manager.get_current_regime(data)
        
        assert 'hmm' in regime
        assert 'adx' in regime
        assert 'volatility' in regime
        assert 'combined' in regime
        assert regime['combined'] in ['favorable', 'neutral', 'unfavorable']
        
    def test_risk_multiplier(self, manager):
        """get_risk_multiplier deve retornar valores do config."""
        assert manager.get_risk_multiplier({'combined': 'favorable'}) == 1.0
        assert manager.get_risk_multiplier({'combined': 'neutral'}) == 0.7
        assert manager.get_risk_multiplier({'combined': 'unfavorable'}) == 0.3
        
    def test_strategy_weights(self, manager):
        """get_strategy_weights deve retornar pesos por regime."""
        weights_trending = manager.get_strategy_weights({'adx': 'trending'})
        assert weights_trending['momentum'] == 0.6
        
        weights_ranging = manager.get_strategy_weights({'adx': 'ranging'})
        assert weights_ranging['mean_reversion'] == 0.6
        
    def test_should_reduce_exposure(self, manager):
        """should_reduce_exposure deve funcionar."""
        np.random.seed(42)
        n_samples = 500
        
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='1h')
        # Cria dados com diferentes regimes
        returns = np.concatenate([
            np.random.randn(200) * 0.001 + 0.001,
            np.random.randn(100) * 0.002,
            np.random.randn(200) * 0.0005
        ])
        prices = 50000 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'close': prices,
            'volume': np.abs(np.random.randn(n_samples) * 1000) + 500,
            'adx': np.full(n_samples, 15)  # Ranging
        }, index=dates)
        
        # Cria série de volatilidade com valores altos
        high_vol = pd.Series(np.random.uniform(0.8, 1.0, 200))
        manager.vol.fit(high_vol)
        
        # Adiciona volatilidade alta aos dados
        data['volatility'] = np.full(n_samples, 0.95)
        
        # Fit HMM
        manager.hmm.fit(data)
        
        # Deve recomendar redução de exposição (ranging + high vol)
        result = manager.should_reduce_exposure(data)
        # Result depende da combinação, mas não deve levantar erro
        assert isinstance(result, bool)


class TestConfigIntegration:
    """Tests for config file integration."""
    
    def test_load_config_from_yaml(self):
        """Deve carregar configuração do arquivo YAML."""
        import yaml
        
        # Use relative path from test file location
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'config', 'default.yaml'
        )
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Verifica estrutura do labeling
        assert 'labeling' in config
        assert 'triple_barrier' in config['labeling']
        assert 'meta_labeling' in config['labeling']
        
        # Verifica estrutura do regime
        assert 'regime' in config
        assert 'hmm' in config['regime']
        assert 'adx_based' in config['regime']
        assert 'volatility' in config['regime']
        
        # Cria instâncias com config do arquivo
        labeler = TripleBarrierLabeler(config)
        hmm = HMMRegimeDetector(config)
        adx = ADXRegimeDetector(config)
        vol = VolatilityRegimeDetector(config)
        manager = RegimeManager(config)
        
        # Verifica valores lidos do config
        assert labeler.tp_mult == 2.0
        assert labeler.sl_mult == 1.0
        assert hmm.n_regimes == 3
        assert adx.trending_threshold == 25
        assert vol.lookback == 100
