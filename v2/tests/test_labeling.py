"""
Tests for the Labeling Module.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
# Add parent directory to path for relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from v2.src.labeling.triple_barrier import TripleBarrierLabeler
from v2.src.labeling.meta_labeling import MetaLabeler
from v2.src.labeling.label_generator import LabelGenerator


class TestTripleBarrierLabeler:
    """Tests for TripleBarrierLabeler."""
    
    @pytest.fixture
    def config(self):
        return {
            'labeling': {
                'triple_barrier': {
                    'tp_multiplier': 2.0,
                    'sl_multiplier': 1.0,
                    'max_holding_bars': 100
                }
            }
        }
        
    @pytest.fixture
    def labeler(self, config):
        return TripleBarrierLabeler(config)
        
    def test_initialization_from_config(self, labeler):
        """Labeler deve ler parâmetros do config."""
        assert labeler.tp_mult == 2.0
        assert labeler.sl_mult == 1.0
        assert labeler.max_bars == 100
        
    def test_get_barriers(self, labeler):
        """Barriers devem ser calculadas corretamente."""
        price = 50000
        atr = 500  # 1% do preço
        
        tp, sl, max_bars = labeler.get_barriers(price, atr)
        
        # TP = 50000 * (1 + 2.0 * 500/50000) = 50000 * 1.02 = 51000
        assert tp == 51000
        # SL = 50000 * (1 - 1.0 * 500/50000) = 50000 * 0.99 = 49500
        assert sl == 49500
        assert max_bars == 100
        
    def test_get_barriers_invalid_inputs(self, labeler):
        """Deve levantar erro para inputs inválidos."""
        with pytest.raises(ValueError):
            labeler.get_barriers(0, 500)
        with pytest.raises(ValueError):
            labeler.get_barriers(50000, 0)
        with pytest.raises(ValueError):
            labeler.get_barriers(-1000, 500)
            
    def test_labels_values(self, labeler):
        """Labels devem ser -1, 0 ou 1."""
        # Cria dados de teste
        dates = pd.date_range('2024-01-01', periods=200, freq='1h')
        np.random.seed(42)
        
        # Simula movimento de preço com tendência
        prices = 50000 + np.cumsum(np.random.randn(200) * 100)
        
        data = pd.DataFrame({
            'close': prices,
            'high': prices + np.abs(np.random.randn(200) * 50),
            'low': prices - np.abs(np.random.randn(200) * 50),
            'atr': np.full(200, 500)
        }, index=dates)
        
        # Eventos para labelar
        events = pd.DataFrame(index=dates[::20][:5])
        
        labels = labeler.label(data, events)
        
        # Labels devem estar em {-1, 0, 1}
        assert all(label in [-1, 0, 1] for label in labels)
        
    def test_label_tp_hit(self, labeler):
        """Deve retornar 1 quando TP é atingido primeiro."""
        # Cria dados onde preço sobe direto
        dates = pd.date_range('2024-01-01', periods=10, freq='1h')
        
        # Preço subindo consistentemente
        data = pd.DataFrame({
            'close': [50000, 50200, 50400, 50600, 50800, 51000, 51200, 51400, 51600, 51800],
            'high': [50100, 50300, 50500, 50700, 50900, 51100, 51300, 51500, 51700, 51900],
            'low': [49900, 50100, 50300, 50500, 50700, 50900, 51100, 51300, 51500, 51700],
            'atr': [500] * 10
        }, index=dates)
        
        events = pd.DataFrame(index=[dates[0]])
        labels = labeler.label(data, events)
        
        # TP = 51000, deve ser atingido
        assert len(labels) == 1
        assert labels.loc[dates[0]] == 1
        
    def test_label_sl_hit(self, labeler):
        """Deve retornar -1 quando SL é atingido primeiro."""
        dates = pd.date_range('2024-01-01', periods=10, freq='1h')
        
        # Preço caindo consistentemente
        data = pd.DataFrame({
            'close': [50000, 49800, 49600, 49400, 49200, 49000, 48800, 48600, 48400, 48200],
            'high': [50100, 49900, 49700, 49500, 49300, 49100, 48900, 48700, 48500, 48300],
            'low': [49900, 49700, 49500, 49300, 49100, 48900, 48700, 48500, 48300, 48100],
            'atr': [500] * 10
        }, index=dates)
        
        events = pd.DataFrame(index=[dates[0]])
        labels = labeler.label(data, events)
        
        # SL = 49500, deve ser atingido
        assert len(labels) == 1
        assert labels.loc[dates[0]] == -1
        
    def test_label_timeout(self, labeler):
        """Deve retornar 0 quando max_holding_bars é atingido."""
        # Config com max_bars pequeno
        config = {
            'labeling': {
                'triple_barrier': {
                    'tp_multiplier': 10.0,  # TP muito distante
                    'sl_multiplier': 10.0,  # SL muito distante
                    'max_holding_bars': 5
                }
            }
        }
        labeler_timeout = TripleBarrierLabeler(config)
        
        dates = pd.date_range('2024-01-01', periods=10, freq='1h')
        
        # Preço lateral
        data = pd.DataFrame({
            'close': [50000, 50010, 50005, 50015, 50008, 50012, 50007, 50011, 50006, 50009],
            'high': [50020, 50030, 50025, 50035, 50028, 50032, 50027, 50031, 50026, 50029],
            'low': [49980, 49990, 49985, 49995, 49988, 49992, 49987, 49991, 49986, 49989],
            'atr': [500] * 10
        }, index=dates)
        
        events = pd.DataFrame(index=[dates[0]])
        labels = labeler_timeout.label(data, events)
        
        # Deve dar timeout
        assert len(labels) == 1
        assert labels.loc[dates[0]] == 0
        
    def test_barrier_info(self, labeler):
        """Deve retornar informações corretas das barreiras."""
        info = labeler.get_barrier_info(50000, 500)
        
        assert info['entry_price'] == 50000
        assert info['take_profit'] == 51000
        assert info['stop_loss'] == 49500
        assert info['max_holding_bars'] == 100
        assert abs(info['tp_distance_pct'] - 2.0) < 0.01
        assert abs(info['sl_distance_pct'] - 1.0) < 0.01
        assert info['risk_reward_ratio'] == 2.0
        

class TestMetaLabeler:
    """Tests for MetaLabeler."""
    
    @pytest.fixture
    def config(self):
        return {
            'labeling': {
                'meta_labeling': {
                    'model_type': 'random_forest',  # Use RF for testing (no external deps)
                    'confidence_threshold': 0.6
                }
            }
        }
        
    @pytest.fixture
    def meta_labeler(self, config):
        return MetaLabeler(config)
        
    def test_initialization_from_config(self, meta_labeler):
        """MetaLabeler deve ler parâmetros do config."""
        assert meta_labeler.model_type == 'random_forest'
        assert meta_labeler.threshold == 0.6
        
    def test_confidence_range(self, meta_labeler):
        """Confiança deve estar entre 0 e 1."""
        # Cria dados de treino sintéticos
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })
        
        primary = pd.Series(np.random.choice([-1, 0, 1], n_samples))
        actual = pd.Series(np.random.choice([-1, 0, 1], n_samples))
        
        meta_labeler.fit(X, primary, actual)
        
        # Testa predição
        test_row = pd.DataFrame({
            'feature1': [0.5],
            'feature2': [-0.3],
            'feature3': [0.1]
        })
        
        confidence = meta_labeler.predict_confidence(test_row, 1)
        
        assert 0 <= confidence <= 1
        
    def test_should_trade(self, meta_labeler):
        """should_trade deve respeitar threshold."""
        assert meta_labeler.should_trade(0.7) == True  # Acima do threshold
        assert meta_labeler.should_trade(0.6) == True  # Igual ao threshold
        assert meta_labeler.should_trade(0.5) == False  # Abaixo do threshold
        
    def test_threshold_validation(self, meta_labeler):
        """Threshold deve estar entre 0 e 1."""
        with pytest.raises(ValueError):
            meta_labeler.set_threshold(-0.1)
        with pytest.raises(ValueError):
            meta_labeler.set_threshold(1.5)
            
    def test_is_fitted_property(self, meta_labeler):
        """is_fitted deve refletir estado do modelo."""
        assert meta_labeler.is_fitted == False
        
        # Treina modelo
        np.random.seed(42)
        X = pd.DataFrame({'f1': np.random.randn(50), 'f2': np.random.randn(50)})
        primary = pd.Series(np.random.choice([-1, 0, 1], 50))
        actual = pd.Series(np.random.choice([-1, 0, 1], 50))
        
        meta_labeler.fit(X, primary, actual)
        
        assert meta_labeler.is_fitted == True
        
    def test_predict_without_fit_raises(self, meta_labeler):
        """Deve levantar erro se predizer sem treinar."""
        test_row = pd.DataFrame({'f1': [0.5], 'f2': [-0.3]})
        
        with pytest.raises(ValueError):
            meta_labeler.predict_confidence(test_row, 1)


class TestLabelGenerator:
    """Tests for LabelGenerator."""
    
    @pytest.fixture
    def config(self):
        return {
            'labeling': {
                'triple_barrier': {
                    'tp_multiplier': 2.0,
                    'sl_multiplier': 1.0,
                    'max_holding_bars': 50
                },
                'meta_labeling': {
                    'enabled': True,
                    'model_type': 'random_forest',
                    'confidence_threshold': 0.6
                }
            }
        }
        
    @pytest.fixture
    def generator(self, config):
        return LabelGenerator(config)
        
    def test_generate_labels_returns_dataframe(self, generator):
        """generate_labels deve retornar DataFrame com colunas corretas."""
        dates = pd.date_range('2024-01-01', periods=100, freq='1h')
        np.random.seed(42)
        
        prices = 50000 + np.cumsum(np.random.randn(100) * 100)
        
        data = pd.DataFrame({
            'close': prices,
            'high': prices + 50,
            'low': prices - 50,
            'atr': np.full(100, 500)
        }, index=dates)
        
        events = pd.DataFrame(index=dates[::10][:5])
        
        result = generator.generate_labels(data, events)
        
        assert isinstance(result, pd.DataFrame)
        assert 'timestamp' in result.columns
        assert 'label' in result.columns
        assert 'tp' in result.columns
        assert 'sl' in result.columns
        
    def test_get_training_data(self, generator):
        """get_training_data deve dividir dados corretamente."""
        np.random.seed(42)
        
        features = pd.DataFrame({
            'f1': np.random.randn(100),
            'f2': np.random.randn(100)
        })
        
        labels = pd.DataFrame({
            'label': np.random.choice([-1, 0, 1], 100)
        })
        
        X_train, X_test, y_train, y_test = generator.get_training_data(
            features, labels, test_size=0.2
        )
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
        
    def test_label_statistics(self, generator):
        """get_label_statistics deve retornar estatísticas corretas."""
        labels = pd.Series([1, 1, 1, -1, -1, 0])
        
        stats = generator.get_label_statistics(labels)
        
        assert stats['total'] == 6
        assert stats['profitable'] == 3
        assert stats['loss'] == 2
        assert stats['timeout'] == 1
        assert abs(stats['profitable_pct'] - 50.0) < 0.01
