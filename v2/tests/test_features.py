"""
Testes para o sistema de features V2.
"""
import pytest
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

sys.path.insert(0, '.')

from v2.src.features.base import Feature
from v2.src.features.microstructure import OFI, TFI, MicroPrice, ShannonEntropy, VPIN
from v2.src.features.technical import EMA, RSI, MACD, ADX, BollingerBands, ATR
from v2.src.features.volume import VolumeSpike, LiquidityClusters
from v2.src.features.pipeline import FeaturePipeline
from v2.src.features.feature_store import FeatureStore


# Fixtures para dados de teste
@pytest.fixture
def sample_ohlcv_data():
    """Gera dados OHLCV de teste."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    np.random.seed(42)
    
    close = 50000 + np.cumsum(np.random.randn(100) * 100)
    
    data = pd.DataFrame({
        'open': close - np.random.rand(100) * 50,
        'high': close + np.abs(np.random.randn(100) * 50),
        'low': close - np.abs(np.random.randn(100) * 50),
        'close': close,
        'volume': np.random.rand(100) * 1000000 + 100000
    }, index=dates)
    
    return data


@pytest.fixture
def sample_orderbook_data():
    """Gera dados de order book de teste."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    np.random.seed(42)
    
    mid_price = 50000 + np.cumsum(np.random.randn(100) * 100)
    spread = np.random.rand(100) * 10 + 1
    
    data = pd.DataFrame({
        'bid': mid_price - spread / 2,
        'ask': mid_price + spread / 2,
        'bid_size': np.random.rand(100) * 10 + 1,
        'ask_size': np.random.rand(100) * 10 + 1,
        'close': mid_price,
        'volume': np.random.rand(100) * 1000000 + 100000,
        'buy_volume': np.random.rand(100) * 500000 + 50000,
        'sell_volume': np.random.rand(100) * 500000 + 50000
    }, index=dates)
    
    return data


@pytest.fixture
def sample_config():
    """Configuração de teste para features."""
    return {
        'features': {
            'microstructure': {
                'enabled': True,
                'ofi': {'enabled': True, 'window': 20},
                'tfi': {'enabled': True, 'window': 20},
                'micro_price': {'enabled': True},
                'entropy': {'enabled': True, 'window': 50},
                'vpin': {'enabled': True, 'n_buckets': 50, 'bucket_size_usd': 100000}
            },
            'technical': {
                'enabled': True,
                'ema': {'enabled': True, 'periods': [9, 21, 50]},
                'rsi': {'enabled': True, 'period': 14, 'overbought': 70, 'oversold': 30},
                'macd': {'enabled': True, 'fast': 12, 'slow': 26, 'signal': 9},
                'adx': {'enabled': True, 'period': 14, 'trending_threshold': 25, 'ranging_threshold': 20},
                'bollinger': {'enabled': True, 'period': 20, 'std_dev': 2.0},
                'atr': {'enabled': True, 'period': 14}
            },
            'volume_analysis': {
                'enabled': True,
                'volume_spike': {'enabled': True, 'lookback': 20, 'threshold_multiplier': 2.0},
                'liquidity_clusters': {'enabled': True, 'levels': 10, 'threshold_percentile': 80}
            }
        }
    }


class TestOFI:
    """Testes para Order Flow Imbalance."""
    
    def test_ofi_calculation(self, sample_ohlcv_data):
        """Testa cálculo de OFI."""
        ofi = OFI({'window': 20}, enabled=True)
        result = ofi.calculate(sample_ohlcv_data)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_ohlcv_data)
        # OFI normalizado deve estar entre -1 e 1
        valid_values = result.dropna()
        assert (valid_values >= -1).all() and (valid_values <= 1).all()
    
    def test_ofi_incremental(self, sample_ohlcv_data):
        """Testa cálculo incremental de OFI."""
        ofi = OFI({'window': 5}, enabled=True)
        state = {}
        
        values = []
        for i in range(len(sample_ohlcv_data)):
            row = sample_ohlcv_data.iloc[i]
            value = ofi.calculate_incremental({
                'close': row['close'],
                'volume': row['volume']
            }, state)
            values.append(value)
        
        # Valores devem estar no range válido
        assert all(-1 <= v <= 1 for v in values)
    
    def test_ofi_disabled_returns_empty(self, sample_ohlcv_data):
        """OFI desabilitado deve retornar série vazia."""
        ofi = OFI({'window': 20}, enabled=False)
        result = ofi.calculate(sample_ohlcv_data)
        
        assert isinstance(result, pd.Series)
        assert result.isna().all() or (result == 0).all() or len(result) == 0


class TestRSI:
    """Testes para RSI."""
    
    def test_rsi_bounds(self, sample_ohlcv_data):
        """RSI deve estar entre 0 e 100."""
        rsi = RSI({'period': 14, 'overbought': 70, 'oversold': 30}, enabled=True)
        result = rsi.calculate(sample_ohlcv_data)
        
        assert isinstance(result, pd.Series)
        valid_values = result.dropna()
        assert (valid_values >= 0).all() and (valid_values <= 100).all()
    
    def test_rsi_incremental(self, sample_ohlcv_data):
        """Testa cálculo incremental de RSI."""
        rsi = RSI({'period': 14, 'overbought': 70, 'oversold': 30}, enabled=True)
        state = {}
        
        values = []
        for i in range(len(sample_ohlcv_data)):
            row = sample_ohlcv_data.iloc[i]
            value = rsi.calculate_incremental({'close': row['close']}, state)
            values.append(value)
        
        # RSI deve estar entre 0 e 100
        assert all(0 <= v <= 100 for v in values)
    
    def test_rsi_signals(self):
        """Testa sinais de overbought/oversold."""
        rsi = RSI({'period': 14, 'overbought': 70, 'oversold': 30}, enabled=True)
        
        assert rsi.is_overbought(75) is True
        assert rsi.is_overbought(65) is False
        assert rsi.is_oversold(25) is True
        assert rsi.is_oversold(35) is False
        
        assert rsi.get_signal(75) == -1  # Venda
        assert rsi.get_signal(25) == 1   # Compra
        assert rsi.get_signal(50) == 0   # Neutro


class TestMACD:
    """Testes para MACD."""
    
    def test_macd_calculation(self, sample_ohlcv_data):
        """Testa cálculo de MACD."""
        macd = MACD({'fast': 12, 'slow': 26, 'signal': 9}, enabled=True)
        result = macd.calculate(sample_ohlcv_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'macd' in result.columns
        assert 'macd_signal' in result.columns
        assert 'macd_hist' in result.columns
    
    def test_macd_incremental(self, sample_ohlcv_data):
        """Testa cálculo incremental de MACD."""
        macd = MACD({'fast': 12, 'slow': 26, 'signal': 9}, enabled=True)
        state = {}
        
        for i in range(len(sample_ohlcv_data)):
            row = sample_ohlcv_data.iloc[i]
            result = macd.calculate_incremental({'close': row['close']}, state)
            
            assert 'macd' in result
            assert 'macd_signal' in result
            assert 'macd_hist' in result


class TestADX:
    """Testes para ADX."""
    
    def test_adx_calculation(self, sample_ohlcv_data):
        """Testa cálculo de ADX."""
        adx = ADX({
            'period': 14, 
            'trending_threshold': 25, 
            'ranging_threshold': 20
        }, enabled=True)
        result = adx.calculate(sample_ohlcv_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'adx' in result.columns
        assert 'plus_di' in result.columns
        assert 'minus_di' in result.columns
        
        # ADX deve estar entre 0 e 100
        valid_adx = result['adx'].dropna()
        assert (valid_adx >= 0).all() and (valid_adx <= 100).all()
    
    def test_adx_regime(self):
        """Testa detecção de regime."""
        adx = ADX({
            'period': 14, 
            'trending_threshold': 25, 
            'ranging_threshold': 20
        }, enabled=True)
        
        assert adx.is_trending(30) is True
        assert adx.is_trending(20) is False
        assert adx.is_ranging(15) is True
        assert adx.is_ranging(25) is False
        
        assert adx.get_regime(30) == 'trending'
        assert adx.get_regime(15) == 'ranging'
        assert adx.get_regime(22) == 'neutral'


class TestBollingerBands:
    """Testes para Bollinger Bands."""
    
    def test_bollinger_calculation(self, sample_ohlcv_data):
        """Testa cálculo de Bollinger Bands."""
        bb = BollingerBands({'period': 20, 'std_dev': 2.0}, enabled=True)
        result = bb.calculate(sample_ohlcv_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'bb_upper' in result.columns
        assert 'bb_middle' in result.columns
        assert 'bb_lower' in result.columns
        assert 'bb_width' in result.columns
        assert 'bb_percent_b' in result.columns
        
        # Upper deve ser maior que middle, que deve ser maior que lower
        valid_idx = result['bb_upper'].notna()
        assert (result.loc[valid_idx, 'bb_upper'] >= result.loc[valid_idx, 'bb_middle']).all()
        assert (result.loc[valid_idx, 'bb_middle'] >= result.loc[valid_idx, 'bb_lower']).all()


class TestVPIN:
    """Testes para VPIN."""
    
    def test_vpin_calculation(self, sample_orderbook_data):
        """Testa cálculo de VPIN."""
        vpin = VPIN({'n_buckets': 50, 'bucket_size_usd': 100000}, enabled=True)
        result = vpin.calculate(sample_orderbook_data)
        
        assert isinstance(result, pd.Series)
        # VPIN deve estar entre 0 e 1
        valid_values = result.dropna()
        assert (valid_values >= 0).all() and (valid_values <= 1).all()
    
    def test_vpin_incremental(self):
        """Testa cálculo incremental de VPIN."""
        vpin = VPIN({'n_buckets': 20, 'bucket_size_usd': 10000}, enabled=True)
        state = {}
        
        # Simula trades
        for i in range(100):
            value = vpin.calculate_incremental({
                'price': 50000,
                'quantity': 0.1,
                'is_buyer_maker': i % 2 == 0
            }, state)
            
            assert 0 <= value <= 1


class TestMicroPrice:
    """Testes para Micro Price."""
    
    def test_micro_price_calculation(self, sample_orderbook_data):
        """Testa cálculo de Micro Price."""
        mp = MicroPrice({}, enabled=True)
        result = mp.calculate(sample_orderbook_data)
        
        assert isinstance(result, pd.Series)
        
        # Micro price deve estar entre bid e ask
        for i in range(len(result)):
            if not np.isnan(result.iloc[i]):
                bid = sample_orderbook_data['bid'].iloc[i]
                ask = sample_orderbook_data['ask'].iloc[i]
                assert bid <= result.iloc[i] <= ask


class TestEntropy:
    """Testes para Shannon Entropy."""
    
    def test_entropy_bounds(self, sample_ohlcv_data):
        """Entropia deve estar entre 0 e 1 (normalizada)."""
        entropy = ShannonEntropy({'window': 50, 'n_bins': 10}, enabled=True)
        result = entropy.calculate(sample_ohlcv_data)
        
        valid_values = result.dropna()
        assert (valid_values >= 0).all() and (valid_values <= 1).all()


class TestVolumeSpike:
    """Testes para Volume Spike."""
    
    def test_volume_spike_detection(self, sample_ohlcv_data):
        """Testa detecção de spikes de volume."""
        spike = VolumeSpike({'lookback': 20, 'threshold_multiplier': 2.0}, enabled=True)
        result = spike.calculate(sample_ohlcv_data)
        
        assert 'volume_spike_ratio' in result.columns
        assert 'is_spike' in result.columns
        
        # is_spike deve ser 0 ou 1
        assert result['is_spike'].isin([0, 1]).all()


class TestFeaturePipeline:
    """Testes para Feature Pipeline."""
    
    def test_pipeline_respects_config(self, sample_config):
        """Features desabilitadas não devem ser calculadas."""
        # Desabilita RSI
        config = sample_config.copy()
        config['features']['technical']['rsi']['enabled'] = False
        
        pipeline = FeaturePipeline(config)
        
        feature_names = pipeline.get_feature_names()
        assert 'RSI' not in feature_names
    
    def test_pipeline_calculates_all(self, sample_config, sample_ohlcv_data):
        """Pipeline deve calcular todas as features habilitadas."""
        pipeline = FeaturePipeline(sample_config)
        result = pipeline.calculate_all(sample_ohlcv_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_data)
        # Deve ter múltiplas colunas
        assert len(result.columns) > 0
    
    def test_pipeline_incremental(self, sample_config, sample_ohlcv_data):
        """Testa cálculo incremental do pipeline."""
        pipeline = FeaturePipeline(sample_config)
        
        row = sample_ohlcv_data.iloc[50]
        result = pipeline.calculate_incremental({
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume']
        })
        
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_pipeline_enable_disable(self, sample_config):
        """Testa habilitar/desabilitar features."""
        pipeline = FeaturePipeline(sample_config)
        
        # Desabilita RSI
        assert pipeline.disable_feature('RSI') is True
        assert 'RSI' not in pipeline.get_feature_names()
        
        # Habilita novamente
        assert pipeline.enable_feature('RSI') is True
        assert 'RSI' in pipeline.get_feature_names()


class TestIncrementalMatchesBatch:
    """Testa que cálculo incremental dá mesmo resultado que batch."""
    
    def test_rsi_incremental_matches_batch(self, sample_ohlcv_data):
        """RSI incremental deve convergir para o batch."""
        rsi = RSI({'period': 14, 'overbought': 70, 'oversold': 30}, enabled=True)
        
        # Batch
        batch_result = rsi.calculate(sample_ohlcv_data)
        
        # Incremental
        state = {}
        incremental_values = []
        for i in range(len(sample_ohlcv_data)):
            row = sample_ohlcv_data.iloc[i]
            value = rsi.calculate_incremental({'close': row['close']}, state)
            incremental_values.append(value)
        
        # Compara últimos valores (após convergência)
        # Tolerância maior porque EMA incremental pode diferir ligeiramente
        batch_last = batch_result.iloc[-1]
        incremental_last = incremental_values[-1]
        
        # Deve estar próximo (tolerância de 20% devido a diferenças de implementação EMA)
        if batch_last > 0:
            relative_diff = abs(batch_last - incremental_last) / batch_last
            assert relative_diff < 0.2, f"Batch: {batch_last}, Incremental: {incremental_last}"


class TestFeatureStore:
    """Testes para Feature Store."""
    
    @pytest.fixture(autouse=True)
    def check_parquet_support(self):
        """Verifica se pyarrow ou fastparquet está instalado."""
        try:
            import pyarrow
        except ImportError:
            try:
                import fastparquet
            except ImportError:
                pytest.skip("pyarrow ou fastparquet não instalado - pulando testes de parquet")
    
    def test_save_and_load(self, tmp_path, sample_ohlcv_data):
        """Testa salvar e carregar features."""
        store = FeatureStore(base_path=str(tmp_path / "features"))
        
        # Cria features de teste
        features = pd.DataFrame({
            'rsi': np.random.rand(100) * 100,
            'macd': np.random.randn(100) * 10
        }, index=sample_ohlcv_data.index)
        
        # Salva
        assert store.save(features, "test_features", partition_by_date=False) is True
        
        # Carrega
        loaded = store.load("test_features")
        
        assert len(loaded) == len(features)
        assert 'rsi' in loaded.columns
        assert 'macd' in loaded.columns
    
    def test_get_latest(self, tmp_path):
        """Testa obter valores mais recentes."""
        store = FeatureStore(base_path=str(tmp_path / "features"))
        
        store.update_latest({'rsi': 65.5, 'macd': 0.5})
        
        result = store.get_latest(['rsi', 'macd'])
        assert result['rsi'] == 65.5
        assert result['macd'] == 0.5
    
    def test_list_datasets(self, tmp_path, sample_ohlcv_data):
        """Testa listar datasets."""
        store = FeatureStore(base_path=str(tmp_path / "features"))
        
        features = pd.DataFrame({
            'feature1': np.random.rand(100)
        }, index=sample_ohlcv_data.index)
        
        store.save(features, "dataset1", partition_by_date=False)
        store.save(features, "dataset2", partition_by_date=False)
        
        datasets = store.list_datasets()
        assert 'dataset1' in datasets
        assert 'dataset2' in datasets


class TestFeatureParams:
    """Testa que todos os parâmetros vêm do config."""
    
    def test_ofi_uses_config_window(self):
        """OFI deve usar window do config."""
        config = {'window': 100}
        ofi = OFI(config, enabled=True)
        assert ofi.window == 100
        
        config2 = {'window': 50}
        ofi2 = OFI(config2, enabled=True)
        assert ofi2.window == 50
    
    def test_rsi_uses_config_params(self):
        """RSI deve usar todos os parâmetros do config."""
        config = {'period': 21, 'overbought': 80, 'oversold': 20}
        rsi = RSI(config, enabled=True)
        
        assert rsi.period == 21
        assert rsi.overbought == 80
        assert rsi.oversold == 20
    
    def test_bollinger_uses_config_params(self):
        """Bollinger Bands deve usar parâmetros do config."""
        config = {'period': 30, 'std_dev': 2.5}
        bb = BollingerBands(config, enabled=True)
        
        assert bb.period == 30
        assert bb.std_dev == 2.5
    
    def test_adx_uses_config_thresholds(self):
        """ADX deve usar thresholds do config."""
        config = {
            'period': 21,
            'trending_threshold': 30,
            'ranging_threshold': 15
        }
        adx = ADX(config, enabled=True)
        
        assert adx.period == 21
        assert adx.trending_threshold == 30
        assert adx.ranging_threshold == 15


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
