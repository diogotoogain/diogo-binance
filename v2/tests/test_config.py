"""
Testes para o módulo de configuração do V2 Trading Bot.

Testa:
- Carregamento de configuração
- Validação de kill switch
- Validação de ranges de risco
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# Adiciona diretório src ao path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.loader import (
    ConfigLoader,
    ConfigValidationError,
    KillSwitchDisabledError,
    get_config
)
from config.schema import (
    OptimizableParam,
    ParamType,
    get_all_optimizable_params,
    count_optimizable_params
)


class TestConfigLoader:
    """Testes para ConfigLoader."""
    
    @pytest.fixture(autouse=True)
    def reset_singleton(self, monkeypatch):
        """Reseta singleton e limpa env vars antes de cada teste."""
        ConfigLoader.reset()
        # Limpa variáveis de ambiente que podem interferir
        monkeypatch.delenv('RISK_PER_TRADE', raising=False)
        monkeypatch.delenv('MAX_LEVERAGE', raising=False)
        monkeypatch.delenv('MAX_DAILY_LOSS', raising=False)
        yield
        ConfigLoader.reset()
    
    @pytest.fixture
    def valid_config(self):
        """Cria config válida temporária."""
        config_data = {
            'environment': {
                'mode': 'demo',
                'use_demo_header': True
            },
            'market': {
                'symbol': 'BTCUSDT'
            },
            'websocket': {
                'buffer_size': 10000
            },
            'risk': {
                'risk_per_trade_pct': 1.0,
                'max_leverage': 10,
                'kill_switch': {
                    'enabled': True,
                    'max_loss_trigger_pct': 5.0
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            return f.name
    
    @pytest.fixture
    def invalid_kill_switch_config(self):
        """Cria config com kill switch desativado."""
        config_data = {
            'environment': {'mode': 'demo'},
            'websocket': {'buffer_size': 10000},
            'risk': {
                'risk_per_trade_pct': 1.0,
                'max_leverage': 10,
                'kill_switch': {
                    'enabled': False  # INVÁLIDO!
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            return f.name
    
    def test_load_default_config(self, valid_config):
        """Testa carregamento de config padrão."""
        config = ConfigLoader(valid_config, skip_env=True)
        
        assert config.get('environment.mode') == 'demo'
        assert config.get('market.symbol') == 'BTCUSDT'
        assert config.get('risk.max_leverage') == 10
    
    def test_kill_switch_always_enabled(self, invalid_kill_switch_config):
        """Testa que kill switch desativado gera erro."""
        with pytest.raises(KillSwitchDisabledError):
            ConfigLoader(invalid_kill_switch_config, skip_env=True)
    
    def test_kill_switch_enabled_passes(self, valid_config):
        """Testa que kill switch ativado passa validação."""
        config = ConfigLoader(valid_config, skip_env=True)
        assert config.get('risk.kill_switch.enabled') is True
    
    def test_risk_range_valid(self, valid_config):
        """Testa ranges de risco válidos."""
        config = ConfigLoader(valid_config, skip_env=True)
        
        risk = config.get('risk.risk_per_trade_pct')
        assert 0.1 <= risk <= 5.0
        
        leverage = config.get('risk.max_leverage')
        assert 1 <= leverage <= 20
    
    def test_risk_per_trade_too_high(self):
        """Testa que risk_per_trade > 5% gera erro."""
        config_data = {
            'environment': {'mode': 'demo'},
            'websocket': {'buffer_size': 10000},
            'risk': {
                'risk_per_trade_pct': 10.0,  # INVÁLIDO! > 5%
                'max_leverage': 10,
                'kill_switch': {'enabled': True}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            
            with pytest.raises(ConfigValidationError):
                ConfigLoader(f.name, skip_env=True)
    
    def test_leverage_too_high(self):
        """Testa que leverage > 20 gera erro."""
        config_data = {
            'environment': {'mode': 'demo'},
            'websocket': {'buffer_size': 10000},
            'risk': {
                'risk_per_trade_pct': 1.0,
                'max_leverage': 50,  # INVÁLIDO! > 20
                'kill_switch': {'enabled': True}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            
            with pytest.raises(ConfigValidationError):
                ConfigLoader(f.name, skip_env=True)
    
    def test_get_with_dot_notation(self, valid_config):
        """Testa acesso com notação de ponto."""
        config = ConfigLoader(valid_config, skip_env=True)
        
        # Acesso direto
        assert config.get('risk.max_leverage') == 10
        
        # Acesso aninhado
        assert config.get('risk.kill_switch.enabled') is True
        
        # Default para chave inexistente
        assert config.get('nonexistent.key', 'default') == 'default'
    
    def test_get_section(self, valid_config):
        """Testa obtenção de seção completa."""
        config = ConfigLoader(valid_config, skip_env=True)
        
        risk_section = config.get_section('risk')
        assert 'risk_per_trade_pct' in risk_section
        assert 'max_leverage' in risk_section
        assert 'kill_switch' in risk_section
    
    def test_is_demo_mode(self, valid_config):
        """Testa propriedade is_demo_mode."""
        config = ConfigLoader(valid_config, skip_env=True)
        assert config.is_demo_mode is True
    
    def test_singleton_pattern(self, valid_config):
        """Testa padrão singleton."""
        config1 = ConfigLoader(valid_config, skip_env=True)
        config2 = ConfigLoader()  # Deve retornar mesma instância
        
        assert config1 is config2
    
    def test_websocket_buffer_validation(self):
        """Testa que buffer < 1000 gera erro."""
        config_data = {
            'environment': {'mode': 'demo'},
            'websocket': {'buffer_size': 100},  # INVÁLIDO! < 1000
            'risk': {
                'risk_per_trade_pct': 1.0,
                'max_leverage': 10,
                'kill_switch': {'enabled': True}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            
            with pytest.raises(ConfigValidationError):
                ConfigLoader(f.name, skip_env=True)


class TestOptimizableParam:
    """Testes para OptimizableParam."""
    
    def test_float_param(self):
        """Testa criação de parâmetro float."""
        param = OptimizableParam(
            name='test.param',
            param_type=ParamType.FLOAT,
            default=1.0,
            low=0.0,
            high=2.0
        )
        
        assert param.name == 'test.param'
        assert param.param_type == ParamType.FLOAT
        assert param.default == 1.0
    
    def test_int_param(self):
        """Testa criação de parâmetro int."""
        param = OptimizableParam(
            name='test.int_param',
            param_type=ParamType.INT,
            default=10,
            low=1,
            high=20
        )
        
        assert param.param_type == ParamType.INT
        assert param.low == 1
        assert param.high == 20
    
    def test_bool_param(self):
        """Testa criação de parâmetro bool."""
        param = OptimizableParam(
            name='test.bool_param',
            param_type=ParamType.BOOL,
            default=True
        )
        
        assert param.param_type == ParamType.BOOL
        assert param.choices == [True, False]
    
    def test_categorical_param(self):
        """Testa criação de parâmetro categórico."""
        param = OptimizableParam(
            name='test.cat_param',
            param_type=ParamType.CATEGORICAL,
            default='option1',
            choices=['option1', 'option2', 'option3']
        )
        
        assert param.param_type == ParamType.CATEGORICAL
        assert len(param.choices) == 3
    
    def test_float_requires_range(self):
        """Testa que FLOAT requer low e high."""
        with pytest.raises(ValueError):
            OptimizableParam(
                name='test.param',
                param_type=ParamType.FLOAT,
                default=1.0
                # Falta low e high
            )
    
    def test_categorical_requires_choices(self):
        """Testa que CATEGORICAL requer choices."""
        with pytest.raises(ValueError):
            OptimizableParam(
                name='test.param',
                param_type=ParamType.CATEGORICAL,
                default='option1'
                # Falta choices
            )
    
    def test_to_dict(self):
        """Testa conversão para dicionário."""
        param = OptimizableParam(
            name='test.param',
            param_type=ParamType.FLOAT,
            default=1.0,
            low=0.0,
            high=2.0,
            description='Test parameter'
        )
        
        d = param.to_dict()
        
        assert d['name'] == 'test.param'
        assert d['type'] == 'float'
        assert d['default'] == 1.0


class TestAllOptimizableParams:
    """Testes para lista completa de parâmetros."""
    
    def test_get_all_params_returns_list(self):
        """Testa que get_all_optimizable_params retorna lista."""
        params = get_all_optimizable_params()
        
        assert isinstance(params, list)
        assert len(params) > 0
    
    def test_param_count_around_150(self):
        """Testa que há aproximadamente 150 parâmetros."""
        count = count_optimizable_params()
        
        # Deve ter entre 100 e 200 parâmetros
        assert 100 <= count <= 200, f"Esperado ~150 params, encontrado {count}"
    
    def test_all_params_have_required_fields(self):
        """Testa que todos os parâmetros têm campos obrigatórios."""
        params = get_all_optimizable_params()
        
        for param in params:
            assert param.name, f"Param sem nome: {param}"
            assert param.param_type is not None, f"Param sem tipo: {param.name}"
            assert param.default is not None, f"Param sem default: {param.name}"
    
    def test_risk_params_exist(self):
        """Testa que parâmetros de risco existem."""
        params = get_all_optimizable_params()
        param_names = [p.name for p in params]
        
        assert any('risk' in name for name in param_names)
        assert any('max_leverage' in name for name in param_names)
    
    def test_strategy_params_exist(self):
        """Testa que parâmetros de estratégia existem."""
        params = get_all_optimizable_params()
        param_names = [p.name for p in params]
        
        assert any('strategies' in name for name in param_names)


# Fixture para usar o config padrão real se existir
@pytest.fixture
def real_config_path():
    """Retorna path do config real se existir."""
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    if config_path.exists():
        return str(config_path)
    return None


class TestRealConfig:
    """Testes com config real (se existir)."""
    
    @pytest.fixture(autouse=True)
    def reset_singleton(self, monkeypatch):
        """Reseta singleton e limpa env vars antes de cada teste."""
        ConfigLoader.reset()
        # Limpa variáveis de ambiente que podem interferir
        monkeypatch.delenv('RISK_PER_TRADE', raising=False)
        monkeypatch.delenv('MAX_LEVERAGE', raising=False)
        monkeypatch.delenv('MAX_DAILY_LOSS', raising=False)
        yield
        ConfigLoader.reset()
    
    def test_load_real_config(self, real_config_path):
        """Testa carregamento do config real."""
        if real_config_path is None:
            pytest.skip("Config real não encontrado")
        
        config = ConfigLoader(real_config_path, skip_env=True)
        
        # Verifica campos básicos
        assert config.get('environment.mode') is not None
        assert config.get('market.symbol') is not None
    
    def test_real_config_kill_switch_enabled(self, real_config_path):
        """Testa que config real tem kill switch ativado."""
        if real_config_path is None:
            pytest.skip("Config real não encontrado")
        
        config = ConfigLoader(real_config_path, skip_env=True)
        
        assert config.get('risk.kill_switch.enabled') is True
    
    def test_real_config_websocket_buffer(self, real_config_path):
        """Testa que config real tem buffer grande."""
        if real_config_path is None:
            pytest.skip("Config real não encontrado")
        
        config = ConfigLoader(real_config_path, skip_env=True)
        
        buffer_size = config.get('websocket.buffer_size')
        assert buffer_size >= 10000, f"Buffer deve ser >= 10000, encontrado {buffer_size}"
