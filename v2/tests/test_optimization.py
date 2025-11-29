"""
Tests for optimization module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Test fixtures
@pytest.fixture
def config():
    """Sample configuration for testing."""
    return {
        "optimization": {
            "n_trials": 10,
            "n_jobs": 1,
            "study_name": "test_study",
            "storage_type": "sqlite",
            "objectives": {
                "primary": "sharpe",
                "secondary": "max_drawdown",
            },
            "pruning": {
                "enabled": True,
                "min_trials": 5,
                "patience": 10,
            },
            "sampler": {
                "type": "TPE",
                "multivariate": True,
                "n_startup_trials": 5,
            },
        },
        "features": {
            "technical": {
                "rsi": {
                    "enabled": True,
                    "period": 14,
                }
            }
        },
        "risk": {
            "max_risk_per_trade_pct": 0.5,
            "kill_switch": {"enabled": True},
        },
    }


class TestParamSpace:
    """Tests for ParamSpace class."""

    def test_param_space_loads_params(self, config):
        """ParamSpace should load optimizable parameters from schema."""
        from v2.src.optimization.param_space import ParamSpace

        param_space = ParamSpace(config)
        assert param_space.params is not None
        assert len(param_space.params) > 0

    def test_param_space_get_param_count(self, config):
        """ParamSpace should count all optimizable parameters."""
        from v2.src.optimization.param_space import ParamSpace

        param_space = ParamSpace(config)
        count = param_space.get_param_count()
        # Should have many parameters from schema
        assert count > 50

    def test_param_space_get_defaults(self, config):
        """ParamSpace should return default values."""
        from v2.src.optimization.param_space import ParamSpace

        param_space = ParamSpace(config)
        defaults = param_space.get_default_params()
        assert isinstance(defaults, dict)
        assert len(defaults) > 0

    def test_param_space_apply_to_config(self, config):
        """ParamSpace should apply params to config."""
        from v2.src.optimization.param_space import ParamSpace

        param_space = ParamSpace(config)
        params = {"features.technical.rsi.period": 21}
        updated = param_space.apply_params_to_config(config, params)
        assert updated["features"]["technical"]["rsi"]["period"] == 21

    def test_param_space_validation(self, config):
        """ParamSpace should validate parameters."""
        from v2.src.optimization.param_space import ParamSpace

        param_space = ParamSpace(config)
        # Valid params should have no errors
        valid_params = {"features.technical.rsi.period": 14}
        errors = param_space.validate_params(valid_params)
        assert len(errors) == 0


class TestObjectiveFunctions:
    """Tests for ObjectiveFunctions class."""

    def test_sharpe_ratio(self):
        """Sharpe ratio should be calculated correctly."""
        from v2.src.optimization.objectives import ObjectiveFunctions

        # Create sample returns
        returns = pd.Series([0.01, -0.005, 0.02, 0.015, -0.01, 0.008])
        sharpe = ObjectiveFunctions.sharpe_ratio(returns)
        assert isinstance(sharpe, float)

    def test_sharpe_ratio_zero_std(self):
        """Sharpe ratio should handle zero std."""
        from v2.src.optimization.objectives import ObjectiveFunctions

        returns = pd.Series([0.0, 0.0, 0.0])
        sharpe = ObjectiveFunctions.sharpe_ratio(returns)
        assert sharpe == 0.0

    def test_sortino_ratio(self):
        """Sortino ratio should be calculated correctly."""
        from v2.src.optimization.objectives import ObjectiveFunctions

        returns = pd.Series([0.01, -0.005, 0.02, 0.015, -0.01, 0.008])
        sortino = ObjectiveFunctions.sortino_ratio(returns)
        assert isinstance(sortino, float)

    def test_max_drawdown(self):
        """Max drawdown should be calculated correctly."""
        from v2.src.optimization.objectives import ObjectiveFunctions

        equity = pd.Series([100, 110, 105, 115, 100, 120])
        max_dd = ObjectiveFunctions.max_drawdown(equity)
        assert max_dd > 0  # Should have some drawdown
        assert max_dd <= 100  # Should be percentage

    def test_profit_factor(self):
        """Profit factor should be calculated correctly."""
        from v2.src.optimization.objectives import ObjectiveFunctions

        trades = pd.DataFrame({"pnl": [100, -50, 75, -25, 50]})
        pf = ObjectiveFunctions.profit_factor(trades)
        assert pf > 1  # More wins than losses

    def test_profit_factor_no_losses(self):
        """Profit factor should handle no losses."""
        from v2.src.optimization.objectives import ObjectiveFunctions

        trades = pd.DataFrame({"pnl": [100, 50, 75]})
        pf = ObjectiveFunctions.profit_factor(trades)
        assert pf == float("inf")

    def test_win_rate(self):
        """Win rate should be calculated correctly."""
        from v2.src.optimization.objectives import ObjectiveFunctions

        trades = pd.DataFrame({"pnl": [100, -50, 75, -25, 50]})
        wr = ObjectiveFunctions.win_rate(trades)
        assert wr == 60.0  # 3 out of 5 winners

    def test_calmar_ratio(self):
        """Calmar ratio should be calculated correctly."""
        from v2.src.optimization.objectives import ObjectiveFunctions

        returns = pd.Series([0.01, -0.005, 0.02, 0.015, -0.01, 0.008])
        calmar = ObjectiveFunctions.calmar_ratio(returns, max_drawdown=5.0)
        assert isinstance(calmar, float)


class TestOptunaOptimizer:
    """Tests for OptunaOptimizer class."""

    def test_optuna_creates_study(self, config, tmp_path):
        """Optuna should create study correctly."""
        from v2.src.optimization.optuna_optimizer import OptunaOptimizer

        config["optimization"]["study_name"] = "test_create"
        optimizer = OptunaOptimizer(config)

        # Use temporary storage
        study = optimizer.create_study(
            storage=f"sqlite:///{tmp_path}/test.db",
            load_if_exists=False,
        )

        assert study is not None
        assert optimizer.study is not None

    def test_optuna_suggest_params(self, config, tmp_path):
        """Optuna should suggest parameters for trial."""
        from v2.src.optimization.optuna_optimizer import OptunaOptimizer
        import optuna

        optimizer = OptunaOptimizer(config)
        study = optimizer.create_study(
            storage=f"sqlite:///{tmp_path}/test.db",
            load_if_exists=False,
        )

        # Create a trial manually
        trial = study.ask()
        params = optimizer.suggest_params(trial)

        assert isinstance(params, dict)
        assert len(params) > 0

    def test_optuna_optimize(self, config, tmp_path):
        """Optuna should run optimization."""
        from v2.src.optimization.optuna_optimizer import OptunaOptimizer

        optimizer = OptunaOptimizer(config)
        optimizer.create_study(
            storage=f"sqlite:///{tmp_path}/test.db",
            load_if_exists=False,
        )

        # Simple objective
        def objective(trial):
            x = trial.suggest_float("x", -10, 10)
            return -(x**2)  # Maximize negative quadratic

        best_params = optimizer.optimize(
            objective,
            n_trials=5,
            show_progress_bar=False,
        )

        assert isinstance(best_params, dict)
        assert optimizer.get_best_value() is not None


class TestStudyManager:
    """Tests for StudyManager class."""

    def test_study_manager_create_study(self, tmp_path):
        """StudyManager should create studies."""
        from v2.src.optimization.study_manager import StudyManager

        manager = StudyManager(
            storage_path=str(tmp_path),
            storage_type="sqlite",
        )

        study = manager.create_study(
            study_name="test_manager",
            direction="maximize",
        )

        assert study is not None

    def test_study_manager_list_studies(self, tmp_path):
        """StudyManager should list studies."""
        from v2.src.optimization.study_manager import StudyManager

        manager = StudyManager(
            storage_path=str(tmp_path),
            storage_type="sqlite",
        )

        manager.create_study("study1")
        manager.create_study("study2")

        studies = manager.list_studies()
        assert "study1" in studies
        assert "study2" in studies

    def test_study_manager_get_summary(self, tmp_path):
        """StudyManager should get study summary."""
        from v2.src.optimization.study_manager import StudyManager

        manager = StudyManager(
            storage_path=str(tmp_path),
            storage_type="sqlite",
        )

        study = manager.create_study("test_summary")

        # Add a trial
        study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=1)

        summary = manager.get_study_summary("test_summary")
        assert summary["study_name"] == "test_summary"
        assert summary["n_trials"] == 1


class TestSchema:
    """Tests for schema module."""

    def test_count_optimizable_params(self):
        """Schema should count optimizable parameters."""
        from v2.src.config.schema import count_optimizable_params

        count = count_optimizable_params()
        # Should have ~150 parameters as specified
        assert count > 100

    def test_get_all_optimizable_params(self):
        """Schema should return all optimizable parameters as a flat list."""
        from v2.src.config.schema import get_all_optimizable_params, get_params_dict

        # get_all_optimizable_params returns a flat list
        params = get_all_optimizable_params()
        assert isinstance(params, list)
        assert len(params) > 0
        
        # get_params_dict returns a dict organized by section
        params_dict = get_params_dict()
        assert isinstance(params_dict, dict)
        assert len(params_dict) > 0

    def test_get_feature_toggles(self):
        """Schema should return feature toggles."""
        from v2.src.config.schema import get_feature_toggles

        toggles = get_feature_toggles()
        assert isinstance(toggles, dict)
        assert len(toggles) > 0

    def test_get_strategy_toggles(self):
        """Schema should return strategy toggles."""
        from v2.src.config.schema import get_strategy_toggles

        toggles = get_strategy_toggles()
        assert isinstance(toggles, dict)
        assert len(toggles) > 0
