"""
Tests for the Multi-Timeframe RL Ensemble System.

Tests for:
- RLEnsemble class
- EnsembleEvaluator class
- Voting logic and confluence decisions
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from pathlib import Path


@pytest.fixture
def sample_observation():
    """Create a sample observation array."""
    return np.array([0.1, 0.2, 50.0, 25.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)


@pytest.fixture
def sample_observations():
    """Create sample observations for all timeframes."""
    base = np.array([0.1, 0.2, 50.0, 25.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return {
        "1m": base.copy(),
        "5m": base.copy() * 1.1,
        "15m": base.copy() * 0.9,
    }


@pytest.fixture
def sample_data():
    """Create sample OHLCV data for testing."""
    n = 500
    np.random.seed(42)
    
    returns = np.random.randn(n) * 0.01
    prices = 50000 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        "open": prices * (1 + np.random.randn(n) * 0.001),
        "high": prices * (1 + np.abs(np.random.randn(n) * 0.005)),
        "low": prices * (1 - np.abs(np.random.randn(n) * 0.005)),
        "close": prices,
        "volume": np.random.randint(100, 1000, n),
        "ofi": np.random.randn(n) * 0.3,
        "tfi": np.random.randn(n) * 0.3,
        "rsi": 50 + np.random.randn(n) * 15,
        "adx": 25 + np.random.randn(n) * 10,
        "regime": np.random.choice([0, 1, 2], n),
    })
    
    return data


class TestRLEnsemble:
    """Tests for RLEnsemble class."""

    def test_ensemble_init(self):
        """Ensemble should initialize with correct defaults."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble()
        
        assert ensemble.models_dir == Path("v2/models/rl")
        assert ensemble.min_votes_to_trade == 2
        assert ensemble.models == {}
        assert not ensemble._loaded

    def test_ensemble_init_custom_dir(self):
        """Ensemble should accept custom models directory."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble(models_dir="/custom/path")
        
        assert ensemble.models_dir == Path("/custom/path")

    def test_ensemble_init_custom_min_votes(self):
        """Ensemble should accept custom minimum votes."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble(min_votes_to_trade=3)
        
        assert ensemble.min_votes_to_trade == 3

    def test_ensemble_timeframes(self):
        """Ensemble should have correct timeframes defined."""
        from v2.src.rl.ensemble import RLEnsemble
        
        assert RLEnsemble.TIMEFRAMES == ["1m", "5m", "15m"]

    def test_confidence_levels(self):
        """Confidence levels should be correctly defined."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble()
        
        assert ensemble.get_confidence_level(3) == "MAXIMA"
        assert ensemble.get_confidence_level(2) == "ALTA"
        assert ensemble.get_confidence_level(1) == "BAIXA"
        assert ensemble.get_confidence_level(0) == "NENHUMA"

    def test_position_size_multipliers(self):
        """Position size multipliers should match confidence levels."""
        from v2.src.rl.ensemble import RLEnsemble
        
        assert RLEnsemble.POSITION_SIZE_MULTIPLIERS["MAXIMA"] == 1.0
        assert RLEnsemble.POSITION_SIZE_MULTIPLIERS["ALTA"] == 0.7
        assert RLEnsemble.POSITION_SIZE_MULTIPLIERS["BAIXA"] == 0.0
        assert RLEnsemble.POSITION_SIZE_MULTIPLIERS["NENHUMA"] == 0.0

    def test_get_status_not_loaded(self):
        """Status should show no models loaded initially."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble()
        status = ensemble.get_status()
        
        assert status["loaded"] is False
        assert status["total_models"] == 0
        assert status["is_ready"] is False

    def test_is_ready_not_enough_models(self):
        """is_ready should return False without enough models."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble(min_votes_to_trade=2)
        ensemble.models = {"1m": MagicMock()}  # Only 1 model
        
        assert not ensemble.is_ready()

    def test_is_ready_with_enough_models(self):
        """is_ready should return True with enough models."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble(min_votes_to_trade=2)
        ensemble.models = {
            "1m": MagicMock(),
            "5m": MagicMock(),
        }
        
        assert ensemble.is_ready()

    def test_get_loaded_timeframes(self):
        """Should return list of loaded timeframes."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble()
        ensemble.models = {
            "1m": MagicMock(),
            "15m": MagicMock(),
        }
        
        loaded = ensemble.get_loaded_timeframes()
        
        assert "1m" in loaded
        assert "15m" in loaded
        assert "5m" not in loaded

    def test_create_hold_result(self):
        """_create_hold_result should return correct structure."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble()
        result = ensemble._create_hold_result()
        
        assert result["decision"] == "HOLD"
        assert result["confidence"] == 0
        assert result["confidence_level"] == "NENHUMA"
        assert result["should_trade"] is False
        assert result["position_size_multiplier"] == 0.0

    def test_action_to_vote_sell(self):
        """Action 0 should map to SELL."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble()
        
        assert ensemble._action_to_vote(0) == "SELL"
        assert ensemble._action_to_vote(3) == "SELL"  # 3 % 3 = 0
        assert ensemble._action_to_vote(6) == "SELL"  # 6 % 3 = 0

    def test_action_to_vote_hold(self):
        """Action 1 should map to HOLD."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble()
        
        assert ensemble._action_to_vote(1) == "HOLD"
        assert ensemble._action_to_vote(4) == "HOLD"  # 4 % 3 = 1

    def test_action_to_vote_buy(self):
        """Action 2 should map to BUY."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble()
        
        assert ensemble._action_to_vote(2) == "BUY"
        assert ensemble._action_to_vote(5) == "BUY"  # 5 % 3 = 2

    def test_determine_decision_buy_majority(self):
        """Should return BUY with buy majority."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble(min_votes_to_trade=2)
        
        decision, confidence = ensemble._determine_decision(2, 1)
        
        assert decision == "BUY"
        assert confidence == 2

    def test_determine_decision_sell_majority(self):
        """Should return SELL with sell majority."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble(min_votes_to_trade=2)
        
        decision, confidence = ensemble._determine_decision(1, 2)
        
        assert decision == "SELL"
        assert confidence == 2

    def test_determine_decision_no_majority(self):
        """Should return HOLD without clear majority."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble(min_votes_to_trade=2)
        
        decision, confidence = ensemble._determine_decision(1, 1)
        
        assert decision == "HOLD"
        assert confidence == 0

    def test_determine_decision_unanimous_buy(self):
        """Should return BUY with confidence 3 for unanimous."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble(min_votes_to_trade=2)
        
        decision, confidence = ensemble._determine_decision(3, 0)
        
        assert decision == "BUY"
        assert confidence == 3

    def test_predict_no_models(self, sample_observations):
        """Predict should return HOLD when no models loaded."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble()
        
        result = ensemble.predict(sample_observations)
        
        assert result["decision"] == "HOLD"
        assert result["should_trade"] is False

    def test_predict_with_mock_models(self, sample_observations):
        """Predict should aggregate votes from models."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble()
        ensemble._loaded = True
        
        # Create mock models that return specific actions
        mock_1m = MagicMock()
        mock_1m.predict.return_value = (np.array([2]), None)  # BUY
        
        mock_5m = MagicMock()
        mock_5m.predict.return_value = (np.array([2]), None)  # BUY
        
        mock_15m = MagicMock()
        mock_15m.predict.return_value = (np.array([1]), None)  # HOLD
        
        ensemble.models = {
            "1m": mock_1m,
            "5m": mock_5m,
            "15m": mock_15m,
        }
        
        result = ensemble.predict(sample_observations)
        
        assert result["decision"] == "BUY"
        assert result["confidence"] == 2
        assert result["buy_votes"] == 2
        assert result["hold_votes"] == 1
        assert result["should_trade"] is True

    def test_predict_with_missing_observation(self):
        """Predict should handle missing observations gracefully."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble()
        ensemble._loaded = True
        
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([2]), None)
        
        ensemble.models = {"1m": mock_model}
        
        # Only provide observation for 1m
        observations = {
            "1m": np.array([0.1, 0.2, 50.0, 25.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        }
        
        result = ensemble.predict(observations)
        
        assert "votes" in result
        assert result["votes"]["1m"] == "BUY"


class TestEnsembleEvaluator:
    """Tests for EnsembleEvaluator class."""

    def test_evaluator_init(self):
        """Evaluator should initialize with correct defaults."""
        from v2.src.rl.ensemble import RLEnsemble
        from v2.src.rl.ensemble_evaluator import EnsembleEvaluator
        
        ensemble = RLEnsemble()
        evaluator = EnsembleEvaluator(ensemble)
        
        assert evaluator.initial_balance == 10000
        assert evaluator.position_size_pct == 0.1

    def test_evaluator_init_custom_config(self):
        """Evaluator should accept custom config."""
        from v2.src.rl.ensemble import RLEnsemble
        from v2.src.rl.ensemble_evaluator import EnsembleEvaluator
        
        ensemble = RLEnsemble()
        config = {"initial_balance": 50000, "position_size_pct": 0.2}
        evaluator = EnsembleEvaluator(ensemble, config)
        
        assert evaluator.initial_balance == 50000
        assert evaluator.position_size_pct == 0.2

    def test_validate_data_empty(self, sample_data):
        """Should reject empty data."""
        from v2.src.rl.ensemble import RLEnsemble
        from v2.src.rl.ensemble_evaluator import EnsembleEvaluator
        
        ensemble = RLEnsemble()
        evaluator = EnsembleEvaluator(ensemble)
        
        result = evaluator._validate_data({"1m": pd.DataFrame()})
        
        assert result is False

    def test_validate_data_valid(self, sample_data):
        """Should accept valid data."""
        from v2.src.rl.ensemble import RLEnsemble
        from v2.src.rl.ensemble_evaluator import EnsembleEvaluator
        
        ensemble = RLEnsemble()
        evaluator = EnsembleEvaluator(ensemble)
        
        result = evaluator._validate_data({"5m": sample_data})
        
        assert result is True

    def test_calculate_pnl_long_profit(self):
        """Should calculate profit for long position."""
        from v2.src.rl.ensemble import RLEnsemble
        from v2.src.rl.ensemble_evaluator import EnsembleEvaluator
        
        ensemble = RLEnsemble()
        config = {"fees": {"taker": 0.0}}
        evaluator = EnsembleEvaluator(ensemble, config)
        
        # Long position: buy at 100, sell at 110
        pnl = evaluator._calculate_pnl(
            position=1,
            entry_price=100,
            exit_price=110,
            balance=10000,
        )
        
        # 10% gain on 10% position = $100
        assert abs(pnl - 100) < 1  # Allow small tolerance

    def test_calculate_pnl_long_loss(self):
        """Should calculate loss for long position."""
        from v2.src.rl.ensemble import RLEnsemble
        from v2.src.rl.ensemble_evaluator import EnsembleEvaluator
        
        ensemble = RLEnsemble()
        config = {"fees": {"taker": 0.0}}
        evaluator = EnsembleEvaluator(ensemble, config)
        
        # Long position: buy at 100, sell at 90
        pnl = evaluator._calculate_pnl(
            position=1,
            entry_price=100,
            exit_price=90,
            balance=10000,
        )
        
        # 10% loss on 10% position = -$100
        assert abs(pnl - (-100)) < 1

    def test_calculate_pnl_short_profit(self):
        """Should calculate profit for short position."""
        from v2.src.rl.ensemble import RLEnsemble
        from v2.src.rl.ensemble_evaluator import EnsembleEvaluator
        
        ensemble = RLEnsemble()
        config = {"fees": {"taker": 0.0}}
        evaluator = EnsembleEvaluator(ensemble, config)
        
        # Short position: sell at 100, buy at 90
        pnl = evaluator._calculate_pnl(
            position=-1,
            entry_price=100,
            exit_price=90,
            balance=10000,
        )
        
        # 10% gain on 10% position = $100
        assert abs(pnl - 100) < 1

    def test_calculate_metrics_empty_trades(self):
        """Should handle empty trades list."""
        from v2.src.rl.ensemble import RLEnsemble
        from v2.src.rl.ensemble_evaluator import EnsembleEvaluator
        
        ensemble = RLEnsemble()
        evaluator = EnsembleEvaluator(ensemble)
        
        metrics = evaluator._calculate_metrics([], [10000], [], 10000)
        
        assert metrics["n_trades"] == 0
        assert metrics["total_pnl"] == 0.0
        assert metrics["win_rate"] == 0.0

    def test_calculate_metrics_with_trades(self):
        """Should calculate correct metrics from trades."""
        from v2.src.rl.ensemble import RLEnsemble
        from v2.src.rl.ensemble_evaluator import EnsembleEvaluator
        
        ensemble = RLEnsemble()
        evaluator = EnsembleEvaluator(ensemble)
        
        trades = [
            {"pnl": 100},
            {"pnl": 50},
            {"pnl": -30},
        ]
        
        metrics = evaluator._calculate_metrics(trades, [10000, 10100, 10150, 10120], [], 10000)
        
        assert metrics["n_trades"] == 3
        assert metrics["total_pnl"] == 120
        assert metrics["winning_trades"] == 2
        assert metrics["losing_trades"] == 1
        assert abs(metrics["win_rate"] - 66.67) < 1

    def test_analyze_voting_patterns_empty(self):
        """Should handle empty voting history."""
        from v2.src.rl.ensemble import RLEnsemble
        from v2.src.rl.ensemble_evaluator import EnsembleEvaluator
        
        ensemble = RLEnsemble()
        evaluator = EnsembleEvaluator(ensemble)
        
        result = evaluator.analyze_voting_patterns([])
        
        assert "error" in result

    def test_analyze_voting_patterns(self):
        """Should analyze voting patterns correctly."""
        from v2.src.rl.ensemble import RLEnsemble
        from v2.src.rl.ensemble_evaluator import EnsembleEvaluator
        
        ensemble = RLEnsemble()
        evaluator = EnsembleEvaluator(ensemble)
        
        voting_history = [
            {"step": 0, "votes": {"1m": "BUY", "5m": "BUY", "15m": "BUY"}, "decision": "BUY", "confidence": 3, "should_trade": True},
            {"step": 1, "votes": {"1m": "SELL", "5m": "SELL", "15m": "HOLD"}, "decision": "SELL", "confidence": 2, "should_trade": True},
            {"step": 2, "votes": {"1m": "HOLD", "5m": "HOLD", "15m": "HOLD"}, "decision": "HOLD", "confidence": 0, "should_trade": False},
        ]
        
        analysis = evaluator.analyze_voting_patterns(voting_history)
        
        assert analysis["total_decisions"] == 3
        assert analysis["unanimous"]["buy"] == 1
        assert analysis["unanimous"]["hold"] == 1
        assert analysis["majority"]["sell"] == 1


class TestIntegration:
    """Integration tests for the ensemble system."""

    def test_ensemble_evaluator_integration(self, sample_data):
        """Test ensemble with evaluator integration."""
        from v2.src.rl.ensemble import RLEnsemble
        from v2.src.rl.ensemble_evaluator import EnsembleEvaluator
        
        ensemble = RLEnsemble()
        evaluator = EnsembleEvaluator(ensemble)
        
        # Without models loaded, backtest should handle gracefully
        data = {"5m": sample_data}
        result = evaluator.backtest_ensemble(data)
        
        # Should have the expected structure
        assert "metrics" in result or "error" in result

    def test_predict_with_metadata(self, sample_observations):
        """Test predict_with_metadata adds correct metadata."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble()
        
        result = ensemble.predict_with_metadata(
            sample_observations,
            market_data={"price": 50000, "volume": 1000}
        )
        
        assert "metadata" in result
        assert result["metadata"]["loaded_models"] == 0


class TestLoadModels:
    """Tests for model loading functionality."""

    def test_load_models_no_sb3(self):
        """Should handle missing stable-baselines3 gracefully."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble(models_dir="/nonexistent/path")
        
        with patch.dict('sys.modules', {'stable_baselines3': None}):
            # This should not crash
            result = ensemble.load_models()
            # May return False if import fails
            assert isinstance(result, bool)

    def test_load_models_no_files(self, tmp_path):
        """Should handle missing model files gracefully."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble(models_dir=str(tmp_path))
        
        # Should return False when no models found
        result = ensemble.load_models()
        
        assert result is False


class TestVotingLogic:
    """Tests for voting logic edge cases."""

    def test_equal_buy_sell_votes(self):
        """Should return HOLD when buy and sell votes are equal."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble(min_votes_to_trade=1)
        
        # 1 BUY, 1 SELL = no clear winner
        decision, confidence = ensemble._determine_decision(1, 1)
        
        assert decision == "HOLD"

    def test_minimum_votes_threshold(self):
        """Should respect minimum votes threshold."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble(min_votes_to_trade=3)
        
        # Only 2 BUY votes, but threshold is 3
        decision, confidence = ensemble._determine_decision(2, 0)
        
        assert decision == "HOLD"

    def test_strict_mode_requires_unanimity(self):
        """Test strict mode (3/3) voting."""
        from v2.src.rl.ensemble import RLEnsemble
        
        ensemble = RLEnsemble(min_votes_to_trade=3)
        
        # Unanimous BUY
        decision1, conf1 = ensemble._determine_decision(3, 0)
        assert decision1 == "BUY"
        assert conf1 == 3
        
        # 2/3 BUY should not trade
        decision2, conf2 = ensemble._determine_decision(2, 1)
        assert decision2 == "HOLD"
