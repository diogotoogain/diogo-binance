"""
Tests for RL module.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch


@pytest.fixture
def config():
    """Sample configuration for testing."""
    return {
        "rl": {
            "enabled": True,
            "environment": {
                "observation_features": ["ofi", "tfi", "rsi", "position", "pnl"],
                "action_space": {
                    "direction": [-1, 0, 1],
                    "position_size": [0.0, 0.25, 0.5, 0.75, 1.0],
                    "sl_atr_mult": [0.5, 1.0, 1.5, 2.0, 3.0],
                    "tp_atr_mult": [1.0, 1.5, 2.0, 3.0, 5.0],
                },
                "reward_function": "sharpe",
                "episode_length": 100,
                "initial_balance": 10000,
            },
            "agents": {
                "ppo": {
                    "enabled": True,
                    "learning_rate": 0.0003,
                    "n_steps": 64,
                    "batch_size": 32,
                },
                "sac": {
                    "enabled": True,
                    "learning_rate": 0.0003,
                    "buffer_size": 10000,
                },
            },
            "training": {
                "total_timesteps": 1000,
                "eval_freq": 500,
                "n_eval_episodes": 2,
            },
        },
        "risk": {
            "kill_switch": {"enabled": True},
        },
    }


@pytest.fixture
def sample_data():
    """Sample OHLCV data for testing."""
    n = 500
    np.random.seed(42)
    
    # Generate price data
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
        "atr": prices * 0.02,
    })
    
    return data


class TestTradingEnvironment:
    """Tests for TradingEnvironment class."""

    def test_environment_init(self, config, sample_data):
        """Environment should initialize correctly."""
        from v2.src.rl.environment import TradingEnvironment

        env = TradingEnvironment(config, sample_data)

        assert env.observation_space is not None
        assert env.action_space is not None
        assert env.episode_length == 100
        assert env.initial_balance == 10000

    def test_environment_reset(self, config, sample_data):
        """Environment should reset correctly."""
        from v2.src.rl.environment import TradingEnvironment

        env = TradingEnvironment(config, sample_data)
        obs, info = env.reset()

        assert obs is not None
        assert len(obs) == len(config["rl"]["environment"]["observation_features"])
        assert env.position == 0
        assert env.balance == env.initial_balance

    def test_environment_step(self, config, sample_data):
        """Step should return obs, reward, done, info."""
        from v2.src.rl.environment import TradingEnvironment

        env = TradingEnvironment(config, sample_data)
        env.reset()

        # Take a step with hold action
        action = {
            "direction": 1,  # index 1 maps to 0 (hold) in [-1, 0, 1]
            "position_size": 0,
            "sl_mult": 0,
            "tp_mult": 0,
        }

        obs, reward, terminated, truncated, info = env.step(action)

        assert obs is not None
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_environment_long_trade(self, config, sample_data):
        """Environment should handle long trades."""
        from v2.src.rl.environment import TradingEnvironment

        env = TradingEnvironment(config, sample_data)
        env.reset()

        # Open long position
        action = {
            "direction": 2,  # index 2 maps to 1 (long) in [-1, 0, 1]
            "position_size": 2,  # 0.5
            "sl_mult": 1,
            "tp_mult": 1,
        }

        obs, reward, terminated, truncated, info = env.step(action)

        # Should have opened a position
        assert env.position == 1  # Long
        assert env.position_size > 0

    def test_environment_short_trade(self, config, sample_data):
        """Environment should handle short trades."""
        from v2.src.rl.environment import TradingEnvironment

        env = TradingEnvironment(config, sample_data)
        env.reset()

        # Open short position
        action = {
            "direction": 0,  # index 0 maps to -1 (short) in [-1, 0, 1]
            "position_size": 2,
            "sl_mult": 1,
            "tp_mult": 1,
        }

        obs, reward, terminated, truncated, info = env.step(action)

        assert env.position == -1  # Short

    def test_environment_episode_stats(self, config, sample_data):
        """Environment should track episode statistics."""
        from v2.src.rl.environment import TradingEnvironment

        env = TradingEnvironment(config, sample_data)
        env.reset()

        # Run a few steps with hold action
        for _ in range(10):
            action = {
                "direction": 1,  # index 1 = hold
                "position_size": 0,
                "sl_mult": 0,
                "tp_mult": 0,
            }
            env.step(action)

        stats = env.get_episode_stats()
        assert "n_trades" in stats
        assert "sharpe" in stats
        assert "max_drawdown" in stats


class TestFlatActionWrapper:
    """Tests for FlatActionWrapper."""

    def test_flat_action_wrapper(self, config, sample_data):
        """FlatActionWrapper should convert flat actions to dict."""
        from v2.src.rl.environment import TradingEnvironment, FlatActionWrapper

        env = TradingEnvironment(config, sample_data)
        wrapped = FlatActionWrapper(env)

        # Check action space is now Discrete
        from gymnasium import spaces
        assert isinstance(wrapped.action_space, spaces.Discrete)

        # Test action conversion
        wrapped.reset()
        obs, reward, term, trunc, info = wrapped.step(0)

        assert obs is not None


class TestRewardFunctions:
    """Tests for RewardFunctions class."""

    def test_pnl_reward(self):
        """P&L reward should be normalized."""
        from v2.src.rl.reward_functions import RewardFunctions

        reward = RewardFunctions.pnl_reward(100, 10000)
        assert reward == 0.01

    def test_sharpe_reward(self):
        """Sharpe reward should use rolling window."""
        from v2.src.rl.reward_functions import RewardFunctions

        returns = [0.01, -0.005, 0.02, 0.015, -0.01, 0.008] * 5
        reward = RewardFunctions.sharpe_reward(returns, window=20)
        assert isinstance(reward, float)

    def test_sharpe_reward_insufficient_data(self):
        """Sharpe reward should handle insufficient data."""
        from v2.src.rl.reward_functions import RewardFunctions

        returns = [0.01, 0.02]
        reward = RewardFunctions.sharpe_reward(returns, window=20)
        assert reward == 0.0

    def test_risk_adjusted_reward(self):
        """Risk-adjusted reward should penalize drawdown."""
        from v2.src.rl.reward_functions import RewardFunctions

        reward = RewardFunctions.risk_adjusted_reward(100, 5, drawdown_penalty=0.5)
        assert reward == 100 - 0.5 * 5

    def test_asymmetric_reward(self):
        """Asymmetric reward should weight losses more."""
        from v2.src.rl.reward_functions import RewardFunctions

        win_reward = RewardFunctions.asymmetric_reward(100, win_multiplier=1.0, loss_multiplier=2.0)
        loss_reward = RewardFunctions.asymmetric_reward(-100, win_multiplier=1.0, loss_multiplier=2.0)

        assert win_reward == 100
        assert loss_reward == -200

    def test_composite_reward(self):
        """Composite reward should combine multiple factors."""
        from v2.src.rl.reward_functions import RewardFunctions

        returns = [0.01, -0.005, 0.02, 0.015, -0.01, 0.008] * 5
        reward = RewardFunctions.composite_reward(
            pnl=50,
            returns=returns,
            drawdown=2.0,
        )
        assert isinstance(reward, float)


class TestRewardScaler:
    """Tests for RewardScaler class."""

    def test_reward_scaler_scale(self):
        """RewardScaler should normalize rewards."""
        from v2.src.rl.reward_functions import RewardScaler

        scaler = RewardScaler(window_size=10, clip_range=(-2, 2))

        # Add some rewards
        for r in [1.0, 2.0, 3.0, 4.0, 5.0]:
            scaled = scaler.scale(r)
            assert isinstance(scaled, (int, float))

    def test_reward_scaler_reset(self):
        """RewardScaler should reset correctly."""
        from v2.src.rl.reward_functions import RewardScaler

        scaler = RewardScaler()
        scaler.scale(1.0)
        scaler.scale(2.0)

        scaler.reset()
        assert len(scaler.rewards) == 0


class TestRLTrainer:
    """Tests for RLTrainer class."""

    def test_trainer_init(self, config, sample_data):
        """RLTrainer should initialize correctly."""
        from v2.src.rl.trainer import RLTrainer
        from v2.src.rl.environment import TradingEnvironment

        env = TradingEnvironment(config, sample_data)
        trainer = RLTrainer(config, env)

        assert trainer.config is not None
        assert trainer.env is not None

    @pytest.mark.skipif(True, reason="Requires stable-baselines3")
    def test_trainer_create_ppo_agent(self, config, sample_data):
        """RLTrainer should create PPO agent."""
        from v2.src.rl.trainer import RLTrainer
        from v2.src.rl.environment import TradingEnvironment

        env = TradingEnvironment(config, sample_data)
        trainer = RLTrainer(config, env)

        agent = trainer.create_agent("ppo")
        assert agent is not None
        assert "ppo" in trainer.agents

    @pytest.mark.skipif(True, reason="Requires stable-baselines3")
    def test_trainer_create_sac_agent(self, config, sample_data):
        """RLTrainer should create SAC agent."""
        from v2.src.rl.trainer import RLTrainer
        from v2.src.rl.environment import TradingEnvironment

        env = TradingEnvironment(config, sample_data)
        trainer = RLTrainer(config, env)

        agent = trainer.create_agent("sac")
        assert agent is not None
        assert "sac" in trainer.agents


class TestPPOAgent:
    """Tests for PPOAgent class."""

    @pytest.mark.skipif(True, reason="Requires stable-baselines3")
    def test_ppo_agent_is_enabled(self, config, sample_data):
        """PPO agent should check enabled status."""
        from v2.src.rl.agents.ppo_agent import PPOAgent
        from v2.src.rl.environment import TradingEnvironment, FlatActionWrapper

        env = FlatActionWrapper(TradingEnvironment(config, sample_data))
        agent = PPOAgent(config, env)

        assert agent.is_enabled() is True


class TestSACAgent:
    """Tests for SACAgent class."""

    @pytest.mark.skipif(True, reason="Requires stable-baselines3")
    def test_sac_agent_is_enabled(self, config, sample_data):
        """SAC agent should check enabled status."""
        from v2.src.rl.agents.sac_agent import SACAgent
        from v2.src.rl.environment import TradingEnvironment, FlatActionWrapper

        env = FlatActionWrapper(TradingEnvironment(config, sample_data))
        agent = SACAgent(config, env)

        assert agent.is_enabled() is True
