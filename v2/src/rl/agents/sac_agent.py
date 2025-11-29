"""
SAC Agent wrapper for trading.

Provides a high-level interface for SAC agent training and inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, TYPE_CHECKING

import numpy as np

logger = logging.getLogger(__name__)

# Try to import stable-baselines3
try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import (
        BaseCallback,
        EvalCallback,
        CheckpointCallback,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    SAC = None  # type: ignore
    BaseCallback = None  # type: ignore
    logger.warning("stable-baselines3 not installed. SAC agent will not be available.")


class SACAgent:
    """
    SAC (Soft Actor-Critic) agent wrapper.

    All parameters come from config - no hardcoded values.

    Config parameters (rl.agents.sac):
    - learning_rate: Learning rate
    - buffer_size: Replay buffer size
    - learning_starts: Steps before learning
    - batch_size: Batch size for updates
    - tau: Soft update coefficient
    - gamma: Discount factor
    """

    def __init__(self, config: Dict[str, Any], env: Any):
        """
        Initialize SAC agent.

        Args:
            config: Full configuration dictionary
            env: Training environment
        """
        if not SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for SAC agent")

        self.config = config.get("rl", {}).get("agents", {}).get("sac", {})
        self.training_config = config.get("rl", {}).get("training", {})
        self.env = env
        self.model: Optional[SAC] = None

    def create_model(
        self,
        policy: str = "MlpPolicy",
        tensorboard_log: Optional[str] = None,
    ) -> Any:
        """
        Create SAC model with config parameters.

        Args:
            policy: Policy network type
            tensorboard_log: Path for tensorboard logs

        Returns:
            SAC model
        """
        # Wrap environment if needed
        if not hasattr(self.env, "num_envs"):
            env = DummyVecEnv([lambda: self.env])
        else:
            env = self.env

        self.model = SAC(
            policy=policy,
            env=env,
            learning_rate=self.config.get("learning_rate", 0.0003),
            buffer_size=self.config.get("buffer_size", 1000000),
            learning_starts=self.config.get("learning_starts", 100),
            batch_size=self.config.get("batch_size", 256),
            tau=self.config.get("tau", 0.005),
            gamma=self.config.get("gamma", 0.99),
            train_freq=self.config.get("train_freq", 1),
            gradient_steps=self.config.get("gradient_steps", 1),
            ent_coef=self.config.get("ent_coef", "auto"),
            target_update_interval=self.config.get("target_update_interval", 1),
            target_entropy=self.config.get("target_entropy", "auto"),
            verbose=1,
            tensorboard_log=tensorboard_log,
        )

        logger.info(
            f"Created SAC model with lr={self.config.get('learning_rate', 0.0003)}, "
            f"buffer_size={self.config.get('buffer_size', 1000000)}"
        )

        return self.model

    def train(
        self,
        total_timesteps: Optional[int] = None,
        callback: Optional[BaseCallback] = None,
        eval_env: Optional[Any] = None,
        eval_freq: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Train the SAC model.

        Args:
            total_timesteps: Total timesteps to train
            callback: Training callback
            eval_env: Environment for evaluation
            eval_freq: Evaluation frequency
            save_path: Path to save checkpoints
        """
        if self.model is None:
            self.create_model()

        total_timesteps = total_timesteps or self.training_config.get(
            "total_timesteps", 100000
        )
        eval_freq = eval_freq or self.training_config.get("eval_freq", 10000)

        # Create callbacks
        callbacks = []

        if callback is not None:
            callbacks.append(callback)

        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=save_path,
                log_path=save_path,
                eval_freq=eval_freq,
                n_eval_episodes=self.training_config.get("n_eval_episodes", 5),
                deterministic=True,
            )
            callbacks.append(eval_callback)

        if save_path:
            save_freq = self.training_config.get("save_freq", 10000)
            checkpoint_callback = CheckpointCallback(
                save_freq=save_freq,
                save_path=save_path,
                name_prefix="sac_trading",
            )
            callbacks.append(checkpoint_callback)

        logger.info(f"Starting SAC training for {total_timesteps} timesteps")

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
            progress_bar=True,
        )

        logger.info("SAC training complete")

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> tuple:
        """
        Get action prediction from model.

        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic action

        Returns:
            Tuple of (action, state)
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")

        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to file.

        Args:
            path: Save path
        """
        if self.model is None:
            raise ValueError("No model to save")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(str(path))
        logger.info(f"Saved SAC model to {path}")

    def load(self, path: Union[str, Path], env: Optional[Any] = None) -> Any:
        """
        Load model from file.

        Args:
            path: Model path
            env: Environment (uses original if None)

        Returns:
            Loaded SAC model
        """
        env = env or self.env

        # Wrap environment if needed
        if not hasattr(env, "num_envs"):
            env = DummyVecEnv([lambda: env])

        self.model = SAC.load(str(path), env=env)
        logger.info(f"Loaded SAC model from {path}")

        return self.model

    def get_policy(self) -> Any:
        """Get the policy network."""
        if self.model is None:
            return None
        return self.model.policy

    def evaluate(
        self,
        env: Any,
        n_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate the model.

        Args:
            env: Evaluation environment
            n_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic actions

        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model to evaluate")

        episode_rewards = []
        episode_lengths = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0

            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

        return {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_length": float(np.mean(episode_lengths)),
            "min_reward": float(np.min(episode_rewards)),
            "max_reward": float(np.max(episode_rewards)),
        }

    def is_enabled(self) -> bool:
        """Check if SAC is enabled in config."""
        return self.config.get("enabled", True)

    def get_replay_buffer_size(self) -> int:
        """Get current replay buffer utilization."""
        if self.model is None:
            return 0
        return self.model.replay_buffer.size() if hasattr(self.model, "replay_buffer") else 0
