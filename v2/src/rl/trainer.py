"""
RL Trainer for trading agents.

Provides high-level training interface for multiple RL algorithms.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .environment import TradingEnvironment, FlatActionWrapper
from .agents.ppo_agent import PPOAgent
from .agents.sac_agent import SACAgent

logger = logging.getLogger(__name__)


class RLTrainer:
    """
    Train RL agents for trading.

    Supports multiple agent types (PPO, SAC) with config-driven parameters.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        env: TradingEnvironment,
        eval_env: Optional[TradingEnvironment] = None,
    ):
        """
        Initialize RL trainer.

        Args:
            config: Full configuration dictionary
            env: Training environment
            eval_env: Evaluation environment (optional)
        """
        self.config = config.get("rl", {})
        self.full_config = config
        self.env = env
        self.eval_env = eval_env
        self.agents: Dict[str, Union[PPOAgent, SACAgent]] = {}

    def create_agent(
        self,
        agent_type: str,
        env: Optional[TradingEnvironment] = None,
    ) -> Union[PPOAgent, SACAgent]:
        """
        Create an RL agent of specified type.

        Args:
            agent_type: Agent type ('ppo' or 'sac')
            env: Environment (uses training env if None)

        Returns:
            Created agent
        """
        env = env or self.env

        # Use flat action wrapper for PPO/SAC compatibility
        wrapped_env = FlatActionWrapper(env)

        if agent_type.lower() == "ppo":
            if not self.config.get("agents", {}).get("ppo", {}).get("enabled", True):
                raise ValueError("PPO agent is disabled in config")

            agent = PPOAgent(self.full_config, wrapped_env)
            agent.create_model()
            self.agents["ppo"] = agent

        elif agent_type.lower() == "sac":
            if not self.config.get("agents", {}).get("sac", {}).get("enabled", True):
                raise ValueError("SAC agent is disabled in config")

            agent = SACAgent(self.full_config, wrapped_env)
            agent.create_model()
            self.agents["sac"] = agent

        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        logger.info(f"Created {agent_type.upper()} agent")
        return agent

    def train(
        self,
        agent_type: str,
        total_timesteps: Optional[int] = None,
        callback: Optional[Any] = None,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Train a specific agent.

        Args:
            agent_type: Agent type to train
            total_timesteps: Total timesteps (uses config if None)
            callback: Training callback
            save_path: Path to save model checkpoints

        Returns:
            Training results
        """
        if agent_type.lower() not in self.agents:
            self.create_agent(agent_type)

        agent = self.agents[agent_type.lower()]

        total_timesteps = total_timesteps or self.config.get("training", {}).get(
            "total_timesteps", 100000
        )

        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)

        # Prepare evaluation environment if available
        eval_env = None
        if self.eval_env is not None:
            eval_env = FlatActionWrapper(self.eval_env)

        # Train agent
        agent.train(
            total_timesteps=total_timesteps,
            callback=callback,
            eval_env=eval_env,
            save_path=str(save_path) if save_path else None,
        )

        # Evaluate after training
        results = {}
        if self.eval_env is not None:
            results = agent.evaluate(
                FlatActionWrapper(self.eval_env),
                n_episodes=self.config.get("training", {}).get("n_eval_episodes", 5),
            )

        return results

    def train_all(
        self,
        total_timesteps: Optional[int] = None,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all enabled agents.

        Args:
            total_timesteps: Total timesteps per agent
            save_dir: Directory to save models

        Returns:
            Results for each agent
        """
        results = {}

        agents_config = self.config.get("agents", {})

        if agents_config.get("ppo", {}).get("enabled", True):
            logger.info("Training PPO agent...")
            save_path = None
            if save_dir:
                save_path = Path(save_dir) / "ppo"

            try:
                results["ppo"] = self.train(
                    "ppo",
                    total_timesteps=total_timesteps,
                    save_path=save_path,
                )
            except Exception as e:
                logger.error(f"PPO training failed: {e}")
                results["ppo"] = {"error": str(e)}

        if agents_config.get("sac", {}).get("enabled", True):
            logger.info("Training SAC agent...")
            save_path = None
            if save_dir:
                save_path = Path(save_dir) / "sac"

            try:
                results["sac"] = self.train(
                    "sac",
                    total_timesteps=total_timesteps,
                    save_path=save_path,
                )
            except Exception as e:
                logger.error(f"SAC training failed: {e}")
                results["sac"] = {"error": str(e)}

        return results

    def save(
        self,
        agent_type: str,
        path: Union[str, Path],
    ) -> None:
        """
        Save trained agent.

        Args:
            agent_type: Agent type to save
            path: Save path
        """
        if agent_type.lower() not in self.agents:
            raise ValueError(f"Agent {agent_type} not trained yet")

        self.agents[agent_type.lower()].save(path)

    def load(
        self,
        agent_type: str,
        path: Union[str, Path],
    ) -> Union[PPOAgent, SACAgent]:
        """
        Load a trained agent.

        Args:
            agent_type: Agent type to load
            path: Model path

        Returns:
            Loaded agent
        """
        env = FlatActionWrapper(self.env)

        if agent_type.lower() == "ppo":
            agent = PPOAgent(self.full_config, env)
            agent.load(path)
            self.agents["ppo"] = agent

        elif agent_type.lower() == "sac":
            agent = SACAgent(self.full_config, env)
            agent.load(path)
            self.agents["sac"] = agent

        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        return agent

    def evaluate(
        self,
        agent_type: str,
        env: Optional[TradingEnvironment] = None,
        n_episodes: int = 10,
    ) -> Dict[str, Any]:
        """
        Evaluate a trained agent.

        Args:
            agent_type: Agent type to evaluate
            env: Evaluation environment (uses eval_env if None)
            n_episodes: Number of episodes

        Returns:
            Evaluation metrics
        """
        if agent_type.lower() not in self.agents:
            raise ValueError(f"Agent {agent_type} not trained yet")

        eval_env = env or self.eval_env or self.env
        wrapped_env = FlatActionWrapper(eval_env)

        return self.agents[agent_type.lower()].evaluate(
            wrapped_env,
            n_episodes=n_episodes,
        )

    def compare_agents(
        self,
        env: Optional[TradingEnvironment] = None,
        n_episodes: int = 20,
    ) -> pd.DataFrame:
        """
        Compare all trained agents.

        Args:
            env: Evaluation environment
            n_episodes: Number of episodes per agent

        Returns:
            DataFrame with comparison results
        """
        results = []

        for agent_type in self.agents:
            try:
                metrics = self.evaluate(agent_type, env, n_episodes)
                metrics["agent"] = agent_type
                results.append(metrics)
            except Exception as e:
                logger.error(f"Evaluation failed for {agent_type}: {e}")

        return pd.DataFrame(results)

    def predict(
        self,
        agent_type: str,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> tuple:
        """
        Get prediction from trained agent.

        Args:
            agent_type: Agent type
            observation: Environment observation
            deterministic: Whether to use deterministic action

        Returns:
            Tuple of (action, state)
        """
        if agent_type.lower() not in self.agents:
            raise ValueError(f"Agent {agent_type} not trained yet")

        return self.agents[agent_type.lower()].predict(observation, deterministic)

    def get_best_agent(self) -> Optional[str]:
        """
        Get the best performing agent based on evaluation.

        Returns:
            Best agent type or None if no agents trained
        """
        if not self.agents:
            return None

        if len(self.agents) == 1:
            return list(self.agents.keys())[0]

        # Compare agents
        comparison = self.compare_agents()
        if comparison.empty:
            return list(self.agents.keys())[0]

        # Sort by mean reward
        best = comparison.sort_values("mean_reward", ascending=False).iloc[0]
        return best["agent"]

    def create_ensemble_predictor(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> Callable:
        """
        Create ensemble predictor combining multiple agents.

        Args:
            weights: Dict mapping agent type to weight

        Returns:
            Ensemble prediction function
        """
        if weights is None:
            weights = {k: 1.0 / len(self.agents) for k in self.agents}

        def ensemble_predict(
            observation: np.ndarray,
            deterministic: bool = True,
        ) -> tuple:
            """Ensemble prediction."""
            weighted_action = None

            for agent_type, weight in weights.items():
                if agent_type in self.agents:
                    action, _ = self.agents[agent_type].predict(
                        observation, deterministic
                    )

                    if weighted_action is None:
                        weighted_action = action * weight
                    else:
                        weighted_action += action * weight

            return weighted_action, None

        return ensemble_predict
