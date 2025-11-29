"""
Multi-Timeframe RL Ensemble for trading decisions.

Uses multiple RL models trained on different timeframes and
makes trading decisions through confluence voting.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Action mapping constants
ACTION_HOLD = 0
ACTION_BUY = 1
ACTION_SELL = 2

# Action names for display
ACTION_NAMES = {
    ACTION_HOLD: "HOLD",
    ACTION_BUY: "BUY",
    ACTION_SELL: "SELL",
}


class RLEnsemble:
    """
    Committee of RL models for confluence-based trading decisions.

    Each RL model studies a different timeframe:
    - 1m: Scalping patterns
    - 5m: Day trade patterns
    - 15m: Short swing patterns

    The final decision is made by voting (majority).
    """

    # Supported timeframes
    TIMEFRAMES = ["1m", "5m", "15m"]

    # Confidence levels based on vote count
    CONFIDENCE_LEVELS = {
        3: "MAXIMA",
        2: "ALTA",
        1: "BAIXA",
        0: "NENHUMA",
    }

    # Position size multipliers based on confidence
    POSITION_SIZE_MULTIPLIERS = {
        "MAXIMA": 1.0,  # 100% of configured position size
        "ALTA": 0.7,  # 70% of configured position size
        "BAIXA": 0.0,  # Don't trade
        "NENHUMA": 0.0,  # Don't trade
    }

    def __init__(
        self,
        models_dir: str = "v2/models/rl",
        min_votes_to_trade: int = 2,
    ):
        """
        Initialize the RL Ensemble.

        Args:
            models_dir: Directory containing trained models
            min_votes_to_trade: Minimum votes required to execute trade (default 2)
        """
        self.models_dir = Path(models_dir)
        self.min_votes_to_trade = min_votes_to_trade
        self.models: Dict[str, Any] = {}
        self.environments: Dict[str, Any] = {}
        self._loaded = False

    def load_models(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Load all trained models from the models directory.

        Args:
            config: Optional configuration dictionary for environments

        Returns:
            True if at least one model was loaded successfully
        """
        try:
            from stable_baselines3 import PPO
        except ImportError:
            logger.error("stable-baselines3 is required for loading models")
            return False

        loaded_count = 0

        for tf in self.TIMEFRAMES:
            model_path = self.models_dir / f"ppo_{tf}_best.zip"

            if not model_path.exists():
                logger.warning(f"Model not found for timeframe {tf}: {model_path}")
                continue

            try:
                # Load model without environment (for prediction only)
                model = PPO.load(str(model_path), env=None)
                self.models[tf] = model
                loaded_count += 1
                logger.info(f"Loaded model for timeframe {tf}")
            except Exception as e:
                logger.error(f"Failed to load model for {tf}: {e}")

        self._loaded = loaded_count > 0

        if self._loaded:
            logger.info(
                f"Ensemble loaded with {loaded_count}/{len(self.TIMEFRAMES)} models"
            )
        else:
            logger.warning("No models loaded for ensemble")

        return self._loaded

    def predict(self, observations: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Collect votes from all RLs and decide by confluence.

        Args:
            observations: Dictionary mapping timeframe to observation array
                Example: {"1m": obs_1m, "5m": obs_5m, "15m": obs_15m}

        Returns:
            Dictionary with:
                - decision: "BUY" | "SELL" | "HOLD"
                - confidence: 3 | 2 | 1 | 0 (how many RLs agreed)
                - confidence_level: "MAXIMA" | "ALTA" | "BAIXA" | "NENHUMA"
                - votes: {"1m": "BUY", "5m": "BUY", "15m": "HOLD"}
                - should_trade: True | False (>= min_votes_to_trade)
                - position_size_multiplier: 0.0 to 1.0
        """
        if not self._loaded or not self.models:
            logger.warning("No models loaded, returning HOLD")
            return self._create_hold_result()

        # Collect votes from each model
        votes: Dict[str, str] = {}
        raw_actions: Dict[str, int] = {}

        for tf, model in self.models.items():
            if tf not in observations:
                logger.warning(f"No observation provided for timeframe {tf}")
                votes[tf] = "HOLD"
                raw_actions[tf] = ACTION_HOLD
                continue

            try:
                obs = observations[tf]
                # Ensure observation is the right shape
                if len(obs.shape) == 1:
                    obs = obs.reshape(1, -1)

                action, _ = model.predict(obs, deterministic=True)

                # Extract scalar from numpy array if needed
                action_value = int(action.item()) if hasattr(action, 'item') else int(action)

                # Convert action to vote
                vote = self._action_to_vote(action_value)
                votes[tf] = vote
                raw_actions[tf] = action_value

            except Exception as e:
                logger.error(f"Prediction failed for {tf}: {e}")
                votes[tf] = "HOLD"
                raw_actions[tf] = ACTION_HOLD

        # Count votes
        buy_votes = sum(1 for v in votes.values() if v == "BUY")
        sell_votes = sum(1 for v in votes.values() if v == "SELL")
        hold_votes = sum(1 for v in votes.values() if v == "HOLD")

        # Determine decision based on confluence
        decision, confidence = self._determine_decision(buy_votes, sell_votes)

        # Get confidence level and position size multiplier
        confidence_level = self.get_confidence_level(confidence)
        position_multiplier = self.POSITION_SIZE_MULTIPLIERS.get(confidence_level, 0.0)

        # Determine if we should trade
        should_trade = confidence >= self.min_votes_to_trade and decision != "HOLD"

        return {
            "decision": decision,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "votes": votes,
            "raw_actions": raw_actions,
            "buy_votes": buy_votes,
            "sell_votes": sell_votes,
            "hold_votes": hold_votes,
            "should_trade": should_trade,
            "position_size_multiplier": position_multiplier,
        }

    def _action_to_vote(self, action: int) -> str:
        """
        Convert model action to vote string.

        The action space in FlatActionWrapper encodes multiple dimensions,
        but we extract the direction component.

        For simple 3-action spaces:
        - 0 = SELL (direction -1)
        - 1 = HOLD (direction 0)
        - 2 = BUY (direction 1)

        For complex action spaces (FlatActionWrapper):
        The direction is encoded in the lower bits.
        """
        # For 3-direction action space
        n_directions = 3  # [-1, 0, 1] = [SELL, HOLD, BUY]

        # Extract direction from flat action
        direction_idx = action % n_directions

        # Map direction index to vote
        if direction_idx == 0:
            return "SELL"
        elif direction_idx == 1:
            return "HOLD"
        else:
            return "BUY"

    def _determine_decision(self, buy_votes: int, sell_votes: int) -> tuple:
        """
        Determine final decision based on vote counts.

        Args:
            buy_votes: Number of BUY votes
            sell_votes: Number of SELL votes

        Returns:
            Tuple of (decision, confidence)
        """
        if buy_votes >= self.min_votes_to_trade and buy_votes > sell_votes:
            return "BUY", buy_votes
        elif sell_votes >= self.min_votes_to_trade and sell_votes > buy_votes:
            return "SELL", sell_votes
        else:
            # No clear majority or not enough votes
            return "HOLD", 0

    def get_confidence_level(self, votes: int) -> str:
        """
        Get confidence level string based on vote count.

        3 votes = "MAXIMA" (position size 100%)
        2 votes = "ALTA" (position size 70%)
        1 vote = "BAIXA" (don't trade)
        0 votes = "NENHUMA" (don't trade)

        Args:
            votes: Number of agreeing votes

        Returns:
            Confidence level string
        """
        return self.CONFIDENCE_LEVELS.get(votes, "NENHUMA")

    def _create_hold_result(self) -> Dict[str, Any]:
        """Create a default HOLD result when no prediction is possible."""
        return {
            "decision": "HOLD",
            "confidence": 0,
            "confidence_level": "NENHUMA",
            "votes": {tf: "HOLD" for tf in self.TIMEFRAMES},
            "raw_actions": {tf: ACTION_HOLD for tf in self.TIMEFRAMES},
            "buy_votes": 0,
            "sell_votes": 0,
            "hold_votes": len(self.TIMEFRAMES),
            "should_trade": False,
            "position_size_multiplier": 0.0,
        }

    def get_loaded_timeframes(self) -> List[str]:
        """Get list of timeframes with loaded models."""
        return list(self.models.keys())

    def is_ready(self) -> bool:
        """Check if ensemble has enough models loaded to make decisions."""
        return len(self.models) >= self.min_votes_to_trade

    def get_status(self) -> Dict[str, Any]:
        """
        Get status of the ensemble.

        Returns:
            Status dictionary with model information
        """
        return {
            "loaded": self._loaded,
            "total_models": len(self.models),
            "loaded_timeframes": list(self.models.keys()),
            "missing_timeframes": [
                tf for tf in self.TIMEFRAMES if tf not in self.models
            ],
            "min_votes_to_trade": self.min_votes_to_trade,
            "is_ready": self.is_ready(),
        }

    def predict_with_metadata(
        self,
        observations: Dict[str, np.ndarray],
        market_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make prediction with additional market context.

        Args:
            observations: Observations for each timeframe
            market_data: Optional additional market data for context

        Returns:
            Prediction result with additional metadata
        """
        result = self.predict(observations)

        # Add metadata
        result["metadata"] = {
            "loaded_models": len(self.models),
            "timeframes": list(self.models.keys()),
        }

        if market_data:
            result["metadata"]["market_data"] = market_data

        return result
