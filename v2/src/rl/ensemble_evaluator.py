"""
Ensemble Evaluator for multi-timeframe RL models.

Provides backtesting and comparison tools for the RL ensemble system.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .ensemble import RLEnsemble

logger = logging.getLogger(__name__)


class EnsembleEvaluator:
    """
    Evaluates performance of the RL ensemble committee.

    Provides methods to:
    - Backtest the ensemble using confluence voting
    - Compare individual model performance vs ensemble
    - Analyze voting patterns and agreement rates
    """

    def __init__(
        self,
        ensemble: RLEnsemble,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the ensemble evaluator.

        Args:
            ensemble: Loaded RLEnsemble instance
            config: Optional configuration dictionary
        """
        self.ensemble = ensemble
        self.config = config or {}
        self.initial_balance = self.config.get("initial_balance", 10000)
        self.position_size_pct = self.config.get("position_size_pct", 0.1)  # 10%
        self.fees = self.config.get("fees", {"taker": 0.0004, "maker": 0.0002})

    def backtest_ensemble(
        self,
        data: Dict[str, pd.DataFrame],
        features: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """
        Run backtest using confluence voting.

        Only enters positions when >= min_votes_to_trade RLs agree.

        Args:
            data: Dictionary of DataFrames for each timeframe
                {"1m": df_1m, "5m": df_5m, "15m": df_15m}
            features: Optional pre-computed features for each timeframe

        Returns:
            Backtest results dictionary
        """
        from .environment import TradingEnvironment

        # Validate data
        if not self._validate_data(data):
            return {"error": "Invalid data provided"}

        # Initialize tracking variables
        balance = self.initial_balance
        equity_curve = [balance]
        trades: List[Dict[str, Any]] = []
        positions: List[Dict[str, Any]] = []
        voting_history: List[Dict[str, Any]] = []

        # Get the reference timeframe (usually 5m for trading decisions)
        ref_tf = "5m" if "5m" in data else list(data.keys())[0]
        ref_data = data[ref_tf]

        # Current position state
        position = 0  # -1 short, 0 flat, 1 long
        entry_price = 0.0
        entry_step = 0

        # Process each step using 5-minute intervals as decision points
        decision_interval = self._get_decision_interval(ref_tf)

        for step in range(0, len(ref_data), decision_interval):
            # Get observations for each timeframe at this point
            observations = self._get_observations(data, features, step)

            if not observations:
                continue

            # Get ensemble prediction
            prediction = self.ensemble.predict(observations)

            # Record voting history
            voting_history.append(
                {
                    "step": step,
                    "votes": prediction["votes"].copy(),
                    "decision": prediction["decision"],
                    "confidence": prediction["confidence"],
                    "should_trade": prediction["should_trade"],
                }
            )

            # Get current price
            current_price = float(ref_data.iloc[step].get("close", 0))
            if current_price <= 0:
                continue

            # Execute trading logic
            if prediction["should_trade"]:
                decision = prediction["decision"]
                position_multiplier = prediction["position_size_multiplier"]

                # Close existing position if direction changes
                if position != 0:
                    if (position == 1 and decision == "SELL") or (
                        position == -1 and decision == "BUY"
                    ):
                        # Close position
                        pnl = self._calculate_pnl(
                            position, entry_price, current_price, balance
                        )
                        balance += pnl

                        trades.append(
                            {
                                "entry_step": entry_step,
                                "exit_step": step,
                                "entry_price": entry_price,
                                "exit_price": current_price,
                                "direction": "LONG" if position == 1 else "SHORT",
                                "pnl": pnl,
                                "reason": "signal_reversal",
                            }
                        )

                        position = 0
                        entry_price = 0.0

                # Open new position if flat
                if position == 0:
                    if decision == "BUY":
                        position = 1
                        entry_price = current_price
                        entry_step = step
                    elif decision == "SELL":
                        position = -1
                        entry_price = current_price
                        entry_step = step

            # Update equity curve
            unrealized_pnl = 0.0
            if position != 0:
                unrealized_pnl = self._calculate_pnl(
                    position, entry_price, current_price, balance
                )
            equity_curve.append(balance + unrealized_pnl)

        # Close any remaining position
        if position != 0 and len(ref_data) > 0:
            final_price = float(ref_data.iloc[-1].get("close", entry_price))
            pnl = self._calculate_pnl(position, entry_price, final_price, balance)
            balance += pnl

            trades.append(
                {
                    "entry_step": entry_step,
                    "exit_step": len(ref_data) - 1,
                    "entry_price": entry_price,
                    "exit_price": final_price,
                    "direction": "LONG" if position == 1 else "SHORT",
                    "pnl": pnl,
                    "reason": "end_of_data",
                }
            )

        # Calculate metrics
        metrics = self._calculate_metrics(
            trades, equity_curve, voting_history, self.initial_balance
        )

        return {
            "metrics": metrics,
            "trades": trades,
            "equity_curve": equity_curve,
            "voting_history": voting_history,
            "final_balance": balance,
        }

    def evaluate_individual_models(
        self,
        data: Dict[str, pd.DataFrame],
        features: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate each individual RL model separately.

        Args:
            data: Dictionary of DataFrames for each timeframe
            features: Optional pre-computed features

        Returns:
            Results for each individual model
        """
        results = {}

        for tf in self.ensemble.get_loaded_timeframes():
            if tf not in data:
                continue

            logger.info(f"Evaluating individual model: {tf}")

            # Create single-model ensemble for evaluation
            individual_results = self._evaluate_single_model(
                tf, data[tf], features.get(tf) if features else None
            )
            results[tf] = individual_results

        return results

    def compare_with_single_model(
        self,
        data: Dict[str, pd.DataFrame],
        features: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        Compare ensemble performance with each individual model.

        Args:
            data: Dictionary of DataFrames for each timeframe
            features: Optional pre-computed features

        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []

        # Evaluate ensemble
        logger.info("Evaluating ensemble...")
        ensemble_results = self.backtest_ensemble(data, features)

        comparison_data.append(
            {
                "model": "ENSEMBLE",
                **ensemble_results["metrics"],
            }
        )

        # Evaluate individual models
        individual_results = self.evaluate_individual_models(data, features)

        for tf, results in individual_results.items():
            comparison_data.append(
                {
                    "model": f"RL_{tf}",
                    **results.get("metrics", {}),
                }
            )

        return pd.DataFrame(comparison_data)

    def analyze_voting_patterns(
        self,
        voting_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze voting patterns and agreement rates.

        Args:
            voting_history: List of voting records from backtest

        Returns:
            Analysis of voting patterns
        """
        if not voting_history:
            return {"error": "No voting history provided"}

        total_decisions = len(voting_history)
        unanimous_buy = 0
        unanimous_sell = 0
        unanimous_hold = 0
        majority_buy = 0
        majority_sell = 0
        no_consensus = 0

        timeframe_stats: Dict[str, Dict[str, int]] = {
            tf: {"BUY": 0, "SELL": 0, "HOLD": 0}
            for tf in self.ensemble.TIMEFRAMES
        }

        for record in voting_history:
            votes = record["votes"]

            # Count individual timeframe votes
            for tf, vote in votes.items():
                if tf in timeframe_stats:
                    timeframe_stats[tf][vote] = timeframe_stats[tf].get(vote, 0) + 1

            # Analyze consensus
            vote_values = list(votes.values())
            buy_count = vote_values.count("BUY")
            sell_count = vote_values.count("SELL")
            hold_count = vote_values.count("HOLD")

            if buy_count == 3:
                unanimous_buy += 1
            elif sell_count == 3:
                unanimous_sell += 1
            elif hold_count == 3:
                unanimous_hold += 1
            elif buy_count >= 2:
                majority_buy += 1
            elif sell_count >= 2:
                majority_sell += 1
            else:
                no_consensus += 1

        return {
            "total_decisions": total_decisions,
            "unanimous": {
                "buy": unanimous_buy,
                "sell": unanimous_sell,
                "hold": unanimous_hold,
                "total": unanimous_buy + unanimous_sell + unanimous_hold,
                "rate": (unanimous_buy + unanimous_sell + unanimous_hold)
                / total_decisions
                if total_decisions > 0
                else 0,
            },
            "majority": {
                "buy": majority_buy,
                "sell": majority_sell,
                "total": majority_buy + majority_sell,
            },
            "no_consensus": {
                "count": no_consensus,
                "rate": no_consensus / total_decisions if total_decisions > 0 else 0,
            },
            "timeframe_stats": timeframe_stats,
            "agreement_rate": 1 - (no_consensus / total_decisions)
            if total_decisions > 0
            else 0,
        }

    def _validate_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Validate that data contains required columns."""
        required_columns = ["close"]

        for tf, df in data.items():
            if df is None or len(df) == 0:
                logger.warning(f"Empty data for timeframe {tf}")
                return False

            for col in required_columns:
                if col not in df.columns and col.capitalize() not in df.columns:
                    logger.warning(f"Missing column {col} in {tf} data")
                    return False

        return True

    def _get_decision_interval(self, timeframe: str) -> int:
        """Get the decision interval in bars for the given timeframe."""
        # Decision every 5 minutes equivalent
        intervals = {
            "1m": 5,  # Every 5 bars
            "5m": 1,  # Every bar
            "15m": 1,  # Every bar (decision at 15m granularity)
        }
        return intervals.get(timeframe, 1)

    def _get_observations(
        self,
        data: Dict[str, pd.DataFrame],
        features: Optional[Dict[str, pd.DataFrame]],
        step: int,
    ) -> Dict[str, np.ndarray]:
        """Get observations for all timeframes at a given step."""
        observations = {}

        # Feature names expected by the environment
        feature_names = [
            "ofi",
            "tfi",
            "rsi",
            "adx",
            "regime",
            "position",
            "pnl",
            "drawdown",
        ]

        for tf in self.ensemble.get_loaded_timeframes():
            if tf not in data:
                continue

            df = data[tf]

            # Adjust step index based on timeframe ratio
            tf_step = self._convert_step_to_timeframe(step, "5m", tf, data)

            if tf_step >= len(df):
                tf_step = len(df) - 1

            if tf_step < 0:
                tf_step = 0

            # Get row data
            if features and tf in features:
                row = features[tf].iloc[tf_step]
            else:
                row = df.iloc[tf_step]

            # Build observation array
            obs_values = []
            for feat in feature_names:
                if feat in ["position", "pnl", "drawdown"]:
                    # These are state variables, set to 0 for now
                    obs_values.append(0.0)
                else:
                    value = row.get(feat, 0.0)
                    if pd.isna(value):
                        value = 0.0
                    obs_values.append(float(value))

            observations[tf] = np.array(obs_values, dtype=np.float32)

        return observations

    def _convert_step_to_timeframe(
        self,
        step: int,
        from_tf: str,
        to_tf: str,
        data: Dict[str, pd.DataFrame],
    ) -> int:
        """Convert step index from one timeframe to another."""
        # Timeframe multipliers (relative to 1m)
        tf_minutes = {"1m": 1, "5m": 5, "15m": 15}

        from_mins = tf_minutes.get(from_tf, 5)
        to_mins = tf_minutes.get(to_tf, 5)

        # Convert step
        if from_mins == to_mins:
            return step
        elif from_mins < to_mins:
            # Going from smaller to larger timeframe
            return step * from_mins // to_mins
        else:
            # Going from larger to smaller timeframe
            return step * from_mins // to_mins

    def _calculate_pnl(
        self,
        position: int,
        entry_price: float,
        exit_price: float,
        balance: float,
    ) -> float:
        """Calculate P&L for a trade."""
        if entry_price <= 0:
            return 0.0

        position_value = balance * self.position_size_pct

        if position == 1:  # Long
            pnl = position_value * (exit_price - entry_price) / entry_price
        else:  # Short
            pnl = position_value * (entry_price - exit_price) / entry_price

        # Apply fees
        fee_rate = self.fees.get("taker", 0.0004)
        pnl -= position_value * fee_rate * 2  # Entry and exit fees

        return pnl

    def _evaluate_single_model(
        self,
        timeframe: str,
        data: pd.DataFrame,
        features: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Evaluate a single model independently."""
        if timeframe not in self.ensemble.models:
            return {"error": f"Model not loaded for {timeframe}"}

        model = self.ensemble.models[timeframe]

        balance = self.initial_balance
        equity_curve = [balance]
        trades: List[Dict[str, Any]] = []

        position = 0
        entry_price = 0.0
        entry_step = 0

        feature_names = [
            "ofi",
            "tfi",
            "rsi",
            "adx",
            "regime",
            "position",
            "pnl",
            "drawdown",
        ]

        for step in range(len(data)):
            # Build observation
            if features is not None and len(features) > step:
                row = features.iloc[step]
            else:
                row = data.iloc[step]

            obs_values = []
            for feat in feature_names:
                if feat in ["position", "pnl", "drawdown"]:
                    obs_values.append(0.0)
                else:
                    value = row.get(feat, 0.0)
                    if pd.isna(value):
                        value = 0.0
                    obs_values.append(float(value))

            obs = np.array(obs_values, dtype=np.float32).reshape(1, -1)

            # Get prediction
            try:
                action, _ = model.predict(obs, deterministic=True)
                vote = self.ensemble._action_to_vote(int(action))
            except Exception:
                vote = "HOLD"

            current_price = float(row.get("close", row.get("Close", 0)))
            if current_price <= 0:
                continue

            # Simple trading logic
            if position == 0:
                if vote == "BUY":
                    position = 1
                    entry_price = current_price
                    entry_step = step
                elif vote == "SELL":
                    position = -1
                    entry_price = current_price
                    entry_step = step
            else:
                should_close = (position == 1 and vote == "SELL") or (
                    position == -1 and vote == "BUY"
                )
                if should_close:
                    pnl = self._calculate_pnl(
                        position, entry_price, current_price, balance
                    )
                    balance += pnl

                    trades.append(
                        {
                            "entry_step": entry_step,
                            "exit_step": step,
                            "entry_price": entry_price,
                            "exit_price": current_price,
                            "direction": "LONG" if position == 1 else "SHORT",
                            "pnl": pnl,
                        }
                    )

                    position = 0
                    entry_price = 0.0

                    # Open new position in opposite direction
                    if vote == "BUY":
                        position = 1
                        entry_price = current_price
                        entry_step = step
                    elif vote == "SELL":
                        position = -1
                        entry_price = current_price
                        entry_step = step

            # Update equity curve
            unrealized_pnl = 0.0
            if position != 0:
                unrealized_pnl = self._calculate_pnl(
                    position, entry_price, current_price, balance
                )
            equity_curve.append(balance + unrealized_pnl)

        # Close any remaining position
        if position != 0 and len(data) > 0:
            final_price = float(data.iloc[-1].get("close", entry_price))
            pnl = self._calculate_pnl(position, entry_price, final_price, balance)
            balance += pnl
            trades.append(
                {
                    "entry_step": entry_step,
                    "exit_step": len(data) - 1,
                    "entry_price": entry_price,
                    "exit_price": final_price,
                    "direction": "LONG" if position == 1 else "SHORT",
                    "pnl": pnl,
                }
            )

        metrics = self._calculate_metrics(trades, equity_curve, [], self.initial_balance)

        return {
            "metrics": metrics,
            "trades": trades,
            "equity_curve": equity_curve,
            "final_balance": balance,
        }

    def _calculate_metrics(
        self,
        trades: List[Dict[str, Any]],
        equity_curve: List[float],
        voting_history: List[Dict[str, Any]],
        initial_balance: float,
    ) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if not trades:
            return {
                "n_trades": 0,
                "total_pnl": 0.0,
                "total_return_pct": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "avg_trade": 0.0,
            }

        # Trade statistics
        n_trades = len(trades)
        total_pnl = sum(t["pnl"] for t in trades)
        winning_trades = [t for t in trades if t["pnl"] > 0]
        losing_trades = [t for t in trades if t["pnl"] <= 0]

        win_rate = len(winning_trades) / n_trades * 100 if n_trades > 0 else 0

        gross_profit = sum(t["pnl"] for t in winning_trades)
        gross_loss = abs(sum(t["pnl"] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Return metrics
        total_return_pct = (total_pnl / initial_balance) * 100

        # Sharpe ratio
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            returns = returns[~np.isnan(returns)]
            if len(returns) > 0:
                sharpe = (
                    returns.mean() / returns.std() * np.sqrt(252)
                    if returns.std() > 0
                    else 0
                )
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Max drawdown
        if equity_curve:
            equity_arr = np.array(equity_curve)
            peak = np.maximum.accumulate(equity_arr)
            drawdown = (peak - equity_arr) / peak
            max_drawdown = np.max(drawdown) * 100
        else:
            max_drawdown = 0

        return {
            "n_trades": n_trades,
            "total_pnl": total_pnl,
            "total_return_pct": total_return_pct,
            "win_rate": win_rate,
            "profit_factor": profit_factor if profit_factor != float("inf") else 999.99,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "avg_trade": total_pnl / n_trades if n_trades > 0 else 0,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
        }
