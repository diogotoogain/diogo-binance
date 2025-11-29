"""
Honest Simulator - Backtesting with Look-Ahead Bias Protection.

This module implements a simulator that GUARANTEES the model never sees
future data (protection against look-ahead bias).

Key Principles:
1. Data is processed in strict chronological order
2. Model only accesses data[:current_index]
3. Decision is made BEFORE seeing the result
4. Learning happens AFTER the decision
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for honest simulation."""

    initial_balance: float = 10000.0
    trade_cost_pct: float = 0.0004  # 0.04% Binance taker fee
    slippage_pct: float = 0.0001   # 0.01% slippage
    max_leverage: float = 5.0
    warmup_samples: int = 1000     # Samples before starting to trade
    lookback_window: int = 100     # Bars to use for feature calculation


@dataclass
class TradeRecord:
    """Record of a single trade."""

    entry_time: datetime
    exit_time: Optional[datetime]
    direction: int  # 1 for long, -1 for short
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: float = 0.0
    fees: float = 0.0
    exit_reason: str = ""
    entry_step: int = 0
    exit_step: int = 0


@dataclass
class SimulationResult:
    """Result from simulation."""

    trades: List[TradeRecord]
    equity_curve: List[float]
    timestamps: List[datetime]
    metrics: Dict[str, float]
    drift_events: List[Dict[str, Any]]
    online_model_stats: Dict[str, Any]
    config: SimulationConfig


class HonestSimulator:
    """
    Simulator that GUARANTEES the model never sees future data.

    Principles:
    1. Data is processed in strict chronological order
    2. Model only accesses data[:current_index]
    3. Decision is made BEFORE seeing the result
    4. Learning happens AFTER the decision
    """

    def __init__(
        self,
        config: SimulationConfig,
        online_learner: Optional[Any] = None,
        strategy: Optional[Any] = None,
        drift_detector: Optional[Any] = None
    ):
        """
        Initialize HonestSimulator.

        Args:
            config: Simulation configuration
            online_learner: Optional online learning model (e.g., OnlineLearner)
            strategy: Optional strategy for signal generation
            drift_detector: Optional drift detector (e.g., MarketDriftDetector)
        """
        self.config = config
        self.online_learner = online_learner
        self.strategy = strategy
        self.drift_detector = drift_detector

        # State
        self.balance = config.initial_balance
        self.position: Optional[TradeRecord] = None
        self.trades: List[TradeRecord] = []
        self.equity_curve: List[float] = [config.initial_balance]
        self.timestamps: List[datetime] = []
        self.drift_events: List[Dict[str, Any]] = []

        # Statistics
        self.n_samples_processed = 0
        self.n_trades = 0
        self.total_pnl = 0.0

    def run(
        self,
        data: pd.DataFrame,
        feature_calculator: Optional[Callable] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> SimulationResult:
        """
        Execute honest simulation.

        CRITICAL: At each step, the model only sees data up to index i-1.
        Decision is made, then result is observed, then learning occurs.

        Args:
            data: DataFrame ordered by timestamp with OHLCV data
            feature_calculator: Optional function to calculate features
            progress_callback: Optional callback for progress updates

        Returns:
            SimulationResult with all trades and metrics
        """
        # Ensure chronological order
        data = data.sort_values('timestamp' if 'timestamp' in data.columns else 'open_time')
        data = data.reset_index(drop=True)

        logger.info(
            f"Starting honest simulation: {len(data):,} candles, "
            f"warmup={self.config.warmup_samples}"
        )

        start_idx = max(self.config.lookback_window, self.config.warmup_samples)

        for i in range(start_idx, len(data)):
            self.n_samples_processed += 1

            # 1. SNAPSHOT OF THE PAST (never sees future!)
            historical = data.iloc[:i].copy()
            current_candle = data.iloc[i]

            # Get timestamp
            timestamp = self._get_timestamp(current_candle)
            self.timestamps.append(timestamp)

            # 2. CALCULATE FEATURES (only with past data)
            features = self._calculate_features(historical, feature_calculator)

            # 3. ONLINE LEARNING MAKES PREDICTION
            ol_prediction = None
            if self.online_learner is not None and features:
                ol_prediction = self._get_ol_prediction(features)

            # 4. STRATEGY GENERATES SIGNAL
            strategy_signal = None
            if self.strategy is not None:
                strategy_signal = self._get_strategy_signal(features, historical)

            # 5. COMBINE DECISIONS (confluence)
            final_decision = self._combine_decisions(ol_prediction, strategy_signal)

            # 6. EXECUTE TRADE (at opening price)
            self._execute_decision(final_decision, current_candle, i)

            # 7. UPDATE EQUITY
            self._update_equity(current_candle)

            # 8. DEFERRED LEARNING: Learn from PREVIOUS candle's outcome
            # We learn what happened from the previous candle now that we can see
            # the current candle. This ensures we never use future data for decisions.
            if self.online_learner is not None and i > start_idx:
                prev_candle = data.iloc[i - 1]
                prev_features = self._calculate_features(data.iloc[:i - 1], None)
                if prev_features:
                    # Label based on whether price went up from prev to current
                    label = self._get_label(prev_candle, current_candle)
                    self._learn_from_sample(prev_features, label)

            # 9. DETECT DRIFT (market change)
            if self.drift_detector is not None:
                self._check_drift(current_candle)

            # Progress callback
            if progress_callback is not None and i % 10000 == 0:
                progress_callback(i - start_idx, len(data) - start_idx)

        # Close any open position at end
        if self.position is not None:
            self._close_position(data.iloc[-1], len(data) - 1, "end_of_simulation")

        # Calculate final metrics
        metrics = self._calculate_metrics()

        logger.info(
            f"Simulation complete: {self.n_trades} trades, "
            f"Final balance: ${self.balance:,.2f}"
        )

        return SimulationResult(
            trades=self.trades,
            equity_curve=self.equity_curve,
            timestamps=self.timestamps,
            metrics=metrics,
            drift_events=self.drift_events,
            online_model_stats=self._get_ol_stats(),
            config=self.config
        )

    def _calculate_features(
        self,
        historical: pd.DataFrame,
        feature_calculator: Optional[Callable]
    ) -> Dict[str, float]:
        """Calculate features from historical data only."""
        if feature_calculator is not None:
            return feature_calculator(historical)

        if len(historical) < 20:
            return {}

        # Default feature calculation
        close = historical['close'].values
        volume = historical['volume'].values if 'volume' in historical.columns else np.ones(len(close))

        features = {}

        # Returns
        returns = np.diff(close) / close[:-1]
        if len(returns) > 0:
            features['return_1'] = float(returns[-1])
            features['return_5'] = float(np.sum(returns[-5:])) if len(returns) >= 5 else 0.0
            features['return_20'] = float(np.sum(returns[-20:])) if len(returns) >= 20 else 0.0

        # Volatility
        if len(returns) >= 20:
            features['volatility_20'] = float(np.std(returns[-20:]))

        # Volume
        if len(volume) >= 20:
            features['volume_ratio'] = float(volume[-1] / np.mean(volume[-20:]))

        # Price position
        if len(close) >= 20:
            high_20 = np.max(close[-20:])
            low_20 = np.min(close[-20:])
            if high_20 != low_20:
                features['price_position'] = float((close[-1] - low_20) / (high_20 - low_20))

        return features

    def _get_ol_prediction(self, features: Dict[str, float]) -> Optional[int]:
        """Get prediction from online learner."""
        try:
            pred = self.online_learner.predict_one(features)
            return pred
        except Exception as e:
            logger.debug(f"OL prediction failed: {e}")
            return None

    def _get_strategy_signal(
        self,
        features: Dict[str, float],
        historical: pd.DataFrame
    ) -> Optional[int]:
        """Get signal from strategy."""
        try:
            # Strategy may expect market_data dict
            market_data = {
                **features,
                'price': float(historical['close'].iloc[-1])
            }
            signal = self.strategy.generate_signal(market_data)
            if signal is not None:
                if hasattr(signal, 'direction'):
                    return signal.direction.value if hasattr(signal.direction, 'value') else int(signal.direction)
                return int(signal)
        except Exception as e:
            logger.debug(f"Strategy signal failed: {e}")
        return None

    def _combine_decisions(
        self,
        ol_prediction: Optional[int],
        strategy_signal: Optional[int]
    ) -> int:
        """
        Combine predictions from multiple sources.

        Returns:
            1 for buy, -1 for sell, 0 for hold
        """
        # If we have both, require agreement (confluence)
        if ol_prediction is not None and strategy_signal is not None:
            if ol_prediction == strategy_signal:
                return ol_prediction
            return 0  # Disagreement = hold

        # Single source
        if ol_prediction is not None:
            return ol_prediction
        if strategy_signal is not None:
            return strategy_signal

        return 0

    def _execute_decision(
        self,
        decision: int,
        candle: pd.Series,
        step: int
    ) -> None:
        """Execute trading decision."""
        current_price = float(candle.get('open', candle.get('close', 0)))

        # Check for exit conditions on existing position
        if self.position is not None:
            should_exit = self._should_exit_position(candle, decision)
            if should_exit:
                self._close_position(candle, step, "signal")

        # Open new position if no current position and signal exists
        if self.position is None and decision != 0:
            self._open_position(decision, candle, step)

    def _should_exit_position(self, candle: pd.Series, new_signal: int) -> bool:
        """Check if current position should be closed."""
        if self.position is None:
            return False

        # Exit on opposite signal
        if new_signal != 0 and new_signal != self.position.direction:
            return True

        return False

    def _open_position(
        self,
        direction: int,
        candle: pd.Series,
        step: int
    ) -> None:
        """Open a new position."""
        price = float(candle.get('open', candle.get('close', 0)))

        # Apply slippage
        if direction == 1:
            entry_price = price * (1 + self.config.slippage_pct)
        else:
            entry_price = price * (1 - self.config.slippage_pct)

        # Calculate position size (fixed fraction of balance)
        position_size = self.balance * 0.1  # 10% of balance

        timestamp = self._get_timestamp(candle)

        self.position = TradeRecord(
            entry_time=timestamp,
            exit_time=None,
            direction=direction,
            entry_price=entry_price,
            exit_price=None,
            size=position_size,
            entry_step=step
        )

    def _close_position(
        self,
        candle: pd.Series,
        step: int,
        reason: str
    ) -> None:
        """Close current position."""
        if self.position is None:
            return

        price = float(candle.get('close', candle.get('open', 0)))

        # Apply slippage
        if self.position.direction == 1:
            exit_price = price * (1 - self.config.slippage_pct)
        else:
            exit_price = price * (1 + self.config.slippage_pct)

        # Calculate P&L
        if self.position.direction == 1:
            pnl_pct = (exit_price - self.position.entry_price) / self.position.entry_price
        else:
            pnl_pct = (self.position.entry_price - exit_price) / self.position.entry_price

        pnl = self.position.size * pnl_pct

        # Deduct fees
        fees = self.position.size * self.config.trade_cost_pct * 2  # Entry + exit
        pnl -= fees

        # Update position
        self.position.exit_time = self._get_timestamp(candle)
        self.position.exit_price = exit_price
        self.position.pnl = pnl
        self.position.fees = fees
        self.position.exit_reason = reason
        self.position.exit_step = step

        # Update balance
        self.balance += pnl
        self.total_pnl += pnl
        self.n_trades += 1

        # Record trade
        self.trades.append(self.position)
        self.position = None

    def _update_equity(self, candle: pd.Series) -> None:
        """Update equity curve."""
        current_price = float(candle.get('close', candle.get('open', 0)))

        if self.position is None:
            equity = self.balance
        else:
            # Calculate unrealized P&L
            if self.position.direction == 1:
                unrealized = self.position.size * (
                    current_price - self.position.entry_price
                ) / self.position.entry_price
            else:
                unrealized = self.position.size * (
                    self.position.entry_price - current_price
                ) / self.position.entry_price
            equity = self.balance + unrealized

        self.equity_curve.append(equity)

    def _get_label(
        self,
        current_candle: pd.Series,
        next_candle: Optional[pd.Series]
    ) -> int:
        """
        Get label for learning (based on actual outcome).

        Returns:
            1 if price went up, 0 otherwise
        """
        if next_candle is None:
            return 0

        current_close = float(current_candle.get('close', 0))
        next_close = float(next_candle.get('close', 0))

        return 1 if next_close > current_close else 0

    def _learn_from_sample(self, features: Dict[str, float], label: int) -> None:
        """Update online learner with new sample."""
        try:
            self.online_learner.learn_one(features, label)
        except Exception as e:
            logger.debug(f"Online learning update failed: {e}")

    def _check_drift(self, candle: pd.Series) -> None:
        """Check for market drift."""
        try:
            # Get recent trade PnL if available
            recent_pnl = 0.0
            if self.trades:
                recent_pnl = self.trades[-1].pnl / self.config.initial_balance

            result = self.drift_detector.update(pnl=recent_pnl)

            if result.get('drift_detected', False):
                self.drift_events.append({
                    'step': self.n_samples_processed,
                    'timestamp': self._get_timestamp(candle).isoformat(),
                    'metrics': result.get('drifting_metrics', [])
                })
        except Exception as e:
            logger.debug(f"Drift detection failed: {e}")

    def _get_timestamp(self, candle: pd.Series) -> datetime:
        """Get timestamp from candle."""
        if 'timestamp' in candle.index:
            ts = candle['timestamp']
        elif 'open_time' in candle.index:
            ts = candle['open_time']
        else:
            return datetime.now(timezone.utc)

        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        return pd.Timestamp(ts).to_pydatetime()

    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate final performance metrics."""
        if not self.trades:
            return {
                'total_return_pct': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'profit_factor': 0.0
            }

        # Basic metrics
        total_return_pct = (
            (self.balance - self.config.initial_balance) /
            self.config.initial_balance * 100
        )

        # Win rate
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]
        win_rate = len(wins) / len(self.trades) * 100 if self.trades else 0

        # Profit factor
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Equity-based metrics
        equity = np.array(self.equity_curve)
        returns = np.diff(equity) / equity[:-1]

        # Sharpe ratio (annualized, assuming minute data)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(525600)  # minutes per year
        else:
            sharpe = 0.0

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdown = abs(np.min(drawdown)) * 100

        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(525600)
        else:
            sortino = 0.0

        return {
            'total_return_pct': float(total_return_pct),
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'total_trades': len(self.trades),
            'profit_factor': float(profit_factor) if profit_factor != float('inf') else 999.99,
            'total_fees': float(sum(t.fees for t in self.trades)),
            'avg_trade_pnl': float(np.mean([t.pnl for t in self.trades])),
            'final_balance': float(self.balance),
            'winning_trades': len(wins),
            'losing_trades': len(losses)
        }

    def _get_ol_stats(self) -> Dict[str, Any]:
        """Get online learner statistics."""
        if self.online_learner is None:
            return {}

        try:
            return self.online_learner.get_stats()
        except Exception:
            return {}

    def reset(self) -> None:
        """Reset simulator state."""
        self.balance = self.config.initial_balance
        self.position = None
        self.trades = []
        self.equity_curve = [self.config.initial_balance]
        self.timestamps = []
        self.drift_events = []
        self.n_samples_processed = 0
        self.n_trades = 0
        self.total_pnl = 0.0
