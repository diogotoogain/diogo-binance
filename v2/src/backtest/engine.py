"""
Backtest engine for strategy evaluation.

Provides event-driven backtesting with realistic simulation.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, Union

import pandas as pd

from .metrics import PerformanceMetrics
from .simulator import Position, SimulationState, Trade, TradeSimulator

logger = logging.getLogger(__name__)


class Strategy(Protocol):
    """Protocol defining strategy interface."""

    def generate_signal(
        self, features: Dict[str, Any], context: Dict[str, Any]
    ) -> Optional[Any]:
        """Generate trading signal from features."""
        ...


class Signal:
    """Represents a trading signal."""

    def __init__(
        self,
        direction: int,
        confidence: float = 1.0,
        size_hint: float = 1.0,
        sl_mult: float = 1.0,
        tp_mult: float = 2.0,
    ):
        """
        Initialize signal.

        Args:
            direction: Trade direction (1 long, -1 short, 0 neutral)
            confidence: Signal confidence (0-1)
            size_hint: Suggested position size multiplier
            sl_mult: Stop loss ATR multiplier
            tp_mult: Take profit ATR multiplier
        """
        self.direction = direction
        self.confidence = confidence
        self.size_hint = size_hint
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult


class BacktestEngine:
    """
    Event-driven backtesting engine.

    All parameters come from config - no hardcoded values.

    Config parameters (backtest):
    - initial_capital: Starting capital
    - slippage.*: Slippage configuration
    - fees.*: Fee configuration
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize backtest engine.

        Args:
            config: Full configuration dictionary
        """
        self.config = config.get("backtest", {})
        self.full_config = config

        # Core settings from config
        self.initial_capital = self.config.get("initial_capital", 10000)

        # Components
        self.simulator = TradeSimulator(config)
        self.metrics = PerformanceMetrics()

        # State
        self.state: Optional[SimulationState] = None

    def run(
        self,
        data: pd.DataFrame,
        strategy: Union[Strategy, Callable],
        features: Optional[pd.DataFrame] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Execute backtest.

        Args:
            data: OHLCV market data
            strategy: Strategy instance or callable
            features: Pre-calculated features DataFrame
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary with results and metrics
        """
        logger.info(f"Starting backtest with {len(data)} bars")

        # Initialize state
        self.state = SimulationState(self.initial_capital)

        # Use data index if features not provided
        if features is None:
            features = data.copy()

        # Ensure same length
        min_len = min(len(data), len(features))
        data = data.iloc[:min_len].reset_index(drop=True)
        features = features.iloc[:min_len].reset_index(drop=True)

        # Run simulation
        for i in range(len(data)):
            self._process_bar(i, data, features, strategy)

            if progress_callback:
                progress_callback(i + 1, len(data))

        # Close any open position at end
        if self.state.position is not None:
            final_price = self._get_price(data.iloc[-1])
            final_time = self._get_timestamp(data, len(data) - 1)
            trade = self.simulator.execute_exit(
                self.state.position,
                final_price,
                final_time,
                len(data) - 1,
                "end_of_data",
            )
            self.state.close_position(trade)

        # Calculate metrics
        equity_series = self.state.get_equity_series()
        trades_df = self.state.get_trades_dataframe()

        results = self.metrics.calculate_all(
            equity_curve=equity_series,
            trades=trades_df,
            initial_capital=self.initial_capital,
        )

        results["equity_curve"] = equity_series
        results["trades"] = trades_df

        logger.info(
            f"Backtest complete: {results['total_trades']} trades, "
            f"{results['total_return_pct']:.2f}% return, "
            f"Sharpe: {results['sharpe_ratio']:.2f}"
        )

        return results

    def _process_bar(
        self,
        index: int,
        data: pd.DataFrame,
        features: pd.DataFrame,
        strategy: Union[Strategy, Callable],
    ) -> None:
        """
        Process a single bar.

        Args:
            index: Bar index
            data: Market data
            features: Feature data
            strategy: Strategy
        """
        current_bar = data.iloc[index]
        current_features = features.iloc[index].to_dict()
        current_price = self._get_price(current_bar)
        current_time = self._get_timestamp(data, index)

        # Check exit conditions for existing position
        if self.state.position is not None:
            should_exit, exit_reason = self._check_exit(
                self.state.position, current_bar, current_price
            )

            if should_exit:
                trade = self.simulator.execute_exit(
                    self.state.position,
                    current_price,
                    current_time,
                    index,
                    exit_reason,
                )
                self.state.close_position(trade)

        # Generate signal for potential new position
        if self.state.position is None:
            context = {
                "balance": self.state.balance,
                "equity": self.state.equity_curve[-1],
                "n_trades": len(self.state.trades),
            }

            signal = self._get_signal(strategy, current_features, context)

            if signal is not None and signal.direction != 0:
                # Calculate position size
                position_size = self._calculate_position_size(
                    signal, current_price, current_bar
                )

                if position_size > 0:
                    # Get ATR for stops
                    atr = current_features.get(
                        "atr", current_features.get("ATR", current_price * 0.02)
                    )

                    position = self.simulator.execute_entry(
                        price=current_price,
                        direction=signal.direction,
                        size=position_size,
                        timestamp=current_time,
                        step=index,
                        atr=atr,
                        sl_mult=signal.sl_mult,
                        tp_mult=signal.tp_mult,
                    )
                    self.state.open_position(position)

        # Update equity
        self.state.update_equity(current_price)

    def _get_signal(
        self,
        strategy: Union[Strategy, Callable],
        features: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Optional[Signal]:
        """
        Get signal from strategy.

        Args:
            strategy: Strategy instance or callable
            features: Current features
            context: Trading context

        Returns:
            Signal or None
        """
        try:
            if hasattr(strategy, "generate_signal"):
                result = strategy.generate_signal(features, context)
            elif callable(strategy):
                result = strategy(features, context)
            else:
                return None

            # Convert result to Signal
            if result is None:
                return None
            elif isinstance(result, Signal):
                return result
            elif isinstance(result, dict):
                return Signal(
                    direction=result.get("direction", 0),
                    confidence=result.get("confidence", 1.0),
                    size_hint=result.get("size_hint", 1.0),
                    sl_mult=result.get("sl_mult", 1.0),
                    tp_mult=result.get("tp_mult", 2.0),
                )
            elif isinstance(result, (int, float)):
                return Signal(direction=int(result))
            else:
                return None

        except Exception as e:
            logger.warning(f"Error getting signal: {e}")
            return None

    def _check_exit(
        self,
        position: Position,
        bar: pd.Series,
        current_price: float,
    ) -> tuple:
        """
        Check if position should be exited.

        Args:
            position: Current position
            bar: Current bar data
            current_price: Current price

        Returns:
            Tuple of (should_exit, reason)
        """
        # Check high/low for stop/target during the bar
        high = bar.get("high", bar.get("High", current_price))
        low = bar.get("low", bar.get("Low", current_price))

        # Check stop loss
        if position.stop_loss > 0:
            if position.direction == 1 and low <= position.stop_loss:
                return True, "stop_loss"
            if position.direction == -1 and high >= position.stop_loss:
                return True, "stop_loss"

        # Check take profit
        if position.take_profit > 0:
            if position.direction == 1 and high >= position.take_profit:
                return True, "take_profit"
            if position.direction == -1 and low <= position.take_profit:
                return True, "take_profit"

        return False, ""

    def _calculate_position_size(
        self,
        signal: Signal,
        price: float,
        bar: pd.Series,
    ) -> float:
        """
        Calculate position size.

        Args:
            signal: Trading signal
            price: Entry price
            bar: Current bar data

        Returns:
            Position size (notional value)
        """
        # Get risk parameters from config
        risk_config = self.full_config.get("risk", {})
        max_risk_pct = risk_config.get("max_risk_per_trade_pct", 0.5) / 100

        # Base position size
        available_capital = self.state.balance
        base_size = available_capital * max_risk_pct

        # Adjust by signal confidence and size hint
        adjusted_size = base_size * signal.confidence * signal.size_hint

        return adjusted_size

    def _get_price(self, bar: pd.Series) -> float:
        """Get price from bar (supports different column names)."""
        return float(bar.get("close", bar.get("Close", 0)))

    def _get_timestamp(self, data: pd.DataFrame, index: int) -> pd.Timestamp:
        """Get timestamp for bar."""
        if isinstance(data.index, pd.DatetimeIndex):
            return data.index[index]
        elif "timestamp" in data.columns:
            return pd.Timestamp(data.iloc[index]["timestamp"])
        elif "time" in data.columns:
            return pd.Timestamp(data.iloc[index]["time"])
        else:
            return pd.Timestamp.now()

    def run_multiple(
        self,
        data: pd.DataFrame,
        strategies: Dict[str, Union[Strategy, Callable]],
        features: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run backtest for multiple strategies.

        Args:
            data: Market data
            strategies: Dictionary of strategy name to strategy
            features: Optional features

        Returns:
            Dictionary of results per strategy
        """
        results = {}

        for name, strategy in strategies.items():
            logger.info(f"Running backtest for strategy: {name}")
            results[name] = self.run(data, strategy, features)

        return results

    def compare_results(
        self, results: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Compare results from multiple backtests.

        Args:
            results: Dictionary of strategy results

        Returns:
            Comparison DataFrame
        """
        comparison = []

        metrics_to_compare = [
            "total_return_pct",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "total_trades",
        ]

        for name, result in results.items():
            row = {"strategy": name}
            for metric in metrics_to_compare:
                row[metric] = result.get(metric, 0)
            comparison.append(row)

        return pd.DataFrame(comparison)

    def get_state(self) -> Optional[SimulationState]:
        """Get current simulation state."""
        return self.state

    def reset(self) -> None:
        """Reset engine state."""
        self.state = None
