"""
Trade simulator for backtesting.

Simulates trade execution with realistic slippage and fees.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Position:
    """Represents an open trading position."""

    direction: int  # 1 for long, -1 for short
    entry_price: float
    size: float
    entry_time: pd.Timestamp
    stop_loss: float = 0.0
    take_profit: float = 0.0
    entry_step: int = 0


@dataclass
class Trade:
    """Represents a completed trade."""

    entry_price: float
    exit_price: float
    direction: int
    size: float
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    pnl: float
    pnl_pct: float
    exit_reason: str
    fees: float = 0.0
    slippage: float = 0.0
    duration: int = 0


class TradeSimulator:
    """
    Simulates trade execution with realistic market conditions.

    All parameters come from config - no hardcoded values.

    Config parameters (backtest):
    - slippage.type: 'fixed' or 'volume_based'
    - slippage.base_bps: Base slippage in basis points
    - slippage.volume_impact: Volume impact factor
    - fees.taker: Taker fee rate
    - fees.maker: Maker fee rate
    - fees.funding_rate: Funding rate for perpetuals
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize trade simulator.

        Args:
            config: Full configuration dictionary
        """
        self.config = config.get("backtest", {})

        # Slippage configuration
        self.slippage_config = self.config.get("slippage", {})
        self.slippage_type = self.slippage_config.get("type", "volume_based")
        self.base_bps = self.slippage_config.get("base_bps", 1.0)
        self.volume_impact = self.slippage_config.get("volume_impact", 0.5)
        self.spread_factor = self.slippage_config.get("spread_factor", 1.0)

        # Fee configuration
        self.fees_config = self.config.get("fees", {})
        self.taker_fee = self.fees_config.get("taker", 0.0004)
        self.maker_fee = self.fees_config.get("maker", 0.0002)
        self.funding_rate = self.fees_config.get("funding_rate", 0.0001)

    def apply_slippage(
        self,
        price: float,
        side: str,
        volume: Optional[float] = None,
        avg_volume: Optional[float] = None,
    ) -> float:
        """
        Apply slippage to a trade price.

        Args:
            price: Base price
            side: Trade side ('BUY' or 'SELL')
            volume: Trade volume
            avg_volume: Average market volume

        Returns:
            Price with slippage applied
        """
        if self.slippage_type == "fixed":
            slippage_pct = self.base_bps / 10000
        else:
            # Volume-based slippage
            slippage_pct = self.base_bps / 10000

            # Add volume impact if available
            if volume is not None and avg_volume is not None and avg_volume > 0:
                volume_ratio = volume / avg_volume
                volume_slippage = self.volume_impact * volume_ratio / 10000
                slippage_pct += volume_slippage

        # Apply slippage based on side
        if side.upper() == "BUY":
            return price * (1 + slippage_pct)
        else:
            return price * (1 - slippage_pct)

    def calculate_fees(
        self,
        notional: float,
        is_maker: bool = False,
    ) -> float:
        """
        Calculate trading fees.

        Args:
            notional: Notional value of trade
            is_maker: Whether this is a maker order

        Returns:
            Fee amount
        """
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        return notional * fee_rate

    def calculate_funding(
        self,
        position_value: float,
        hours_held: float,
    ) -> float:
        """
        Calculate funding costs for perpetual futures.

        Args:
            position_value: Value of position
            hours_held: Hours position was held

        Returns:
            Funding cost (negative = payment received)
        """
        # Funding is typically applied every 8 hours
        funding_periods = hours_held / 8
        return position_value * self.funding_rate * funding_periods

    def execute_entry(
        self,
        price: float,
        direction: int,
        size: float,
        timestamp: pd.Timestamp,
        step: int = 0,
        atr: float = 0.0,
        sl_mult: float = 1.0,
        tp_mult: float = 2.0,
        volume: Optional[float] = None,
        avg_volume: Optional[float] = None,
    ) -> Position:
        """
        Execute trade entry.

        Args:
            price: Market price
            direction: Trade direction (1 for long, -1 for short)
            size: Position size (notional value)
            timestamp: Entry timestamp
            step: Entry step index
            atr: ATR for stop/target calculation
            sl_mult: Stop loss ATR multiplier
            tp_mult: Take profit ATR multiplier
            volume: Trade volume
            avg_volume: Average volume

        Returns:
            Opened Position
        """
        # Apply slippage
        side = "BUY" if direction == 1 else "SELL"
        entry_price = self.apply_slippage(price, side, volume, avg_volume)

        # Calculate stop loss and take profit
        if direction == 1:  # Long
            stop_loss = entry_price - atr * sl_mult if atr > 0 else 0
            take_profit = entry_price + atr * tp_mult if atr > 0 else 0
        else:  # Short
            stop_loss = entry_price + atr * sl_mult if atr > 0 else 0
            take_profit = entry_price - atr * tp_mult if atr > 0 else 0

        return Position(
            direction=direction,
            entry_price=entry_price,
            size=size,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_step=step,
        )

    def execute_exit(
        self,
        position: Position,
        price: float,
        timestamp: pd.Timestamp,
        step: int = 0,
        reason: str = "signal",
        volume: Optional[float] = None,
        avg_volume: Optional[float] = None,
    ) -> Trade:
        """
        Execute trade exit.

        Args:
            position: Position to close
            price: Market price
            timestamp: Exit timestamp
            step: Exit step index
            reason: Exit reason
            volume: Trade volume
            avg_volume: Average volume

        Returns:
            Completed Trade
        """
        # Apply slippage (opposite side of entry)
        side = "SELL" if position.direction == 1 else "BUY"
        exit_price = self.apply_slippage(price, side, volume, avg_volume)

        # Calculate P&L
        if position.direction == 1:  # Long
            pnl = position.size * (exit_price - position.entry_price) / position.entry_price
        else:  # Short
            pnl = position.size * (position.entry_price - exit_price) / position.entry_price

        pnl_pct = (
            (exit_price / position.entry_price - 1) * position.direction * 100
        )

        # Calculate fees
        entry_notional = position.size
        exit_notional = position.size * (1 + pnl / position.size if position.size > 0 else 1)
        entry_fee = self.calculate_fees(entry_notional)
        exit_fee = self.calculate_fees(exit_notional)
        total_fees = entry_fee + exit_fee

        # Calculate slippage cost
        slippage_entry = abs(position.entry_price - price) / price * entry_notional
        slippage_exit = abs(exit_price - price) / price * exit_notional
        total_slippage = slippage_entry + slippage_exit

        # Adjust P&L for fees
        pnl -= total_fees

        # Calculate duration
        duration = step - position.entry_step

        return Trade(
            entry_price=position.entry_price,
            exit_price=exit_price,
            direction=position.direction,
            size=position.size,
            entry_time=position.entry_time,
            exit_time=timestamp,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason,
            fees=total_fees,
            slippage=total_slippage,
            duration=duration,
        )

    def check_stop_loss(
        self,
        position: Position,
        current_price: float,
    ) -> bool:
        """
        Check if stop loss is hit.

        Args:
            position: Open position
            current_price: Current market price

        Returns:
            True if stop loss is hit
        """
        if position.stop_loss == 0:
            return False

        if position.direction == 1:  # Long
            return current_price <= position.stop_loss
        else:  # Short
            return current_price >= position.stop_loss

    def check_take_profit(
        self,
        position: Position,
        current_price: float,
    ) -> bool:
        """
        Check if take profit is hit.

        Args:
            position: Open position
            current_price: Current market price

        Returns:
            True if take profit is hit
        """
        if position.take_profit == 0:
            return False

        if position.direction == 1:  # Long
            return current_price >= position.take_profit
        else:  # Short
            return current_price <= position.take_profit

    def calculate_unrealized_pnl(
        self,
        position: Position,
        current_price: float,
    ) -> float:
        """
        Calculate unrealized P&L for open position.

        Args:
            position: Open position
            current_price: Current market price

        Returns:
            Unrealized P&L
        """
        if position.direction == 1:  # Long
            return position.size * (current_price - position.entry_price) / position.entry_price
        else:  # Short
            return position.size * (position.entry_price - current_price) / position.entry_price

    def simulate_market_impact(
        self,
        price: float,
        order_size: float,
        market_depth: Dict[str, List[tuple]],
    ) -> float:
        """
        Simulate market impact based on order book depth.

        Args:
            price: Mid price
            order_size: Order size
            market_depth: Order book with 'bids' and 'asks'

        Returns:
            Execution price after market impact
        """
        # Simplified market impact model
        # In reality, this would walk the order book

        if not market_depth or "asks" not in market_depth:
            return price

        # Estimate impact based on size vs available liquidity
        total_liquidity = sum(qty for _, qty in market_depth.get("asks", [])[:5])

        if total_liquidity == 0:
            return price

        impact_ratio = order_size / total_liquidity
        impact_bps = impact_ratio * 10  # Simplified: 10 bps per 100% of liquidity

        return price * (1 + impact_bps / 10000)


class SimulationState:
    """
    Manages simulation state during backtesting.
    """

    def __init__(self, initial_capital: float):
        """
        Initialize simulation state.

        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]

    def update_equity(self, current_price: float) -> float:
        """
        Update and record current equity.

        Args:
            current_price: Current market price

        Returns:
            Current equity
        """
        if self.position is None:
            equity = self.balance
        else:
            unrealized = self.position.size * (
                current_price - self.position.entry_price
            ) / self.position.entry_price * self.position.direction
            equity = self.balance + unrealized

        self.equity_curve.append(equity)
        return equity

    def open_position(self, position: Position) -> None:
        """Record position opening."""
        self.position = position

    def close_position(self, trade: Trade) -> None:
        """Record position closing."""
        self.trades.append(trade)
        self.balance += trade.pnl
        self.position = None

    def get_trades_dataframe(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "direction": t.direction,
                "size": t.size,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "exit_reason": t.exit_reason,
                "fees": t.fees,
                "slippage": t.slippage,
                "duration": t.duration,
            }
            for t in self.trades
        ])

    def get_equity_series(self) -> pd.Series:
        """Get equity curve as Series."""
        return pd.Series(self.equity_curve)
