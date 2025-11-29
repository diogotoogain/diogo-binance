"""
Backtesting module for V2 Trading Bot.

This module provides honest simulation with look-ahead bias protection.
"""

from .honest_simulator import (
    HonestSimulator,
    SimulationConfig,
    SimulationResult,
    TradeRecord
)

__all__ = [
    'HonestSimulator',
    'SimulationConfig',
    'SimulationResult',
    'TradeRecord'
]
