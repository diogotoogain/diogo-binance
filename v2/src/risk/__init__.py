"""
V2 Risk Management Package.

This package provides comprehensive risk management for the trading bot:
- Kill Switch (CRITICAL - always enabled!)
- Daily/Weekly/Monthly limits
- Position sizing (fixed, Kelly, vol_target)
- Stop Loss calculation
- Take Profit calculation
- Trailing Stop
- Regime-based risk adjustment
- Seasonality-based risk adjustment
"""
from .kill_switch import KillSwitch
from .daily_limits import DailyLimits
from .position_sizer import PositionSizer
from .stop_loss import StopLossCalculator
from .take_profit import TakeProfitCalculator
from .trailing_stop import TrailingStop
from .regime_adjustment import RegimeAdjustment
from .seasonality_adjuster import SeasonalityAdjuster
from .risk_manager import RiskManager

__all__ = [
    'KillSwitch',
    'DailyLimits',
    'PositionSizer',
    'StopLossCalculator',
    'TakeProfitCalculator',
    'TrailingStop',
    'RegimeAdjustment',
    'SeasonalityAdjuster',
    'RiskManager',
]
