"""
V2 Strategies Package

Estratégias multi-modelo para o V2 do trading bot.
Cada estratégia pode ser ON/OFF via config.
ZERO números hardcoded - todos os parâmetros vêm do config.
"""
from v2.src.strategies.base import Strategy, Signal, SignalDirection
from v2.src.strategies.throttling import Throttler
from v2.src.strategies.hft_ofi_scalper import HFTOFIScalper
from v2.src.strategies.momentum_intraday import MomentumIntraday
from v2.src.strategies.mean_reversion_intraday import MeanReversionIntraday
from v2.src.strategies.volatility_breakout import VolatilityBreakout
from v2.src.strategies.ensemble import EnsembleManager

__all__ = [
    'Strategy',
    'Signal',
    'SignalDirection',
    'Throttler',
    'HFTOFIScalper',
    'MomentumIntraday',
    'MeanReversionIntraday',
    'VolatilityBreakout',
    'EnsembleManager',
]
