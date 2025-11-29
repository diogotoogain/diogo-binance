"""
Módulo de dados do V2 Trading Bot.

Inclui download de dados históricos e construção de barras.
"""

from .historical import HistoricalDataDownloader
from .bar_builder import BarBuilder, create_bar_builder

__all__ = [
    'HistoricalDataDownloader',
    'BarBuilder',
    'create_bar_builder'
]
