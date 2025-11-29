"""
Módulo de dados do V2 Trading Bot.

Inclui download de dados históricos e construção de barras.
"""

from .local_data_loader import LocalDataLoader, load_local_data

# Optional imports that depend on external dependencies
try:
    from .historical import HistoricalDataDownloader
    from .bar_builder import BarBuilder, create_bar_builder
    HISTORICAL_AVAILABLE = True
except ImportError:
    HISTORICAL_AVAILABLE = False
    HistoricalDataDownloader = None
    BarBuilder = None
    create_bar_builder = None

__all__ = [
    'LocalDataLoader',
    'load_local_data',
    'HistoricalDataDownloader',
    'BarBuilder',
    'create_bar_builder'
]
