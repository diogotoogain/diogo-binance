"""
Módulo de conectores para Binance.

Inclui cliente REST async e handler WebSocket robusto.
"""

from .binance_client import BinanceClient
from .binance_derivatives import BinanceDerivativesConnector
from .websocket_handler import WebSocketHandler, WebSocketHealthMonitor

__all__ = [
    'BinanceClient',
    'BinanceDerivativesConnector',
    'WebSocketHandler',
    'WebSocketHealthMonitor'
]
