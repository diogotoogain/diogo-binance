"""
WebSocket Handler - Handler robusto para streams da Binance

Features:
- Buffer GRANDE: 10000 mensagens (evita overflow!)
- Reconexão automática
- Health check
- Callbacks async
- Múltiplos streams (trades, klines, depth)
"""

import asyncio
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException


class StreamType(Enum):
    """Tipos de stream disponíveis."""
    TRADE = "trade"
    KLINE = "kline"
    DEPTH = "depth"
    TICKER = "ticker"
    BOOK_TICKER = "bookTicker"
    AGG_TRADE = "aggTrade"


@dataclass
class WebSocketMessage:
    """Mensagem do WebSocket."""
    stream_type: StreamType
    symbol: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class WebSocketStats:
    """Estatísticas do WebSocket."""
    messages_received: int = 0
    messages_processed: int = 0
    messages_dropped: int = 0
    reconnections: int = 0
    last_message_time: Optional[datetime] = None
    connected_since: Optional[datetime] = None


class WebSocketHealthMonitor:
    """
    Monitor de saúde do WebSocket.
    
    Verifica se o WebSocket está saudável com base em:
    - Tempo desde última mensagem
    - Taxa de mensagens
    - Conexão ativa
    """
    
    def __init__(
        self,
        max_silent_seconds: float = 30.0,
        min_messages_per_minute: int = 1
    ):
        """
        Inicializa o monitor.
        
        Args:
            max_silent_seconds: Máximo de segundos sem mensagem
            min_messages_per_minute: Mínimo de mensagens por minuto
        """
        self.max_silent_seconds = max_silent_seconds
        self.min_messages_per_minute = min_messages_per_minute
        self._message_times: deque = deque(maxlen=1000)
        self._last_message_time: Optional[float] = None
        self._is_connected = False
    
    def record_message(self) -> None:
        """Registra recebimento de mensagem."""
        now = time.time()
        self._last_message_time = now
        self._message_times.append(now)
    
    def set_connected(self, connected: bool) -> None:
        """Define estado de conexão."""
        self._is_connected = connected
    
    def is_healthy(self) -> bool:
        """
        Verifica se o WebSocket está saudável.
        
        Returns:
            True se saudável, False caso contrário
        """
        if not self._is_connected:
            return False
        
        if self._last_message_time is None:
            return True  # Ainda não recebeu mensagens
        
        # Verifica tempo desde última mensagem
        elapsed = time.time() - self._last_message_time
        if elapsed > self.max_silent_seconds:
            return False
        
        return True
    
    def get_messages_per_minute(self) -> float:
        """
        Calcula taxa de mensagens por minuto.
        
        Returns:
            Mensagens por minuto
        """
        if len(self._message_times) < 2:
            return 0.0
        
        now = time.time()
        one_minute_ago = now - 60
        
        recent_messages = sum(1 for t in self._message_times if t > one_minute_ago)
        return recent_messages
    
    def get_last_message_age(self) -> Optional[float]:
        """
        Retorna idade da última mensagem em segundos.
        
        Returns:
            Segundos desde última mensagem ou None
        """
        if self._last_message_time is None:
            return None
        return time.time() - self._last_message_time


class WebSocketHandler:
    """
    Handler robusto para WebSocket da Binance.
    
    Features:
    - Buffer grande (10000 mensagens) para evitar overflow
    - Reconexão automática com backoff
    - Health monitoring
    - Callbacks assíncronos
    - Múltiplos streams
    
    Attributes:
        buffer_size: Tamanho do buffer (padrão 10000)
        reconnect_delay: Delay inicial de reconexão
        max_reconnect_attempts: Máximo de tentativas
    """
    
    # URLs do WebSocket
    FUTURES_WS_URL = "wss://fstream.binance.com/ws"
    TESTNET_WS_URL = "wss://stream.binancefuture.com/ws"
    
    # Buffer GRANDE para evitar overflow - 10000 mensagens!
    DEFAULT_BUFFER_SIZE = 10000
    
    def __init__(
        self,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        reconnect_delay: float = 1.0,
        max_reconnect_attempts: int = 10,
        ping_interval: float = 30.0,
        ping_timeout: float = 10.0,
        testnet: bool = False
    ):
        """
        Inicializa o WebSocket handler.
        
        Args:
            buffer_size: Tamanho do buffer de mensagens (MÍNIMO 10000!)
            reconnect_delay: Delay inicial de reconexão em segundos
            max_reconnect_attempts: Máximo de tentativas de reconexão
            ping_interval: Intervalo de ping em segundos
            ping_timeout: Timeout do ping em segundos
            testnet: Usar URL do testnet
        """
        # Garante buffer mínimo de 10000
        if buffer_size < 10000:
            buffer_size = 10000
        
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        
        if testnet:
            self.base_url = self.TESTNET_WS_URL
        else:
            self.base_url = self.FUTURES_WS_URL
        
        # Buffer de mensagens
        self._buffer: deque = deque(maxlen=buffer_size)
        
        # Estado
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._reconnect_count = 0
        
        # Subscriptions
        self._subscribed_streams: Set[str] = set()
        
        # Callbacks
        self._callbacks: Dict[StreamType, List[Callable]] = {
            st: [] for st in StreamType
        }
        self._global_callbacks: List[Callable] = []
        
        # Stats e Health
        self.stats = WebSocketStats()
        self.health_monitor = WebSocketHealthMonitor()
        
        # Tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
    
    def _build_stream_name(
        self,
        symbol: str,
        stream_type: StreamType,
        interval: Optional[str] = None,
        depth_levels: int = 20
    ) -> str:
        """Constrói nome do stream."""
        symbol_lower = symbol.lower()
        
        if stream_type == StreamType.TRADE:
            return f"{symbol_lower}@trade"
        elif stream_type == StreamType.AGG_TRADE:
            return f"{symbol_lower}@aggTrade"
        elif stream_type == StreamType.KLINE:
            interval = interval or "1m"
            return f"{symbol_lower}@kline_{interval}"
        elif stream_type == StreamType.DEPTH:
            return f"{symbol_lower}@depth{depth_levels}@100ms"
        elif stream_type == StreamType.TICKER:
            return f"{symbol_lower}@ticker"
        elif stream_type == StreamType.BOOK_TICKER:
            return f"{symbol_lower}@bookTicker"
        
        raise ValueError(f"Stream type inválido: {stream_type}")
    
    async def connect(self) -> None:
        """Conecta ao WebSocket."""
        if self._running:
            return
        
        self._running = True
        self._reconnect_count = 0
        
        await self._connect()
        
        # Inicia tasks
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._ping_task = asyncio.create_task(self._ping_loop())
        self._health_task = asyncio.create_task(self._health_check_loop())
    
    async def _connect(self) -> None:
        """Estabelece conexão WebSocket."""
        try:
            self._websocket = await websockets.connect(
                self.base_url,
                ping_interval=None,  # Gerenciamos manualmente
                ping_timeout=self.ping_timeout,
                max_size=2**24  # 16MB max message size
            )
            
            self.health_monitor.set_connected(True)
            self.stats.connected_since = datetime.now()
            
            # Re-subscribe aos streams
            if self._subscribed_streams:
                await self._subscribe(list(self._subscribed_streams))
            
        except Exception as e:
            self.health_monitor.set_connected(False)
            raise WebSocketException(f"Erro ao conectar: {e}")
    
    async def disconnect(self) -> None:
        """Desconecta do WebSocket."""
        self._running = False
        
        # Cancela tasks
        for task in [self._receive_task, self._ping_task, self._health_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Fecha WebSocket
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        
        self.health_monitor.set_connected(False)
    
    async def _reconnect(self) -> None:
        """Tenta reconectar com backoff exponencial."""
        if not self._running:
            return
        
        self.health_monitor.set_connected(False)
        self._reconnect_count += 1
        self.stats.reconnections += 1
        
        if self._reconnect_count > self.max_reconnect_attempts:
            self._running = False
            raise WebSocketException(
                f"Máximo de reconexões ({self.max_reconnect_attempts}) excedido"
            )
        
        # Backoff exponencial
        delay = self.reconnect_delay * (2 ** (self._reconnect_count - 1))
        delay = min(delay, 60)  # Máximo 60 segundos
        
        await asyncio.sleep(delay)
        
        try:
            await self._connect()
            self._reconnect_count = 0  # Reset ao conectar com sucesso
        except Exception:
            await self._reconnect()  # Tenta novamente
    
    async def _receive_loop(self) -> None:
        """Loop de recebimento de mensagens."""
        while self._running:
            try:
                if not self._websocket:
                    await asyncio.sleep(0.1)
                    continue
                
                message = await self._websocket.recv()
                await self._process_message(message)
                
            except ConnectionClosed:
                if self._running:
                    await self._reconnect()
            except Exception:
                if self._running:
                    await asyncio.sleep(0.1)
    
    async def _process_message(self, raw_message: str) -> None:
        """Processa mensagem recebida."""
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError:
            return
        
        # Atualiza stats
        self.stats.messages_received += 1
        self.stats.last_message_time = datetime.now()
        self.health_monitor.record_message()
        
        # Identifica tipo de stream
        stream_type = self._identify_stream_type(data)
        symbol = self._extract_symbol(data)
        
        # Cria mensagem
        msg = WebSocketMessage(
            stream_type=stream_type,
            symbol=symbol,
            data=data
        )
        
        # Adiciona ao buffer
        if len(self._buffer) >= self.buffer_size:
            self.stats.messages_dropped += 1
        self._buffer.append(msg)
        
        # Chama callbacks
        await self._dispatch_callbacks(msg)
        
        self.stats.messages_processed += 1
    
    def _identify_stream_type(self, data: Dict) -> StreamType:
        """Identifica tipo de stream da mensagem."""
        if 'e' in data:
            event_type = data['e']
            if event_type == 'trade':
                return StreamType.TRADE
            elif event_type == 'aggTrade':
                return StreamType.AGG_TRADE
            elif event_type == 'kline':
                return StreamType.KLINE
            elif event_type == 'depthUpdate':
                return StreamType.DEPTH
            elif event_type == '24hrTicker':
                return StreamType.TICKER
            elif event_type == 'bookTicker':
                return StreamType.BOOK_TICKER
        
        # Fallback
        if 'bids' in data or 'asks' in data:
            return StreamType.DEPTH
        
        return StreamType.TRADE
    
    def _extract_symbol(self, data: Dict) -> str:
        """Extrai símbolo da mensagem."""
        return data.get('s', data.get('symbol', 'UNKNOWN'))
    
    async def _dispatch_callbacks(self, msg: WebSocketMessage) -> None:
        """Despacha callbacks para uma mensagem."""
        # Callbacks específicos do tipo
        for callback in self._callbacks.get(msg.stream_type, []):
            try:
                result = callback(msg)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass  # Não deixa callback quebrar o handler
        
        # Callbacks globais
        for callback in self._global_callbacks:
            try:
                result = callback(msg)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass
    
    async def _ping_loop(self) -> None:
        """Loop de ping para manter conexão."""
        while self._running:
            await asyncio.sleep(self.ping_interval)
            
            if self._websocket and not self._websocket.closed:
                try:
                    await self._websocket.ping()
                except Exception:
                    pass
    
    async def _health_check_loop(self) -> None:
        """Loop de verificação de saúde."""
        while self._running:
            await asyncio.sleep(60)  # Verifica a cada minuto
            
            if not self.health_monitor.is_healthy():
                # Tenta reconectar se não estiver saudável
                await self._reconnect()
    
    async def _subscribe(self, streams: List[str]) -> None:
        """Envia mensagem de subscription."""
        if not self._websocket:
            return
        
        subscribe_msg = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": int(time.time() * 1000)
        }
        
        await self._websocket.send(json.dumps(subscribe_msg))
    
    async def _unsubscribe(self, streams: List[str]) -> None:
        """Envia mensagem de unsubscription."""
        if not self._websocket:
            return
        
        unsubscribe_msg = {
            "method": "UNSUBSCRIBE",
            "params": streams,
            "id": int(time.time() * 1000)
        }
        
        await self._websocket.send(json.dumps(unsubscribe_msg))
    
    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC API - SUBSCRIBE
    # ═══════════════════════════════════════════════════════════════════════
    
    async def subscribe_trades(
        self,
        symbol: str,
        callback: Optional[Callable] = None
    ) -> None:
        """
        Subscreve ao stream de trades.
        
        Args:
            symbol: Símbolo do par (ex: BTCUSDT)
            callback: Função callback para mensagens
        """
        stream = self._build_stream_name(symbol, StreamType.TRADE)
        self._subscribed_streams.add(stream)
        
        if callback:
            self._callbacks[StreamType.TRADE].append(callback)
        
        if self._websocket:
            await self._subscribe([stream])
    
    async def subscribe_klines(
        self,
        symbol: str,
        interval: str = "1m",
        callback: Optional[Callable] = None
    ) -> None:
        """
        Subscreve ao stream de klines.
        
        Args:
            symbol: Símbolo do par
            interval: Intervalo (1m, 5m, 15m, 1h, etc)
            callback: Função callback para mensagens
        """
        stream = self._build_stream_name(symbol, StreamType.KLINE, interval=interval)
        self._subscribed_streams.add(stream)
        
        if callback:
            self._callbacks[StreamType.KLINE].append(callback)
        
        if self._websocket:
            await self._subscribe([stream])
    
    async def subscribe_depth(
        self,
        symbol: str,
        levels: int = 20,
        callback: Optional[Callable] = None
    ) -> None:
        """
        Subscreve ao stream de orderbook depth.
        
        Args:
            symbol: Símbolo do par
            levels: Níveis de profundidade (5, 10, 20)
            callback: Função callback para mensagens
        """
        stream = self._build_stream_name(symbol, StreamType.DEPTH, depth_levels=levels)
        self._subscribed_streams.add(stream)
        
        if callback:
            self._callbacks[StreamType.DEPTH].append(callback)
        
        if self._websocket:
            await self._subscribe([stream])
    
    async def subscribe_ticker(
        self,
        symbol: str,
        callback: Optional[Callable] = None
    ) -> None:
        """
        Subscreve ao stream de ticker.
        
        Args:
            symbol: Símbolo do par
            callback: Função callback para mensagens
        """
        stream = self._build_stream_name(symbol, StreamType.TICKER)
        self._subscribed_streams.add(stream)
        
        if callback:
            self._callbacks[StreamType.TICKER].append(callback)
        
        if self._websocket:
            await self._subscribe([stream])
    
    async def subscribe_book_ticker(
        self,
        symbol: str,
        callback: Optional[Callable] = None
    ) -> None:
        """
        Subscreve ao stream de book ticker (best bid/ask).
        
        Args:
            symbol: Símbolo do par
            callback: Função callback para mensagens
        """
        stream = self._build_stream_name(symbol, StreamType.BOOK_TICKER)
        self._subscribed_streams.add(stream)
        
        if callback:
            self._callbacks[StreamType.BOOK_TICKER].append(callback)
        
        if self._websocket:
            await self._subscribe([stream])
    
    def add_callback(
        self,
        callback: Callable[[WebSocketMessage], Any],
        stream_type: Optional[StreamType] = None
    ) -> None:
        """
        Adiciona callback para mensagens.
        
        Args:
            callback: Função callback
            stream_type: Tipo de stream específico ou None para todos
        """
        if stream_type:
            self._callbacks[stream_type].append(callback)
        else:
            self._global_callbacks.append(callback)
    
    def remove_callback(
        self,
        callback: Callable,
        stream_type: Optional[StreamType] = None
    ) -> None:
        """
        Remove callback.
        
        Args:
            callback: Função callback
            stream_type: Tipo de stream ou None para global
        """
        if stream_type:
            if callback in self._callbacks[stream_type]:
                self._callbacks[stream_type].remove(callback)
        else:
            if callback in self._global_callbacks:
                self._global_callbacks.remove(callback)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC API - BUFFER
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_messages(self, count: int = 100) -> List[WebSocketMessage]:
        """
        Obtém mensagens do buffer.
        
        Args:
            count: Número de mensagens (mais recentes)
            
        Returns:
            Lista de mensagens
        """
        return list(self._buffer)[-count:]
    
    def get_latest_message(
        self,
        stream_type: Optional[StreamType] = None
    ) -> Optional[WebSocketMessage]:
        """
        Obtém última mensagem.
        
        Args:
            stream_type: Filtrar por tipo de stream
            
        Returns:
            Última mensagem ou None
        """
        if not self._buffer:
            return None
        
        if stream_type:
            for msg in reversed(self._buffer):
                if msg.stream_type == stream_type:
                    return msg
            return None
        
        return self._buffer[-1]
    
    def clear_buffer(self) -> None:
        """Limpa buffer de mensagens."""
        self._buffer.clear()
    
    # ═══════════════════════════════════════════════════════════════════════
    # PUBLIC API - STATUS
    # ═══════════════════════════════════════════════════════════════════════
    
    @property
    def is_connected(self) -> bool:
        """Retorna se está conectado."""
        return self._websocket is not None and not self._websocket.closed
    
    @property
    def is_healthy(self) -> bool:
        """Retorna se está saudável."""
        return self.health_monitor.is_healthy()
    
    def get_stats(self) -> WebSocketStats:
        """Retorna estatísticas."""
        return self.stats
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT MANAGER
    # ═══════════════════════════════════════════════════════════════════════
    
    async def __aenter__(self) -> 'WebSocketHandler':
        """Entra no context manager."""
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Sai do context manager."""
        await self.disconnect()
