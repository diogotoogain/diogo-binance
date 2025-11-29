"""
Binance REST Client - Cliente async para API REST da Binance

Features:
- Suporte a Demo Mode (header X-MBX-DEMO)
- Assinatura HMAC SHA256
- Métodos para account, market data e orders
- Rate limiting automático
"""

import asyncio
import hashlib
import hmac
import os
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import aiohttp
from dotenv import load_dotenv


class BinanceClientError(Exception):
    """Erro do cliente Binance."""
    pass


class BinanceAPIError(BinanceClientError):
    """Erro da API Binance."""
    def __init__(self, status_code: int, message: str, code: int = 0):
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(f"[{status_code}] Code {code}: {message}")


class BinanceClient:
    """
    Cliente REST async para Binance Futures.
    
    Suporta modo demo com header X-MBX-DEMO e assinatura HMAC.
    
    Attributes:
        api_key: Chave da API
        secret_key: Chave secreta
        use_demo: Se deve usar modo demo
        base_url: URL base da API
    """
    
    # URLs da API
    FUTURES_URL = "https://fapi.binance.com"
    TESTNET_URL = "https://testnet.binancefuture.com"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        use_demo: bool = True,
        testnet: bool = False
    ):
        """
        Inicializa o cliente Binance.
        
        Args:
            api_key: Chave da API (ou de env BINANCE_API_KEY)
            secret_key: Chave secreta (ou de env BINANCE_SECRET_KEY)
            use_demo: Usar modo demo com header X-MBX-DEMO
            testnet: Usar URL do testnet
        """
        load_dotenv()
        
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.secret_key = secret_key or os.getenv('BINANCE_SECRET_KEY')
        self.use_demo = use_demo
        
        if testnet:
            self.base_url = self.TESTNET_URL
        else:
            self.base_url = self.FUTURES_URL
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_delay = 0.1  # 100ms entre requests
        self._last_request_time = 0
    
    @property
    def _headers(self) -> Dict[str, str]:
        """Retorna headers para requisições."""
        headers = {
            'X-MBX-APIKEY': self.api_key or '',
            'Content-Type': 'application/json'
        }
        
        # Header para modo demo
        if self.use_demo:
            headers['X-MBX-DEMO'] = 'true'
        
        return headers
    
    def _sign(self, params: Dict[str, Any]) -> str:
        """
        Gera assinatura HMAC SHA256.
        
        Args:
            params: Parâmetros da requisição
            
        Returns:
            Assinatura em hexadecimal
        """
        if not self.secret_key:
            raise BinanceClientError("Secret key não configurada")
        
        query_string = urlencode(params)
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Obtém ou cria sessão HTTP."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(headers=self._headers)
        return self._session
    
    async def _rate_limit(self) -> None:
        """Aplica rate limiting."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._rate_limit_delay:
            await asyncio.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False
    ) -> Any:
        """
        Faz requisição à API.
        
        Args:
            method: Método HTTP (GET, POST, DELETE)
            endpoint: Endpoint da API
            params: Parâmetros da requisição
            signed: Se a requisição precisa de assinatura
            
        Returns:
            Resposta JSON da API
        """
        await self._rate_limit()
        
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        
        params = params or {}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._sign(params)
        
        try:
            if method == 'GET':
                async with session.get(url, params=params) as response:
                    return await self._handle_response(response)
            elif method == 'POST':
                async with session.post(url, params=params) as response:
                    return await self._handle_response(response)
            elif method == 'DELETE':
                async with session.delete(url, params=params) as response:
                    return await self._handle_response(response)
            else:
                raise BinanceClientError(f"Método HTTP inválido: {method}")
                
        except aiohttp.ClientError as e:
            raise BinanceClientError(f"Erro de conexão: {e}")
    
    async def _handle_response(self, response: aiohttp.ClientResponse) -> Any:
        """
        Processa resposta da API.
        
        Args:
            response: Resposta HTTP
            
        Returns:
            Dados JSON ou raise BinanceAPIError
        """
        try:
            data = await response.json()
        except Exception:
            text = await response.text()
            if response.status != 200:
                raise BinanceAPIError(response.status, text)
            return text
        
        if response.status != 200:
            code = data.get('code', 0)
            msg = data.get('msg', str(data))
            raise BinanceAPIError(response.status, msg, code)
        
        return data
    
    async def close(self) -> None:
        """Fecha sessão HTTP."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    # ═══════════════════════════════════════════════════════════════════════
    # ACCOUNT ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════════
    
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Obtém informações da conta.
        
        Returns:
            Dados da conta incluindo balanços e posições
        """
        return await self._request('GET', '/fapi/v2/account', signed=True)
    
    async def get_balance(self, asset: str = "USDT") -> Dict[str, Any]:
        """
        Obtém balanço de um ativo específico.
        
        Args:
            asset: Símbolo do ativo (ex: USDT, BTC)
            
        Returns:
            Balanço do ativo
        """
        account = await self.get_account_info()
        for balance in account.get('assets', []):
            if balance['asset'] == asset:
                return balance
        return {'asset': asset, 'walletBalance': '0', 'availableBalance': '0'}
    
    async def get_positions(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Obtém posições abertas.
        
        Args:
            symbol: Símbolo específico ou None para todos
            
        Returns:
            Lista de posições
        """
        account = await self.get_account_info()
        positions = account.get('positions', [])
        
        if symbol:
            positions = [p for p in positions if p['symbol'] == symbol]
        
        # Filtra apenas posições com quantidade != 0
        return [p for p in positions if float(p.get('positionAmt', 0)) != 0]
    
    # ═══════════════════════════════════════════════════════════════════════
    # MARKET DATA ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════════
    
    async def get_ticker_price(self, symbol: str) -> Dict[str, Any]:
        """
        Obtém preço atual do ticker.
        
        Args:
            symbol: Símbolo do par (ex: BTCUSDT)
            
        Returns:
            Dados do ticker com preço
        """
        params = {'symbol': symbol}
        return await self._request('GET', '/fapi/v1/ticker/price', params)
    
    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Obtém orderbook.
        
        Args:
            symbol: Símbolo do par
            limit: Número de níveis (5, 10, 20, 50, 100, 500, 1000)
            
        Returns:
            Orderbook com bids e asks
        """
        params = {'symbol': symbol, 'limit': limit}
        return await self._request('GET', '/fapi/v1/depth', params)
    
    async def get_klines(
        self,
        symbol: str,
        interval: str = "1m",
        limit: int = 500,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[List]:
        """
        Obtém candlesticks/klines.
        
        Args:
            symbol: Símbolo do par
            interval: Intervalo (1m, 5m, 15m, 1h, etc)
            limit: Número de candles (max 1500)
            start_time: Timestamp início (ms)
            end_time: Timestamp fim (ms)
            
        Returns:
            Lista de candles [open_time, open, high, low, close, volume, ...]
        """
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
        
        return await self._request('GET', '/fapi/v1/klines', params)
    
    async def get_recent_trades(
        self,
        symbol: str,
        limit: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Obtém trades recentes.
        
        Args:
            symbol: Símbolo do par
            limit: Número de trades (max 1000)
            
        Returns:
            Lista de trades recentes
        """
        params = {'symbol': symbol, 'limit': limit}
        return await self._request('GET', '/fapi/v1/trades', params)
    
    async def get_exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtém informações da exchange.
        
        Args:
            symbol: Símbolo específico ou None para todos
            
        Returns:
            Informações da exchange
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        return await self._request('GET', '/fapi/v1/exchangeInfo', params)
    
    # ═══════════════════════════════════════════════════════════════════════
    # ORDER ENDPOINTS
    # ═══════════════════════════════════════════════════════════════════════
    
    async def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        close_position: bool = False,
        position_side: str = "BOTH"
    ) -> Dict[str, Any]:
        """
        Cria uma ordem.
        
        Args:
            symbol: Símbolo do par
            side: BUY ou SELL
            order_type: LIMIT, MARKET, STOP, STOP_MARKET, etc
            quantity: Quantidade
            price: Preço (para LIMIT)
            stop_price: Preço de stop
            time_in_force: GTC, IOC, FOK
            reduce_only: Apenas reduzir posição
            close_position: Fechar posição
            position_side: BOTH, LONG, SHORT (hedge mode)
            
        Returns:
            Dados da ordem criada
        """
        params = {
            'symbol': symbol,
            'side': side,
            'type': order_type,
            'positionSide': position_side
        }
        
        if quantity:
            params['quantity'] = quantity
        
        if price:
            params['price'] = price
            
        if stop_price:
            params['stopPrice'] = stop_price
        
        if order_type in ('LIMIT', 'STOP', 'TAKE_PROFIT'):
            params['timeInForce'] = time_in_force
        
        if reduce_only:
            params['reduceOnly'] = 'true'
            
        if close_position:
            params['closePosition'] = 'true'
        
        return await self._request('POST', '/fapi/v1/order', params, signed=True)
    
    async def cancel_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Cancela uma ordem.
        
        Args:
            symbol: Símbolo do par
            order_id: ID da ordem
            client_order_id: ID do cliente
            
        Returns:
            Dados da ordem cancelada
        """
        params = {'symbol': symbol}
        
        if order_id:
            params['orderId'] = order_id
        elif client_order_id:
            params['origClientOrderId'] = client_order_id
        else:
            raise BinanceClientError("Precisa de order_id ou client_order_id")
        
        return await self._request('DELETE', '/fapi/v1/order', params, signed=True)
    
    async def cancel_all_orders(self, symbol: str) -> Dict[str, Any]:
        """
        Cancela todas as ordens abertas de um símbolo.
        
        Args:
            symbol: Símbolo do par
            
        Returns:
            Resultado da operação
        """
        params = {'symbol': symbol}
        return await self._request('DELETE', '/fapi/v1/allOpenOrders', params, signed=True)
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Obtém ordens abertas.
        
        Args:
            symbol: Símbolo específico ou None para todos
            
        Returns:
            Lista de ordens abertas
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        return await self._request('GET', '/fapi/v1/openOrders', params, signed=True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # LEVERAGE & MARGIN
    # ═══════════════════════════════════════════════════════════════════════
    
    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """
        Define alavancagem para um símbolo.
        
        Args:
            symbol: Símbolo do par
            leverage: Alavancagem (1-125)
            
        Returns:
            Resultado da operação
        """
        params = {
            'symbol': symbol,
            'leverage': leverage
        }
        return await self._request('POST', '/fapi/v1/leverage', params, signed=True)
    
    async def set_margin_type(self, symbol: str, margin_type: str) -> Dict[str, Any]:
        """
        Define tipo de margem.
        
        Args:
            symbol: Símbolo do par
            margin_type: ISOLATED ou CROSSED
            
        Returns:
            Resultado da operação
        """
        params = {
            'symbol': symbol,
            'marginType': margin_type
        }
        return await self._request('POST', '/fapi/v1/marginType', params, signed=True)
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT MANAGER
    # ═══════════════════════════════════════════════════════════════════════
    
    async def __aenter__(self) -> 'BinanceClient':
        """Entra no context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Sai do context manager."""
        await self.close()
