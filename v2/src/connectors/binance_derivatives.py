"""
Binance Derivatives Data Connector.

Coleta Funding Rate, Open Interest e Long/Short Ratio.
"""
import asyncio
import time
from typing import Any, Dict, List, Optional

import aiohttp


class BinanceDerivativesError(Exception):
    """Erro do conector de derivativos."""
    pass


class BinanceDerivativesConnector:
    """
    Conector para dados de derivativos da Binance Futures.
    
    Endpoints utilizados:
    - GET /fapi/v1/fundingRate - Histórico de funding rate
    - GET /fapi/v1/openInterest - Open Interest atual
    - GET /futures/data/openInterestHist - Histórico de OI
    - GET /futures/data/globalLongShortAccountRatio - Long/Short ratio
    """
    
    BASE_URL = "https://fapi.binance.com"
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Inicializa o conector de derivativos.
        
        Args:
            base_url: URL base da API (opcional, usa BASE_URL padrão)
        """
        self.base_url = base_url or self.BASE_URL
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limit_delay = 0.1  # 100ms entre requests
        self._last_request_time = 0.0
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Obtém ou cria sessão HTTP."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
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
        endpoint: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Faz requisição GET à API.
        
        Args:
            endpoint: Endpoint da API
            params: Parâmetros da requisição
            
        Returns:
            Resposta JSON da API
        """
        await self._rate_limit()
        
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        
        try:
            async with session.get(url, params=params or {}) as response:
                if response.status != 200:
                    text = await response.text()
                    raise BinanceDerivativesError(
                        f"API error {response.status}: {text}"
                    )
                return await response.json()
        except aiohttp.ClientError as e:
            raise BinanceDerivativesError(f"Connection error: {e}")
    
    async def close(self) -> None:
        """Fecha sessão HTTP."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self) -> 'BinanceDerivativesConnector':
        """Entra no context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Sai do context manager."""
        await self.close()
    
    # ═══════════════════════════════════════════════════════════════════════
    # FUNDING RATE
    # ═══════════════════════════════════════════════════════════════════════
    
    async def get_funding_rate_history(
        self, 
        symbol: str = "BTCUSDT",
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtém histórico de funding rate.
        
        Args:
            symbol: Par de trading (ex: BTCUSDT)
            limit: Número de registros (max 1000)
            start_time: Timestamp início em ms (opcional)
            end_time: Timestamp fim em ms (opcional)
            
        Returns:
            Lista de dicts com fundingTime, fundingRate, symbol, markPrice
        """
        params: Dict[str, Any] = {
            'symbol': symbol,
            'limit': min(limit, 1000)
        }
        
        if start_time is not None:
            params['startTime'] = start_time
        if end_time is not None:
            params['endTime'] = end_time
        
        return await self._request('/fapi/v1/fundingRate', params)
    
    async def get_current_funding_rate(self, symbol: str = "BTCUSDT") -> float:
        """
        Obtém funding rate atual (próximo pagamento).
        
        Args:
            symbol: Par de trading
            
        Returns:
            Funding rate como float (ex: 0.0001 = 0.01%)
        """
        # Pega o último registro para obter o funding rate atual
        history = await self.get_funding_rate_history(symbol=symbol, limit=1)
        
        if not history:
            return 0.0
        
        return float(history[0].get('fundingRate', 0))
    
    async def get_premium_index(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """
        Obtém índice premium com informações de funding.
        
        Args:
            symbol: Par de trading
            
        Returns:
            Dict com lastFundingRate, nextFundingTime, etc.
        """
        params = {'symbol': symbol}
        return await self._request('/fapi/v1/premiumIndex', params)
    
    # ═══════════════════════════════════════════════════════════════════════
    # OPEN INTEREST
    # ═══════════════════════════════════════════════════════════════════════
    
    async def get_open_interest(self, symbol: str = "BTCUSDT") -> float:
        """
        Obtém open interest atual em contratos.
        
        Args:
            symbol: Par de trading
            
        Returns:
            Open interest em quantidade de contratos
        """
        params = {'symbol': symbol}
        result = await self._request('/fapi/v1/openInterest', params)
        return float(result.get('openInterest', 0))
    
    async def get_open_interest_history(
        self,
        symbol: str = "BTCUSDT",
        period: str = "5m",
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtém histórico de open interest.
        
        Args:
            symbol: Par de trading
            period: Período de agregação (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
            limit: Número de registros (max 500)
            start_time: Timestamp início em ms (opcional)
            end_time: Timestamp fim em ms (opcional)
            
        Returns:
            Lista de dicts com timestamp, sumOpenInterest, sumOpenInterestValue
        """
        params: Dict[str, Any] = {
            'symbol': symbol,
            'period': period,
            'limit': min(limit, 500)
        }
        
        if start_time is not None:
            params['startTime'] = start_time
        if end_time is not None:
            params['endTime'] = end_time
        
        return await self._request('/futures/data/openInterestHist', params)
    
    # ═══════════════════════════════════════════════════════════════════════
    # LONG/SHORT RATIO
    # ═══════════════════════════════════════════════════════════════════════
    
    async def get_long_short_ratio(
        self,
        symbol: str = "BTCUSDT",
        period: str = "5m",
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtém histórico de long/short ratio (contas de usuários).
        
        Args:
            symbol: Par de trading
            period: Período (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
            limit: Número de registros (max 500)
            start_time: Timestamp início em ms (opcional)
            end_time: Timestamp fim em ms (opcional)
            
        Returns:
            Lista de dicts com timestamp, longShortRatio, longAccount, shortAccount
        """
        params: Dict[str, Any] = {
            'symbol': symbol,
            'period': period,
            'limit': min(limit, 500)
        }
        
        if start_time is not None:
            params['startTime'] = start_time
        if end_time is not None:
            params['endTime'] = end_time
        
        return await self._request(
            '/futures/data/globalLongShortAccountRatio',
            params
        )
    
    async def get_top_trader_long_short_ratio(
        self,
        symbol: str = "BTCUSDT",
        period: str = "5m",
        limit: int = 100,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Obtém long/short ratio dos top traders (por posição).
        
        Args:
            symbol: Par de trading
            period: Período (5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d)
            limit: Número de registros (max 500)
            start_time: Timestamp início em ms (opcional)
            end_time: Timestamp fim em ms (opcional)
            
        Returns:
            Lista de dicts com timestamp, longShortRatio, longAccount, shortAccount
        """
        params: Dict[str, Any] = {
            'symbol': symbol,
            'period': period,
            'limit': min(limit, 500)
        }
        
        if start_time is not None:
            params['startTime'] = start_time
        if end_time is not None:
            params['endTime'] = end_time
        
        return await self._request(
            '/futures/data/topLongShortPositionRatio',
            params
        )
    
    async def get_current_long_short_ratio(
        self,
        symbol: str = "BTCUSDT"
    ) -> float:
        """
        Obtém long/short ratio atual.
        
        Args:
            symbol: Par de trading
            
        Returns:
            Ratio como float (ex: 1.5 = 50% mais longs que shorts)
        """
        history = await self.get_long_short_ratio(
            symbol=symbol,
            period="5m",
            limit=1
        )
        
        if not history:
            return 1.0  # Neutro
        
        return float(history[0].get('longShortRatio', 1.0))
    
    # ═══════════════════════════════════════════════════════════════════════
    # AGGREGATED DATA
    # ═══════════════════════════════════════════════════════════════════════
    
    async def get_all_derivatives_data(
        self,
        symbol: str = "BTCUSDT"
    ) -> Dict[str, Any]:
        """
        Obtém todos os dados de derivativos de uma vez.
        
        Args:
            symbol: Par de trading
            
        Returns:
            Dict com funding_rate, open_interest, long_short_ratio, premium_index
        """
        # Faz todas as requisições em paralelo
        results = await asyncio.gather(
            self.get_current_funding_rate(symbol),
            self.get_open_interest(symbol),
            self.get_current_long_short_ratio(symbol),
            self.get_premium_index(symbol),
            return_exceptions=True
        )
        
        funding_rate = results[0] if not isinstance(results[0], Exception) else 0.0
        open_interest = results[1] if not isinstance(results[1], Exception) else 0.0
        long_short_ratio = results[2] if not isinstance(results[2], Exception) else 1.0
        premium_index = results[3] if not isinstance(results[3], Exception) else {}
        
        return {
            'symbol': symbol,
            'funding_rate': funding_rate,
            'open_interest': open_interest,
            'long_short_ratio': long_short_ratio,
            'premium_index': premium_index,
            'timestamp': int(time.time() * 1000)
        }
