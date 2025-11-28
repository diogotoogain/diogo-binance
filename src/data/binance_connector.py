import asyncio
import logging
import time
from binance import AsyncClient, BinanceSocketManager
from binance.exceptions import BinanceWebsocketQueueOverflow, BinanceWebsocketClosed
from src.core.event_bus import EventBus

logger = logging.getLogger("BinanceConnector")


class WebSocketHealthCheck:
    """Monitora a saúde do WebSocket e detecta desconexões."""
    
    def __init__(self, timeout_seconds: int = 30):
        self.last_message_time: float = None
        self.timeout_seconds = timeout_seconds
        self.last_valid_price: float = None
    
    def on_message(self, price: float = None):
        """Atualiza timestamp da última mensagem recebida."""
        self.last_message_time = time.time()
        if price and price > 0:
            self.last_valid_price = price
    
    def is_healthy(self) -> bool:
        """Verifica se o WebSocket está saudável."""
        if self.last_message_time is None:
            return False
        return (time.time() - self.last_message_time) < self.timeout_seconds
    
    def get_last_valid_price(self) -> float:
        """Retorna o último preço válido."""
        return self.last_valid_price


class BinanceConnector:
    def __init__(self, api_key: str, api_secret: str, event_bus: EventBus, demo_mode: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.event_bus = event_bus
        self.demo_mode = demo_mode
        self.client = None
        self.bsm = None
        self.health_check = WebSocketHealthCheck(timeout_seconds=30)

    async def connect(self):
        if self.demo_mode:
            logger.info("🔌 Conectando à Binance Demo Trading...")
            # Demo Trading usa demo=True - a biblioteca cuida das URLs corretas
            self.client = await AsyncClient.create(
                self.api_key, 
                self.api_secret, 
                testnet=False,
                demo=True  # ISSO FAZ A MÁGICA - USA demo-fapi.binance.com
            )
        else:
            logger.info("🔌 Conectando à Binance Produção...")
            self.client = await AsyncClient.create(self.api_key, self.api_secret, testnet=False)
        
        self.bsm = BinanceSocketManager(self.client)
        logger.info("✅ Binance Conectada.")

    async def start_streams(self, symbol: str):
        """
        Versão BLINDADA: Se cair ou lotar a memória, ele reconecta sozinho.
        Inclui tratamento específico para BinanceWebsocketQueueOverflow.
        """
        symbol_lower = symbol.lower()
        streams = [
            f"{symbol_lower}@aggTrade",    
            f"{symbol_lower}@forceOrder",
            f"{symbol_lower}@depth@100ms",
        ]
        
        # LOOP DE VIDA INFINITA
        while True:
            try:
                # Salvar último preço válido antes de reconectar
                saved_price = self.health_check.get_last_valid_price()
                
                logger.info(f"🌊 (Re)Iniciando MULTIPLEX STREAM...")
                if saved_price:
                    logger.info(f"   📊 Último preço válido salvo: ${saved_price:.2f}")
                
                socket = self.bsm.multiplex_socket(streams)
                
                async with socket as ts:
                    while True:
                        try:
                            msg = await ts.recv()
                            
                            if msg and 'data' in msg:
                                payload = msg['data']
                                stream_name = msg['stream']
                                
                                if 'aggTrade' in stream_name:
                                    price = float(payload['p'])
                                    
                                    # Atualizar health check com o preço
                                    self.health_check.on_message(price)
                                    
                                    data_normalizada = {
                                        'symbol': payload['s'],
                                        'trade_id': payload['a'],
                                        'timestamp': payload['T'],
                                        'price': payload['p'],
                                        'quantity': payload['q'],
                                        'is_buyer_maker': payload['m']
                                    }
                                    await self.event_bus.publish('market_data', data_normalizada)

                                elif 'forceOrder' in stream_name:
                                    # Atualizar health check
                                    self.health_check.on_message()
                                    
                                    o = payload['o']
                                    data_liquidacao = {
                                        'symbol': o['s'],
                                        'side': o['S'],
                                        'price': o['p'],
                                        'quantity': o['q'],
                                        'amount_usd': float(o['p']) * float(o['q']),
                                        'timestamp': payload['T']
                                    }
                                    await self.event_bus.publish('liquidation_data', data_liquidacao)

                                elif 'depth' in stream_name:
                                    # Atualizar health check
                                    self.health_check.on_message()
                                    
                                    data_orderbook = {
                                        'event_type': 'orderbook',
                                        'bids': payload.get('b', []),
                                        'asks': payload.get('a', []),
                                        'timestamp': payload.get('E', 0)
                                    }
                                    await self.event_bus.publish('orderbook_data', data_orderbook)
                                    
                        except BinanceWebsocketQueueOverflow as e:
                            # Tratamento específico para Queue Overflow
                            logger.warning(f"⚠️ Buffer WebSocket cheio: {e}. Reconectando com buffer limpo...")
                            break
                        except BinanceWebsocketClosed as e:
                            logger.warning(f"⚠️ WebSocket fechado: {e}. Reconectando...")
                            break
                        except Exception as e:
                            error_str = str(e)
                            if 'queue' in error_str.lower() or 'overflow' in error_str.lower():
                                logger.warning(f"⚠️ Buffer WebSocket cheio: {e}. Reconectando...")
                            else:
                                logger.warning(f"⚠️ Engasgo no Stream: {e}")
                            break
                            
            except Exception as e:
                logger.error(f"❌ Erro Crítico na Conexão: {e}. Tentando voltar em 5s...")
                await asyncio.sleep(5)
    
    def is_websocket_healthy(self) -> bool:
        """Verifica se o WebSocket está saudável."""
        return self.health_check.is_healthy()
    
    def get_last_valid_price(self) -> float:
        """Retorna o último preço válido do WebSocket."""
        return self.health_check.get_last_valid_price()

    async def close(self):
        if self.client:
            await self.client.close_connection()