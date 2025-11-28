import asyncio
import logging
from binance import AsyncClient, BinanceSocketManager
from src.core.event_bus import EventBus

logger = logging.getLogger("BinanceConnector")

class BinanceConnector:
    def __init__(self, api_key: str, api_secret: str, event_bus: EventBus, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.event_bus = event_bus
        self.testnet = testnet
        self.client = None
        self.bsm = None

    async def connect(self):
        logger.info(f"🔌 Conectando à Binance (Testnet={self.testnet})...")
        self.client = await AsyncClient.create(self.api_key, self.api_secret, testnet=self.testnet)
        self.bsm = BinanceSocketManager(self.client)
        logger.info("✅ Binance Conectada.")

    async def start_streams(self, symbol: str):
        """
        Versão BLINDADA: Se cair ou lotar a memória, ele reconecta sozinho.
        """
        symbol_lower = symbol.lower()
        streams = [
            f"{symbol_lower}@aggTrade",    
            f"{symbol_lower}@forceOrder",
            f"{symbol_lower}@depth@100ms",  # Order Book a cada 100ms
        ]
        
        # LOOP DE VIDA INFINITA
        while True:
            try:
                logger.info(f"🌊 (Re)Iniciando MULTIPLEX STREAM...")
                socket = self.bsm.multiplex_socket(streams)
                
                async with socket as ts:
                    while True:
                        try:
                            msg = await ts.recv()
                            
                            if msg and 'data' in msg:
                                payload = msg['data']
                                stream_name = msg['stream']
                                
                                if 'aggTrade' in stream_name:
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
                                    data_orderbook = {
                                        'event_type': 'orderbook',
                                        'bids': payload.get('b', []),
                                        'asks': payload.get('a', []),
                                        'timestamp': payload.get('E', 0)
                                    }
                                    await self.event_bus.publish('orderbook_data', data_orderbook)
                                    
                        except Exception as e:
                            # Se der erro de Overflow (fila cheia), ele cai aqui
                            logger.warning(f"⚠️ Engasgo no Stream (Buffer cheio?): {e}")
                            break # Sai do loop interno para reiniciar a conexão limpa
                            
            except Exception as e:
                # Se a conexão cair de vez
                logger.error(f"❌ Erro Crítico na Conexão: {e}. Tentando voltar em 5s...")
                await asyncio.sleep(5)

    async def close(self):
        if self.client:
            await self.client.close_connection()