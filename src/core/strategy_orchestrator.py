import logging
from src.core.event_bus import EventBus

# Importa a Tropa de Elite Completa
from src.strategies.sentiment_strategy import SentimentStrategy
from src.strategies.smart_strategy import SmartStrategy
from src.strategies.liquidation_strategy import LiquidationStrategy
from src.strategies.vpin_strategy import VPINStrategy

logger = logging.getLogger("Orchestrator")

class StrategyOrchestrator:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.strategies = [] # Estratégias de Ticks (Preço)
        self.liquidation_strategies = [] # Estratégias de Eventos (Quebras)
        
        # --- EQUIPE DE TICKS ---
        self.strategies.append(SentimentStrategy())
        self.strategies.append(SmartStrategy())
        self.strategies.append(VPINStrategy())
        
        # --- EQUIPE DE LIQUIDAÇÃO ---
        # Aqui está o Caçador que faltava!
        self.liquidation_strategies.append(LiquidationStrategy())
        
        logger.info(f"🧠 Orquestrador iniciado. Tick-Strats: {len(self.strategies)} | Event-Strats: {len(self.liquidation_strategies)}")

    # --- PROCESSA PREÇO (TICKS) ---
    async def _process_tick(self, data: dict):
        data['event_type'] = 'tick'
        for strategy in self.strategies:
            signal = await strategy.on_tick(data)
            await self._handle_signal(signal, data, strategy.name)

    # --- PROCESSA LIQUIDAÇÃO (O NOVO FLUXO) ---
    async def _process_liquidation(self, data: dict):
        data['event_type'] = 'liquidation'
        for strategy in self.liquidation_strategies:
            signal = await strategy.on_tick(data)
            await self._handle_signal(signal, data, strategy.name)

    # --- DISPARADOR UNIFICADO ---
    async def _handle_signal(self, signal, data, strat_name):
        if signal:
            logger.info(f"🚨 SINAL >>> {strat_name}: {signal['action']} | Motivo: {signal['reason']}")
            
            # Garante que o preço existe (na liquidação o preço vem no dado)
            price = data.get('price', 0)
            
            await self.event_bus.publish('trade_signal', {
                'strategy': strat_name,
                'action': signal['action'],
                'price': price,
                'reason': signal['reason']
            })

    async def start(self):
        # Assina os dois canais
        self.event_bus.subscribe('market_data', self._process_tick)
        self.event_bus.subscribe('liquidation_data', self._process_liquidation)
        logger.info("🧠 Cérebro conectado aos fluxos de Mercado e Liquidação.")