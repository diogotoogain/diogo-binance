import logging
from typing import Optional
from src.core.event_bus import EventBus

# Importa a Tropa de Elite Completa
from src.strategies.sentiment_strategy import SentimentStrategy
from src.strategies.smart_strategy import SmartStrategy
from src.strategies.liquidation_strategy import LiquidationStrategy
from src.strategies.vpin_strategy import VPINStrategy

# NOVAS ESTRATÉGIAS INSTITUCIONAIS
from src.strategies.cascade_liquidation_strategy import CascadeLiquidationStrategy
from src.strategies.flow_imbalance_strategy import FlowImbalanceStrategy
from src.strategies.rolling_vwap_strategy import RollingVWAPStrategy
from src.strategies.obi_strategy import OBIStrategy

logger = logging.getLogger("Orchestrator")

class StrategyOrchestrator:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.strategies = []  # Estratégias de Ticks (Preço)
        self.liquidation_strategies = []  # Estratégias de Eventos (Quebras)
        self.orderbook_strategies = []  # Estratégias de Order Book (NOVO!)
        
        # Meta-Controller e Executor (injetados depois)
        self.meta_controller = None
        self.executor = None
        
        # --- EQUIPE DE TICKS ---
        self.strategies.append(SentimentStrategy())
        self.strategies.append(SmartStrategy())
        self.strategies.append(VPINStrategy())
        self.strategies.append(FlowImbalanceStrategy())  # NOVA!
        self.strategies.append(RollingVWAPStrategy())    # NOVA!
        
        # --- EQUIPE DE LIQUIDAÇÃO ---
        self.liquidation_strategies.append(LiquidationStrategy())
        self.liquidation_strategies.append(CascadeLiquidationStrategy())  # NOVA!
        
        # --- EQUIPE DE ORDER BOOK --- (NOVA!)
        self.orderbook_strategies.append(OBIStrategy())
        
        logger.info(f"🧠 Orquestrador iniciado. Tick-Strats: {len(self.strategies)} | Event-Strats: {len(self.liquidation_strategies)} | Orderbook-Strats: {len(self.orderbook_strategies)}")

    def set_meta_controller(self, meta_controller) -> None:
        """Injeta o MetaController (Oráculo)"""
        self.meta_controller = meta_controller
        logger.info("🧠 MetaController conectado ao Orchestrator")

    def set_executor(self, executor) -> None:
        """Injeta o TradeExecutor"""
        self.executor = executor
        logger.info("⚡ TradeExecutor conectado ao Orchestrator")

    # --- PROCESSA PREÇO (TICKS) ---
    async def _process_tick(self, data: dict):
        data['event_type'] = 'tick'
        for strategy in self.strategies:
            signal = await strategy.on_tick(data)
            await self._handle_signal(signal, data, strategy.name)

    # --- PROCESSA LIQUIDAÇÃO ---
    async def _process_liquidation(self, data: dict):
        data['event_type'] = 'liquidation'
        for strategy in self.liquidation_strategies:
            signal = await strategy.on_tick(data)
            await self._handle_signal(signal, data, strategy.name)

    # --- PROCESSA ORDER BOOK --- (NOVO!)
    async def _process_orderbook(self, data: dict):
        data['event_type'] = 'orderbook'
        for strategy in self.orderbook_strategies:
            signal = await strategy.on_tick(data)
            await self._handle_signal(signal, data, strategy.name)

    # --- DISPARADOR UNIFICADO ---
    async def _handle_signal(self, signal, data, strat_name):
        if signal:
            logger.info(f"🚨 SINAL >>> {strat_name}: {signal['action']} | Motivo: {signal['reason']}")
            
            # Garante que o preço existe (na liquidação o preço vem no dado)
            price = data.get('price', 0)
            
            # Se tiver MetaController, passa o sinal para votação
            if self.meta_controller:
                combined_signal = self.meta_controller.receive_signal(strat_name, signal)
                
                # Se o MetaController retornar um sinal combinado, executa
                if combined_signal and self.executor:
                    try:
                        current_price = float(price) if price else 0
                        await self.executor.execute_signal(combined_signal, current_price)
                    except Exception as e:
                        logger.error(f"❌ Erro ao executar sinal: {e}")
            
            await self.event_bus.publish('trade_signal', {
                'strategy': strat_name,
                'action': signal['action'],
                'price': price,
                'reason': signal['reason']
            })

    async def start(self):
        # Assina os três canais
        self.event_bus.subscribe('market_data', self._process_tick)
        self.event_bus.subscribe('liquidation_data', self._process_liquidation)
        self.event_bus.subscribe('orderbook_data', self._process_orderbook)  # NOVO!
        logger.info("🧠 Cérebro conectado aos fluxos de Mercado, Liquidação e Order Book.")
