import asyncio
import logging
import os
from dotenv import load_dotenv
from datetime import datetime

from src.core.event_bus import EventBus
from src.data.database_manager import DatabaseManager
from src.data.binance_connector import BinanceConnector
from src.core.strategy_orchestrator import StrategyOrchestrator
from src.core.dashboard import TerminalDashboard
from src.utils.trade_logger import TradeLogger
from src.core.meta_controller import MetaController
from src.execution import RiskManager, PositionManager, TradeExecutor
from src.api.dashboard_api import WebDashboard

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.FileHandler("robo_madrugada.log"), logging.StreamHandler()]
)
logger = logging.getLogger("Maestro")

async def main():
    load_dotenv()
    logger.info("--- 🚀 SNAME-MR V2: INGESTÃO MULTI-STREAM ---")

    # 1. Infra
    event_bus = EventBus()
    db_manager = DatabaseManager()
    trade_logger = TradeLogger()
    
    # 2. Banco
    await db_manager.init_models()
    await db_manager.register_listeners(event_bus)

    # 3. Cérebro
    orchestrator = StrategyOrchestrator(event_bus)
    await orchestrator.start()

    # 4. Trade Logger
    event_bus.subscribe('trade_signal', trade_logger.on_signal)

    # 5. Bus
    asyncio.create_task(event_bus.start())

    # 6. Meta-Controller (Oráculo)
    meta_controller = MetaController()
    orchestrator.set_meta_controller(meta_controller)
    
    # 7. Conectar à Binance
    api_key = os.getenv("BINANCE_API_KEY")
    secret = os.getenv("BINANCE_SECRET_KEY")
    demo_mode = os.getenv("USE_DEMO", "false").lower() == "true"
    connector = BinanceConnector(api_key, secret, event_bus, demo_mode=demo_mode)
    await connector.connect()
    
    # 8. Risk & Position Managers
    risk_manager = RiskManager()
    position_manager = PositionManager()
    
    # 9. Trade Executor
    executor = TradeExecutor(
        client=connector.client,
        position_manager=position_manager,
        risk_manager=risk_manager
    )
    await executor.initialize()
    orchestrator.set_executor(executor)
    
    # 10. Iniciar Orchestrator
    await orchestrator.start()

    # 11. Bus (redundante, mas mantido para compatibilidade)
    asyncio.create_task(event_bus.start())
    
    # 12. Dashboard Terminal (agora com acesso ao connector e managers)
    dashboard = TerminalDashboard(
        event_bus=event_bus,
        connector=connector,
        position_manager=position_manager,
        risk_manager=risk_manager
    )
    await dashboard.start()
    
    # 13. Web Dashboard (porta 8080)
    web_dashboard = WebDashboard(
        event_bus=event_bus,
        connector=connector,
        orchestrator=orchestrator,
        meta_controller=meta_controller,
        port=8080
    )
    await web_dashboard.start()

    # 14. Ligar TUDO (Ticks + Liquidações)
    stream_task = asyncio.create_task(connector.start_streams("BTCUSDT"))

    print("\n💀 CAÇADOR DE LIQUIDEZ ATIVO... Monitorando Ticks e Quebras...\n")
    print("🌐 Dashboard Web: http://localhost:8080\n")

    try:
        await stream_task
    except asyncio.CancelledError:
        logger.info("Desligando...")
    finally:
        await web_dashboard.stop()
        await dashboard.stop()
        await connector.close()
        await event_bus.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Parando...")