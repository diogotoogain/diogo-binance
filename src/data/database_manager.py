from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from src.core.database import Base, MarketData, Signal, LiquidationData
from src.core.event_bus import EventBus
import logging
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
logger = logging.getLogger("DatabaseManager")

class DatabaseManager:
    def __init__(self):
        database_url = os.getenv("DATABASE_URL")
        if database_url and database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")
            
        self.engine = create_async_engine(database_url, echo=False, future=True)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_models(self):
        try:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("🗄️  Tabelas verificadas/criadas (Incluindo Liquidações).")
        except Exception as e:
            logger.critical(f"❌ Falha no Banco: {e}")
            raise

    async def _persist_market_data(self, data: dict):
        try:
            async with self.async_session() as session:
                async with session.begin():
                    ts_obj = datetime.fromtimestamp(int(data['timestamp']) / 1000)
                    record = MarketData(
                        id=int(data['trade_id']),
                        timestamp=ts_obj,
                        timestamp_ms=int(data['timestamp']),
                        symbol=data['symbol'],
                        price=float(data['price']),
                        quantity=float(data['quantity']),
                        is_buyer_maker=data['is_buyer_maker']
                    )
                    await session.merge(record)
        except Exception:
            pass 

    # --- NOVO: SALVA LIQUIDAÇÕES ---
    async def _persist_liquidation(self, data: dict):
        try:
            async with self.async_session() as session:
                async with session.begin():
                    record = LiquidationData(
                        timestamp=datetime.utcnow(),
                        symbol=data['symbol'],
                        side=data['side'],
                        price=float(data['price']),
                        quantity=float(data['quantity']),
                        amount_usd=float(data['amount_usd'])
                    )
                    session.add(record)
            logger.info(f"💀 LIQUIDAÇÃO SALVA: {data['side']} de ${data['amount_usd']:.2f}")
        except Exception as e:
            logger.error(f"❌ Erro ao salvar Liquidação: {e}")

    async def _persist_signal(self, data: dict):
        try:
            async with self.async_session() as session:
                async with session.begin():
                    record = Signal(
                        strategy=data['strategy'],
                        timestamp=datetime.utcnow(),
                        signal_type=data['action'],
                        strength=1.0,
                        metadata_info=data
                    )
                    session.add(record)
        except Exception as e:
            logger.error(f"❌ Erro ao salvar Sinal: {e}")

    async def register_listeners(self, event_bus: EventBus):
        event_bus.subscribe("market_data", self._persist_market_data)
        event_bus.subscribe("liquidation_data", self._persist_liquidation)
        event_bus.subscribe("trade_signal", self._persist_signal)
        logger.info("💾 DatabaseManager ouvindo Ticks, Liquidações e Sinais.")