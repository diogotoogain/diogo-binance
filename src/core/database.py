from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, BigInteger, Boolean
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True, index=True)
    strategy = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    pnl = Column(Float, nullable=True)

class MarketData(Base): # Ticks (AggTrade)
    __tablename__ = 'market_data'
    id = Column(BigInteger, primary_key=True, index=True) 
    timestamp = Column(DateTime, nullable=False, index=True)
    timestamp_ms = Column(BigInteger, nullable=False)
    symbol = Column(String, nullable=False, index=True)
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    is_buyer_maker = Column(Boolean, nullable=False)

# --- NOVO: Tabela de Liquidações (Gente Quebrando) ---
class LiquidationData(Base):
    __tablename__ = 'liquidations'
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False) # BUY (Short Liquidado) ou SELL (Long Liquidado)
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    amount_usd = Column(Float, nullable=False) # Valor total em Dólar

class Signal(Base):
    __tablename__ = 'signals'
    id = Column(Integer, primary_key=True, index=True)
    strategy = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    signal_type = Column(String, nullable=False)
    strength = Column(Float, nullable=False)
    metadata_info = Column(JSON, nullable=True)