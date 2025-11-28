"""
SNAME-MR Risk Manager
Gerencia risco por trade, posição máxima e stop diário.
TUDO PARAMETRIZÁVEL via .env ou construtor.
"""
import logging
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("RiskManager")


class RiskManager:
    def __init__(self,
        risk_per_trade: float = None,
        max_position_size: float = None,
        max_daily_loss: float = None,
        max_concurrent_positions: int = None
    ):
        # Carrega do .env se não fornecido
        self.risk_per_trade = risk_per_trade or float(os.getenv("RISK_PER_TRADE", 0.01))
        self.max_position_size = max_position_size or float(os.getenv("MAX_POSITION_SIZE", 0.05))
        self.max_daily_loss = max_daily_loss or float(os.getenv("MAX_DAILY_LOSS", 0.03))
        self.max_concurrent_positions = max_concurrent_positions or int(os.getenv("MAX_CONCURRENT_POSITIONS", 1))
        
        self.daily_pnl = 0.0
        self.initial_balance = 0.0
        self.trades_today = 0
        
        logger.info(f"🛡️ RiskManager iniciado:")
        logger.info(f"   📊 Risco/Trade: {self.risk_per_trade*100}%")
        logger.info(f"   📊 Max Posição: {self.max_position_size*100}%")
        logger.info(f"   📊 Stop Diário: {self.max_daily_loss*100}%")
        logger.info(f"   📊 Max Posições: {self.max_concurrent_positions}")

    def set_initial_balance(self, balance: float):
        self.initial_balance = balance
        logger.info(f"💰 Saldo inicial: ${balance:.2f}")

    def calculate_position_size(self, balance: float, entry_price: float, stop_loss_price: float) -> float:
        risk_amount = balance * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            logger.warning("⚠️ Stop Loss igual ao Entry! Usando 1% do preço.")
            price_risk = entry_price * 0.01
        
        position_size = risk_amount / price_risk
        max_size = (balance * self.max_position_size) / entry_price
        position_size = min(position_size, max_size)
        position_size = round(position_size, 3)
        
        logger.info(f"📊 Position Size: {position_size} BTC | Risco: ${risk_amount:.2f}")
        return position_size

    def can_open_position(self, current_positions: int) -> tuple:
        if current_positions >= self.max_concurrent_positions:
            return False, f"Máximo de {self.max_concurrent_positions} posições atingido"
        
        if self.initial_balance > 0:
            daily_loss_percent = -self.daily_pnl / self.initial_balance
            if daily_loss_percent >= self.max_daily_loss:
                return False, f"STOP DIÁRIO: Perda de {daily_loss_percent*100:.2f}%"
        
        return True, "OK"

    def update_daily_pnl(self, pnl: float):
        self.daily_pnl += pnl
        self.trades_today += 1
        emoji = "🟢" if self.daily_pnl >= 0 else "🔴"
        logger.info(f"{emoji} P&L Diário: ${self.daily_pnl:.2f} | Trades: {self.trades_today}")

    def reset_daily(self):
        logger.info(f"🔄 Reset diário | P&L final: ${self.daily_pnl:.2f} | Trades: {self.trades_today}")
        self.daily_pnl = 0.0
        self.trades_today = 0

    def get_status(self) -> dict:
        return {
            'daily_pnl': self.daily_pnl,
            'trades_today': self.trades_today,
            'initial_balance': self.initial_balance,
            'risk_per_trade': self.risk_per_trade,
            'max_position_size': self.max_position_size,
            'max_daily_loss': self.max_daily_loss
        }