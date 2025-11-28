"""
SNAME-MR Position Manager
Gerencia posições abertas, SL, TP, trailing stop.
TUDO PARAMETRIZÁVEL via .env ou construtor.
"""
import logging
import os
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("PositionManager")


@dataclass
class Position:
    symbol: str
    side: str  # 'LONG' ou 'SHORT'
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    
    @property
    def is_long(self) -> bool:
        return self.side == 'LONG'


class PositionManager:
    def __init__(self,
        default_sl_percent: float = None,
        default_tp_percent: float = None,
        use_trailing_stop: bool = None,
        trailing_percent: float = None
    ):
        self.default_sl_percent = default_sl_percent or float(os.getenv("DEFAULT_SL_PERCENT", 0.01))
        self.default_tp_percent = default_tp_percent or float(os.getenv("DEFAULT_TP_PERCENT", 0.02))
        self.use_trailing_stop = use_trailing_stop if use_trailing_stop is not None else os.getenv("USE_TRAILING_STOP", "false").lower() == "true"
        self.trailing_percent = trailing_percent or float(os.getenv("TRAILING_PERCENT", 0.005))
        
        self.current_position: Optional[Position] = None
        self.highest_price = 0.0
        self.lowest_price = float('inf')
        
        logger.info(f"📍 PositionManager iniciado:")
        logger.info(f"   🛑 SL Padrão: {self.default_sl_percent*100}%")
        logger.info(f"   🎯 TP Padrão: {self.default_tp_percent*100}%")
        logger.info(f"   📈 Trailing Stop: {'Ativo' if self.use_trailing_stop else 'Desativado'}")

    def has_position(self) -> bool:
        return self.current_position is not None

    def calculate_sl_tp(self, entry_price: float, side: str) -> tuple:
        if side in ['BUY', 'LONG']:
            stop_loss = entry_price * (1 - self.default_sl_percent)
            take_profit = entry_price * (1 + self.default_tp_percent)
        else:
            stop_loss = entry_price * (1 + self.default_sl_percent)
            take_profit = entry_price * (1 - self.default_tp_percent)
        return round(stop_loss, 2), round(take_profit, 2)

    def open_position(self, symbol: str, side: str, entry_price: float, quantity: float,
                      stop_loss: float = None, take_profit: float = None) -> Optional[Position]:
        if self.has_position():
            logger.warning("⚠️ Já existe posição aberta!")
            return None
        
        if stop_loss is None or take_profit is None:
            stop_loss, take_profit = self.calculate_sl_tp(entry_price, side)
        
        position_side = 'LONG' if side in ['BUY', 'LONG'] else 'SHORT'
        
        self.current_position = Position(
            symbol=symbol, side=position_side, entry_price=entry_price,
            quantity=quantity, stop_loss=stop_loss, take_profit=take_profit,
            entry_time=datetime.now()
        )
        
        self.highest_price = entry_price
        self.lowest_price = entry_price
        
        logger.info(f"📈 POSIÇÃO ABERTA: {position_side} {quantity} {symbol} @ ${entry_price:.2f}")
        logger.info(f"   🛑 SL: ${stop_loss:.2f} | 🎯 TP: ${take_profit:.2f}")
        return self.current_position

    def close_position(self, exit_price: float) -> float:
        if not self.has_position():
            return 0.0
        
        pos = self.current_position
        if pos.is_long:
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity
        
        if pos.entry_price > 0 and pos.quantity > 0:
            pnl_percent = (pnl / (pos.entry_price * pos.quantity)) * 100
        else:
            pnl_percent = 0.0
        emoji = "🟢" if pnl > 0 else "🔴"
        logger.info(f"{emoji} POSIÇÃO FECHADA: {pos.side} @ ${exit_price:.2f} | P&L: ${pnl:.2f} ({pnl_percent:+.2f}%)")
        
        self.current_position = None
        return pnl

    def update_trailing_stop(self, current_price: float):
        if not self.has_position() or not self.use_trailing_stop:
            return
        
        pos = self.current_position
        if pos.is_long and current_price > self.highest_price:
            self.highest_price = current_price
            new_sl = current_price * (1 - self.trailing_percent)
            if new_sl > pos.stop_loss:
                pos.stop_loss = round(new_sl, 2)
                logger.info(f"📈 Trailing SL: ${pos.stop_loss:.2f}")
        elif not pos.is_long and current_price < self.lowest_price:
            self.lowest_price = current_price
            new_sl = current_price * (1 + self.trailing_percent)
            if new_sl < pos.stop_loss:
                pos.stop_loss = round(new_sl, 2)
                logger.info(f"📉 Trailing SL: ${pos.stop_loss:.2f}")

    def should_close(self, current_price: float) -> Optional[str]:
        if not self.has_position():
            return None
        self.update_trailing_stop(current_price)
        pos = self.current_position
        if pos.is_long:
            if current_price <= pos.stop_loss: return 'STOP_LOSS'
            if current_price >= pos.take_profit: return 'TAKE_PROFIT'
        else:
            if current_price >= pos.stop_loss: return 'STOP_LOSS'
            if current_price <= pos.take_profit: return 'TAKE_PROFIT'
        return None

    def get_status(self) -> dict:
        if not self.has_position():
            return {'has_position': False}
        pos = self.current_position
        return {
            'has_position': True, 'symbol': pos.symbol, 'side': pos.side,
            'entry_price': pos.entry_price, 'quantity': pos.quantity,
            'stop_loss': pos.stop_loss, 'take_profit': pos.take_profit
        }