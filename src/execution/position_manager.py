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
        self.last_valid_price: Optional[float] = None  # Cache do último preço válido
        
        logger.info(f"📍 PositionManager iniciado:")
        logger.info(f"   🛑 SL Padrão: {self.default_sl_percent*100}%")
        logger.info(f"   🎯 TP Padrão: {self.default_tp_percent*100}%")
        logger.info(f"   📈 Trailing Stop: {'Ativo' if self.use_trailing_stop else 'Desativado'}")

    def has_position(self) -> bool:
        return self.current_position is not None

    def is_valid_price(self, price: float) -> bool:
        """Valida se o preço é válido (não None, não zero, não negativo)."""
        return price is not None and price > 0

    def update_last_valid_price(self, price: float):
        """Atualiza cache do último preço válido."""
        if self.is_valid_price(price):
            self.last_valid_price = price

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

    def close_position(self, exit_price: float) -> Optional[float]:
        """
        Fecha a posição atual e calcula o PnL.
        Retorna None se o preço for inválido (não fecha a posição).
        Retorna o PnL se a posição foi fechada com sucesso.
        """
        if not self.has_position():
            return 0.0
        
        # VALIDAÇÃO CRÍTICA: Rejeitar preço inválido
        if not self.is_valid_price(exit_price):
            logger.error(f"❌ PREÇO INVÁLIDO: {exit_price}. Abortando fechamento de posição!")
            return None
        
        # Validar que o preço está dentro de um range razoável (±50% do último preço conhecido)
        if self.last_valid_price and abs(exit_price - self.last_valid_price) / self.last_valid_price > 0.5:
            logger.error(f"❌ PREÇO SUSPEITO: {exit_price} vs último válido {self.last_valid_price}. Abortando!")
            return None
        
        pos = self.current_position
        
        # Validar entry_price da posição
        if not self.is_valid_price(pos.entry_price):
            logger.error(f"❌ Entry price inválido: {pos.entry_price}. Abortando fechamento!")
            return None
        
        if pos.is_long:
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity
        
        # FIX: Verificar antes de dividir para evitar division by zero
        if pos.entry_price > 0 and pos.quantity > 0:
            pnl_percent = (pnl / (pos.entry_price * pos.quantity)) * 100
        else:
            pnl_percent = 0.0
            logger.warning(f"⚠️ Entry price ou quantity inválido: entry={pos.entry_price}, qty={pos.quantity}")
        
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
        
        # VALIDAÇÃO CRÍTICA: Não processar preço inválido
        if not self.is_valid_price(current_price):
            logger.warning(f"⚠️ Preço inválido em should_close: {current_price}. Ignorando.")
            return None
        
        # Atualiza cache do último preço válido
        self.update_last_valid_price(current_price)
        
        self.update_trailing_stop(current_price)
        pos = self.current_position
        if pos.is_long:
            if current_price <= pos.stop_loss: return 'STOP_LOSS'
            if current_price >= pos.take_profit: return 'TAKE_PROFIT'
        else:
            if current_price >= pos.stop_loss: return 'STOP_LOSS'
            if current_price <= pos.take_profit: return 'TAKE_PROFIT'
        return None

    def sync_position(self, symbol: str, side: str, entry_price: float, quantity: float,
                       unrealized_pnl: float = 0.0, stop_loss: float = None, take_profit: float = None):
        """
        Sincroniza posição local com dados reais da Binance.
        Usado para atualizar o estado sem abrir nova posição.
        """
        if stop_loss is None or take_profit is None:
            stop_loss, take_profit = self.calculate_sl_tp(entry_price, 'BUY' if side == 'LONG' else 'SELL')
        
        self.current_position = Position(
            symbol=symbol, side=side, entry_price=entry_price,
            quantity=quantity, stop_loss=stop_loss, take_profit=take_profit,
            entry_time=datetime.now()
        )
        
        self.highest_price = entry_price
        self.lowest_price = entry_price
        
        logger.info(f"🔄 POSIÇÃO SINCRONIZADA: {side} {quantity} {symbol} @ ${entry_price:.2f} | PnL: ${unrealized_pnl:.2f}")

    def clear_position(self):
        """Limpa posição local (usado quando Binance reporta sem posição)."""
        if self.has_position():
            logger.info("🔄 Posição local limpa (sincronizada com Binance)")
        self.current_position = None

    def get_status(self) -> dict:
        if not self.has_position():
            return {'has_position': False}
        pos = self.current_position
        return {
            'has_position': True, 'symbol': pos.symbol, 'side': pos.side,
            'entry_price': pos.entry_price, 'quantity': pos.quantity,
            'stop_loss': pos.stop_loss, 'take_profit': pos.take_profit
        }