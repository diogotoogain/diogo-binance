"""
SNAME-MR Trade Executor
Executa ordens na Binance Demo/Real.
TUDO PARAMETRIZÁVEL via .env ou construtor.
"""
import logging
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("TradeExecutor")


class TradeExecutor:
    def __init__(self,
        client,
        position_manager,
        risk_manager,
        symbol: str = None,
        min_confidence: str = None,
        execution_enabled: bool = None
    ):
        self.client = client
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        
        self.symbol = symbol or os.getenv("SYMBOL", "BTCUSDT")
        self.min_confidence = min_confidence or os.getenv("MIN_CONFIDENCE", "HIGH")
        self.execution_enabled = execution_enabled if execution_enabled is not None else os.getenv("EXECUTION_ENABLED", "true").lower() == "true"
        
        self.total_trades = 0
        self.winning_trades = 0
        
        logger.info(f"⚡ TradeExecutor | Symbol: {self.symbol} | Min Conf: {self.min_confidence} | Ativo: {self.execution_enabled}")

    async def initialize(self):
        try:
            account = await self.client.futures_account()
            balance = float(account['totalWalletBalance'])
            available = float(account['availableBalance'])
            self.risk_manager.set_initial_balance(balance)
            logger.info(f"💰 Conta | Saldo: ${balance:.2f} | Disponível: ${available:.2f}")
            
            positions = await self.client.futures_position_information(symbol=self.symbol)
            for pos in positions:
                qty = float(pos['positionAmt'])
                if qty != 0:
                    side = 'LONG' if qty > 0 else 'SHORT'
                    entry = float(pos['entryPrice'])
                    logger.warning(f"⚠️ Posição existente: {side} {abs(qty)} @ ${entry:.2f}")
                    self.position_manager.open_position(self.symbol, 'BUY' if qty > 0 else 'SELL', entry, abs(qty))
            return True
        except Exception as e:
            logger.error(f"❌ Erro inicialização: {e}")
            return False

    def _check_confidence(self, confidence: str) -> bool:
        levels = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'MAX': 4}
        return levels.get(confidence, 0) >= levels.get(self.min_confidence, 3)

    async def execute_signal(self, signal: dict, current_price: float) -> bool:
        if not self.execution_enabled:
            logger.info(f"⏸️ [OBS] {signal.get('action')} @ ${current_price:.2f}")
            return False
        
        action = signal.get('action')
        confidence = signal.get('confidence', 'MEDIUM')
        
        if not self._check_confidence(confidence):
            return False
        
        if self.position_manager.has_position():
            pos = self.position_manager.current_position
            if (pos.is_long and action == 'SELL') or (not pos.is_long and action == 'BUY'):
                return await self._close_position(current_price, "Sinal_Oposto")
            return False
        
        can_open, msg = self.risk_manager.can_open_position(0)
        if not can_open:
            logger.warning(f"⚠️ Bloqueado: {msg}")
            return False
        
        sl, tp = self.position_manager.calculate_sl_tp(current_price, action)
        account = await self.client.futures_account()
        balance = float(account['availableBalance'])
        qty = self.risk_manager.calculate_position_size(balance, current_price, sl)
        
        if qty <= 0:
            return False
        
        return await self._open_position(action, qty, current_price, sl, tp)

    async def _open_position(self, side: str, qty: float, price: float, sl: float, tp: float) -> bool:
        try:
            order = await self.client.futures_create_order(symbol=self.symbol, side=side, type='MARKET', quantity=qty)
            fill = float(order.get('avgPrice', price))
            logger.info(f"✅ {side} {qty} @ ${fill:.2f}")
            
            self.position_manager.open_position(self.symbol, side, fill, qty, sl, tp)
            
            sl_side = 'SELL' if side == 'BUY' else 'BUY'
            await self.client.futures_create_order(symbol=self.symbol, side=sl_side, type='STOP_MARKET', stopPrice=round(sl, 2), closePosition=True)
            await self.client.futures_create_order(symbol=self.symbol, side=sl_side, type='TAKE_PROFIT_MARKET', stopPrice=round(tp, 2), closePosition=True)
            
            self.total_trades += 1
            return True
        except Exception as e:
            logger.error(f"❌ Erro: {e}")
            return False

    async def _close_position(self, price: float, reason: str) -> bool:
        if not self.position_manager.has_position():
            return False
        try:
            pos = self.position_manager.current_position
            await self.client.futures_cancel_all_open_orders(symbol=self.symbol)
            order = await self.client.futures_create_order(symbol=self.symbol, side='SELL' if pos.is_long else 'BUY', type='MARKET', quantity=pos.quantity)
            fill = float(order.get('avgPrice', price))
            pnl = self.position_manager.close_position(fill)
            self.risk_manager.update_daily_pnl(pnl)
            if pnl > 0: self.winning_trades += 1
            return True
        except Exception as e:
            logger.error(f"❌ Erro: {e}")
            return False

    def enable(self): self.execution_enabled = True
    def disable(self): self.execution_enabled = False