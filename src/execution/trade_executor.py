"""
SNAME-MR Trade Executor
Executa ordens na Binance Demo/Real.
TUDO PARAMETRIZÁVEL via .env ou construtor.
"""
import logging
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from src.notifications.webhook_notifier import WebhookNotifier

load_dotenv()
logger = logging.getLogger("TradeExecutor")


class TradeExecutor:
    def __init__(self,
        client,
        position_manager,
        risk_manager,
        symbol: str = None,
        min_confidence: str = None,
        execution_enabled: bool = None,
        notifier: WebhookNotifier = None
    ):
        self.client = client
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        
        self.symbol = symbol or os.getenv("SYMBOL", "BTCUSDT")
        self.min_confidence = min_confidence or os.getenv("MIN_CONFIDENCE", "HIGH")
        self.execution_enabled = execution_enabled if execution_enabled is not None else os.getenv("EXECUTION_ENABLED", "true").lower() == "true"
        self.notifier = notifier or WebhookNotifier()
        
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
            
            # Sincroniza posição com Binance
            await self.sync_with_binance()
            return True
        except Exception as e:
            logger.error(f"❌ Erro inicialização: {e}")
            return False

    async def _get_binance_position(self) -> Optional[Dict[str, Any]]:
        """Busca posição REAL na Binance."""
        try:
            positions = await self.client.futures_position_information(symbol=self.symbol)
            for pos in positions:
                qty = float(pos.get('positionAmt', 0))
                if qty != 0:
                    return {
                        'positionAmt': qty,
                        'side': 'LONG' if qty > 0 else 'SHORT',
                        'entryPrice': float(pos.get('entryPrice', 0)),
                        'unRealizedProfit': float(pos.get('unRealizedProfit', 0)),
                        'markPrice': float(pos.get('markPrice', 0))
                    }
            return None
        except Exception as e:
            logger.error(f"❌ Erro ao buscar posição Binance: {e}")
            return None

    async def sync_with_binance(self) -> Optional[Dict[str, Any]]:
        """Sincroniza estado local com Binance."""
        try:
            # Posições
            binance_pos = await self._get_binance_position()
            if binance_pos:
                qty = binance_pos['positionAmt']
                self.position_manager.sync_position(
                    symbol=self.symbol,
                    side='LONG' if qty > 0 else 'SHORT',
                    entry_price=binance_pos['entryPrice'],
                    quantity=abs(qty),
                    unrealized_pnl=binance_pos['unRealizedProfit']
                )
                logger.info(f"🔄 Sincronizado: {binance_pos['side']} {abs(qty)} @ ${binance_pos['entryPrice']:.2f}")
            else:
                # Limpa posição local se Binance não tem posição
                self.position_manager.clear_position()
            
            # Ordens abertas
            orders = await self.client.futures_get_open_orders(symbol=self.symbol)
            
            # Conta
            account = await self.client.futures_account()
            
            return {
                'position': binance_pos,
                'open_orders': orders,
                'account': account
            }
        except Exception as e:
            logger.error(f"❌ Erro ao sincronizar com Binance: {e}")
            return None

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
        
        # CRÍTICO: Verificar posição REAL na Binance, não só no PositionManager local
        binance_position = await self._get_binance_position()
        
        if binance_position:
            qty = binance_position['positionAmt']
            is_long = qty > 0
            is_short = qty < 0
            
            # Se tem posição e sinal é oposto, FECHAR primeiro
            if (is_long and action == 'SELL') or (is_short and action == 'BUY'):
                success = await self._close_position(current_price, "Sinal_Oposto")
                if not success:
                    logger.error("❌ Falha ao fechar posição! NÃO abrindo nova!")
                    return False  # NÃO CONTINUA se não conseguiu fechar!
                # Após fechar com sucesso, continua para abrir nova posição
            
            # Se tem posição e sinal é mesmo lado, IGNORAR (não acumula!)
            elif (is_long and action == 'BUY') or (is_short and action == 'SELL'):
                logger.info(f"⏭️ Já em posição {'LONG' if is_long else 'SHORT'}. Ignorando sinal {action}.")
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
            fill = float(order.get('avgPrice', 0))
            if fill <= 0:
                fill = price  # Usa o preço passado como parâmetro
            if fill <= 0:
                logger.error("❌ Preço inválido! Não é possível abrir posição.")
                return False
            
            # FIX: Validar preço retornado
            fill = float(order.get('avgPrice', 0))
            if fill <= 0:
                fill = float(order.get('price', 0))
            if fill <= 0:
                fill = price  # Fallback para preço passado
            if fill <= 0:
                logger.error("❌ Preço de fill inválido! Abortando.")
                return False
            
            logger.info(f"✅ {side} {qty} @ ${fill:.2f}")
            
            self.position_manager.open_position(self.symbol, side, fill, qty, sl, tp)
            
            sl_side = 'SELL' if side == 'BUY' else 'BUY'
            await self.client.futures_create_order(symbol=self.symbol, side=sl_side, type='STOP_MARKET', stopPrice=round(sl, 2), closePosition=True)
            await self.client.futures_create_order(symbol=self.symbol, side=sl_side, type='TAKE_PROFIT_MARKET', stopPrice=round(tp, 2), closePosition=True)
            
            self.total_trades += 1
            await self.notifier.notify_trade_open(side, self.symbol, fill, qty, sl, tp)
            return True
        except Exception as e:
            logger.error(f"❌ Erro: {e}")
            return False

    async def _close_position(self, price: float, reason: str) -> bool:
        if not self.position_manager.has_position():
            # Tenta sincronizar para ter certeza
            binance_pos = await self._get_binance_position()
            if not binance_pos:
                return False
            # Sincroniza para poder fechar corretamente
            self.position_manager.sync_position(
                symbol=self.symbol,
                side=binance_pos['side'],
                entry_price=binance_pos['entryPrice'],
                quantity=abs(binance_pos['positionAmt'])
            )
        
        try:
            pos = self.position_manager.current_position
            entry = pos.entry_price
            await self.client.futures_cancel_all_open_orders(symbol=self.symbol)
            order = await self.client.futures_create_order(symbol=self.symbol, side='SELL' if pos.is_long else 'BUY', type='MARKET', quantity=pos.quantity)
            
            # FIX: Validar preço de fill
            fill = float(order.get('avgPrice', 0))
            if fill <= 0:
                fill = float(order.get('price', 0))
            if fill <= 0:
                fill = price
            
            pnl = self.position_manager.close_position(fill)
            self.risk_manager.update_daily_pnl(pnl)
            if pnl > 0: self.winning_trades += 1
            await self.notifier.notify_trade_close(self.symbol, entry, fill, pnl, reason)
            return True
        except Exception as e:
            logger.error(f"❌ Erro ao fechar: {e}")
            return False

    def enable(self): self.execution_enabled = True
    def disable(self): self.execution_enabled = False