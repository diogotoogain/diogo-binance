"""
SNAME-MR Sync Manager
Sincroniza estado local com dados reais da Binance.
"""
import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger("SyncManager")


class SyncManager:
    """Gerencia sincronização entre estado local e Binance."""
    
    def __init__(self, client, symbol: str = "BTCUSDT"):
        self.client = client
        self.symbol = symbol
        self._last_sync = None
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Busca posições abertas na Binance.
        
        Returns:
            Lista de posições com informações formatadas
        """
        try:
            positions = await self.client.futures_position_information(symbol=self.symbol)
            result = []
            for pos in positions:
                qty = float(pos.get('positionAmt', 0))
                if qty != 0:
                    entry_price = float(pos.get('entryPrice', 0))
                    mark_price = float(pos.get('markPrice', 0))
                    unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                    
                    # Calcular ROE%
                    roe_percent = 0.0
                    if entry_price > 0 and qty != 0:
                        notional = entry_price * abs(qty)
                        if notional > 0:
                            roe_percent = (unrealized_pnl / notional) * 100
                    
                    result.append({
                        'symbol': pos.get('symbol'),
                        'side': 'LONG' if qty > 0 else 'SHORT',
                        'size': abs(qty),
                        'entry_price': entry_price,
                        'mark_price': mark_price,
                        'liquidation_price': float(pos.get('liquidationPrice', 0)),
                        'unrealized_pnl': unrealized_pnl,
                        'roe_percent': roe_percent,
                        'leverage': int(pos.get('leverage', 1)),
                        'margin_type': pos.get('marginType', 'cross')
                    })
            return result
        except Exception as e:
            logger.error(f"❌ Erro ao buscar posições: {e}")
            return []
    
    async def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Busca ordens abertas (SL, TP pendentes).
        
        Returns:
            Lista de ordens abertas formatadas
        """
        try:
            orders = await self.client.futures_get_open_orders(symbol=self.symbol)
            result = []
            for order in orders:
                result.append({
                    'order_id': order.get('orderId'),
                    'symbol': order.get('symbol'),
                    'side': order.get('side'),
                    'type': order.get('type'),
                    'price': float(order.get('price', 0)),
                    'stop_price': float(order.get('stopPrice', 0)),
                    'quantity': float(order.get('origQty', 0)),
                    'status': order.get('status'),
                    'time': order.get('time')
                })
            return result
        except Exception as e:
            logger.error(f"❌ Erro ao buscar ordens abertas: {e}")
            return []
    
    async def get_order_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Busca histórico de ordens.
        
        Returns:
            Lista de ordens históricas
        """
        try:
            orders = await self.client.futures_get_all_orders(symbol=self.symbol, limit=limit)
            result = []
            for order in orders:
                result.append({
                    'order_id': order.get('orderId'),
                    'symbol': order.get('symbol'),
                    'side': order.get('side'),
                    'type': order.get('type'),
                    'price': float(order.get('price', 0)),
                    'avg_price': float(order.get('avgPrice', 0)),
                    'quantity': float(order.get('origQty', 0)),
                    'executed_qty': float(order.get('executedQty', 0)),
                    'status': order.get('status'),
                    'time': order.get('time'),
                    'update_time': order.get('updateTime')
                })
            return sorted(result, key=lambda x: x.get('update_time', 0), reverse=True)
        except Exception as e:
            logger.error(f"❌ Erro ao buscar histórico de ordens: {e}")
            return []
    
    async def get_trade_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Busca histórico de trades executados.
        
        Returns:
            Lista de trades executados
        """
        try:
            trades = await self.client.futures_account_trades(symbol=self.symbol, limit=limit)
            result = []
            for trade in trades:
                result.append({
                    'trade_id': trade.get('id'),
                    'order_id': trade.get('orderId'),
                    'symbol': trade.get('symbol'),
                    'side': trade.get('side'),
                    'price': float(trade.get('price', 0)),
                    'qty': float(trade.get('qty', 0)),
                    'realized_pnl': float(trade.get('realizedPnl', 0)),
                    'commission': float(trade.get('commission', 0)),
                    'commission_asset': trade.get('commissionAsset'),
                    'time': trade.get('time'),
                    'buyer': trade.get('buyer', False),
                    'maker': trade.get('maker', False)
                })
            return sorted(result, key=lambda x: x.get('time', 0), reverse=True)
        except Exception as e:
            logger.error(f"❌ Erro ao buscar histórico de trades: {e}")
            return []
    
    async def get_transactions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Busca histórico de transações (income history).
        
        Returns:
            Lista de transações
        """
        try:
            transactions = await self.client.futures_income_history(symbol=self.symbol, limit=limit)
            result = []
            for tx in transactions:
                result.append({
                    'symbol': tx.get('symbol'),
                    'type': tx.get('incomeType'),
                    'income': float(tx.get('income', 0)),
                    'asset': tx.get('asset'),
                    'time': tx.get('time'),
                    'info': tx.get('info', '')
                })
            return sorted(result, key=lambda x: x.get('time', 0), reverse=True)
        except Exception as e:
            logger.error(f"❌ Erro ao buscar transações: {e}")
            return []
    
    async def get_assets(self) -> List[Dict[str, Any]]:
        """
        Busca saldos da carteira.
        
        Returns:
            Lista de assets com saldos
        """
        try:
            balances = await self.client.futures_account_balance()
            result = []
            for bal in balances:
                balance = float(bal.get('balance', 0))
                if balance > 0:
                    result.append({
                        'asset': bal.get('asset'),
                        'balance': balance,
                        'available': float(bal.get('availableBalance', 0)),
                        'cross_wallet': float(bal.get('crossWalletBalance', 0)),
                        'cross_unrealized_pnl': float(bal.get('crossUnPnl', 0))
                    })
            return result
        except Exception as e:
            logger.error(f"❌ Erro ao buscar assets: {e}")
            return []
    
    async def get_account_summary(self) -> Dict[str, Any]:
        """
        Busca resumo da conta.
        
        Returns:
            Resumo da conta com saldos e PnL
        """
        try:
            account = await self.client.futures_account()
            return {
                'total_wallet_balance': float(account.get('totalWalletBalance', 0)),
                'total_margin_balance': float(account.get('totalMarginBalance', 0)),
                'total_unrealized_profit': float(account.get('totalUnrealizedProfit', 0)),
                'available_balance': float(account.get('availableBalance', 0)),
                'max_withdraw_amount': float(account.get('maxWithdrawAmount', 0)),
                'total_position_initial_margin': float(account.get('totalPositionInitialMargin', 0)),
                'total_open_order_initial_margin': float(account.get('totalOpenOrderInitialMargin', 0)),
                'can_trade': account.get('canTrade', False),
                'can_deposit': account.get('canDeposit', False),
                'can_withdraw': account.get('canWithdraw', False)
            }
        except Exception as e:
            logger.error(f"❌ Erro ao buscar resumo da conta: {e}")
            return {}
    
    async def full_sync(self) -> Dict[str, Any]:
        """
        Sincronização completa de todos os dados.
        
        Returns:
            Dicionário com todos os dados sincronizados
        """
        try:
            positions = await self.get_positions()
            open_orders = await self.get_open_orders()
            account = await self.get_account_summary()
            assets = await self.get_assets()
            
            self._last_sync = {
                'positions': positions,
                'open_orders': open_orders,
                'account': account,
                'assets': assets
            }
            
            logger.info(f"🔄 Sync completo | Posições: {len(positions)} | Ordens: {len(open_orders)}")
            return self._last_sync
        except Exception as e:
            logger.error(f"❌ Erro na sincronização completa: {e}")
            return {}
