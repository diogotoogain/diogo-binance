"""
DASHBOARD TERMINAL: Visualização em tempo real
- Mostra preço atual, posição, P&L
- Atualiza a cada 5 segundos
- Mostra sinais recentes
- Sincronizado com Binance
"""
import os
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger("Dashboard")


class TerminalDashboard:
    def __init__(self, event_bus, connector=None, position_manager=None, risk_manager=None):
        self.event_bus = event_bus
        self.connector = connector
        self.position_manager = position_manager
        self.risk_manager = risk_manager
        
        self.current_price = 0.0
        self.position = None
        self.account_data = {
            'wallet_balance': 0.0,
            'margin_balance': 0.0,
            'unrealized_pnl': 0.0,
            'unrealized_pnl_percent': 0.0
        }
        self.daily_pnl = 0.0
        self.signals_count = 0
        self.decisions_count = 0
        self.last_signal = None
        self.open_orders_count = 0
        self.start_time = datetime.now()
        self.running = False

    async def start(self):
        self.running = True
        self.event_bus.subscribe('market_data', self._on_price)
        self.event_bus.subscribe('trade_signal', self._on_signal)
        self.event_bus.subscribe('position_update', self._on_position)
        asyncio.create_task(self._refresh_loop())
        logger.info("📊 Dashboard Terminal iniciado")

    async def stop(self):
        self.running = False

    async def _on_price(self, data):
        self.current_price = float(data.get('price', 0))

    async def _on_signal(self, signal):
        self.signals_count += 1
        self.last_signal = signal

    async def _on_position(self, data):
        self.position = data

    async def _sync_binance_data(self):
        """Sincroniza dados com Binance se disponível."""
        if not self.connector or not self.connector.client:
            return
        
        try:
            # Buscar conta
            account = await self.connector.client.futures_account()
            self.account_data['wallet_balance'] = float(account.get('totalWalletBalance', 0))
            self.account_data['margin_balance'] = float(account.get('totalMarginBalance', 0))
            self.account_data['unrealized_pnl'] = float(account.get('totalUnrealizedProfit', 0))
            
            if self.account_data['wallet_balance'] > 0:
                self.account_data['unrealized_pnl_percent'] = (
                    self.account_data['unrealized_pnl'] / self.account_data['wallet_balance']
                ) * 100
            
            # Buscar posição
            positions = await self.connector.client.futures_position_information(symbol='BTCUSDT')
            for pos in positions:
                qty = float(pos.get('positionAmt', 0))
                if qty != 0:
                    entry_price = float(pos.get('entryPrice', 0))
                    unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                    
                    pnl_percent = 0.0
                    if entry_price > 0 and qty != 0:
                        notional = entry_price * abs(qty)
                        if notional > 0:
                            pnl_percent = (unrealized_pnl / notional) * 100
                    
                    self.position = {
                        'has_position': True,
                        'side': 'LONG' if qty > 0 else 'SHORT',
                        'quantity': abs(qty),
                        'entry_price': entry_price,
                        'mark_price': float(pos.get('markPrice', 0)),
                        'unrealized_pnl': unrealized_pnl,
                        'pnl_percent': pnl_percent,
                        'stop_loss': float(pos.get('liquidationPrice', 0)),
                        'take_profit': 0.0  # Será preenchido por ordens abertas
                    }
                    break
            else:
                self.position = {'has_position': False}
            
            # Buscar ordens abertas
            orders = await self.connector.client.futures_get_open_orders(symbol='BTCUSDT')
            self.open_orders_count = len(orders)
            
            # Atualizar SL/TP da posição baseado nas ordens
            if self.position and self.position.get('has_position'):
                for order in orders:
                    order_type = order.get('type', '')
                    stop_price = float(order.get('stopPrice', 0))
                    if 'STOP' in order_type and stop_price > 0:
                        self.position['stop_loss'] = stop_price
                    elif 'PROFIT' in order_type and stop_price > 0:
                        self.position['take_profit'] = stop_price
            
            # Daily PnL do RiskManager
            if self.risk_manager:
                self.daily_pnl = self.risk_manager.daily_pnl
                
        except Exception as e:
            logger.error(f"Erro sincronização Binance: {e}")

    async def _refresh_loop(self):
        while self.running:
            await asyncio.sleep(5)
            await self._sync_binance_data()
            self._render()

    def _render(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        uptime = datetime.now() - self.start_time
        
        print("═" * 55)
        print("  🏦 SNAME-MR TRADING BOT - LIVE DASHBOARD")
        print("═" * 55)
        print(f"  ⏱️  Uptime: {str(uptime).split('.')[0]}")
        print(f"  📈 BTC/USDT: ${self.current_price:,.2f}")
        print(f"  📊 Sinais recebidos: {self.signals_count}")
        print("─" * 55)
        
        # Conta (Sincronizado com Binance)
        print("  💰 CONTA (Sincronizado com Binance)")
        print(f"  ├─ Wallet Balance:    ${self.account_data['wallet_balance']:,.2f}")
        print(f"  ├─ Margin Balance:    ${self.account_data['margin_balance']:,.2f}")
        
        pnl = self.account_data['unrealized_pnl']
        pnl_pct = self.account_data['unrealized_pnl_percent']
        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
        print(f"  └─ Unrealized PNL:    {pnl_emoji} ${pnl:,.2f} ({pnl_pct:+.2f}%)")
        print("─" * 55)
        
        # Posição
        print("  📍 POSIÇÃO ATUAL")
        if self.position and self.position.get('has_position'):
            p = self.position
            side_emoji = "🟢" if p['side'] == 'LONG' else "🔴"
            print(f"  ├─ Side:      {side_emoji} {p['side']}")
            
            qty = p.get('quantity', 0)
            entry = p.get('entry_price', 0)
            usd_value = qty * entry
            print(f"  ├─ Size:      {qty:.4f} BTC (${usd_value:,.2f} USDT)")
            print(f"  ├─ Entry:     ${entry:,.2f}")
            print(f"  ├─ Mark:      ${p.get('mark_price', 0):,.2f}")
            
            pos_pnl = p.get('unrealized_pnl', 0)
            pos_pnl_pct = p.get('pnl_percent', 0)
            pos_emoji = "🟢" if pos_pnl >= 0 else "🔴"
            print(f"  ├─ PNL:       {pos_emoji} ${pos_pnl:,.2f} ({pos_pnl_pct:+.2f}%)")
            
            sl = p.get('stop_loss', 0)
            tp = p.get('take_profit', 0)
            print(f"  ├─ Stop Loss: ${sl:,.2f}" if sl > 0 else "  ├─ Stop Loss: --")
            print(f"  └─ Take Profit: ${tp:,.2f}" if tp > 0 else "  └─ Take Profit: --")
        else:
            print("  └─ Nenhuma posição aberta")
        
        print("─" * 55)
        print(f"  📋 ORDENS ABERTAS: {self.open_orders_count}")
        print(f"  💰 P&L Diário: ${self.daily_pnl:,.2f}")
        
        if self.last_signal:
            action = self.last_signal.get('action', 'HOLD')
            reason = self.last_signal.get('reason', '')[:30]
            print(f"  📨 Último sinal: {action} | {reason}")
        
        print("═" * 55)
        print("  💡 Ctrl+C para parar | 🌐 Dashboard: http://localhost:8080")
