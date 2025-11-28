"""
DASHBOARD TERMINAL: Visualização em tempo real
- Mostra preço atual, posição, P&L
- Atualiza a cada 5 segundos
- Mostra sinais recentes
"""
import os
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger("Dashboard")


class TerminalDashboard:
    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.current_price = 0.0
        self.position = None
        self.daily_pnl = 0.0
        self.signals_count = 0
        self.decisions_count = 0
        self.last_signal = None
        self.start_time = datetime.now()
        self.running = False

    async def start(self):
        self.running = True
        self.event_bus.subscribe('market_data', self._on_price)
        self.event_bus.subscribe('trade_signal', self._on_signal)
        self.event_bus.subscribe('position_update', self._on_position)
        asyncio.create_task(self._refresh_loop())
        logger.info("📊 Dashboard iniciado")

    async def stop(self):
        self.running = False

    async def _on_price(self, data):
        self.current_price = float(data.get('price', 0))

    async def _on_signal(self, signal):
        self.signals_count += 1
        self.last_signal = signal

    async def _on_position(self, data):
        self.position = data

    async def _refresh_loop(self):
        while self.running:
            await asyncio.sleep(5)
            self._render()

    def _render(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        uptime = datetime.now() - self.start_time
        
        print("=" * 60)
        print("       🤖 SNAME-MR TRADING BOT - LIVE DASHBOARD")
        print("=" * 60)
        print(f"  ⏱️  Uptime: {str(uptime).split('.')[0]}")
        print(f"  💵 BTC/USDT: ${self.current_price:,.2f}")
        print(f"  📊 Sinais recebidos: {self.signals_count}")
        print("-" * 60)
        
        if self.position and self.position.get('has_position'):
            p = self.position
            print(f"  📍 POSIÇÃO: {p['side']} {p['quantity']} @ ${p['entry_price']:,.2f}")
            print(f"     🛑 SL: ${p['stop_loss']:,.2f} | 🎯 TP: ${p['take_profit']:,.2f}")
        else:
            print("  📍 POSIÇÃO: Nenhuma")
        
        print("-" * 60)
        print(f"  💰 P&L Diário: ${self.daily_pnl:,.2f}")
        
        if self.last_signal:
            print("-" * 60)
            print(f"  📨 Último sinal: {self.last_signal.get('action')} | {self.last_signal.get('reason', '')[:40]}")
        
        print("=" * 60)
        print("  💡 Ctrl+C para parar")
