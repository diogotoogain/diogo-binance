"""
BACKTESTER: Testa estratégias com dados históricos
"""
import logging
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger("Backtester")


@dataclass
class BacktestTrade:
    entry_time: datetime
    exit_time: datetime
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float


class Backtester:
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trades: List[BacktestTrade] = []
        self.position = None

    async def run(self, candles: List[dict], strategy) -> dict:
        """
        Roda backtest com candles históricos
        candles = [{'open': x, 'high': x, 'low': x, 'close': x, 'volume': x, 'timestamp': x}, ...]
        """
        logger.info(f"🔬 Iniciando backtest com {len(candles)} candles...")
        
        for candle in candles:
            price = candle['close']
            
            # Checar posição existente
            if self.position:
                # Simular SL/TP
                if self.position['side'] == 'LONG':
                    if price <= self.position['sl']:
                        self._close_position(price, candle['timestamp'], 'SL')
                    elif price >= self.position['tp']:
                        self._close_position(price, candle['timestamp'], 'TP')
                else:
                    if price >= self.position['sl']:
                        self._close_position(price, candle['timestamp'], 'SL')
                    elif price <= self.position['tp']:
                        self._close_position(price, candle['timestamp'], 'TP')
            
            # Pedir sinal da estratégia
            signal = strategy.analyze(price, candle.get('high'), candle.get('low'))
            
            if signal and signal.get('action') and not self.position:
                self._open_position(signal, price, candle['timestamp'])
        
        return self._calculate_metrics()

    def _open_position(self, signal: dict, price: float, timestamp):
        side = 'LONG' if signal['action'] == 'BUY' else 'SHORT'
        qty = (self.balance * 0.01) / price  # 1% of balance
        
        if side == 'LONG':
            sl = price * 0.99
            tp = price * 1.02
        else:
            sl = price * 1.01
            tp = price * 0.98
        
        self.position = {
            'side': side,
            'entry_price': price,
            'quantity': qty,
            'sl': sl,
            'tp': tp,
            'entry_time': timestamp
        }

    def _close_position(self, price: float, timestamp, reason: str):
        if not self.position:
            return
        
        p = self.position
        if p['side'] == 'LONG':
            pnl = (price - p['entry_price']) * p['quantity']
        else:
            pnl = (p['entry_price'] - price) * p['quantity']
        
        pnl_pct = (pnl / (p['entry_price'] * p['quantity'])) * 100
        self.balance += pnl
        
        trade = BacktestTrade(
            entry_time=p['entry_time'],
            exit_time=timestamp,
            side=p['side'],
            entry_price=p['entry_price'],
            exit_price=price,
            quantity=p['quantity'],
            pnl=pnl,
            pnl_percent=pnl_pct
        )
        self.trades.append(trade)
        self.position = None

    def _calculate_metrics(self) -> dict:
        if not self.trades:
            return {'error': 'Nenhum trade executado'}
        
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        
        total_pnl = sum(t.pnl for t in self.trades)
        win_rate = len(wins) / len(self.trades) * 100
        
        total_wins = sum(t.pnl for t in wins)
        total_losses = sum(t.pnl for t in losses)
        avg_win = total_wins / len(wins) if wins else 0
        avg_loss = total_losses / len(losses) if losses else 0
        profit_factor = abs(total_wins / total_losses) if total_losses != 0 else 0
        
        return {
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'final_balance': round(self.balance, 2),
            'return_pct': round((self.balance - self.initial_balance) / self.initial_balance * 100, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2)
        }
