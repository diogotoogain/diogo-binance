"""
Script para rodar backtest
Uso: python scripts/run_backtest.py
"""
import asyncio
import sys
sys.path.insert(0, '.')

from src.backtest import Backtester


class SimpleStrategy:
    """
    Estratégia simples de exemplo para backtest.
    Compra quando preço cai abaixo da banda inferior e vende quando sobe acima da banda superior.
    """
    def __init__(self):
        self.prices = []
        self.window = 20
    
    def analyze(self, price: float, high: float = None, low: float = None) -> dict:
        self.prices.append(price)
        if len(self.prices) > self.window:
            self.prices.pop(0)
        
        if len(self.prices) < self.window:
            return None
        
        avg = sum(self.prices) / len(self.prices)
        std = (sum((p - avg) ** 2 for p in self.prices) / len(self.prices)) ** 0.5
        
        if std == 0:
            return None
        
        upper = avg + (2 * std)
        lower = avg - (2 * std)
        
        if price < lower:
            return {"action": "BUY", "reason": "Price below lower band"}
        elif price > upper:
            return {"action": "SELL", "reason": "Price above upper band"}
        
        return None


async def main():
    # Dados de exemplo (substituir por dados reais)
    candles = [
        {'open': 90000, 'high': 91000, 'low': 89500, 'close': 90500, 'timestamp': '2024-01-01'},
        {'open': 90500, 'high': 91500, 'low': 90000, 'close': 91000, 'timestamp': '2024-01-02'},
        {'open': 91000, 'high': 92000, 'low': 90500, 'close': 91500, 'timestamp': '2024-01-03'},
        {'open': 91500, 'high': 92500, 'low': 91000, 'close': 92000, 'timestamp': '2024-01-04'},
        {'open': 92000, 'high': 93000, 'low': 91500, 'close': 92500, 'timestamp': '2024-01-05'},
        {'open': 92500, 'high': 93500, 'low': 92000, 'close': 93000, 'timestamp': '2024-01-06'},
        {'open': 93000, 'high': 94000, 'low': 92500, 'close': 93500, 'timestamp': '2024-01-07'},
        {'open': 93500, 'high': 94500, 'low': 93000, 'close': 94000, 'timestamp': '2024-01-08'},
        {'open': 94000, 'high': 95000, 'low': 93500, 'close': 94500, 'timestamp': '2024-01-09'},
        {'open': 94500, 'high': 95500, 'low': 94000, 'close': 95000, 'timestamp': '2024-01-10'},
        {'open': 95000, 'high': 96000, 'low': 94500, 'close': 95500, 'timestamp': '2024-01-11'},
        {'open': 95500, 'high': 96500, 'low': 95000, 'close': 96000, 'timestamp': '2024-01-12'},
        {'open': 96000, 'high': 97000, 'low': 95500, 'close': 96500, 'timestamp': '2024-01-13'},
        {'open': 96500, 'high': 97500, 'low': 96000, 'close': 97000, 'timestamp': '2024-01-14'},
        {'open': 97000, 'high': 98000, 'low': 96500, 'close': 97500, 'timestamp': '2024-01-15'},
        {'open': 97500, 'high': 98500, 'low': 97000, 'close': 98000, 'timestamp': '2024-01-16'},
        {'open': 98000, 'high': 99000, 'low': 97500, 'close': 98500, 'timestamp': '2024-01-17'},
        {'open': 98500, 'high': 99500, 'low': 98000, 'close': 99000, 'timestamp': '2024-01-18'},
        {'open': 99000, 'high': 100000, 'low': 98500, 'close': 99500, 'timestamp': '2024-01-19'},
        {'open': 99500, 'high': 100500, 'low': 99000, 'close': 100000, 'timestamp': '2024-01-20'},
        {'open': 100000, 'high': 101000, 'low': 99500, 'close': 100500, 'timestamp': '2024-01-21'},
        {'open': 100500, 'high': 101500, 'low': 100000, 'close': 101000, 'timestamp': '2024-01-22'},
        {'open': 101000, 'high': 102000, 'low': 100500, 'close': 101500, 'timestamp': '2024-01-23'},
        {'open': 101500, 'high': 102500, 'low': 101000, 'close': 102000, 'timestamp': '2024-01-24'},
        {'open': 102000, 'high': 103000, 'low': 101500, 'close': 102500, 'timestamp': '2024-01-25'},
        {'open': 102500, 'high': 103500, 'low': 102000, 'close': 98000, 'timestamp': '2024-01-26'},
        {'open': 98000, 'high': 99000, 'low': 96000, 'close': 96500, 'timestamp': '2024-01-27'},
        {'open': 96500, 'high': 98000, 'low': 95000, 'close': 97000, 'timestamp': '2024-01-28'},
        {'open': 97000, 'high': 99000, 'low': 96500, 'close': 98500, 'timestamp': '2024-01-29'},
        {'open': 98500, 'high': 100000, 'low': 98000, 'close': 99500, 'timestamp': '2024-01-30'},
    ]
    
    strategy = SimpleStrategy()
    backtester = Backtester(initial_balance=10000)
    
    results = await backtester.run(candles, strategy)
    
    print("\n" + "=" * 50)
    print("       📊 RESULTADO DO BACKTEST")
    print("=" * 50)
    for k, v in results.items():
        print(f"  {k}: {v}")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
