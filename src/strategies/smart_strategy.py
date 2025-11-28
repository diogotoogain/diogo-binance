from src.strategies.base_strategy import BaseStrategy
import math

class SmartStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("PreditorVWAP")
        self.tick_prices = []
        self.window_size = 1000 
        self.cum_vol = 0
        self.cum_pv = 0 
        self.vwap = 0
        self.std_dev = 0 

    async def on_tick(self, data: dict):
        price = float(data['price'])
        qty = float(data['quantity'])
        
        self.cum_pv += price * qty
        self.cum_vol += qty
        
        if self.cum_vol > 0:
            self.vwap = self.cum_pv / self.cum_vol
            
        self.tick_prices.append(price)
        if len(self.tick_prices) > self.window_size:
            self.tick_prices.pop(0)
            
        if len(self.tick_prices) > 10:
            mean = sum(self.tick_prices) / len(self.tick_prices)
            variance = sum([((x - mean) ** 2) for x in self.tick_prices]) / len(self.tick_prices)
            self.std_dev = math.sqrt(variance)
        
        if self.std_dev > 0:
            upper_band = self.vwap + (2 * self.std_dev)
            lower_band = self.vwap - (2 * self.std_dev)

            if price > upper_band:
                return {"action": "SELL", "confidence": "HIGH", "reason": "Overbought_VWAP"}
            elif price < lower_band:
                return {"action": "BUY", "confidence": "HIGH", "reason": "Oversold_VWAP"}
            
        return None