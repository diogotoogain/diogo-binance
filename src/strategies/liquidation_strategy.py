from src.strategies.base_strategy import BaseStrategy
from datetime import datetime

class LiquidationStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("CacadorLiquidez")
        self.min_liquidation_usd = 10000 

    async def on_tick(self, data: dict):
        if data.get('event_type') != 'liquidation':
            return None

        side = data['side'] 
        amount = float(data['amount_usd'])
        
        if amount >= self.min_liquidation_usd:
            self.log(f"💀 BALEIA LIQUIDADA! Lado: {side} | Valor: ${amount:,.2f}")
            
            if side == 'SELL': 
                return {"action": "BUY", "confidence": "MAX", "reason": f"Liq_Long_Reversal_${amount:.0f}"}
            elif side == 'BUY': 
                return {"action": "SELL", "confidence": "MAX", "reason": f"Liq_Short_Reversal_${amount:.0f}"}
        
        return None