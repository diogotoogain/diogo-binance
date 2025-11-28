from src.strategies.base_strategy import BaseStrategy

class SentimentStrategy(BaseStrategy):
    def __init__(self):
        super().__init__("FluxoBrabo")
        self.balance_pressure = 0 # Positivo = Compra, Negativo = Venda
        self.threshold = 10 # Sensibilidade

    async def on_tick(self, data: dict):
        # is_buyer_maker = True -> VENDEDOR Agrediu (Preço cai)
        # is_buyer_maker = False -> COMPRADOR Agrediu (Preço sobe)
        
        if data['is_buyer_maker'] == False: 
            self.balance_pressure += 1 
        else:
            self.balance_pressure -= 1 

        # --- O GATILHO ---
        if self.balance_pressure >= self.threshold:
            self.log(f"🔥 FLUXO COMPRADOR! (+{self.balance_pressure})")
            self.balance_pressure = 0 
            return {"action": "BUY", "confidence": "HIGH", "reason": "Fluxo_Agressao_Compra"}
            
        elif self.balance_pressure <= -self.threshold:
            self.log(f"❄️ FLUXO VENDEDOR! ({self.balance_pressure})")
            self.balance_pressure = 0 
            return {"action": "SELL", "confidence": "HIGH", "reason": "Fluxo_Agressao_Venda"}
            
        return None