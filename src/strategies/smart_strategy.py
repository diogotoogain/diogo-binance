from src.strategies.base_strategy import BaseStrategy
import math
import time

class SmartStrategy(BaseStrategy):
    """
    PreditorVWAP - Estratégia CRITERIOSA de VWAP
    
    REGRAS DE TRADER PROFISSIONAL:
    1. Só entra quando o preço ROMPE a banda com FORÇA (não só toca)
    2. Precisa de CONFIRMAÇÃO (ficar X segundos fora da banda)
    3. Cooldown entre sinais (evita overtrading)
    4. Só sinaliza na MUDANÇA de estado (não repete)
    """
    
    def __init__(self):
        super().__init__("PreditorVWAP")
        self.tick_prices = []
        self.window_size = 1000 
        self.cum_vol = 0
        self.cum_pv = 0 
        self.vwap = 0
        self.std_dev = 0
        
        # === CONTROLES DE TRADER PROFISSIONAL ===
        self.last_state = "NEUTRAL"  # NEUTRAL, OVERSOLD, OVERBOUGHT
        self.state_entry_time = 0
        self.last_signal_time = 0
        
        # === PARÂMETROS CRITERIOSOS ===
        self.confirmation_seconds = 3.0    # Precisa ficar 3s fora da banda
        self.min_deviation_percent = 0.15  # Precisa desviar 0.15% além da banda
        self.cooldown_seconds = 30.0       # 30s entre sinais
        self.band_multiplier = 2.0         # 2 desvios padrão

    async def on_tick(self, data: dict):
        price = float(data['price'])
        qty = float(data['quantity'])
        now = time.time()
        
        # Atualiza VWAP
        self.cum_pv += price * qty
        self.cum_vol += qty
        
        if self.cum_vol > 0:
            self.vwap = self.cum_pv / self.cum_vol
            
        # Atualiza janela de preços
        self.tick_prices.append(price)
        if len(self.tick_prices) > self.window_size:
            self.tick_prices.pop(0)
            
        # Calcula desvio padrão
        if len(self.tick_prices) > 50:  # Precisa de dados suficientes
            mean = sum(self.tick_prices) / len(self.tick_prices)
            variance = sum([((x - mean) ** 2) for x in self.tick_prices]) / len(self.tick_prices)
            self.std_dev = math.sqrt(variance)
        else:
            return None  # Dados insuficientes
        
        if self.std_dev == 0:
            return None
            
        # Calcula bandas
        upper_band = self.vwap + (self.band_multiplier * self.std_dev)
        lower_band = self.vwap - (self.band_multiplier * self.std_dev)
        
        # === LÓGICA DE TRADER PROFISSIONAL ===
        
        # Calcula desvio percentual ALÉM da banda
        deviation_below = (lower_band - price) / lower_band if lower_band > 0 else 0
        deviation_above = (price - upper_band) / upper_band if upper_band > 0 else 0
        
        # Determina estado atual (precisa romper com FORÇA)
        current_state = "NEUTRAL"
        if deviation_below > (self.min_deviation_percent / 100):
            current_state = "OVERSOLD"
        elif deviation_above > (self.min_deviation_percent / 100):
            current_state = "OVERBOUGHT"
        
        # Verifica cooldown global
        if now - self.last_signal_time < self.cooldown_seconds:
            # Atualiza estado mas não sinaliza
            if current_state != self.last_state:
                self.last_state = current_state
                self.state_entry_time = now
            return None
        
        # MUDOU de estado? Reseta o timer de confirmação
        if current_state != self.last_state:
            self.last_state = current_state
            self.state_entry_time = now
            return None  # Aguarda confirmação
        
        # === CONFIRMAÇÃO: Ficou X segundos no estado extremo? ===
        if current_state != "NEUTRAL":
            time_in_state = now - self.state_entry_time
            
            if time_in_state >= self.confirmation_seconds:
                # SINAL CONFIRMADO!
                self.last_signal_time = now
                self.state_entry_time = now + 9999  # Evita re-sinalizar
                
                if current_state == "OVERSOLD":
                    self.log(f"📉 OVERSOLD CONFIRMADO! Price: {price:.2f} | VWAP: {self.vwap:.2f} | Desvio: {deviation_below*100:.2f}%")
                    return {
                        "action": "BUY", 
                        "confidence": "HIGH", 
                        "reason": f"VWAP_Oversold_{deviation_below*100:.1f}%"
                    }
                elif current_state == "OVERBOUGHT":
                    self.log(f"📈 OVERBOUGHT CONFIRMADO! Price: {price:.2f} | VWAP: {self.vwap:.2f} | Desvio: {deviation_above*100:.2f}%")
                    return {
                        "action": "SELL", 
                        "confidence": "HIGH", 
                        "reason": f"VWAP_Overbought_{deviation_above*100:.1f}%"
                    }
        
        return None