from src.strategies.base_strategy import BaseStrategy
import time

class SentimentStrategy(BaseStrategy):
    """
    FluxoBrabo - Detector de Agressão de Fluxo
    
    REGRAS DE TRADER PROFISSIONAL:
    1. Threshold alto para evitar ruído (50 trades, não 10)
    2. Pondera por VOLUME, não só contagem
    3. Cooldown entre sinais
    4. Confirmação: pressão precisa se manter
    5. Decay: pressão diminui com o tempo se não confirmada
    """
    
    def __init__(self):
        super().__init__("FluxoBrabo")
        
        # === PRESSÃO DE FLUXO ===
        self.buy_volume = 0.0
        self.sell_volume = 0.0
        self.trade_count_buy = 0
        self.trade_count_sell = 0
        
        # === CONTROLES DE TRADER PROFISSIONAL ===
        self.last_signal_time = 0
        self.window_start_time = 0
        
        # === PARÂMETROS CRITERIOSOS ===
        self.threshold_ratio = 0.70        # 70% do volume em uma direção
        self.min_trades = 50               # Mínimo de 50 trades na janela
        self.min_volume_usd = 100_000      # Mínimo $100k de volume
        self.window_seconds = 10.0         # Janela de 10 segundos
        self.cooldown_seconds = 30.0       # 30s entre sinais

    async def on_tick(self, data: dict):
        now = time.time()
        price = float(data['price'])
        qty = float(data['quantity'])
        volume_usd = price * qty
        
        # Reseta janela se passou muito tempo
        if self.window_start_time == 0:
            self.window_start_time = now
        
        if now - self.window_start_time > self.window_seconds:
            # Nova janela - reseta contadores
            self.buy_volume = 0.0
            self.sell_volume = 0.0
            self.trade_count_buy = 0
            self.trade_count_sell = 0
            self.window_start_time = now
        
        # Acumula volume
        # is_buyer_maker = True -> VENDEDOR Agrediu (Market Sell)
        # is_buyer_maker = False -> COMPRADOR Agrediu (Market Buy)
        if data['is_buyer_maker'] == False:
            self.buy_volume += volume_usd
            self.trade_count_buy += 1
        else:
            self.sell_volume += volume_usd
            self.trade_count_sell += 1
        
        total_volume = self.buy_volume + self.sell_volume
        total_trades = self.trade_count_buy + self.trade_count_sell
        
        # Verifica cooldown
        if now - self.last_signal_time < self.cooldown_seconds:
            return None
        
        # Verifica mínimos
        if total_trades < self.min_trades:
            return None
        if total_volume < self.min_volume_usd:
            return None
        
        # Calcula ratio
        buy_ratio = self.buy_volume / total_volume if total_volume > 0 else 0.5
        
        # === SINAL CRITERIOSO ===
        if buy_ratio >= self.threshold_ratio:
            self.last_signal_time = now
            self.log(f"🔥 FLUXO COMPRADOR FORTE! Ratio: {buy_ratio:.1%} | Vol: ${total_volume:,.0f} | Trades: {total_trades}")
            # Reseta após sinal
            self.buy_volume = 0
            self.sell_volume = 0
            self.trade_count_buy = 0
            self.trade_count_sell = 0
            self.window_start_time = now
            return {
                "action": "BUY", 
                "confidence": "HIGH", 
                "reason": f"Fluxo_Agressao_Compra_{buy_ratio:.0%}"
            }
            
        elif buy_ratio <= (1 - self.threshold_ratio):
            self.last_signal_time = now
            sell_ratio = 1 - buy_ratio
            self.log(f"❄️ FLUXO VENDEDOR FORTE! Ratio: {sell_ratio:.1%} | Vol: ${total_volume:,.0f} | Trades: {total_trades}")
            # Reseta após sinal
            self.buy_volume = 0
            self.sell_volume = 0
            self.trade_count_buy = 0
            self.trade_count_sell = 0
            self.window_start_time = now
            return {
                "action": "SELL", 
                "confidence": "HIGH", 
                "reason": f"Fluxo_Agressao_Venda_{sell_ratio:.0%}"
            }
            
        return None