from src.strategies.base_strategy import BaseStrategy
from collections import deque
import time


class VPINStrategy(BaseStrategy):
    """
    Estratégia VPIN (Volume-Synchronized Probability of Informed Trading).
    
    Detecta fluxo tóxico de traders informados em tempo real.
    - VPIN alto (> 0.7) = Smart money agindo = Sinal forte
    - VPIN baixo (< 0.3) = Mercado calmo = Não operar
    
    A matemática:
    VPIN = Σ|V_buy - V_sell| / Σ(V_buy + V_sell)
    
    Onde:
    - V_buy = Volume de compra agressiva (is_buyer_maker = False)
    - V_sell = Volume de venda agressiva (is_buyer_maker = True)
    - Σ = Soma sobre todos os buckets na janela
    """
    
    def __init__(
        self,
        bucket_size_usd: float = 100_000,      # Tamanho do bucket em USD
        n_buckets: int = 50,                    # Número de buckets na janela
        vpin_threshold_high: float = 0.7,       # Limite superior de VPIN
        vpin_threshold_low: float = 0.3,        # Limite inferior de VPIN
        signal_cooldown_seconds: int = 30,      # Cooldown entre sinais
    ):
        super().__init__("VPINDetector")
        
        # Parâmetros configuráveis
        self.bucket_size_usd = bucket_size_usd
        self.n_buckets = n_buckets
        self.vpin_threshold_high = vpin_threshold_high
        self.vpin_threshold_low = vpin_threshold_low
        self.signal_cooldown_seconds = signal_cooldown_seconds
        
        # Janela deslizante de buckets (cada bucket é um dict com buy_volume e sell_volume)
        self.buckets = deque(maxlen=n_buckets)
        
        # Bucket atual em construção
        self.current_bucket_buy_volume = 0.0
        self.current_bucket_sell_volume = 0.0
        self.current_bucket_total_volume = 0.0
        
        # Controle de cooldown
        self.last_signal_time = 0
        
        # VPIN atual
        self.current_vpin = 0.0
        
        self.log(f"📊 VPIN iniciado | Bucket: ${bucket_size_usd:,.0f} | Janela: {n_buckets} buckets")

    def _calculate_vpin(self) -> float:
        """
        Calcula o VPIN atual baseado nos buckets completos.
        
        VPIN = Σ|V_buy - V_sell| / Σ(V_buy + V_sell)
        """
        if len(self.buckets) < 10:
            return 0.0
        
        total_imbalance = 0.0
        total_volume = 0.0
        
        for bucket in self.buckets:
            buy_vol = bucket['buy_volume']
            sell_vol = bucket['sell_volume']
            total_volume += buy_vol + sell_vol
            total_imbalance += abs(buy_vol - sell_vol)
        
        if total_volume == 0:
            return 0.0
        
        # VPIN = soma dos desequilíbrios / volume total
        vpin = total_imbalance / total_volume
        return min(vpin, 1.0)  # Garantir que VPIN está entre 0 e 1

    def _detect_direction(self) -> str:
        """
        Detecta a direção do fluxo tóxico baseado nos últimos 5 buckets.
        
        Returns:
            "BUY" se fluxo de compra dominante
            "SELL" se fluxo de venda dominante
            "INDEFINIDO" se não há direção clara
        """
        if len(self.buckets) < 5:
            return "INDEFINIDO"
        
        recent_buckets = list(self.buckets)[-5:]
        buy_dominant = 0
        sell_dominant = 0
        
        for bucket in recent_buckets:
            if bucket['buy_volume'] > bucket['sell_volume']:
                buy_dominant += 1
            elif bucket['sell_volume'] > bucket['buy_volume']:
                sell_dominant += 1
        
        # Precisa de maioria clara (3+ de 5)
        if buy_dominant >= 3:
            return "BUY"
        elif sell_dominant >= 3:
            return "SELL"
        else:
            return "INDEFINIDO"

    def _can_signal(self) -> bool:
        """Verifica se está fora do período de cooldown."""
        current_time = time.time()
        return (current_time - self.last_signal_time) >= self.signal_cooldown_seconds

    def _complete_bucket(self):
        """
        Finaliza o bucket atual e adiciona à janela deslizante.
        """
        bucket = {
            'buy_volume': self.current_bucket_buy_volume,
            'sell_volume': self.current_bucket_sell_volume,
        }
        self.buckets.append(bucket)
        
        # Reset do bucket atual
        self.current_bucket_buy_volume = 0.0
        self.current_bucket_sell_volume = 0.0
        self.current_bucket_total_volume = 0.0
        
        # Recalcula VPIN
        self.current_vpin = self._calculate_vpin()
        
        # FIX: Só loga quando VPIN está alto E pode gerar sinal (fora do cooldown)
        if self.current_vpin >= self.vpin_threshold_high and self._can_signal():
            direction = self._detect_direction()
            self.log(f"🚨 VPIN ALTO: {self.current_vpin:.2%} | Direção: {direction}")

    async def on_tick(self, data: dict):
        """
        Processa cada tick e atualiza o VPIN.
        
        Args:
            data: Dict com price, quantity, is_buyer_maker, event_type
        
        Returns:
            Signal dict ou None
        """
        # Ignora eventos de liquidação
        if data.get('event_type') == 'liquidation':
            return None
        
        price = float(data['price'])
        quantity = float(data['quantity'])
        volume_usd = price * quantity
        
        # Classificação do trade:
        # is_buyer_maker = True -> Vendedor agrediu (volume de venda)
        # is_buyer_maker = False -> Comprador agrediu (volume de compra)
        if data['is_buyer_maker']:
            self.current_bucket_sell_volume += volume_usd
        else:
            self.current_bucket_buy_volume += volume_usd
        
        self.current_bucket_total_volume += volume_usd
        
        # Verifica se o bucket atual está cheio
        if self.current_bucket_total_volume >= self.bucket_size_usd:
            self._complete_bucket()
            
            # Gera sinal se VPIN alto e cooldown respeitado
            if self.current_vpin >= self.vpin_threshold_high and self._can_signal():
                direction = self._detect_direction()
                
                if direction == "BUY":
                    self.last_signal_time = time.time()
                    return {
                        "action": "BUY",
                        "confidence": "HIGH",
                        "reason": f"VPIN_Toxic_Buy_{self.current_vpin:.0%}",
                        "vpin": self.current_vpin,
                        "direction": "BUY"
                    }
                elif direction == "SELL":
                    self.last_signal_time = time.time()
                    return {
                        "action": "SELL",
                        "confidence": "HIGH",
                        "reason": f"VPIN_Toxic_Sell_{self.current_vpin:.0%}",
                        "vpin": self.current_vpin,
                        "direction": "SELL"
                    }
                else:
                    # Direção indefinida - não operar (perigo)
                    self.log(f"⚠️ VPIN alto mas direção INDEFINIDA - não operar")
        
        return None

    def get_status(self) -> dict:
        """
        Retorna o status atual da estratégia para monitoramento externo.
        
        Returns:
            Dict com informações de status
        """
        return {
            "name": self.name,
            "vpin": self.current_vpin,
            "buckets_filled": len(self.buckets),
            "buckets_required": self.n_buckets,
            "current_bucket_progress": self.current_bucket_total_volume / self.bucket_size_usd,
            "current_bucket_buy": self.current_bucket_buy_volume,
            "current_bucket_sell": self.current_bucket_sell_volume,
            "direction": self._detect_direction(),
            "is_vpin_high": self.current_vpin >= self.vpin_threshold_high,
            "is_vpin_low": self.current_vpin <= self.vpin_threshold_low,
            "can_signal": self._can_signal(),
            "threshold_high": self.vpin_threshold_high,
            "threshold_low": self.vpin_threshold_low,
        }
