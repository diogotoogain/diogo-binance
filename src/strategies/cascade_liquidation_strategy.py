from src.strategies.base_strategy import BaseStrategy
from collections import deque
import time


class CascadeLiquidationStrategy(BaseStrategy):
    """
    Estratégia para detectar CASCATAS de liquidações (múltiplas liquidações em sequência rápida).
    Cascata de LONGS liquidados → Sinal de COMPRA (fundo local)
    Cascata de SHORTS liquidados → Sinal de VENDA (topo local)
    """

    def __init__(
        self,
        window_seconds: float = 5.0,
        min_cascade_usd: float = 100_000,
        min_liquidations: int = 3,
        direction_threshold: float = 0.7,
        cooldown_seconds: int = 60,
    ):
        super().__init__("CascadeLiquidation")
        self.window_seconds = window_seconds
        self.min_cascade_usd = min_cascade_usd
        self.min_liquidations = min_liquidations
        self.direction_threshold = direction_threshold
        self.cooldown_seconds = cooldown_seconds

        # Janela deslizante de liquidações: (timestamp, side, amount_usd)
        self.liquidations = deque()
        self.last_signal_time = 0

    async def on_tick(self, data: dict):
        if data.get('event_type') != 'liquidation':
            return None

        current_time = time.time()
        timestamp = data.get('timestamp', current_time * 1000) / 1000  # Convert to seconds
        side = data['side']
        amount_usd = float(data['amount_usd'])

        # Adiciona a liquidação atual
        self.liquidations.append((timestamp, side, amount_usd))

        # Remove liquidações fora da janela
        cutoff_time = current_time - self.window_seconds
        while self.liquidations and self.liquidations[0][0] < cutoff_time:
            self.liquidations.popleft()

        # Verifica cooldown
        if current_time - self.last_signal_time < self.cooldown_seconds:
            return None

        # Verifica número mínimo de liquidações
        if len(self.liquidations) < self.min_liquidations:
            return None

        # Calcula volumes por direção
        long_volume = 0  # SELL side = longs sendo liquidados
        short_volume = 0  # BUY side = shorts sendo liquidados

        for _, liq_side, liq_amount in self.liquidations:
            if liq_side == 'SELL':
                long_volume += liq_amount
            elif liq_side == 'BUY':
                short_volume += liq_amount

        total_volume = long_volume + short_volume

        # Verifica volume mínimo
        if total_volume < self.min_cascade_usd:
            return None

        # Calcula direção dominante
        long_ratio = long_volume / total_volume if total_volume > 0 else 0
        short_ratio = short_volume / total_volume if total_volume > 0 else 0

        signal = None

        # Cascata de LONGS liquidados → BUY (fundo local)
        if long_ratio >= self.direction_threshold:
            self.last_signal_time = current_time
            self.log(f"🔥 CASCATA LONG! Vol: ${total_volume:,.0f} | Liqs: {len(self.liquidations)}")
            signal = {
                "action": "BUY",
                "confidence": "MAX",
                "reason": f"Cascade_Long_Liq_${total_volume:.0f}",
                "total_volume_usd": total_volume,
                "num_liquidations": len(self.liquidations),
                "long_ratio": long_ratio,
            }

        # Cascata de SHORTS liquidados → SELL (topo local)
        elif short_ratio >= self.direction_threshold:
            self.last_signal_time = current_time
            self.log(f"❄️ CASCATA SHORT! Vol: ${total_volume:,.0f} | Liqs: {len(self.liquidations)}")
            signal = {
                "action": "SELL",
                "confidence": "MAX",
                "reason": f"Cascade_Short_Liq_${total_volume:.0f}",
                "total_volume_usd": total_volume,
                "num_liquidations": len(self.liquidations),
                "short_ratio": short_ratio,
            }

        return signal

    def get_status(self) -> dict:
        return {
            "name": self.name,
            "liquidations_in_window": len(self.liquidations),
            "window_seconds": self.window_seconds,
            "min_cascade_usd": self.min_cascade_usd,
            "min_liquidations": self.min_liquidations,
            "direction_threshold": self.direction_threshold,
            "cooldown_seconds": self.cooldown_seconds,
        }
