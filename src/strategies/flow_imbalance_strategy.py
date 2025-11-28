from src.strategies.base_strategy import BaseStrategy
from collections import deque
import time


class FlowImbalanceStrategy(BaseStrategy):
    """
    Versão corrigida do FluxoBrabo. Pondera por VOLUME EM USD, não por contagem de trades.
    is_buyer_maker = True → Vendedor agrediu (venda)
    is_buyer_maker = False → Comprador agrediu (compra)
    """

    def __init__(
        self,
        window_seconds: float = 30.0,
        ratio_threshold_buy: float = 0.60,
        ratio_threshold_sell: float = 0.40,
        min_volume_usd: float = 500_000,
        cooldown_seconds: int = 15,
    ):
        super().__init__("FlowImbalance")
        self.window_seconds = window_seconds
        self.ratio_threshold_buy = ratio_threshold_buy
        self.ratio_threshold_sell = ratio_threshold_sell
        self.min_volume_usd = min_volume_usd
        self.cooldown_seconds = cooldown_seconds

        # Janela deslizante: (timestamp, is_buy, volume_usd)
        self.trades = deque()
        self.last_signal_time = 0

    async def on_tick(self, data: dict):
        if data.get('event_type') == 'liquidation':
            return None

        current_time = time.time()
        timestamp = data.get('timestamp', current_time * 1000) / 1000  # Convert to seconds

        price = float(data['price'])
        quantity = float(data['quantity'])
        volume_usd = price * quantity

        # is_buyer_maker = False → Comprador agrediu (compra)
        is_buy = not data['is_buyer_maker']

        # Adiciona o trade atual
        self.trades.append((timestamp, is_buy, volume_usd))

        # Remove trades fora da janela
        cutoff_time = current_time - self.window_seconds
        while self.trades and self.trades[0][0] < cutoff_time:
            self.trades.popleft()

        # Verifica cooldown
        if current_time - self.last_signal_time < self.cooldown_seconds:
            return None

        # Calcula volumes
        buy_volume = 0
        sell_volume = 0

        for _, trade_is_buy, trade_volume in self.trades:
            if trade_is_buy:
                buy_volume += trade_volume
            else:
                sell_volume += trade_volume

        total_volume = buy_volume + sell_volume

        # Verifica volume mínimo
        if total_volume < self.min_volume_usd:
            return None

        # Calcula ratio
        buy_ratio = buy_volume / total_volume if total_volume > 0 else 0

        signal = None

        # Compra: ratio de compra > threshold
        if buy_ratio >= self.ratio_threshold_buy:
            self.last_signal_time = current_time
            self.log(f"🔥 FLUXO COMPRADOR! Ratio: {buy_ratio:.2%} | Vol: ${total_volume:,.0f}")
            signal = {
                "action": "BUY",
                "confidence": "HIGH",
                "reason": f"Flow_Buy_{buy_ratio:.0%}",
                "buy_ratio": buy_ratio,
                "total_volume_usd": total_volume,
                "buy_volume_usd": buy_volume,
                "sell_volume_usd": sell_volume,
            }

        # Venda: ratio de compra < threshold (ou seja, venda dominante)
        elif buy_ratio <= self.ratio_threshold_sell:
            self.last_signal_time = current_time
            self.log(f"❄️ FLUXO VENDEDOR! Ratio: {buy_ratio:.2%} | Vol: ${total_volume:,.0f}")
            signal = {
                "action": "SELL",
                "confidence": "HIGH",
                "reason": f"Flow_Sell_{buy_ratio:.0%}",
                "buy_ratio": buy_ratio,
                "total_volume_usd": total_volume,
                "buy_volume_usd": buy_volume,
                "sell_volume_usd": sell_volume,
            }

        return signal

    def get_status(self) -> dict:
        buy_vol = sum(v for _, is_buy, v in self.trades if is_buy)
        sell_vol = sum(v for _, is_buy, v in self.trades if not is_buy)
        total = buy_vol + sell_vol
        return {
            "name": self.name,
            "trades_in_window": len(self.trades),
            "buy_volume_usd": buy_vol,
            "sell_volume_usd": sell_vol,
            "current_ratio": buy_vol / total if total > 0 else 0,
            "window_seconds": self.window_seconds,
            "min_volume_usd": self.min_volume_usd,
            "ratio_threshold_buy": self.ratio_threshold_buy,
            "ratio_threshold_sell": self.ratio_threshold_sell,
        }
