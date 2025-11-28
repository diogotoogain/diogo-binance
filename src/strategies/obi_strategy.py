from src.strategies.base_strategy import BaseStrategy
from collections import deque
import time
import math


class OBIStrategy(BaseStrategy):
    """
    Order Book Imbalance com Decaimento Exponencial.
    OBI ponderado usa peso exponencial: w_i = e^(-λ × i)
    Ordens perto do preço = muito peso
    Ordens longe do preço = ignoradas (possível spoofing)
    """

    def __init__(
        self,
        depth_levels: int = 20,
        decay_lambda: float = 0.3,
        obi_threshold_buy: float = 0.15,
        obi_threshold_sell: float = -0.15,
        min_total_liquidity_btc: float = 50.0,
        cooldown_seconds: int = 20,
        divergence_lookback: int = 10,
    ):
        super().__init__("OBI")
        self.depth_levels = depth_levels
        self.decay_lambda = decay_lambda
        self.obi_threshold_buy = obi_threshold_buy
        self.obi_threshold_sell = obi_threshold_sell
        self.min_total_liquidity_btc = min_total_liquidity_btc
        self.cooldown_seconds = cooldown_seconds
        self.divergence_lookback = divergence_lookback

        # Histórico para detectar divergências: (timestamp, obi, price)
        self.obi_history = deque(maxlen=divergence_lookback)
        self.last_signal_time = 0
        self.last_obi = 0
        self.last_price = 0

    async def on_tick(self, data: dict):
        if data.get('event_type') != 'orderbook':
            return None

        current_time = time.time()

        bids = data.get('bids', [])
        asks = data.get('asks', [])

        if not bids or not asks:
            return None

        # Verifica cooldown
        if current_time - self.last_signal_time < self.cooldown_seconds:
            return None

        # Calcula OBI ponderado com decaimento exponencial
        weighted_bids = 0
        weighted_asks = 0
        total_bids = 0
        total_asks = 0

        # Processa bids (níveis mais altos = mais perto do preço)
        for i, bid in enumerate(bids[:self.depth_levels]):
            price = float(bid[0])
            qty = float(bid[1])
            weight = math.exp(-self.decay_lambda * i)
            weighted_bids += qty * weight
            total_bids += qty

        # Processa asks (níveis mais baixos = mais perto do preço)
        for i, ask in enumerate(asks[:self.depth_levels]):
            price = float(ask[0])
            qty = float(ask[1])
            weight = math.exp(-self.decay_lambda * i)
            weighted_asks += qty * weight
            total_asks += qty

        total_liquidity = total_bids + total_asks

        # Verifica liquidez mínima
        if total_liquidity < self.min_total_liquidity_btc:
            return None

        # Calcula OBI ponderado
        total_weighted = weighted_bids + weighted_asks
        if total_weighted == 0:
            return None

        obi = (weighted_bids - weighted_asks) / total_weighted

        # Obtém preço mid
        best_bid = float(bids[0][0]) if bids else 0
        best_ask = float(asks[0][0]) if asks else 0
        mid_price = (best_bid + best_ask) / 2 if best_bid and best_ask else 0

        # Armazena histórico
        self.obi_history.append((current_time, obi, mid_price))
        self.last_obi = obi
        self.last_price = mid_price

        signal = None

        # Detecta divergências
        divergence = self._detect_divergence()

        # OBI > threshold → BUY (mais bids que asks)
        if obi > self.obi_threshold_buy:
            self.last_signal_time = current_time
            self.log(f"📗 OBI POSITIVO! OBI: {obi:.2%} | Liq: {total_liquidity:.2f} BTC")
            signal = {
                "action": "BUY",
                "confidence": "HIGH",
                "reason": f"OBI_Buy_{obi:.0%}",
                "obi": obi,
                "weighted_bids": weighted_bids,
                "weighted_asks": weighted_asks,
                "total_liquidity_btc": total_liquidity,
                "mid_price": mid_price,
                "divergence": divergence,
            }

        # OBI < threshold → SELL (mais asks que bids)
        elif obi < self.obi_threshold_sell:
            self.last_signal_time = current_time
            self.log(f"📕 OBI NEGATIVO! OBI: {obi:.2%} | Liq: {total_liquidity:.2f} BTC")
            signal = {
                "action": "SELL",
                "confidence": "HIGH",
                "reason": f"OBI_Sell_{obi:.0%}",
                "obi": obi,
                "weighted_bids": weighted_bids,
                "weighted_asks": weighted_asks,
                "total_liquidity_btc": total_liquidity,
                "mid_price": mid_price,
                "divergence": divergence,
            }

        return signal

    def _detect_divergence(self) -> str:
        """Detecta divergências entre OBI e preço."""
        if len(self.obi_history) < self.divergence_lookback:
            return "insufficient_data"

        # Compara primeiro e último ponto
        first_obi = self.obi_history[0][1]
        last_obi = self.obi_history[-1][1]
        first_price = self.obi_history[0][2]
        last_price = self.obi_history[-1][2]

        obi_direction = "up" if last_obi > first_obi else "down"
        price_direction = "up" if last_price > first_price else "down"

        if obi_direction != price_direction:
            return f"divergence_{obi_direction}_obi_{price_direction}_price"

        return "none"

    def get_status(self) -> dict:
        return {
            "name": self.name,
            "last_obi": self.last_obi,
            "last_price": self.last_price,
            "history_size": len(self.obi_history),
            "depth_levels": self.depth_levels,
            "decay_lambda": self.decay_lambda,
            "obi_threshold_buy": self.obi_threshold_buy,
            "obi_threshold_sell": self.obi_threshold_sell,
            "min_total_liquidity_btc": self.min_total_liquidity_btc,
            "cooldown_seconds": self.cooldown_seconds,
        }
