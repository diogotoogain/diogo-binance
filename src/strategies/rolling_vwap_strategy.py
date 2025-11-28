from src.strategies.base_strategy import BaseStrategy
from collections import deque
import time
import math


class RollingVWAPStrategy(BaseStrategy):
    """
    VWAP com janela deslizante (Rolling VWAP), não acumulado infinito.
    Preço > VWAP + 2σ → SELL (sobrecomprado)
    Preço < VWAP - 2σ → BUY (sobrevendido)
    """

    def __init__(
        self,
        window_seconds: float = 300.0,
        std_dev_periods: float = 2.0,
        min_trades: int = 50,
        cooldown_seconds: int = 30,
    ):
        super().__init__("RollingVWAP")
        self.window_seconds = window_seconds
        self.std_dev_periods = std_dev_periods
        self.min_trades = min_trades
        self.cooldown_seconds = cooldown_seconds

        # Janela deslizante: (timestamp, price, volume)
        self.ticks = deque()
        self.last_signal_time = 0

    async def on_tick(self, data: dict):
        if data.get('event_type') == 'liquidation':
            return None

        current_time = time.time()
        timestamp = data.get('timestamp', current_time * 1000) / 1000  # Convert to seconds

        price = float(data['price'])
        quantity = float(data['quantity'])

        # Adiciona o tick atual
        self.ticks.append((timestamp, price, quantity))

        # Remove ticks fora da janela
        cutoff_time = current_time - self.window_seconds
        while self.ticks and self.ticks[0][0] < cutoff_time:
            self.ticks.popleft()

        # Verifica mínimo de trades
        if len(self.ticks) < self.min_trades:
            return None

        # Verifica cooldown
        if current_time - self.last_signal_time < self.cooldown_seconds:
            return None

        # Calcula Rolling VWAP
        cum_pv = 0
        cum_vol = 0
        prices = []

        for _, tick_price, tick_vol in self.ticks:
            cum_pv += tick_price * tick_vol
            cum_vol += tick_vol
            prices.append(tick_price)

        if cum_vol == 0:
            return None

        vwap = cum_pv / cum_vol

        # Calcula desvio padrão dos preços
        mean_price = sum(prices) / len(prices)
        variance = sum([(p - mean_price) ** 2 for p in prices]) / len(prices)
        std_dev = math.sqrt(variance)

        if std_dev == 0:
            return None

        # Calcula bandas
        upper_band = vwap + (self.std_dev_periods * std_dev)
        lower_band = vwap - (self.std_dev_periods * std_dev)

        signal = None

        # Sobrecomprado → SELL
        if price > upper_band:
            self.last_signal_time = current_time
            self.log(f"📈 SOBRECOMPRADO! Price: {price:.2f} > Upper: {upper_band:.2f}")
            signal = {
                "action": "SELL",
                "confidence": "HIGH",
                "reason": f"VWAP_Overbought_{price:.0f}>{upper_band:.0f}",
                "price": price,
                "vwap": vwap,
                "upper_band": upper_band,
                "lower_band": lower_band,
                "std_dev": std_dev,
            }

        # Sobrevendido → BUY
        elif price < lower_band:
            self.last_signal_time = current_time
            self.log(f"📉 SOBREVENDIDO! Price: {price:.2f} < Lower: {lower_band:.2f}")
            signal = {
                "action": "BUY",
                "confidence": "HIGH",
                "reason": f"VWAP_Oversold_{price:.0f}<{lower_band:.0f}",
                "price": price,
                "vwap": vwap,
                "upper_band": upper_band,
                "lower_band": lower_band,
                "std_dev": std_dev,
            }

        return signal

    def get_status(self) -> dict:
        cum_pv = 0
        cum_vol = 0
        for _, tick_price, tick_vol in self.ticks:
            cum_pv += tick_price * tick_vol
            cum_vol += tick_vol
        vwap = cum_pv / cum_vol if cum_vol > 0 else 0

        return {
            "name": self.name,
            "ticks_in_window": len(self.ticks),
            "current_vwap": vwap,
            "window_seconds": self.window_seconds,
            "std_dev_periods": self.std_dev_periods,
            "min_trades": self.min_trades,
            "cooldown_seconds": self.cooldown_seconds,
        }
