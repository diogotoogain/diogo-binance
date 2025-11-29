"""
Liquidation Cascade Strategy.

Opera baseado em cascatas de liquidação.
ZERO hardcoded - todos os parâmetros vêm do config.
"""
from typing import Any, Dict, Optional
import time

from v2.src.strategies.base import Strategy, Signal, SignalDirection
from v2.src.strategies.throttling import Throttler
from v2.src.features.microstructure.liquidation_features import LiquidationFeatures


class LiquidationCascadeStrategy(Strategy):
    """
    Estratégia que opera cascatas de liquidação.

    Modos:
    1. FOLLOW: Segue a cascata (shorts sendo liquidados = compra)
    2. FADE: Opera reversão após cascata terminar

    Regras FOLLOW:
    - Detecta cascata de liquidações de shorts → Compra
    - Detecta cascata de liquidações de longs → Vende
    - SL apertado (cascatas são rápidas)
    - TP baseado em extensão típica de cascata

    Regras FADE:
    - Espera cascata terminar
    - Opera reversão
    - Mais arriscado mas maior R:R
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa a estratégia de cascata de liquidações.

        Args:
            config: Configuração da estratégia do YAML
        """
        # Get liquidations config section
        liq_config = config.get('liquidations', {})
        strategy_config = liq_config.get('strategy', {})

        enabled = strategy_config.get('enabled', True)
        super().__init__("LiquidationCascade", config, enabled)

        # Full config for LiquidationFeatures
        self._full_config = config

        # Strategy parameters
        self.follow_cascade = strategy_config.get('follow_cascade', True)
        self.min_cascade_size_usd = strategy_config.get('min_cascade_size_usd', 500000)

        self.fade_cascade = strategy_config.get('fade_cascade', False)
        self.fade_after_seconds = strategy_config.get('fade_after_seconds', 300)
        self.fade_min_cascade_size = strategy_config.get('fade_min_cascade_size', 2000000)

        # Throttling
        throttle_config = strategy_config.get('throttling', {})
        if not throttle_config:
            throttle_config = {
                'enabled': True,
                'max_trades_per_minute': 2,
                'cooldown_after_loss_seconds': 60
            }
        self.throttler = Throttler(throttle_config)

        # State tracking
        self._last_cascade_end_time: Optional[float] = None
        self._last_cascade_direction: int = 0
        self._last_cascade_volume: float = 0

        # Initialize liquidation features
        self.liq_features = LiquidationFeatures(config)

        self.log(
            f"Inicializado | Follow: {self.follow_cascade} | "
            f"Fade: {self.fade_cascade} | Min size: ${self.min_cascade_size_usd:,.0f}"
        )

    def _check_follow_signal(
        self,
        cascade_active: bool,
        cascade_direction: int,
        cascade_volume: float
    ) -> Optional[Signal]:
        """
        Verifica sinal de follow cascade.

        Args:
            cascade_active: Se cascata está ativa
            cascade_direction: Direção (1=longs liq, -1=shorts liq)
            cascade_volume: Volume USD

        Returns:
            Signal ou None
        """
        if not self.follow_cascade:
            return None

        if not cascade_active:
            return None

        if cascade_volume < self.min_cascade_size_usd:
            return None

        # Shorts sendo liquidados (cascade_direction = -1) → BUY
        # Longs sendo liquidados (cascade_direction = 1) → SELL
        if cascade_direction == -1:
            return Signal(
                direction=SignalDirection.BUY,
                strategy_name=self.name,
                confidence=min(1.0, cascade_volume / (self.min_cascade_size_usd * 2)),
                reason=f"Follow_Short_Cascade_${cascade_volume:,.0f}",
                metadata={
                    'mode': 'follow',
                    'cascade_direction': cascade_direction,
                    'cascade_volume': cascade_volume,
                }
            )
        elif cascade_direction == 1:
            return Signal(
                direction=SignalDirection.SELL,
                strategy_name=self.name,
                confidence=min(1.0, cascade_volume / (self.min_cascade_size_usd * 2)),
                reason=f"Follow_Long_Cascade_${cascade_volume:,.0f}",
                metadata={
                    'mode': 'follow',
                    'cascade_direction': cascade_direction,
                    'cascade_volume': cascade_volume,
                }
            )

        return None

    def _check_fade_signal(
        self,
        cascade_active: bool,
        cascade_direction: int,
        cascade_volume: float,
        current_time: float
    ) -> Optional[Signal]:
        """
        Verifica sinal de fade cascade (reversão).

        Args:
            cascade_active: Se cascata está ativa
            cascade_direction: Direção atual
            cascade_volume: Volume USD
            current_time: Timestamp atual

        Returns:
            Signal ou None
        """
        if not self.fade_cascade:
            return None

        # Se cascata acabou de terminar e volume foi significativo
        if cascade_active:
            # Atualiza estado da última cascata
            self._last_cascade_direction = cascade_direction
            self._last_cascade_volume = cascade_volume
            return None

        # Verifica se teve cascata recente
        if self._last_cascade_direction == 0:
            return None

        if self._last_cascade_volume < self.fade_min_cascade_size:
            return None

        # Verifica se já passou tempo suficiente para fade
        if self._last_cascade_end_time is None:
            self._last_cascade_end_time = current_time
            return None

        elapsed = current_time - self._last_cascade_end_time
        if elapsed < self.fade_after_seconds:
            return None

        # Gera sinal de reversão
        # Longs foram liquidados → agora BUY (reversão)
        # Shorts foram liquidados → agora SELL (reversão)
        direction = self._last_cascade_direction

        signal: Optional[Signal] = None
        if direction == 1:  # Longs foram liquidados → BUY reversão
            signal = Signal(
                direction=SignalDirection.BUY,
                strategy_name=self.name,
                confidence=0.7,
                reason=f"Fade_Long_Cascade_${self._last_cascade_volume:,.0f}",
                metadata={
                    'mode': 'fade',
                    'original_cascade_direction': direction,
                    'cascade_volume': self._last_cascade_volume,
                    'fade_after': elapsed,
                }
            )
        elif direction == -1:  # Shorts foram liquidados → SELL reversão
            signal = Signal(
                direction=SignalDirection.SELL,
                strategy_name=self.name,
                confidence=0.7,
                reason=f"Fade_Short_Cascade_${self._last_cascade_volume:,.0f}",
                metadata={
                    'mode': 'fade',
                    'original_cascade_direction': direction,
                    'cascade_volume': self._last_cascade_volume,
                    'fade_after': elapsed,
                }
            )

        # Reset state after generating signal
        self._last_cascade_direction = 0
        self._last_cascade_volume = 0
        self._last_cascade_end_time = None

        return signal

    def add_liquidation(self, liquidation: Dict[str, Any]) -> None:
        """
        Adiciona uma liquidação ao tracker interno.

        Args:
            liquidation: Dados da liquidação
        """
        self.liq_features.add_liquidation(liquidation)

    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        Gera sinal baseado em cascatas de liquidação.

        Args:
            market_data: Dict com:
                - liq_features: Dict de features de liquidação calculadas
                - price: Preço atual

        Returns:
            Signal ou None
        """
        if not self.enabled:
            return None

        # Verifica throttling
        if not self.throttler.can_trade():
            return None

        # Get liquidation features from market_data or calculate
        liq_features = market_data.get('liq_features')
        if liq_features is None:
            liq_features = self.liq_features.calculate()

        cascade_active = bool(liq_features.get('cascade_active', False))
        cascade_direction = int(liq_features.get('cascade_direction', 0))

        # Get cascade volume from market_data or calculate
        cascade_volume = market_data.get('cascade_volume')
        if cascade_volume is None:
            cascade_volume = self.liq_features.get_cascade_volume()

        current_time = time.time()

        # Track cascade end
        if not cascade_active and self._last_cascade_direction != 0:
            if self._last_cascade_end_time is None:
                self._last_cascade_end_time = current_time

        signal = None

        # Check follow mode first
        if self.follow_cascade:
            signal = self._check_follow_signal(
                cascade_active, cascade_direction, cascade_volume
            )

        # Check fade mode if no follow signal
        if signal is None and self.fade_cascade:
            signal = self._check_fade_signal(
                cascade_active, cascade_direction, cascade_volume, current_time
            )

        if signal:
            self._total_signals += 1
            self._last_signal_time = current_time
            self.throttler.record_trade()
            self.log(
                f"📈 SINAL: {signal.direction.value} | "
                f"Conf: {signal.confidence:.2f} | {signal.reason}"
            )

        return signal

    def get_status(self) -> Dict[str, Any]:
        """Retorna status estendido."""
        base_status = super().get_status()
        base_status.update({
            'follow_cascade': self.follow_cascade,
            'fade_cascade': self.fade_cascade,
            'min_cascade_size_usd': self.min_cascade_size_usd,
            'fade_min_cascade_size': self.fade_min_cascade_size,
            'liq_features_status': self.liq_features.get_status(),
            'throttler': self.throttler.get_status(),
        })
        return base_status
