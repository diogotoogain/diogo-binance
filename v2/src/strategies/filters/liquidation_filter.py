"""
Liquidation-based Trading Filter.

Pausa ou ajusta trades durante cascatas violentas.
ZERO hardcoded - todos os parâmetros vêm do config.
"""
from typing import Any, Dict, Optional, Tuple
import time

from v2.src.features.microstructure.liquidation_features import LiquidationFeatures


class LiquidationFilter:
    """
    Filtro baseado em liquidações.

    Funcionalidades:
    - Pausa todas as estratégias durante cascata extrema
    - Ajusta position size baseado em intensidade de liquidações
    - Previne entrar contra uma cascata em andamento
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o filtro de liquidações.

        Args:
            config: Configuração do YAML com seção 'liquidations.filter'
        """
        liq_config = config.get('liquidations', {})
        filter_config = liq_config.get('filter', {})

        self.enabled = filter_config.get('enabled', True)
        self.pause_during_cascade = filter_config.get('pause_during_cascade', True)
        self.pause_threshold_usd = filter_config.get('pause_threshold_usd', 5000000)
        self.resume_after_seconds = filter_config.get('resume_after_seconds', 120)

        # Initialize liquidation features tracker
        self.liq_features = LiquidationFeatures(config)

        # State
        self._pause_start_time: Optional[float] = None
        self._paused = False

    def add_liquidation(self, liquidation: Dict[str, Any]) -> None:
        """
        Adiciona uma liquidação ao tracker.

        Args:
            liquidation: Dados da liquidação
        """
        self.liq_features.add_liquidation(liquidation)

    def should_trade(
        self,
        signal: Optional[Any] = None,
        liq_features: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, float, str]:
        """
        Verifica se deve executar o trade.

        Args:
            signal: Sinal de trading (opcional, para verificar direção)
            liq_features: Features de liquidação pré-calculadas (opcional)

        Returns:
            Tuple[allowed, size_multiplier, reason]
            - allowed: Se o trade é permitido
            - size_multiplier: Multiplicador de tamanho (0.0-1.0)
            - reason: Motivo da decisão
        """
        if not self.enabled:
            return (True, 1.0, "filter_disabled")

        # Get or calculate liquidation features
        if liq_features is None:
            liq_features = self.liq_features.calculate()

        current_time = time.time()

        # Check if we're in a pause state
        if self._paused:
            elapsed = current_time - (self._pause_start_time or current_time)
            if elapsed < self.resume_after_seconds:
                return (
                    False,
                    0.0,
                    f"cascade_pause_active_{elapsed:.0f}s/{self.resume_after_seconds}s"
                )
            else:
                # Resume trading
                self._paused = False
                self._pause_start_time = None

        cascade_active = bool(liq_features.get('cascade_active', False))
        cascade_volume = self.liq_features.get_cascade_volume()

        # Check for extreme cascade
        if self.pause_during_cascade and cascade_active:
            if cascade_volume >= self.pause_threshold_usd:
                self._paused = True
                self._pause_start_time = current_time
                return (
                    False,
                    0.0,
                    f"extreme_cascade_detected_${cascade_volume:,.0f}"
                )

        # Check for moderate cascade - reduce size
        if cascade_active and cascade_volume > 0:
            # Reduce size based on cascade intensity
            intensity = min(1.0, cascade_volume / self.pause_threshold_usd)
            size_mult = max(0.25, 1.0 - (intensity * 0.75))
            return (
                True,
                size_mult,
                f"cascade_size_reduction_{size_mult:.2f}"
            )

        # Check for direction conflict
        if signal is not None and cascade_active:
            cascade_direction = int(liq_features.get('cascade_direction', 0))
            signal_direction = getattr(signal, 'direction', None)

            if signal_direction is not None:
                # Prevent entering against cascade
                # cascade_direction = 1 means longs being liquidated (bearish)
                # cascade_direction = -1 means shorts being liquidated (bullish)
                is_buy = signal_direction.value == 'BUY' if hasattr(signal_direction, 'value') else str(signal_direction).upper() == 'BUY'

                if cascade_direction == 1 and is_buy:
                    return (
                        False,
                        0.0,
                        "signal_against_long_cascade"
                    )
                elif cascade_direction == -1 and not is_buy:
                    return (
                        False,
                        0.0,
                        "signal_against_short_cascade"
                    )

        return (True, 1.0, "allowed")

    def get_status(self) -> Dict[str, Any]:
        """
        Retorna status atual do filtro.

        Returns:
            Dict com status
        """
        return {
            'enabled': self.enabled,
            'paused': self._paused,
            'pause_threshold_usd': self.pause_threshold_usd,
            'resume_after_seconds': self.resume_after_seconds,
            'liq_features_status': self.liq_features.get_status(),
        }

    def reset(self) -> None:
        """Reseta o estado do filtro."""
        self._paused = False
        self._pause_start_time = None
        self.liq_features.reset()
