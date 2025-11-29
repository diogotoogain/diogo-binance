"""
Liquidation Feature Calculator.

Calcula features baseadas em liquidações forçadas.
ZERO hardcoded - todos os parâmetros vêm do config.
"""
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from collections import deque

import pandas as pd


class LiquidationFeatures:
    """
    Features baseadas em liquidações.

    Features geradas:
    - liq_volume_{window}s: Volume USD liquidado na janela
    - liq_count_{window}s: Número de liquidações na janela
    - liq_imbalance_{window}s: (long_liqs - short_liqs) / total - Imbalance direcional
    - liq_acceleration: Volume atual vs média móvel
    - liq_largest_single: Maior liquidação individual recente
    - cascade_active: bool - Cascata em andamento
    - cascade_direction: 1 (longs sendo liquidados) ou -1 (shorts), 0 (nenhuma)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o calculador de features de liquidação.

        Args:
            config: Dicionário de configuração com seção 'liquidations'
        """
        self.config = config.get('liquidations', {})
        self.enabled = self.config.get('enabled', True)

        # Features config
        features_config = self.config.get('features', {})
        self.lookback_windows = features_config.get(
            'lookback_windows', [60, 300, 900]
        )
        self.include_liq_volume = features_config.get('include_liq_volume', True)
        self.include_liq_count = features_config.get('include_liq_count', True)
        self.include_liq_imbalance = features_config.get('include_liq_imbalance', True)
        self.include_liq_acceleration = features_config.get('include_liq_acceleration', True)

        # Cascade detection config
        cascade_config = self.config.get('cascade_detection', {})
        self.cascade_enabled = cascade_config.get('enabled', True)
        self.cascade_window = self.config.get('cascade_window_seconds', 60)
        self.min_liquidations = cascade_config.get('min_liquidations', 5)
        self.volume_acceleration = cascade_config.get('volume_acceleration', 2.0)
        self.significant_cascade_usd = self.config.get('significant_cascade_usd', 1000000)

        # Buffer de liquidações recentes
        max_window = max(self.lookback_windows) if self.lookback_windows else 900
        # Estimate max liquidations: assume up to 100 per second in extreme cases
        max_buffer = max(10000, max_window * 10)
        self.liquidation_buffer: deque = deque(maxlen=max_buffer)

        # Estado para aceleração
        self._volume_history: deque = deque(maxlen=100)

    def add_liquidation(self, liquidation: Dict[str, Any]) -> None:
        """
        Adiciona uma liquidação ao buffer.

        Args:
            liquidation: Dict com dados da liquidação:
                - timestamp: datetime ou float (unix timestamp)
                - side: 'BUY' (short liq) ou 'SELL' (long liq)
                - quantity: quantidade liquidada
                - price: preço de liquidação
        """
        if not self.enabled:
            return

        timestamp = liquidation.get('timestamp')
        if timestamp is None:
            timestamp = datetime.now()
        elif isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp)

        quantity = float(liquidation.get('quantity', 0))
        price = float(liquidation.get('price', 0))
        usd_value = quantity * price

        self.liquidation_buffer.append({
            'timestamp': timestamp,
            'side': liquidation.get('side'),  # 'BUY' = short liq, 'SELL' = long liq
            'quantity': quantity,
            'price': price,
            'usd_value': usd_value
        })

    def calculate(self, current_time: Optional[datetime] = None) -> Dict[str, float]:
        """
        Calcula todas as features de liquidação.

        Args:
            current_time: Tempo atual (default: datetime.now())

        Returns:
            Dict com features calculadas
        """
        if not self.enabled:
            return {}

        if current_time is None:
            current_time = datetime.now()

        features: Dict[str, float] = {}

        # Calculate features for each lookback window
        for window in self.lookback_windows:
            window_liqs = self._get_liquidations_in_window(current_time, window)
            suffix = f"_{window}s"

            # Volume total
            if self.include_liq_volume:
                features[f'liq_volume{suffix}'] = sum(
                    liq['usd_value'] for liq in window_liqs
                )

            # Contagem
            if self.include_liq_count:
                features[f'liq_count{suffix}'] = float(len(window_liqs))

            # Imbalance
            if self.include_liq_imbalance:
                long_liqs = sum(
                    liq['usd_value'] for liq in window_liqs
                    if liq['side'] == 'SELL'
                )
                short_liqs = sum(
                    liq['usd_value'] for liq in window_liqs
                    if liq['side'] == 'BUY'
                )
                total = long_liqs + short_liqs
                if total > 0:
                    features[f'liq_imbalance{suffix}'] = (long_liqs - short_liqs) / total
                else:
                    features[f'liq_imbalance{suffix}'] = 0.0

        # Aceleração (volume atual vs média móvel)
        if self.include_liq_acceleration:
            features['liq_acceleration'] = self._calculate_acceleration(current_time)

        # Maior liquidação individual recente
        cascade_liqs = self._get_liquidations_in_window(current_time, self.cascade_window)
        if cascade_liqs:
            features['liq_largest_single'] = max(
                liq['usd_value'] for liq in cascade_liqs
            )
        else:
            features['liq_largest_single'] = 0.0

        # Cascade detection
        if self.cascade_enabled:
            features['cascade_active'] = float(self._detect_cascade(current_time))
            features['cascade_direction'] = float(self._get_cascade_direction(current_time))
        else:
            features['cascade_active'] = 0.0
            features['cascade_direction'] = 0.0

        return features

    def _get_liquidations_in_window(
        self,
        current_time: datetime,
        window_seconds: int
    ) -> List[Dict]:
        """
        Retorna liquidações dentro da janela de tempo.

        Args:
            current_time: Tempo atual
            window_seconds: Tamanho da janela em segundos

        Returns:
            Lista de liquidações dentro da janela
        """
        cutoff = current_time - timedelta(seconds=window_seconds)
        return [
            liq for liq in self.liquidation_buffer
            if liq['timestamp'] >= cutoff
        ]

    def _calculate_acceleration(self, current_time: datetime) -> float:
        """
        Calcula aceleração do volume de liquidações.

        Args:
            current_time: Tempo atual

        Returns:
            Ratio do volume atual vs média histórica
        """
        # Volume no cascade window
        recent_liqs = self._get_liquidations_in_window(current_time, self.cascade_window)
        current_volume = sum(liq['usd_value'] for liq in recent_liqs)

        # Atualiza histórico
        self._volume_history.append(current_volume)

        # Calcula média
        if len(self._volume_history) < 2:
            return 1.0

        avg_volume = sum(self._volume_history) / len(self._volume_history)
        if avg_volume == 0:
            return 1.0

        return current_volume / avg_volume

    def _detect_cascade(self, current_time: datetime) -> bool:
        """
        Detecta se uma cascata está em andamento.

        Args:
            current_time: Tempo atual

        Returns:
            True se cascata detectada
        """
        window_liqs = self._get_liquidations_in_window(current_time, self.cascade_window)

        # Mínimo de liquidações
        if len(window_liqs) < self.min_liquidations:
            return False

        # Volume total
        total_volume = sum(liq['usd_value'] for liq in window_liqs)
        if total_volume < self.significant_cascade_usd:
            return False

        # Aceleração
        acceleration = self._calculate_acceleration(current_time)
        if acceleration < self.volume_acceleration:
            return False

        return True

    def _get_cascade_direction(self, current_time: datetime) -> int:
        """
        Retorna direção da cascata.

        Args:
            current_time: Tempo atual

        Returns:
            1 se longs sendo liquidados (bearish)
            -1 se shorts sendo liquidados (bullish)
            0 se nenhuma cascata
        """
        if not self._detect_cascade(current_time):
            return 0

        window_liqs = self._get_liquidations_in_window(current_time, self.cascade_window)

        long_liqs = sum(
            liq['usd_value'] for liq in window_liqs
            if liq['side'] == 'SELL'
        )
        short_liqs = sum(
            liq['usd_value'] for liq in window_liqs
            if liq['side'] == 'BUY'
        )

        if long_liqs > short_liqs:
            return 1  # Longs sendo liquidados (bearish)
        elif short_liqs > long_liqs:
            return -1  # Shorts sendo liquidados (bullish)
        return 0

    def get_cascade_volume(self, current_time: Optional[datetime] = None) -> float:
        """
        Retorna volume total da cascata atual.

        Args:
            current_time: Tempo atual

        Returns:
            Volume USD da cascata
        """
        if current_time is None:
            current_time = datetime.now()

        window_liqs = self._get_liquidations_in_window(current_time, self.cascade_window)
        return sum(liq['usd_value'] for liq in window_liqs)

    def reset(self) -> None:
        """Reseta todos os buffers e estados."""
        self.liquidation_buffer.clear()
        self._volume_history.clear()

    def get_status(self) -> Dict[str, Any]:
        """
        Retorna status atual.

        Returns:
            Dict com status
        """
        return {
            'enabled': self.enabled,
            'buffer_size': len(self.liquidation_buffer),
            'lookback_windows': self.lookback_windows,
            'cascade_window': self.cascade_window,
            'min_liquidations': self.min_liquidations,
        }
