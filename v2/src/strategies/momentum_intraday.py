"""
Momentum Intraday Strategy

Estratégia de momentum intradiária baseada em EMA crossover + RSI + ADX.
Opera na direção da tendência quando há confirmação de momentum.

Parâmetros (strategies.momentum_intraday):
- ema_fast: Período da EMA rápida (OPTIMIZE: [5-15])
- ema_slow: Período da EMA lenta (OPTIMIZE: [15-50])
- filters.adx_filter.min_adx: ADX mínimo para confirmar tendência
- throttling.max_trades_per_hour: Rate limit

ZERO hardcoded - todos os parâmetros vêm do config.
"""
from typing import Any, Dict, Optional
import time

from v2.src.strategies.base import Strategy, Signal, SignalDirection
from v2.src.strategies.throttling import Throttler


class MomentumIntraday(Strategy):
    """
    Estratégia de momentum intradiária.
    
    Princípios:
    - EMA fast > EMA slow = tendência de alta
    - EMA fast < EMA slow = tendência de baixa
    - ADX alto = tendência forte = melhor para momentum
    - RSI confirma se não está sobrecomprado/sobrevendido
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa a estratégia de Momentum.
        
        Args:
            config: Configuração da estratégia do YAML
        """
        enabled = config.get('enabled', True)
        super().__init__("MomentumIntraday", config, enabled)
        
        # Parâmetros principais
        params = config.get('params', {})
        self.ema_fast = params.get('ema_fast', 9)
        self.ema_slow = params.get('ema_slow', 21)
        self.rsi_entry_threshold = params.get('rsi_entry_threshold', 50)
        self.holding_minutes = params.get('holding_minutes', 60)
        
        # Filtros
        filters = config.get('filters', {})
        self.adx_filter_enabled = filters.get('adx_filter', {}).get('enabled', True)
        self.min_adx = filters.get('adx_filter', {}).get('min_adx', 25)
        self.volume_confirmation_enabled = filters.get('volume_confirmation', {}).get('enabled', True)
        self.min_volume_percentile = filters.get('volume_confirmation', {}).get('min_volume_percentile', 60)
        
        # Throttling
        throttle_config = config.get('throttling', {})
        self.throttler = Throttler(throttle_config)
        
        # Estado interno
        self._last_ema_fast: Optional[float] = None
        self._last_ema_slow: Optional[float] = None
        self._crossover_detected = False
        
        self.log(
            f"Inicializado | EMA Fast: {self.ema_fast} | "
            f"EMA Slow: {self.ema_slow} | Min ADX: {self.min_adx}"
        )
    
    def _check_filters(self, market_data: Dict[str, Any]) -> bool:
        """
        Verifica se todos os filtros passam.
        
        Args:
            market_data: Dados de mercado
        
        Returns:
            True se todos os filtros passam
        """
        # Filtro ADX - queremos tendência forte para momentum
        if self.adx_filter_enabled:
            adx = market_data.get('adx', 0)
            if adx < self.min_adx:
                return False
        
        # Confirmação de volume
        if self.volume_confirmation_enabled:
            volume_percentile = market_data.get('volume_percentile', 0)
            if volume_percentile < self.min_volume_percentile:
                return False
        
        return True
    
    def _detect_crossover(
        self, 
        ema_fast: float, 
        ema_slow: float
    ) -> Optional[str]:
        """
        Detecta crossover de EMAs.
        
        Args:
            ema_fast: Valor atual da EMA rápida
            ema_slow: Valor atual da EMA lenta
        
        Returns:
            'bullish', 'bearish' ou None
        """
        if self._last_ema_fast is None or self._last_ema_slow is None:
            self._last_ema_fast = ema_fast
            self._last_ema_slow = ema_slow
            return None
        
        # Bullish crossover: fast cruza acima de slow
        if self._last_ema_fast <= self._last_ema_slow and ema_fast > ema_slow:
            crossover = 'bullish'
        # Bearish crossover: fast cruza abaixo de slow
        elif self._last_ema_fast >= self._last_ema_slow and ema_fast < ema_slow:
            crossover = 'bearish'
        else:
            crossover = None
        
        # Atualiza estado
        self._last_ema_fast = ema_fast
        self._last_ema_slow = ema_slow
        
        return crossover
    
    def _check_rsi_entry(
        self, 
        rsi: float, 
        direction: SignalDirection
    ) -> bool:
        """
        Verifica se RSI permite entrada.
        
        Args:
            rsi: Valor atual do RSI
            direction: Direção pretendida
        
        Returns:
            True se RSI OK para entrada
        """
        if direction == SignalDirection.BUY:
            # Para compra, RSI não deve estar muito alto
            return rsi >= self.rsi_entry_threshold
        elif direction == SignalDirection.SELL:
            # Para venda, RSI não deve estar muito baixo
            return rsi <= (100 - self.rsi_entry_threshold)
        return False
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        Gera sinal baseado em EMA crossover + RSI + ADX.
        
        Args:
            market_data: Dict com:
                - ema_fast: EMA rápida (calculada externamente)
                - ema_slow: EMA lenta
                - rsi: RSI atual
                - adx: Average Directional Index
                - volume_percentile: Percentil do volume
                - price: Preço atual
        
        Returns:
            Signal ou None
        """
        if not self.enabled:
            return None
        
        # Verifica throttling
        if not self.throttler.can_trade():
            return None
        
        # Verifica filtros
        if not self._check_filters(market_data):
            return None
        
        # Extrai indicadores
        ema_fast = market_data.get('ema_fast', 0)
        ema_slow = market_data.get('ema_slow', 0)
        rsi = market_data.get('rsi', 50)
        adx = market_data.get('adx', 0)
        price = market_data.get('price', 0)
        
        # Detecta crossover
        crossover = self._detect_crossover(ema_fast, ema_slow)
        
        if crossover is None:
            return None
        
        signal = None
        
        # Bullish crossover
        if crossover == 'bullish':
            if self._check_rsi_entry(rsi, SignalDirection.BUY):
                # Confiança baseada na força do ADX
                confidence = min(1.0, adx / 50)  # ADX 50+ = confiança máxima
                signal = Signal(
                    direction=SignalDirection.BUY,
                    strategy_name=self.name,
                    confidence=confidence,
                    reason=f"Bullish_EMA_Cross ADX={adx:.1f} RSI={rsi:.1f}",
                    metadata={
                        'ema_fast': ema_fast,
                        'ema_slow': ema_slow,
                        'rsi': rsi,
                        'adx': adx,
                        'price': price,
                        'holding_minutes': self.holding_minutes,
                    }
                )
        
        # Bearish crossover
        elif crossover == 'bearish':
            if self._check_rsi_entry(rsi, SignalDirection.SELL):
                confidence = min(1.0, adx / 50)
                signal = Signal(
                    direction=SignalDirection.SELL,
                    strategy_name=self.name,
                    confidence=confidence,
                    reason=f"Bearish_EMA_Cross ADX={adx:.1f} RSI={rsi:.1f}",
                    metadata={
                        'ema_fast': ema_fast,
                        'ema_slow': ema_slow,
                        'rsi': rsi,
                        'adx': adx,
                        'price': price,
                        'holding_minutes': self.holding_minutes,
                    }
                )
        
        if signal:
            self._total_signals += 1
            self._last_signal_time = time.time()
            self.throttler.record_trade()
            self.log(f"📈 SINAL: {signal.direction.value} | Conf: {signal.confidence:.2f} | {signal.reason}")
        
        return signal
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna status estendido."""
        base_status = super().get_status()
        base_status.update({
            'ema_fast_period': self.ema_fast,
            'ema_slow_period': self.ema_slow,
            'min_adx': self.min_adx,
            'rsi_threshold': self.rsi_entry_threshold,
            'holding_minutes': self.holding_minutes,
            'throttler': self.throttler.get_status(),
        })
        return base_status
