"""
Volatility Breakout Strategy

Estratégia de breakout baseada em ATR e Volume.
Detecta squeezes de volatilidade e opera o breakout.

Parâmetros (strategies.volatility_breakout):
- breakout_atr_mult: Multiplicador de ATR para breakout (OPTIMIZE: [1.0-2.5])
- volume_confirm_mult: Multiplicador de volume para confirmação
- throttling.max_trades_per_day: Rate limit

ZERO hardcoded - todos os parâmetros vêm do config.
"""
from typing import Any, Dict, Optional
import time

from v2.src.strategies.base import Strategy, Signal, SignalDirection
from v2.src.strategies.throttling import Throttler


class VolatilityBreakout(Strategy):
    """
    Estratégia de breakout de volatilidade.
    
    Princípios:
    - Squeeze = Bollinger Bands apertadas = baixa volatilidade
    - Breakout = preço rompe faixa + volume alto
    - ATR define a magnitude do movimento esperado
    - Após breakout, confirma com ADX crescente
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa a estratégia de Volatility Breakout.
        
        Args:
            config: Configuração da estratégia do YAML
        """
        enabled = config.get('enabled', True)
        super().__init__("VolatilityBreakout", config, enabled)
        
        # Parâmetros principais
        params = config.get('params', {})
        self.squeeze_bb_width_percentile = params.get('squeeze_bb_width_percentile', 20)
        self.breakout_atr_mult = params.get('breakout_atr_mult', 1.5)
        self.volume_confirm_mult = params.get('volume_confirm_mult', 1.5)
        self.holding_minutes = params.get('holding_minutes', 120)
        
        # Filtros
        filters = config.get('filters', {})
        self.adx_filter_enabled = filters.get('adx_filter', {}).get('enabled', True)
        self.min_adx_after_breakout = filters.get('adx_filter', {}).get('min_adx_after_breakout', 20)
        
        # Throttling
        throttle_config = config.get('throttling', {})
        self.throttler = Throttler(throttle_config)
        
        # Estado interno
        self._in_squeeze = False
        self._squeeze_start_price: Optional[float] = None
        self._last_bb_width: Optional[float] = None
        
        self.log(
            f"Inicializado | ATR Mult: {self.breakout_atr_mult} | "
            f"Vol Mult: {self.volume_confirm_mult} | Squeeze Pct: {self.squeeze_bb_width_percentile}"
        )
    
    def _detect_squeeze(self, bb_width_percentile: float) -> bool:
        """
        Detecta se está em squeeze (baixa volatilidade).
        
        Args:
            bb_width_percentile: Percentil da largura das BB
        
        Returns:
            True se em squeeze
        """
        return bb_width_percentile <= self.squeeze_bb_width_percentile
    
    def _detect_breakout(
        self,
        price: float,
        high: float,
        low: float,
        atr: float,
        prev_high: float,
        prev_low: float
    ) -> Optional[str]:
        """
        Detecta breakout de preço.
        
        Args:
            price: Preço atual
            high: Máxima atual
            low: Mínima atual
            atr: ATR atual
            prev_high: Máxima anterior
            prev_low: Mínima anterior
        
        Returns:
            'bullish', 'bearish' ou None
        """
        # Breakout threshold baseado em ATR
        threshold = atr * self.breakout_atr_mult
        
        # Bullish breakout: preço rompe acima da máxima + threshold
        if high > prev_high + threshold:
            return 'bullish'
        
        # Bearish breakout: preço rompe abaixo da mínima - threshold
        if low < prev_low - threshold:
            return 'bearish'
        
        return None
    
    def _check_volume_confirmation(self, volume_ratio: float) -> bool:
        """
        Verifica confirmação de volume.
        
        Args:
            volume_ratio: Ratio do volume atual vs média
        
        Returns:
            True se volume confirma
        """
        return volume_ratio >= self.volume_confirm_mult
    
    def _check_adx_confirmation(self, adx: float) -> bool:
        """
        Verifica confirmação de ADX (tendência se formando).
        
        Args:
            adx: ADX atual
        
        Returns:
            True se ADX confirma
        """
        if not self.adx_filter_enabled:
            return True
        return adx >= self.min_adx_after_breakout
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        Gera sinal baseado em breakout de volatilidade.
        
        Args:
            market_data: Dict com:
                - price: Preço atual
                - high: Máxima do período
                - low: Mínima do período
                - prev_high: Máxima do período anterior
                - prev_low: Mínima do período anterior
                - atr: Average True Range
                - adx: Average Directional Index
                - bb_width_percentile: Percentil da largura das BB
                - volume_ratio: Ratio de volume vs média
        
        Returns:
            Signal ou None
        """
        if not self.enabled:
            return None
        
        # Verifica throttling
        if not self.throttler.can_trade():
            return None
        
        # Extrai indicadores
        price = market_data.get('price', 0)
        high = market_data.get('high', price)
        low = market_data.get('low', price)
        prev_high = market_data.get('prev_high', high)
        prev_low = market_data.get('prev_low', low)
        atr = market_data.get('atr', 0)
        adx = market_data.get('adx', 0)
        bb_width_percentile = market_data.get('bb_width_percentile', 50)
        volume_ratio = market_data.get('volume_ratio', 1.0)
        
        # Detecta squeeze
        is_in_squeeze = self._detect_squeeze(bb_width_percentile)
        
        # Atualiza estado de squeeze
        if is_in_squeeze and not self._in_squeeze:
            self._in_squeeze = True
            self._squeeze_start_price = price
            self.log(f"🔄 Squeeze detectado | BB Width Pct: {bb_width_percentile}")
        elif not is_in_squeeze:
            self._in_squeeze = False
            self._squeeze_start_price = None
        
        # Só opera breakout se estiver/estava em squeeze
        if not self._in_squeeze and self._squeeze_start_price is None:
            return None
        
        # Detecta breakout
        breakout = self._detect_breakout(price, high, low, atr, prev_high, prev_low)
        
        if breakout is None:
            return None
        
        # Verifica confirmações
        has_volume = self._check_volume_confirmation(volume_ratio)
        has_adx = self._check_adx_confirmation(adx)
        
        if not has_volume:
            self.log(f"⚠️ Breakout sem volume | Ratio: {volume_ratio:.2f}")
            return None
        
        signal = None
        
        # Calcula confiança
        confidence = 0.5  # Base
        if has_volume:
            confidence += 0.2
        if has_adx:
            confidence += 0.2
        # Bônus se estava em squeeze profundo
        if bb_width_percentile <= 10:
            confidence += 0.1
        confidence = min(1.0, confidence)
        
        # Bullish breakout
        if breakout == 'bullish':
            signal = Signal(
                direction=SignalDirection.BUY,
                strategy_name=self.name,
                confidence=confidence,
                reason=f"Vol_Breakout_Up ATR_Mult={self.breakout_atr_mult} Vol={volume_ratio:.2f}",
                metadata={
                    'breakout_type': 'bullish',
                    'price': price,
                    'atr': atr,
                    'adx': adx,
                    'volume_ratio': volume_ratio,
                    'bb_width_percentile': bb_width_percentile,
                    'holding_minutes': self.holding_minutes,
                }
            )
        
        # Bearish breakout
        elif breakout == 'bearish':
            signal = Signal(
                direction=SignalDirection.SELL,
                strategy_name=self.name,
                confidence=confidence,
                reason=f"Vol_Breakout_Down ATR_Mult={self.breakout_atr_mult} Vol={volume_ratio:.2f}",
                metadata={
                    'breakout_type': 'bearish',
                    'price': price,
                    'atr': atr,
                    'adx': adx,
                    'volume_ratio': volume_ratio,
                    'bb_width_percentile': bb_width_percentile,
                    'holding_minutes': self.holding_minutes,
                }
            )
        
        if signal:
            self._total_signals += 1
            self._last_signal_time = time.time()
            self.throttler.record_trade()
            # Reset squeeze state após sinal
            self._in_squeeze = False
            self._squeeze_start_price = None
            self.log(f"📈 SINAL: {signal.direction.value} | Conf: {signal.confidence:.2f} | {signal.reason}")
        
        return signal
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna status estendido."""
        base_status = super().get_status()
        base_status.update({
            'squeeze_bb_width_percentile': self.squeeze_bb_width_percentile,
            'breakout_atr_mult': self.breakout_atr_mult,
            'volume_confirm_mult': self.volume_confirm_mult,
            'min_adx_after_breakout': self.min_adx_after_breakout,
            'holding_minutes': self.holding_minutes,
            'in_squeeze': self._in_squeeze,
            'throttler': self.throttler.get_status(),
        })
        return base_status
