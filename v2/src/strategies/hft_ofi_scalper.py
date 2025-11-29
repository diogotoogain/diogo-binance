"""
HFT OFI Scalper Strategy

Estratégia de HFT (High Frequency Trading) baseada em OFI (Order Flow Imbalance).
Usa OFI e TFI para detectar desequilíbrios de fluxo e scalp rápido.

Parâmetros (strategies.hft_ofi_scalper):
- ofi_threshold: Limite de OFI para sinal (OPTIMIZE: [0.1-0.5])
- tfi_threshold: Limite de TFI
- holding_seconds: Tempo máximo de holding (OPTIMIZE: [10-300])
- filters.adx_filter.max_adx: ADX máximo para operar
- throttling.max_trades_per_minute: Rate limit

ZERO hardcoded - todos os parâmetros vêm do config.
"""
from typing import Any, Dict, Optional
import time

from v2.src.strategies.base import Strategy, Signal, SignalDirection
from v2.src.strategies.throttling import Throttler


class HFTOFIScalper(Strategy):
    """
    Estratégia de scalping de alta frequência baseada em Order Flow Imbalance.
    
    Princípios:
    - OFI alto positivo = pressão de compra = BUY
    - OFI alto negativo = pressão de venda = SELL
    - TFI confirma direção do fluxo
    - ADX baixo = mercado lateral = melhor para scalping
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o HFT OFI Scalper.
        
        Args:
            config: Configuração da estratégia do YAML
        """
        # Verifica se está habilitado
        enabled = config.get('enabled', True)
        super().__init__("HFTOFIScalper", config, enabled)
        
        # Parâmetros principais (do config)
        params = config.get('params', {})
        self.ofi_threshold = params.get('ofi_threshold', 0.3)
        self.tfi_threshold = params.get('tfi_threshold', 0.3)
        self.min_spread_bps = params.get('min_spread_bps', 1.0)
        self.holding_seconds = params.get('holding_seconds', 30)
        
        # Filtros
        filters = config.get('filters', {})
        self.adx_filter_enabled = filters.get('adx_filter', {}).get('enabled', True)
        self.max_adx = filters.get('adx_filter', {}).get('max_adx', 30)
        self.volume_spike_filter_enabled = filters.get('volume_spike_filter', {}).get('enabled', True)
        self.min_spike_multiplier = filters.get('volume_spike_filter', {}).get('min_spike_multiplier', 1.5)
        
        # Throttling
        throttle_config = config.get('throttling', {})
        self.throttler = Throttler(throttle_config)
        
        # Estado interno
        self._entry_time: Optional[float] = None
        self._entry_price: Optional[float] = None
        
        self.log(
            f"Inicializado | OFI thresh: {self.ofi_threshold} | "
            f"TFI thresh: {self.tfi_threshold} | Hold: {self.holding_seconds}s"
        )
    
    def _check_filters(self, market_data: Dict[str, Any]) -> bool:
        """
        Verifica se todos os filtros passam.
        
        Args:
            market_data: Dados de mercado
        
        Returns:
            True se todos os filtros passam
        """
        # Filtro ADX - queremos mercado lateral para scalping
        if self.adx_filter_enabled:
            adx = market_data.get('adx', 0)
            if adx > self.max_adx:
                return False
        
        # Filtro de Volume Spike
        if self.volume_spike_filter_enabled:
            volume_spike = market_data.get('volume_spike', 1.0)
            if volume_spike < self.min_spike_multiplier:
                return False
        
        return True
    
    def _check_spread(self, market_data: Dict[str, Any]) -> bool:
        """
        Verifica se o spread é adequado.
        
        Args:
            market_data: Dados de mercado
        
        Returns:
            True se spread OK
        """
        spread_bps = market_data.get('spread_bps', 0)
        return spread_bps >= self.min_spread_bps
    
    def _should_exit(self, current_time: float) -> bool:
        """
        Verifica se deve sair da posição (holding time).
        
        Args:
            current_time: Timestamp atual
        
        Returns:
            True se deve sair
        """
        if self._entry_time is None:
            return False
        
        elapsed = current_time - self._entry_time
        return elapsed >= self.holding_seconds
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        Gera sinal baseado em OFI e TFI.
        
        Args:
            market_data: Dict com:
                - ofi: Order Flow Imbalance normalizado [-1, 1]
                - tfi: Trade Flow Imbalance [-1, 1]
                - adx: Average Directional Index
                - volume_spike: Múltiplo do volume médio
                - spread_bps: Spread em basis points
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
        
        # Verifica spread
        if not self._check_spread(market_data):
            return None
        
        # Extrai indicadores
        ofi = market_data.get('ofi', 0)
        tfi = market_data.get('tfi', 0)
        price = market_data.get('price', 0)
        
        # Lógica de entrada
        signal = None
        
        # BUY: OFI positivo alto + TFI positivo
        if ofi >= self.ofi_threshold and tfi >= self.tfi_threshold:
            signal = Signal(
                direction=SignalDirection.BUY,
                strategy_name=self.name,
                confidence=min(1.0, (abs(ofi) + abs(tfi)) / 2),
                reason=f"OFI_High_Buy OFI={ofi:.2f} TFI={tfi:.2f}",
                metadata={
                    'ofi': ofi,
                    'tfi': tfi,
                    'price': price,
                    'holding_seconds': self.holding_seconds,
                }
            )
            self._entry_time = time.time()
            self._entry_price = price
        
        # SELL: OFI negativo alto + TFI negativo
        elif ofi <= -self.ofi_threshold and tfi <= -self.tfi_threshold:
            signal = Signal(
                direction=SignalDirection.SELL,
                strategy_name=self.name,
                confidence=min(1.0, (abs(ofi) + abs(tfi)) / 2),
                reason=f"OFI_High_Sell OFI={ofi:.2f} TFI={tfi:.2f}",
                metadata={
                    'ofi': ofi,
                    'tfi': tfi,
                    'price': price,
                    'holding_seconds': self.holding_seconds,
                }
            )
            self._entry_time = time.time()
            self._entry_price = price
        
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
            'ofi_threshold': self.ofi_threshold,
            'tfi_threshold': self.tfi_threshold,
            'holding_seconds': self.holding_seconds,
            'max_adx': self.max_adx,
            'throttler': self.throttler.get_status(),
        })
        return base_status
