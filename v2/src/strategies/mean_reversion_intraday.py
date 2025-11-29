"""
Mean Reversion Intraday Strategy

Estratégia de mean reversion baseada em Z-Score e Bollinger Bands.
Opera contra a tendência quando preço está muito afastado da média.

Parâmetros (strategies.mean_reversion_intraday):
- zscore_entry: Z-Score para entrada (OPTIMIZE: [1.5-3.0])
- filters.adx_filter.max_adx: ADX máximo (mercado lateral)
- throttling.max_trades_per_hour: Rate limit

ZERO hardcoded - todos os parâmetros vêm do config.
"""
from typing import Any, Dict, Optional
import time

from v2.src.strategies.base import Strategy, Signal, SignalDirection
from v2.src.strategies.throttling import Throttler


class MeanReversionIntraday(Strategy):
    """
    Estratégia de mean reversion intradiária.
    
    Princípios:
    - Z-Score alto positivo = preço acima da média = SELL (reversão para baixo)
    - Z-Score alto negativo = preço abaixo da média = BUY (reversão para cima)
    - ADX baixo = mercado lateral = melhor para mean reversion
    - Bollinger Bands confirmam extremos
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa a estratégia de Mean Reversion.
        
        Args:
            config: Configuração da estratégia do YAML
        """
        enabled = config.get('enabled', True)
        super().__init__("MeanReversionIntraday", config, enabled)
        
        # Parâmetros principais
        params = config.get('params', {})
        self.zscore_entry = params.get('zscore_entry', 2.0)
        self.zscore_exit = params.get('zscore_exit', 0.5)
        self.bollinger_entry_std = params.get('bollinger_entry_std', 2.0)
        self.holding_minutes = params.get('holding_minutes', 30)
        
        # Filtros
        filters = config.get('filters', {})
        self.adx_filter_enabled = filters.get('adx_filter', {}).get('enabled', True)
        self.max_adx = filters.get('adx_filter', {}).get('max_adx', 20)
        self.liquidity_filter_enabled = filters.get('liquidity_cluster_filter', {}).get('enabled', True)
        self.proximity_threshold_pct = filters.get('liquidity_cluster_filter', {}).get('proximity_threshold_pct', 0.5)
        
        # Throttling
        throttle_config = config.get('throttling', {})
        self.throttler = Throttler(throttle_config)
        
        # Estado interno
        self._in_position = False
        self._position_direction: Optional[SignalDirection] = None
        
        self.log(
            f"Inicializado | Z-Score Entry: {self.zscore_entry} | "
            f"Z-Score Exit: {self.zscore_exit} | Max ADX: {self.max_adx}"
        )
    
    def _check_filters(self, market_data: Dict[str, Any]) -> bool:
        """
        Verifica se todos os filtros passam.
        
        Args:
            market_data: Dados de mercado
        
        Returns:
            True se todos os filtros passam
        """
        # Filtro ADX - queremos mercado lateral para mean reversion
        if self.adx_filter_enabled:
            adx = market_data.get('adx', 0)
            if adx > self.max_adx:
                return False
        
        # Filtro de proximidade a cluster de liquidez
        if self.liquidity_filter_enabled:
            # Verifica se há suporte/resistência próximo
            liquidity_proximity = market_data.get('liquidity_proximity_pct', 100)
            if liquidity_proximity > self.proximity_threshold_pct:
                # Sem suporte/resistência próximo para reverter
                pass  # Ainda pode operar, mas com menos confiança
        
        return True
    
    def _check_bollinger_confirmation(
        self, 
        price: float, 
        bb_upper: float, 
        bb_lower: float,
        direction: SignalDirection
    ) -> bool:
        """
        Verifica confirmação das Bollinger Bands.
        
        Args:
            price: Preço atual
            bb_upper: Banda superior
            bb_lower: Banda inferior
            direction: Direção pretendida
        
        Returns:
            True se Bollinger confirma
        """
        if direction == SignalDirection.BUY:
            # Para compra (reversão para cima), preço deve estar na banda inferior
            return price <= bb_lower
        elif direction == SignalDirection.SELL:
            # Para venda (reversão para baixo), preço deve estar na banda superior
            return price >= bb_upper
        return False
    
    def _calculate_confidence(
        self, 
        zscore: float, 
        has_bollinger_confirm: bool,
        has_liquidity_support: bool
    ) -> float:
        """
        Calcula confiança do sinal.
        
        Args:
            zscore: Z-Score absoluto
            has_bollinger_confirm: Se Bollinger confirma
            has_liquidity_support: Se há suporte/resistência
        
        Returns:
            Confiança entre 0 e 1
        """
        # Base: Z-Score mais extremo = mais confiança
        base_confidence = min(1.0, abs(zscore) / 3.0)
        
        # Bônus por confirmação de Bollinger
        if has_bollinger_confirm:
            base_confidence = min(1.0, base_confidence + 0.2)
        
        # Bônus por suporte/resistência próximo
        if has_liquidity_support:
            base_confidence = min(1.0, base_confidence + 0.1)
        
        return base_confidence
    
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        Gera sinal baseado em Z-Score e Bollinger Bands.
        
        Args:
            market_data: Dict com:
                - zscore: Z-Score do preço
                - bb_upper: Bollinger Band superior
                - bb_lower: Bollinger Band inferior
                - bb_middle: Bollinger Band média (SMA)
                - adx: Average Directional Index
                - liquidity_proximity_pct: % de distância ao suporte/resistência
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
        zscore = market_data.get('zscore', 0)
        bb_upper = market_data.get('bb_upper', 0)
        bb_lower = market_data.get('bb_lower', 0)
        bb_middle = market_data.get('bb_middle', 0)
        price = market_data.get('price', 0)
        liquidity_proximity = market_data.get('liquidity_proximity_pct', 100)
        
        signal = None
        has_liquidity_support = liquidity_proximity <= self.proximity_threshold_pct
        
        # Z-Score alto negativo = preço muito abaixo da média = BUY
        if zscore <= -self.zscore_entry:
            has_bb_confirm = self._check_bollinger_confirmation(
                price, bb_upper, bb_lower, SignalDirection.BUY
            )
            confidence = self._calculate_confidence(
                zscore, has_bb_confirm, has_liquidity_support
            )
            signal = Signal(
                direction=SignalDirection.BUY,
                strategy_name=self.name,
                confidence=confidence,
                reason=f"MeanRev_Buy Z={zscore:.2f} BB_Confirm={has_bb_confirm}",
                metadata={
                    'zscore': zscore,
                    'bb_upper': bb_upper,
                    'bb_lower': bb_lower,
                    'bb_middle': bb_middle,
                    'price': price,
                    'bb_confirmed': has_bb_confirm,
                    'liquidity_support': has_liquidity_support,
                    'holding_minutes': self.holding_minutes,
                }
            )
        
        # Z-Score alto positivo = preço muito acima da média = SELL
        elif zscore >= self.zscore_entry:
            has_bb_confirm = self._check_bollinger_confirmation(
                price, bb_upper, bb_lower, SignalDirection.SELL
            )
            confidence = self._calculate_confidence(
                zscore, has_bb_confirm, has_liquidity_support
            )
            signal = Signal(
                direction=SignalDirection.SELL,
                strategy_name=self.name,
                confidence=confidence,
                reason=f"MeanRev_Sell Z={zscore:.2f} BB_Confirm={has_bb_confirm}",
                metadata={
                    'zscore': zscore,
                    'bb_upper': bb_upper,
                    'bb_lower': bb_lower,
                    'bb_middle': bb_middle,
                    'price': price,
                    'bb_confirmed': has_bb_confirm,
                    'liquidity_support': has_liquidity_support,
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
            'zscore_entry': self.zscore_entry,
            'zscore_exit': self.zscore_exit,
            'bollinger_entry_std': self.bollinger_entry_std,
            'max_adx': self.max_adx,
            'holding_minutes': self.holding_minutes,
            'throttler': self.throttler.get_status(),
        })
        return base_status
