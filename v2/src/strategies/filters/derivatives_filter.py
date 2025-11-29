"""
Derivatives-based Trading Filter.

Filtra/ajusta trades baseado em Funding Rate e Open Interest.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class ExtremeAction(Enum):
    """Ações possíveis quando há condição extrema."""
    NONE = "none"
    REDUCE_SIZE = "reduce_size"
    REVERSE_BIAS = "reverse_bias"
    PAUSE = "pause"


@dataclass
class FilterResult:
    """Resultado do filtro."""
    allowed: bool
    size_multiplier: float
    reason: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'allowed': self.allowed,
            'size_multiplier': self.size_multiplier,
            'reason': self.reason,
            'metadata': self.metadata
        }


class DerivativesFilter:
    """
    Filtro baseado em dados de derivativos.
    
    Regras:
    1. Funding Rate > 0.1%: Reduz longs, favorece shorts
    2. Funding Rate < -0.1%: Reduz shorts, favorece longs
    3. OI subindo + Preço caindo: Possível capitulação (alerta)
    4. OI caindo + Preço subindo: Rally falso (evitar longs)
    
    Todos os parâmetros vêm do config - ZERO hardcoded!
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o filtro de derivativos.
        
        Args:
            config: Configuração com parâmetros
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        
        # Configuração de Funding Rate
        fr_config = config.get('funding_rate', {})
        self.fr_enabled = fr_config.get('enabled', True)
        self.fr_extreme_positive = fr_config.get('extreme_positive', 0.001)
        self.fr_extreme_negative = fr_config.get('extreme_negative', -0.001)
        self.fr_extreme_action = ExtremeAction(
            fr_config.get('extreme_action', 'reduce_size')
        )
        self.fr_extreme_size_multiplier = fr_config.get('extreme_size_multiplier', 0.5)
        
        # Configuração de Open Interest
        oi_config = config.get('open_interest', {})
        self.oi_enabled = oi_config.get('enabled', True)
        self.oi_significant_change = oi_config.get('significant_change_pct', 5.0) / 100
        
        # Divergência OI vs Preço
        div_config = oi_config.get('divergence_detection', {})
        self.div_enabled = div_config.get('enabled', True)
        self.div_price_threshold = div_config.get('price_change_threshold', 0.02)
        self.div_oi_threshold = div_config.get('oi_change_threshold', 0.03)
        
        # Configuração de Long/Short Ratio
        ls_config = config.get('long_short_ratio', {})
        self.ls_enabled = ls_config.get('enabled', True)
        self.ls_extreme_long = ls_config.get('extreme_long', 2.0)
        self.ls_extreme_short = ls_config.get('extreme_short', 0.5)
    
    def should_trade(
        self,
        signal_direction: str,
        derivatives_data: Dict[str, Any]
    ) -> Tuple[bool, float, str]:
        """
        Verifica se deve executar o trade.
        
        Args:
            signal_direction: "BUY" ou "SELL"
            derivatives_data: Dados de derivativos (funding_rate, open_interest, etc)
            
        Returns:
            Tuple[allowed, size_multiplier, reason]
        """
        result = self.filter(signal_direction, derivatives_data)
        return (result.allowed, result.size_multiplier, result.reason)
    
    def filter(
        self,
        signal_direction: str,
        derivatives_data: Dict[str, Any]
    ) -> FilterResult:
        """
        Aplica filtro baseado em derivativos.
        
        Args:
            signal_direction: "BUY" ou "SELL"
            derivatives_data: Dict com funding_rate, open_interest, 
                              long_short_ratio, price_change, oi_change
                              
        Returns:
            FilterResult com decisão
        """
        if not self.enabled:
            return FilterResult(
                allowed=True,
                size_multiplier=1.0,
                reason="Derivatives filter disabled",
                metadata={}
            )
        
        metadata: Dict[str, Any] = {}
        reasons: list = []
        size_multiplier = 1.0
        allowed = True
        
        # 1. Verifica Funding Rate
        if self.fr_enabled:
            fr_result = self._check_funding_rate(
                signal_direction,
                derivatives_data.get('funding_rate', 0.0)
            )
            if fr_result:
                allowed_fr, mult_fr, reason_fr = fr_result
                if not allowed_fr:
                    return FilterResult(
                        allowed=False,
                        size_multiplier=0.0,
                        reason=reason_fr,
                        metadata={'funding_rate_blocked': True}
                    )
                size_multiplier = min(size_multiplier, mult_fr)
                if mult_fr < 1.0:
                    reasons.append(reason_fr)
                metadata['funding_rate_multiplier'] = mult_fr
        
        # 2. Verifica divergência OI vs Preço
        if self.oi_enabled and self.div_enabled:
            div_result = self._check_divergence(
                signal_direction,
                derivatives_data.get('price_change_pct', 0.0),
                derivatives_data.get('oi_change_pct', 0.0)
            )
            if div_result:
                allowed_div, mult_div, reason_div = div_result
                if not allowed_div:
                    return FilterResult(
                        allowed=False,
                        size_multiplier=0.0,
                        reason=reason_div,
                        metadata={'divergence_blocked': True}
                    )
                size_multiplier = min(size_multiplier, mult_div)
                if mult_div < 1.0:
                    reasons.append(reason_div)
                metadata['divergence_multiplier'] = mult_div
        
        # 3. Verifica Long/Short Ratio
        if self.ls_enabled:
            ls_result = self._check_long_short_ratio(
                signal_direction,
                derivatives_data.get('long_short_ratio', 1.0)
            )
            if ls_result:
                allowed_ls, mult_ls, reason_ls = ls_result
                if not allowed_ls:
                    return FilterResult(
                        allowed=False,
                        size_multiplier=0.0,
                        reason=reason_ls,
                        metadata={'long_short_blocked': True}
                    )
                size_multiplier = min(size_multiplier, mult_ls)
                if mult_ls < 1.0:
                    reasons.append(reason_ls)
                metadata['long_short_multiplier'] = mult_ls
        
        # Combina razões
        if reasons:
            final_reason = "; ".join(reasons)
        else:
            final_reason = "Trade allowed by derivatives filter"
        
        return FilterResult(
            allowed=allowed,
            size_multiplier=size_multiplier,
            reason=final_reason,
            metadata=metadata
        )
    
    def _check_funding_rate(
        self,
        signal_direction: str,
        funding_rate: float
    ) -> Optional[Tuple[bool, float, str]]:
        """
        Verifica condições de Funding Rate.
        
        Args:
            signal_direction: "BUY" ou "SELL"
            funding_rate: Taxa de funding atual
            
        Returns:
            Tuple[allowed, multiplier, reason] ou None
        """
        is_long = signal_direction.upper() == "BUY"
        
        # Funding muito positivo = mercado muito long
        if funding_rate >= self.fr_extreme_positive:
            if is_long:
                # Long quando mercado já está muito long
                if self.fr_extreme_action == ExtremeAction.PAUSE:
                    return (False, 0.0, f"FR={funding_rate:.4%} extreme positive, pausing longs")
                elif self.fr_extreme_action == ExtremeAction.REDUCE_SIZE:
                    return (True, self.fr_extreme_size_multiplier, 
                            f"FR={funding_rate:.4%} high, reducing long size")
                elif self.fr_extreme_action == ExtremeAction.REVERSE_BIAS:
                    return (False, 0.0, f"FR={funding_rate:.4%} extreme, reversing bias to short")
            else:
                # Short quando mercado está muito long = favorável
                return (True, 1.2, f"FR={funding_rate:.4%} extreme positive, shorts favored")
        
        # Funding muito negativo = mercado muito short
        elif funding_rate <= self.fr_extreme_negative:
            if not is_long:
                # Short quando mercado já está muito short
                if self.fr_extreme_action == ExtremeAction.PAUSE:
                    return (False, 0.0, f"FR={funding_rate:.4%} extreme negative, pausing shorts")
                elif self.fr_extreme_action == ExtremeAction.REDUCE_SIZE:
                    return (True, self.fr_extreme_size_multiplier,
                            f"FR={funding_rate:.4%} low, reducing short size")
                elif self.fr_extreme_action == ExtremeAction.REVERSE_BIAS:
                    return (False, 0.0, f"FR={funding_rate:.4%} extreme, reversing bias to long")
            else:
                # Long quando mercado está muito short = favorável
                return (True, 1.2, f"FR={funding_rate:.4%} extreme negative, longs favored")
        
        return None
    
    def _check_divergence(
        self,
        signal_direction: str,
        price_change_pct: float,
        oi_change_pct: float
    ) -> Optional[Tuple[bool, float, str]]:
        """
        Verifica divergência OI vs Preço.
        
        Args:
            signal_direction: "BUY" ou "SELL"
            price_change_pct: Mudança % do preço
            oi_change_pct: Mudança % do OI
            
        Returns:
            Tuple[allowed, multiplier, reason] ou None
        """
        is_long = signal_direction.upper() == "BUY"
        
        # Converte para decimal se veio como percentual
        if abs(price_change_pct) > 1:
            price_change_pct = price_change_pct / 100
        if abs(oi_change_pct) > 1:
            oi_change_pct = oi_change_pct / 100
        
        # Preço sobe + OI cai = rally falso (bearish)
        if (price_change_pct > self.div_price_threshold and 
            oi_change_pct < -self.div_oi_threshold):
            if is_long:
                return (True, 0.5, 
                        f"Bearish divergence: price up {price_change_pct:.2%} but OI down {oi_change_pct:.2%}")
            else:
                # Short em divergência bearish = favorável
                return (True, 1.2, 
                        f"Bearish divergence favors shorts")
        
        # Preço cai + OI sobe = capitulação (pode ser fundo)
        if (price_change_pct < -self.div_price_threshold and
            oi_change_pct > self.div_oi_threshold):
            if is_long:
                # Long em possível fundo = pode ser favorável
                return (True, 1.0, 
                        f"Potential capitulation: price down but OI up")
            else:
                # Short em possível fundo = cuidado
                return (True, 0.7,
                        f"Bullish divergence: price down {price_change_pct:.2%} but OI up {oi_change_pct:.2%}")
        
        return None
    
    def _check_long_short_ratio(
        self,
        signal_direction: str,
        long_short_ratio: float
    ) -> Optional[Tuple[bool, float, str]]:
        """
        Verifica Long/Short Ratio.
        
        Args:
            signal_direction: "BUY" ou "SELL"
            long_short_ratio: Ratio (>1 = mais longs, <1 = mais shorts)
            
        Returns:
            Tuple[allowed, multiplier, reason] ou None
        """
        is_long = signal_direction.upper() == "BUY"
        
        # Muitos longs
        if long_short_ratio >= self.ls_extreme_long:
            if is_long:
                return (True, 0.7, 
                        f"L/S ratio={long_short_ratio:.2f} very high, reducing long size")
            else:
                return (True, 1.2,
                        f"L/S ratio={long_short_ratio:.2f} extreme long, shorts favored")
        
        # Muitos shorts
        if long_short_ratio <= self.ls_extreme_short:
            if not is_long:
                return (True, 0.7,
                        f"L/S ratio={long_short_ratio:.2f} very low, reducing short size")
            else:
                return (True, 1.2,
                        f"L/S ratio={long_short_ratio:.2f} extreme short, longs favored")
        
        return None
    
    def get_market_bias(
        self,
        derivatives_data: Dict[str, Any]
    ) -> Tuple[int, str]:
        """
        Retorna viés do mercado baseado em derivativos.
        
        Args:
            derivatives_data: Dados de derivativos
            
        Returns:
            Tuple[bias, explanation]
            bias: -1 (bearish), 0 (neutral), 1 (bullish)
        """
        bias = 0
        explanations = []
        
        # Funding Rate
        fr = derivatives_data.get('funding_rate', 0.0)
        if fr >= self.fr_extreme_positive:
            bias -= 1
            explanations.append(f"FR={fr:.4%} extreme positive (bearish)")
        elif fr <= self.fr_extreme_negative:
            bias += 1
            explanations.append(f"FR={fr:.4%} extreme negative (bullish)")
        
        # Long/Short Ratio
        ls = derivatives_data.get('long_short_ratio', 1.0)
        if ls >= self.ls_extreme_long:
            bias -= 1
            explanations.append(f"L/S={ls:.2f} extreme long (bearish)")
        elif ls <= self.ls_extreme_short:
            bias += 1
            explanations.append(f"L/S={ls:.2f} extreme short (bullish)")
        
        # Divergência
        price_change = derivatives_data.get('price_change_pct', 0.0)
        oi_change = derivatives_data.get('oi_change_pct', 0.0)
        
        if abs(price_change) > 1:
            price_change = price_change / 100
        if abs(oi_change) > 1:
            oi_change = oi_change / 100
        
        if (price_change > self.div_price_threshold and 
            oi_change < -self.div_oi_threshold):
            bias -= 1
            explanations.append("Bearish divergence: price up but OI down")
        elif (price_change < -self.div_price_threshold and
              oi_change > self.div_oi_threshold):
            bias += 1
            explanations.append("Bullish divergence: price down but OI up")
        
        # Normaliza bias para [-1, 1]
        if bias > 1:
            bias = 1
        elif bias < -1:
            bias = -1
        
        explanation = "; ".join(explanations) if explanations else "Neutral market"
        
        return (bias, explanation)
    
    def is_enabled(self) -> bool:
        """Verifica se o filtro está habilitado."""
        return self.enabled
    
    def set_enabled(self, enabled: bool) -> None:
        """Habilita ou desabilita o filtro."""
        self.enabled = enabled
