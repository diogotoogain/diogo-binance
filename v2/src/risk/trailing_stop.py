"""
Trailing Stop - Move o SL conforme preço avança.

Permite travar lucros à medida que o preço se move a favor.
"""
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TrailingStop:
    """
    Trailing Stop que move o SL conforme preço avança.
    
    Parâmetros do config (position.trailing_stop):
    - enabled: false                  # TOGGLE
    - type: "atr"                     # TOGGLE: [fixed_pct, atr]
    - activation_pct: 1.0             # OPTIMIZE: [0.5, 1.0, 1.5, 2.0]
    - trail_pct: 0.5                  # OPTIMIZE: [0.3, 0.5, 0.75, 1.0]
    - trail_atr_mult: 1.0             # OPTIMIZE: [0.5, 1.0, 1.5, 2.0]
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trailing stop.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config['position']['trailing_stop']
        self.enabled = self.config['enabled']
        self.trailing_type = self.config['type']
        
        # Estado de tracking
        self.highest_price: Optional[float] = None
        self.lowest_price: Optional[float] = None
        self.activated = False
        self.current_trailing_sl: Optional[float] = None
        
    def reset(self) -> None:
        """Reset trailing stop state for new position."""
        self.highest_price = None
        self.lowest_price = None
        self.activated = False
        self.current_trailing_sl = None
        
    def update(self, current_price: float, entry_price: float, 
               side: str, atr: Optional[float] = None) -> Optional[float]:
        """
        Atualiza trailing stop e retorna novo SL se necessário.
        
        Args:
            current_price: Preço atual do mercado
            entry_price: Preço de entrada da posição
            side: "LONG" ou "SHORT"
            atr: ATR atual (para tipo atr)
            
        Returns:
            Novo preço de SL ou None se não mudou
        """
        if not self.enabled:
            return None
            
        if current_price <= 0 or entry_price <= 0:
            return None
            
        # Verifica se ativou o trailing
        if not self.activated:
            pnl_pct = self._calculate_pnl_pct(current_price, entry_price, side)
            activation_pct = self.config['activation_pct']
            
            if pnl_pct >= activation_pct:
                self.activated = True
                logger.info(
                    f"🎯 Trailing stop ativado! "
                    f"P&L: {pnl_pct:.2f}% >= {activation_pct:.2f}%"
                )
                
        if not self.activated:
            return None
            
        # Atualiza extremos e calcula novo SL
        new_sl = None
        
        if side.upper() == "LONG":
            if self.highest_price is None or current_price > self.highest_price:
                self.highest_price = current_price
                new_sl = self._calculate_trailing_sl(self.highest_price, side, atr)
                
                # Só atualiza se o novo SL for maior que o atual
                if self.current_trailing_sl is None or new_sl > self.current_trailing_sl:
                    self.current_trailing_sl = new_sl
                    logger.debug(f"📈 Novo trailing SL (LONG): {new_sl:.2f}")
                    return new_sl
                    
        else:  # SHORT
            if self.lowest_price is None or current_price < self.lowest_price:
                self.lowest_price = current_price
                new_sl = self._calculate_trailing_sl(self.lowest_price, side, atr)
                
                # Só atualiza se o novo SL for menor que o atual
                if self.current_trailing_sl is None or new_sl < self.current_trailing_sl:
                    self.current_trailing_sl = new_sl
                    logger.debug(f"📉 Novo trailing SL (SHORT): {new_sl:.2f}")
                    return new_sl
                    
        return None
        
    def _calculate_pnl_pct(self, current_price: float, entry_price: float,
                           side: str) -> float:
        """
        Calcula P&L atual em %.
        
        Args:
            current_price: Preço atual
            entry_price: Preço de entrada
            side: "LONG" ou "SHORT"
            
        Returns:
            P&L em % (positivo = lucro, negativo = perda)
        """
        if entry_price <= 0:
            return 0.0
            
        if side.upper() == "LONG":
            return (current_price - entry_price) / entry_price * 100
        else:  # SHORT
            return (entry_price - current_price) / entry_price * 100
            
    def _calculate_trailing_sl(self, reference_price: float, side: str,
                                atr: Optional[float] = None) -> float:
        """
        Calcula preço do trailing SL.
        
        Args:
            reference_price: Preço de referência (highest/lowest)
            side: "LONG" ou "SHORT"
            atr: ATR atual (para tipo atr)
            
        Returns:
            Preço do trailing SL
        """
        if self.trailing_type == "fixed_pct":
            trail_pct = self.config['trail_pct'] / 100
            distance = reference_price * trail_pct
        elif self.trailing_type == "atr" and atr is not None and atr > 0:
            trail_mult = self.config['trail_atr_mult']
            distance = atr * trail_mult
        else:
            # Fallback para fixed_pct
            trail_pct = self.config['trail_pct'] / 100
            distance = reference_price * trail_pct
            
        if side.upper() == "LONG":
            return reference_price - distance
        else:  # SHORT
            return reference_price + distance
            
    def should_trigger(self, current_price: float, side: str) -> bool:
        """
        Verifica se o trailing stop foi atingido.
        
        Args:
            current_price: Preço atual
            side: "LONG" ou "SHORT"
            
        Returns:
            True se trailing stop foi atingido
        """
        if not self.activated or self.current_trailing_sl is None:
            return False
            
        if side.upper() == "LONG":
            return current_price <= self.current_trailing_sl
        else:  # SHORT
            return current_price >= self.current_trailing_sl
            
    @property
    def is_active(self) -> bool:
        """Check if trailing stop is activated."""
        return self.enabled and self.activated
        
    @property
    def current_stop_loss(self) -> Optional[float]:
        """Get current trailing stop loss price."""
        return self.current_trailing_sl
