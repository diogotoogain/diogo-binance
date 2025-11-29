"""
Take Profit Calculator - Calcula preço de Take Profit.

Suporta múltiplos métodos: fixed_pct, ATR-based, R:R ratio.
"""
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TakeProfitCalculator:
    """
    Calcula Take Profit.
    
    Parâmetros do config (position):
    - tp_type: "atr"                  # TOGGLE: [fixed_pct, atr, rr_ratio]
    - tp_fixed_pct: 2.0               # OPTIMIZE: [1.0, 1.5, 2.0, 3.0, 5.0]
    - tp_atr_multiplier: 2.5          # OPTIMIZE: [1.0, 1.5, 2.0, 2.5, 3.0, 5.0]
    - tp_rr_ratio: 2.0                # OPTIMIZE: [1.5, 2.0, 2.5, 3.0]
    """
    
    def __init__(self, config: Dict):
        """
        Initialize take profit calculator.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config['position']
        self.tp_type = self.config['tp_type']
        
    def calculate(self, entry_price: float, side: str, 
                  stop_loss: Optional[float] = None, 
                  atr: Optional[float] = None) -> float:
        """
        Calcula preço de Take Profit.
        
        Args:
            entry_price: Preço de entrada
            side: "LONG" ou "SHORT"
            stop_loss: Preço do stop loss (para rr_ratio)
            atr: ATR atual (para tipo atr)
            
        Returns:
            Preço do take profit
        """
        if entry_price <= 0:
            logger.warning("Entry price inválido para cálculo de TP")
            return 0.0
            
        # Calcula distância do TP em %
        if self.tp_type == "fixed_pct":
            pct = self.config['tp_fixed_pct'] / 100
        elif self.tp_type == "atr" and atr is not None and atr > 0:
            pct = (atr * self.config['tp_atr_multiplier']) / entry_price
        elif self.tp_type == "rr_ratio" and stop_loss is not None:
            pct = self._calculate_rr_ratio_pct(entry_price, stop_loss, side)
        else:
            # Fallback para fixed_pct
            pct = self.config['tp_fixed_pct'] / 100
            logger.debug(f"Usando fallback para fixed_pct TP: {pct*100:.2f}%")
            
        # Calcula preço do TP baseado no lado
        if side.upper() == "LONG":
            return entry_price * (1 + pct)
        else:  # SHORT
            return entry_price * (1 - pct)
            
    def _calculate_rr_ratio_pct(self, entry_price: float, stop_loss: float,
                                 side: str) -> float:
        """
        Calcula % do TP baseado no ratio R:R e distância do SL.
        
        Args:
            entry_price: Preço de entrada
            stop_loss: Preço do stop loss
            side: "LONG" ou "SHORT"
            
        Returns:
            % de distância para o TP
        """
        rr_ratio = self.config['tp_rr_ratio']
        
        # Calcula distância do SL
        sl_distance = abs(entry_price - stop_loss)
        
        # TP distance = SL distance * R:R ratio
        tp_distance = sl_distance * rr_ratio
        
        # Converte para %
        return tp_distance / entry_price
        
    def calculate_distance_pct(self, entry_price: float, take_profit_price: float,
                                side: str) -> float:
        """
        Calcula distância do TP em %.
        
        Args:
            entry_price: Preço de entrada
            take_profit_price: Preço do take profit
            side: "LONG" ou "SHORT"
            
        Returns:
            Distância em % (sempre positivo)
        """
        if entry_price <= 0:
            return 0.0
            
        if side.upper() == "LONG":
            return (take_profit_price - entry_price) / entry_price * 100
        else:  # SHORT
            return (entry_price - take_profit_price) / entry_price * 100
            
    def validate_take_profit(self, entry_price: float, take_profit_price: float,
                              side: str) -> bool:
        """
        Valida se o take profit está no lado correto.
        
        Args:
            entry_price: Preço de entrada
            take_profit_price: Preço do take profit
            side: "LONG" ou "SHORT"
            
        Returns:
            True se válido
        """
        if entry_price <= 0 or take_profit_price <= 0:
            return False
            
        if side.upper() == "LONG":
            # TP deve estar acima do preço de entrada para LONG
            return take_profit_price > entry_price
        else:  # SHORT
            # TP deve estar abaixo do preço de entrada para SHORT
            return take_profit_price < entry_price
            
    def calculate_reward_risk_ratio(self, entry_price: float, 
                                     stop_loss_price: float,
                                     take_profit_price: float) -> float:
        """
        Calcula o ratio Reward:Risk.
        
        Args:
            entry_price: Preço de entrada
            stop_loss_price: Preço do stop loss
            take_profit_price: Preço do take profit
            
        Returns:
            R:R ratio (ex: 2.0 significa ganho potencial 2x a perda potencial)
        """
        risk = abs(entry_price - stop_loss_price)
        reward = abs(take_profit_price - entry_price)
        
        if risk <= 0:
            return 0.0
            
        return reward / risk
