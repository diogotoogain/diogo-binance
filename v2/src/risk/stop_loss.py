"""
Stop Loss Calculator - Calcula preço de Stop Loss.

Suporta múltiplos métodos: fixed_pct, ATR-based, volatility-based.
"""
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class StopLossCalculator:
    """
    Calcula Stop Loss.
    
    Parâmetros do config (position):
    - sl_type: "atr"                  # TOGGLE: [fixed_pct, atr, volatility]
    - sl_fixed_pct: 1.0               # OPTIMIZE: [0.5, 1.0, 1.5, 2.0, 3.0]
    - sl_atr_multiplier: 1.5          # OPTIMIZE: [0.5, 1.0, 1.5, 2.0, 3.0]
    """
    
    def __init__(self, config: Dict):
        """
        Initialize stop loss calculator.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config['position']
        self.sl_type = self.config['sl_type']
        
    def calculate(self, entry_price: float, side: str, atr: float = None,
                  volatility: float = None) -> float:
        """
        Calcula preço de Stop Loss.
        
        Args:
            entry_price: Preço de entrada
            side: "LONG" ou "SHORT"
            atr: ATR atual (para tipo atr)
            volatility: Volatilidade atual (para tipo volatility)
            
        Returns:
            Preço do stop loss
        """
        if entry_price <= 0:
            logger.warning("Entry price inválido para cálculo de SL")
            return 0.0
            
        # Calcula distância do SL em %
        if self.sl_type == "fixed_pct":
            pct = self.config['sl_fixed_pct'] / 100
        elif self.sl_type == "atr" and atr is not None and atr > 0:
            pct = (atr * self.config['sl_atr_multiplier']) / entry_price
        elif self.sl_type == "volatility" and volatility is not None and volatility > 0:
            # Usa volatilidade como % direta
            pct = volatility * self.config.get('sl_volatility_multiplier', 1.5)
        else:
            # Fallback para fixed_pct
            pct = self.config['sl_fixed_pct'] / 100
            logger.debug(f"Usando fallback para fixed_pct SL: {pct*100:.2f}%")
            
        # Calcula preço do SL baseado no lado
        if side.upper() == "LONG":
            return entry_price * (1 - pct)
        else:  # SHORT
            return entry_price * (1 + pct)
            
    def calculate_distance_pct(self, entry_price: float, stop_loss_price: float,
                                side: str) -> float:
        """
        Calcula distância do SL em %.
        
        Args:
            entry_price: Preço de entrada
            stop_loss_price: Preço do stop loss
            side: "LONG" ou "SHORT"
            
        Returns:
            Distância em % (sempre positivo)
        """
        if entry_price <= 0:
            return 0.0
            
        if side.upper() == "LONG":
            return (entry_price - stop_loss_price) / entry_price * 100
        else:  # SHORT
            return (stop_loss_price - entry_price) / entry_price * 100
            
    def validate_stop_loss(self, entry_price: float, stop_loss_price: float,
                           side: str) -> bool:
        """
        Valida se o stop loss está no lado correto.
        
        Args:
            entry_price: Preço de entrada
            stop_loss_price: Preço do stop loss
            side: "LONG" ou "SHORT"
            
        Returns:
            True se válido
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            return False
            
        if side.upper() == "LONG":
            # SL deve estar abaixo do preço de entrada para LONG
            return stop_loss_price < entry_price
        else:  # SHORT
            # SL deve estar acima do preço de entrada para SHORT
            return stop_loss_price > entry_price
            
    def adjust_for_risk(self, entry_price: float, side: str, 
                        balance: float, position_size: float,
                        max_risk_pct: float) -> float:
        """
        Ajusta SL para respeitar risco máximo por trade.
        
        Args:
            entry_price: Preço de entrada
            side: "LONG" ou "SHORT"
            balance: Saldo da conta
            position_size: Tamanho da posição
            max_risk_pct: Risco máximo por trade em %
            
        Returns:
            Preço do stop loss ajustado
        """
        if balance <= 0 or position_size <= 0 or entry_price <= 0:
            return 0.0
            
        # Quanto podemos perder no máximo
        max_loss = balance * max_risk_pct / 100
        
        # Quanto perdemos por unidade de movimento de preço
        value_at_risk = position_size
        
        # Máximo movimento de preço permitido
        max_price_move = max_loss / value_at_risk
        
        if side.upper() == "LONG":
            return max(0, entry_price - max_price_move)
        else:  # SHORT
            return entry_price + max_price_move
