"""
Position Sizer - Calcula tamanho da posição.

Suporta múltiplos métodos de sizing: fixed, Kelly, vol_target.
"""
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Calcula tamanho da posição.
    
    Parâmetros do config (bet_sizing):
    - method: "kelly"                 # TOGGLE: [fixed, kelly, vol_target, rl]
    - fixed.size_pct: 2.0             # OPTIMIZE: [1, 2, 3, 5]
    - kelly.fraction: 0.25            # OPTIMIZE: [0.1, 0.25, 0.5, 0.75]
    - kelly.max_size_pct: 10.0        # OPTIMIZE: [5, 10, 15, 20]
    - vol_target.annual_vol: 0.15     # OPTIMIZE: [0.10, 0.15, 0.20, 0.25]
    """
    
    # Annualization factor (trading days per year)
    TRADING_DAYS_PER_YEAR = 252
    
    def __init__(self, config: Dict):
        """
        Initialize position sizer.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config
        self.method = config['bet_sizing']['method']
        
    def calculate(self, balance: float, current_price: float, 
                  atr: float, signal_confidence: float = 1.0) -> float:
        """
        Calcula tamanho da posição baseado no método configurado.
        
        Args:
            balance: Saldo atual da conta
            current_price: Preço atual do ativo
            atr: Average True Range atual
            signal_confidence: Confiança do sinal [0, 1]
            
        Returns:
            Tamanho da posição em unidades do ativo (ex: BTC)
        """
        if balance <= 0 or current_price <= 0:
            logger.warning("Balance ou preço inválido para cálculo de posição")
            return 0.0
            
        if self.method == "fixed":
            return self._fixed_size(balance, current_price)
        elif self.method == "kelly":
            return self._kelly_size(balance, current_price, signal_confidence)
        elif self.method == "vol_target":
            return self._vol_target_size(balance, current_price, atr)
        else:
            logger.warning(f"Método desconhecido: {self.method}, usando fixed")
            return self._fixed_size(balance, current_price)
            
    def _fixed_size(self, balance: float, price: float) -> float:
        """
        Tamanho fixo como % do balance.
        
        Args:
            balance: Saldo atual
            price: Preço atual
            
        Returns:
            Tamanho em unidades do ativo
        """
        pct = self.config['bet_sizing']['fixed']['size_pct']
        size_usd = balance * pct / 100
        return size_usd / price
        
    def _kelly_size(self, balance: float, price: float, confidence: float) -> float:
        """
        Fração de Kelly para sizing.
        
        f* = (p * b - q) / b
        
        Onde:
        - p = probabilidade de ganho (usamos confidence)
        - q = 1 - p
        - b = ratio de ganho/perda (assumimos 1:1 inicialmente)
        
        Args:
            balance: Saldo atual
            price: Preço atual
            confidence: Confiança do sinal (usada como prob de ganho)
            
        Returns:
            Tamanho em unidades do ativo
        """
        kelly_config = self.config['bet_sizing']['kelly']
        fraction = kelly_config['fraction']
        max_pct = kelly_config['max_size_pct']
        
        # Parâmetros Kelly
        p = max(0.01, min(0.99, confidence))  # Clamp entre 0.01 e 0.99
        q = 1 - p
        b = 1.0  # Risk-reward ratio (assumido 1:1)
        
        # Fórmula de Kelly
        kelly_fraction = (p * b - q) / b
        kelly_fraction = max(0, kelly_fraction)  # Não permite negativo
        
        # Aplica fração do Kelly (half-Kelly, quarter-Kelly, etc)
        adjusted = kelly_fraction * fraction
        
        # Limite máximo
        size_pct = min(adjusted * 100, max_pct)
        
        # Converte para unidades do ativo
        size_usd = balance * size_pct / 100
        return size_usd / price
        
    def _vol_target_size(self, balance: float, price: float, atr: float) -> float:
        """
        Volatility targeting - ajusta tamanho para atingir volatilidade alvo.
        
        Args:
            balance: Saldo atual
            price: Preço atual
            atr: Average True Range
            
        Returns:
            Tamanho em unidades do ativo
        """
        target_vol = self.config['bet_sizing']['vol_target']['annual_vol']
        
        # Calcula volatilidade diária estimada
        if price > 0 and atr > 0:
            daily_vol = atr / price  # Volatilidade diária aproximada
            annual_vol_estimate = daily_vol * (self.TRADING_DAYS_PER_YEAR ** 0.5)
        else:
            annual_vol_estimate = 0.0
            
        # Calcula multiplicador de tamanho
        if annual_vol_estimate > 0:
            size_mult = target_vol / annual_vol_estimate
        else:
            size_mult = 1.0
            
        # Tamanho base (2% do balance)
        base_pct = 2.0  # Poderia ser configurável
        base_size = balance * base_pct / 100 / price
        
        # Aplica multiplicador
        adjusted_size = base_size * size_mult
        
        # Limita para não ser excessivo
        max_size = balance * 0.1 / price  # Max 10% do balance
        return min(adjusted_size, max_size)
        
    def calculate_with_stop_loss(self, balance: float, current_price: float,
                                  stop_loss_price: float, 
                                  risk_per_trade_pct: float) -> float:
        """
        Calcula tamanho baseado no risco por trade e stop loss.
        
        Args:
            balance: Saldo atual
            current_price: Preço de entrada
            stop_loss_price: Preço do stop loss
            risk_per_trade_pct: % do balance a arriscar
            
        Returns:
            Tamanho em unidades do ativo
        """
        if balance <= 0 or current_price <= 0:
            return 0.0
            
        # Calcula risco por unidade
        risk_per_unit = abs(current_price - stop_loss_price)
        
        if risk_per_unit <= 0:
            logger.warning("Risco por unidade <= 0, não pode calcular posição")
            return 0.0
            
        # Quanto podemos arriscar em USD
        risk_amount = balance * risk_per_trade_pct / 100
        
        # Quantas unidades podemos comprar
        position_size = risk_amount / risk_per_unit
        
        return position_size
