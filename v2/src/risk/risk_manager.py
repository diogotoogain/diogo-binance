"""
Risk Manager - Gerenciador principal de risco.

Orquestra todos os componentes de gestão de risco.
"""
import logging
from typing import Dict, Tuple, Optional

from .kill_switch import KillSwitch
from .daily_limits import DailyLimits
from .position_sizer import PositionSizer
from .stop_loss import StopLossCalculator
from .take_profit import TakeProfitCalculator
from .trailing_stop import TrailingStop
from .regime_adjustment import RegimeAdjustment

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Gerenciador principal de risco.
    
    Parâmetros do config (risk):
    - risk_per_trade_pct: 1.0         # OPTIMIZE: [0.25, 0.5, 1.0, 1.5, 2.0]
    - max_position_size_pct: 5.0      # OPTIMIZE: [2, 5, 10, 15, 20]
    - max_daily_loss_pct: 3.0         # OPTIMIZE: [2, 3, 5, 7]
    - max_daily_trades: 50            # OPTIMIZE: [20, 50, 100]
    - max_weekly_loss_pct: 7.0        # OPTIMIZE: [5, 7, 10, 15]
    - max_monthly_loss_pct: 15.0      # OPTIMIZE: [10, 15, 20, 25]
    - max_drawdown_pct: 20.0          # OPTIMIZE: [10, 15, 20, 25, 30]
    - max_concurrent_positions: 1     # OPTIMIZE: [1, 2, 3, 5]
    - max_leverage: 10                # OPTIMIZE: [3, 5, 10, 15, 20]
    """
    
    def __init__(self, config: Dict):
        """
        Initialize risk manager.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config['risk']
        self.full_config = config
        
        # Componentes de risco
        self.kill_switch = KillSwitch(config)
        self.daily_limits = DailyLimits(config)
        self.position_sizer = PositionSizer(config)
        self.stop_loss_calculator = StopLossCalculator(config)
        self.take_profit_calculator = TakeProfitCalculator(config)
        self.trailing_stop = TrailingStop(config)
        self.regime_adjuster = RegimeAdjustment(config)
        
        # Estado
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        self._monthly_pnl = 0.0
        self._current_drawdown = 0.0
        self._peak_balance = 0.0
        self._open_positions = 0
        self._initial_balance = 0.0
        
    def set_initial_balance(self, balance: float) -> None:
        """
        Define saldo inicial.
        
        Args:
            balance: Saldo inicial
        """
        self._initial_balance = balance
        self._peak_balance = balance
        self.daily_limits.set_initial_balance(balance)
        logger.info(f"💰 Saldo inicial definido: ${balance:.2f}")
        
    def can_open_position(self, balance: float, 
                          regime: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Verifica se pode abrir nova posição.
        
        Checks:
        1. Kill switch não ativo
        2. Dentro dos limites diários
        3. Drawdown aceitável
        4. Posições concorrentes
        5. Regime permite
        
        Args:
            balance: Saldo atual
            regime: Informações do regime atual (opcional)
            
        Returns:
            (pode_abrir, motivo)
        """
        # Kill switch é SEMPRE o primeiro check!
        if self.kill_switch.is_triggered():
            reason = self.kill_switch.trigger_reason or "KILL SWITCH ATIVO"
            return False, f"🚨 {reason}"
            
        # Verifica limites diários
        if self.daily_limits.is_exceeded():
            exceeded, reason = self.daily_limits.check_all_limits()
            return False, reason
            
        # Verifica drawdown máximo
        if self._current_drawdown >= self.config['max_drawdown_pct']:
            return False, f"❌ Drawdown máximo atingido: {self._current_drawdown:.1f}%"
            
        # Verifica número de posições concorrentes
        max_positions = self.config['max_concurrent_positions']
        if self._open_positions >= max_positions:
            return False, f"❌ Máximo de posições atingido: {self._open_positions}/{max_positions}"
            
        # Verifica regime (se fornecido)
        if regime:
            regime_profile = self.regime_adjuster.get_regime_risk_profile(regime)
            if not regime_profile['allow_new_positions']:
                return False, f"❌ Regime não permite novas posições: {regime_profile['recommendation']}"
                
        return True, "✅ OK para abrir posição"
        
    def calculate_position_size(self, balance: float, current_price: float, 
                                 atr: float, signal_confidence: float = 1.0,
                                 regime: Optional[Dict] = None) -> float:
        """
        Calcula tamanho da posição.
        
        Considera:
        - Risk per trade
        - Ajuste por regime
        - Bet sizing method (Kelly, vol target, etc)
        
        Args:
            balance: Saldo atual
            current_price: Preço atual
            atr: Average True Range
            signal_confidence: Confiança do sinal
            regime: Informações do regime (opcional)
            
        Returns:
            Tamanho da posição em unidades do ativo
        """
        # Calcula tamanho base
        base_size = self.position_sizer.calculate(
            balance=balance,
            current_price=current_price,
            atr=atr,
            signal_confidence=signal_confidence
        )
        
        # Ajuste por regime
        if regime:
            regime_mult = self.regime_adjuster.get_multiplier(regime)
            adjusted_size = base_size * regime_mult
        else:
            adjusted_size = base_size
            
        # Aplica limite máximo de posição
        max_position_pct = self.config['max_position_size_pct']
        max_size = (balance * max_position_pct / 100) / current_price
        
        final_size = min(adjusted_size, max_size)
        
        logger.debug(
            f"Position sizing: base={base_size:.6f}, "
            f"adjusted={adjusted_size:.6f}, max={max_size:.6f}, "
            f"final={final_size:.6f}"
        )
        
        return final_size
        
    def calculate_stop_loss(self, entry_price: float, side: str,
                            atr: Optional[float] = None) -> float:
        """
        Calcula preço do stop loss.
        
        Args:
            entry_price: Preço de entrada
            side: "LONG" ou "SHORT"
            atr: ATR atual
            
        Returns:
            Preço do stop loss
        """
        return self.stop_loss_calculator.calculate(entry_price, side, atr)
        
    def calculate_take_profit(self, entry_price: float, side: str,
                               stop_loss: Optional[float] = None,
                               atr: Optional[float] = None) -> float:
        """
        Calcula preço do take profit.
        
        Args:
            entry_price: Preço de entrada
            side: "LONG" ou "SHORT"
            stop_loss: Preço do stop loss
            atr: ATR atual
            
        Returns:
            Preço do take profit
        """
        return self.take_profit_calculator.calculate(entry_price, side, stop_loss, atr)
        
    def update_trailing_stop(self, current_price: float, entry_price: float,
                              side: str, atr: Optional[float] = None) -> Optional[float]:
        """
        Atualiza trailing stop.
        
        Args:
            current_price: Preço atual
            entry_price: Preço de entrada
            side: "LONG" ou "SHORT"
            atr: ATR atual
            
        Returns:
            Novo preço de SL ou None se não mudou
        """
        return self.trailing_stop.update(current_price, entry_price, side, atr)
        
    def reset_trailing_stop(self) -> None:
        """Reset trailing stop para nova posição."""
        self.trailing_stop.reset()
        
    def update_pnl(self, pnl: float, balance: float) -> None:
        """
        Atualiza P&L e verifica limites.
        
        Args:
            pnl: P&L do trade
            balance: Saldo atual
        """
        # Validação de PnL suspeito
        if pnl is None:
            logger.error("❌ PnL None ignorado")
            return
            
        if balance > 0 and abs(pnl) > balance:
            logger.error(f"❌ PnL suspeito ignorado: {pnl} (maior que balance {balance})")
            return
            
        # Atualiza P&L
        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        self._monthly_pnl += pnl
        
        # Atualiza nos daily limits
        self.daily_limits.update_pnl(pnl)
        
        # Atualiza drawdown
        current_balance = balance
        if current_balance > self._peak_balance:
            self._peak_balance = current_balance
            
        if self._peak_balance > 0:
            self._current_drawdown = (
                (self._peak_balance - current_balance) / self._peak_balance * 100
            )
        else:
            self._current_drawdown = 0.0
            
        # Verifica kill switch
        self.kill_switch.check(self._daily_pnl, balance)
        
        logger.debug(
            f"P&L atualizado: daily={self._daily_pnl:.2f}, "
            f"drawdown={self._current_drawdown:.2f}%"
        )
        
    def on_position_opened(self) -> None:
        """Callback quando posição é aberta."""
        self._open_positions += 1
        logger.debug(f"Posição aberta. Total: {self._open_positions}")
        
    def on_position_closed(self, pnl: float, balance: float) -> None:
        """
        Callback quando posição é fechada.
        
        Args:
            pnl: P&L da posição
            balance: Saldo atual
        """
        self._open_positions = max(0, self._open_positions - 1)
        self.update_pnl(pnl, balance)
        self.reset_trailing_stop()
        logger.debug(f"Posição fechada. P&L: {pnl:.2f}. Total: {self._open_positions}")
        
    def reset_daily(self) -> None:
        """Reset contadores diários."""
        self._daily_pnl = 0.0
        self.daily_limits.reset_daily()
        logger.info("📅 Reset diário de risco")
        
    def reset_weekly(self) -> None:
        """Reset contadores semanais."""
        self._weekly_pnl = 0.0
        logger.info("📆 Reset semanal de risco")
        
    def reset_monthly(self) -> None:
        """Reset contadores mensais."""
        self._monthly_pnl = 0.0
        logger.info("🗓️ Reset mensal de risco")
        
    def force_kill_switch(self, reason: str = "Manual") -> None:
        """
        Força ativação do kill switch.
        
        Args:
            reason: Motivo da ativação
        """
        self.kill_switch.force_trigger(reason)
        
    @property
    def daily_pnl(self) -> float:
        """Get daily P&L."""
        return self._daily_pnl
        
    @property
    def weekly_pnl(self) -> float:
        """Get weekly P&L."""
        return self._weekly_pnl
        
    @property
    def monthly_pnl(self) -> float:
        """Get monthly P&L."""
        return self._monthly_pnl
        
    @property
    def current_drawdown(self) -> float:
        """Get current drawdown %."""
        return self._current_drawdown
        
    @property
    def open_positions(self) -> int:
        """Get number of open positions."""
        return self._open_positions
        
    @property
    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active."""
        return self.kill_switch.is_triggered()
        
    def get_risk_summary(self) -> Dict:
        """
        Get summary of current risk state.
        
        Returns:
            Dictionary with risk metrics
        """
        return {
            'daily_pnl': self._daily_pnl,
            'weekly_pnl': self._weekly_pnl,
            'monthly_pnl': self._monthly_pnl,
            'current_drawdown_pct': self._current_drawdown,
            'peak_balance': self._peak_balance,
            'open_positions': self._open_positions,
            'kill_switch_active': self.kill_switch.is_triggered(),
            'kill_switch_reason': self.kill_switch.trigger_reason,
            'daily_limits_exceeded': self.daily_limits.is_exceeded(),
            'remaining_capacity': self.daily_limits.get_remaining_capacity(),
        }
