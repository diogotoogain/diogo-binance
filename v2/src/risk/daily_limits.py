"""
Daily Limits - Controle de limites diários, semanais e mensais.

Monitora e controla os limites de trading para evitar perdas excessivas.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class DailyLimits:
    """
    Controla limites de trading diários, semanais e mensais.
    
    Parâmetros do config (risk):
    - max_daily_loss_pct: 3.0         # OPTIMIZE: [2, 3, 5, 7]
    - max_daily_trades: 50            # OPTIMIZE: [20, 50, 100]
    - max_weekly_loss_pct: 7.0        # OPTIMIZE: [5, 7, 10, 15]
    - max_monthly_loss_pct: 15.0      # OPTIMIZE: [10, 15, 20, 25]
    """
    
    def __init__(self, config: Dict):
        """
        Initialize daily limits manager.
        
        Args:
            config: Full configuration dictionary
        """
        self.config = config['risk']
        
        # Limites do config
        self.max_daily_loss_pct = self.config['max_daily_loss_pct']
        self.max_daily_trades = self.config.get('max_daily_trades', 50)
        self.max_weekly_loss_pct = self.config.get('max_weekly_loss_pct', 7.0)
        self.max_monthly_loss_pct = self.config.get('max_monthly_loss_pct', 15.0)
        
        # Estado
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        self._monthly_pnl = 0.0
        self._daily_trades = 0
        self._initial_balance = 0.0
        
        # Timestamps para reset
        self._last_daily_reset = datetime.now(timezone.utc)
        self._last_weekly_reset = datetime.now(timezone.utc)
        self._last_monthly_reset = datetime.now(timezone.utc)
        
    def set_initial_balance(self, balance: float) -> None:
        """
        Define saldo inicial para cálculos de porcentagem.
        
        Args:
            balance: Saldo inicial
        """
        self._initial_balance = balance
        
    def update_pnl(self, pnl: float) -> None:
        """
        Atualiza P&L após um trade.
        
        Args:
            pnl: P&L do trade
        """
        self._check_and_reset()
        
        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        self._monthly_pnl += pnl
        self._daily_trades += 1
        
        logger.debug(
            f"P&L atualizado: Daily={self._daily_pnl:.2f}, "
            f"Weekly={self._weekly_pnl:.2f}, Monthly={self._monthly_pnl:.2f}"
        )
        
    def is_exceeded(self) -> bool:
        """
        Verifica se algum limite foi excedido.
        
        Returns:
            True se algum limite foi excedido
        """
        self._check_and_reset()
        
        exceeded, _ = self.check_all_limits()
        return exceeded
        
    def check_all_limits(self) -> Tuple[bool, str]:
        """
        Verifica todos os limites.
        
        Returns:
            Tuple (exceeded, reason)
        """
        self._check_and_reset()
        
        if self._initial_balance <= 0:
            return False, "Balance not set"
            
        # Check daily trades limit
        if self._daily_trades >= self.max_daily_trades:
            msg = f"❌ Daily trades limit reached: {self._daily_trades}"
            logger.warning(msg)
            return True, msg
            
        # Check daily loss limit
        daily_loss_pct = abs(self._daily_pnl / self._initial_balance * 100)
        if self._daily_pnl < 0 and daily_loss_pct >= self.max_daily_loss_pct:
            msg = f"❌ Daily loss limit exceeded: {daily_loss_pct:.2f}%"
            logger.warning(msg)
            return True, msg
            
        # Check weekly loss limit
        weekly_loss_pct = abs(self._weekly_pnl / self._initial_balance * 100)
        if self._weekly_pnl < 0 and weekly_loss_pct >= self.max_weekly_loss_pct:
            msg = f"❌ Weekly loss limit exceeded: {weekly_loss_pct:.2f}%"
            logger.warning(msg)
            return True, msg
            
        # Check monthly loss limit
        monthly_loss_pct = abs(self._monthly_pnl / self._initial_balance * 100)
        if self._monthly_pnl < 0 and monthly_loss_pct >= self.max_monthly_loss_pct:
            msg = f"❌ Monthly loss limit exceeded: {monthly_loss_pct:.2f}%"
            logger.warning(msg)
            return True, msg
            
        return False, "✅ Within limits"
        
    def _check_and_reset(self) -> None:
        """Verifica e reseta contadores se necessário."""
        now = datetime.now(timezone.utc)
        
        # Reset diário (novo dia)
        if now.date() > self._last_daily_reset.date():
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._last_daily_reset = now
            logger.info("📅 Reset diário de limites")
            
        # Reset semanal (nova semana)
        days_since_weekly = (now - self._last_weekly_reset).days
        if days_since_weekly >= 7:
            self._weekly_pnl = 0.0
            self._last_weekly_reset = now
            logger.info("📆 Reset semanal de limites")
            
        # Reset mensal (novo mês - considerando ano também)
        if (now.year > self._last_monthly_reset.year or 
            (now.year == self._last_monthly_reset.year and 
             now.month > self._last_monthly_reset.month)):
            self._monthly_pnl = 0.0
            self._last_monthly_reset = now
            logger.info("🗓️ Reset mensal de limites")
            
    def reset_daily(self) -> None:
        """Força reset dos contadores diários."""
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._last_daily_reset = datetime.now(timezone.utc)
        logger.info("📅 Reset manual de limites diários")
        
    def reset_all(self) -> None:
        """Força reset de todos os contadores."""
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        self._monthly_pnl = 0.0
        self._daily_trades = 0
        now = datetime.now(timezone.utc)
        self._last_daily_reset = now
        self._last_weekly_reset = now
        self._last_monthly_reset = now
        logger.info("🔄 Reset total de limites")
        
    @property
    def daily_pnl(self) -> float:
        """Get current daily P&L."""
        return self._daily_pnl
        
    @property
    def weekly_pnl(self) -> float:
        """Get current weekly P&L."""
        return self._weekly_pnl
        
    @property
    def monthly_pnl(self) -> float:
        """Get current monthly P&L."""
        return self._monthly_pnl
        
    @property
    def daily_trades(self) -> int:
        """Get current daily trade count."""
        return self._daily_trades
        
    def get_remaining_capacity(self) -> Dict[str, float]:
        """
        Get remaining capacity before hitting limits.
        
        Returns:
            Dictionary with remaining capacity for each limit
        """
        if self._initial_balance <= 0:
            return {}
            
        return {
            'daily_trades_remaining': max(0, self.max_daily_trades - self._daily_trades),
            'daily_loss_remaining_pct': max(
                0, 
                self.max_daily_loss_pct - abs(self._daily_pnl / self._initial_balance * 100)
            ) if self._daily_pnl < 0 else self.max_daily_loss_pct,
            'weekly_loss_remaining_pct': max(
                0,
                self.max_weekly_loss_pct - abs(self._weekly_pnl / self._initial_balance * 100)
            ) if self._weekly_pnl < 0 else self.max_weekly_loss_pct,
            'monthly_loss_remaining_pct': max(
                0,
                self.max_monthly_loss_pct - abs(self._monthly_pnl / self._initial_balance * 100)
            ) if self._monthly_pnl < 0 else self.max_monthly_loss_pct,
        }
