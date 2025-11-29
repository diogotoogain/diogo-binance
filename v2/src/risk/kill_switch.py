"""
Kill Switch - PROTEÇÃO CRÍTICA!

Este componente é CRÍTICO e NUNCA pode ser desativado.
Protege contra perdas catastróficas pausando o bot automaticamente.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class KillSwitch:
    """
    KILL SWITCH - PROTEÇÃO CRÍTICA!
    
    Parâmetros do config (risk.kill_switch):
    - enabled: true                   # NUNCA DESATIVAR!!!
    - trigger_loss_pct: 5.0           # OPTIMIZE: [3, 5, 7, 10]
    - pause_hours: 24                 # OPTIMIZE: [12, 24, 48]
    
    ⚠️ ESTE COMPONENTE É CRÍTICO!
    - NUNCA pode ser desabilitado
    - Protege contra perdas catastróficas
    - Pausa o bot automaticamente
    """
    
    def __init__(self, config: Dict):
        """
        Initialize kill switch.
        
        Args:
            config: Full configuration dictionary
            
        Raises:
            ValueError: If kill switch is disabled (NEVER allowed!)
        """
        self.config = config['risk']['kill_switch']
        
        # VALIDAÇÃO CRÍTICA - Kill switch DEVE estar ativo!
        if not self.config.get('enabled', True):
            raise ValueError("🚨 KILL SWITCH NÃO PODE SER DESATIVADO!")
            
        self.trigger_loss_pct = self.config['trigger_loss_pct']
        self.pause_hours = self.config['pause_hours']
        self.triggered = False
        self.trigger_time: Optional[datetime] = None
        self._trigger_reason: str = ""
        
    def check(self, daily_pnl: float, balance: float) -> bool:
        """
        Verifica se deve ativar kill switch.
        
        Args:
            daily_pnl: P&L diário atual
            balance: Saldo atual da conta
            
        Returns:
            True se kill switch foi ativado
        """
        if balance <= 0:
            logger.warning("⚠️ Balance <= 0, cannot calculate loss percentage")
            return False
            
        loss_pct = abs(daily_pnl / balance * 100) if daily_pnl < 0 else 0
        
        if daily_pnl < 0 and loss_pct >= self.trigger_loss_pct:
            self.triggered = True
            self.trigger_time = datetime.now(timezone.utc)
            self._trigger_reason = f"Daily loss exceeded threshold: {loss_pct:.2f}%"
            logger.critical(
                f"🚨🚨🚨 KILL SWITCH ATIVADO! "
                f"Loss: {loss_pct:.2f}% >= {self.trigger_loss_pct:.2f}%"
            )
            return True
            
        return False
        
    def is_triggered(self) -> bool:
        """
        Verifica se kill switch está ativo.
        
        Returns:
            True se kill switch está ativo
        """
        if not self.triggered:
            return False
            
        # Verifica se passou o tempo de pausa
        if self.trigger_time:
            elapsed_hours = (datetime.now(timezone.utc) - self.trigger_time).total_seconds() / 3600
            if elapsed_hours >= self.pause_hours:
                self.triggered = False
                self.trigger_time = None
                self._trigger_reason = ""
                logger.info(f"✅ Kill switch desativado após {self.pause_hours}h de pausa")
                return False
                
        return True
        
    def force_trigger(self, reason: str = "Manual") -> None:
        """
        Força ativação do kill switch.
        
        Args:
            reason: Motivo da ativação forçada
        """
        self.triggered = True
        self.trigger_time = datetime.now(timezone.utc)
        self._trigger_reason = reason
        logger.critical(f"🚨 KILL SWITCH FORÇADO: {reason}")
        
    def reset(self) -> None:
        """
        Reset kill switch (use com cuidado!).
        
        ATENÇÃO: Só use em situações onde você tem certeza
        que o problema foi resolvido.
        """
        logger.warning("⚠️ Kill switch sendo resetado manualmente!")
        self.triggered = False
        self.trigger_time = None
        self._trigger_reason = ""
        
    @property
    def trigger_reason(self) -> str:
        """Get the reason for the kill switch trigger."""
        return self._trigger_reason
        
    @property
    def time_until_reset(self) -> Optional[float]:
        """
        Get hours until automatic reset.
        
        Returns:
            Hours until reset, or None if not triggered
        """
        if not self.triggered or not self.trigger_time:
            return None
            
        elapsed_hours = (datetime.now(timezone.utc) - self.trigger_time).total_seconds() / 3600
        remaining = self.pause_hours - elapsed_hours
        return max(0, remaining)
