"""
Throttling Module - Rate Limiting por Estratégia

Implementa rate limiting para evitar overtrading.
Cada estratégia tem seu próprio throttler com limites configuráveis.
ZERO hardcoded - todos os parâmetros vêm do config.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ThrottleConfig:
    """
    Configuração de throttling.
    
    Attributes:
        enabled: Se throttling está habilitado
        max_trades_per_minute: Máximo de trades por minuto
        max_trades_per_hour: Máximo de trades por hora
        max_trades_per_day: Máximo de trades por dia
        cooldown_after_loss_seconds: Cooldown após loss
    """
    enabled: bool = True
    max_trades_per_minute: Optional[int] = None
    max_trades_per_hour: Optional[int] = None
    max_trades_per_day: Optional[int] = None
    cooldown_after_loss_seconds: int = 0


class Throttler:
    """
    Rate limiter para estratégias.
    
    Controla a frequência de trades por estratégia.
    Suporta limites por minuto, hora e dia.
    Implementa cooldown após loss.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa o throttler com config do YAML.
        
        Args:
            config: Configuração de throttling do YAML
        """
        self.enabled = config.get('enabled', True)
        self.max_per_minute = config.get('max_trades_per_minute')
        self.max_per_hour = config.get('max_trades_per_hour')
        self.max_per_day = config.get('max_trades_per_day')
        self.cooldown_after_loss = config.get('cooldown_after_loss_seconds', 0)
        
        # Histórico de trades
        self._trades: list = []  # Lista de timestamps
        self._last_loss_time: Optional[float] = None
        
        # Constantes de tempo
        self._MINUTE = 60
        self._HOUR = 3600
        self._DAY = 86400
    
    def can_trade(self) -> bool:
        """
        Verifica se pode fazer trade agora.
        
        Returns:
            True se pode, False se throttled
        """
        if not self.enabled:
            return True
        
        now = time.time()
        
        # Verifica cooldown após loss
        if self._last_loss_time is not None:
            if now - self._last_loss_time < self.cooldown_after_loss:
                return False
        
        # Remove trades antigos
        self._cleanup_old_trades(now)
        
        # Verifica limites
        if self.max_per_minute is not None:
            minute_trades = sum(1 for t in self._trades if now - t < self._MINUTE)
            if minute_trades >= self.max_per_minute:
                return False
        
        if self.max_per_hour is not None:
            hour_trades = sum(1 for t in self._trades if now - t < self._HOUR)
            if hour_trades >= self.max_per_hour:
                return False
        
        if self.max_per_day is not None:
            day_trades = sum(1 for t in self._trades if now - t < self._DAY)
            if day_trades >= self.max_per_day:
                return False
        
        return True
    
    def record_trade(self, is_loss: bool = False) -> None:
        """
        Registra um trade executado.
        
        Args:
            is_loss: Se foi uma loss (para cooldown)
        """
        now = time.time()
        self._trades.append(now)
        
        if is_loss:
            self._last_loss_time = now
    
    def _cleanup_old_trades(self, now: float) -> None:
        """Remove trades com mais de 1 dia."""
        self._trades = [t for t in self._trades if now - t < self._DAY]
    
    def get_remaining_trades(self) -> Dict[str, Optional[int]]:
        """
        Retorna quantos trades restam em cada período.
        
        Returns:
            Dict com trades restantes por período
        """
        now = time.time()
        self._cleanup_old_trades(now)
        
        remaining = {}
        
        if self.max_per_minute is not None:
            minute_trades = sum(1 for t in self._trades if now - t < self._MINUTE)
            remaining['minute'] = self.max_per_minute - minute_trades
        
        if self.max_per_hour is not None:
            hour_trades = sum(1 for t in self._trades if now - t < self._HOUR)
            remaining['hour'] = self.max_per_hour - hour_trades
        
        if self.max_per_day is not None:
            day_trades = sum(1 for t in self._trades if now - t < self._DAY)
            remaining['day'] = self.max_per_day - day_trades
        
        return remaining
    
    def get_cooldown_remaining(self) -> float:
        """
        Retorna tempo restante de cooldown após loss.
        
        Returns:
            Segundos restantes ou 0 se não está em cooldown
        """
        if self._last_loss_time is None:
            return 0.0
        
        elapsed = time.time() - self._last_loss_time
        remaining = self.cooldown_after_loss - elapsed
        
        return max(0.0, remaining)
    
    def reset(self) -> None:
        """Reseta estado do throttler."""
        self._trades.clear()
        self._last_loss_time = None
    
    def get_status(self) -> Dict[str, Any]:
        """
        Retorna status do throttler.
        
        Returns:
            Dict com status atual
        """
        return {
            'enabled': self.enabled,
            'remaining': self.get_remaining_trades(),
            'cooldown_remaining': self.get_cooldown_remaining(),
            'total_trades_today': len(self._trades),
        }
