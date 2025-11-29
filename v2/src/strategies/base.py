"""
Base Strategy Class and Signal Dataclass

Define a estrutura base para todas as estratégias V2.
Cada estratégia herda de Strategy e implementa generate_signal().
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union
import logging
import time

import numpy as np


class SignalDirection(Enum):
    """Direção do sinal de trading."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """
    Representa um sinal de trading gerado por uma estratégia.
    
    Attributes:
        direction: Direção do sinal (BUY, SELL, HOLD)
        strategy_name: Nome da estratégia que gerou o sinal
        confidence: Confiança do sinal (0.0 a 1.0)
        timestamp: Timestamp Unix do sinal
        reason: Motivo/explicação do sinal
        metadata: Dados adicionais específicos da estratégia
    """
    direction: SignalDirection
    strategy_name: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte o sinal para dict."""
        return {
            'direction': self.direction.value,
            'strategy_name': self.strategy_name,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'reason': self.reason,
            'metadata': self.metadata,
        }


class Strategy(ABC):
    """
    Classe base abstrata para todas as estratégias V2.
    
    ZERO hardcoded - todos os parâmetros vêm do config.
    Cada estratégia pode ser habilitada/desabilitada via config.
    """
    
    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        enabled: bool = True
    ):
        """
        Inicializa a estratégia.
        
        Args:
            name: Nome único da estratégia
            config: Configuração da estratégia (do YAML)
            enabled: Se a estratégia está habilitada
        """
        self.name = name
        self.config = config
        self.enabled = enabled
        self.logger = logging.getLogger(f"Strategy-{name}")
        
        # Performance tracking
        self._total_signals = 0
        self._profitable_signals = 0
        self._last_signal_time: Optional[float] = None
        
        # Drawdown tracking for ensemble
        self._peak_pnl = 0.0
        self._current_pnl = 0.0
        self._current_drawdown = 0.0
        
        # Sharpe tracking (rolling)
        self._returns: list = []
        self._max_returns_history = 252  # ~1 ano de dias úteis
        
        # Day of week filter config - carregado dos filtros da estratégia
        self._day_of_week_filter_config = self._load_day_of_week_filter_config()
        
        self.log(f"Estratégia inicializada | Enabled: {enabled}")
    
    def _load_day_of_week_filter_config(self) -> Dict[str, Any]:
        """
        Carrega configuração do filtro de dia da semana.
        
        Returns:
            Dict com configuração do filtro
        """
        filters = self.config.get('filters', {})
        dow_filter = filters.get('day_of_week_filter', {})
        
        return {
            'enabled': dow_filter.get('enabled', False),
            'allowed_days': dow_filter.get('allowed_days', [0, 1, 2, 3, 4]),
            'monday_multiplier': dow_filter.get('monday_multiplier', 1.0),
            'friday_multiplier': dow_filter.get('friday_multiplier', 1.0),
            'weekend_allowed': dow_filter.get('weekend_allowed', False),
        }
    
    def _check_day_of_week_filter(
        self, timestamp: Union[datetime, float, int, None] = None
    ) -> Tuple[bool, float]:
        """
        Verifica se o dia da semana permite trading.
        
        Args:
            timestamp: Timestamp para verificar (usa hora atual se None)
        
        Returns:
            Tuple[allowed: bool, size_multiplier: float]
            - allowed: True se trading é permitido neste dia
            - size_multiplier: Multiplicador de tamanho de posição (ex: 0.7 na segunda)
        """
        config = self._day_of_week_filter_config
        
        # Se filtro desabilitado, permite tudo com multiplier 1.0
        if not config.get('enabled', False):
            return (True, 1.0)
        
        # Determinar dia da semana
        if timestamp is None:
            dt = datetime.now(timezone.utc)
        elif isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        else:
            dt = timestamp
        
        day_of_week = dt.weekday()  # 0=Monday, 6=Sunday
        
        # Verificar fim de semana
        is_weekend = day_of_week >= 5
        if is_weekend and not config.get('weekend_allowed', False):
            return (False, 0.0)
        
        # Verificar se dia está na lista permitida
        allowed_days = config.get('allowed_days', [0, 1, 2, 3, 4])
        if day_of_week not in allowed_days:
            return (False, 0.0)
        
        # Calcular multiplicador de tamanho
        size_multiplier = 1.0
        
        # Segunda-feira
        if day_of_week == 0:
            size_multiplier = config.get('monday_multiplier', 1.0)
        # Sexta-feira
        elif day_of_week == 4:
            size_multiplier = config.get('friday_multiplier', 1.0)
        
        return (True, size_multiplier)

    @abstractmethod
    def generate_signal(self, market_data: Dict[str, Any]) -> Optional[Signal]:
        """
        Gera um sinal de trading baseado nos dados de mercado.
        
        Args:
            market_data: Dados de mercado (preço, volume, indicadores, etc.)
        
        Returns:
            Signal se houver sinal, None caso contrário
        """
        pass
    
    def log(self, message: str) -> None:
        """Log com prefixo da estratégia."""
        self.logger.info(f"[{self.name}] {message}")
    
    def update_performance(self, pnl: float) -> None:
        """
        Atualiza métricas de performance.
        
        Args:
            pnl: Lucro/prejuízo do último trade
        """
        self._current_pnl += pnl
        self._returns.append(pnl)
        
        # Mantém histórico limitado
        if len(self._returns) > self._max_returns_history:
            self._returns.pop(0)
        
        # Atualiza peak e drawdown
        if self._current_pnl > self._peak_pnl:
            self._peak_pnl = self._current_pnl
        
        if self._peak_pnl > 0:
            self._current_drawdown = (self._peak_pnl - self._current_pnl) / self._peak_pnl
        
        if pnl > 0:
            self._profitable_signals += 1
    
    def get_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calcula Sharpe Ratio rolling.
        
        Args:
            risk_free_rate: Taxa livre de risco
        
        Returns:
            Sharpe ratio ou 0.0 se dados insuficientes
        """
        if len(self._returns) < 2:
            return 0.0
        
        returns = np.array(self._returns)
        mean_return = np.mean(returns) - risk_free_rate
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        # Anualização (assumindo retornos diários)
        return (mean_return / std_return) * np.sqrt(252)
    
    def get_sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calcula Sortino Ratio (usa apenas volatilidade negativa).
        
        Args:
            risk_free_rate: Taxa livre de risco
        
        Returns:
            Sortino ratio ou 0.0 se dados insuficientes
        """
        if len(self._returns) < 2:
            return 0.0
        
        returns = np.array(self._returns)
        negative_returns = returns[returns < 0]
        
        if len(negative_returns) == 0:
            return float('inf')  # Sem retornos negativos
        
        mean_return = np.mean(returns) - risk_free_rate
        downside_std = np.std(negative_returns, ddof=1)
        
        if downside_std == 0:
            return 0.0
        
        return (mean_return / downside_std) * np.sqrt(252)
    
    def get_win_rate(self) -> float:
        """Retorna taxa de acerto."""
        if self._total_signals == 0:
            return 0.0
        return self._profitable_signals / self._total_signals
    
    def get_current_drawdown(self) -> float:
        """Retorna drawdown atual em percentual."""
        return self._current_drawdown
    
    def get_status(self) -> Dict[str, Any]:
        """
        Retorna status atual da estratégia.
        
        Returns:
            Dict com métricas e status
        """
        return {
            'name': self.name,
            'enabled': self.enabled,
            'total_signals': self._total_signals,
            'profitable_signals': self._profitable_signals,
            'win_rate': self.get_win_rate(),
            'sharpe_ratio': self.get_sharpe_ratio(),
            'sortino_ratio': self.get_sortino_ratio(),
            'current_pnl': self._current_pnl,
            'peak_pnl': self._peak_pnl,
            'current_drawdown': self._current_drawdown,
            'last_signal_time': self._last_signal_time,
        }
    
    def reset_performance(self) -> None:
        """Reseta métricas de performance."""
        self._total_signals = 0
        self._profitable_signals = 0
        self._last_signal_time = None
        self._peak_pnl = 0.0
        self._current_pnl = 0.0
        self._current_drawdown = 0.0
        self._returns.clear()
