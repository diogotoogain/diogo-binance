"""
Bar Builder - Construtor de barras (Time, Volume, Dollar)

Features:
- Time bars (padrão)
- Volume bars
- Dollar bars
- Factory function para criar builder
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np


@dataclass
class Bar:
    """Representa uma barra OHLCV."""
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int
    dollar_volume: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'open_time': self.open_time,
            'close_time': self.close_time,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'trades': self.trades,
            'dollar_volume': self.dollar_volume
        }


class BarBuilder(ABC):
    """
    Classe base abstrata para construção de barras.
    
    Subclasses implementam diferentes métodos de agregação:
    - TimeBarBuilder: Barras baseadas em tempo
    - VolumeBarBuilder: Barras baseadas em volume
    - DollarBarBuilder: Barras baseadas em valor em dólares
    """
    
    def __init__(self):
        """Inicializa o builder."""
        self._current_bar: Optional[Dict] = None
        self._completed_bars: List[Bar] = []
    
    @abstractmethod
    def add_tick(self, timestamp: datetime, price: float, volume: float) -> Optional[Bar]:
        """
        Adiciona um tick e retorna barra completa se formada.
        
        Args:
            timestamp: Timestamp do tick
            price: Preço do tick
            volume: Volume do tick
            
        Returns:
            Bar completa se formada, None caso contrário
        """
        pass
    
    @abstractmethod
    def build_from_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        Constrói barras a partir de DataFrame de trades.
        
        Args:
            trades: DataFrame com colunas [timestamp, price, volume]
            
        Returns:
            DataFrame com barras OHLCV
        """
        pass
    
    def _start_new_bar(self, timestamp: datetime, price: float) -> None:
        """Inicia uma nova barra."""
        self._current_bar = {
            'open_time': timestamp,
            'close_time': timestamp,
            'open': price,
            'high': price,
            'low': price,
            'close': price,
            'volume': 0.0,
            'trades': 0,
            'dollar_volume': 0.0
        }
    
    def _update_bar(self, timestamp: datetime, price: float, volume: float) -> None:
        """Atualiza a barra atual com novo tick."""
        if self._current_bar is None:
            self._start_new_bar(timestamp, price)
            return
        
        self._current_bar['close_time'] = timestamp
        self._current_bar['high'] = max(self._current_bar['high'], price)
        self._current_bar['low'] = min(self._current_bar['low'], price)
        self._current_bar['close'] = price
        self._current_bar['volume'] += volume
        self._current_bar['trades'] += 1
        self._current_bar['dollar_volume'] += price * volume
    
    def _complete_bar(self) -> Bar:
        """Completa a barra atual e retorna."""
        bar = Bar(**self._current_bar)
        self._completed_bars.append(bar)
        self._current_bar = None
        return bar
    
    def get_completed_bars(self) -> List[Bar]:
        """Retorna lista de barras completas."""
        return self._completed_bars.copy()
    
    def reset(self) -> None:
        """Reseta o estado do builder."""
        self._current_bar = None
        self._completed_bars = []


class TimeBarBuilder(BarBuilder):
    """
    Construtor de barras baseadas em tempo.
    
    Agrupa ticks em intervalos de tempo fixos.
    """
    
    def __init__(self, interval_seconds: int = 60):
        """
        Inicializa o builder.
        
        Args:
            interval_seconds: Intervalo em segundos (60 = 1 minuto)
        """
        super().__init__()
        self.interval_seconds = interval_seconds
        self._bar_start_time: Optional[datetime] = None
    
    def add_tick(self, timestamp: datetime, price: float, volume: float) -> Optional[Bar]:
        """Adiciona tick e retorna barra se intervalo completou."""
        # Primeira tick
        if self._bar_start_time is None:
            self._bar_start_time = timestamp
            self._start_new_bar(timestamp, price)
        
        # Verifica se nova barra deve começar
        elapsed = (timestamp - self._bar_start_time).total_seconds()
        
        if elapsed >= self.interval_seconds:
            # Completa barra anterior
            completed = self._complete_bar()
            
            # Inicia nova barra
            self._bar_start_time = timestamp
            self._start_new_bar(timestamp, price)
            self._update_bar(timestamp, price, volume)
            
            return completed
        
        # Atualiza barra atual
        self._update_bar(timestamp, price, volume)
        return None
    
    def build_from_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Constrói barras de tempo a partir de trades."""
        self.reset()
        
        # Garante colunas necessárias
        if 'time' in trades.columns:
            trades = trades.rename(columns={'time': 'timestamp'})
        
        required_cols = ['timestamp', 'price']
        if not all(col in trades.columns for col in required_cols):
            raise ValueError(f"DataFrame precisa das colunas: {required_cols}")
        
        # Volume pode ser 'qty' ou 'volume'
        vol_col = 'qty' if 'qty' in trades.columns else 'volume'
        if vol_col not in trades.columns:
            trades = trades.copy()
            trades['volume'] = 1.0
            vol_col = 'volume'
        
        bars = []
        for _, row in trades.iterrows():
            bar = self.add_tick(
                timestamp=row['timestamp'],
                price=float(row['price']),
                volume=float(row[vol_col])
            )
            if bar:
                bars.append(bar.to_dict())
        
        # Adiciona última barra se existir
        if self._current_bar:
            bars.append(self._complete_bar().to_dict())
        
        return pd.DataFrame(bars)


class VolumeBarBuilder(BarBuilder):
    """
    Construtor de barras baseadas em volume.
    
    Cria nova barra quando volume acumulado atinge threshold.
    """
    
    def __init__(self, volume_threshold: float = 100.0):
        """
        Inicializa o builder.
        
        Args:
            volume_threshold: Volume necessário para completar barra
        """
        super().__init__()
        self.volume_threshold = volume_threshold
        self._accumulated_volume = 0.0
    
    def add_tick(self, timestamp: datetime, price: float, volume: float) -> Optional[Bar]:
        """Adiciona tick e retorna barra se volume atingido."""
        # Primeira tick
        if self._current_bar is None:
            self._start_new_bar(timestamp, price)
            self._accumulated_volume = 0.0
        
        # Atualiza barra
        self._update_bar(timestamp, price, volume)
        self._accumulated_volume += volume
        
        # Verifica threshold
        if self._accumulated_volume >= self.volume_threshold:
            self._accumulated_volume = 0.0
            return self._complete_bar()
        
        return None
    
    def build_from_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Constrói barras de volume a partir de trades."""
        self.reset()
        self._accumulated_volume = 0.0
        
        # Garante colunas necessárias
        if 'time' in trades.columns:
            trades = trades.rename(columns={'time': 'timestamp'})
        
        vol_col = 'qty' if 'qty' in trades.columns else 'volume'
        if vol_col not in trades.columns:
            raise ValueError("DataFrame precisa de coluna 'qty' ou 'volume'")
        
        bars = []
        for _, row in trades.iterrows():
            bar = self.add_tick(
                timestamp=row['timestamp'],
                price=float(row['price']),
                volume=float(row[vol_col])
            )
            if bar:
                bars.append(bar.to_dict())
        
        # Adiciona última barra se existir
        if self._current_bar:
            bars.append(self._complete_bar().to_dict())
        
        return pd.DataFrame(bars)


class DollarBarBuilder(BarBuilder):
    """
    Construtor de barras baseadas em valor em dólares.
    
    Cria nova barra quando valor transacionado atinge threshold.
    """
    
    def __init__(self, dollar_threshold: float = 10_000_000.0):
        """
        Inicializa o builder.
        
        Args:
            dollar_threshold: Valor em dólares para completar barra
        """
        super().__init__()
        self.dollar_threshold = dollar_threshold
        self._accumulated_dollars = 0.0
    
    def add_tick(self, timestamp: datetime, price: float, volume: float) -> Optional[Bar]:
        """Adiciona tick e retorna barra se valor atingido."""
        dollar_value = price * volume
        
        # Primeira tick
        if self._current_bar is None:
            self._start_new_bar(timestamp, price)
            self._accumulated_dollars = 0.0
        
        # Atualiza barra
        self._update_bar(timestamp, price, volume)
        self._accumulated_dollars += dollar_value
        
        # Verifica threshold
        if self._accumulated_dollars >= self.dollar_threshold:
            self._accumulated_dollars = 0.0
            return self._complete_bar()
        
        return None
    
    def build_from_trades(self, trades: pd.DataFrame) -> pd.DataFrame:
        """Constrói barras de dólar a partir de trades."""
        self.reset()
        self._accumulated_dollars = 0.0
        
        # Garante colunas necessárias
        if 'time' in trades.columns:
            trades = trades.rename(columns={'time': 'timestamp'})
        
        vol_col = 'qty' if 'qty' in trades.columns else 'volume'
        if vol_col not in trades.columns:
            raise ValueError("DataFrame precisa de coluna 'qty' ou 'volume'")
        
        bars = []
        for _, row in trades.iterrows():
            bar = self.add_tick(
                timestamp=row['timestamp'],
                price=float(row['price']),
                volume=float(row[vol_col])
            )
            if bar:
                bars.append(bar.to_dict())
        
        # Adiciona última barra se existir
        if self._current_bar:
            bars.append(self._complete_bar().to_dict())
        
        return pd.DataFrame(bars)


def create_bar_builder(
    bar_type: str = "time",
    config: Optional[Dict] = None
) -> BarBuilder:
    """
    Factory function para criar bar builder.
    
    Args:
        bar_type: Tipo de barra ("time", "volume", "dollar")
        config: Configuração opcional com parâmetros
        
    Returns:
        Instância do BarBuilder apropriado
        
    Raises:
        ValueError: Se tipo de barra inválido
    """
    config = config or {}
    
    if bar_type == "time":
        interval = config.get('interval_seconds', 60)
        return TimeBarBuilder(interval_seconds=interval)
    
    elif bar_type == "volume":
        threshold = config.get('volume_threshold', 
                              config.get('volume_bar_threshold', 100))
        return VolumeBarBuilder(volume_threshold=threshold)
    
    elif bar_type == "dollar":
        threshold = config.get('dollar_threshold',
                              config.get('dollar_bar_threshold', 10_000_000))
        return DollarBarBuilder(dollar_threshold=threshold)
    
    else:
        raise ValueError(
            f"Tipo de barra inválido: {bar_type}. "
            f"Use 'time', 'volume' ou 'dollar'."
        )
