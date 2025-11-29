"""Signal data class for trading signals."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Signal:
    """
    Trading signal data class.
    
    Attributes:
        direction: Signal direction (LONG, SHORT, NEUTRAL)
        confidence: Signal confidence [0, 1]
        timestamp: Signal timestamp
        strategy: Strategy that generated the signal
        entry_price: Suggested entry price
        stop_loss: Suggested stop loss price
        take_profit: Suggested take profit price
        metadata: Additional signal metadata
    """
    direction: str  # LONG, SHORT, NEUTRAL
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    strategy: str = ""
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate signal attributes."""
        if self.direction not in ('LONG', 'SHORT', 'NEUTRAL'):
            raise ValueError(f"Invalid direction: {self.direction}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1: {self.confidence}")
