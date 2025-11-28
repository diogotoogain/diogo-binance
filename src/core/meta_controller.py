"""
SNAME-MR Meta Controller
Aggregates signals from multiple strategies and decides actions.
Requires multiple strategies to agree before taking action.
"""
import logging
import time
from typing import Optional

logger = logging.getLogger("MetaController")


class MetaController:
    def __init__(self,
                 min_strategies_agree: int = 2,
                 signal_timeout_seconds: float = 5.0,
                 debounce_seconds: float = 0.1):
        self.min_strategies_agree = min_strategies_agree
        self.signal_timeout_seconds = signal_timeout_seconds
        self.debounce_seconds = debounce_seconds
        
        self.pending_signals = {}  # strategy_name -> {action, timestamp}
        self.total_decisions = 0
        self.last_decision_time = 0.0
        
        logger.info(f"🧠 MetaController iniciado:")
        logger.info(f"   📊 Min Estratégias: {self.min_strategies_agree}")
        logger.info(f"   ⏱️ Timeout: {self.signal_timeout_seconds}s")
        logger.info(f"   🔄 Debounce: {self.debounce_seconds}s")

    def receive_signal(self, strategy_name: str, signal: dict) -> Optional[dict]:
        """
        Receives a signal from a strategy and returns an aggregated decision
        if enough strategies agree.
        """
        current_time = time.time()
        action = signal.get('action')
        
        if not action:
            return None
        
        # Check debounce
        if current_time - self.last_decision_time < self.debounce_seconds:
            return None
        
        # Clean expired signals
        self._clean_expired_signals(current_time)
        
        # Store new signal
        self.pending_signals[strategy_name] = {
            'action': action,
            'timestamp': current_time,
            'signal': signal
        }
        
        # Count agreeing signals
        agreeing_strategies = self._count_agreeing_strategies(action)
        
        if len(agreeing_strategies) >= self.min_strategies_agree:
            # Make decision
            self.total_decisions += 1
            self.last_decision_time = current_time
            
            # Calculate confidence
            confidence = 'HIGH' if len(agreeing_strategies) >= 3 else 'MEDIUM'
            
            result = {
                'action': action,
                'confidence': confidence,
                'strategies': agreeing_strategies,
                'count': len(agreeing_strategies)
            }
            
            logger.info(f"✅ DECISÃO: {action} | Confiança: {confidence} | Estratégias: {agreeing_strategies}")
            return result
        
        return None

    def _clean_expired_signals(self, current_time: float):
        """Remove signals that have timed out."""
        expired = []
        for name, data in self.pending_signals.items():
            if current_time - data['timestamp'] > self.signal_timeout_seconds:
                expired.append(name)
        
        for name in expired:
            del self.pending_signals[name]

    def _count_agreeing_strategies(self, action: str) -> list:
        """Returns list of strategy names agreeing with the action."""
        agreeing = []
        for name, data in self.pending_signals.items():
            if data['action'] == action:
                agreeing.append(name)
        return agreeing

    def get_status(self) -> dict:
        return {
            'total_decisions': self.total_decisions,
            'pending_signals': len(self.pending_signals),
            'min_strategies_agree': self.min_strategies_agree
        }
