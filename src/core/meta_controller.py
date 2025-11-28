"""
META-CONTROLADOR (ORÁCULO): O CÉREBRO DO SISTEMA
- Combina sinais de TODAS as estratégias
- Filtra sinais duplicados (debounce)
- Sistema de votação ponderada
- Só executa quando há consenso
"""
import logging
import os
import time
from typing import Dict, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("MetaController")


class MetaController:
    def __init__(self,
        min_strategies_agree: int = None,
        signal_timeout_seconds: float = None,
        debounce_seconds: float = None
    ):
        self.min_strategies_agree = min_strategies_agree or int(os.getenv("MIN_STRATEGIES_AGREE", 2))
        self.signal_timeout = signal_timeout_seconds or float(os.getenv("SIGNAL_TIMEOUT", 5.0))
        self.debounce = debounce_seconds or float(os.getenv("DEBOUNCE_SECONDS", 3.0))
        
        self.weights = {
            'FluxoBrabo': float(os.getenv("WEIGHT_FLUXO", 0.20)),
            'RollingVWAP': float(os.getenv("WEIGHT_VWAP", 0.15)),
            'PreditorVWAP': float(os.getenv("WEIGHT_PREDITOR", 0.15)),
            'VPINDetector': float(os.getenv("WEIGHT_VPIN", 0.25)),
            'OBI': float(os.getenv("WEIGHT_OBI", 0.15)),
            'LiquidationHunter': float(os.getenv("WEIGHT_LIQ", 0.10))
        }
        
        self.pending_signals: Dict[str, dict] = {}
        self.last_execution_time = 0
        self.last_action = None
        self.total_decisions = 0
        
        logger.info(f"🧠 MetaController | Min Agree: {self.min_strategies_agree} | Timeout: {self.signal_timeout}s | Debounce: {self.debounce}s")

    def receive_signal(self, strategy_name: str, signal: dict) -> Optional[dict]:
        now = time.time()
        
        if now - self.last_execution_time < self.debounce:
            return None
        
        if signal.get('action') == self.last_action and now - self.last_execution_time < 60:
            return None
        
        self.pending_signals[strategy_name] = {
            **signal,
            'timestamp': now,
            'weight': self.weights.get(strategy_name, 0.10)
        }
        
        self._cleanup_old_signals(now)
        return self._try_combine()

    def _cleanup_old_signals(self, now: float):
        expired = [k for k, v in self.pending_signals.items() 
                   if now - v['timestamp'] > self.signal_timeout]
        for k in expired:
            del self.pending_signals[k]

    def _try_combine(self) -> Optional[dict]:
        if len(self.pending_signals) < self.min_strategies_agree:
            return None
        
        buy_votes = []
        sell_votes = []
        
        for strat, sig in self.pending_signals.items():
            if sig['action'] == 'BUY':
                buy_votes.append((strat, sig['weight']))
            elif sig['action'] == 'SELL':
                sell_votes.append((strat, sig['weight']))
        
        buy_score = sum(w for _, w in buy_votes)
        sell_score = sum(w for _, w in sell_votes)
        
        if len(buy_votes) >= self.min_strategies_agree and buy_score > sell_score:
            action = 'BUY'
            strategies = [s for s, _ in buy_votes]
            score = buy_score
        elif len(sell_votes) >= self.min_strategies_agree and sell_score > buy_score:
            action = 'SELL'
            strategies = [s for s, _ in sell_votes]
            score = sell_score
        else:
            return None
        
        if len(strategies) >= 4:
            confidence = 'MAX'
        elif len(strategies) >= 3:
            confidence = 'HIGH'
        else:
            confidence = 'MEDIUM'
        
        self.pending_signals.clear()
        self.last_execution_time = time.time()
        self.last_action = action
        self.total_decisions += 1
        
        combined = {
            'action': action,
            'confidence': confidence,
            'score': score,
            'strategies': strategies,
            'reason': f"Consenso: {', '.join(strategies)} (score: {score:.2f})"
        }
        
        logger.info(f"🧠 DECISÃO #{self.total_decisions}: {action} | Conf: {confidence} | Estratégias: {strategies}")
        return combined
