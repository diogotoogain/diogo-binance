"""
Market Drift Detection using River's ADWIN.

This module detects when market conditions change significantly,
allowing the trading system to adapt or pause.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from river import drift
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False


class MarketDriftDetector:
    """
    Detects market regime changes using ADWIN drift detectors.
    
    Monitors multiple metrics to detect when market conditions change:
    - PnL drift: When profitability pattern changes
    - Win rate drift: When win/loss ratio shifts
    - Spread drift: When market liquidity changes
    
    Example:
        >>> detector = MarketDriftDetector()
        >>> # Update with new metrics
        >>> alert = detector.update(pnl=0.02, win_rate=0.6, spread=0.001)
        >>> if alert['drift_detected']:
        ...     print(f"Drift in: {alert['drifting_metrics']}")
        >>> # Check if should pause trading
        >>> if detector.should_pause_trading():
        ...     print("Market changed significantly - consider pausing")
    """
    
    # Default history management thresholds
    DEFAULT_MAX_HISTORY_SIZE = 1000
    DEFAULT_TRIM_HISTORY_SIZE = 500
    
    def __init__(
        self,
        pnl_delta: float = 0.002,
        win_rate_delta: float = 0.01,
        spread_delta: float = 0.005,
        pause_threshold: int = 2,
        cooldown_samples: int = 50,
        max_history_size: int = 1000,
        trim_history_size: int = 500
    ):
        """
        Initialize MarketDriftDetector.
        
        Args:
            pnl_delta: ADWIN sensitivity for PnL drift (lower = more sensitive)
            win_rate_delta: ADWIN sensitivity for win rate drift
            spread_delta: ADWIN sensitivity for spread drift
            pause_threshold: Number of metrics in drift to trigger pause
            cooldown_samples: Samples to wait after drift before resuming
            max_history_size: Maximum number of metric history entries to keep
            trim_history_size: Number of entries to keep when trimming history
        """
        if not RIVER_AVAILABLE:
            raise ImportError(
                "River library is required for MarketDriftDetector. "
                "Install with: pip install river"
            )
        
        self.pnl_delta = pnl_delta
        self.win_rate_delta = win_rate_delta
        self.spread_delta = spread_delta
        self.pause_threshold = pause_threshold
        self.cooldown_samples = cooldown_samples
        self.max_history_size = max_history_size
        self.trim_history_size = trim_history_size
        
        # Initialize ADWIN detectors
        self.pnl_detector = drift.ADWIN(delta=pnl_delta)
        self.win_rate_detector = drift.ADWIN(delta=win_rate_delta)
        self.spread_detector = drift.ADWIN(delta=spread_delta)
        
        # State tracking
        self.n_samples = 0
        self.n_pnl_drifts = 0
        self.n_win_rate_drifts = 0
        self.n_spread_drifts = 0
        self.last_drift_sample = -cooldown_samples  # Allow trading from start
        
        # History for analysis
        self.drift_history: List[Dict[str, Any]] = []
        self.metric_history: List[Dict[str, float]] = []
        
    def update(
        self,
        pnl: Optional[float] = None,
        win_rate: Optional[float] = None,
        spread: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Update detectors with new metrics and check for drift.
        
        Args:
            pnl: Current PnL value (can be trade return)
            win_rate: Current win rate (0-1)
            spread: Current spread value
            
        Returns:
            Dictionary with drift detection results
        """
        self.n_samples += 1
        
        drifting_metrics = []
        
        # Check PnL drift
        if pnl is not None:
            self.pnl_detector.update(pnl)
            if self.pnl_detector.drift_detected:
                drifting_metrics.append('pnl')
                self.n_pnl_drifts += 1
        
        # Check win rate drift
        if win_rate is not None:
            self.win_rate_detector.update(win_rate)
            if self.win_rate_detector.drift_detected:
                drifting_metrics.append('win_rate')
                self.n_win_rate_drifts += 1
        
        # Check spread drift
        if spread is not None:
            self.spread_detector.update(spread)
            if self.spread_detector.drift_detected:
                drifting_metrics.append('spread')
                self.n_spread_drifts += 1
        
        drift_detected = len(drifting_metrics) > 0
        
        if drift_detected:
            self.last_drift_sample = self.n_samples
            self.drift_history.append({
                'sample': self.n_samples,
                'metrics': drifting_metrics,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        
        # Track metrics
        self.metric_history.append({
            'sample': self.n_samples,
            'pnl': pnl,
            'win_rate': win_rate,
            'spread': spread
        })
        
        # Keep history manageable using configurable thresholds
        if len(self.metric_history) > self.max_history_size:
            self.metric_history = self.metric_history[-self.trim_history_size:]
        
        return {
            'drift_detected': drift_detected,
            'drifting_metrics': drifting_metrics,
            'n_samples': self.n_samples,
            'samples_since_drift': self.n_samples - self.last_drift_sample
        }
    
    def should_pause_trading(self) -> bool:
        """
        Check if trading should be paused due to market changes.
        
        Returns True if multiple metrics show drift and cooldown
        hasn't elapsed.
        
        Returns:
            True if trading should be paused
        """
        # Check if we're in cooldown period
        samples_since_drift = self.n_samples - self.last_drift_sample
        if samples_since_drift < self.cooldown_samples:
            # Check recent drift severity
            recent_drifts = [
                d for d in self.drift_history
                if d['sample'] > self.n_samples - self.cooldown_samples
            ]
            
            # Count unique metrics that drifted recently
            drifting_metrics = set()
            for d in recent_drifts:
                drifting_metrics.update(d['metrics'])
            
            if len(drifting_metrics) >= self.pause_threshold:
                return True
        
        return False
    
    def get_alert_level(self) -> str:
        """
        Get current alert level based on drift detection.
        
        Returns:
            'normal', 'warning', or 'critical'
        """
        if self.should_pause_trading():
            return 'critical'
        
        samples_since_drift = self.n_samples - self.last_drift_sample
        if samples_since_drift < self.cooldown_samples // 2:
            return 'warning'
        
        return 'normal'
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get drift detection statistics.
        
        Returns:
            Dictionary with all statistics
        """
        return {
            'n_samples': self.n_samples,
            'n_pnl_drifts': self.n_pnl_drifts,
            'n_win_rate_drifts': self.n_win_rate_drifts,
            'n_spread_drifts': self.n_spread_drifts,
            'total_drifts': self.n_pnl_drifts + self.n_win_rate_drifts + self.n_spread_drifts,
            'samples_since_drift': self.n_samples - self.last_drift_sample,
            'alert_level': self.get_alert_level(),
            'should_pause': self.should_pause_trading(),
            'recent_drifts': self.drift_history[-10:]  # Last 10 drifts
        }
    
    def get_detector_estimates(self) -> Dict[str, Optional[float]]:
        """
        Get current estimates from ADWIN detectors.
        
        Returns:
            Dictionary with mean estimates for each metric
        """
        estimates = {}
        
        # ADWIN tracks windowed mean
        estimates['pnl_mean'] = self.pnl_detector.estimation if self.n_samples > 0 else None
        estimates['win_rate_mean'] = self.win_rate_detector.estimation if self.n_samples > 0 else None
        estimates['spread_mean'] = self.spread_detector.estimation if self.n_samples > 0 else None
        
        return estimates
    
    def reset(self) -> None:
        """Reset all detectors and state."""
        self.pnl_detector = drift.ADWIN(delta=self.pnl_delta)
        self.win_rate_detector = drift.ADWIN(delta=self.win_rate_delta)
        self.spread_detector = drift.ADWIN(delta=self.spread_delta)
        
        self.n_samples = 0
        self.n_pnl_drifts = 0
        self.n_win_rate_drifts = 0
        self.n_spread_drifts = 0
        self.last_drift_sample = -self.cooldown_samples
        self.drift_history = []
        self.metric_history = []


class MultiMetricDriftDetector:
    """
    Generic drift detector for any number of metrics.
    
    Useful for monitoring custom metrics beyond PnL, win rate, and spread.
    """
    
    def __init__(
        self,
        metric_names: List[str],
        delta: float = 0.002,
        pause_threshold: int = 2,
        cooldown_samples: int = 50
    ):
        """
        Initialize MultiMetricDriftDetector.
        
        Args:
            metric_names: List of metric names to monitor
            delta: ADWIN sensitivity (same for all metrics)
            pause_threshold: Number of metrics in drift to trigger pause
            cooldown_samples: Samples to wait after drift before resuming
        """
        if not RIVER_AVAILABLE:
            raise ImportError(
                "River library is required for MultiMetricDriftDetector. "
                "Install with: pip install river"
            )
        
        self.metric_names = metric_names
        self.delta = delta
        self.pause_threshold = pause_threshold
        self.cooldown_samples = cooldown_samples
        
        # Initialize detectors
        self.detectors = {
            name: drift.ADWIN(delta=delta)
            for name in metric_names
        }
        
        # State
        self.n_samples = 0
        self.drift_counts = {name: 0 for name in metric_names}
        self.last_drift_sample = -cooldown_samples
        self.drift_history: List[Dict[str, Any]] = []
    
    def update(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Update detectors with new metrics.
        
        Args:
            metrics: Dictionary of metric_name -> value
            
        Returns:
            Dictionary with drift detection results
        """
        self.n_samples += 1
        drifting = []
        
        for name, value in metrics.items():
            if name in self.detectors:
                self.detectors[name].update(value)
                if self.detectors[name].drift_detected:
                    drifting.append(name)
                    self.drift_counts[name] += 1
        
        if drifting:
            self.last_drift_sample = self.n_samples
            self.drift_history.append({
                'sample': self.n_samples,
                'metrics': drifting,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
        
        return {
            'drift_detected': len(drifting) > 0,
            'drifting_metrics': drifting,
            'n_samples': self.n_samples
        }
    
    def should_pause(self) -> bool:
        """Check if action should be paused."""
        samples_since_drift = self.n_samples - self.last_drift_sample
        if samples_since_drift >= self.cooldown_samples:
            return False
        
        recent_drifts = [
            d for d in self.drift_history
            if d['sample'] > self.n_samples - self.cooldown_samples
        ]
        
        drifting_metrics = set()
        for d in recent_drifts:
            drifting_metrics.update(d['metrics'])
        
        return len(drifting_metrics) >= self.pause_threshold
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all metrics."""
        return {
            'n_samples': self.n_samples,
            'drift_counts': self.drift_counts,
            'total_drifts': sum(self.drift_counts.values()),
            'samples_since_drift': self.n_samples - self.last_drift_sample,
            'should_pause': self.should_pause()
        }
