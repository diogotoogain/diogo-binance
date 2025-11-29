"""
Online Learning Module for adaptive market learning.

This module provides:
- OnlineLearner: Incremental learning with Hoeffding Adaptive Tree
- MarketDriftDetector: Drift detection with ADWIN
"""

from .river_models import OnlineLearner
from .drift_detector import MarketDriftDetector

__all__ = ["OnlineLearner", "MarketDriftDetector"]
