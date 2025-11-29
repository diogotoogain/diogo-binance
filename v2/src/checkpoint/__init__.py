"""
Checkpoint module for MEGA Historical Simulator.

Provides persistence and recovery capabilities including:
- Auto-save checkpoints at configurable intervals
- Graceful shutdown handling (Ctrl+C)
- Resume from checkpoint functionality
"""

from .checkpoint_manager import CheckpointManager, SimulationCheckpoint
from .graceful_shutdown import GracefulShutdown

__all__ = ['CheckpointManager', 'SimulationCheckpoint', 'GracefulShutdown']
