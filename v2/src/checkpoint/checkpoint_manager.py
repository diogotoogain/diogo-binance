"""
Checkpoint Manager for MEGA Historical Simulator.

Manages save/load of simulation state for persistence and recovery.
"""

import gzip
import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Constants for time tracking
SECONDS_PER_MINUTE = 60


@dataclass
class SimulationCheckpoint:
    """Complete simulation state for checkpoint/recovery."""
    
    # Identification
    checkpoint_id: str
    created_at: datetime
    config_hash: str  # Hash to validate if config changed
    
    # Progress
    current_index: int
    total_candles: int
    progress_pct: float
    
    # Simulator State
    balance: float
    position: Optional[Any]  # TradeRecord if position is open
    trades: List[Any]  # List of TradeRecord
    equity_curve: List[float]
    timestamps: List[datetime]
    
    # Online Learning State
    online_model_state: Optional[bytes] = None
    n_samples_seen: int = 0
    model_accuracy: float = 0.0
    
    # Drift Detection
    drift_events: List[Dict] = field(default_factory=list)
    drift_detector_state: Optional[bytes] = None
    
    # Features State (rolling windows, etc.)
    features_state: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    elapsed_time_seconds: float = 0.0
    last_candle_timestamp: Optional[datetime] = None
    data_file_hashes: Dict[str, str] = field(default_factory=dict)


class CheckpointManager:
    """
    Manages checkpoint save/load operations.
    
    Features:
    - Auto-save at configurable intervals (time and candles)
    - Compression support (gzip)
    - Retention policy (keep last N checkpoints)
    - Manifest tracking
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "v2/checkpoints",
        save_interval_minutes: int = 5,
        save_interval_candles: int = 10000,
        keep_last_n: int = 5,
        compress: bool = True
    ):
        """
        Initialize CheckpointManager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            save_interval_minutes: Auto-save interval in minutes
            save_interval_candles: Auto-save interval in candles
            keep_last_n: Number of recent checkpoints to keep
            compress: Whether to compress checkpoints with gzip
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_interval_minutes = save_interval_minutes
        self.save_interval_candles = save_interval_candles
        self.keep_last_n = keep_last_n
        self.compress = compress
        
        # Tracking
        self._last_save_time = time.time()
        self._last_save_candle = 0
        self._start_time = time.time()
        
        # Manifest file
        self.manifest_path = self.checkpoint_dir / "checkpoint_manifest.json"
        self._load_manifest()
    
    def _load_manifest(self) -> None:
        """Load or create checkpoint manifest."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r') as f:
                    self.manifest = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.manifest = {"checkpoints": []}
        else:
            self.manifest = {"checkpoints": []}
    
    def _save_manifest(self) -> None:
        """Save checkpoint manifest."""
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2, default=str)
    
    def should_save(self, current_candle: int) -> bool:
        """
        Check if it's time to auto-save.
        
        Args:
            current_candle: Current candle index
            
        Returns:
            True if should save, False otherwise
        """
        # Time-based check
        elapsed_minutes = (time.time() - self._last_save_time) / SECONDS_PER_MINUTE
        if elapsed_minutes >= self.save_interval_minutes:
            return True
        
        # Candle-based check
        candles_since_save = current_candle - self._last_save_candle
        if candles_since_save >= self.save_interval_candles:
            return True
        
        return False
    
    def save(
        self,
        checkpoint: SimulationCheckpoint,
        is_final: bool = False
    ) -> str:
        """
        Save checkpoint to disk.
        
        Args:
            checkpoint: SimulationCheckpoint object
            is_final: If True, this is the final checkpoint
            
        Returns:
            Path to saved checkpoint file
        """
        # Generate filename
        timestamp_str = checkpoint.created_at.strftime("%Y-%m-%d_%H-%M-%S")
        suffix = "final" if is_final else timestamp_str
        filename = f"simulation_state_{suffix}.pkl"
        if self.compress:
            filename += ".gz"
        
        filepath = self.checkpoint_dir / filename
        
        # Serialize and save
        data = pickle.dumps(checkpoint)
        
        if self.compress:
            with gzip.open(filepath, 'wb') as f:
                f.write(data)
        else:
            with open(filepath, 'wb') as f:
                f.write(data)
        
        # Update latest symlink/copy
        latest_path = self.checkpoint_dir / ("simulation_state_latest.pkl.gz" if self.compress else "simulation_state_latest.pkl")
        if latest_path.exists():
            latest_path.unlink()
        
        # Copy to latest instead of symlink for better compatibility
        if self.compress:
            with gzip.open(latest_path, 'wb') as f:
                f.write(data)
        else:
            with open(latest_path, 'wb') as f:
                f.write(data)
        
        # Update manifest
        checkpoint_info = {
            "id": checkpoint.checkpoint_id,
            "filepath": str(filepath),
            "created_at": checkpoint.created_at.isoformat(),
            "progress_pct": checkpoint.progress_pct,
            "candle_index": checkpoint.current_index,
            "total_candles": checkpoint.total_candles,
            "balance": checkpoint.balance,
            "trades": len(checkpoint.trades),
            "is_final": is_final
        }
        self.manifest["checkpoints"].append(checkpoint_info)
        self._save_manifest()
        
        # Update tracking
        self._last_save_time = time.time()
        self._last_save_candle = checkpoint.current_index
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        logger.info(f"💾 Checkpoint saved: {filepath}")
        
        return str(filepath)
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the last N."""
        checkpoints = self.manifest.get("checkpoints", [])
        
        # Filter out final checkpoints from cleanup
        regular = [c for c in checkpoints if not c.get("is_final", False)]
        finals = [c for c in checkpoints if c.get("is_final", False)]
        
        # Keep only last N regular checkpoints
        if len(regular) > self.keep_last_n:
            to_remove = regular[:-self.keep_last_n]
            for cp in to_remove:
                filepath = Path(cp["filepath"])
                if filepath.exists():
                    filepath.unlink()
                    logger.debug(f"Removed old checkpoint: {filepath}")
            
            regular = regular[-self.keep_last_n:]
        
        self.manifest["checkpoints"] = finals + regular
        self._save_manifest()
    
    def load(self, filepath: Optional[str] = None) -> Optional[SimulationCheckpoint]:
        """
        Load checkpoint from disk.
        
        Args:
            filepath: Specific checkpoint file path.
                     If None, loads latest checkpoint.
                     
        Returns:
            SimulationCheckpoint or None if not found
        """
        if filepath is None:
            # Try to load latest
            latest_path = self.checkpoint_dir / ("simulation_state_latest.pkl.gz" if self.compress else "simulation_state_latest.pkl")
            if not latest_path.exists():
                logger.warning("No latest checkpoint found")
                return None
            filepath = str(latest_path)
        
        filepath = Path(filepath)
        if not filepath.exists():
            logger.warning(f"Checkpoint file not found: {filepath}")
            return None
        
        try:
            if str(filepath).endswith('.gz'):
                with gzip.open(filepath, 'rb') as f:
                    data = f.read()
            else:
                with open(filepath, 'rb') as f:
                    data = f.read()
            
            checkpoint = pickle.loads(data)
            logger.info(f"📂 Loaded checkpoint: {filepath}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def get_latest_checkpoint_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the latest checkpoint.
        
        Returns:
            Dictionary with checkpoint info or None
        """
        if not self.manifest.get("checkpoints"):
            return None
        
        # Get latest non-final checkpoint
        checkpoints = [c for c in self.manifest["checkpoints"] if not c.get("is_final", False)]
        if not checkpoints:
            return None
        
        return checkpoints[-1]
    
    def has_checkpoint(self) -> bool:
        """Check if any checkpoint exists."""
        latest_path = self.checkpoint_dir / ("simulation_state_latest.pkl.gz" if self.compress else "simulation_state_latest.pkl")
        return latest_path.exists()
    
    def clear_checkpoints(self) -> None:
        """Remove all checkpoints."""
        for f in self.checkpoint_dir.glob("simulation_state_*.pkl*"):
            f.unlink()
        self.manifest = {"checkpoints": []}
        self._save_manifest()
        logger.info("🗑️ All checkpoints cleared")
    
    def get_elapsed_time(self) -> float:
        """Get total elapsed simulation time."""
        return time.time() - self._start_time
    
    @staticmethod
    def compute_config_hash(config: Dict[str, Any]) -> str:
        """
        Compute hash of configuration for validation.
        
        Uses SHA-256 for reliable configuration change detection.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            SHA-256 hash string
        """
        import hashlib
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def create_checkpoint(
        self,
        simulator: Any,
        current_index: int,
        total_candles: int,
        config: Dict[str, Any],
        online_learner: Optional[Any] = None,
        drift_detector: Optional[Any] = None,
        last_timestamp: Optional[datetime] = None
    ) -> SimulationCheckpoint:
        """
        Create checkpoint from current simulator state.
        
        Args:
            simulator: HonestSimulator instance
            current_index: Current candle index
            total_candles: Total candles to process
            config: Configuration dictionary
            online_learner: Optional online learner instance
            drift_detector: Optional drift detector instance
            last_timestamp: Timestamp of last processed candle
            
        Returns:
            SimulationCheckpoint object
        """
        import uuid
        
        # Serialize online learner state
        ol_state = None
        n_samples = 0
        accuracy = 0.0
        if online_learner is not None:
            try:
                ol_state = pickle.dumps(online_learner)
                stats = online_learner.get_stats() if hasattr(online_learner, 'get_stats') else {}
                n_samples = stats.get('n_samples_seen', 0)
                accuracy = stats.get('accuracy', 0.0)
            except Exception as e:
                logger.warning(f"Failed to serialize online learner: {e}")
        
        # Serialize drift detector state
        dd_state = None
        if drift_detector is not None:
            try:
                dd_state = pickle.dumps(drift_detector)
            except Exception as e:
                logger.warning(f"Failed to serialize drift detector: {e}")
        
        return SimulationCheckpoint(
            checkpoint_id=str(uuid.uuid4())[:8],
            created_at=datetime.now(),
            config_hash=self.compute_config_hash(config),
            current_index=current_index,
            total_candles=total_candles,
            progress_pct=(current_index / total_candles * 100) if total_candles > 0 else 0.0,
            balance=simulator.balance,
            position=simulator.position,
            trades=list(simulator.trades),
            equity_curve=list(simulator.equity_curve),
            timestamps=list(simulator.timestamps) if hasattr(simulator, 'timestamps') else [],
            online_model_state=ol_state,
            n_samples_seen=n_samples,
            model_accuracy=accuracy,
            drift_events=list(simulator.drift_events) if hasattr(simulator, 'drift_events') else [],
            drift_detector_state=dd_state,
            elapsed_time_seconds=self.get_elapsed_time(),
            last_candle_timestamp=last_timestamp
        )
    
    def restore_simulator(
        self,
        checkpoint: SimulationCheckpoint,
        simulator: Any,
    ) -> tuple:
        """
        Restore simulator state from checkpoint.
        
        Args:
            checkpoint: SimulationCheckpoint to restore from
            simulator: HonestSimulator instance to restore to
            
        Returns:
            Tuple of (online_learner, drift_detector) if they were saved
        """
        # Restore simulator state
        simulator.balance = checkpoint.balance
        simulator.position = checkpoint.position
        simulator.trades = list(checkpoint.trades)
        simulator.equity_curve = list(checkpoint.equity_curve)
        if hasattr(simulator, 'timestamps'):
            simulator.timestamps = list(checkpoint.timestamps)
        if hasattr(simulator, 'drift_events'):
            simulator.drift_events = list(checkpoint.drift_events)
        simulator.n_samples_processed = checkpoint.current_index
        
        # Restore online learner
        online_learner = None
        if checkpoint.online_model_state is not None:
            try:
                online_learner = pickle.loads(checkpoint.online_model_state)
            except Exception as e:
                logger.warning(f"Failed to restore online learner: {e}")
        
        # Restore drift detector
        drift_detector = None
        if checkpoint.drift_detector_state is not None:
            try:
                drift_detector = pickle.loads(checkpoint.drift_detector_state)
            except Exception as e:
                logger.warning(f"Failed to restore drift detector: {e}")
        
        logger.info(f"✅ Restored simulator state from checkpoint")
        
        return online_learner, drift_detector
