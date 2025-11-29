#!/usr/bin/env python3
"""
Online Learning Update Script.

This script runs alongside data collection to continuously update
the online learning model. Runs every 6 hours after new data is collected.

Usage:
    python v2/scripts/online_learning_update.py [--data-dir PATH] [--model-dir PATH]
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_recent_data(data_dir: Path, hours: int = 6) -> Optional[pd.DataFrame]:
    """
    Load most recent parquet data.
    
    Args:
        data_dir: Directory containing parquet files
        hours: Look for data from last N hours
        
    Returns:
        DataFrame with recent data or None
    """
    parquet_files = sorted(data_dir.glob("**/*.parquet"))
    
    if not parquet_files:
        logger.warning(f"No parquet files found in {data_dir}")
        return None
    
    # Load most recent files (by modification time)
    recent_files = []
    cutoff_time = datetime.utcnow().timestamp() - (hours * 3600)
    
    for pf in parquet_files:
        if pf.stat().st_mtime > cutoff_time:
            recent_files.append(pf)
    
    # If no recent files, use the most recent one anyway
    if not recent_files:
        recent_files = [parquet_files[-1]]
        logger.info(f"No files modified in last {hours}h, using most recent: {recent_files[0].name}")
    
    logger.info(f"Loading {len(recent_files)} parquet files")
    
    dfs = []
    for pf in recent_files:
        try:
            df = pd.read_parquet(pf)
            dfs.append(df)
            logger.debug(f"Loaded {pf.name}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"Failed to load {pf}: {e}")
    
    if not dfs:
        return None
    
    combined = pd.concat(dfs, ignore_index=True)
    
    # Sort by time
    time_col = 'open_time' if 'open_time' in combined.columns else 'timestamp'
    if time_col in combined.columns:
        combined = combined.sort_values(time_col).reset_index(drop=True)
    
    combined = combined.drop_duplicates()
    
    logger.info(f"Loaded {len(combined)} total rows")
    return combined


def prepare_features(data: pd.DataFrame) -> Tuple[List[Dict[str, float]], List[int]]:
    """
    Prepare features for online learning.
    
    Features:
    - price_change: Percentage price change
    - volume_change: Volume change ratio
    - high_low_range: High-low range as % of price
    - trades_intensity: Number of trades normalized
    
    Labels:
    - 1: Price went up (positive return)
    - 0: Price went down or neutral
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        Tuple of (features_list, labels)
    """
    # Standardize column names
    df = data.copy()
    
    col_map = {
        'Open': 'open', 'High': 'high', 'Low': 'low',
        'Close': 'close', 'Volume': 'volume'
    }
    df.rename(columns=col_map, inplace=True)
    
    # Calculate features
    features_list = []
    labels = []
    
    # Need at least 2 rows for features
    if len(df) < 2:
        logger.warning("Not enough data for feature calculation")
        return [], []
    
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        
        # Skip if essential columns missing
        if 'close' not in curr or 'close' not in prev:
            continue
            
        # Price change
        if prev['close'] != 0:
            price_change = (curr['close'] - prev['close']) / prev['close']
        else:
            price_change = 0
        
        # Volume change
        volume_change = 0
        if 'volume' in curr and 'volume' in prev:
            if prev['volume'] != 0:
                volume_change = (curr['volume'] - prev['volume']) / prev['volume']
        
        # High-low range
        high_low_range = 0
        if 'high' in curr and 'low' in curr:
            if curr['close'] != 0:
                high_low_range = (curr['high'] - curr['low']) / curr['close']
        
        # Trades intensity (normalized)
        trades_intensity = 0
        if 'trades_count' in curr:
            trades_intensity = min(curr['trades_count'] / 100.0, 5.0)  # Cap at 5
        
        features = {
            'price_change': price_change,
            'volume_change': min(max(volume_change, -5), 5),  # Clip extreme values
            'high_low_range': high_low_range,
            'trades_intensity': trades_intensity
        }
        
        # Label: 1 if price went up in next period, 0 otherwise
        # For online learning, we use current return as label
        label = 1 if price_change > 0 else 0
        
        features_list.append(features)
        labels.append(label)
    
    logger.info(f"Prepared {len(features_list)} samples")
    logger.info(f"Label distribution: {sum(labels)} positive, {len(labels) - sum(labels)} negative")
    
    return features_list, labels


def run_online_learning_update(
    data_dir: Path,
    model_dir: Path,
    results_dir: Path
) -> Dict[str, Any]:
    """
    Run online learning update.
    
    Args:
        data_dir: Directory with parquet data
        model_dir: Directory for model storage
        results_dir: Directory for results/metrics
        
    Returns:
        Dictionary with update results
    """
    # Import here to handle missing dependencies gracefully
    try:
        from v2.src.online_learning.river_models import OnlineLearner
        from v2.src.online_learning.drift_detector import MarketDriftDetector
    except ImportError as e:
        logger.error(f"Failed to import online learning modules: {e}")
        logger.error("Install river with: pip install river")
        return {'status': 'error', 'error': str(e)}
    
    # Load recent data
    data = load_recent_data(data_dir)
    if data is None or len(data) == 0:
        logger.warning("No data to process")
        return {'status': 'no_data'}
    
    # Prepare features
    features_list, labels = prepare_features(data)
    if not features_list:
        logger.warning("No features to learn from")
        return {'status': 'no_features'}
    
    # Model file path
    model_path = model_dir / "online_model.pkl"
    
    # Load existing model or create new one
    if model_path.exists():
        logger.info(f"Loading existing model from {model_path}")
        try:
            learner = OnlineLearner.load(str(model_path))
            logger.info(f"Model loaded with {learner.n_samples_seen} previous samples")
        except Exception as e:
            logger.warning(f"Failed to load model, creating new: {e}")
            learner = OnlineLearner()
    else:
        logger.info("Creating new OnlineLearner model")
        learner = OnlineLearner()
    
    # Initialize drift detector
    drift_detector = MarketDriftDetector()
    
    # Calculate PnL-like metric for drift detection
    # Using simulated "trades" based on predictions
    correct_predictions = 0
    total_predictions = 0
    
    # Learn from new data
    logger.info(f"Learning from {len(features_list)} new samples...")
    
    for features, label in zip(features_list, labels):
        # Predict before learning
        pred = learner.predict_one(features)
        if pred is not None:
            total_predictions += 1
            if pred == label:
                correct_predictions += 1
        
        # Learn
        learner.learn_one(features, label)
        
        # Update drift detector with simulated PnL
        if total_predictions > 0:
            win_rate = correct_predictions / total_predictions
            pnl = features['price_change'] if pred == label else -abs(features['price_change'])
            drift_detector.update(pnl=pnl, win_rate=win_rate)
    
    # Get final metrics
    metrics = learner.get_metrics()
    drift_stats = drift_detector.get_stats()
    
    logger.info(f"Learning complete:")
    logger.info(f"  - Total samples: {learner.n_samples_seen}")
    logger.info(f"  - Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  - F1 Score: {metrics['f1']:.4f}")
    logger.info(f"  - Drifts detected: {drift_stats['total_drifts']}")
    logger.info(f"  - Alert level: {drift_stats['alert_level']}")
    
    # Save model
    logger.info(f"Saving model to {model_path}")
    learner.save(str(model_path))
    
    # Save metrics
    results = {
        'status': 'success',
        'timestamp': datetime.utcnow().isoformat(),
        'samples_processed': len(features_list),
        'total_samples': learner.n_samples_seen,
        'metrics': metrics,
        'drift_stats': drift_stats,
        'should_pause_trading': drift_detector.should_pause_trading()
    }
    
    # Save results
    results_file = results_dir / f"update_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {results_file}")
    
    # Save latest metrics (overwrite)
    latest_file = results_dir / "latest_metrics.json"
    with open(latest_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Online Learning Update Script")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="v2/data/raw/1s",
        help="Directory containing parquet data files"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="v2/models/online",
        help="Directory to store/load model"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="v2/results/online_learning",
        help="Directory for results and metrics"
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=6,
        help="Look for data from last N hours"
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)
    
    # Create directories if needed
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Online Learning Update Starting")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Results directory: {results_dir}")
    logger.info("=" * 60)
    
    # Run update
    results = run_online_learning_update(data_dir, model_dir, results_dir)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Update Summary:")
    logger.info(f"  Status: {results.get('status', 'unknown')}")
    
    if results.get('status') == 'success':
        logger.info(f"  Samples processed: {results.get('samples_processed', 0)}")
        logger.info(f"  Total model samples: {results.get('total_samples', 0)}")
        metrics = results.get('metrics', {})
        logger.info(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        logger.info(f"  F1 Score: {metrics.get('f1', 0):.4f}")
        
        if results.get('should_pause_trading'):
            logger.warning("⚠️  ALERT: Market drift detected! Consider pausing trading.")
    
    logger.info("=" * 60)
    
    # Exit with appropriate code
    if results.get('status') == 'error':
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
