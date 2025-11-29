#!/usr/bin/env python3
"""
Weekly RL Training Script.

This script trains the RL agent on historical data and saves the model
along with performance metrics. Focused on MAXIMIZING PROFIT using
Sortino-based reward function.

Usage:
    python v2/scripts/train_weekly.py [--timesteps N] [--data-dir PATH]
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

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


def load_parquet_data(data_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load and combine parquet files from data directory.

    Args:
        data_dir: Path to directory containing parquet files

    Returns:
        Combined DataFrame or None if no data found
    """
    parquet_files = list(data_dir.glob("**/*.parquet"))

    if not parquet_files:
        logger.warning(f"No parquet files found in {data_dir}")
        return None

    logger.info(f"Found {len(parquet_files)} parquet files")

    dfs = []
    for pf in sorted(parquet_files):
        try:
            df = pd.read_parquet(pf)
            dfs.append(df)
            logger.debug(f"Loaded {pf.name}: {len(df)} rows")
        except Exception as e:
            logger.warning(f"Failed to load {pf}: {e}")

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=True)

    # Sort by timestamp if available
    if "timestamp" in combined.columns:
        combined = combined.sort_values("timestamp").reset_index(drop=True)
    elif "time" in combined.columns:
        combined = combined.sort_values("time").reset_index(drop=True)

    # Remove duplicates
    combined = combined.drop_duplicates()

    logger.info(f"Combined data: {len(combined)} rows")
    return combined


def prepare_training_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for RL training by adding required features.

    Args:
        data: Raw OHLCV data

    Returns:
        Data with features added
    """
    df = data.copy()

    # Standardize column names
    column_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df = df.rename(columns=column_map)

    # Ensure required columns exist
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            # Try to find alternative names
            alt_col = col.capitalize()
            if alt_col in df.columns:
                df[col] = df[alt_col]
            elif col == "volume" and "Volume" not in df.columns:
                # Generate synthetic volume if missing
                df[col] = 1000

    # Calculate technical features if not present
    if "atr" not in df.columns:
        # Calculate ATR (14-period)
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

    if "rsi" not in df.columns:
        # Calculate RSI (14-period)
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi"] = df["rsi"].fillna(50)

    if "adx" not in df.columns:
        # Simplified ADX calculation
        df["adx"] = 25  # Default value

    if "ofi" not in df.columns:
        # Order Flow Imbalance proxy
        df["ofi"] = np.random.randn(len(df)) * 0.1

    if "tfi" not in df.columns:
        # Trade Flow Imbalance proxy
        df["tfi"] = np.random.randn(len(df)) * 0.1

    if "regime" not in df.columns:
        # Market regime (0=crash, 1=normal, 2=trending)
        df["regime"] = 1

    # Fill NaN values
    df = df.ffill().bfill()

    return df


def get_profit_focused_config() -> Dict[str, Any]:
    """
    Get configuration optimized for maximizing profit.

    Returns:
        Configuration dictionary
    """
    return {
        "rl": {
            "enabled": True,
            "environment": {
                "observation_features": [
                    "ofi",
                    "tfi",
                    "rsi",
                    "adx",
                    "regime",
                    "position",
                    "pnl",
                    "drawdown",
                ],
                "action_space": {
                    "direction": [-1, 0, 1],
                    "position_size": [0.0, 0.25, 0.5, 0.75, 1.0],
                    "sl_atr_mult": [0.5, 1.0, 1.5, 2.0, 3.0],
                    "tp_atr_mult": [1.0, 1.5, 2.0, 3.0, 5.0],
                },
                "reward_function": "sortino",  # Maximizes profit, penalizes losses
                "episode_length": 1440,  # 24 hours of 1-minute bars
                "initial_balance": 10000,
            },
            "agents": {
                "ppo": {
                    "enabled": True,
                    "learning_rate": 3e-4,
                    "n_steps": 2048,
                    "batch_size": 64,
                    "n_epochs": 10,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_range": 0.2,
                    "ent_coef": 0.01,  # Encourage exploration
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                },
                "sac": {
                    "enabled": False,  # Focus on PPO for weekly training
                },
            },
            "training": {
                "total_timesteps": 500000,
                "eval_freq": 10000,
                "n_eval_episodes": 5,
                "save_freq": 50000,
            },
        },
        "risk": {
            "kill_switch": {"enabled": True},
            "max_position_size_pct": 50.0,  # Up to 50% of capital per trade
            "max_leverage": 10,
            "max_drawdown_pct": 20.0,
        },
        "backtest": {
            "initial_capital": 10000,
            "fees": {
                "taker": 0.0004,  # 0.04% Binance fee
                "maker": 0.0002,
            },
        },
    }


def train_rl_agent(
    data: pd.DataFrame,
    config: Dict[str, Any],
    total_timesteps: int,
    save_dir: Path,
) -> Dict[str, Any]:
    """
    Train the RL agent on the provided data.

    Args:
        data: Training data
        config: Configuration dictionary
        total_timesteps: Number of training timesteps
        save_dir: Directory to save model

    Returns:
        Training results
    """
    from v2.src.rl.environment import FlatActionWrapper, TradingEnvironment
    from v2.src.rl.trainer import RLTrainer

    # Split data for training and evaluation
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx].reset_index(drop=True)
    eval_data = data.iloc[split_idx:].reset_index(drop=True)

    logger.info(f"Training data: {len(train_data)} rows")
    logger.info(f"Evaluation data: {len(eval_data)} rows")

    # Create environments
    train_env = TradingEnvironment(config, train_data)
    eval_env = TradingEnvironment(config, eval_data)

    # Create trainer
    trainer = RLTrainer(config, train_env, eval_env)

    # Train PPO agent
    logger.info(f"Starting PPO training for {total_timesteps} timesteps...")
    results = trainer.train(
        agent_type="ppo",
        total_timesteps=total_timesteps,
        save_path=save_dir / "checkpoints",
    )

    # Save final model
    model_path = save_dir / "ppo_trading_final.zip"
    trainer.save("ppo", model_path)
    logger.info(f"Model saved to {model_path}")

    return results


def run_backtest(
    data: pd.DataFrame,
    model_path: Path,
    config: Dict[str, Any],
    days: int = 30,
) -> Dict[str, Any]:
    """
    Run backtest on recent data.

    Args:
        data: Full dataset
        model_path: Path to trained model
        config: Configuration dictionary
        days: Number of days to backtest

    Returns:
        Backtest metrics
    """
    from v2.src.rl.agents.ppo_agent import PPOAgent
    from v2.src.rl.environment import FlatActionWrapper, TradingEnvironment

    # Get last N days of data (approximate)
    # Assuming roughly 1440 bars per day for 1-minute data
    bars_per_day = 1440
    n_bars = min(days * bars_per_day, len(data))
    backtest_data = data.iloc[-n_bars:].reset_index(drop=True)

    logger.info(f"Backtesting on last {n_bars} bars (~{days} days)")

    # Create environment
    env = TradingEnvironment(config, backtest_data)
    wrapped_env = FlatActionWrapper(env)

    # Load model
    agent = PPOAgent(config, wrapped_env)
    agent.load(model_path)

    # Run backtest episodes
    n_episodes = 5
    all_metrics = []

    for ep in range(n_episodes):
        obs, _ = wrapped_env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            done = terminated or truncated
            total_reward += reward

        # Get episode stats
        stats = env.get_episode_stats()
        stats["total_reward"] = total_reward
        all_metrics.append(stats)

    # Aggregate metrics
    metrics_df = pd.DataFrame(all_metrics)

    results = {
        "total_return": float(metrics_df["total_pnl"].mean()),
        "total_return_pct": float(
            metrics_df["total_pnl"].mean() / config["rl"]["environment"]["initial_balance"] * 100
        ),
        "sharpe_ratio": float(metrics_df["sharpe"].mean()),
        "max_drawdown": float(metrics_df["max_drawdown"].mean()),
        "win_rate": float(metrics_df["win_rate"].mean()),
        "avg_trades": float(metrics_df["n_trades"].mean()),
        "n_episodes": n_episodes,
    }

    return results


def save_results(
    results: Dict[str, Any],
    metrics: Dict[str, Any],
    save_dir: Path,
) -> None:
    """
    Save training results and metrics.

    Args:
        results: Training results
        metrics: Backtest metrics
        save_dir: Directory to save results
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Combine results
    combined = {
        "timestamp": timestamp,
        "training": results,
        "backtest": metrics,
    }

    # Save as JSON
    results_file = save_dir / f"training_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(combined, f, indent=2, default=str)

    logger.info(f"Results saved to {results_file}")

    # Also save latest results
    latest_file = save_dir / "latest_results.json"
    with open(latest_file, "w") as f:
        json.dump(combined, f, indent=2, default=str)


def print_metrics_summary(metrics: Dict[str, Any]) -> None:
    """Print a summary of the backtest metrics."""
    print("\n" + "=" * 60)
    print("📊 BACKTEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Total Return:     {metrics.get('total_return_pct', 0):.2f}%")
    print(f"  Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  Max Drawdown:     {metrics.get('max_drawdown', 0):.2f}%")
    print(f"  Win Rate:         {metrics.get('win_rate', 0):.2f}%")
    print(f"  Avg Trades:       {metrics.get('avg_trades', 0):.1f}")
    print("=" * 60 + "\n")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Weekly RL Training Script")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500000,
        help="Total training timesteps (default: 500000)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="v2/data/raw",
        help="Directory containing parquet data files",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="v2/models/rl",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="v2/results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--backtest-days",
        type=int,
        default=30,
        help="Number of days for backtesting (default: 30)",
    )

    args = parser.parse_args()

    # Convert to paths
    data_dir = project_root / args.data_dir
    model_dir = project_root / args.model_dir
    results_dir = project_root / args.results_dir

    # Create directories
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("🚀 Starting Weekly RL Training")
    logger.info("=" * 60)
    logger.info(f"  Data directory:    {data_dir}")
    logger.info(f"  Model directory:   {model_dir}")
    logger.info(f"  Results directory: {results_dir}")
    logger.info(f"  Timesteps:         {args.timesteps}")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading training data...")
    data = load_parquet_data(data_dir)

    if data is None or len(data) == 0:
        logger.error("No data available for training!")
        # Create synthetic data for testing
        logger.info("Creating synthetic data for demonstration...")
        n = 50000
        np.random.seed(42)
        returns = np.random.randn(n) * 0.001
        prices = 50000 * np.exp(np.cumsum(returns))
        data = pd.DataFrame(
            {
                "open": prices * (1 + np.random.randn(n) * 0.0005),
                "high": prices * (1 + np.abs(np.random.randn(n) * 0.002)),
                "low": prices * (1 - np.abs(np.random.randn(n) * 0.002)),
                "close": prices,
                "volume": np.random.randint(100, 1000, n),
            }
        )

    # Prepare data
    logger.info("Preparing training data...")
    data = prepare_training_data(data)
    logger.info(f"Prepared data: {len(data)} rows, {len(data.columns)} columns")

    # Get profit-focused config
    config = get_profit_focused_config()

    # Train agent
    logger.info("Training RL agent...")
    training_results = train_rl_agent(
        data=data,
        config=config,
        total_timesteps=args.timesteps,
        save_dir=model_dir,
    )

    # Run backtest
    logger.info("Running backtest...")
    model_path = model_dir / "ppo_trading_final.zip"
    backtest_metrics = run_backtest(
        data=data,
        model_path=model_path,
        config=config,
        days=args.backtest_days,
    )

    # Save results
    save_results(training_results, backtest_metrics, results_dir)

    # Print summary
    print_metrics_summary(backtest_metrics)

    logger.info("✅ Weekly training complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
