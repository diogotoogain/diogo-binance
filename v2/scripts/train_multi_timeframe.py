#!/usr/bin/env python3
"""
Multi-Timeframe RL Training Script.

Trains separate RL models for different timeframes (1m, 5m, 15m)
to be used in the ensemble voting system.

Usage:
    python v2/scripts/train_multi_timeframe.py [--timeframe TF] [--timesteps N]
    
    # Train all timeframes
    python v2/scripts/train_multi_timeframe.py
    
    # Train specific timeframe
    python v2/scripts/train_multi_timeframe.py --timeframe 5m --timesteps 200000
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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

# Supported timeframes for training
TIMEFRAMES = ["1m", "5m", "15m"]

# Timeframe-specific configurations
TIMEFRAME_CONFIGS = {
    "1m": {
        "episode_length": 60,  # 1 hour of 1-minute bars
        "n_steps": 1024,
        "batch_size": 32,
        "description": "Scalping patterns",
    },
    "5m": {
        "episode_length": 288,  # 24 hours of 5-minute bars
        "n_steps": 2048,
        "batch_size": 64,
        "description": "Day trade patterns",
    },
    "15m": {
        "episode_length": 192,  # 2 days of 15-minute bars
        "n_steps": 2048,
        "batch_size": 64,
        "description": "Short swing patterns",
    },
}


def load_timeframe_data(data_dir: Path, timeframe: str) -> Optional[pd.DataFrame]:
    """
    Load data for a specific timeframe.

    Args:
        data_dir: Path to data directory
        timeframe: Timeframe to load (1m, 5m, 15m)

    Returns:
        DataFrame with OHLCV data or None if not found
    """
    # Try different file naming patterns
    patterns = [
        f"BTCUSDT_{timeframe}.parquet",
        f"btcusdt_{timeframe}.parquet",
        f"*_{timeframe}.parquet",
        f"*{timeframe}*.parquet",
    ]

    for pattern in patterns:
        files = list(data_dir.glob(pattern))
        if files:
            logger.info(f"Found {len(files)} files for {timeframe}: {pattern}")
            dfs = []
            for f in sorted(files):
                try:
                    df = pd.read_parquet(f)
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"Failed to load {f}: {e}")

            if dfs:
                combined = pd.concat(dfs, ignore_index=True)
                # Sort by timestamp if available
                for col in ["timestamp", "time", "datetime"]:
                    if col in combined.columns:
                        combined = combined.sort_values(col).reset_index(drop=True)
                        break
                combined = combined.drop_duplicates()
                return combined

    logger.warning(f"No data found for timeframe {timeframe}")
    return None


def generate_synthetic_data(timeframe: str, n_bars: int = 50000) -> pd.DataFrame:
    """
    Generate synthetic data for testing when real data is not available.

    Args:
        timeframe: Timeframe for data generation
        n_bars: Number of bars to generate

    Returns:
        DataFrame with synthetic OHLCV data
    """
    logger.info(f"Generating synthetic data for {timeframe} ({n_bars} bars)")

    np.random.seed(42 + hash(timeframe) % 1000)

    # Volatility per bar varies by timeframe:
    # - 1m: Lower volatility per bar (~0.05% per minute)
    # - 5m: Medium volatility (~0.1% per 5 minutes)
    # - 15m: Higher volatility (~0.2% per 15 minutes)
    # These values approximate typical BTC/USDT volatility patterns
    VOLATILITY_BY_TIMEFRAME = {
        "1m": 0.0005,   # ~0.05% per minute
        "5m": 0.001,    # ~0.1% per 5 minutes
        "15m": 0.002,   # ~0.2% per 15 minutes
    }
    volatility = VOLATILITY_BY_TIMEFRAME.get(timeframe, 0.001)

    returns = np.random.randn(n_bars) * volatility
    prices = 50000 * np.exp(np.cumsum(returns))

    data = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_bars) * volatility / 2),
            "high": prices * (1 + np.abs(np.random.randn(n_bars) * volatility * 2)),
            "low": prices * (1 - np.abs(np.random.randn(n_bars) * volatility * 2)),
            "close": prices,
            "volume": np.random.randint(100, 1000, n_bars),
        }
    )

    return data


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
            alt_col = col.capitalize()
            if alt_col in df.columns:
                df[col] = df[alt_col]
            elif col == "volume":
                df[col] = 1000

    # Calculate technical features
    if "atr" not in df.columns:
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift(1))
        low_close = abs(df["low"] - df["close"].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(14).mean()

    if "rsi" not in df.columns:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        df["rsi"] = df["rsi"].fillna(50)

    if "adx" not in df.columns:
        tr = pd.concat(
            [
                df["high"] - df["low"],
                abs(df["high"] - df["close"].shift(1)),
                abs(df["low"] - df["close"].shift(1)),
            ],
            axis=1,
        ).max(axis=1)

        plus_dm = df["high"].diff()
        minus_dm = df["low"].diff().abs()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        period = 14
        atr_smooth = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr_smooth)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr_smooth)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df["adx"] = dx.rolling(period).mean().fillna(25)

    if "ofi" not in df.columns:
        price_change = df["close"].diff()
        volume_dir = np.sign(price_change) * df["volume"]
        df["ofi"] = (
            volume_dir.rolling(20).sum() / (df["volume"].rolling(20).sum() + 1e-8)
        ).fillna(0)

    if "tfi" not in df.columns:
        hl_range = df["high"] - df["low"]
        close_position = (df["close"] - df["low"]) / (hl_range + 1e-8)
        df["tfi"] = (close_position * 2 - 1).rolling(20).mean().fillna(0)

    if "regime" not in df.columns:
        returns = df["close"].pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252)
        vol_pct = volatility.rank(pct=True)

        adx_threshold = 25
        trend_mask = df["adx"] > adx_threshold

        df["regime"] = 1  # Default normal
        df.loc[vol_pct > 0.8, "regime"] = 0  # High volatility
        df.loc[trend_mask & (vol_pct <= 0.8), "regime"] = 2  # Trending

    # Fill NaN values
    df = df.ffill().bfill()

    return df


def get_timeframe_config(timeframe: str, base_timesteps: int = 200000) -> Dict[str, Any]:
    """
    Get configuration optimized for a specific timeframe.

    Args:
        timeframe: Timeframe (1m, 5m, 15m)
        base_timesteps: Base number of training timesteps

    Returns:
        Configuration dictionary
    """
    tf_config = TIMEFRAME_CONFIGS.get(timeframe, TIMEFRAME_CONFIGS["5m"])

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
                "reward_function": "sortino",
                "episode_length": tf_config["episode_length"],
                "initial_balance": 10000,
            },
            "agents": {
                "ppo": {
                    "enabled": True,
                    "learning_rate": 3e-4,
                    "n_steps": tf_config["n_steps"],
                    "batch_size": tf_config["batch_size"],
                    "n_epochs": 10,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_range": 0.2,
                    "ent_coef": 0.01,
                    "vf_coef": 0.5,
                    "max_grad_norm": 0.5,
                },
                "sac": {"enabled": False},
            },
            "training": {
                "total_timesteps": base_timesteps,
                "eval_freq": 10000,
                "n_eval_episodes": 5,
                "save_freq": 50000,
            },
        },
        "risk": {
            "kill_switch": {"enabled": True},
            "max_position_size_pct": 10.0,
            "max_leverage": 5,
            "max_drawdown_pct": 20.0,
        },
        "backtest": {
            "initial_capital": 10000,
            "fees": {"taker": 0.0004, "maker": 0.0002},
        },
    }


def train_timeframe_model(
    data: pd.DataFrame,
    config: Dict[str, Any],
    timeframe: str,
    total_timesteps: int,
    save_dir: Path,
) -> Dict[str, Any]:
    """
    Train the RL agent for a specific timeframe.

    Args:
        data: Training data
        config: Configuration dictionary
        timeframe: Timeframe being trained
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

    logger.info(f"[{timeframe}] Training data: {len(train_data)} rows")
    logger.info(f"[{timeframe}] Evaluation data: {len(eval_data)} rows")

    # Create environments
    train_env = TradingEnvironment(config, train_data)
    eval_env = TradingEnvironment(config, eval_data)

    # Create trainer
    trainer = RLTrainer(config, train_env, eval_env)

    # Train PPO agent
    logger.info(f"[{timeframe}] Starting PPO training for {total_timesteps} timesteps...")
    results = trainer.train(
        agent_type="ppo",
        total_timesteps=total_timesteps,
        save_path=save_dir / f"checkpoints_{timeframe}",
    )

    # Save final model with timeframe-specific name
    model_path = save_dir / f"ppo_{timeframe}_best.zip"
    trainer.save("ppo", model_path)
    logger.info(f"[{timeframe}] Model saved to {model_path}")

    return results


def run_backtest(
    data: pd.DataFrame,
    model_path: Path,
    config: Dict[str, Any],
    timeframe: str,
) -> Dict[str, Any]:
    """
    Run backtest on the trained model.

    Args:
        data: Test data
        model_path: Path to trained model
        config: Configuration dictionary
        timeframe: Timeframe being tested

    Returns:
        Backtest metrics
    """
    from v2.src.rl.agents.ppo_agent import PPOAgent
    from v2.src.rl.environment import FlatActionWrapper, TradingEnvironment

    # Use last portion of data for backtest
    backtest_data = data.iloc[-int(len(data) * 0.2) :].reset_index(drop=True)

    logger.info(f"[{timeframe}] Backtesting on {len(backtest_data)} bars")

    # Create environment
    env = TradingEnvironment(config, backtest_data)
    wrapped_env = FlatActionWrapper(env)

    # Load model
    agent = PPOAgent(config, wrapped_env)
    agent.load(model_path, env=wrapped_env)

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

        stats = wrapped_env.unwrapped.get_episode_stats()
        stats["total_reward"] = total_reward
        all_metrics.append(stats)

    # Aggregate metrics
    metrics_df = pd.DataFrame(all_metrics)

    results = {
        "timeframe": timeframe,
        "total_return": float(metrics_df["total_pnl"].mean()),
        "total_return_pct": float(
            metrics_df["total_pnl"].mean()
            / config["rl"]["environment"]["initial_balance"]
            * 100
        ),
        "sharpe_ratio": float(metrics_df["sharpe"].mean()),
        "max_drawdown": float(metrics_df["max_drawdown"].mean()),
        "win_rate": float(metrics_df["win_rate"].mean()),
        "avg_trades": float(metrics_df["n_trades"].mean()),
        "n_episodes": n_episodes,
    }

    return results


def save_training_results(
    results: Dict[str, Any],
    save_dir: Path,
    timeframe: str,
) -> None:
    """Save training results to JSON file."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Save timeframe-specific results
    results_file = save_dir / f"training_results_{timeframe}_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Also save as latest
    latest_file = save_dir / f"latest_results_{timeframe}.json"
    with open(latest_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"[{timeframe}] Results saved to {results_file}")


def print_summary(all_results: Dict[str, Dict[str, Any]]) -> None:
    """Print summary of all training results."""
    print("\n" + "=" * 70)
    print("📊 MULTI-TIMEFRAME TRAINING SUMMARY")
    print("=" * 70)

    for tf, results in all_results.items():
        print(f"\n🕐 Timeframe: {tf} ({TIMEFRAME_CONFIGS.get(tf, {}).get('description', '')})")
        print("-" * 40)

        backtest = results.get("backtest", {})
        print(f"  Total Return:     {backtest.get('total_return_pct', 0):.2f}%")
        print(f"  Sharpe Ratio:     {backtest.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown:     {backtest.get('max_drawdown', 0):.2f}%")
        print(f"  Win Rate:         {backtest.get('win_rate', 0):.2f}%")
        print(f"  Avg Trades:       {backtest.get('avg_trades', 0):.1f}")

    print("\n" + "=" * 70)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Multi-Timeframe RL Training Script")
    parser.add_argument(
        "--timeframe",
        "-t",
        type=str,
        choices=TIMEFRAMES + ["all"],
        default="all",
        help="Timeframe to train (1m, 5m, 15m, or all)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200000,
        help="Total training timesteps per model (default: 200000)",
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

    args = parser.parse_args()

    # Convert to paths
    data_dir = project_root / args.data_dir
    model_dir = project_root / args.model_dir
    results_dir = project_root / args.results_dir

    # Create directories
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine which timeframes to train
    timeframes_to_train = TIMEFRAMES if args.timeframe == "all" else [args.timeframe]

    logger.info("=" * 70)
    logger.info("🚀 Starting Multi-Timeframe RL Training")
    logger.info("=" * 70)
    logger.info(f"  Timeframes:        {', '.join(timeframes_to_train)}")
    logger.info(f"  Data directory:    {data_dir}")
    logger.info(f"  Model directory:   {model_dir}")
    logger.info(f"  Results directory: {results_dir}")
    logger.info(f"  Timesteps:         {args.timesteps}")
    logger.info("=" * 70)

    all_results: Dict[str, Dict[str, Any]] = {}

    for tf in timeframes_to_train:
        logger.info(f"\n{'='*50}")
        logger.info(f"🕐 Training {tf} model ({TIMEFRAME_CONFIGS[tf]['description']})")
        logger.info("=" * 50)

        # Load data for this timeframe
        data = load_timeframe_data(data_dir, tf)

        if data is None or len(data) == 0:
            logger.warning(f"No data found for {tf}, generating synthetic data...")
            data = generate_synthetic_data(tf)

        # Prepare data
        data = prepare_training_data(data)
        logger.info(f"[{tf}] Prepared data: {len(data)} rows, {len(data.columns)} columns")

        # Get timeframe-specific config
        config = get_timeframe_config(tf, args.timesteps)

        # Train model
        training_results = train_timeframe_model(
            data=data,
            config=config,
            timeframe=tf,
            total_timesteps=args.timesteps,
            save_dir=model_dir,
        )

        # Run backtest
        model_path = model_dir / f"ppo_{tf}_best.zip"
        backtest_results = run_backtest(data, model_path, config, tf)

        # Combine results
        combined_results = {
            "timeframe": tf,
            "description": TIMEFRAME_CONFIGS[tf]["description"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "training": training_results,
            "backtest": backtest_results,
            "config": {
                "timesteps": args.timesteps,
                "episode_length": config["rl"]["environment"]["episode_length"],
            },
        }

        # Save results
        save_training_results(combined_results, results_dir, tf)
        all_results[tf] = combined_results

        logger.info(f"✅ Completed training for {tf}")

    # Print summary
    print_summary(all_results)

    # Save combined results
    combined_file = results_dir / "multi_timeframe_results.json"
    with open(combined_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Combined results saved to {combined_file}")

    logger.info("\n✅ Multi-timeframe training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
