"""
Configuration schema with optimizable parameter definitions.

This module defines all optimizable parameters for the trading system.
All parameters marked with OPTIMIZE in the config are defined here.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class ParamType(Enum):
    """Parameter type for optimization."""

    FLOAT = "float"
    INT = "int"
    CATEGORICAL = "categorical"
    BOOL = "bool"


@dataclass
class OptimizableParam:
    """Definition of an optimizable parameter."""

    name: str
    param_type: ParamType
    low: Optional[Union[float, int]] = None
    high: Optional[Union[float, int]] = None
    choices: Optional[List[Any]] = None
    default: Any = None
    description: str = ""
    log_scale: bool = False
    step: Optional[Union[float, int]] = None


# Define all optimizable parameters organized by section
OPTIMIZABLE_PARAMS: Dict[str, List[OptimizableParam]] = {
    # Features - Microstructure
    "features.microstructure.ofi": [
        OptimizableParam(
            name="window",
            param_type=ParamType.INT,
            low=5,
            high=100,
            default=20,
            description="OFI calculation window",
        ),
    ],
    "features.microstructure.tfi": [
        OptimizableParam(
            name="window",
            param_type=ParamType.INT,
            low=5,
            high=100,
            default=20,
            description="TFI calculation window",
        ),
    ],
    "features.microstructure.entropy": [
        OptimizableParam(
            name="window",
            param_type=ParamType.INT,
            low=20,
            high=100,
            default=50,
            description="Entropy calculation window",
        ),
    ],
    "features.microstructure.vpin": [
        OptimizableParam(
            name="n_buckets",
            param_type=ParamType.INT,
            low=20,
            high=100,
            default=50,
            description="Number of VPIN buckets",
        ),
    ],
    # Features - Technical
    "features.technical.rsi": [
        OptimizableParam(
            name="period",
            param_type=ParamType.INT,
            low=7,
            high=21,
            default=14,
            description="RSI period",
        ),
        OptimizableParam(
            name="overbought",
            param_type=ParamType.INT,
            low=65,
            high=80,
            default=70,
            description="RSI overbought threshold",
        ),
        OptimizableParam(
            name="oversold",
            param_type=ParamType.INT,
            low=20,
            high=35,
            default=30,
            description="RSI oversold threshold",
        ),
    ],
    "features.technical.macd": [
        OptimizableParam(
            name="fast",
            param_type=ParamType.INT,
            low=8,
            high=16,
            default=12,
            description="MACD fast period",
        ),
        OptimizableParam(
            name="slow",
            param_type=ParamType.INT,
            low=20,
            high=32,
            default=26,
            description="MACD slow period",
        ),
        OptimizableParam(
            name="signal",
            param_type=ParamType.INT,
            low=6,
            high=12,
            default=9,
            description="MACD signal period",
        ),
    ],
    "features.technical.adx": [
        OptimizableParam(
            name="period",
            param_type=ParamType.INT,
            low=10,
            high=20,
            default=14,
            description="ADX period",
        ),
    ],
    "features.technical.bollinger": [
        OptimizableParam(
            name="period",
            param_type=ParamType.INT,
            low=10,
            high=30,
            default=20,
            description="Bollinger period",
        ),
        OptimizableParam(
            name="std_dev",
            param_type=ParamType.FLOAT,
            low=1.5,
            high=3.0,
            default=2.0,
            description="Bollinger standard deviation",
        ),
    ],
    "features.technical.atr": [
        OptimizableParam(
            name="period",
            param_type=ParamType.INT,
            low=7,
            high=21,
            default=14,
            description="ATR period",
        ),
    ],
    # Features - Volume Analysis
    "features.volume_analysis.volume_spike": [
        OptimizableParam(
            name="lookback",
            param_type=ParamType.INT,
            low=10,
            high=30,
            default=20,
            description="Volume spike lookback",
        ),
        OptimizableParam(
            name="threshold_multiplier",
            param_type=ParamType.FLOAT,
            low=1.5,
            high=3.0,
            default=2.0,
            description="Volume spike threshold multiplier",
        ),
    ],
    "features.volume_analysis.liquidity_clusters": [
        OptimizableParam(
            name="levels",
            param_type=ParamType.INT,
            low=5,
            high=20,
            default=10,
            description="Number of liquidity levels",
        ),
        OptimizableParam(
            name="threshold_percentile",
            param_type=ParamType.INT,
            low=70,
            high=90,
            default=80,
            description="Liquidity threshold percentile",
        ),
    ],
    # Regime Detection
    "regime_detection.hmm": [
        OptimizableParam(
            name="n_regimes",
            param_type=ParamType.INT,
            low=2,
            high=4,
            default=3,
            description="Number of HMM regimes",
        ),
    ],
    "regime_detection.adx_regime": [
        OptimizableParam(
            name="trending_threshold",
            param_type=ParamType.INT,
            low=20,
            high=30,
            default=25,
            description="ADX trending threshold",
        ),
        OptimizableParam(
            name="ranging_threshold",
            param_type=ParamType.INT,
            low=15,
            high=25,
            default=20,
            description="ADX ranging threshold",
        ),
    ],
    # Labeling
    "labeling.triple_barrier": [
        OptimizableParam(
            name="tp_multiplier",
            param_type=ParamType.FLOAT,
            low=1.5,
            high=3.0,
            default=2.0,
            description="Take profit ATR multiplier",
        ),
        OptimizableParam(
            name="sl_multiplier",
            param_type=ParamType.FLOAT,
            low=0.5,
            high=2.0,
            default=1.0,
            description="Stop loss ATR multiplier",
        ),
        OptimizableParam(
            name="max_holding_bars",
            param_type=ParamType.INT,
            low=50,
            high=200,
            default=100,
            description="Maximum holding bars",
        ),
    ],
    "labeling.meta_labeling": [
        OptimizableParam(
            name="confidence_threshold",
            param_type=ParamType.FLOAT,
            low=0.5,
            high=0.7,
            default=0.6,
            description="Meta-labeling confidence threshold",
        ),
        OptimizableParam(
            name="model_type",
            param_type=ParamType.CATEGORICAL,
            choices=["xgboost", "lightgbm", "random_forest"],
            default="xgboost",
            description="Meta-labeling model type",
        ),
    ],
    # Strategies - HFT OFI Scalper
    "strategies.hft_ofi_scalper.params": [
        OptimizableParam(
            name="ofi_threshold",
            param_type=ParamType.FLOAT,
            low=0.1,
            high=0.5,
            default=0.3,
            description="OFI entry threshold",
        ),
        OptimizableParam(
            name="tfi_threshold",
            param_type=ParamType.FLOAT,
            low=0.1,
            high=0.5,
            default=0.3,
            description="TFI entry threshold",
        ),
        OptimizableParam(
            name="min_spread_bps",
            param_type=ParamType.FLOAT,
            low=0.5,
            high=2.0,
            default=1.0,
            description="Minimum spread in basis points",
        ),
        OptimizableParam(
            name="holding_seconds",
            param_type=ParamType.INT,
            low=10,
            high=60,
            default=30,
            description="Position holding time in seconds",
        ),
    ],
    "strategies.hft_ofi_scalper.filters.adx_filter": [
        OptimizableParam(
            name="max_adx",
            param_type=ParamType.INT,
            low=25,
            high=35,
            default=30,
            description="Maximum ADX for entry",
        ),
    ],
    "strategies.hft_ofi_scalper.filters.volume_spike_filter": [
        OptimizableParam(
            name="min_spike_multiplier",
            param_type=ParamType.FLOAT,
            low=1.2,
            high=2.0,
            default=1.5,
            description="Minimum volume spike multiplier",
        ),
    ],
    "strategies.hft_ofi_scalper.throttling": [
        OptimizableParam(
            name="max_trades_per_minute",
            param_type=ParamType.INT,
            low=3,
            high=10,
            default=5,
            description="Maximum trades per minute",
        ),
        OptimizableParam(
            name="cooldown_after_loss_seconds",
            param_type=ParamType.INT,
            low=15,
            high=60,
            default=30,
            description="Cooldown after loss",
        ),
    ],
    # Strategies - Momentum Intraday
    "strategies.momentum_intraday.params": [
        OptimizableParam(
            name="ema_fast",
            param_type=ParamType.INT,
            low=5,
            high=12,
            default=9,
            description="Fast EMA period",
        ),
        OptimizableParam(
            name="ema_slow",
            param_type=ParamType.INT,
            low=15,
            high=30,
            default=21,
            description="Slow EMA period",
        ),
        OptimizableParam(
            name="rsi_entry_threshold",
            param_type=ParamType.INT,
            low=45,
            high=55,
            default=50,
            description="RSI entry threshold",
        ),
        OptimizableParam(
            name="holding_minutes",
            param_type=ParamType.INT,
            low=30,
            high=120,
            default=60,
            description="Position holding time",
        ),
    ],
    "strategies.momentum_intraday.filters.adx_filter": [
        OptimizableParam(
            name="min_adx",
            param_type=ParamType.INT,
            low=20,
            high=30,
            default=25,
            description="Minimum ADX for entry",
        ),
    ],
    "strategies.momentum_intraday.filters.volume_confirmation": [
        OptimizableParam(
            name="min_volume_percentile",
            param_type=ParamType.INT,
            low=50,
            high=70,
            default=60,
            description="Minimum volume percentile",
        ),
    ],
    "strategies.momentum_intraday.throttling": [
        OptimizableParam(
            name="max_trades_per_hour",
            param_type=ParamType.INT,
            low=2,
            high=8,
            default=4,
            description="Maximum trades per hour",
        ),
    ],
    # Strategies - Mean Reversion Intraday
    "strategies.mean_reversion_intraday.params": [
        OptimizableParam(
            name="zscore_entry",
            param_type=ParamType.FLOAT,
            low=1.5,
            high=2.5,
            default=2.0,
            description="Z-score entry threshold",
        ),
        OptimizableParam(
            name="zscore_exit",
            param_type=ParamType.FLOAT,
            low=0.25,
            high=0.75,
            default=0.5,
            description="Z-score exit threshold",
        ),
        OptimizableParam(
            name="bollinger_entry_std",
            param_type=ParamType.FLOAT,
            low=1.5,
            high=2.5,
            default=2.0,
            description="Bollinger entry standard deviation",
        ),
        OptimizableParam(
            name="holding_minutes",
            param_type=ParamType.INT,
            low=15,
            high=60,
            default=30,
            description="Position holding time",
        ),
    ],
    "strategies.mean_reversion_intraday.filters.adx_filter": [
        OptimizableParam(
            name="max_adx",
            param_type=ParamType.INT,
            low=15,
            high=25,
            default=20,
            description="Maximum ADX for entry",
        ),
    ],
    "strategies.mean_reversion_intraday.filters.liquidity_cluster_filter": [
        OptimizableParam(
            name="proximity_threshold_pct",
            param_type=ParamType.FLOAT,
            low=0.25,
            high=1.0,
            default=0.5,
            description="Liquidity proximity threshold",
        ),
    ],
    "strategies.mean_reversion_intraday.throttling": [
        OptimizableParam(
            name="max_trades_per_hour",
            param_type=ParamType.INT,
            low=4,
            high=10,
            default=6,
            description="Maximum trades per hour",
        ),
    ],
    # Strategies - Volatility Breakout
    "strategies.volatility_breakout.params": [
        OptimizableParam(
            name="squeeze_bb_width_percentile",
            param_type=ParamType.INT,
            low=10,
            high=30,
            default=20,
            description="Squeeze Bollinger width percentile",
        ),
        OptimizableParam(
            name="breakout_atr_multiplier",
            param_type=ParamType.FLOAT,
            low=1.0,
            high=2.0,
            default=1.5,
            description="Breakout ATR multiplier",
        ),
        OptimizableParam(
            name="volume_confirmation_mult",
            param_type=ParamType.FLOAT,
            low=1.2,
            high=2.0,
            default=1.5,
            description="Volume confirmation multiplier",
        ),
        OptimizableParam(
            name="holding_minutes",
            param_type=ParamType.INT,
            low=60,
            high=240,
            default=120,
            description="Position holding time",
        ),
    ],
    "strategies.volatility_breakout.filters.adx_filter": [
        OptimizableParam(
            name="min_adx_after_breakout",
            param_type=ParamType.INT,
            low=15,
            high=25,
            default=20,
            description="Minimum ADX after breakout",
        ),
    ],
    "strategies.volatility_breakout.throttling": [
        OptimizableParam(
            name="max_trades_per_day",
            param_type=ParamType.INT,
            low=2,
            high=5,
            default=3,
            description="Maximum trades per day",
        ),
    ],
    # Ensemble
    "ensemble": [
        OptimizableParam(
            name="weighting_method",
            param_type=ParamType.CATEGORICAL,
            choices=["sharpe", "equal", "inverse_vol"],
            default="sharpe",
            description="Ensemble weighting method",
        ),
        OptimizableParam(
            name="lookback_days",
            param_type=ParamType.INT,
            low=14,
            high=60,
            default=30,
            description="Lookback window for weighting",
        ),
        OptimizableParam(
            name="min_confidence",
            param_type=ParamType.FLOAT,
            low=0.4,
            high=0.6,
            default=0.5,
            description="Minimum confidence for signal",
        ),
        OptimizableParam(
            name="rebalance_frequency_hours",
            param_type=ParamType.INT,
            low=6,
            high=48,
            default=24,
            description="Rebalancing frequency",
        ),
    ],
    "ensemble.strategy_constraints": [
        OptimizableParam(
            name="max_weight_per_strategy",
            param_type=ParamType.FLOAT,
            low=0.4,
            high=0.8,
            default=0.6,
            description="Maximum weight per strategy",
        ),
    ],
    "ensemble.disable_strategy_if": [
        OptimizableParam(
            name="max_drawdown_pct",
            param_type=ParamType.FLOAT,
            low=10.0,
            high=20.0,
            default=15.0,
            description="Max drawdown to disable strategy",
        ),
        OptimizableParam(
            name="min_sharpe",
            param_type=ParamType.FLOAT,
            low=0.3,
            high=0.7,
            default=0.5,
            description="Minimum Sharpe to keep strategy",
        ),
    ],
    "ensemble.meta_labeling_filter": [
        OptimizableParam(
            name="min_meta_confidence",
            param_type=ParamType.FLOAT,
            low=0.5,
            high=0.7,
            default=0.6,
            description="Minimum meta-labeling confidence",
        ),
    ],
    # Risk Management
    "risk": [
        OptimizableParam(
            name="max_risk_per_trade_pct",
            param_type=ParamType.FLOAT,
            low=0.25,
            high=1.0,
            default=0.5,
            description="Max risk per trade percentage",
        ),
        OptimizableParam(
            name="max_daily_loss_pct",
            param_type=ParamType.FLOAT,
            low=2.0,
            high=5.0,
            default=3.0,
            description="Max daily loss percentage",
        ),
        OptimizableParam(
            name="max_weekly_loss_pct",
            param_type=ParamType.FLOAT,
            low=5.0,
            high=10.0,
            default=7.0,
            description="Max weekly loss percentage",
        ),
        OptimizableParam(
            name="max_monthly_loss_pct",
            param_type=ParamType.FLOAT,
            low=10.0,
            high=20.0,
            default=15.0,
            description="Max monthly loss percentage",
        ),
        OptimizableParam(
            name="max_drawdown_pct",
            param_type=ParamType.FLOAT,
            low=15.0,
            high=25.0,
            default=20.0,
            description="Max drawdown percentage",
        ),
        OptimizableParam(
            name="max_open_positions",
            param_type=ParamType.INT,
            low=1,
            high=3,
            default=1,
            description="Max open positions",
        ),
        OptimizableParam(
            name="max_leverage",
            param_type=ParamType.INT,
            low=5,
            high=20,
            default=10,
            description="Max leverage",
        ),
    ],
    "risk.regime_risk_adjustment": [
        OptimizableParam(
            name="crash_regime_multiplier",
            param_type=ParamType.FLOAT,
            low=0.1,
            high=0.5,
            default=0.3,
            description="Risk multiplier in crash regime",
        ),
        OptimizableParam(
            name="high_vol_regime_multiplier",
            param_type=ParamType.FLOAT,
            low=0.3,
            high=0.7,
            default=0.5,
            description="Risk multiplier in high vol regime",
        ),
    ],
    "risk.kill_switch": [
        OptimizableParam(
            name="max_loss_trigger_pct",
            param_type=ParamType.FLOAT,
            low=3.0,
            high=7.0,
            default=5.0,
            description="Kill switch trigger percentage",
        ),
        OptimizableParam(
            name="pause_duration_hours",
            param_type=ParamType.INT,
            low=12,
            high=48,
            default=24,
            description="Pause duration after kill switch",
        ),
    ],
    # Bet Sizing
    "bet_sizing.kelly": [
        OptimizableParam(
            name="fraction",
            param_type=ParamType.FLOAT,
            low=0.1,
            high=0.5,
            default=0.25,
            description="Kelly fraction",
        ),
        OptimizableParam(
            name="max_bet_pct",
            param_type=ParamType.FLOAT,
            low=0.05,
            high=0.2,
            default=0.1,
            description="Maximum bet percentage",
        ),
    ],
    "bet_sizing.vol_target": [
        OptimizableParam(
            name="target_annual_vol",
            param_type=ParamType.FLOAT,
            low=0.1,
            high=0.2,
            default=0.15,
            description="Target annual volatility",
        ),
    ],
    "bet_sizing.fixed": [
        OptimizableParam(
            name="position_size_pct",
            param_type=ParamType.FLOAT,
            low=0.01,
            high=0.05,
            default=0.02,
            description="Fixed position size percentage",
        ),
    ],
    # RL - PPO Agent
    "rl.agents.ppo": [
        OptimizableParam(
            name="learning_rate",
            param_type=ParamType.FLOAT,
            low=0.0001,
            high=0.001,
            default=0.0003,
            log_scale=True,
            description="PPO learning rate",
        ),
        OptimizableParam(
            name="n_steps",
            param_type=ParamType.INT,
            low=1024,
            high=4096,
            default=2048,
            description="PPO number of steps",
        ),
        OptimizableParam(
            name="batch_size",
            param_type=ParamType.INT,
            low=32,
            high=128,
            default=64,
            description="PPO batch size",
        ),
        OptimizableParam(
            name="n_epochs",
            param_type=ParamType.INT,
            low=5,
            high=20,
            default=10,
            description="PPO number of epochs",
        ),
        OptimizableParam(
            name="gamma",
            param_type=ParamType.FLOAT,
            low=0.95,
            high=0.999,
            default=0.99,
            description="PPO discount factor",
        ),
        OptimizableParam(
            name="gae_lambda",
            param_type=ParamType.FLOAT,
            low=0.9,
            high=0.99,
            default=0.95,
            description="PPO GAE lambda",
        ),
        OptimizableParam(
            name="clip_range",
            param_type=ParamType.FLOAT,
            low=0.1,
            high=0.3,
            default=0.2,
            description="PPO clip range",
        ),
    ],
    # RL - SAC Agent
    "rl.agents.sac": [
        OptimizableParam(
            name="learning_rate",
            param_type=ParamType.FLOAT,
            low=0.0001,
            high=0.001,
            default=0.0003,
            log_scale=True,
            description="SAC learning rate",
        ),
        OptimizableParam(
            name="buffer_size",
            param_type=ParamType.INT,
            low=100000,
            high=1000000,
            default=1000000,
            description="SAC replay buffer size",
        ),
        OptimizableParam(
            name="learning_starts",
            param_type=ParamType.INT,
            low=50,
            high=200,
            default=100,
            description="SAC learning starts",
        ),
        OptimizableParam(
            name="batch_size",
            param_type=ParamType.INT,
            low=128,
            high=512,
            default=256,
            description="SAC batch size",
        ),
        OptimizableParam(
            name="tau",
            param_type=ParamType.FLOAT,
            low=0.001,
            high=0.01,
            default=0.005,
            description="SAC target update rate",
        ),
        OptimizableParam(
            name="gamma",
            param_type=ParamType.FLOAT,
            low=0.95,
            high=0.999,
            default=0.99,
            description="SAC discount factor",
        ),
    ],
    # RL Training
    "rl.training": [
        OptimizableParam(
            name="total_timesteps",
            param_type=ParamType.INT,
            low=50000,
            high=500000,
            default=100000,
            description="Total training timesteps",
        ),
    ],
    # Online Learning
    "online_learning.drift_detection": [
        OptimizableParam(
            name="delta",
            param_type=ParamType.FLOAT,
            low=0.001,
            high=0.005,
            default=0.002,
            description="ADWIN delta parameter",
        ),
    ],
    # Backtest
    "backtest.slippage": [
        OptimizableParam(
            name="base_bps",
            param_type=ParamType.FLOAT,
            low=0.5,
            high=2.0,
            default=1.0,
            description="Base slippage in basis points",
        ),
        OptimizableParam(
            name="volume_impact",
            param_type=ParamType.FLOAT,
            low=0.3,
            high=1.0,
            default=0.5,
            description="Volume impact factor",
        ),
        OptimizableParam(
            name="spread_factor",
            param_type=ParamType.FLOAT,
            low=0.5,
            high=1.5,
            default=1.0,
            description="Spread impact factor",
        ),
    ],
    "backtest.walk_forward": [
        OptimizableParam(
            name="train_days",
            param_type=ParamType.INT,
            low=30,
            high=90,
            default=60,
            description="Training window days",
        ),
        OptimizableParam(
            name="test_days",
            param_type=ParamType.INT,
            low=7,
            high=21,
            default=14,
            description="Testing window days",
        ),
        OptimizableParam(
            name="step_days",
            param_type=ParamType.INT,
            low=3,
            high=14,
            default=7,
            description="Step days between windows",
        ),
    ],
}

# Feature toggles (enabled/disabled)
FEATURE_TOGGLES: Dict[str, OptimizableParam] = {
    "features.microstructure.ofi.enabled": OptimizableParam(
        name="enabled",
        param_type=ParamType.BOOL,
        default=True,
        description="Enable OFI feature",
    ),
    "features.microstructure.tfi.enabled": OptimizableParam(
        name="enabled",
        param_type=ParamType.BOOL,
        default=True,
        description="Enable TFI feature",
    ),
    "features.microstructure.micro_price.enabled": OptimizableParam(
        name="enabled",
        param_type=ParamType.BOOL,
        default=True,
        description="Enable Micro-Price feature",
    ),
    "features.microstructure.entropy.enabled": OptimizableParam(
        name="enabled",
        param_type=ParamType.BOOL,
        default=True,
        description="Enable Entropy feature",
    ),
    "features.microstructure.vpin.enabled": OptimizableParam(
        name="enabled",
        param_type=ParamType.BOOL,
        default=True,
        description="Enable VPIN feature",
    ),
    "features.technical.rsi.enabled": OptimizableParam(
        name="enabled",
        param_type=ParamType.BOOL,
        default=True,
        description="Enable RSI feature",
    ),
    "features.technical.macd.enabled": OptimizableParam(
        name="enabled",
        param_type=ParamType.BOOL,
        default=True,
        description="Enable MACD feature",
    ),
    "features.technical.adx.enabled": OptimizableParam(
        name="enabled",
        param_type=ParamType.BOOL,
        default=True,
        description="Enable ADX feature",
    ),
    "features.technical.bollinger.enabled": OptimizableParam(
        name="enabled",
        param_type=ParamType.BOOL,
        default=True,
        description="Enable Bollinger Bands feature",
    ),
    "features.technical.atr.enabled": OptimizableParam(
        name="enabled",
        param_type=ParamType.BOOL,
        default=True,
        description="Enable ATR feature",
    ),
    "features.volume_analysis.volume_spike.enabled": OptimizableParam(
        name="enabled",
        param_type=ParamType.BOOL,
        default=True,
        description="Enable Volume Spike feature",
    ),
    "features.volume_analysis.liquidity_clusters.enabled": OptimizableParam(
        name="enabled",
        param_type=ParamType.BOOL,
        default=True,
        description="Enable Liquidity Clusters feature",
    ),
}

# Strategy toggles
STRATEGY_TOGGLES: Dict[str, OptimizableParam] = {
    "strategies.hft_ofi_scalper.enabled": OptimizableParam(
        name="enabled",
        param_type=ParamType.BOOL,
        default=True,
        description="Enable HFT OFI Scalper strategy",
    ),
    "strategies.momentum_intraday.enabled": OptimizableParam(
        name="enabled",
        param_type=ParamType.BOOL,
        default=True,
        description="Enable Momentum Intraday strategy",
    ),
    "strategies.mean_reversion_intraday.enabled": OptimizableParam(
        name="enabled",
        param_type=ParamType.BOOL,
        default=True,
        description="Enable Mean Reversion Intraday strategy",
    ),
    "strategies.volatility_breakout.enabled": OptimizableParam(
        name="enabled",
        param_type=ParamType.BOOL,
        default=True,
        description="Enable Volatility Breakout strategy",
    ),
}


def get_all_optimizable_params() -> Tuple[Dict[str, List[OptimizableParam]], int]:
    """
    Get all optimizable parameters with their definitions.

    Returns:
        Tuple of (params dict, total count)
    """
    total_count = 0
    for params_list in OPTIMIZABLE_PARAMS.values():
        total_count += len(params_list)

    # Add toggles
    total_count += len(FEATURE_TOGGLES)
    total_count += len(STRATEGY_TOGGLES)

    return OPTIMIZABLE_PARAMS, total_count


def get_feature_toggles() -> Dict[str, OptimizableParam]:
    """Get all feature toggle parameters."""
    return FEATURE_TOGGLES


def get_strategy_toggles() -> Dict[str, OptimizableParam]:
    """Get all strategy toggle parameters."""
    return STRATEGY_TOGGLES


def count_optimizable_params() -> int:
    """Count total optimizable parameters."""
    _, count = get_all_optimizable_params()
    return count


# Verify we have ~150 parameters as specified
if __name__ == "__main__":
    params, count = get_all_optimizable_params()
    print(f"Total optimizable parameters: {count}")
    print("\nBy section:")
    for section, params_list in params.items():
        print(f"  {section}: {len(params_list)} params")
    print(f"\nFeature toggles: {len(FEATURE_TOGGLES)}")
    print(f"Strategy toggles: {len(STRATEGY_TOGGLES)}")
