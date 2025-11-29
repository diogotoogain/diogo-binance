#!/usr/bin/env python3
"""
Backtest Script for V2 Trading Strategies

CLI script to run backtests on historical data with proper configuration.

Usage:
    python v2/scripts/run_backtest.py --symbol BTCUSDT --timeframe 1m \
        --strategy momentum_intraday --data-path v2/v2/data/raw/BTCUSDT_1m.parquet
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import numpy as np

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from v2.src.config.loader import ConfigLoader, get_config
from v2.src.risk.risk_manager import RiskManager
from v2.src.backtest.engine import BacktestEngine, Signal
from v2.src.backtest.report import BacktestReport
from v2.src.strategies.base import Strategy, SignalDirection


# Strategy mapping
STRATEGY_MAP = {
    'momentum_intraday': 'MomentumIntraday',
    'hft_ofi_scalper': 'HFTOFIScalper',
    'mean_reversion_intraday': 'MeanReversionIntraday',
    'volatility_breakout': 'VolatilityBreakout',
}


def get_strategy_class(strategy_name: str):
    """
    Get strategy class by name.
    
    Args:
        strategy_name: Strategy identifier (e.g., 'momentum_intraday')
        
    Returns:
        Strategy class
    """
    if strategy_name == 'momentum_intraday':
        from v2.src.strategies.momentum_intraday import MomentumIntraday
        return MomentumIntraday
    elif strategy_name == 'hft_ofi_scalper':
        from v2.src.strategies.hft_ofi_scalper import HFTOFIScalper
        return HFTOFIScalper
    elif strategy_name == 'mean_reversion_intraday':
        from v2.src.strategies.mean_reversion_intraday import MeanReversionIntraday
        return MeanReversionIntraday
    elif strategy_name == 'volatility_breakout':
        from v2.src.strategies.volatility_breakout import VolatilityBreakout
        return VolatilityBreakout
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGY_MAP.keys())}")


class BacktestStrategyAdapter:
    """
    Adapter to use V2 strategies with BacktestEngine.
    
    Converts between strategy signal format and backtest engine format.
    """
    
    def __init__(self, strategy: Strategy, risk_manager: RiskManager):
        """
        Initialize adapter.
        
        Args:
            strategy: V2 Strategy instance
            risk_manager: RiskManager instance
        """
        self.strategy = strategy
        self.risk_manager = risk_manager
        
    def generate_signal(self, features: Dict[str, Any], context: Dict[str, Any]) -> Optional[Signal]:
        """
        Generate signal compatible with BacktestEngine.
        
        Args:
            features: Current bar features
            context: Trading context (balance, equity, etc.)
            
        Returns:
            BacktestEngine.Signal or None
        """
        # Check risk manager before generating signal
        can_trade, reason = self.risk_manager.can_open_position(
            balance=context.get('balance', 10000)
        )
        
        if not can_trade:
            return None
            
        # Build market data for strategy
        market_data = {
            'price': features.get('close', features.get('Close', 0)),
            'ofi': features.get('ofi', 0),
            'tfi': features.get('tfi', 0),
            'rsi': features.get('rsi', 50),
            'adx': features.get('adx', 25),
            'atr': features.get('atr', features.get('ATR', 0)),
            'ema_fast': features.get('ema_fast', features.get('ema_9', 0)),
            'ema_slow': features.get('ema_slow', features.get('ema_21', 0)),
            'volume_percentile': features.get('volume_percentile', 50),
            'volume_spike': features.get('volume_spike', 1.0),
            'spread_bps': features.get('spread_bps', 1.0),
            'zscore': features.get('zscore', 0),
            'bb_upper': features.get('bb_upper', 0),
            'bb_lower': features.get('bb_lower', 0),
            'bb_width_percentile': features.get('bb_width_percentile', 50),
        }
        
        # Generate strategy signal
        strategy_signal = self.strategy.generate_signal(market_data)
        
        if strategy_signal is None:
            return None
            
        # Convert to backtest engine signal format
        direction = 0
        if strategy_signal.direction == SignalDirection.BUY:
            direction = 1
        elif strategy_signal.direction == SignalDirection.SELL:
            direction = -1
            
        if direction == 0:
            return None
            
        return Signal(
            direction=direction,
            confidence=strategy_signal.confidence,
            size_hint=1.0,
            sl_mult=1.5,
            tp_mult=2.5,
        )


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load market data from file.
    
    Args:
        data_path: Path to data file (parquet or CSV)
        
    Returns:
        DataFrame with OHLCV data
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    if data_path.suffix == '.parquet':
        df = pd.read_parquet(data_path)
    elif data_path.suffix == '.csv':
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
    # Ensure required columns exist (normalize column names)
    column_mapping = {
        'Open': 'open', 'High': 'high', 'Low': 'low', 
        'Close': 'close', 'Volume': 'volume'
    }
    df.rename(columns=column_mapping, inplace=True)
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
        
    # Set index if timestamp column exists
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
    return df


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic technical features for backtesting.
    
    Args:
        df: OHLCV DataFrame
        
    Returns:
        DataFrame with added features
    """
    features = df.copy()
    
    # EMAs
    features['ema_9'] = features['close'].ewm(span=9, adjust=False).mean()
    features['ema_21'] = features['close'].ewm(span=21, adjust=False).mean()
    features['ema_fast'] = features['ema_9']
    features['ema_slow'] = features['ema_21']
    
    # RSI
    delta = features['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.inf)
    features['rsi'] = 100 - (100 / (1 + rs))
    features['rsi'] = features['rsi'].fillna(50)
    
    # ATR
    high_low = features['high'] - features['low']
    high_close = (features['high'] - features['close'].shift()).abs()
    low_close = (features['low'] - features['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    features['atr'] = tr.rolling(window=14).mean()
    features['ATR'] = features['atr']
    
    # ADX (simplified)
    plus_dm = features['high'].diff()
    minus_dm = features['low'].diff().abs()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    smoothed_plus_dm = plus_dm.rolling(window=14).mean()
    smoothed_minus_dm = minus_dm.rolling(window=14).mean()
    smoothed_tr = tr.rolling(window=14).mean()
    
    plus_di = 100 * (smoothed_plus_dm / smoothed_tr.replace(0, np.inf))
    minus_di = 100 * (smoothed_minus_dm / smoothed_tr.replace(0, np.inf))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.inf))
    features['adx'] = dx.rolling(window=14).mean()
    features['adx'] = features['adx'].fillna(25)
    
    # Volume features
    features['volume_sma'] = features['volume'].rolling(window=20).mean()
    features['volume_spike'] = features['volume'] / features['volume_sma'].replace(0, 1)
    features['volume_percentile'] = features['volume'].rolling(window=100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).fillna(50)
    
    # Z-score
    features['zscore'] = (features['close'] - features['close'].rolling(window=20).mean()) / \
                         features['close'].rolling(window=20).std().replace(0, 1)
    features['zscore'] = features['zscore'].fillna(0)
    
    # Bollinger Bands
    bb_middle = features['close'].rolling(window=20).mean()
    bb_std = features['close'].rolling(window=20).std()
    features['bb_upper'] = bb_middle + (bb_std * 2)
    features['bb_lower'] = bb_middle - (bb_std * 2)
    features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / bb_middle
    features['bb_width_percentile'] = features['bb_width'].rolling(window=100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).fillna(50)
    
    # OFI and TFI (simulated for backtest)
    features['ofi'] = np.random.randn(len(features)) * 0.2
    features['tfi'] = np.random.randn(len(features)) * 0.2
    
    # Spread (simulated)
    features['spread_bps'] = 1.0 + np.random.rand(len(features)) * 0.5
    
    # Fill NaN values
    features = features.fillna(method='ffill').fillna(method='bfill')
    
    return features


def generate_synthetic_data(n_bars: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for testing.
    
    Args:
        n_bars: Number of bars to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic OHLCV data
    """
    np.random.seed(seed)
    
    # Generate price series with random walk
    returns = np.random.randn(n_bars) * 0.002  # ~0.2% volatility per bar
    prices = 50000 * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    dates = pd.date_range(start='2024-01-01', periods=n_bars, freq='1min')
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(n_bars) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n_bars) * 0.003)),
        'low': prices * (1 - np.abs(np.random.randn(n_bars) * 0.003)),
        'close': prices,
        'volume': np.random.randint(100, 10000, n_bars),
    }, index=dates)
    
    # Ensure high > low and high >= close >= low
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


def print_metrics(results: Dict[str, Any], strategy_name: str) -> None:
    """
    Print backtest metrics in a formatted way.
    
    Args:
        results: Backtest results dictionary
        strategy_name: Name of the strategy
    """
    print("\n" + "=" * 70)
    print(f"📊 BACKTEST RESULTS - {strategy_name.upper()}")
    print("=" * 70)
    
    print(f"\n💰 Performance:")
    print(f"   Total Return:     {results.get('total_return_pct', 0):.2f}%")
    print(f"   Final Equity:     ${results.get('final_equity', 0):,.2f}")
    
    print(f"\n📈 Risk Metrics:")
    print(f"   Sharpe Ratio:     {results.get('sharpe_ratio', 0):.3f}")
    print(f"   Sortino Ratio:    {results.get('sortino_ratio', 0):.3f}")
    print(f"   Max Drawdown:     {results.get('max_drawdown', 0):.2f}%")
    print(f"   Volatility:       {results.get('volatility', 0):.2f}%")
    
    print(f"\n🎯 Trading Stats:")
    print(f"   Total Trades:     {results.get('total_trades', 0)}")
    print(f"   Win Rate:         {results.get('win_rate', 0):.1f}%")
    print(f"   Profit Factor:    {results.get('profit_factor', 0):.2f}")
    print(f"   Avg Trade:        ${results.get('avg_trade_pnl', 0):.2f}")
    
    print(f"\n⚡ Additional:")
    print(f"   VaR (95%):        {results.get('var_95', 0):.2f}%")
    print(f"   CVaR (95%):       {results.get('cvar_95', 0):.2f}%")
    
    print("\n" + "=" * 70)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run backtest on V2 trading strategies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading symbol"
    )
    
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1m",
        help="Data timeframe"
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        default="momentum_intraday",
        choices=list(STRATEGY_MAP.keys()),
        help="Strategy to backtest"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to data file (parquet or CSV). If not provided, uses synthetic data."
    )
    
    parser.add_argument(
        "--config-path",
        type=str,
        default="v2/config/default.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000,
        help="Initial capital for backtest"
    )
    
    parser.add_argument(
        "--synthetic-bars",
        type=int,
        default=5000,
        help="Number of bars for synthetic data (if no data-path)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("🚀 V2 BACKTEST ENGINE")
    print("=" * 70)
    print(f"   Symbol:     {args.symbol}")
    print(f"   Timeframe:  {args.timeframe}")
    print(f"   Strategy:   {args.strategy}")
    print(f"   Capital:    ${args.initial_capital:,.2f}")
    print("=" * 70)
    
    # Reset ConfigLoader singleton to ensure fresh load
    ConfigLoader.reset()
    
    # Load configuration
    try:
        config = get_config(args.config_path, skip_env=True)
        print(f"\n✅ Configuration loaded from: {args.config_path}")
    except Exception as e:
        print(f"\n❌ Error loading config: {e}")
        sys.exit(1)
    
    # Get full config dictionary
    full_config = config.config
    
    # Override initial capital in config
    if 'backtest' not in full_config:
        full_config['backtest'] = {}
    full_config['backtest']['initial_capital'] = args.initial_capital
    
    # Initialize RiskManager
    try:
        risk_manager = RiskManager(full_config)
        risk_manager.set_initial_balance(args.initial_capital)
        print("✅ RiskManager initialized")
    except Exception as e:
        print(f"❌ Error initializing RiskManager: {e}")
        sys.exit(1)
    
    # Load or generate data
    if args.data_path:
        try:
            print(f"\n📥 Loading data from: {args.data_path}")
            data = load_data(args.data_path)
            print(f"   Loaded {len(data)} bars")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            sys.exit(1)
    else:
        print(f"\n🎲 Generating synthetic data ({args.synthetic_bars} bars)...")
        data = generate_synthetic_data(n_bars=args.synthetic_bars)
        print(f"   Generated {len(data)} bars")
    
    # Add features
    print("🔧 Adding technical features...")
    features = add_basic_features(data)
    
    # Get strategy configuration
    strategy_config = full_config.get('strategies', {}).get(args.strategy, {})
    if not strategy_config:
        print(f"⚠️  No specific config for {args.strategy}, using defaults")
        strategy_config = {'enabled': True, 'params': {}, 'filters': {}, 'throttling': {}}
    
    # Initialize strategy
    try:
        strategy_class = get_strategy_class(args.strategy)
        strategy = strategy_class(strategy_config)
        print(f"✅ Strategy '{args.strategy}' initialized")
    except Exception as e:
        print(f"❌ Error initializing strategy: {e}")
        sys.exit(1)
    
    # Create strategy adapter for backtest engine
    adapter = BacktestStrategyAdapter(strategy, risk_manager)
    
    # Initialize backtest engine
    engine = BacktestEngine(full_config)
    print("✅ BacktestEngine initialized")
    
    # Run backtest
    print("\n🏃 Running backtest...")
    try:
        results = engine.run(
            data=data,
            strategy=adapter,
            features=features,
            progress_callback=lambda i, n: print(f"\r   Progress: {i}/{n} ({100*i/n:.1f}%)", end="") if args.verbose else None
        )
        print("\n✅ Backtest complete!")
    except Exception as e:
        print(f"\n❌ Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print metrics
    print_metrics(results, args.strategy)
    
    # Save results if output path provided
    if args.output:
        try:
            report = BacktestReport.from_backtest_results(results)
            report.save_json(args.output)
            print(f"\n💾 Results saved to: {args.output}")
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ BACKTEST COMPLETE")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
