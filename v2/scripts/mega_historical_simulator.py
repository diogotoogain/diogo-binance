#!/usr/bin/env python3
"""
MEGA SIMULADOR HISTÓRICO

Simulates 6 years of trading with look-ahead bias protection.
Trains Online Learning (ADWIN + Hoeffding) incrementally.
Optimizes parameters automatically with Walk-Forward.

Usage:
    python v2/scripts/mega_historical_simulator.py \
        --config v2/config/mega_sim_config.yaml

    # Quick test (1 month)
    python v2/scripts/mega_historical_simulator.py \
        --config v2/config/mega_sim_config.yaml \
        --start-date 2024-01-01 \
        --end-date 2024-01-31

    # With optimization
    python v2/scripts/mega_historical_simulator.py \
        --config v2/config/mega_sim_config.yaml \
        --optimize
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from v2.src.data.local_data_loader import LocalDataLoader
from v2.src.backtesting.honest_simulator import HonestSimulator, SimulationConfig
from v2.src.optimization.parameter_optimizer import ParameterOptimizer, OptimizationConfig
from v2.src.reporting.simulation_report import SimulationReport

# Optional imports
try:
    from v2.src.online_learning.river_models import OnlineLearner
    from v2.src.online_learning.drift_detector import MarketDriftDetector
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False

try:
    from v2.src.strategies.hft_ofi_scalper import HFTOFIScalper
    STRATEGY_AVAILABLE = True
except ImportError:
    STRATEGY_AVAILABLE = False


def setup_logging(config: dict) -> logging.Logger:
    """Setup logging based on configuration."""
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))

    # Create logger
    logger = logging.getLogger('mega_sim')
    logger.setLevel(level)

    # Console handler
    if log_config.get('console', True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_config.get('file', False):
        file_path = log_config.get('file_path', 'v2/logs/mega_sim.log')
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def print_banner():
    """Print welcome banner."""
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║   ███╗   ███╗███████╗ ██████╗  █████╗                            ║
║   ████╗ ████║██╔════╝██╔════╝ ██╔══██╗                           ║
║   ██╔████╔██║█████╗  ██║  ███╗███████║                           ║
║   ██║╚██╔╝██║██╔══╝  ██║   ██║██╔══██║                           ║
║   ██║ ╚═╝ ██║███████╗╚██████╔╝██║  ██║                           ║
║   ╚═╝     ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝                           ║
║                                                                   ║
║   SIMULADOR HISTÓRICO                                            ║
║   ────────────────────                                           ║
║   🔒 Look-Ahead Bias Protection                                  ║
║   🧠 Online Learning (ADWIN + Hoeffding)                         ║
║   ⚙️  Walk-Forward Optimization                                   ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)


def create_simulation_config(config: dict) -> SimulationConfig:
    """Create SimulationConfig from YAML config."""
    sim_config = config.get('simulation', {})
    return SimulationConfig(
        initial_balance=sim_config.get('initial_balance', 10000),
        trade_cost_pct=sim_config.get('trade_cost_pct', 0.0004),
        slippage_pct=sim_config.get('slippage_pct', 0.0001),
        max_leverage=sim_config.get('max_leverage', 5),
        warmup_samples=sim_config.get('warmup_samples', 1000),
        lookback_window=sim_config.get('lookback_window', 100)
    )


def create_online_learner(config: dict) -> 'OnlineLearner':
    """Create OnlineLearner from config."""
    if not RIVER_AVAILABLE:
        return None

    ol_config = config.get('online_learning', {})
    if not ol_config.get('enabled', True):
        return None

    model_config = ol_config.get('model', {})
    return OnlineLearner(
        grace_period=model_config.get('grace_period', 200),
        delta=model_config.get('delta', 1e-7),
        seed=model_config.get('seed', 42)
    )


def create_drift_detector(config: dict) -> 'MarketDriftDetector':
    """Create MarketDriftDetector from config."""
    if not RIVER_AVAILABLE:
        return None

    ol_config = config.get('online_learning', {})
    drift_config = ol_config.get('drift_detection', {})

    if not drift_config.get('enabled', True):
        return None

    return MarketDriftDetector(
        pnl_delta=drift_config.get('pnl_delta', 0.002),
        win_rate_delta=drift_config.get('win_rate_delta', 0.01),
        spread_delta=drift_config.get('spread_delta', 0.005),
        pause_threshold=drift_config.get('pause_threshold', 2),
        cooldown_samples=drift_config.get('cooldown_samples', 50)
    )


def create_strategy(config: dict):
    """Create strategy from config."""
    if not STRATEGY_AVAILABLE:
        return None

    strategy_config = config.get('strategy', {})
    strategy_type = strategy_config.get('type', 'hft_ofi_scalper')

    if strategy_type == 'hft_ofi_scalper':
        return HFTOFIScalper(strategy_config)

    return None


def run_simulation(
    config: dict,
    data,
    logger: logging.Logger,
    optimize: bool = False
) -> dict:
    """Run the simulation."""

    # Create components
    sim_config = create_simulation_config(config)
    online_learner = create_online_learner(config)
    drift_detector = create_drift_detector(config)
    strategy = create_strategy(config)

    logger.info("Components created:")
    logger.info(f"  - Online Learner: {'Enabled' if online_learner else 'Disabled'}")
    logger.info(f"  - Drift Detector: {'Enabled' if drift_detector else 'Disabled'}")
    logger.info(f"  - Strategy: {strategy.__class__.__name__ if strategy else 'None'}")

    # Run optimization if requested
    if optimize and config.get('optimization', {}).get('enabled', False):
        logger.info("Running Walk-Forward Optimization...")
        opt_result = run_optimization(config, data, logger)
        logger.info(f"Best parameters: {opt_result.best_params}")
        # Update strategy with optimized parameters
        # (In production, would update strategy config here)

    # Create simulator
    simulator = HonestSimulator(
        config=sim_config,
        online_learner=online_learner,
        strategy=strategy,
        drift_detector=drift_detector
    )

    # Progress callback
    def progress_callback(current: int, total: int):
        if current % 50000 == 0:
            pct = current / total * 100
            logger.info(f"Progress: {current:,}/{total:,} ({pct:.1f}%)")

    # Run simulation
    logger.info(f"Starting simulation with {len(data):,} candles...")
    start_time = datetime.now()

    result = simulator.run(
        data=data,
        progress_callback=progress_callback
    )

    elapsed = datetime.now() - start_time
    logger.info(f"Simulation completed in {elapsed}")

    return result


def run_optimization(config: dict, data, logger: logging.Logger):
    """Run walk-forward optimization."""
    opt_config = config.get('optimization', {})

    optimizer = ParameterOptimizer(OptimizationConfig(
        metric=opt_config.get('metric', 'sortino'),
        train_pct=opt_config.get('train_pct', 0.8),
        walk_forward_windows=opt_config.get('walk_forward_windows', 12),
        min_samples_per_window=opt_config.get('min_samples_per_window', 5000)
    ))

    param_grid = opt_config.get('param_grid', {})

    def progress_callback(current: int, total: int):
        if current % 100 == 0:
            logger.info(f"Optimization: {current}/{total}")

    result = optimizer.optimize(
        data=data,
        param_grid=param_grid,
        progress_callback=progress_callback
    )

    # Print report
    report = optimizer.generate_report(result)
    print(report)

    return result


def save_results(result, config: dict, logger: logging.Logger):
    """Save simulation results."""
    output_config = config.get('output', {})
    output_dir = output_config.get('dir', 'v2/results/mega_sim')

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate report
    report = SimulationReport(result, config)

    if output_config.get('generate_report', True):
        report.generate(output_dir)

    if output_config.get('save_model', True) and RIVER_AVAILABLE:
        # Save online model if available
        if hasattr(result, 'online_model_stats') and result.online_model_stats:
            import json
            with open(Path(output_dir) / 'online_model_stats.json', 'w') as f:
                json.dump(result.online_model_stats, f, indent=2, default=str)

    # Print summary
    report.print_summary()

    logger.info(f"Results saved to {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='MEGA SIMULADOR HISTÓRICO - Backtesting with Look-Ahead Bias Protection'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='v2/config/mega_sim_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Override data directory from config'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        help='Override timeframe from config'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Override start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='Override end date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Run walk-forward optimization'
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with limited data'
    )

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Load config
    config = load_config(args.config)

    # Override config with command line arguments
    if args.data_dir:
        config['data']['base_dir'] = args.data_dir
    if args.timeframe:
        config['data']['primary_timeframe'] = args.timeframe
    if args.start_date:
        config['data']['start_date'] = args.start_date
    if args.end_date:
        config['data']['end_date'] = args.end_date
    if args.quick_test:
        # Use only 1 month of data for quick test
        from datetime import timedelta
        start_dt = datetime.strptime(config['data']['start_date'], "%Y-%m-%d")
        end_dt = start_dt + timedelta(days=30)
        config['data']['end_date'] = end_dt.strftime("%Y-%m-%d")

    # Setup logging
    logger = setup_logging(config)
    logger.info("MEGA SIMULADOR HISTÓRICO starting...")

    # Get data configuration
    data_config = config.get('data', {})
    data_dir = data_config.get('base_dir')
    timeframe = data_config.get('primary_timeframe', '1m')
    start_date = data_config.get('start_date', '2019-12-31')
    end_date = data_config.get('end_date', '2025-11-28')
    symbol = data_config.get('symbol', 'BTCUSDT')

    logger.info(f"Data configuration:")
    logger.info(f"  - Directory: {data_dir}")
    logger.info(f"  - Timeframe: {timeframe}")
    logger.info(f"  - Period: {start_date} to {end_date}")
    logger.info(f"  - Symbol: {symbol}")

    # Load data
    logger.info("Loading data...")
    loader = LocalDataLoader()

    try:
        data = loader.load_timeframe(
            data_dir=data_dir,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            symbol=symbol
        )
    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}")
        logger.info("Please ensure you have downloaded historical data to the specified directory.")
        logger.info(f"Expected structure: {data_dir}/{timeframe}/{symbol}-{timeframe}-YYYY-MM-DD.zip")
        sys.exit(1)

    if data.empty:
        logger.error("No data loaded. Please check your data directory and date range.")
        sys.exit(1)

    logger.info(f"Loaded {len(data):,} candles")

    # Estimate candle count
    expected = loader.estimate_candle_count(start_date, end_date, timeframe)
    logger.info(f"Expected ~{expected:,} candles, loaded {len(data):,}")

    # Run simulation
    result = run_simulation(config, data, logger, optimize=args.optimize)

    # Save results
    save_results(result, config, logger)

    logger.info("MEGA SIMULADOR HISTÓRICO completed successfully!")


if __name__ == '__main__':
    main()
