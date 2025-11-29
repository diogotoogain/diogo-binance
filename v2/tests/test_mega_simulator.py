"""
Tests for the MEGA Historical Simulator components.
"""

import sys
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest

# Add project root to path for relative imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestLocalDataLoader:
    """Tests for LocalDataLoader class."""

    @pytest.fixture
    def loader(self):
        """Create LocalDataLoader instance."""
        # Import directly to avoid chain imports
        from v2.src.data.local_data_loader import LocalDataLoader
        return LocalDataLoader()

    @pytest.fixture
    def sample_data_dir(self, tmp_path):
        """Create a temporary directory with sample data."""
        import zipfile
        import csv

        # Create directory structure
        data_dir = tmp_path / "BTCUSDT" / "1m"
        data_dir.mkdir(parents=True)

        # Create sample CSV data
        columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ]

        # Create a few days of sample data
        for day in range(1, 4):
            date_str = f"2024-01-0{day}"
            csv_path = data_dir / f"BTCUSDT-1m-{date_str}.csv"

            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # Write 100 candles per file
                base_ts = int(datetime(2024, 1, day).timestamp() * 1000)
                for i in range(100):
                    row = [
                        base_ts + i * 60000,  # open_time
                        42000 + i,            # open
                        42100 + i,            # high
                        41900 + i,            # low
                        42050 + i,            # close
                        1000 + i * 10,        # volume
                        base_ts + (i + 1) * 60000,  # close_time
                        42000000,             # quote_volume
                        100,                  # trades
                        500,                  # taker_buy_volume
                        21000000,             # taker_buy_quote_volume
                        0                     # ignore
                    ]
                    writer.writerow(row)

        return tmp_path / "BTCUSDT"

    def test_initialization(self, loader):
        """LocalDataLoader should initialize correctly."""
        assert loader is not None
        assert loader.data_dir is None

    def test_load_timeframe_csv(self, loader, sample_data_dir):
        """load_timeframe should load CSV files correctly."""
        df = loader.load_timeframe(
            data_dir=str(sample_data_dir),
            timeframe="1m",
            start_date="2024-01-01",
            end_date="2024-01-03",
            symbol="BTCUSDT"
        )

        # Should have loaded data
        assert len(df) > 0
        # Should have required columns
        assert 'open' in df.columns
        assert 'close' in df.columns
        assert 'timestamp' in df.columns
        # Should be sorted
        assert df['open_time'].is_monotonic_increasing

    def test_load_timeframe_empty_dir(self, loader, tmp_path):
        """load_timeframe should handle missing directory."""
        # Create directory but no files
        data_dir = tmp_path / "empty"
        data_dir.mkdir()
        (data_dir / "1m").mkdir()

        df = loader.load_timeframe(
            data_dir=str(data_dir),
            timeframe="1m",
            start_date="2024-01-01",
            end_date="2024-01-31",
            symbol="BTCUSDT"
        )

        # Should return empty DataFrame
        assert len(df) == 0

    def test_estimate_candle_count(self, loader):
        """estimate_candle_count should return reasonable estimates."""
        # 1 day of 1m candles = 1440
        count_1m = loader.estimate_candle_count("2024-01-01", "2024-01-02", "1m")
        assert count_1m == 1440

        # 1 day of 5m candles = 288
        count_5m = loader.estimate_candle_count("2024-01-01", "2024-01-02", "5m")
        assert count_5m == 288

    def test_extract_date_from_filename(self, loader):
        """_extract_date_from_filename should parse dates correctly."""
        date = loader._extract_date_from_filename("BTCUSDT-1m-2024-01-15.zip")
        assert date is not None
        assert date.year == 2024
        assert date.month == 1
        assert date.day == 15

        # Invalid filename
        date = loader._extract_date_from_filename("invalid.zip")
        assert date is None


class TestHonestSimulator:
    """Tests for HonestSimulator class."""

    @pytest.fixture
    def simulator_config(self):
        """Create SimulationConfig instance."""
        from v2.src.backtesting.honest_simulator import SimulationConfig
        return SimulationConfig(
            initial_balance=10000,
            trade_cost_pct=0.0004,
            slippage_pct=0.0001,
            max_leverage=5,
            warmup_samples=10,
            lookback_window=5
        )

    @pytest.fixture
    def simulator(self, simulator_config):
        """Create HonestSimulator instance."""
        from v2.src.backtesting.honest_simulator import HonestSimulator
        return HonestSimulator(config=simulator_config)

    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
        np.random.seed(42)
        n_samples = 100

        # Generate synthetic price data
        prices = 42000 + np.cumsum(np.random.randn(n_samples) * 10)

        df = pd.DataFrame({
            'open_time': range(n_samples),
            'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='1min'),
            'open': prices,
            'high': prices + np.abs(np.random.randn(n_samples) * 5),
            'low': prices - np.abs(np.random.randn(n_samples) * 5),
            'close': prices + np.random.randn(n_samples) * 2,
            'volume': np.random.randint(100, 1000, n_samples)
        })

        return df

    def test_initialization(self, simulator, simulator_config):
        """HonestSimulator should initialize correctly."""
        assert simulator.balance == simulator_config.initial_balance
        assert simulator.position is None
        assert len(simulator.trades) == 0
        assert len(simulator.equity_curve) == 1

    def test_run_simulation(self, simulator, sample_data):
        """run should execute simulation and return results."""
        result = simulator.run(sample_data)

        assert result is not None
        assert hasattr(result, 'trades')
        assert hasattr(result, 'equity_curve')
        assert hasattr(result, 'metrics')
        # Equity curve should have grown
        assert len(result.equity_curve) > 1

    def test_no_look_ahead_bias(self, simulator, sample_data):
        """Simulator should not access future data."""
        # Track what data the simulator sees
        data_access_log = []

        def tracking_feature_calculator(historical):
            # Log the last timestamp in historical data
            data_access_log.append(len(historical))
            return {'test_feature': 1.0}

        simulator.run(sample_data, feature_calculator=tracking_feature_calculator)

        # Each call should see progressively more data
        # and never see the full dataset until the end
        for i, size in enumerate(data_access_log[:-1]):
            assert size < len(sample_data), f"Saw future data at step {i}"

    def test_metrics_calculation(self, simulator, sample_data):
        """Metrics should be calculated correctly."""
        result = simulator.run(sample_data)
        metrics = result.metrics

        # Should have standard metrics
        assert 'total_return_pct' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'win_rate' in metrics
        assert 'total_trades' in metrics

    def test_reset(self, simulator, sample_data):
        """reset should clear all state."""
        # Run simulation
        simulator.run(sample_data)
        assert simulator.n_samples_processed > 0

        # Reset
        simulator.reset()

        assert simulator.balance == simulator.config.initial_balance
        assert simulator.n_samples_processed == 0
        assert len(simulator.trades) == 0
        assert simulator.position is None


class TestParameterOptimizer:
    """Tests for ParameterOptimizer class."""

    @pytest.fixture
    def optimizer_config(self):
        """Create OptimizationConfig instance."""
        # Import directly to avoid optuna chain import
        from v2.src.optimization.parameter_optimizer import OptimizationConfig
        return OptimizationConfig(
            metric="sortino",
            train_pct=0.8,
            walk_forward_windows=3,
            min_samples_per_window=20
        )

    @pytest.fixture
    def optimizer(self, optimizer_config):
        """Create ParameterOptimizer instance."""
        # Import directly to avoid optuna chain import
        from v2.src.optimization.parameter_optimizer import ParameterOptimizer
        return ParameterOptimizer(config=optimizer_config)

    @pytest.fixture
    def sample_data(self):
        """Create sample market data for optimization."""
        np.random.seed(42)
        n_samples = 500

        prices = 42000 + np.cumsum(np.random.randn(n_samples) * 10)

        return pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='1min'),
            'open': prices,
            'high': prices + 10,
            'low': prices - 10,
            'close': prices + np.random.randn(n_samples),
            'volume': np.random.randint(100, 1000, n_samples)
        })

    def test_initialization(self, optimizer, optimizer_config):
        """ParameterOptimizer should initialize correctly."""
        assert optimizer.config.metric == "sortino"
        assert optimizer.config.walk_forward_windows == 3

    def test_generate_windows(self, optimizer, sample_data):
        """_generate_windows should create valid windows."""
        windows = optimizer._generate_windows(sample_data)

        assert len(windows) > 0

        for window in windows:
            assert window['train_start'] < window['train_end']
            assert window['test_start'] < window['test_end']
            assert window['train_end'] <= window['test_start']

    def test_optimize(self, optimizer, sample_data):
        """optimize should return valid results."""
        param_grid = {
            'param_a': [0.1, 0.2, 0.3],
            'param_b': [10, 20]
        }

        result = optimizer.optimize(
            data=sample_data,
            param_grid=param_grid
        )

        assert result is not None
        assert 'param_a' in result.best_params
        assert 'param_b' in result.best_params
        assert result.best_score is not None
        assert len(result.all_results) == 6  # 3 * 2 combinations

    def test_generate_report(self, optimizer, sample_data):
        """generate_report should create formatted report."""
        param_grid = {
            'param_a': [0.1, 0.2],
        }

        result = optimizer.optimize(
            data=sample_data,
            param_grid=param_grid
        )

        report = optimizer.generate_report(result)

        assert "PARAMETER OPTIMIZATION REPORT" in report
        assert "Best Score" in report


class TestSimulationReport:
    """Tests for SimulationReport class."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample simulation result."""
        from v2.src.backtesting.honest_simulator import (
            SimulationResult, SimulationConfig, TradeRecord
        )

        trades = [
            TradeRecord(
                entry_time=datetime(2024, 1, 1, 10, 0),
                exit_time=datetime(2024, 1, 1, 11, 0),
                direction=1,
                entry_price=42000,
                exit_price=42100,
                size=1000,
                pnl=100,
                fees=0.8,
                exit_reason="signal"
            ),
            TradeRecord(
                entry_time=datetime(2024, 1, 1, 12, 0),
                exit_time=datetime(2024, 1, 1, 13, 0),
                direction=-1,
                entry_price=42200,
                exit_price=42100,
                size=1000,
                pnl=50,
                fees=0.8,
                exit_reason="signal"
            )
        ]

        return SimulationResult(
            trades=trades,
            equity_curve=[10000, 10100, 10150],
            timestamps=[datetime(2024, 1, 1, i) for i in range(3)],
            metrics={
                'total_return_pct': 1.5,
                'sharpe_ratio': 0.8,
                'max_drawdown': 5.0,
                'win_rate': 100.0,
                'total_trades': 2,
                'final_balance': 10150,
                'winning_trades': 2,
                'losing_trades': 0,
                'profit_factor': 999.99,
                'avg_trade_pnl': 75,
                'sortino_ratio': 1.2,
                'total_fees': 1.6
            },
            drift_events=[],
            online_model_stats={},
            config=SimulationConfig()
        )

    def test_generate_report(self, sample_result, tmp_path):
        """generate should create report files."""
        from v2.src.reporting.simulation_report import SimulationReport

        report = SimulationReport(sample_result)
        output_dir = tmp_path / "reports"

        report.generate(str(output_dir))

        # Check files were created
        assert (output_dir / "equity_curve.csv").exists()
        assert (output_dir / "trades.csv").exists()
        assert (output_dir / "metrics.json").exists()
        assert (output_dir / "report.html").exists()

    def test_print_summary(self, sample_result, capsys):
        """print_summary should output to console."""
        from v2.src.reporting.simulation_report import SimulationReport

        report = SimulationReport(sample_result)
        report.print_summary()

        captured = capsys.readouterr()
        assert "MEGA SIMULATION RESULTS" in captured.out
        assert "Total Return" in captured.out


class TestIntegration:
    """Integration tests for the full simulation pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create sample market data for integration testing."""
        np.random.seed(42)
        n_samples = 200

        prices = 42000 + np.cumsum(np.random.randn(n_samples) * 10)

        return pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='1min'),
            'open': prices,
            'high': prices + 10,
            'low': prices - 10,
            'close': prices + np.random.randn(n_samples),
            'volume': np.random.randint(100, 1000, n_samples)
        })

    def test_full_pipeline(self, sample_data, tmp_path):
        """Test the full simulation pipeline."""
        from v2.src.backtesting.honest_simulator import HonestSimulator, SimulationConfig
        from v2.src.reporting.simulation_report import SimulationReport

        # Create simulator
        config = SimulationConfig(
            initial_balance=10000,
            warmup_samples=20,
            lookback_window=10
        )
        simulator = HonestSimulator(config=config)

        # Run simulation
        result = simulator.run(sample_data)

        # Generate report
        report = SimulationReport(result)
        output_dir = tmp_path / "output"
        report.generate(str(output_dir))

        # Verify outputs
        assert result.metrics is not None
        assert (output_dir / "report.html").exists()

    @pytest.mark.skipif(
        not Path(PROJECT_ROOT / "v2/src/online_learning/river_models.py").exists(),
        reason="Online learning module not available"
    )
    def test_with_online_learning(self, sample_data, tmp_path):
        """Test simulation with online learning enabled."""
        try:
            from v2.src.online_learning.river_models import OnlineLearner
            from v2.src.online_learning.drift_detector import MarketDriftDetector
        except ImportError:
            pytest.skip("River library not installed")

        from v2.src.backtesting.honest_simulator import HonestSimulator, SimulationConfig

        # Create components
        config = SimulationConfig(
            initial_balance=10000,
            warmup_samples=20,
            lookback_window=10
        )
        online_learner = OnlineLearner(grace_period=10)
        drift_detector = MarketDriftDetector(
            pnl_delta=0.1,
            cooldown_samples=10
        )

        # Create simulator with online learning
        simulator = HonestSimulator(
            config=config,
            online_learner=online_learner,
            drift_detector=drift_detector
        )

        # Run simulation
        result = simulator.run(sample_data)

        # Verify online learning was used
        assert result.online_model_stats is not None
