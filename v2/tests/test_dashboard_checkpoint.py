"""
Tests for the Checkpoint and Dashboard functionality.
"""

import sys
import tempfile
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest

# Add project root to path for relative imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestCheckpointManager:
    """Tests for CheckpointManager class."""

    @pytest.fixture
    def checkpoint_manager(self, tmp_path):
        """Create CheckpointManager instance with temp directory."""
        from v2.src.checkpoint.checkpoint_manager import CheckpointManager
        return CheckpointManager(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            save_interval_minutes=1,
            save_interval_candles=100,
            keep_last_n=3,
            compress=True
        )

    @pytest.fixture
    def sample_checkpoint(self):
        """Create a sample checkpoint."""
        from v2.src.checkpoint.checkpoint_manager import SimulationCheckpoint
        return SimulationCheckpoint(
            checkpoint_id="test123",
            created_at=datetime.now(),
            config_hash="abc123",
            current_index=5000,
            total_candles=10000,
            progress_pct=50.0,
            balance=12500.0,
            position=None,
            trades=[],
            equity_curve=[10000.0, 10500.0, 11000.0, 12000.0, 12500.0],
            timestamps=[]
        )

    def test_initialization(self, checkpoint_manager):
        """CheckpointManager should initialize correctly."""
        assert checkpoint_manager is not None
        assert checkpoint_manager.checkpoint_dir.exists()
        assert checkpoint_manager.keep_last_n == 3

    def test_save_and_load(self, checkpoint_manager, sample_checkpoint):
        """save and load should work correctly."""
        # Save checkpoint
        filepath = checkpoint_manager.save(sample_checkpoint)
        assert Path(filepath).exists()

        # Load checkpoint
        loaded = checkpoint_manager.load(filepath)
        assert loaded is not None
        assert loaded.checkpoint_id == sample_checkpoint.checkpoint_id
        assert loaded.balance == sample_checkpoint.balance
        assert loaded.progress_pct == sample_checkpoint.progress_pct

    def test_load_latest(self, checkpoint_manager, sample_checkpoint):
        """load without filepath should load latest checkpoint."""
        # Save checkpoint
        checkpoint_manager.save(sample_checkpoint)

        # Load latest
        loaded = checkpoint_manager.load()
        assert loaded is not None
        assert loaded.checkpoint_id == sample_checkpoint.checkpoint_id

    def test_should_save_by_candles(self, checkpoint_manager):
        """should_save should return True after enough candles."""
        # Initially should not save (0 candles)
        assert not checkpoint_manager.should_save(50)
        
        # After 100 candles should save
        assert checkpoint_manager.should_save(100)

    def test_has_checkpoint(self, checkpoint_manager, sample_checkpoint):
        """has_checkpoint should detect existing checkpoints."""
        assert not checkpoint_manager.has_checkpoint()
        
        checkpoint_manager.save(sample_checkpoint)
        
        assert checkpoint_manager.has_checkpoint()

    def test_clear_checkpoints(self, checkpoint_manager, sample_checkpoint):
        """clear_checkpoints should remove all checkpoints."""
        checkpoint_manager.save(sample_checkpoint)
        assert checkpoint_manager.has_checkpoint()
        
        checkpoint_manager.clear_checkpoints()
        assert not checkpoint_manager.has_checkpoint()

    def test_cleanup_old_checkpoints(self, checkpoint_manager, sample_checkpoint):
        """Old checkpoints should be cleaned up."""
        # Save more checkpoints than keep_last_n
        for i in range(5):
            sample_checkpoint.checkpoint_id = f"test{i}"
            checkpoint_manager.save(sample_checkpoint)
            # Reset save time to allow next save
            checkpoint_manager._last_save_candle = 0
            checkpoint_manager._last_save_time = 0

        # Should only have keep_last_n checkpoints (+ manifest)
        checkpoint_files = list(checkpoint_manager.checkpoint_dir.glob("simulation_state_*.pkl.gz"))
        # Filter out 'latest' file
        regular_files = [f for f in checkpoint_files if 'latest' not in f.name]
        
        # Should have at most keep_last_n + 1 (latest symlink)
        assert len(regular_files) <= checkpoint_manager.keep_last_n

    def test_compute_config_hash(self, checkpoint_manager):
        """compute_config_hash should produce consistent hashes."""
        config1 = {"key": "value", "nested": {"a": 1}}
        config2 = {"key": "value", "nested": {"a": 1}}
        config3 = {"key": "different"}

        hash1 = checkpoint_manager.compute_config_hash(config1)
        hash2 = checkpoint_manager.compute_config_hash(config2)
        hash3 = checkpoint_manager.compute_config_hash(config3)

        assert hash1 == hash2
        assert hash1 != hash3

    def test_get_latest_checkpoint_info(self, checkpoint_manager, sample_checkpoint):
        """get_latest_checkpoint_info should return correct info."""
        # No checkpoint
        info = checkpoint_manager.get_latest_checkpoint_info()
        assert info is None

        # Save checkpoint
        checkpoint_manager.save(sample_checkpoint)

        info = checkpoint_manager.get_latest_checkpoint_info()
        assert info is not None
        assert info['id'] == sample_checkpoint.checkpoint_id
        assert info['balance'] == sample_checkpoint.balance


class TestGracefulShutdown:
    """Tests for GracefulShutdown class."""

    @pytest.fixture
    def graceful_shutdown(self):
        """Create GracefulShutdown instance."""
        from v2.src.checkpoint.graceful_shutdown import GracefulShutdown
        return GracefulShutdown()

    def test_initialization(self, graceful_shutdown):
        """GracefulShutdown should initialize correctly."""
        assert graceful_shutdown is not None
        assert not graceful_shutdown.shutdown_requested

    def test_update_state(self, graceful_shutdown):
        """update_state should update internal state."""
        graceful_shutdown.update_state(
            current_index=1000,
            total_candles=5000,
            balance=15000.0,
            trades_count=50
        )

        assert graceful_shutdown.current_index == 1000
        assert graceful_shutdown.total_candles == 5000
        assert graceful_shutdown.balance == 15000.0
        assert graceful_shutdown.trades_count == 50

    def test_format_duration(self):
        """format_duration should format correctly."""
        from v2.src.checkpoint.graceful_shutdown import format_duration

        assert format_duration(30) == "30s"
        assert format_duration(90) == "1min 30s"
        assert format_duration(3700) == "1h 1min"


class TestDataService:
    """Tests for DataService class."""

    @pytest.fixture
    def data_service(self, tmp_path):
        """Create DataService instance with temp directory."""
        from v2.dashboard.data_service import DataService
        return DataService(results_dir=str(tmp_path))

    @pytest.fixture
    def sample_trades_df(self):
        """Create sample trades DataFrame."""
        return pd.DataFrame({
            'trade_id': [1, 2, 3, 4, 5],
            'entry_time': pd.date_range('2024-01-01', periods=5, freq='D'),
            'exit_time': pd.date_range('2024-01-01 01:00', periods=5, freq='D'),
            'direction': [1, -1, 1, 1, -1],
            'entry_price': [42000, 42100, 42200, 42300, 42400],
            'exit_price': [42100, 42000, 42150, 42400, 42350],
            'size': [1000, 1000, 1000, 1000, 1000],
            'pnl': [100, 100, -50, 100, 50],
            'fees': [0.8, 0.8, 0.8, 0.8, 0.8],
            'exit_reason': ['signal', 'signal', 'sl', 'tp', 'signal']
        })

    def test_initialization(self, data_service):
        """DataService should initialize correctly."""
        assert data_service is not None

    def test_filter_trades_by_direction(self, data_service, sample_trades_df):
        """filter_trades should filter by direction."""
        data_service._trades_df = sample_trades_df

        # Filter longs
        longs = data_service.filter_trades(direction=1)
        assert len(longs) == 3
        assert all(longs['direction'] == 1)

        # Filter shorts
        shorts = data_service.filter_trades(direction=-1)
        assert len(shorts) == 2
        assert all(shorts['direction'] == -1)

    def test_filter_trades_by_result(self, data_service, sample_trades_df):
        """filter_trades should filter by result."""
        data_service._trades_df = sample_trades_df

        # Winners
        winners = data_service.filter_trades(result_filter='winners')
        assert len(winners) == 4
        assert all(winners['pnl'] > 0)

        # Losers
        losers = data_service.filter_trades(result_filter='losers')
        assert len(losers) == 1
        assert all(losers['pnl'] < 0)

    def test_get_trades_paginated(self, data_service, sample_trades_df):
        """get_trades_paginated should return correct pages."""
        data_service._trades_df = sample_trades_df

        # First page
        df, total = data_service.get_trades_paginated(page=1, per_page=2)
        assert len(df) == 2
        assert total == 5

        # Second page
        df, total = data_service.get_trades_paginated(page=2, per_page=2)
        assert len(df) == 2

        # Third page
        df, total = data_service.get_trades_paginated(page=3, per_page=2)
        assert len(df) == 1

    def test_get_summary_stats(self, data_service, sample_trades_df):
        """get_summary_stats should calculate correctly."""
        data_service._trades_df = sample_trades_df
        data_service._metrics = {}

        stats = data_service.get_summary_stats()

        assert stats['total_trades'] == 5
        assert stats['winning_trades'] == 4
        assert stats['losing_trades'] == 1
        assert stats['total_pnl'] == 300
        assert stats['long_trades'] == 3
        assert stats['short_trades'] == 2


class TestBacktestAnalytics:
    """Tests for BacktestAnalytics class."""

    @pytest.fixture
    def analytics(self, tmp_path):
        """Create BacktestAnalytics instance."""
        from v2.dashboard.data_service import DataService
        from v2.dashboard.analytics import BacktestAnalytics

        data_service = DataService(results_dir=str(tmp_path))
        return BacktestAnalytics(data_service)

    @pytest.fixture
    def analytics_with_data(self, tmp_path):
        """Create BacktestAnalytics with sample data."""
        from v2.dashboard.data_service import DataService
        from v2.dashboard.analytics import BacktestAnalytics

        data_service = DataService(results_dir=str(tmp_path))

        # Add sample trades
        data_service._trades_df = pd.DataFrame({
            'trade_id': range(1, 51),
            'entry_time': pd.date_range('2024-01-01', periods=50, freq='h'),
            'exit_time': pd.date_range('2024-01-01 00:30', periods=50, freq='h'),
            'direction': np.random.choice([1, -1], 50),
            'entry_price': 42000 + np.random.randn(50) * 100,
            'exit_price': 42000 + np.random.randn(50) * 100,
            'size': [1000] * 50,
            'pnl': np.random.randn(50) * 50,
            'fees': [0.8] * 50,
            'exit_reason': ['signal'] * 50
        })

        # Add equity curve
        equity = 10000 + np.cumsum(np.random.randn(100) * 10)
        data_service._equity_df = pd.DataFrame({
            'equity': equity,
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='min')
        })

        data_service._metrics = {
            'initial_balance': 10000,
            'final_balance': equity[-1]
        }

        return BacktestAnalytics(data_service)

    def test_analyze_by_hour(self, analytics_with_data):
        """analyze_by_hour should return data for all 24 hours."""
        result = analytics_with_data.analyze_by_hour()
        assert len(result) == 24
        assert all('hour' in r for r in result)
        assert all('trades' in r for r in result)

    def test_analyze_by_day_of_week(self, analytics_with_data):
        """analyze_by_day_of_week should return data for all 7 days."""
        result = analytics_with_data.analyze_by_day_of_week()
        assert len(result) == 7
        assert all('day_name' in r for r in result)

    def test_calculate_var(self, analytics_with_data):
        """calculate_var should return VaR values."""
        result = analytics_with_data.calculate_var()
        assert 'var_95' in result
        assert 'var_99' in result
        assert 'cvar_95' in result
        assert 'cvar_99' in result

    def test_analyze_sequences(self, analytics_with_data):
        """analyze_sequences should return streak information."""
        result = analytics_with_data.analyze_sequences()
        assert 'max_win_streak' in result
        assert 'max_loss_streak' in result
        assert 'current_streak' in result
        assert 'current_streak_type' in result

    def test_monte_carlo_simulation(self, analytics_with_data):
        """monte_carlo_simulation should return distribution."""
        result = analytics_with_data.monte_carlo_simulation(n_simulations=100)
        assert 'simulations' in result
        assert result['simulations'] == 100
        assert 'percentiles' in result
        assert '50th' in result['percentiles']


class TestDashboardServer:
    """Tests for BacktestDashboard server."""

    @pytest.fixture
    def dashboard(self, tmp_path):
        """Create BacktestDashboard instance."""
        from v2.dashboard.server import BacktestDashboard
        from v2.dashboard.data_service import DataService
        from v2.dashboard.analytics import BacktestAnalytics

        data_service = DataService(results_dir=str(tmp_path))
        analytics = BacktestAnalytics(data_service)

        return BacktestDashboard(
            data_service=data_service,
            analytics=analytics,
            port=8081
        )

    def test_initialization(self, dashboard):
        """BacktestDashboard should initialize correctly."""
        assert dashboard is not None
        assert dashboard.app is not None
        assert dashboard.port == 8081

    def test_routes_exist(self, dashboard):
        """All expected routes should be registered."""
        routes = [route.path for route in dashboard.app.routes]

        # Page routes
        assert "/" in routes
        assert "/trades" in routes
        assert "/performance" in routes
        assert "/temporal" in routes
        assert "/risk" in routes
        assert "/model" in routes
        assert "/compare" in routes

        # API routes
        assert "/api/summary" in routes
        assert "/api/metrics" in routes
        assert "/api/equity" in routes
        assert "/api/trades" in routes


class TestIntegration:
    """Integration tests for the full checkpoint and dashboard system."""

    @pytest.fixture
    def sample_data(self):
        """Create sample market data."""
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

    def test_checkpoint_create_and_restore(self, sample_data, tmp_path):
        """Test creating checkpoint from simulator and restoring."""
        from v2.src.backtesting.honest_simulator import HonestSimulator, SimulationConfig
        from v2.src.checkpoint.checkpoint_manager import CheckpointManager

        # Create simulator and run partial simulation
        config = SimulationConfig(
            initial_balance=10000,
            warmup_samples=20,
            lookback_window=10
        )
        simulator = HonestSimulator(config=config)
        
        # Run simulation
        result = simulator.run(sample_data)

        # Create checkpoint
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(tmp_path / "checkpoints"),
            compress=True
        )

        sim_config = {'test': 'config'}
        checkpoint = checkpoint_manager.create_checkpoint(
            simulator=simulator,
            current_index=100,
            total_candles=200,
            config=sim_config
        )

        # Save checkpoint
        filepath = checkpoint_manager.save(checkpoint)
        assert Path(filepath).exists()

        # Create new simulator and restore
        new_simulator = HonestSimulator(config=config)
        loaded_checkpoint = checkpoint_manager.load(filepath)
        
        checkpoint_manager.restore_simulator(loaded_checkpoint, new_simulator)

        # Verify state was restored
        assert new_simulator.balance == simulator.balance
        assert len(new_simulator.trades) == len(simulator.trades)
        assert len(new_simulator.equity_curve) == len(simulator.equity_curve)

    def test_data_service_from_simulation_result(self, sample_data, tmp_path):
        """Test loading DataService from SimulationResult."""
        from v2.src.backtesting.honest_simulator import HonestSimulator, SimulationConfig
        from v2.dashboard.data_service import DataService
        from v2.dashboard.analytics import BacktestAnalytics

        # Run simulation
        config = SimulationConfig(
            initial_balance=10000,
            warmup_samples=20,
            lookback_window=10
        )
        simulator = HonestSimulator(config=config)
        result = simulator.run(sample_data)

        # Load into DataService
        data_service = DataService(results_dir=str(tmp_path))
        data_service.load_from_simulation_result(result)

        # Verify data was loaded
        assert len(data_service.equity) > 0
        assert data_service.metrics is not None

        # Create analytics
        analytics = BacktestAnalytics(data_service)
        
        # Verify analytics work
        summary = data_service.get_summary_stats()
        assert 'equity_points' in summary
