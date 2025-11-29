"""
Tests for backtest module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def config():
    """Sample configuration for testing."""
    return {
        "backtest": {
            "initial_capital": 10000,
            "slippage": {
                "type": "volume_based",
                "base_bps": 1.0,
                "volume_impact": 0.5,
                "spread_factor": 1.0,
            },
            "fees": {
                "taker": 0.0004,
                "maker": 0.0002,
                "funding_rate": 0.0001,
            },
            "walk_forward": {
                "enabled": True,
                "train_days": 30,
                "test_days": 7,
                "step_days": 7,
            },
        },
        "risk": {
            "max_risk_per_trade_pct": 0.5,
            "kill_switch": {"enabled": True},
        },
    }


@pytest.fixture
def sample_data():
    """Sample OHLCV data for testing."""
    n = 200
    np.random.seed(42)
    
    # Generate price data
    returns = np.random.randn(n) * 0.01
    prices = 50000 * np.exp(np.cumsum(returns))
    
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1D")
    
    data = pd.DataFrame({
        "open": prices * (1 + np.random.randn(n) * 0.001),
        "high": prices * (1 + np.abs(np.random.randn(n) * 0.005)),
        "low": prices * (1 - np.abs(np.random.randn(n) * 0.005)),
        "close": prices,
        "volume": np.random.randint(100, 1000, n),
    }, index=dates)
    
    return data


@pytest.fixture
def sample_features(sample_data):
    """Sample features for testing."""
    n = len(sample_data)
    np.random.seed(42)
    
    features = sample_data.copy()
    features["ofi"] = np.random.randn(n) * 0.3
    features["tfi"] = np.random.randn(n) * 0.3
    features["rsi"] = 50 + np.random.randn(n) * 15
    features["atr"] = sample_data["close"] * 0.02
    features["adx"] = 25 + np.random.randn(n) * 10
    
    return features


class TestTradeSimulator:
    """Tests for TradeSimulator class."""

    def test_simulator_init(self, config):
        """TradeSimulator should initialize from config."""
        from v2.src.backtest.simulator import TradeSimulator

        sim = TradeSimulator(config)

        assert sim.slippage_type == "volume_based"
        assert sim.base_bps == 1.0
        assert sim.taker_fee == 0.0004

    def test_apply_slippage_fixed(self, config):
        """Slippage should be applied correctly for fixed type."""
        from v2.src.backtest.simulator import TradeSimulator

        config["backtest"]["slippage"]["type"] = "fixed"
        sim = TradeSimulator(config)

        price = 50000
        slipped_buy = sim.apply_slippage(price, "BUY")
        slipped_sell = sim.apply_slippage(price, "SELL")

        # Buy should be higher, sell should be lower
        assert slipped_buy > price
        assert slipped_sell < price

    def test_apply_slippage_volume_based(self, config):
        """Slippage should increase with volume."""
        from v2.src.backtest.simulator import TradeSimulator

        sim = TradeSimulator(config)

        price = 50000
        
        # Small order
        small_slip = sim.apply_slippage(price, "BUY", volume=10, avg_volume=1000)
        
        # Large order
        large_slip = sim.apply_slippage(price, "BUY", volume=500, avg_volume=1000)

        # Larger orders should have more slippage
        assert large_slip > small_slip

    def test_calculate_fees(self, config):
        """Fees should be calculated correctly."""
        from v2.src.backtest.simulator import TradeSimulator

        sim = TradeSimulator(config)

        notional = 10000
        taker_fee = sim.calculate_fees(notional, is_maker=False)
        maker_fee = sim.calculate_fees(notional, is_maker=True)

        assert taker_fee == 10000 * 0.0004
        assert maker_fee == 10000 * 0.0002

    def test_execute_entry(self, config):
        """Trade entry should create position correctly."""
        from v2.src.backtest.simulator import TradeSimulator

        sim = TradeSimulator(config)

        position = sim.execute_entry(
            price=50000,
            direction=1,
            size=1000,
            timestamp=pd.Timestamp.now(),
            step=0,
            atr=1000,
            sl_mult=1.0,
            tp_mult=2.0,
        )

        assert position.direction == 1
        assert position.size == 1000
        assert position.stop_loss > 0
        assert position.take_profit > 0

    def test_execute_exit(self, config):
        """Trade exit should close position correctly."""
        from v2.src.backtest.simulator import TradeSimulator

        sim = TradeSimulator(config)

        # Create position
        position = sim.execute_entry(
            price=50000,
            direction=1,
            size=1000,
            timestamp=pd.Timestamp.now(),
            step=0,
            atr=1000,
            sl_mult=1.0,
            tp_mult=2.0,
        )

        # Close position with profit
        trade = sim.execute_exit(
            position=position,
            price=51000,  # Higher than entry
            timestamp=pd.Timestamp.now(),
            step=10,
            reason="signal",
        )

        assert trade.pnl > 0  # Should be profitable
        assert trade.exit_reason == "signal"

    def test_check_stop_loss(self, config):
        """Stop loss should be detected correctly."""
        from v2.src.backtest.simulator import TradeSimulator, Position

        sim = TradeSimulator(config)

        # Long position
        position = Position(
            direction=1,
            entry_price=50000,
            size=1000,
            entry_time=pd.Timestamp.now(),
            stop_loss=49000,
            take_profit=52000,
        )

        # Price above stop - no trigger
        assert sim.check_stop_loss(position, 49500) is False
        
        # Price at or below stop - trigger
        assert sim.check_stop_loss(position, 49000) is True
        assert sim.check_stop_loss(position, 48000) is True

    def test_check_take_profit(self, config):
        """Take profit should be detected correctly."""
        from v2.src.backtest.simulator import TradeSimulator, Position

        sim = TradeSimulator(config)

        # Long position
        position = Position(
            direction=1,
            entry_price=50000,
            size=1000,
            entry_time=pd.Timestamp.now(),
            stop_loss=49000,
            take_profit=52000,
        )

        # Price below target - no trigger
        assert sim.check_take_profit(position, 51000) is False
        
        # Price at or above target - trigger
        assert sim.check_take_profit(position, 52000) is True
        assert sim.check_take_profit(position, 53000) is True


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics class."""

    def test_metrics_calculate_all(self, sample_data):
        """Metrics should calculate all measures."""
        from v2.src.backtest.metrics import PerformanceMetrics

        metrics = PerformanceMetrics()

        equity = pd.Series([10000, 10100, 10050, 10200, 10150, 10300])
        trades = pd.DataFrame({
            "pnl": [100, -50, 150, -50, 150],
            "entry_time": pd.date_range("2024-01-01", periods=5),
            "exit_time": pd.date_range("2024-01-02", periods=5),
        })

        results = metrics.calculate_all(equity, trades)

        assert "total_return_pct" in results
        assert "sharpe_ratio" in results
        assert "max_drawdown" in results
        assert "win_rate" in results
        assert "profit_factor" in results

    def test_metrics_empty_data(self):
        """Metrics should handle empty data."""
        from v2.src.backtest.metrics import PerformanceMetrics

        metrics = PerformanceMetrics()

        results = metrics.calculate_all(pd.Series([]), pd.DataFrame())

        assert results["total_return_pct"] == 0.0
        assert results["sharpe_ratio"] == 0.0

    def test_sharpe_ratio_calculation(self):
        """Sharpe ratio should be calculated correctly."""
        from v2.src.backtest.metrics import PerformanceMetrics

        metrics = PerformanceMetrics()

        # Create returns with known Sharpe
        equity = pd.Series([10000, 10100, 10200, 10300, 10400])
        trades = pd.DataFrame()

        results = metrics.calculate_all(equity, trades)

        # Should be positive for consistently positive returns
        assert results["sharpe_ratio"] > 0

    def test_max_drawdown_calculation(self):
        """Max drawdown should be calculated correctly."""
        from v2.src.backtest.metrics import PerformanceMetrics

        metrics = PerformanceMetrics()

        # Equity with 10% drawdown
        equity = pd.Series([100, 110, 99, 105, 95, 100])
        trades = pd.DataFrame()

        results = metrics.calculate_all(equity, trades)

        # Max drawdown should be ~13.6% (from 110 to 95)
        assert results["max_drawdown"] > 10


class TestBacktestEngine:
    """Tests for BacktestEngine class."""

    def test_engine_init(self, config):
        """BacktestEngine should initialize correctly."""
        from v2.src.backtest.engine import BacktestEngine

        engine = BacktestEngine(config)

        assert engine.initial_capital == 10000
        assert engine.simulator is not None

    def test_engine_run_simple_strategy(self, config, sample_data, sample_features):
        """Engine should run a simple strategy."""
        from v2.src.backtest.engine import BacktestEngine, Signal

        engine = BacktestEngine(config)

        # Simple strategy that always holds
        def strategy(features, context):
            return Signal(direction=0)  # Hold

        results = engine.run(sample_data, strategy, sample_features)

        assert "total_return_pct" in results
        assert "sharpe_ratio" in results
        assert results["total_trades"] == 0  # No trades for hold strategy

    def test_engine_run_long_strategy(self, config, sample_data, sample_features):
        """Engine should execute long trades."""
        from v2.src.backtest.engine import BacktestEngine, Signal

        engine = BacktestEngine(config)

        # Strategy that goes long when RSI < 40
        def strategy(features, context):
            rsi = features.get("rsi", 50)
            if rsi < 40:
                return Signal(direction=1, confidence=0.8)
            return Signal(direction=0)

        results = engine.run(sample_data, strategy, sample_features)

        assert "total_return_pct" in results
        assert "trades" in results

    def test_engine_run_multiple_strategies(self, config, sample_data, sample_features):
        """Engine should run multiple strategies."""
        from v2.src.backtest.engine import BacktestEngine, Signal

        engine = BacktestEngine(config)

        strategies = {
            "hold": lambda f, c: Signal(direction=0),
            "always_long": lambda f, c: Signal(direction=1),
        }

        results = engine.run_multiple(sample_data, strategies, sample_features)

        assert "hold" in results
        assert "always_long" in results


class TestWalkForwardOptimizer:
    """Tests for WalkForwardOptimizer class."""

    def test_walk_forward_windows(self, config, sample_data):
        """Walk-forward should create windows correctly."""
        from v2.src.backtest.walk_forward import WalkForwardOptimizer
        from v2.src.optimization.optuna_optimizer import OptunaOptimizer
        from v2.src.backtest.engine import BacktestEngine

        optimizer = OptunaOptimizer(config)
        engine = BacktestEngine(config)

        wfo = WalkForwardOptimizer(config, optimizer, engine)

        # Generate windows
        windows = wfo._generate_windows(sample_data)

        # Should have multiple windows
        assert len(windows) > 0

        # Each window should have correct structure
        for window in windows:
            assert window.train_start_idx < window.train_end_idx
            assert window.test_start_idx == window.train_end_idx
            assert window.test_end_idx > window.test_start_idx


class TestBacktestReport:
    """Tests for BacktestReport class."""

    def test_report_generate_summary(self):
        """Report should generate text summary."""
        from v2.src.backtest.report import BacktestReport

        results = {
            "initial_capital": 10000,
            "final_equity": 11000,
            "total_return_pct": 10.0,
            "sharpe_ratio": 1.5,
            "sortino_ratio": 2.0,
            "max_drawdown": 5.0,
            "win_rate": 60.0,
            "profit_factor": 1.8,
            "total_trades": 20,
            "volatility": 15.0,
            "var_95": 2.0,
            "cvar_95": 3.0,
        }

        trades = pd.DataFrame({
            "pnl": [100, -50, 150],
            "entry_price": [50000, 50100, 50200],
            "exit_price": [50100, 50050, 50350],
            "direction": [1, 1, 1],
        })

        equity = pd.Series([10000, 10100, 10050, 10200])

        report = BacktestReport(results, trades, equity)
        summary = report.generate_summary()

        assert "BACKTEST REPORT" in summary
        assert "10.00%" in summary  # Total return
        assert "1.50" in summary  # Sharpe ratio

    def test_report_save_json(self, tmp_path):
        """Report should save JSON file."""
        from v2.src.backtest.report import BacktestReport

        results = {
            "total_return_pct": 10.0,
            "sharpe_ratio": 1.5,
        }
        trades = pd.DataFrame()
        equity = pd.Series([10000, 10100])

        report = BacktestReport(results, trades, equity)
        report.save_json(tmp_path / "results.json")

        assert (tmp_path / "results.json").exists()

    def test_report_from_backtest_results(self):
        """Report should create from backtest results."""
        from v2.src.backtest.report import BacktestReport

        results = {
            "total_return_pct": 10.0,
            "sharpe_ratio": 1.5,
            "equity_curve": pd.Series([10000, 10100]),
            "trades": pd.DataFrame(),
        }

        report = BacktestReport.from_backtest_results(results)

        assert report.results is not None
