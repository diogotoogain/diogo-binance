"""
Tests for real-time 1-second data collection and merge scripts.
"""

import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

from scripts.collect_realtime import OneSecondCandleAggregator
from scripts.merge_1s_data import find_data_files


class TestOneSecondCandleAggregator:
    """Tests for the OneSecondCandleAggregator class."""

    def test_first_trade_starts_new_candle(self):
        """First trade should start a new candle."""
        aggregator = OneSecondCandleAggregator()

        # First trade at timestamp 1000ms (second 1)
        result = aggregator.add_trade(
            trade_time_ms=1000,
            price=50000.0,
            quantity=0.1
        )

        # First trade doesn't complete a candle
        assert result is None
        assert aggregator.current_candle is not None
        assert aggregator.current_candle["open"] == 50000.0
        assert aggregator.current_candle["high"] == 50000.0
        assert aggregator.current_candle["low"] == 50000.0
        assert aggregator.current_candle["close"] == 50000.0
        assert aggregator.current_candle["volume"] == 0.1
        assert aggregator.current_candle["trades_count"] == 1

    def test_same_second_updates_candle(self):
        """Trades in the same second should update the current candle."""
        aggregator = OneSecondCandleAggregator()

        # First trade
        aggregator.add_trade(trade_time_ms=1000, price=50000.0, quantity=0.1)

        # Second trade in same second (1500ms is still second 1)
        result = aggregator.add_trade(trade_time_ms=1500, price=50100.0, quantity=0.2)

        assert result is None
        assert aggregator.current_candle["high"] == 50100.0
        assert aggregator.current_candle["close"] == 50100.0
        assert aggregator.current_candle["volume"] == pytest.approx(0.3)
        assert aggregator.current_candle["trades_count"] == 2

    def test_new_second_completes_previous_candle(self):
        """Trade in new second should complete previous candle."""
        aggregator = OneSecondCandleAggregator()

        # Trades in first second (second 1)
        aggregator.add_trade(trade_time_ms=1000, price=50000.0, quantity=0.1)
        aggregator.add_trade(trade_time_ms=1500, price=50100.0, quantity=0.2)

        # Trade in new second (second 2)
        completed = aggregator.add_trade(trade_time_ms=2000, price=50200.0, quantity=0.15)

        assert completed is not None
        assert completed["open_time"] == 1000
        assert completed["open"] == 50000.0
        assert completed["high"] == 50100.0
        assert completed["low"] == 50000.0
        assert completed["close"] == 50100.0
        assert completed["volume"] == pytest.approx(0.3)
        assert completed["trades_count"] == 2

    def test_low_price_tracking(self):
        """Low price should be tracked correctly."""
        aggregator = OneSecondCandleAggregator()

        aggregator.add_trade(trade_time_ms=1000, price=50000.0, quantity=0.1)
        aggregator.add_trade(trade_time_ms=1200, price=49900.0, quantity=0.1)
        aggregator.add_trade(trade_time_ms=1500, price=50050.0, quantity=0.1)

        assert aggregator.current_candle["low"] == 49900.0
        assert aggregator.current_candle["high"] == 50050.0

    def test_flush_current_candle(self):
        """flush_current should return incomplete candle."""
        aggregator = OneSecondCandleAggregator()

        aggregator.add_trade(trade_time_ms=1000, price=50000.0, quantity=0.1)

        flushed = aggregator.flush_current()

        assert flushed is not None
        assert flushed["open"] == 50000.0
        assert aggregator.current_candle is None
        assert len(aggregator.completed_candles) == 1

    def test_get_and_clear_completed(self):
        """get_and_clear_completed should return and clear buffer."""
        aggregator = OneSecondCandleAggregator()

        # Create multiple completed candles
        aggregator.add_trade(trade_time_ms=1000, price=50000.0, quantity=0.1)
        aggregator.add_trade(trade_time_ms=2000, price=50100.0, quantity=0.1)
        aggregator.add_trade(trade_time_ms=3000, price=50200.0, quantity=0.1)

        candles = aggregator.get_and_clear_completed()

        assert len(candles) == 2  # 2 completed candles
        assert len(aggregator.completed_candles) == 0  # Buffer cleared

    def test_quote_volume_calculation(self):
        """Quote volume should be price * quantity sum."""
        aggregator = OneSecondCandleAggregator()

        aggregator.add_trade(trade_time_ms=1000, price=50000.0, quantity=0.1)  # 5000
        aggregator.add_trade(trade_time_ms=1500, price=51000.0, quantity=0.2)  # 10200

        expected_quote_vol = 50000.0 * 0.1 + 51000.0 * 0.2
        assert aggregator.current_candle["quote_volume"] == pytest.approx(expected_quote_vol)


class TestFindDataFiles:
    """Tests for find_data_files function."""

    def test_find_files_with_date_filter(self):
        """Should filter files by date range."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Create test files
            (data_dir / "BTCUSDT_1s_2024-01-10.parquet").touch()
            (data_dir / "BTCUSDT_1s_2024-01-15.parquet").touch()
            (data_dir / "BTCUSDT_1s_2024-01-20.parquet").touch()

            # Filter by date range
            files = find_data_files(
                data_dir=data_dir,
                symbol="BTCUSDT",
                start_date="2024-01-12",
                end_date="2024-01-18"
            )

            assert len(files) == 1
            assert files[0].name == "BTCUSDT_1s_2024-01-15.parquet"

    def test_find_files_sorted_by_date(self):
        """Should return files sorted by date."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)

            # Create files in random order
            (data_dir / "BTCUSDT_1s_2024-01-20.parquet").touch()
            (data_dir / "BTCUSDT_1s_2024-01-10.parquet").touch()
            (data_dir / "BTCUSDT_1s_2024-01-15.parquet").touch()

            files = find_data_files(data_dir=data_dir, symbol="BTCUSDT")

            assert len(files) == 3
            assert files[0].name == "BTCUSDT_1s_2024-01-10.parquet"
            assert files[1].name == "BTCUSDT_1s_2024-01-15.parquet"
            assert files[2].name == "BTCUSDT_1s_2024-01-20.parquet"

    def test_find_files_empty_directory(self):
        """Should return empty list for empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = find_data_files(data_dir=Path(tmpdir), symbol="BTCUSDT")
            assert len(files) == 0

    def test_find_files_wrong_symbol(self):
        """Should not find files for different symbol."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "BTCUSDT_1s_2024-01-15.parquet").touch()

            files = find_data_files(data_dir=data_dir, symbol="ETHUSDT")
            assert len(files) == 0
