#!/usr/bin/env python3
"""
Real-time 1-second candle data collector from Binance WebSocket.

Connects to Binance Futures WebSocket trade stream and aggregates trades into
1-second OHLCV candles. Saves data to Parquet files separated by day.

Usage:
    python collect_realtime.py --duration 20700  # Run for 5h45min
    python collect_realtime.py --symbol BTCUSDT --duration 3600
"""

import argparse
import asyncio
import json
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    sys.exit(1)

from src.utils.logger import setup_logging, get_logger


class OneSecondCandleAggregator:
    """Aggregates trades into 1-second OHLCV candles."""

    def __init__(self):
        """Initialize aggregator."""
        self.current_candle: Optional[dict] = None
        self.current_second: Optional[int] = None
        self.completed_candles: list = []

    def add_trade(self, trade_time_ms: int, price: float, quantity: float) -> Optional[dict]:
        """
        Add a trade and return completed candle if second boundary crossed.

        Args:
            trade_time_ms: Trade timestamp in milliseconds
            price: Trade price
            quantity: Trade quantity

        Returns:
            Completed candle dict if second boundary crossed, None otherwise
        """
        trade_second = trade_time_ms // 1000

        # First trade or new second
        if self.current_second is None or trade_second > self.current_second:
            completed = None

            # Complete previous candle if exists
            if self.current_candle is not None:
                completed = self.current_candle.copy()
                self.completed_candles.append(completed)

            # Start new candle
            self.current_second = trade_second
            self.current_candle = {
                "open_time": trade_second * 1000,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": quantity,
                "trades_count": 1,
                "quote_volume": price * quantity,
            }

            return completed

        # Same second - update candle
        self.current_candle["high"] = max(self.current_candle["high"], price)
        self.current_candle["low"] = min(self.current_candle["low"], price)
        self.current_candle["close"] = price
        self.current_candle["volume"] += quantity
        self.current_candle["trades_count"] += 1
        self.current_candle["quote_volume"] += price * quantity

        return None

    def flush_current(self) -> Optional[dict]:
        """Flush current incomplete candle."""
        if self.current_candle is not None:
            completed = self.current_candle.copy()
            self.completed_candles.append(completed)
            self.current_candle = None
            self.current_second = None
            return completed
        return None

    def get_and_clear_completed(self) -> list:
        """Get all completed candles and clear the buffer."""
        candles = self.completed_candles.copy()
        self.completed_candles = []
        return candles


class RealtimeDataCollector:
    """Collects real-time trade data from Binance and saves as 1-second candles."""

    # Binance Futures WebSocket URL
    WS_URL_TEMPLATE = "wss://fstream.binance.com/ws/{symbol}@trade"

    def __init__(
        self,
        symbol: str = "btcusdt",
        data_dir: str = "v2/data/raw/1s",
        flush_interval: int = 60,
    ):
        """
        Initialize collector.

        Args:
            symbol: Trading pair symbol (lowercase)
            data_dir: Directory to save Parquet files
            flush_interval: Seconds between disk flushes
        """
        self.symbol = symbol.lower()
        self.symbol_upper = symbol.upper()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.flush_interval = flush_interval

        self.aggregator = OneSecondCandleAggregator()
        self.logger = get_logger(__name__)
        self.running = False
        self.shutdown_event = asyncio.Event()

        # Statistics
        self.trades_received = 0
        self.candles_saved = 0
        self.reconnect_count = 0

        # Current day buffer
        self.current_day: Optional[str] = None
        self.day_candles: list = []

    def _get_ws_url(self) -> str:
        """Get WebSocket URL for the symbol."""
        return self.WS_URL_TEMPLATE.format(symbol=self.symbol)

    def _get_parquet_path(self, date_str: str) -> Path:
        """Get Parquet file path for a given date."""
        return self.data_dir / f"{self.symbol_upper}_1s_{date_str}.parquet"

    def _save_candles(self, candles: list, date_str: str) -> None:
        """Save candles to Parquet file, appending if file exists."""
        if not candles:
            return

        filepath = self._get_parquet_path(date_str)
        df_new = pd.DataFrame(candles)

        # Load existing data if file exists
        if filepath.exists():
            try:
                df_existing = pd.read_parquet(filepath)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                # Remove duplicates based on open_time
                df_combined = df_combined.drop_duplicates(subset=["open_time"], keep="last")
                df_combined = df_combined.sort_values("open_time").reset_index(drop=True)
            except Exception as e:
                self.logger.warning(f"Error reading existing file, overwriting: {e}")
                df_combined = df_new
        else:
            df_combined = df_new

        # Save to Parquet
        df_combined.to_parquet(filepath, index=False, compression="snappy")
        self.candles_saved += len(candles)
        self.logger.info(
            f"💾 Saved {len(candles)} candles to {filepath.name} "
            f"(total in file: {len(df_combined)})"
        )

    def _flush_to_disk(self) -> None:
        """Flush accumulated candles to disk."""
        candles = self.aggregator.get_and_clear_completed()

        if not candles:
            return

        # Group candles by day
        candles_by_day: dict = {}
        for candle in candles:
            dt = datetime.fromtimestamp(candle["open_time"] / 1000, tz=timezone.utc)
            date_str = dt.strftime("%Y-%m-%d")
            if date_str not in candles_by_day:
                candles_by_day[date_str] = []
            candles_by_day[date_str].append(candle)

        # Save each day's candles
        for date_str, day_candles in candles_by_day.items():
            self._save_candles(day_candles, date_str)

    async def _handle_message(self, message: str) -> None:
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)

            # Verify it's a trade event
            if data.get("e") != "trade":
                return

            # Extract trade data
            trade_time = int(data["T"])  # Trade time in milliseconds
            price = float(data["p"])
            quantity = float(data["q"])

            self.trades_received += 1

            # Add to aggregator
            self.aggregator.add_trade(trade_time, price, quantity)

        except (KeyError, ValueError, json.JSONDecodeError) as e:
            self.logger.warning(f"Error processing message: {e}")

    async def _connect_and_receive(self) -> None:
        """Connect to WebSocket and receive messages."""
        ws_url = self._get_ws_url()
        self.logger.info(f"🔌 Connecting to {ws_url}")

        try:
            async with websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            ) as ws:
                self.logger.info("✅ Connected to Binance WebSocket")

                while self.running and not self.shutdown_event.is_set():
                    try:
                        # Wait for message with timeout
                        message = await asyncio.wait_for(ws.recv(), timeout=30)
                        await self._handle_message(message)
                    except asyncio.TimeoutError:
                        # No message received, check if still running
                        self.logger.debug("No message received for 30s, checking connection")
                        continue
                    except websockets.ConnectionClosed:
                        self.logger.warning("⚠️ WebSocket connection closed")
                        break

        except Exception as e:
            self.logger.error(f"❌ WebSocket error: {e}")
            raise

    async def _flush_loop(self) -> None:
        """Periodically flush candles to disk."""
        while self.running and not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.flush_interval)
                if self.running:
                    self._flush_to_disk()
                    self.logger.info(
                        f"📊 Stats: {self.trades_received:,} trades, "
                        f"{self.candles_saved:,} candles saved, "
                        f"{self.reconnect_count} reconnects"
                    )
            except asyncio.CancelledError:
                break

    async def run(self, duration_seconds: int) -> None:
        """
        Run the collector for specified duration.

        Args:
            duration_seconds: How long to run in seconds
        """
        self.running = True
        self.logger.info(f"🚀 Starting data collection for {duration_seconds}s")
        self.logger.info(f"   Symbol: {self.symbol_upper}")
        self.logger.info(f"   Data dir: {self.data_dir}")
        self.logger.info(f"   Flush interval: {self.flush_interval}s")

        # Setup shutdown handlers
        loop = asyncio.get_event_loop()

        def handle_shutdown(sig):
            self.logger.info(f"📛 Received signal {sig.name}, shutting down gracefully...")
            self.running = False
            self.shutdown_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: handle_shutdown(s))

        # Start flush loop
        flush_task = asyncio.create_task(self._flush_loop())

        # Main collection loop with reconnection
        start_time = asyncio.get_event_loop().time()
        end_time = start_time + duration_seconds

        while self.running and asyncio.get_event_loop().time() < end_time:
            try:
                await self._connect_and_receive()
            except Exception as e:
                self.logger.error(f"❌ Connection error: {e}")

            if self.running and asyncio.get_event_loop().time() < end_time:
                self.reconnect_count += 1
                wait_time = min(5 * self.reconnect_count, 30)
                self.logger.info(f"🔄 Reconnecting in {wait_time}s...")
                await asyncio.sleep(wait_time)

        # Shutdown
        self.running = False
        self.shutdown_event.set()
        flush_task.cancel()

        try:
            await flush_task
        except asyncio.CancelledError:
            pass

        # Final flush
        self.aggregator.flush_current()
        self._flush_to_disk()

        self.logger.info("="*60)
        self.logger.info("📈 COLLECTION COMPLETE")
        self.logger.info(f"   Total trades: {self.trades_received:,}")
        self.logger.info(f"   Total candles saved: {self.candles_saved:,}")
        self.logger.info(f"   Reconnections: {self.reconnect_count}")
        self.logger.info("="*60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect real-time 1-second candle data from Binance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading pair symbol"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=20700,  # 5h45min
        help="Duration in seconds (default: 5h45min = 20700s)"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="v2/data/raw/1s",
        help="Directory to save Parquet files"
    )

    parser.add_argument(
        "--flush-interval",
        type=int,
        default=60,
        help="Seconds between disk flushes"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level, file_enabled=False)

    logger = get_logger(__name__)
    logger.info("="*60)
    logger.info("📡 BINANCE REAL-TIME 1S DATA COLLECTOR")
    logger.info("="*60)

    collector = RealtimeDataCollector(
        symbol=args.symbol,
        data_dir=args.data_dir,
        flush_interval=args.flush_interval,
    )

    await collector.run(args.duration)


if __name__ == "__main__":
    asyncio.run(main())
