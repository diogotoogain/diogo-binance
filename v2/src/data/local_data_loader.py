"""
Local Data Loader for Binance Historical Data.

Loads historical kline data from local ZIP/CSV files downloaded from Binance.
Supports multiple timeframes and efficient memory management.
"""

import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class LocalDataLoader:
    """
    Loads historical data from local Binance ZIP/CSV files.

    Expected file structure:
    downloads/futures/um/daily/klines/BTCUSDT/1m/
    ├── BTCUSDT-1m-2019-12-31.zip
    ├── BTCUSDT-1m-2020-01-01.zip
    └── ...

    Each ZIP contains a CSV with columns:
    open_time, open, high, low, close, volume, close_time,
    quote_volume, trades, taker_buy_volume, taker_buy_quote_volume, ignore
    """

    # Standard Binance kline columns
    KLINE_COLUMNS = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
        'taker_buy_quote_volume', 'ignore'
    ]

    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize LocalDataLoader.

        Args:
            data_dir: Base directory containing downloaded data.
        """
        self.data_dir = Path(data_dir) if data_dir else None

    def load_timeframe(
        self,
        data_dir: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        symbol: str = "BTCUSDT"
    ) -> pd.DataFrame:
        """
        Load and concatenate data for a single timeframe.

        Args:
            data_dir: Base directory for data files
            timeframe: Timeframe (1m, 5m, 15m, etc.)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbol: Trading pair symbol

        Returns:
            DataFrame with all candles for the period
        """
        data_path = Path(data_dir) / timeframe
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        logger.info(
            f"Loading {symbol} {timeframe} data from {start_date} to {end_date}"
        )

        all_data = []
        files_loaded = 0
        files_skipped = 0

        # Find and process files in date range
        for file_path in sorted(data_path.glob(f"{symbol}-{timeframe}-*.zip")):
            file_date = self._extract_date_from_filename(file_path.name)
            if file_date is None:
                continue

            if start <= file_date <= end:
                try:
                    df = self._load_zip_file(file_path)
                    all_data.append(df)
                    files_loaded += 1
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    files_skipped += 1

        # Also check for plain CSV files
        for file_path in sorted(data_path.glob(f"{symbol}-{timeframe}-*.csv")):
            file_date = self._extract_date_from_filename(file_path.name)
            if file_date is None:
                continue

            if start <= file_date <= end:
                try:
                    df = self._load_csv_file(file_path)
                    all_data.append(df)
                    files_loaded += 1
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")
                    files_skipped += 1

        if not all_data:
            logger.warning(f"No data files found for {symbol} {timeframe}")
            return pd.DataFrame(columns=self.KLINE_COLUMNS[:-1])

        # Concatenate and process
        df = pd.concat(all_data, ignore_index=True)

        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['open_time'], keep='last')
        df = df.sort_values('open_time').reset_index(drop=True)

        # Ensure datetime types
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')

        logger.info(
            f"Loaded {len(df):,} candles from {files_loaded} files "
            f"({files_skipped} skipped)"
        )

        return df

    def load_multi_timeframe(
        self,
        data_dir: str,
        timeframes: Optional[List[str]] = None,
        start_date: str = "2019-12-31",
        end_date: str = "2025-11-28",
        symbol: str = "BTCUSDT"
    ) -> Dict[str, pd.DataFrame]:
        """
        Load multiple timeframes simultaneously.

        Args:
            data_dir: Base directory for data files
            timeframes: List of timeframes to load (default: 1m, 5m, 15m)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbol: Trading pair symbol

        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        if timeframes is None:
            timeframes = ["1m", "5m", "15m"]

        results = {}
        for tf in timeframes:
            try:
                results[tf] = self.load_timeframe(
                    data_dir, tf, start_date, end_date, symbol
                )
            except Exception as e:
                logger.error(f"Failed to load {tf}: {e}")
                results[tf] = pd.DataFrame()

        return results

    def _load_zip_file(self, file_path: Path) -> pd.DataFrame:
        """Load data from a ZIP file."""
        with zipfile.ZipFile(file_path, 'r') as zf:
            # Get CSV name from ZIP
            csv_names = [n for n in zf.namelist() if n.endswith('.csv')]
            if not csv_names:
                raise ValueError(f"No CSV found in {file_path}")

            with zf.open(csv_names[0]) as f:
                df = pd.read_csv(f, header=None, names=self.KLINE_COLUMNS)

        return self._process_dataframe(df)

    def _load_csv_file(self, file_path: Path) -> pd.DataFrame:
        """Load data from a CSV file."""
        # Try reading with and without header
        try:
            df = pd.read_csv(file_path, header=None, names=self.KLINE_COLUMNS)
            # Check if first row might be a header
            if df.iloc[0]['open_time'] == 'open_time':
                df = pd.read_csv(file_path, names=self.KLINE_COLUMNS, skiprows=1)
        except Exception:
            df = pd.read_csv(file_path)

        return self._process_dataframe(df)

    def _process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process and clean loaded dataframe."""
        # Drop ignore column if present
        if 'ignore' in df.columns:
            df = df.drop(columns=['ignore'])

        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                        'quote_volume', 'taker_buy_volume', 'taker_buy_quote_volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Convert time columns
        if 'open_time' in df.columns:
            df['open_time'] = pd.to_numeric(df['open_time'], errors='coerce')
        if 'close_time' in df.columns:
            df['close_time'] = pd.to_numeric(df['close_time'], errors='coerce')
        if 'trades' in df.columns:
            df['trades'] = pd.to_numeric(df['trades'], errors='coerce').fillna(0).astype(int)

        return df

    def _extract_date_from_filename(self, filename: str) -> Optional[datetime]:
        """Extract date from filename like BTCUSDT-1m-2020-01-01.zip."""
        try:
            # Remove extension
            name = filename.replace('.zip', '').replace('.csv', '')
            parts = name.split('-')
            # Date is last 3 parts: YYYY-MM-DD
            if len(parts) >= 4:
                date_str = '-'.join(parts[-3:])
                return datetime.strptime(date_str, "%Y-%m-%d")
        except (ValueError, IndexError):
            pass
        return None

    def get_data_info(self, data_dir: str, symbol: str = "BTCUSDT") -> Dict[str, Dict]:
        """
        Get information about available data.

        Args:
            data_dir: Base directory for data files
            symbol: Trading pair symbol

        Returns:
            Dictionary with info for each timeframe
        """
        base_path = Path(data_dir)
        info = {}

        for tf_dir in base_path.iterdir():
            if not tf_dir.is_dir():
                continue

            tf = tf_dir.name
            files = list(tf_dir.glob(f"{symbol}-{tf}-*.zip"))
            files.extend(tf_dir.glob(f"{symbol}-{tf}-*.csv"))

            if not files:
                continue

            dates = []
            for f in files:
                date = self._extract_date_from_filename(f.name)
                if date:
                    dates.append(date)

            if dates:
                info[tf] = {
                    'files': len(files),
                    'start_date': min(dates).strftime("%Y-%m-%d"),
                    'end_date': max(dates).strftime("%Y-%m-%d"),
                    'path': str(tf_dir)
                }

        return info

    def estimate_candle_count(
        self,
        start_date: str,
        end_date: str,
        timeframe: str
    ) -> int:
        """
        Estimate number of candles for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Timeframe string (1m, 5m, 15m, etc.)

        Returns:
            Estimated number of candles
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end - start).days

        # Minutes per timeframe
        tf_minutes = {
            '1s': 1/60,
            '1m': 1,
            '5m': 5,
            '15m': 15,
            '30m': 30,
            '1h': 60,
            '4h': 240,
            '1d': 1440
        }

        minutes_per_candle = tf_minutes.get(timeframe, 1)
        candles_per_day = 1440 / minutes_per_candle

        return int(days * candles_per_day)


def load_local_data(
    data_dir: str,
    timeframe: str = "1m",
    start_date: str = "2019-12-31",
    end_date: str = "2025-11-28",
    symbol: str = "BTCUSDT"
) -> pd.DataFrame:
    """
    Convenience function to load local data.

    Args:
        data_dir: Base directory for data files
        timeframe: Timeframe to load
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        symbol: Trading pair symbol

    Returns:
        DataFrame with candle data
    """
    loader = LocalDataLoader()
    return loader.load_timeframe(data_dir, timeframe, start_date, end_date, symbol)
