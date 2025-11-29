#!/usr/bin/env python3
"""
Merge 1-second candle data files utility.

Consolidates multiple daily Parquet files into a single file for backtesting.
Removes duplicates and ensures data is sorted by timestamp.

Usage:
    python merge_1s_data.py --start 2024-01-01 --end 2024-01-31
    python merge_1s_data.py --output combined_btcusdt_1s.parquet
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logging, get_logger


def find_data_files(
    data_dir: Path,
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list:
    """
    Find all 1s data files for a symbol within date range.

    Args:
        data_dir: Directory containing Parquet files
        symbol: Trading pair symbol (uppercase)
        start_date: Start date string (YYYY-MM-DD), optional
        end_date: End date string (YYYY-MM-DD), optional

    Returns:
        List of Path objects for matching files, sorted by date
    """
    pattern = f"{symbol}_1s_*.parquet"
    files = list(data_dir.glob(pattern))

    if not files:
        return []

    # Parse dates from filenames and filter
    filtered_files = []
    for f in files:
        # Extract date from filename: BTCUSDT_1s_2024-01-15.parquet
        try:
            date_str = f.stem.split("_")[-1]
            file_date = datetime.strptime(date_str, "%Y-%m-%d")

            # Apply date filters
            if start_date:
                start = datetime.strptime(start_date, "%Y-%m-%d")
                if file_date < start:
                    continue

            if end_date:
                end = datetime.strptime(end_date, "%Y-%m-%d")
                if file_date > end:
                    continue

            filtered_files.append((file_date, f))
        except ValueError:
            continue

    # Sort by date
    filtered_files.sort(key=lambda x: x[0])
    return [f for _, f in filtered_files]


def merge_files(
    files: list,
    output_path: Path,
    logger,
) -> pd.DataFrame:
    """
    Merge multiple Parquet files into one.

    Args:
        files: List of Path objects to merge
        output_path: Output file path
        logger: Logger instance

    Returns:
        Merged DataFrame
    """
    if not files:
        raise ValueError("No files to merge")

    logger.info(f"📁 Found {len(files)} files to merge")

    # Read and concatenate all files
    dfs = []
    total_rows = 0

    for f in files:
        try:
            df = pd.read_parquet(f)
            rows = len(df)
            total_rows += rows
            dfs.append(df)
            logger.debug(f"   Loaded {f.name}: {rows:,} rows")
        except Exception as e:
            logger.warning(f"   ⚠️ Error loading {f.name}: {e}")
            continue

    if not dfs:
        raise ValueError("No data loaded from files")

    logger.info(f"📊 Total rows before merge: {total_rows:,}")

    # Concatenate
    merged = pd.concat(dfs, ignore_index=True)

    # Remove duplicates based on open_time
    before_dedup = len(merged)
    merged = merged.drop_duplicates(subset=["open_time"], keep="last")
    after_dedup = len(merged)
    duplicates_removed = before_dedup - after_dedup

    if duplicates_removed > 0:
        logger.info(f"🔄 Removed {duplicates_removed:,} duplicate candles")

    # Sort by timestamp
    merged = merged.sort_values("open_time").reset_index(drop=True)

    # Save merged file
    merged.to_parquet(output_path, index=False, compression="snappy")
    logger.info(f"💾 Saved merged data to {output_path}")
    logger.info(f"   Final rows: {len(merged):,}")

    return merged


def get_data_summary(df: pd.DataFrame, logger) -> None:
    """Print summary statistics for the merged data."""
    if df.empty:
        logger.info("No data to summarize")
        return

    # Convert timestamps
    start_ts = datetime.fromtimestamp(df["open_time"].min() / 1000)
    end_ts = datetime.fromtimestamp(df["open_time"].max() / 1000)
    duration = end_ts - start_ts

    logger.info("\n" + "="*60)
    logger.info("📈 DATA SUMMARY")
    logger.info("="*60)
    logger.info(f"   Period: {start_ts} to {end_ts}")
    logger.info(f"   Duration: {duration}")
    logger.info(f"   Total candles: {len(df):,}")
    logger.info(f"   Price range: {df['low'].min():.2f} - {df['high'].max():.2f}")
    logger.info(f"   Total volume: {df['volume'].sum():,.2f}")
    logger.info(f"   Total trades: {df['trades_count'].sum():,}")

    # Check for gaps (more than 1 second between candles)
    if len(df) > 1:
        time_diffs = df["open_time"].diff().dropna()
        gaps = time_diffs[time_diffs > 1000]  # More than 1 second
        if len(gaps) > 0:
            logger.info(f"   ⚠️ Data gaps: {len(gaps):,} (>1 second)")
        else:
            logger.info("   ✅ No data gaps detected")

    logger.info("="*60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge 1-second candle data files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading pair symbol"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="v2/data/raw/1s",
        help="Directory containing Parquet files"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (default: v2/data/processed/<symbol>_1s_merged.parquet)"
    )

    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level, file_enabled=False)

    logger = get_logger(__name__)
    logger.info("="*60)
    logger.info("🔗 1-SECOND DATA MERGE UTILITY")
    logger.info("="*60)

    data_dir = Path(args.data_dir)
    symbol = args.symbol.upper()

    if not data_dir.exists():
        logger.error(f"❌ Data directory not found: {data_dir}")
        sys.exit(1)

    # Find files to merge
    files = find_data_files(
        data_dir=data_dir,
        symbol=symbol,
        start_date=args.start,
        end_date=args.end,
    )

    if not files:
        logger.error(f"❌ No files found matching {symbol}_1s_*.parquet")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        processed_dir = Path("v2/data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = processed_dir / f"{symbol}_1s_merged.parquet"

    logger.info(f"   Symbol: {symbol}")
    logger.info(f"   Data dir: {data_dir}")
    logger.info(f"   Output: {output_path}")
    if args.start:
        logger.info(f"   Start date: {args.start}")
    if args.end:
        logger.info(f"   End date: {args.end}")
    logger.info("-"*60)

    try:
        merged_df = merge_files(files, output_path, logger)
        get_data_summary(merged_df, logger)
        logger.info("✅ Merge complete!")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
