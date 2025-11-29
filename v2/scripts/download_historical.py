#!/usr/bin/env python3
"""
Download Historical Data Script

CLI para download de dados históricos da Binance.

Uso:
    python download_historical.py --symbol BTCUSDT --months 6 --timeframes 1m 5m 1h
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Adiciona diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.historical import HistoricalDataDownloader
from src.utils.logger import setup_logging, get_logger


def parse_args():
    """Parse argumentos da linha de comando."""
    parser = argparse.ArgumentParser(
        description="Download de dados históricos da Binance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Símbolo do par (ex: BTCUSDT, ETHUSDT)"
    )
    
    parser.add_argument(
        "--months",
        type=int,
        default=6,
        help="Número de meses para baixar"
    )
    
    parser.add_argument(
        "--timeframes",
        type=str,
        nargs="+",
        default=["1m"],
        help="Timeframes para baixar (ex: 1m 5m 15m 1h)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="v2/data/raw",
        help="Diretório de saída"
    )
    
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        help="Delay entre requests em segundos"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Modo verboso (DEBUG)"
    )
    
    return parser.parse_args()


async def main():
    """Função principal."""
    args = parse_args()
    
    # Configura logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)
    logger = get_logger(__name__)
    
    logger.info("="*60)
    logger.info("📥 DOWNLOAD DE DADOS HISTÓRICOS")
    logger.info("="*60)
    logger.info(f"   Símbolo: {args.symbol}")
    logger.info(f"   Meses: {args.months}")
    logger.info(f"   Timeframes: {args.timeframes}")
    logger.info(f"   Output: {args.output_dir}")
    logger.info("="*60)
    
    try:
        # Cria downloader
        downloader = HistoricalDataDownloader(
            data_dir=args.output_dir,
            rate_limit_delay=args.rate_limit
        )
        
        # Download
        results = await downloader.download_multiple_timeframes(
            symbol=args.symbol,
            intervals=args.timeframes,
            months=args.months,
            save=True
        )
        
        # Sumário
        logger.info("\n" + "="*60)
        logger.info("📊 SUMÁRIO")
        logger.info("="*60)
        
        total_candles = 0
        for interval, df in results.items():
            count = len(df)
            total_candles += count
            logger.info(f"   {interval}: {count:,} candles")
        
        logger.info(f"\n   TOTAL: {total_candles:,} candles")
        logger.info("="*60)
        logger.info("✅ Download completo!")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Download interrompido pelo usuário")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Erro: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
