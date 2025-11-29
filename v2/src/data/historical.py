"""
Historical Data Downloader - Download assíncrono de dados históricos

Features:
- Download assíncrono com rate limiting
- Múltiplos timeframes
- Salva em Parquet
- Progress logging
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..connectors.binance_client import BinanceClient
from ..utils.logger import get_logger
from ..utils.parquet_manager import ParquetManager

logger = get_logger(__name__)


class HistoricalDataDownloader:
    """
    Downloader de dados históricos da Binance.
    
    Baixa klines de forma assíncrona e salva em formato Parquet.
    
    Attributes:
        client: Cliente Binance
        parquet_manager: Gerenciador de arquivos Parquet
        rate_limit_delay: Delay entre requests (segundos)
    """
    
    # Colunas dos klines da Binance
    KLINE_COLUMNS = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
        'taker_buy_quote_volume', 'ignore'
    ]
    
    def __init__(
        self,
        client: Optional[BinanceClient] = None,
        data_dir: str = "v2/data/raw",
        rate_limit_delay: float = 0.5
    ):
        """
        Inicializa o downloader.
        
        Args:
            client: Cliente Binance (cria um se não fornecido)
            data_dir: Diretório para salvar dados
            rate_limit_delay: Delay entre requests em segundos
        """
        self.client = client or BinanceClient(use_demo=True)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_delay = rate_limit_delay
        self.parquet_manager = ParquetManager(base_dir=str(self.data_dir.parent))
    
    async def download_klines(
        self,
        symbol: str,
        interval: str = "1m",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        months: int = 6
    ) -> pd.DataFrame:
        """
        Baixa klines históricos.
        
        Args:
            symbol: Símbolo do par (ex: BTCUSDT)
            interval: Intervalo (1m, 5m, 15m, 1h, etc)
            start_time: Data/hora de início
            end_time: Data/hora de fim
            months: Número de meses para baixar (se start_time não definido)
            
        Returns:
            DataFrame com os klines
        """
        # Define período
        if end_time is None:
            end_time = datetime.now()
        
        if start_time is None:
            start_time = end_time - timedelta(days=months * 30)
        
        logger.info(
            f"📥 Iniciando download: {symbol} {interval} "
            f"de {start_time.strftime('%Y-%m-%d')} até {end_time.strftime('%Y-%m-%d')}"
        )
        
        all_klines = []
        current_start = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        # Calcula intervalo em ms para paginação
        interval_ms = self._interval_to_ms(interval)
        batch_size = 1000  # Máximo de klines por request
        
        total_batches = 0
        
        try:
            async with self.client:
                while current_start < end_ms:
                    # Rate limiting
                    await asyncio.sleep(self.rate_limit_delay)
                    
                    # Download batch
                    klines = await self.client.get_klines(
                        symbol=symbol,
                        interval=interval,
                        start_time=current_start,
                        end_time=end_ms,
                        limit=batch_size
                    )
                    
                    if not klines:
                        break
                    
                    all_klines.extend(klines)
                    total_batches += 1
                    
                    # Atualiza start para próximo batch
                    last_close_time = klines[-1][6]  # close_time
                    current_start = last_close_time + 1
                    
                    # Log progresso
                    if total_batches % 10 == 0:
                        progress_date = datetime.fromtimestamp(current_start / 1000)
                        logger.info(f"  📊 Progresso: {total_batches} batches, até {progress_date.strftime('%Y-%m-%d')}")
                    
                    # Se recebeu menos que batch_size, chegamos ao fim
                    if len(klines) < batch_size:
                        break
        
        except Exception as e:
            logger.error(f"❌ Erro no download: {e}")
            if not all_klines:
                raise
        
        # Converte para DataFrame
        df = pd.DataFrame(all_klines, columns=self.KLINE_COLUMNS)
        
        # Converte tipos
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                    'taker_buy_volume', 'taker_buy_quote_volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['trades'] = pd.to_numeric(df['trades'], errors='coerce').astype('Int64')
        
        # Remove coluna ignore
        df = df.drop(columns=['ignore'], errors='ignore')
        
        # Remove duplicatas
        df = df.drop_duplicates(subset=['open_time'], keep='last')
        df = df.sort_values('open_time').reset_index(drop=True)
        
        logger.info(
            f"✅ Download completo: {len(df)} candles, "
            f"{total_batches} batches"
        )
        
        return df
    
    def _interval_to_ms(self, interval: str) -> int:
        """Converte intervalo para milissegundos."""
        unit = interval[-1]
        value = int(interval[:-1])
        
        multipliers = {
            's': 1000,
            'm': 60 * 1000,
            'h': 60 * 60 * 1000,
            'd': 24 * 60 * 60 * 1000,
            'w': 7 * 24 * 60 * 60 * 1000,
            'M': 30 * 24 * 60 * 60 * 1000
        }
        
        return value * multipliers.get(unit, 60 * 1000)
    
    async def download_multiple_timeframes(
        self,
        symbol: str,
        intervals: List[str],
        months: int = 6,
        save: bool = True
    ) -> dict:
        """
        Baixa múltiplos timeframes.
        
        Args:
            symbol: Símbolo do par
            intervals: Lista de intervalos (ex: ['1m', '5m', '1h'])
            months: Número de meses
            save: Se deve salvar em Parquet
            
        Returns:
            Dicionário com DataFrames por intervalo
        """
        results = {}
        
        for interval in intervals:
            logger.info(f"\n{'='*50}")
            logger.info(f"⏰ Timeframe: {interval}")
            logger.info(f"{'='*50}")
            
            df = await self.download_klines(
                symbol=symbol,
                interval=interval,
                months=months
            )
            
            results[interval] = df
            
            if save and not df.empty:
                filename = f"{symbol}_{interval}"
                self.parquet_manager.save(df, f"raw/{filename}")
                logger.info(f"💾 Salvo em: raw/{filename}.parquet")
        
        return results
    
    async def download_trades(
        self,
        symbol: str,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Baixa trades recentes.
        
        Args:
            symbol: Símbolo do par
            limit: Número de trades (max 1000)
            
        Returns:
            DataFrame com trades
        """
        logger.info(f"📥 Baixando {limit} trades recentes de {symbol}")
        
        async with self.client:
            trades = await self.client.get_recent_trades(symbol, limit)
        
        df = pd.DataFrame(trades)
        
        if not df.empty:
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            df['qty'] = pd.to_numeric(df['qty'], errors='coerce')
            df['quoteQty'] = pd.to_numeric(df['quoteQty'], errors='coerce')
        
        logger.info(f"✅ {len(df)} trades baixados")
        return df
