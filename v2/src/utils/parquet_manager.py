"""
Parquet Manager - Gerenciador de arquivos Parquet

Features:
- Compressão snappy
- Particionamento por data
- Append eficiente
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .logger import get_logger

logger = get_logger(__name__)


class ParquetManager:
    """
    Gerenciador de arquivos Parquet.
    
    Fornece métodos para salvar, carregar e gerenciar dados em formato Parquet
    com compressão snappy e particionamento opcional.
    
    Attributes:
        base_dir: Diretório base para arquivos
        compression: Tipo de compressão (snappy, gzip, zstd)
    """
    
    def __init__(
        self,
        base_dir: str = "v2/data",
        compression: str = "snappy"
    ):
        """
        Inicializa o gerenciador.
        
        Args:
            base_dir: Diretório base para arquivos Parquet
            compression: Tipo de compressão (snappy, gzip, zstd, none)
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.compression = compression
    
    def save(
        self,
        df: pd.DataFrame,
        name: str,
        partition_cols: Optional[List[str]] = None,
        append: bool = False
    ) -> Path:
        """
        Salva DataFrame em formato Parquet.
        
        Args:
            df: DataFrame a salvar
            name: Nome do arquivo (sem extensão)
            partition_cols: Colunas para particionamento
            append: Se True, adiciona aos dados existentes
            
        Returns:
            Path do arquivo/diretório salvo
        """
        if df.empty:
            logger.warning(f"⚠️ DataFrame vazio, não salvando: {name}")
            return Path()
        
        filepath = self.base_dir / f"{name}.parquet"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Converte para PyArrow Table
        table = pa.Table.from_pandas(df)
        
        if partition_cols:
            # Salva com particionamento
            pq.write_to_dataset(
                table,
                root_path=str(filepath),
                partition_cols=partition_cols,
                compression=self.compression,
                existing_data_behavior='overwrite_or_ignore' if append else 'delete_matching'
            )
        else:
            if append and filepath.exists():
                # Carrega dados existentes e concatena
                existing_df = self.load(name)
                df = pd.concat([existing_df, df], ignore_index=True)
                table = pa.Table.from_pandas(df)
            
            # Salva arquivo único
            pq.write_table(
                table,
                filepath,
                compression=self.compression
            )
        
        logger.debug(f"💾 Salvo: {filepath} ({len(df)} linhas)")
        return filepath
    
    def load(
        self,
        name: str,
        columns: Optional[List[str]] = None,
        filters: Optional[List] = None
    ) -> pd.DataFrame:
        """
        Carrega dados de arquivo Parquet.
        
        Args:
            name: Nome do arquivo (sem extensão)
            columns: Colunas a carregar (None = todas)
            filters: Filtros PyArrow para partições
            
        Returns:
            DataFrame com os dados
        """
        filepath = self.base_dir / f"{name}.parquet"
        
        if not filepath.exists():
            # Tenta como diretório particionado
            if filepath.is_dir():
                pass
            else:
                logger.warning(f"⚠️ Arquivo não encontrado: {filepath}")
                return pd.DataFrame()
        
        try:
            if filepath.is_dir():
                # Dataset particionado
                dataset = pq.ParquetDataset(filepath, filters=filters)
                table = dataset.read(columns=columns)
            else:
                # Arquivo único
                table = pq.read_table(filepath, columns=columns)
            
            df = table.to_pandas()
            logger.debug(f"📖 Carregado: {filepath} ({len(df)} linhas)")
            return df
            
        except Exception as e:
            logger.error(f"❌ Erro ao carregar {filepath}: {e}")
            return pd.DataFrame()
    
    def save_partitioned_by_date(
        self,
        df: pd.DataFrame,
        name: str,
        date_column: str = "open_time",
        append: bool = True
    ) -> Path:
        """
        Salva DataFrame particionado por data.
        
        Args:
            df: DataFrame a salvar
            name: Nome base do arquivo
            date_column: Coluna com data/timestamp
            append: Se True, adiciona aos dados existentes
            
        Returns:
            Path do diretório
        """
        if df.empty:
            return Path()
        
        # Cria coluna de partição
        df = df.copy()
        
        if date_column in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[date_column]):
                df['_date'] = df[date_column].dt.strftime('%Y-%m-%d')
            else:
                df['_date'] = pd.to_datetime(df[date_column]).dt.strftime('%Y-%m-%d')
        else:
            df['_date'] = datetime.now().strftime('%Y-%m-%d')
        
        return self.save(df, name, partition_cols=['_date'], append=append)
    
    def load_date_range(
        self,
        name: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        date_column: str = "_date"
    ) -> pd.DataFrame:
        """
        Carrega dados de um range de datas.
        
        Args:
            name: Nome do arquivo/diretório
            start_date: Data inicial
            end_date: Data final
            date_column: Coluna de partição de data
            
        Returns:
            DataFrame filtrado
        """
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        
        filters = [
            (date_column, '>=', start_date),
            (date_column, '<=', end_date)
        ]
        
        return self.load(name, filters=filters)
    
    def append(
        self,
        df: pd.DataFrame,
        name: str
    ) -> Path:
        """
        Adiciona dados a arquivo existente.
        
        Args:
            df: DataFrame a adicionar
            name: Nome do arquivo
            
        Returns:
            Path do arquivo
        """
        return self.save(df, name, append=True)
    
    def delete(self, name: str) -> bool:
        """
        Remove arquivo Parquet.
        
        Args:
            name: Nome do arquivo
            
        Returns:
            True se removido, False caso contrário
        """
        filepath = self.base_dir / f"{name}.parquet"
        
        try:
            if filepath.is_dir():
                shutil.rmtree(filepath)
            elif filepath.exists():
                filepath.unlink()
            else:
                return False
            
            logger.info(f"🗑️ Removido: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erro ao remover {filepath}: {e}")
            return False
    
    def list_files(self, pattern: str = "*.parquet") -> List[Path]:
        """
        Lista arquivos Parquet no diretório.
        
        Args:
            pattern: Padrão glob para filtrar
            
        Returns:
            Lista de paths
        """
        return list(self.base_dir.rglob(pattern))
    
    def get_metadata(self, name: str) -> dict:
        """
        Obtém metadados do arquivo Parquet.
        
        Args:
            name: Nome do arquivo
            
        Returns:
            Dicionário com metadados
        """
        filepath = self.base_dir / f"{name}.parquet"
        
        if not filepath.exists():
            return {}
        
        try:
            metadata = pq.read_metadata(filepath)
            return {
                'num_rows': metadata.num_rows,
                'num_columns': metadata.num_columns,
                'num_row_groups': metadata.num_row_groups,
                'created_by': metadata.created_by,
                'format_version': str(metadata.format_version),
                'serialized_size': metadata.serialized_size,
            }
        except Exception:
            return {}
    
    def optimize(self, name: str, row_group_size: int = 100000) -> Path:
        """
        Otimiza arquivo Parquet (reescreve com row groups uniformes).
        
        Args:
            name: Nome do arquivo
            row_group_size: Tamanho dos row groups
            
        Returns:
            Path do arquivo otimizado
        """
        df = self.load(name)
        
        if df.empty:
            return Path()
        
        filepath = self.base_dir / f"{name}.parquet"
        table = pa.Table.from_pandas(df)
        
        pq.write_table(
            table,
            filepath,
            compression=self.compression,
            row_group_size=row_group_size
        )
        
        logger.info(f"✨ Otimizado: {filepath}")
        return filepath
