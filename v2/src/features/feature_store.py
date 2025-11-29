"""
Feature Store.

Armazena e gerencia features calculadas em formato Parquet.
"""
from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime


logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Armazena e gerencia features calculadas.
    
    Usa formato Parquet para:
    - Compressão eficiente
    - Leitura rápida de colunas específicas
    - Particionamento por data
    """
    
    def __init__(
        self, 
        base_path: str = "v2/data/features",
        compression: str = "snappy"
    ):
        """
        Inicializa o Feature Store.
        
        Args:
            base_path: Diretório base para armazenamento
            compression: Tipo de compressão ('snappy', 'gzip', 'zstd')
        """
        self.base_path = Path(base_path)
        self.compression = compression
        self._ensure_directory()
        
        # Cache em memória para últimos valores
        self._cache: Dict[str, pd.DataFrame] = {}
        self._latest_values: Dict[str, float] = {}
        
    def _ensure_directory(self) -> None:
        """Garante que o diretório base existe."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    def _get_file_path(self, name: str, date: Optional[str] = None) -> Path:
        """
        Gera caminho do arquivo para uma feature.
        
        Args:
            name: Nome da feature ou dataset
            date: Data opcional para particionamento (YYYY-MM-DD)
            
        Returns:
            Path do arquivo
        """
        if date:
            # Particionamento por data
            return self.base_path / f"{name}" / f"date={date}" / "data.parquet"
        return self.base_path / f"{name}.parquet"
        
    def save(
        self, 
        features: pd.DataFrame, 
        name: str,
        partition_by_date: bool = True
    ) -> bool:
        """
        Salva features em Parquet.
        
        Args:
            features: DataFrame com features calculadas
            name: Nome do dataset
            partition_by_date: Se deve particionar por data
            
        Returns:
            True se salvou com sucesso
        """
        try:
            if features.empty:
                logger.warning(f"DataFrame vazio, nada para salvar: {name}")
                return False
            
            if partition_by_date and isinstance(features.index, pd.DatetimeIndex):
                # Particiona por data
                return self._save_partitioned(features, name)
            else:
                # Salva em arquivo único
                file_path = self._get_file_path(name)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                features.to_parquet(
                    file_path,
                    compression=self.compression,
                    index=True
                )
                
                logger.info(f"Features salvas: {file_path}")
                
                # Atualiza cache
                self._cache[name] = features
                
            return True
            
        except Exception as e:
            logger.error(f"Erro salvando features {name}: {e}")
            return False
            
    def _save_partitioned(self, features: pd.DataFrame, name: str) -> bool:
        """Salva features particionadas por data."""
        try:
            # Agrupa por data
            features_copy = features.copy()
            features_copy['_date'] = features_copy.index.date
            
            for date, group in features_copy.groupby('_date'):
                date_str = str(date)
                file_path = self._get_file_path(name, date_str)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Remove coluna auxiliar
                group = group.drop(columns=['_date'])
                
                group.to_parquet(
                    file_path,
                    compression=self.compression,
                    index=True
                )
                
            logger.info(f"Features salvas (particionado): {name}")
            return True
            
        except Exception as e:
            logger.error(f"Erro salvando features particionadas {name}: {e}")
            return False
            
    def load(
        self, 
        name: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Carrega features.
        
        Args:
            name: Nome do dataset
            start_date: Data inicial (YYYY-MM-DD)
            end_date: Data final (YYYY-MM-DD)
            columns: Colunas específicas a carregar
            
        Returns:
            DataFrame com features
        """
        try:
            # Tenta carregar arquivo único primeiro
            file_path = self._get_file_path(name)
            
            if file_path.exists():
                df = pd.read_parquet(file_path, columns=columns)
                
                # Filtra por data se especificado
                if start_date or end_date:
                    if isinstance(df.index, pd.DatetimeIndex):
                        if start_date:
                            df = df[df.index >= start_date]
                        if end_date:
                            df = df[df.index <= end_date]
                            
                return df
            
            # Tenta carregar particionado
            partition_path = self.base_path / name
            if partition_path.exists():
                return self._load_partitioned(name, start_date, end_date, columns)
            
            logger.warning(f"Features não encontradas: {name}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Erro carregando features {name}: {e}")
            return pd.DataFrame()
            
    def _load_partitioned(
        self, 
        name: str, 
        start_date: Optional[str],
        end_date: Optional[str],
        columns: Optional[List[str]]
    ) -> pd.DataFrame:
        """Carrega features particionadas."""
        partition_path = self.base_path / name
        dfs = []
        
        for date_dir in sorted(partition_path.iterdir()):
            if not date_dir.is_dir():
                continue
                
            # Extrai data do nome do diretório (date=YYYY-MM-DD)
            if not date_dir.name.startswith('date='):
                continue
                
            date_str = date_dir.name.replace('date=', '')
            
            # Filtra por data
            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue
            
            parquet_file = date_dir / "data.parquet"
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file, columns=columns)
                dfs.append(df)
        
        if dfs:
            return pd.concat(dfs, axis=0).sort_index()
        return pd.DataFrame()
        
    def get_latest(self, feature_names: List[str]) -> Dict[str, float]:
        """
        Retorna valores mais recentes das features.
        
        Args:
            feature_names: Lista de nomes de features
            
        Returns:
            Dict com nome -> valor mais recente
        """
        result = {}
        
        for name in feature_names:
            if name in self._latest_values:
                result[name] = self._latest_values[name]
            else:
                result[name] = 0.0
                
        return result
        
    def update_latest(self, values: Dict[str, float]) -> None:
        """
        Atualiza valores mais recentes (para uso em real-time).
        
        Args:
            values: Dict com nome -> valor
        """
        self._latest_values.update(values)
        
    def append(
        self, 
        features: pd.DataFrame, 
        name: str
    ) -> bool:
        """
        Adiciona novas linhas a um dataset existente.
        
        Args:
            features: Novas features a adicionar
            name: Nome do dataset
            
        Returns:
            True se adicionou com sucesso
        """
        try:
            existing = self.load(name)
            
            if existing.empty:
                return self.save(features, name)
            
            # Concatena e remove duplicatas
            combined = pd.concat([existing, features], axis=0)
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
            
            return self.save(combined, name)
            
        except Exception as e:
            logger.error(f"Erro adicionando features {name}: {e}")
            return False
            
    def delete(self, name: str) -> bool:
        """
        Remove um dataset de features.
        
        Args:
            name: Nome do dataset
            
        Returns:
            True se removeu com sucesso
        """
        try:
            import shutil
            
            file_path = self._get_file_path(name)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Features removidas: {file_path}")
                
            # Remove também diretório particionado
            partition_path = self.base_path / name
            if partition_path.exists():
                shutil.rmtree(partition_path)
                logger.info(f"Features particionadas removidas: {partition_path}")
            
            # Limpa cache
            if name in self._cache:
                del self._cache[name]
                
            return True
            
        except Exception as e:
            logger.error(f"Erro removendo features {name}: {e}")
            return False
            
    def list_datasets(self) -> List[str]:
        """
        Lista datasets disponíveis.
        
        Returns:
            Lista de nomes de datasets
        """
        datasets = []
        
        for item in self.base_path.iterdir():
            if item.is_file() and item.suffix == '.parquet':
                datasets.append(item.stem)
            elif item.is_dir():
                datasets.append(item.name)
                
        return sorted(datasets)
        
    def get_info(self, name: str) -> Dict[str, Any]:
        """
        Retorna informações sobre um dataset.
        
        Args:
            name: Nome do dataset
            
        Returns:
            Dict com informações (colunas, linhas, datas, etc)
        """
        df = self.load(name)
        
        if df.empty:
            return {'error': 'Dataset não encontrado'}
        
        info = {
            'name': name,
            'rows': len(df),
            'columns': list(df.columns),
            'dtypes': {str(col): str(dtype) for col, dtype in df.dtypes.items()},
            'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        if isinstance(df.index, pd.DatetimeIndex):
            info['start_date'] = str(df.index.min())
            info['end_date'] = str(df.index.max())
            
        return info
        
    def clear_cache(self) -> None:
        """Limpa cache em memória."""
        self._cache.clear()
        self._latest_values.clear()
