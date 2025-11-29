"""
Logger - Logging colorido com rotação de arquivo

Features:
- Logs coloridos no console
- Logs em arquivo com rotação
- Configuração via setup_logging()
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


class ColoredFormatter(logging.Formatter):
    """Formatter com cores para o console."""
    
    # Cores por nível
    LEVEL_COLORS = {
        logging.DEBUG: Fore.CYAN if COLORAMA_AVAILABLE else '',
        logging.INFO: Fore.GREEN if COLORAMA_AVAILABLE else '',
        logging.WARNING: Fore.YELLOW if COLORAMA_AVAILABLE else '',
        logging.ERROR: Fore.RED if COLORAMA_AVAILABLE else '',
        logging.CRITICAL: Fore.RED + Style.BRIGHT if COLORAMA_AVAILABLE else '',
    }
    
    RESET = Style.RESET_ALL if COLORAMA_AVAILABLE else ''
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        """
        Inicializa o formatter.
        
        Args:
            fmt: Formato da mensagem
            datefmt: Formato da data
        """
        if fmt is None:
            fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        if datefmt is None:
            datefmt = "%Y-%m-%d %H:%M:%S"
        super().__init__(fmt, datefmt)
    
    def format(self, record: logging.LogRecord) -> str:
        """Formata o registro com cores."""
        # Salva valores originais
        original_levelname = record.levelname
        
        # Aplica cor
        color = self.LEVEL_COLORS.get(record.levelno, '')
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        
        # Formata
        result = super().format(record)
        
        # Restaura
        record.levelname = original_levelname
        
        return result


class FileFormatter(logging.Formatter):
    """Formatter simples para arquivos (sem cores)."""
    
    def __init__(self, fmt: Optional[str] = None, datefmt: Optional[str] = None):
        """
        Inicializa o formatter.
        
        Args:
            fmt: Formato da mensagem
            datefmt: Formato da data
        """
        if fmt is None:
            fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        if datefmt is None:
            datefmt = "%Y-%m-%d %H:%M:%S"
        super().__init__(fmt, datefmt)


# Estado global de configuração
_logging_configured = False
_log_dir: Optional[Path] = None


def setup_logging(
    level: str = "INFO",
    log_dir: str = "v2/logs",
    console_enabled: bool = True,
    file_enabled: bool = True,
    max_file_size_mb: int = 100,
    backup_count: int = 5,
    colored: bool = True
) -> None:
    """
    Configura o sistema de logging.
    
    Args:
        level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Diretório para arquivos de log
        console_enabled: Habilitar log no console
        file_enabled: Habilitar log em arquivo
        max_file_size_mb: Tamanho máximo do arquivo em MB
        backup_count: Número de backups a manter
        colored: Usar cores no console
    """
    global _logging_configured, _log_dir
    
    if _logging_configured:
        return
    
    # Cria diretório de logs
    _log_dir = Path(log_dir)
    _log_dir.mkdir(parents=True, exist_ok=True)
    
    # Nível de log
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configura root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove handlers existentes
    root_logger.handlers = []
    
    # Handler do console
    if console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        if colored and COLORAMA_AVAILABLE:
            console_handler.setFormatter(ColoredFormatter())
        else:
            console_handler.setFormatter(FileFormatter())
        
        root_logger.addHandler(console_handler)
    
    # Handler de arquivo
    if file_enabled:
        log_file = _log_dir / f"v2_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)  # Arquivo sempre DEBUG
        file_handler.setFormatter(FileFormatter())
        
        root_logger.addHandler(file_handler)
    
    _logging_configured = True
    
    # Log inicial
    logger = get_logger(__name__)
    logger.info("="*60)
    logger.info("🚀 Sistema de logging inicializado")
    logger.info(f"   Nível: {level}")
    logger.info(f"   Console: {console_enabled}")
    logger.info(f"   Arquivo: {file_enabled}")
    if file_enabled:
        logger.info(f"   Diretório: {log_dir}")
    logger.info("="*60)


def get_logger(name: str) -> logging.Logger:
    """
    Obtém logger com nome especificado.
    
    Args:
        name: Nome do logger (geralmente __name__)
        
    Returns:
        Logger configurado
    """
    # Configura logging se ainda não foi configurado
    if not _logging_configured:
        setup_logging()
    
    return logging.getLogger(name)


def log_exception(logger: logging.Logger, exc: Exception, context: str = "") -> None:
    """
    Loga exceção com contexto.
    
    Args:
        logger: Logger a usar
        exc: Exceção
        context: Contexto adicional
    """
    if context:
        logger.error(f"❌ {context}: {type(exc).__name__}: {exc}")
    else:
        logger.error(f"❌ {type(exc).__name__}: {exc}")
    
    logger.debug("Stack trace:", exc_info=True)


class LoggerMixin:
    """Mixin para adicionar logger a classes."""
    
    @property
    def logger(self) -> logging.Logger:
        """Retorna logger da classe."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(self.__class__.__name__)
        return self._logger
