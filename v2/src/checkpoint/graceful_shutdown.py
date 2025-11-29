"""
Graceful Shutdown Handler for MEGA Historical Simulator.

Handles signals (SIGINT/SIGTERM) to perform clean shutdown with checkpoint save.
"""

import asyncio
import logging
import signal
import sys
import threading
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class GracefulShutdown:
    """
    Handles graceful shutdown with checkpoint save.
    
    Features:
    - Captures SIGINT (Ctrl+C) and SIGTERM
    - Pauses simulation
    - Saves checkpoint
    - Displays progress summary
    """
    
    def __init__(
        self,
        checkpoint_manager: Any = None,
        save_callback: Optional[Callable[[], None]] = None
    ):
        """
        Initialize GracefulShutdown.
        
        Args:
            checkpoint_manager: CheckpointManager instance for saving
            save_callback: Optional callback to save state before exit
        """
        self.checkpoint_manager = checkpoint_manager
        self.save_callback = save_callback
        
        self._shutdown_requested = False
        self._original_sigint = None
        self._original_sigterm = None
        
        # State for display
        self.current_index = 0
        self.total_candles = 0
        self.balance = 0.0
        self.trades_count = 0
        self.start_time = datetime.now()
    
    @property
    def shutdown_requested(self) -> bool:
        """Check if shutdown was requested."""
        return self._shutdown_requested
    
    def register_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        # Store original handlers
        self._original_sigint = signal.signal(signal.SIGINT, self._handle_signal)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)
        logger.debug("Registered graceful shutdown handlers")
    
    def unregister_handlers(self) -> None:
        """Restore original signal handlers."""
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)
        logger.debug("Unregistered graceful shutdown handlers")
    
    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle interrupt signal."""
        if self._shutdown_requested:
            # Second interrupt - force exit
            print("\n⚠️  Force exit. Exiting without saving...")
            sys.exit(1)
        
        self._shutdown_requested = True
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        
        print(f"\n\n^C")
        print("⏸️  Pausando simulação...")
        
        # Execute save callback if provided
        if self.save_callback is not None:
            try:
                self.save_callback()
            except Exception as e:
                logger.error(f"Error in save callback: {e}")
        
        # Display summary
        self._display_summary()
    
    def _display_summary(self) -> None:
        """Display progress summary on shutdown."""
        elapsed = datetime.now() - self.start_time
        progress_pct = (self.current_index / self.total_candles * 100) if self.total_candles > 0 else 0
        
        # Format elapsed time
        hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        elapsed_str = f"{hours}h {minutes}min" if hours > 0 else f"{minutes}min {seconds}s"
        
        print(f"""
📊 Progresso: {progress_pct:.1f}% completo ({self.current_index:,} / {self.total_candles:,} candles)
💰 Balance: ${self.balance:,.2f}
📈 Trades: {self.trades_count:,}
⏱️  Tempo rodando: {elapsed_str}

Para resumir depois: python v2/scripts/mega_historical_simulator.py --resume
""")
    
    def update_state(
        self,
        current_index: int,
        total_candles: int,
        balance: float,
        trades_count: int
    ) -> None:
        """
        Update state for summary display.
        
        Args:
            current_index: Current candle index
            total_candles: Total candles to process
            balance: Current balance
            trades_count: Number of trades executed
        """
        self.current_index = current_index
        self.total_candles = total_candles
        self.balance = balance
        self.trades_count = trades_count


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "2h 34min"
    """
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(int(td.total_seconds()), 3600)
    minutes, secs = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}h {minutes}min"
    elif minutes > 0:
        return f"{minutes}min {secs}s"
    else:
        return f"{secs}s"


def prompt_resume_or_new(checkpoint_info: dict) -> str:
    """
    Display checkpoint info and prompt user for action.
    
    Args:
        checkpoint_info: Dictionary with checkpoint information
        
    Returns:
        User choice: 'R' (resume), 'N' (new), 'V' (view), 'Q' (quit)
    """
    print("""
🔄 Checkpoint encontrado!
""")
    
    created_at = checkpoint_info.get('created_at', 'Unknown')
    progress_pct = checkpoint_info.get('progress_pct', 0)
    candle_index = checkpoint_info.get('candle_index', 0)
    total_candles = checkpoint_info.get('total_candles', 0)
    balance = checkpoint_info.get('balance', 0)
    trades = checkpoint_info.get('trades', 0)
    
    print(f"""📅 Último save: {created_at}
📊 Progresso: {candle_index:,} / {total_candles:,} candles ({progress_pct:.1f}%)
💰 Balance atual: ${balance:,.2f}
📈 Trades executados: {trades}

Opções:
[R] Resumir de onde parou
[N] Começar nova simulação (apaga checkpoint)
[V] Ver detalhes do checkpoint
[Q] Sair

""")
    
    while True:
        try:
            choice = input("Escolha: ").strip().upper()
            if choice in ['R', 'N', 'V', 'Q']:
                return choice
            print("Opção inválida. Digite R, N, V ou Q.")
        except (KeyboardInterrupt, EOFError):
            return 'Q'


def display_checkpoint_details(checkpoint: Any) -> None:
    """
    Display detailed checkpoint information.
    
    Args:
        checkpoint: SimulationCheckpoint object
    """
    print(f"""
═══════════════════════════════════════════════════════════════
                 DETALHES DO CHECKPOINT
═══════════════════════════════════════════════════════════════

📌 ID: {checkpoint.checkpoint_id}
📅 Criado em: {checkpoint.created_at}
🔑 Config Hash: {checkpoint.config_hash[:16]}...

─────────────────────────────────────────────────────────────────
📊 PROGRESSO
─────────────────────────────────────────────────────────────────
   Candles: {checkpoint.current_index:,} / {checkpoint.total_candles:,}
   Progresso: {checkpoint.progress_pct:.2f}%
   Última candle: {checkpoint.last_candle_timestamp}

─────────────────────────────────────────────────────────────────
💰 ESTADO FINANCEIRO
─────────────────────────────────────────────────────────────────
   Balance: ${checkpoint.balance:,.2f}
   Trades executados: {len(checkpoint.trades)}
   Posição aberta: {"Sim" if checkpoint.position else "Não"}

─────────────────────────────────────────────────────────────────
🧠 ONLINE LEARNING
─────────────────────────────────────────────────────────────────
   Modelo salvo: {"Sim" if checkpoint.online_model_state else "Não"}
   Amostras vistas: {checkpoint.n_samples_seen:,}
   Accuracy: {checkpoint.model_accuracy:.2%}
   Drifts detectados: {len(checkpoint.drift_events)}

─────────────────────────────────────────────────────────────────
⏱️ TEMPO
─────────────────────────────────────────────────────────────────
   Tempo decorrido: {format_duration(checkpoint.elapsed_time_seconds)}

═══════════════════════════════════════════════════════════════
""")
