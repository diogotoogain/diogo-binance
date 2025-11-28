"""
LOGGER DE TRADES: Salva histórico de trades em JSON
"""
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("TradeLogger")


class TradeLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.trades_file = self.log_dir / f"trades_{datetime.now().strftime('%Y%m%d')}.json"
        self.signals_file = self.log_dir / f"signals_{datetime.now().strftime('%Y%m%d')}.json"

    def log_trade(self, trade: dict):
        trade['logged_at'] = datetime.now().isoformat()
        self._append_json(self.trades_file, trade)

    def log_signal(self, signal: dict):
        signal['logged_at'] = datetime.now().isoformat()
        self._append_json(self.signals_file, signal)

    def _append_json(self, file_path: Path, data: dict):
        try:
            existing = []
            if file_path.exists():
                with open(file_path, 'r') as f:
                    existing = json.load(f)
            existing.append(data)
            with open(file_path, 'w') as f:
                json.dump(existing, f, indent=2)
        except (json.JSONDecodeError, IOError, OSError) as e:
            logger.error(f"Erro ao salvar log em {file_path}: {e}")

    def get_today_trades(self) -> list:
        try:
            if self.trades_file.exists():
                with open(self.trades_file, 'r') as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError, OSError) as e:
            logger.error(f"Erro ao ler trades: {e}")
        return []

    async def on_signal(self, signal: dict):
        """Handler async para eventos de sinal do EventBus"""
        self.log_signal(signal)
