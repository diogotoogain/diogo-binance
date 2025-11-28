from abc import ABC, abstractmethod
import logging

class BaseStrategy(ABC):
    """
    O DNA de qualquer estratégia.
    Define que toda estratégia precisa ter um nome e saber processar um tick.
    """
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"Strategy-{name}")

    @abstractmethod
    async def on_tick(self, tick_data: dict):
        """
        Recebe o dado do mercado (Tick) e retorna uma decisão.
        """
        pass

    def log(self, message: str):
        self.logger.info(f"[{self.name}] {message}")