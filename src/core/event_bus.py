import asyncio
import logging
from typing import Dict, List, Callable, Any, Awaitable

logger = logging.getLogger("EventBus")

class EventBus:
    def __init__(self):
        self.topics: Dict[str, List[Callable[[Any], Awaitable[None]]]] = {
            'market_data': [],
            'trade_signal': [],
            'system_log': []
        }
        self.queue = asyncio.Queue()
        self.is_running = False

    async def start(self):
        self.is_running = True
        logger.info("🧠 Event Bus iniciado e ouvindo...")
        while self.is_running:
            topic, data = await self.queue.get()
            if topic in self.topics:
                for handler in self.topics[topic]:
                    asyncio.create_task(self._safe_execute(handler, data))
            self.queue.task_done()

    async def _safe_execute(self, handler, data):
        try:
            await handler(data)
        except Exception as e:
            logger.error(f"❌ Erro ao processar evento no Bus: {e}")

    async def publish(self, topic: str, data: Any):
        await self.queue.put((topic, data))

    def subscribe(self, topic: str, handler: Callable[[Any], Awaitable[None]]):
        if topic not in self.topics:
            self.topics[topic] = []
        self.topics[topic].append(handler)
        logger.info(f"👂 Novo assinante registrado no tópico: {topic}")

    async def stop(self):
        self.is_running = False
        logger.info("🛑 Event Bus parando...")