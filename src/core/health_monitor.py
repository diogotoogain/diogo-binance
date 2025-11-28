import os
import asyncio
import logging
from datetime import datetime
from typing import Dict
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("HealthMonitor")


class HealthMonitor:
    def __init__(self, event_bus, connector, check_interval: int = None):
        self.event_bus = event_bus
        self.connector = connector
        self.check_interval = check_interval or int(os.getenv("HEALTH_CHECK_INTERVAL", 60))
        
        self.last_tick_time = datetime.now()
        self.last_health_check = datetime.now()
        self.ticks_received = 0
        self.errors_count = 0
        self.restarts_count = 0
        self.is_healthy = True
        self.running = False
        
        self.max_tick_gap = int(os.getenv("MAX_TICK_GAP_SECONDS", 30))
        self.max_errors = int(os.getenv("MAX_ERRORS_BEFORE_RESTART", 10))

    async def start(self):
        self.running = True
        self.event_bus.subscribe('market_data', self._on_tick)
        asyncio.create_task(self._health_loop())
        logger.info(f"Health Monitor iniciado | Intervalo: {self.check_interval}s")

    async def stop(self):
        self.running = False

    async def _on_tick(self, data):
        self.last_tick_time = datetime.now()
        self.ticks_received += 1

    async def _health_loop(self):
        while self.running:
            await asyncio.sleep(self.check_interval)
            await self._check_health()

    async def _check_health(self):
        self.last_health_check = datetime.now()
        issues = []
        
        tick_gap = (datetime.now() - self.last_tick_time).total_seconds()
        if tick_gap > self.max_tick_gap:
            issues.append(f"Sem ticks ha {tick_gap:.0f}s")
            self.errors_count += 1
        
        if self.errors_count >= self.max_errors:
            issues.append(f"Muitos erros: {self.errors_count}")
            await self._attempt_recovery()
        
        if issues:
            self.is_healthy = False
            logger.warning(f"HEALTH CHECK FAILED: {issues}")
        else:
            self.is_healthy = True
            self.errors_count = 0
            logger.debug("Health check OK")

    async def _attempt_recovery(self):
        logger.warning("Tentando recovery...")
        self.restarts_count += 1
        self.errors_count = 0
        
        try:
            await self.connector.close()
            await asyncio.sleep(5)
            await self.connector.connect()
            symbol = os.getenv("SYMBOL", "BTCUSDT")
            asyncio.create_task(self.connector.start_streams(symbol))
            logger.info("Recovery bem sucedido!")
        except Exception as e:
            logger.error(f"Recovery falhou: {e}")

    def get_status(self) -> Dict:
        return {
            'is_healthy': self.is_healthy,
            'last_tick': self.last_tick_time.isoformat(),
            'ticks_received': self.ticks_received,
            'errors_count': self.errors_count,
            'restarts_count': self.restarts_count,
            'uptime_seconds': (datetime.now() - self.last_health_check).total_seconds()
        }
