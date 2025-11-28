import os
from aiohttp import web
import logging

logger = logging.getLogger("HealthEndpoint")


class HealthEndpoint:
    def __init__(self, health_monitor, port: int = None):
        self.health_monitor = health_monitor
        self.port = port or int(os.getenv("HEALTH_PORT", 8080))
        self.app = web.Application()
        self.app.router.add_get('/health', self.health_handler)
        self.app.router.add_get('/status', self.status_handler)

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        logger.info(f"Health endpoint em http://0.0.0.0:{self.port}/health")

    async def health_handler(self, request):
        status = self.health_monitor.get_status()
        if status['is_healthy']:
            return web.json_response({'status': 'healthy'}, status=200)
        else:
            return web.json_response({'status': 'unhealthy', 'details': status}, status=503)

    async def status_handler(self, request):
        return web.json_response(self.health_monitor.get_status())
