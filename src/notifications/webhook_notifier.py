import os
import logging
import aiohttp
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("WebhookNotifier")


class WebhookNotifier:
    def __init__(self, webhook_url: str = None, enabled: bool = None):
        self.webhook_url = webhook_url or os.getenv("WEBHOOK_URL", "")
        self.enabled = enabled if enabled is not None else os.getenv("NOTIFICATIONS_ENABLED", "false").lower() == "true"
        self.bot_name = os.getenv("BOT_NAME", "SNAME-MR Bot")
        
        if self.enabled and self.webhook_url:
            logger.info("Notificacoes ativadas")
        else:
            logger.info("Notificacoes desativadas")

    async def send(self, title: str, message: str, color: int = 0x00ff00, fields: list = None):
        if not self.enabled or not self.webhook_url:
            return False
        
        embed = {
            "title": title,
            "description": message,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": self.bot_name}
        }
        
        if fields:
            embed["fields"] = fields
        
        payload = {"embeds": [embed]}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.webhook_url, json=payload) as resp:
                    return resp.status == 204
        except Exception as e:
            logger.error(f"Erro webhook: {e}")
            return False

    async def notify_trade_open(self, side: str, symbol: str, price: float, quantity: float, sl: float, tp: float):
        color = 0x00ff00 if side == 'BUY' else 0xff0000
        await self.send(
            title=f"TRADE ABERTO - {side}",
            message=f"{symbol} @ ${price:,.2f}",
            color=color,
            fields=[
                {"name": "Quantidade", "value": str(quantity), "inline": True},
                {"name": "Stop Loss", "value": f"${sl:,.2f}", "inline": True},
                {"name": "Take Profit", "value": f"${tp:,.2f}", "inline": True}
            ]
        )

    async def notify_trade_close(self, symbol: str, entry: float, exit_price: float, pnl: float, reason: str):
        color = 0x00ff00 if pnl > 0 else 0xff0000
        pnl_pct = (pnl / entry) * 100
        await self.send(
            title=f"TRADE FECHADO - {reason}",
            message=f"{symbol} | P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)",
            color=color,
            fields=[
                {"name": "Entrada", "value": f"${entry:,.2f}", "inline": True},
                {"name": "Saida", "value": f"${exit_price:,.2f}", "inline": True}
            ]
        )

    async def notify_daily_summary(self, total_trades: int, wins: int, pnl: float, balance: float):
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        color = 0x00ff00 if pnl > 0 else 0xff0000
        await self.send(
            title="RESUMO DIARIO",
            message=f"P&L: ${pnl:,.2f}",
            color=color,
            fields=[
                {"name": "Trades", "value": str(total_trades), "inline": True},
                {"name": "Win Rate", "value": f"{win_rate:.1f}%", "inline": True},
                {"name": "Saldo", "value": f"${balance:,.2f}", "inline": True}
            ]
        )
