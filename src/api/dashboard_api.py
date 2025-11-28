"""
Web Dashboard API for SNAME-MR Trading Bot
FastAPI + WebSocket for real-time updates
Runs on http://localhost:8080
"""
import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from collections import deque
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn

logger = logging.getLogger("DashboardAPI")


class WebDashboard:
    """Web Dashboard with FastAPI + WebSocket for real-time trading data."""
    
    def __init__(self, event_bus, connector=None, orchestrator=None, meta_controller=None, port: int = 8080):
        self.event_bus = event_bus
        self.connector = connector
        self.orchestrator = orchestrator
        self.meta_controller = meta_controller
        self.port = port
        
        # Data stores
        self.current_price = 0.0
        self.price_history: deque = deque(maxlen=100)
        self.price_24h_change = 0.0
        self.volume_24h = 0.0
        self.start_price = 0.0
        
        # Position & Balance
        self.position = None
        self.balance_data = {
            "total_usdt": 0.0,
            "available": 0.0,
            "margin_balance": 0.0,
            "unrealized_pnl": 0.0,
            "daily_pnl": 0.0
        }
        
        # Strategy states
        self.strategy_states: Dict[str, Dict] = {}
        
        # Meta-controller state
        self.meta_state = {
            "decision": "HOLD",
            "confidence": "NONE",
            "agreeing_count": 0,
            "score": 0.0,
            "total_decisions": 0
        }
        
        # Trade history
        self.trade_history: deque = deque(maxlen=20)
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0.0
        
        # Liquidations
        self.liquidations: deque = deque(maxlen=10)
        self.total_liquidation_volume = 0.0
        
        # Logs
        self.logs: deque = deque(maxlen=50)
        
        # WebSocket clients
        self.active_connections: List[WebSocket] = []
        
        # Server reference for proper shutdown
        self._server: Optional[uvicorn.Server] = None
        self._refresh_task: Optional[asyncio.Task] = None
        
        # FastAPI app
        self.app = FastAPI(title="SNAME-MR Dashboard")
        self._setup_routes()
        
        # Start time
        self.start_time = datetime.now()
        
    def _setup_routes(self):
        static_dir = Path(__file__).parent / "static"
        
        @self.app.get("/", response_class=HTMLResponse)
        async def get_dashboard():
            html_path = static_dir / "index.html"
            if html_path.exists():
                return FileResponse(html_path)
            return HTMLResponse("<h1>Dashboard not found</h1>")
        
        @self.app.get("/api/status")
        async def get_status():
            return await self._get_full_status()
        
        @self.app.get("/api/balance")
        async def get_balance():
            return await self._fetch_balance()
        
        @self.app.get("/api/position")
        async def get_position():
            return await self._fetch_position()
        
        # New Binance-style endpoints
        @self.app.get("/api/positions")
        async def get_positions():
            """Busca posições abertas (symbol, size, entry_price, mark_price, pnl, roe%)"""
            return await self._fetch_positions()
        
        @self.app.get("/api/open-orders")
        async def get_open_orders():
            """Busca ordens abertas (SL, TP pendentes)"""
            return await self._fetch_open_orders()
        
        @self.app.get("/api/order-history")
        async def get_order_history():
            """Busca histórico de ordens"""
            return await self._fetch_order_history()
        
        @self.app.get("/api/trade-history")
        async def get_trade_history():
            """Busca histórico de trades executados"""
            return await self._fetch_trade_history()
        
        @self.app.get("/api/transactions")
        async def get_transactions():
            """Busca histórico de transações"""
            return await self._fetch_transactions()
        
        @self.app.get("/api/assets")
        async def get_assets():
            """Busca saldo da carteira (USDT, BTC, etc)"""
            return await self._fetch_assets()
        
        @self.app.get("/api/account")
        async def get_account():
            """Busca resumo da conta (margin balance, wallet balance, unrealized pnl)"""
            return await self._fetch_account_summary()
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self._handle_websocket(websocket)
        
        if static_dir.exists():
            self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    async def _handle_websocket(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total clients: {len(self.active_connections)}")
        
        try:
            initial_data = await self._get_full_status()
            await websocket.send_json({"type": "initial", "data": initial_data})
            
            while True:
                try:
                    data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                    if data == "ping":
                        await websocket.send_text("pong")
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "ping"})
                 
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
    
    async def _broadcast(self, message: dict):
        if not self.active_connections:
            return
            
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)
        
        for conn in dead_connections:
            if conn in self.active_connections:
                self.active_connections.remove(conn)
    
    async def _get_full_status(self) -> dict:
        uptime = datetime.now() - self.start_time
        
        strategies = {}
        if self.orchestrator:
            for strategy in self.orchestrator.strategies:
                if hasattr(strategy, 'get_status'):
                    strategies[strategy.name] = strategy.get_status()
            for strategy in self.orchestrator.liquidation_strategies:
                if hasattr(strategy, 'get_status'):
                    strategies[strategy.name] = strategy.get_status()
            for strategy in self.orchestrator.orderbook_strategies:
                if hasattr(strategy, 'get_status'):
                    strategies[strategy.name] = strategy.get_status()
        
        meta_status = self.meta_state.copy()
        if self.meta_controller:
            mc_status = self.meta_controller.get_status()
            meta_status["total_decisions"] = mc_status.get("total_decisions", 0)
        
        win_rate = 0.0
        total_trades = self.win_count + self.loss_count
        if total_trades > 0:
            win_rate = (self.win_count / total_trades) * 100
        
        is_demo = getattr(self.connector, 'demo_mode', False) if self.connector else False;
        
        return {
            "price": {
                "current": self.current_price,
                "change_24h": self.price_24h_change,
                "volume_24h": self.volume_24h,
                "history": list(self.price_history)
            },
            "balance": self.balance_data,
            "position": self.position,
            "strategies": strategies,
            "meta_controller": meta_status,
            "trades": {
                "history": list(self.trade_history),
                "win_rate": win_rate,
                "total_pnl": self.total_pnl,
                "win_count": self.win_count,
                "loss_count": self.loss_count
            },
            "liquidations": {
                "recent": list(self.liquidations),
                "total_volume": self.total_liquidation_volume
            },
            "logs": list(self.logs),
            "system": {
                "uptime": str(uptime).split('.')[0],
                "connected_clients": len(self.active_connections),
                "demo_mode": is_demo
            }
        }
    
    async def _fetch_balance(self) -> dict:
        if not self.connector or not self.connector.client:
            return self.balance_data;
        
        try:
            balances = await self.connector.client.futures_account_balance()
            for b in balances:
                if b.get('asset') == 'USDT':
                    self.balance_data["total_usdt"] = float(b.get('balance', 0))
                    self.balance_data["available"] = float(b.get('availableBalance', 0))
                    break;
            
            account = await self.connector.client.futures_account()
            self.balance_data["margin_balance"] = float(account.get('totalMarginBalance', 0))
            self.balance_data["unrealized_pnl"] = float(account.get('totalUnrealizedProfit', 0))
            
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
        
        return self.balance_data
    
    async def _fetch_position(self) -> Optional[dict]:
        if not self.connector or not self.connector.client:
            return self.position;
        
        try:
            positions = await self.connector.client.futures_position_information(symbol='BTCUSDT')
            for pos in positions:
                qty = float(pos.get('positionAmt', 0))
                if qty != 0:
                    entry_price = float(pos.get('entryPrice', 0))
                    unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                    
                    self.position = {
                        "has_position": True,
                        "symbol": pos.get('symbol'),
                        "side": "LONG" if qty > 0 else "SHORT",
                        "quantity": abs(qty),
                        "entry_price": entry_price,
                        "mark_price": float(pos.get('markPrice', 0)),
                        "liquidation_price": float(pos.get('liquidationPrice', 0)),
                        "unrealized_pnl": unrealized_pnl,
                        "pnl_percent": (unrealized_pnl / (entry_price * abs(qty)) * 100) if entry_price > 0 and abs(qty) > 0 else 0,
                        "leverage": int(pos.get('leverage', 1))
                    }
                    return self.position;
            
            self.position = {"has_position": False};
            
        except Exception as e:
            logger.error(f"Error fetching position: {e}")
        
        return self.position
    
    async def _fetch_positions(self) -> List[dict]:
        """Busca todas as posições abertas no formato Binance."""
        if not self.connector or not self.connector.client:
            return []
        
        try:
            positions = await self.connector.client.futures_position_information(symbol='BTCUSDT')
            result = []
            for pos in positions:
                qty = float(pos.get('positionAmt', 0))
                if qty != 0:
                    entry_price = float(pos.get('entryPrice', 0))
                    mark_price = float(pos.get('markPrice', 0))
                    unrealized_pnl = float(pos.get('unRealizedProfit', 0))
                    
                    # Calcular ROE%
                    roe_percent = 0.0
                    if entry_price > 0 and qty != 0:
                        notional = entry_price * abs(qty)
                        if notional > 0:
                            roe_percent = (unrealized_pnl / notional) * 100
                    
                    result.append({
                        'symbol': pos.get('symbol'),
                        'side': 'LONG' if qty > 0 else 'SHORT',
                        'size': abs(qty),
                        'entry_price': entry_price,
                        'mark_price': mark_price,
                        'liquidation_price': float(pos.get('liquidationPrice', 0)),
                        'unrealized_pnl': unrealized_pnl,
                        'roe_percent': roe_percent,
                        'leverage': int(pos.get('leverage', 1)),
                        'margin_type': pos.get('marginType', 'cross')
                    })
            return result
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return []
    
    async def _fetch_open_orders(self) -> List[dict]:
        """Busca ordens abertas (SL, TP pendentes)."""
        if not self.connector or not self.connector.client:
            return []
        
        try:
            orders = await self.connector.client.futures_get_open_orders(symbol='BTCUSDT')
            result = []
            for order in orders:
                result.append({
                    'order_id': order.get('orderId'),
                    'symbol': order.get('symbol'),
                    'side': order.get('side'),
                    'type': order.get('type'),
                    'price': float(order.get('price', 0)),
                    'stop_price': float(order.get('stopPrice', 0)),
                    'quantity': float(order.get('origQty', 0)),
                    'status': order.get('status'),
                    'time': order.get('time')
                })
            return result
        except Exception as e:
            logger.error(f"Error fetching open orders: {e}")
            return []
    
    async def _fetch_order_history(self, limit: int = 50) -> List[dict]:
        """Busca histórico de ordens."""
        if not self.connector or not self.connector.client:
            return []
        
        try:
            orders = await self.connector.client.futures_get_all_orders(symbol='BTCUSDT', limit=limit)
            result = []
            for order in orders:
                result.append({
                    'order_id': order.get('orderId'),
                    'symbol': order.get('symbol'),
                    'side': order.get('side'),
                    'type': order.get('type'),
                    'price': float(order.get('price', 0)),
                    'avg_price': float(order.get('avgPrice', 0)),
                    'quantity': float(order.get('origQty', 0)),
                    'executed_qty': float(order.get('executedQty', 0)),
                    'status': order.get('status'),
                    'time': order.get('time'),
                    'update_time': order.get('updateTime')
                })
            return sorted(result, key=lambda x: x.get('update_time', 0), reverse=True)
        except Exception as e:
            logger.error(f"Error fetching order history: {e}")
            return []
    
    async def _fetch_trade_history(self, limit: int = 50) -> List[dict]:
        """Busca histórico de trades executados."""
        if not self.connector or not self.connector.client:
            return []
        
        try:
            trades = await self.connector.client.futures_account_trades(symbol='BTCUSDT', limit=limit)
            result = []
            for trade in trades:
                result.append({
                    'trade_id': trade.get('id'),
                    'order_id': trade.get('orderId'),
                    'symbol': trade.get('symbol'),
                    'side': trade.get('side'),
                    'price': float(trade.get('price', 0)),
                    'qty': float(trade.get('qty', 0)),
                    'realized_pnl': float(trade.get('realizedPnl', 0)),
                    'commission': float(trade.get('commission', 0)),
                    'commission_asset': trade.get('commissionAsset'),
                    'time': trade.get('time'),
                    'buyer': trade.get('buyer', False),
                    'maker': trade.get('maker', False)
                })
            return sorted(result, key=lambda x: x.get('time', 0), reverse=True)
        except Exception as e:
            logger.error(f"Error fetching trade history: {e}")
            return []
    
    async def _fetch_transactions(self, limit: int = 50) -> List[dict]:
        """Busca histórico de transações."""
        if not self.connector or not self.connector.client:
            return []
        
        try:
            transactions = await self.connector.client.futures_income_history(symbol='BTCUSDT', limit=limit)
            result = []
            for tx in transactions:
                result.append({
                    'symbol': tx.get('symbol'),
                    'type': tx.get('incomeType'),
                    'income': float(tx.get('income', 0)),
                    'asset': tx.get('asset'),
                    'time': tx.get('time'),
                    'info': tx.get('info', '')
                })
            return sorted(result, key=lambda x: x.get('time', 0), reverse=True)
        except Exception as e:
            logger.error(f"Error fetching transactions: {e}")
            return []
    
    async def _fetch_assets(self) -> List[dict]:
        """Busca saldos da carteira."""
        if not self.connector or not self.connector.client:
            return []
        
        try:
            balances = await self.connector.client.futures_account_balance()
            result = []
            for bal in balances:
                balance = float(bal.get('balance', 0))
                if balance > 0:
                    result.append({
                        'asset': bal.get('asset'),
                        'balance': balance,
                        'available': float(bal.get('availableBalance', 0)),
                        'cross_wallet': float(bal.get('crossWalletBalance', 0)),
                        'cross_unrealized_pnl': float(bal.get('crossUnPnl', 0))
                    })
            return result
        except Exception as e:
            logger.error(f"Error fetching assets: {e}")
            return []
    
    async def _fetch_account_summary(self) -> dict:
        """Busca resumo da conta."""
        if not self.connector or not self.connector.client:
            return {}
        
        try:
            account = await self.connector.client.futures_account()
            return {
                'total_wallet_balance': float(account.get('totalWalletBalance', 0)),
                'total_margin_balance': float(account.get('totalMarginBalance', 0)),
                'total_unrealized_profit': float(account.get('totalUnrealizedProfit', 0)),
                'available_balance': float(account.get('availableBalance', 0)),
                'max_withdraw_amount': float(account.get('maxWithdrawAmount', 0)),
                'total_position_initial_margin': float(account.get('totalPositionInitialMargin', 0)),
                'total_open_order_initial_margin': float(account.get('totalOpenOrderInitialMargin', 0)),
                'can_trade': account.get('canTrade', False),
                'can_deposit': account.get('canDeposit', False),
                'can_withdraw': account.get('canWithdraw', False)
            }
        except Exception as e:
            logger.error(f"Error fetching account summary: {e}")
            return {}
    
    async def _on_market_data(self, data: dict):
        try:
            price = float(data.get('price', 0))
            if price > 0:
                self.current_price = price;
                
                if self.start_price == 0:
                    self.start_price = price;
                
                if self.start_price > 0:
                    self.price_24h_change = ((price - self.start_price) / self.start_price) * 100;
                
                timestamp = data.get('timestamp', int(time.time() * 1000));
                self.price_history.append({
                    "time": timestamp,
                    "price": price
                });
                
                await self._broadcast({
                    "type": "price",
                    "data": {
                        "current": price,
                        "change_24h": self.price_24h_change,
                        "timestamp": timestamp
                    }
                });
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
    
    async def _on_trade_signal(self, data: dict):
        try:
            strategy = data.get('strategy', 'Unknown')
            action = data.get('action', 'HOLD')
            reason = data.get('reason', '')
            price = data.get('price', self.current_price);
            
            self.strategy_states[strategy] = {
                "action": action,
                "reason": reason,
                "price": price,
                "timestamp": time.time()
            };
            
            self._add_log("INFO", f"[{strategy}] {action}: {reason}");
            
            await self._broadcast({
                "type": "signal",
                "data": {
                    "strategy": strategy,
                    "action": action,
                    "reason": reason,
                    "price": price
                }
            });
        except Exception as e:
            logger.error(f"Error processing trade signal: {e}")
    
    async def _on_position_update(self, data: dict):
        try:
            self.position = data;
            await self._broadcast({
                "type": "position",
                "data": data
            });
        except Exception as e:
            logger.error(f"Error processing position update: {e}")
    
    async def _on_liquidation(self, data: dict):
        try:
            liquidation = {
                "side": data.get('side'),
                "amount_usd": float(data.get('amount_usd', 0)),
                "price": float(data.get('price', 0)),
                "timestamp": data.get('timestamp', int(time.time() * 1000))
            };
            
            self.liquidations.append(liquidation);
            self.total_liquidation_volume += liquidation["amount_usd"];
            
            amount = liquidation["amount_usd"];
            emoji = "🔴" if liquidation["side"] == "SELL" else "🟢";
            self._add_log("WARNING", f"💀 Liquidação {liquidation['side']}: ${amount:,.0f} {emoji}");
            
            await self._broadcast({
                "type": "liquidation",
                "data": liquidation
            });
        except Exception as e:
            logger.error(f"Error processing liquidation: {e}")
    
    def _add_log(self, level: str, message: str):
        log_entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "message": message
        }
        self.logs.append(log_entry);
    
    async def start(self):
        self.event_bus.subscribe('market_data', self._on_market_data);
        self.event_bus.subscribe('trade_signal', self._on_trade_signal);
        self.event_bus.subscribe('position_update', self._on_position_update);
        self.event_bus.subscribe('liquidation_data', self._on_liquidation);
        
        logger.info(f"📊 Dashboard subscribed to EventBus topics");
        
        asyncio.create_task(self._fetch_balance());
        asyncio.create_task(self._fetch_position());
        
        self._refresh_task = asyncio.create_task(self._periodic_refresh());
        
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="warning"
        );
        self._server = uvicorn.Server(config);
        
        logger.info(f"🌐 Dashboard starting on http://localhost:{self.port}");
        self._add_log("INFO", f"Dashboard iniciado na porta {self.port}");
        
        asyncio.create_task(self._server.serve());
    
    async def _periodic_refresh(self):
        while True:
            await asyncio.sleep(10);
            try:
                await self._fetch_balance();
                await self._fetch_position();
                
                strategies = {};
                if self.orchestrator:
                    for strategy in self.orchestrator.strategies:
                        if hasattr(strategy, 'get_status'):
                            strategies[strategy.name] = strategy.get_status();
                    for strategy in self.orchestrator.liquidation_strategies:
                        if hasattr(strategy, 'get_status'):
                            strategies[strategy.name] = strategy.get_status();
                    for strategy in self.orchestrator.orderbook_strategies:
                        if hasattr(strategy, 'get_status'):
                            strategies[strategy.name] = strategy.get_status();
                
                meta_status = self.meta_state.copy();
                if self.meta_controller:
                    mc_status = self.meta_controller.get_status();
                    meta_status["total_decisions"] = mc_status.get("total_decisions", 0);
                
                await self._broadcast({
                    "type": "refresh",
                    "data": {
                        "balance": self.balance_data,
                        "position": self.position,
                        "strategies": strategies,
                        "meta_controller": meta_status
                    }
                });
            except Exception as e:
                logger.error(f"Error in periodic refresh: {e}");
    
    async def stop(self):
        logger.info("🛑 Dashboard stopping...");
        
        if self._refresh_task:
            self._refresh_task.cancel();
            try:
                await self._refresh_task;
            except asyncio.CancelledError:
                pass;
        
        if self._server:
            self._server.should_exit = True;
        
        for connection in self.active_connections:
            try:
                await connection.close();
            except Exception:
                pass;
        self.active_connections.clear();