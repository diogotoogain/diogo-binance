"""
Backtest Dashboard Server.

FastAPI-based web server for the backtest analytics dashboard.
Runs on localhost:8081 (separate from trading dashboard on 8080).
"""

import asyncio
import logging
import webbrowser
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

# Dashboard directory
DASHBOARD_DIR = Path(__file__).parent


class BacktestDashboard:
    """
    FastAPI-based backtest analytics dashboard.
    
    Provides:
    - Web interface for viewing backtest results
    - REST API for data access
    - Export functionality
    """
    
    def __init__(
        self,
        data_service: Any = None,
        analytics: Any = None,
        port: int = 8081
    ):
        """
        Initialize BacktestDashboard.
        
        Args:
            data_service: DataService instance with loaded data
            analytics: BacktestAnalytics instance
            port: Port to run server on
        """
        self.data_service = data_service
        self.analytics = analytics
        self.port = port
        
        # Create FastAPI app
        self.app = FastAPI(
            title="MEGA Backtest Analytics",
            description="Analytics dashboard for historical simulation results",
            version="1.0.0"
        )
        
        self._setup_routes()
        self._setup_static_files()
    
    def _setup_static_files(self) -> None:
        """Setup static file serving."""
        static_dir = DASHBOARD_DIR / "static"
        if static_dir.exists():
            self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    
    def _setup_routes(self) -> None:
        """Setup API routes."""
        
        # ========== Page Routes ==========
        
        @self.app.get("/", response_class=HTMLResponse)
        async def home():
            """Serve main dashboard page."""
            template_path = DASHBOARD_DIR / "templates" / "dashboard.html"
            if template_path.exists():
                return FileResponse(template_path)
            return HTMLResponse(self._generate_dashboard_html())
        
        @self.app.get("/trades", response_class=HTMLResponse)
        async def trades_page():
            """Serve trades table page."""
            template_path = DASHBOARD_DIR / "templates" / "trades.html"
            if template_path.exists():
                return FileResponse(template_path)
            return HTMLResponse(self._generate_trades_html())
        
        @self.app.get("/performance", response_class=HTMLResponse)
        async def performance_page():
            """Serve performance analysis page."""
            template_path = DASHBOARD_DIR / "templates" / "performance.html"
            if template_path.exists():
                return FileResponse(template_path)
            return HTMLResponse(self._generate_performance_html())
        
        @self.app.get("/temporal", response_class=HTMLResponse)
        async def temporal_page():
            """Serve temporal analysis page."""
            template_path = DASHBOARD_DIR / "templates" / "temporal.html"
            if template_path.exists():
                return FileResponse(template_path)
            return HTMLResponse(self._generate_temporal_html())
        
        @self.app.get("/risk", response_class=HTMLResponse)
        async def risk_page():
            """Serve risk analysis page."""
            template_path = DASHBOARD_DIR / "templates" / "risk.html"
            if template_path.exists():
                return FileResponse(template_path)
            return HTMLResponse(self._generate_risk_html())
        
        @self.app.get("/model", response_class=HTMLResponse)
        async def model_page():
            """Serve model analysis page."""
            template_path = DASHBOARD_DIR / "templates" / "model.html"
            if template_path.exists():
                return FileResponse(template_path)
            return HTMLResponse(self._generate_model_html())
        
        @self.app.get("/compare", response_class=HTMLResponse)
        async def compare_page():
            """Serve comparison page."""
            template_path = DASHBOARD_DIR / "templates" / "compare.html"
            if template_path.exists():
                return FileResponse(template_path)
            return HTMLResponse(self._generate_compare_html())
        
        # ========== API Routes ==========
        
        @self.app.get("/api/summary")
        async def api_summary():
            """Get summary statistics."""
            if self.data_service is None:
                raise HTTPException(status_code=503, detail="Data not loaded")
            return JSONResponse(self.data_service.get_summary_stats())
        
        @self.app.get("/api/metrics")
        async def api_metrics():
            """Get performance metrics."""
            if self.data_service is None:
                raise HTTPException(status_code=503, detail="Data not loaded")
            return JSONResponse(self.data_service.metrics)
        
        @self.app.get("/api/equity")
        async def api_equity(max_points: int = Query(5000, ge=100, le=50000)):
            """Get equity curve data (downsampled for performance)."""
            if self.data_service is None:
                raise HTTPException(status_code=503, detail="Data not loaded")
            return JSONResponse(self.data_service.get_equity_downsampled(max_points))
        
        @self.app.get("/api/trades")
        async def api_trades(
            page: int = Query(1, ge=1),
            per_page: int = Query(25, ge=1, le=100),
            sort_by: str = Query("entry_time"),
            sort_desc: bool = Query(True),
            direction: Optional[int] = Query(None),
            result_filter: Optional[str] = Query(None),
            start_date: Optional[str] = Query(None),
            end_date: Optional[str] = Query(None)
        ):
            """Get paginated trades list."""
            if self.data_service is None:
                raise HTTPException(status_code=503, detail="Data not loaded")
            
            filters = {}
            if direction is not None:
                filters['direction'] = direction
            if result_filter is not None:
                filters['result_filter'] = result_filter
            if start_date is not None:
                from datetime import datetime
                filters['start_date'] = datetime.fromisoformat(start_date)
            if end_date is not None:
                from datetime import datetime
                filters['end_date'] = datetime.fromisoformat(end_date)
            
            df, total = self.data_service.get_trades_paginated(
                page=page,
                per_page=per_page,
                sort_by=sort_by,
                sort_desc=sort_desc,
                filters=filters if filters else None
            )
            
            # Convert to list of dicts
            trades = df.to_dict('records') if len(df) > 0 else []
            
            # Serialize datetime objects
            for trade in trades:
                for key, value in trade.items():
                    if hasattr(value, 'isoformat'):
                        trade[key] = value.isoformat()
                    elif hasattr(value, 'total_seconds'):
                        trade[key] = str(value)
            
            return JSONResponse({
                'trades': trades,
                'total': total,
                'page': page,
                'per_page': per_page,
                'pages': (total + per_page - 1) // per_page
            })
        
        @self.app.get("/api/analysis/period/{period}")
        async def api_period_analysis(period: str):
            """Get analysis by period (day, week, month, year)."""
            if self.analytics is None:
                raise HTTPException(status_code=503, detail="Analytics not loaded")
            return JSONResponse(self.analytics.analyze_by_period(period))
        
        @self.app.get("/api/analysis/hour")
        async def api_hour_analysis():
            """Get analysis by hour of day."""
            if self.analytics is None:
                raise HTTPException(status_code=503, detail="Analytics not loaded")
            return JSONResponse(self.analytics.analyze_by_hour())
        
        @self.app.get("/api/analysis/day-of-week")
        async def api_dow_analysis():
            """Get analysis by day of week."""
            if self.analytics is None:
                raise HTTPException(status_code=503, detail="Analytics not loaded")
            return JSONResponse(self.analytics.analyze_by_day_of_week())
        
        @self.app.get("/api/analysis/drawdowns")
        async def api_drawdowns(threshold: float = Query(1.0, ge=0.1)):
            """Get drawdown analysis."""
            if self.analytics is None:
                raise HTTPException(status_code=503, detail="Analytics not loaded")
            return JSONResponse(self.analytics.analyze_drawdowns(threshold))
        
        @self.app.get("/api/analysis/var")
        async def api_var():
            """Get Value at Risk analysis."""
            if self.analytics is None:
                raise HTTPException(status_code=503, detail="Analytics not loaded")
            return JSONResponse(self.analytics.calculate_var())
        
        @self.app.get("/api/analysis/sequences")
        async def api_sequences():
            """Get win/loss sequence analysis."""
            if self.analytics is None:
                raise HTTPException(status_code=503, detail="Analytics not loaded")
            return JSONResponse(self.analytics.analyze_sequences())
        
        @self.app.get("/api/analysis/monte-carlo")
        async def api_monte_carlo(simulations: int = Query(1000, ge=100, le=10000)):
            """Run Monte Carlo simulation."""
            if self.analytics is None:
                raise HTTPException(status_code=503, detail="Analytics not loaded")
            return JSONResponse(self.analytics.monte_carlo_simulation(simulations))
        
        @self.app.get("/api/analysis/heatmap")
        async def api_heatmap():
            """Get returns heatmap data."""
            if self.analytics is None:
                raise HTTPException(status_code=503, detail="Analytics not loaded")
            return JSONResponse(self.analytics.generate_returns_heatmap())
        
        @self.app.get("/api/analysis/rolling")
        async def api_rolling(window: int = Query(30, ge=10, le=200)):
            """Get rolling metrics."""
            if self.analytics is None:
                raise HTTPException(status_code=503, detail="Analytics not loaded")
            return JSONResponse(self.analytics.get_rolling_metrics(window))
        
        @self.app.get("/api/drift-events")
        async def api_drift_events():
            """Get drift events."""
            if self.data_service is None:
                raise HTTPException(status_code=503, detail="Data not loaded")
            return JSONResponse(self.data_service.drift_events)
        
        @self.app.get("/api/ol-stats")
        async def api_ol_stats():
            """Get online learning statistics."""
            if self.data_service is None:
                raise HTTPException(status_code=503, detail="Data not loaded")
            return JSONResponse(self.data_service.ol_stats)
        
        # ========== Export Routes ==========
        
        @self.app.get("/api/export/csv")
        async def export_csv():
            """Export trades to CSV."""
            if self.data_service is None:
                raise HTTPException(status_code=503, detail="Data not loaded")
            
            import io
            trades = self.data_service.trades
            if len(trades) == 0:
                raise HTTPException(status_code=404, detail="No trades to export")
            
            csv_buffer = io.StringIO()
            trades.to_csv(csv_buffer, index=False)
            
            return JSONResponse({
                'filename': 'trades_export.csv',
                'content': csv_buffer.getvalue()
            })
    
    def _generate_dashboard_html(self) -> str:
        """Generate main dashboard HTML."""
        return self._base_template("Dashboard", self._dashboard_content())
    
    def _generate_trades_html(self) -> str:
        """Generate trades page HTML."""
        return self._base_template("Trades", self._trades_content())
    
    def _generate_performance_html(self) -> str:
        """Generate performance page HTML."""
        return self._base_template("Performance", self._performance_content())
    
    def _generate_temporal_html(self) -> str:
        """Generate temporal analysis page HTML."""
        return self._base_template("Temporal Analysis", self._temporal_content())
    
    def _generate_risk_html(self) -> str:
        """Generate risk analysis page HTML."""
        return self._base_template("Risk Analysis", self._risk_content())
    
    def _generate_model_html(self) -> str:
        """Generate model analysis page HTML."""
        return self._base_template("Model Analysis", self._model_content())
    
    def _generate_compare_html(self) -> str:
        """Generate comparison page HTML."""
        return self._base_template("Comparisons", self._compare_content())
    
    def _base_template(self, title: str, content: str) -> str:
        """Generate base HTML template."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - MEGA Backtest Analytics</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <nav class="sidebar">
        <div class="logo">
            <span class="logo-icon">📊</span>
            <span class="logo-text">MEGA Analytics</span>
        </div>
        <ul class="nav-links">
            <li><a href="/" class="{'active' if title == 'Dashboard' else ''}">🏠 Dashboard</a></li>
            <li><a href="/trades" class="{'active' if title == 'Trades' else ''}">📋 Trades</a></li>
            <li><a href="/performance" class="{'active' if title == 'Performance' else ''}">📈 Performance</a></li>
            <li><a href="/temporal" class="{'active' if title == 'Temporal Analysis' else ''}">⏰ Temporal</a></li>
            <li><a href="/risk" class="{'active' if title == 'Risk Analysis' else ''}">⚠️ Risk</a></li>
            <li><a href="/model" class="{'active' if title == 'Model Analysis' else ''}">🧠 Model</a></li>
            <li><a href="/compare" class="{'active' if title == 'Comparisons' else ''}">📊 Compare</a></li>
        </ul>
    </nav>
    
    <main class="content">
        <header class="page-header">
            <h1>{title}</h1>
            <div class="header-actions">
                <button id="export-csv" class="btn btn-secondary">📥 Export CSV</button>
                <button id="refresh-data" class="btn btn-primary">🔄 Refresh</button>
            </div>
        </header>
        
        <div class="page-content">
            {content}
        </div>
    </main>
    
    <script src="/static/js/main.js"></script>
    <script src="/static/js/charts.js"></script>
</body>
</html>
"""
    
    def _dashboard_content(self) -> str:
        """Dashboard page content."""
        return """
        <!-- KPI Cards -->
        <div class="kpi-grid">
            <div class="kpi-card" id="kpi-return">
                <div class="kpi-label">Total Return</div>
                <div class="kpi-value" id="total-return">--</div>
            </div>
            <div class="kpi-card" id="kpi-sharpe">
                <div class="kpi-label">Sharpe Ratio</div>
                <div class="kpi-value" id="sharpe-ratio">--</div>
            </div>
            <div class="kpi-card" id="kpi-sortino">
                <div class="kpi-label">Sortino Ratio</div>
                <div class="kpi-value" id="sortino-ratio">--</div>
            </div>
            <div class="kpi-card" id="kpi-drawdown">
                <div class="kpi-label">Max Drawdown</div>
                <div class="kpi-value negative" id="max-drawdown">--</div>
            </div>
            <div class="kpi-card" id="kpi-winrate">
                <div class="kpi-label">Win Rate</div>
                <div class="kpi-value" id="win-rate">--</div>
            </div>
            <div class="kpi-card" id="kpi-pf">
                <div class="kpi-label">Profit Factor</div>
                <div class="kpi-value" id="profit-factor">--</div>
            </div>
        </div>
        
        <!-- Equity Curve -->
        <div class="card">
            <div class="card-header">
                <h2>📈 Equity Curve</h2>
            </div>
            <div class="card-body">
                <canvas id="equity-chart" height="300"></canvas>
            </div>
        </div>
        
        <!-- Summary Stats -->
        <div class="stats-grid">
            <div class="card">
                <div class="card-header">
                    <h2>💰 Balance</h2>
                </div>
                <div class="card-body">
                    <div class="stat-row">
                        <span class="stat-label">Initial Balance</span>
                        <span class="stat-value" id="initial-balance">--</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Final Balance</span>
                        <span class="stat-value" id="final-balance">--</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Total P&L</span>
                        <span class="stat-value" id="total-pnl">--</span>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <h2>📊 Trades</h2>
                </div>
                <div class="card-body">
                    <div class="stat-row">
                        <span class="stat-label">Total Trades</span>
                        <span class="stat-value" id="total-trades">--</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Winners</span>
                        <span class="stat-value positive" id="winning-trades">--</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Losers</span>
                        <span class="stat-value negative" id="losing-trades">--</span>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <h2>📅 Period</h2>
                </div>
                <div class="card-body">
                    <div class="stat-row">
                        <span class="stat-label">Start Date</span>
                        <span class="stat-value" id="start-date">--</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">End Date</span>
                        <span class="stat-value" id="end-date">--</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Data Points</span>
                        <span class="stat-value" id="equity-points">--</span>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _trades_content(self) -> str:
        """Trades page content."""
        return """
        <!-- Filters -->
        <div class="filters-bar">
            <div class="filter-group">
                <label>Direction</label>
                <select id="filter-direction">
                    <option value="">All</option>
                    <option value="1">Long</option>
                    <option value="-1">Short</option>
                </select>
            </div>
            <div class="filter-group">
                <label>Result</label>
                <select id="filter-result">
                    <option value="">All</option>
                    <option value="winners">Winners</option>
                    <option value="losers">Losers</option>
                </select>
            </div>
            <div class="filter-group">
                <label>Per Page</label>
                <select id="filter-per-page">
                    <option value="25">25</option>
                    <option value="50">50</option>
                    <option value="100">100</option>
                </select>
            </div>
            <button id="apply-filters" class="btn btn-primary">Apply Filters</button>
        </div>
        
        <!-- Trades Table -->
        <div class="card">
            <div class="card-header">
                <h2>📋 Trade History</h2>
                <span id="trades-count">-- trades</span>
            </div>
            <div class="card-body">
                <div class="table-container">
                    <table id="trades-table" class="data-table">
                        <thead>
                            <tr>
                                <th data-sort="trade_id">#</th>
                                <th data-sort="entry_time">Entry Time</th>
                                <th data-sort="exit_time">Exit Time</th>
                                <th>Duration</th>
                                <th data-sort="direction">Direction</th>
                                <th data-sort="entry_price">Entry Price</th>
                                <th data-sort="exit_price">Exit Price</th>
                                <th data-sort="size">Size ($)</th>
                                <th data-sort="pnl">P&L</th>
                                <th>P&L %</th>
                                <th>Fees</th>
                                <th data-sort="exit_reason">Exit Reason</th>
                            </tr>
                        </thead>
                        <tbody id="trades-tbody">
                            <tr><td colspan="12" class="loading">Loading trades...</td></tr>
                        </tbody>
                        <tfoot>
                            <tr>
                                <td colspan="8">Totals</td>
                                <td id="total-pnl-footer">--</td>
                                <td id="avg-pnl-footer">--</td>
                                <td id="total-fees-footer">--</td>
                                <td></td>
                            </tr>
                        </tfoot>
                    </table>
                </div>
                
                <!-- Pagination -->
                <div class="pagination">
                    <button id="prev-page" class="btn btn-secondary">&lt; Prev</button>
                    <span id="page-info">Page 1 of 1</span>
                    <button id="next-page" class="btn btn-secondary">Next &gt;</button>
                </div>
            </div>
        </div>
        """
    
    def _performance_content(self) -> str:
        """Performance page content."""
        return """
        <!-- Period Selector -->
        <div class="period-selector">
            <button class="period-btn active" data-period="day">Daily</button>
            <button class="period-btn" data-period="week">Weekly</button>
            <button class="period-btn" data-period="month">Monthly</button>
            <button class="period-btn" data-period="year">Yearly</button>
        </div>
        
        <!-- Performance Table -->
        <div class="card">
            <div class="card-header">
                <h2>📊 Performance by Period</h2>
            </div>
            <div class="card-body">
                <table id="period-table" class="data-table">
                    <thead>
                        <tr>
                            <th>Period</th>
                            <th>Trades</th>
                            <th>Total P&L</th>
                            <th>Avg P&L</th>
                            <th>Win Rate</th>
                            <th>Longs</th>
                            <th>Shorts</th>
                        </tr>
                    </thead>
                    <tbody id="period-tbody">
                        <tr><td colspan="7" class="loading">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Returns Distribution -->
        <div class="card">
            <div class="card-header">
                <h2>📈 Returns Distribution</h2>
            </div>
            <div class="card-body">
                <canvas id="distribution-chart" height="250"></canvas>
            </div>
        </div>
        
        <!-- Heatmap -->
        <div class="card">
            <div class="card-header">
                <h2>📅 Returns Heatmap</h2>
            </div>
            <div class="card-body">
                <div id="heatmap-container"></div>
            </div>
        </div>
        """
    
    def _temporal_content(self) -> str:
        """Temporal analysis page content."""
        return """
        <div class="charts-grid">
            <!-- By Hour -->
            <div class="card">
                <div class="card-header">
                    <h2>⏰ Performance by Hour</h2>
                </div>
                <div class="card-body">
                    <canvas id="hour-chart" height="250"></canvas>
                </div>
            </div>
            
            <!-- By Day of Week -->
            <div class="card">
                <div class="card-header">
                    <h2>📅 Performance by Day</h2>
                </div>
                <div class="card-body">
                    <canvas id="dow-chart" height="250"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Best/Worst Periods -->
        <div class="stats-grid">
            <div class="card">
                <div class="card-header">
                    <h2>🏆 Best Days</h2>
                </div>
                <div class="card-body">
                    <div id="best-days-list"></div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <h2>💔 Worst Days</h2>
                </div>
                <div class="card-body">
                    <div id="worst-days-list"></div>
                </div>
            </div>
        </div>
        """
    
    def _risk_content(self) -> str:
        """Risk analysis page content."""
        return """
        <!-- Risk Metrics -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">VaR 95%</div>
                <div class="kpi-value negative" id="var-95">--</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">VaR 99%</div>
                <div class="kpi-value negative" id="var-99">--</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">CVaR 95%</div>
                <div class="kpi-value negative" id="cvar-95">--</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">CVaR 99%</div>
                <div class="kpi-value negative" id="cvar-99">--</div>
            </div>
        </div>
        
        <!-- Drawdown Chart -->
        <div class="card">
            <div class="card-header">
                <h2>📉 Underwater Curve (Drawdown)</h2>
            </div>
            <div class="card-body">
                <canvas id="drawdown-chart" height="250"></canvas>
            </div>
        </div>
        
        <!-- Drawdown Table -->
        <div class="card">
            <div class="card-header">
                <h2>📊 Drawdown Events</h2>
            </div>
            <div class="card-body">
                <table id="drawdown-table" class="data-table">
                    <thead>
                        <tr>
                            <th>Start</th>
                            <th>Trough</th>
                            <th>End</th>
                            <th>Max DD %</th>
                            <th>Duration</th>
                            <th>Recovery</th>
                        </tr>
                    </thead>
                    <tbody id="drawdown-tbody">
                        <tr><td colspan="6" class="loading">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <!-- Sequences -->
        <div class="stats-grid">
            <div class="card">
                <div class="card-header">
                    <h2>🔥 Win/Loss Streaks</h2>
                </div>
                <div class="card-body">
                    <div class="stat-row">
                        <span class="stat-label">Max Win Streak</span>
                        <span class="stat-value positive" id="max-win-streak">--</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Max Loss Streak</span>
                        <span class="stat-value negative" id="max-loss-streak">--</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Current Streak</span>
                        <span class="stat-value" id="current-streak">--</span>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <div class="card-header">
                    <h2>📊 Streak Distribution</h2>
                </div>
                <div class="card-body">
                    <canvas id="streak-chart" height="200"></canvas>
                </div>
            </div>
        </div>
        """
    
    def _model_content(self) -> str:
        """Model analysis page content."""
        return """
        <!-- Model Stats -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Samples Seen</div>
                <div class="kpi-value" id="samples-seen">--</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Model Accuracy</div>
                <div class="kpi-value" id="model-accuracy">--</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Drifts Detected</div>
                <div class="kpi-value" id="drifts-count">--</div>
            </div>
        </div>
        
        <!-- Rolling Accuracy -->
        <div class="card">
            <div class="card-header">
                <h2>📈 Rolling Performance</h2>
            </div>
            <div class="card-body">
                <canvas id="rolling-chart" height="250"></canvas>
            </div>
        </div>
        
        <!-- Drift Events -->
        <div class="card">
            <div class="card-header">
                <h2>🔄 Drift Events</h2>
            </div>
            <div class="card-body">
                <table id="drift-table" class="data-table">
                    <thead>
                        <tr>
                            <th>Step</th>
                            <th>Timestamp</th>
                            <th>Metrics</th>
                        </tr>
                    </thead>
                    <tbody id="drift-tbody">
                        <tr><td colspan="3" class="loading">Loading...</td></tr>
                    </tbody>
                </table>
            </div>
        </div>
        """
    
    def _compare_content(self) -> str:
        """Comparison page content."""
        return """
        <!-- Comparison Chart -->
        <div class="card">
            <div class="card-header">
                <h2>📊 Strategy vs Buy & Hold</h2>
            </div>
            <div class="card-body">
                <canvas id="compare-chart" height="300"></canvas>
                <div class="compare-stats">
                    <div class="compare-stat">
                        <span class="label">Strategy Return</span>
                        <span class="value" id="strategy-return">--</span>
                    </div>
                    <div class="compare-stat">
                        <span class="label">Buy & Hold Return</span>
                        <span class="value" id="buyhold-return">--</span>
                    </div>
                    <div class="compare-stat">
                        <span class="label">Outperformance</span>
                        <span class="value" id="outperformance">--</span>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Monte Carlo -->
        <div class="card">
            <div class="card-header">
                <h2>🎲 Monte Carlo Simulation</h2>
                <button id="run-monte-carlo" class="btn btn-primary">Run 1000 Simulations</button>
            </div>
            <div class="card-body">
                <canvas id="monte-carlo-chart" height="250"></canvas>
                <div class="monte-carlo-stats">
                    <div class="stat-row">
                        <span class="stat-label">5th Percentile</span>
                        <span class="stat-value" id="mc-5th">--</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">50th Percentile (Median)</span>
                        <span class="stat-value" id="mc-50th">--</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">95th Percentile</span>
                        <span class="stat-value" id="mc-95th">--</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Your Result Rank</span>
                        <span class="stat-value" id="mc-rank">--</span>
                    </div>
                </div>
            </div>
        </div>
        """
    
    async def start(self, open_browser: bool = True) -> None:
        """
        Start the dashboard server.
        
        Args:
            open_browser: If True, open browser to dashboard URL
        """
        import uvicorn
        
        # Bind to localhost only for security (local analytics dashboard)
        config = uvicorn.Config(
            self.app,
            host="127.0.0.1",
            port=self.port,
            log_level="warning"
        )
        server = uvicorn.Server(config)
        
        logger.info(f"🌐 Starting backtest dashboard on http://localhost:{self.port}")
        
        if open_browser:
            # Open browser after short delay
            asyncio.get_event_loop().call_later(
                1.0,
                lambda: webbrowser.open(f"http://localhost:{self.port}")
            )
        
        await server.serve()


def run_dashboard(
    results_dir: str = "v2/results/mega_sim",
    port: int = 8081,
    open_browser: bool = True
) -> None:
    """
    Run the backtest dashboard.
    
    Args:
        results_dir: Directory containing simulation results
        port: Port to run server on
        open_browser: If True, open browser to dashboard
    """
    from .data_service import DataService
    from .analytics import BacktestAnalytics
    
    # Load data
    data_service = DataService(results_dir)
    if not data_service.load_results():
        logger.warning("No results found to display")
    
    analytics = BacktestAnalytics(data_service)
    
    # Create and start dashboard
    dashboard = BacktestDashboard(
        data_service=data_service,
        analytics=analytics,
        port=port
    )
    
    asyncio.run(dashboard.start(open_browser))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MEGA Backtest Analytics Dashboard")
    parser.add_argument("--results-dir", default="v2/results/mega_sim", help="Results directory")
    parser.add_argument("--port", type=int, default=8081, help="Port number")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    
    args = parser.parse_args()
    
    run_dashboard(
        results_dir=args.results_dir,
        port=args.port,
        open_browser=not args.no_browser
    )
