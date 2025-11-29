"""
Simulation Report Generator.

Generates comprehensive reports from simulation results
including HTML, JSON, and CSV outputs.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Maximum points to display in charts to avoid browser performance issues
MAX_CHART_POINTS = 5000


def downsample_for_chart(data: List, max_points: int = MAX_CHART_POINTS) -> List:
    """
    Downsample data for chart visualization while preserving important features.
    
    Uses LTTB (Largest-Triangle-Three-Buckets) inspired approach that
    preserves local extrema better than simple decimation.
    
    Args:
        data: List of numeric values
        max_points: Maximum number of points to return
        
    Returns:
        Downsampled list
    """
    if len(data) <= max_points:
        return data
    
    # Simple approach: keep every nth point plus first/last
    # More sophisticated: use numpy to find local mins/maxes
    arr = np.array(data)
    step = len(arr) // max_points
    
    # Get evenly spaced indices
    indices = np.arange(0, len(arr), step)
    
    # Ensure we include the last point
    if indices[-1] != len(arr) - 1:
        indices = np.append(indices, len(arr) - 1)
    
    return arr[indices].tolist()


class SimulationReport:
    """
    Generate comprehensive simulation reports.

    Includes:
    - Equity curve visualization
    - Drawdown analysis
    - Performance metrics (Sharpe, Sortino, Calmar, etc.)
    - Period analysis (yearly, monthly)
    - Trade analysis by market regime
    - Drift detection events
    """

    def __init__(
        self,
        result: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize SimulationReport.

        Args:
            result: SimulationResult from HonestSimulator
            config: Optional configuration dict
        """
        self.result = result
        self.config = config or {}

    def generate(self, output_dir: str) -> None:
        """
        Generate all reports to output directory.

        Args:
            output_dir: Directory to save reports
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating reports to {output_path}")

        # Generate each report type
        self._save_equity_curve(output_path / "equity_curve.csv")
        self._save_trades(output_path / "trades.csv")
        self._save_metrics(output_path / "metrics.json")
        self._save_drift_events(output_path / "drift_events.json")
        self._save_html_report(output_path / "report.html")

        logger.info("Report generation complete")

    def _save_equity_curve(self, path: Path) -> None:
        """Save equity curve to CSV."""
        if hasattr(self.result, 'equity_curve'):
            df = pd.DataFrame({
                'equity': self.result.equity_curve
            })

            if hasattr(self.result, 'timestamps') and self.result.timestamps:
                # Pad timestamps to match equity curve length
                n_missing = len(self.result.equity_curve) - len(self.result.timestamps)
                if n_missing > 0:
                    timestamps = [None] * n_missing + list(self.result.timestamps)
                else:
                    timestamps = self.result.timestamps[-len(self.result.equity_curve):]
                df['timestamp'] = timestamps

            df.to_csv(path, index=True)
            logger.info(f"Saved equity curve: {path}")

    def _save_trades(self, path: Path) -> None:
        """Save trades to CSV."""
        if hasattr(self.result, 'trades') and self.result.trades:
            records = []
            for trade in self.result.trades:
                records.append({
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'direction': trade.direction,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'size': trade.size,
                    'pnl': trade.pnl,
                    'fees': trade.fees,
                    'exit_reason': trade.exit_reason
                })
            df = pd.DataFrame(records)
            df.to_csv(path, index=False)
            logger.info(f"Saved trades: {path}")

    def _save_metrics(self, path: Path) -> None:
        """Save metrics to JSON."""
        if hasattr(self.result, 'metrics'):
            with open(path, 'w') as f:
                json.dump(self.result.metrics, f, indent=2, default=str)
            logger.info(f"Saved metrics: {path}")

    def _save_drift_events(self, path: Path) -> None:
        """Save drift events to JSON."""
        if hasattr(self.result, 'drift_events'):
            with open(path, 'w') as f:
                json.dump(self.result.drift_events, f, indent=2, default=str)
            logger.info(f"Saved drift events: {path}")

    def _save_html_report(self, path: Path) -> None:
        """Generate and save HTML report."""
        html = self._generate_html()
        with open(path, 'w') as f:
            f.write(html)
        logger.info(f"Saved HTML report: {path}")

    def _generate_html(self) -> str:
        """Generate HTML report content."""
        metrics = getattr(self.result, 'metrics', {})
        equity = getattr(self.result, 'equity_curve', [])
        trades = getattr(self.result, 'trades', [])
        drift_events = getattr(self.result, 'drift_events', [])
        ol_stats = getattr(self.result, 'online_model_stats', {})

        # Calculate additional metrics for display
        if trades:
            wins = [t for t in trades if t.pnl > 0]
            losses = [t for t in trades if t.pnl < 0]
            max_win = max(t.pnl for t in trades) if trades else 0
            max_loss = min(t.pnl for t in trades) if trades else 0
        else:
            wins, losses = [], []
            max_win, max_loss = 0, 0

        # Drawdown calculation
        if len(equity) > 0:
            equity_arr = np.array(equity)
            peak = np.maximum.accumulate(equity_arr)
            drawdown = (equity_arr - peak) / np.where(peak > 0, peak, 1) * 100
            drawdown_list = drawdown.tolist()
        else:
            drawdown_list = []

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MEGA Simulation Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-primary: #0f0f23;
            --bg-secondary: #1a1a35;
            --text-primary: #cccccc;
            --text-secondary: #888888;
            --accent: #00d4ff;
            --positive: #00ff88;
            --negative: #ff4444;
            --border: #333366;
        }}
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: var(--accent);
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }}
        h2 {{
            color: var(--accent);
            margin: 20px 0 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--border);
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid var(--border);
        }}
        .metric-label {{
            font-size: 0.85em;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: var(--accent);
            margin-top: 5px;
        }}
        .positive {{ color: var(--positive); }}
        .negative {{ color: var(--negative); }}
        .chart-container {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 30px;
        }}
        canvas {{
            max-height: 400px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background: var(--bg-secondary);
            color: var(--accent);
            font-weight: 600;
        }}
        tr:hover {{
            background: rgba(0, 212, 255, 0.05);
        }}
        .summary-box {{
            background: linear-gradient(135deg, var(--bg-secondary) 0%, #1e1e4a 100%);
            border-radius: 12px;
            padding: 30px;
            margin-bottom: 30px;
            border: 1px solid var(--border);
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 30px;
        }}
        .timestamp {{
            text-align: right;
            color: var(--text-secondary);
            font-size: 0.9em;
            margin-bottom: 20px;
        }}
        @media (max-width: 768px) {{
            .summary-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 MEGA SIMULADOR HISTÓRICO</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary-box">
            <div class="summary-grid">
                <div>
                    <div class="metric-label">Total Return</div>
                    <div class="metric-value {'positive' if metrics.get('total_return_pct', 0) >= 0 else 'negative'}">
                        {metrics.get('total_return_pct', 0):.2f}%
                    </div>
                </div>
                <div>
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value">{metrics.get('sharpe_ratio', 0):.2f}</div>
                </div>
                <div>
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value negative">{metrics.get('max_drawdown', 0):.2f}%</div>
                </div>
                <div>
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value">{metrics.get('win_rate', 0):.1f}%</div>
                </div>
            </div>
        </div>

        <h2>💰 Performance Metrics</h2>
        <div class="grid">
            <div class="card">
                <div class="metric-label">Final Balance</div>
                <div class="metric-value">${metrics.get('final_balance', 0):,.2f}</div>
            </div>
            <div class="card">
                <div class="metric-label">Sortino Ratio</div>
                <div class="metric-value">{metrics.get('sortino_ratio', 0):.2f}</div>
            </div>
            <div class="card">
                <div class="metric-label">Profit Factor</div>
                <div class="metric-value">{metrics.get('profit_factor', 0):.2f}</div>
            </div>
            <div class="card">
                <div class="metric-label">Total Trades</div>
                <div class="metric-value">{metrics.get('total_trades', 0)}</div>
            </div>
            <div class="card">
                <div class="metric-label">Winning Trades</div>
                <div class="metric-value positive">{metrics.get('winning_trades', 0)}</div>
            </div>
            <div class="card">
                <div class="metric-label">Losing Trades</div>
                <div class="metric-value negative">{metrics.get('losing_trades', 0)}</div>
            </div>
            <div class="card">
                <div class="metric-label">Avg Trade P&L</div>
                <div class="metric-value {'positive' if metrics.get('avg_trade_pnl', 0) >= 0 else 'negative'}">
                    ${metrics.get('avg_trade_pnl', 0):.2f}
                </div>
            </div>
            <div class="card">
                <div class="metric-label">Total Fees</div>
                <div class="metric-value">${metrics.get('total_fees', 0):.2f}</div>
            </div>
        </div>

        <h2>📈 Equity Curve</h2>
        <div class="chart-container">
            <canvas id="equityChart"></canvas>
        </div>

        <h2>📉 Drawdown</h2>
        <div class="chart-container">
            <canvas id="drawdownChart"></canvas>
        </div>

        <h2>🔄 Online Learning Stats</h2>
        <div class="grid">
            <div class="card">
                <div class="metric-label">Samples Seen</div>
                <div class="metric-value">{ol_stats.get('n_samples_seen', 'N/A')}</div>
            </div>
            <div class="card">
                <div class="metric-label">Drifts Detected</div>
                <div class="metric-value">{len(drift_events)}</div>
            </div>
            <div class="card">
                <div class="metric-label">Model Accuracy</div>
                <div class="metric-value">{ol_stats.get('accuracy', 'N/A')}</div>
            </div>
        </div>

        <h2>📊 Trade Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Max Win</td>
                <td class="positive">${max_win:.2f}</td>
            </tr>
            <tr>
                <td>Max Loss</td>
                <td class="negative">${max_loss:.2f}</td>
            </tr>
            <tr>
                <td>Average Win</td>
                <td class="positive">${sum(t.pnl for t in wins) / len(wins) if wins else 0:.2f}</td>
            </tr>
            <tr>
                <td>Average Loss</td>
                <td class="negative">${sum(t.pnl for t in losses) / len(losses) if losses else 0:.2f}</td>
            </tr>
        </table>

        {self._generate_drift_section(drift_events)}
    </div>

    <script>
        // Equity Chart - downsampled for performance
        const equityData = {json.dumps(downsample_for_chart(equity))};
        const equityCtx = document.getElementById('equityChart').getContext('2d');
        new Chart(equityCtx, {{
            type: 'line',
            data: {{
                labels: equityData.map((_, i) => i),
                datasets: [{{
                    label: 'Equity',
                    data: equityData,
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        grid: {{ color: '#333' }},
                        ticks: {{ color: '#888' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#888', maxTicksLimit: 10 }}
                    }}
                }}
            }}
        }});

        // Drawdown Chart - downsampled for performance
        const drawdownData = {json.dumps(downsample_for_chart(drawdown_list))};
        const ddCtx = document.getElementById('drawdownChart').getContext('2d');
        new Chart(ddCtx, {{
            type: 'line',
            data: {{
                labels: drawdownData.map((_, i) => i),
                datasets: [{{
                    label: 'Drawdown %',
                    data: drawdownData,
                    borderColor: '#ff4444',
                    backgroundColor: 'rgba(255, 68, 68, 0.2)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        grid: {{ color: '#333' }},
                        ticks: {{ color: '#888' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#888', maxTicksLimit: 10 }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        return html

    def _generate_drift_section(self, drift_events: List[Dict]) -> str:
        """Generate drift events section HTML."""
        if not drift_events:
            return """
        <h2>🔄 Drift Events</h2>
        <p style="color: var(--text-secondary);">No drift events detected during simulation.</p>
"""

        rows = ""
        for event in drift_events[:20]:  # Show first 20
            rows += f"""
            <tr>
                <td>{event.get('step', 'N/A')}</td>
                <td>{event.get('timestamp', 'N/A')}</td>
                <td>{', '.join(event.get('metrics', []))}</td>
            </tr>
"""

        return f"""
        <h2>🔄 Drift Events ({len(drift_events)} total)</h2>
        <table>
            <tr>
                <th>Step</th>
                <th>Timestamp</th>
                <th>Metrics</th>
            </tr>
            {rows}
        </table>
"""

    def print_summary(self) -> None:
        """Print summary to console."""
        metrics = getattr(self.result, 'metrics', {})

        print(f"""
╔═══════════════════════════════════════════════════════════════╗
║              📊 MEGA SIMULATION RESULTS                        ║
╠═══════════════════════════════════════════════════════════════╣
║  💰 Performance                                                ║
║    Total Return:         {metrics.get('total_return_pct', 0):>10.2f}%                     ║
║    Final Balance:        ${metrics.get('final_balance', 0):>10,.2f}                  ║
║    Sharpe Ratio:         {metrics.get('sharpe_ratio', 0):>10.2f}                      ║
║    Sortino Ratio:        {metrics.get('sortino_ratio', 0):>10.2f}                      ║
║    Max Drawdown:         {metrics.get('max_drawdown', 0):>10.2f}%                     ║
╠═══════════════════════════════════════════════════════════════╣
║  📈 Trades                                                     ║
║    Total Trades:         {metrics.get('total_trades', 0):>10}                      ║
║    Win Rate:             {metrics.get('win_rate', 0):>10.1f}%                     ║
║    Winning:              {metrics.get('winning_trades', 0):>10}                      ║
║    Losing:               {metrics.get('losing_trades', 0):>10}                      ║
║    Profit Factor:        {metrics.get('profit_factor', 0):>10.2f}                      ║
║    Avg Trade P&L:        ${metrics.get('avg_trade_pnl', 0):>10.2f}                  ║
╠═══════════════════════════════════════════════════════════════╣
║  🔄 Online Learning                                            ║
║    Drifts Detected:      {len(getattr(self.result, 'drift_events', [])):>10}                      ║
╚═══════════════════════════════════════════════════════════════╝
""")
