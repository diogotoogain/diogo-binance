"""
Backtest report generation.

Generates comprehensive reports in various formats.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


class BacktestReport:
    """
    Generate and save backtest reports.

    Supports multiple output formats: text, HTML, JSON.
    """

    def __init__(
        self,
        results: Dict[str, Any],
        trades: pd.DataFrame,
        equity_curve: pd.Series,
    ):
        """
        Initialize backtest report.

        Args:
            results: Backtest results dictionary
            trades: DataFrame of executed trades
            equity_curve: Series of equity values
        """
        self.results = results
        self.trades = trades
        self.equity_curve = equity_curve

    def generate_summary(self) -> str:
        """
        Generate text summary report.

        Returns:
            Formatted summary string
        """
        r = self.results

        return f"""
╔═══════════════════════════════════════════════════════════════╗
║                    📊 BACKTEST REPORT                          ║
╠═══════════════════════════════════════════════════════════════╣
║  CAPITAL                                                       ║
║    Initial Capital:      ${r.get('initial_capital', 0):>10,.2f}                   ║
║    Final Equity:         ${r.get('final_equity', 0):>10,.2f}                   ║
║    Total Return:         {r.get('total_return_pct', 0):>10.2f}%                      ║
╠═══════════════════════════════════════════════════════════════╣
║  RISK-ADJUSTED RETURNS                                        ║
║    Sharpe Ratio:         {r.get('sharpe_ratio', 0):>10.2f}                       ║
║    Sortino Ratio:        {r.get('sortino_ratio', 0):>10.2f}                       ║
║    Calmar Ratio:         {r.get('calmar_ratio', 0):>10.2f}                       ║
╠═══════════════════════════════════════════════════════════════╣
║  RISK METRICS                                                  ║
║    Max Drawdown:         {r.get('max_drawdown', 0):>10.2f}%                      ║
║    Volatility:           {r.get('volatility', 0):>10.2f}%                      ║
║    VaR (95%):            {r.get('var_95', 0):>10.2f}%                      ║
║    CVaR (95%):           {r.get('cvar_95', 0):>10.2f}%                      ║
╠═══════════════════════════════════════════════════════════════╣
║  TRADE STATISTICS                                             ║
║    Total Trades:         {r.get('total_trades', 0):>10}                       ║
║    Winning Trades:       {r.get('winning_trades', 0):>10}                       ║
║    Losing Trades:        {r.get('losing_trades', 0):>10}                       ║
║    Win Rate:             {r.get('win_rate', 0):>10.2f}%                      ║
║    Profit Factor:        {r.get('profit_factor', 0):>10.2f}                       ║
║    Avg Trade P&L:        ${r.get('avg_trade_pnl', 0):>10.2f}                     ║
║    Avg Win:              ${r.get('avg_win', 0):>10.2f}                     ║
║    Avg Loss:             ${r.get('avg_loss', 0):>10.2f}                     ║
║    Expectancy:           ${r.get('expectancy', 0):>10.2f}                     ║
║    SQN:                  {r.get('sqn', 0):>10.2f}                       ║
╠═══════════════════════════════════════════════════════════════╣
║  TIME ANALYSIS                                                ║
║    Trading Days:         {r.get('trading_days', 0):>10}                       ║
║    Positive Days:        {r.get('positive_days', 0):>10}                       ║
║    Positive Day %:       {r.get('positive_day_pct', 0):>10.2f}%                      ║
╚═══════════════════════════════════════════════════════════════╝
"""

    def generate_trade_summary(self) -> str:
        """
        Generate trade-by-trade summary.

        Returns:
            Formatted trade summary
        """
        if len(self.trades) == 0:
            return "No trades executed"

        summary = "\n📈 TRADE LOG\n"
        summary += "=" * 80 + "\n"
        summary += f"{'#':<4} {'Direction':<6} {'Entry':<12} {'Exit':<12} {'P&L':<12} {'Reason':<15}\n"
        summary += "-" * 80 + "\n"

        for i, trade in self.trades.iterrows():
            direction = "LONG" if trade.get("direction", 0) == 1 else "SHORT"
            entry = f"${trade.get('entry_price', 0):,.2f}"
            exit_price = f"${trade.get('exit_price', 0):,.2f}"
            pnl = trade.get("pnl", 0)
            pnl_str = f"${pnl:+,.2f}"
            reason = trade.get("exit_reason", "")[:15]

            summary += f"{i+1:<4} {direction:<6} {entry:<12} {exit_price:<12} {pnl_str:<12} {reason:<15}\n"

        summary += "-" * 80 + "\n"

        return summary

    def generate_monthly_returns(self) -> pd.DataFrame:
        """
        Generate monthly returns table.

        Returns:
            DataFrame with monthly returns
        """
        if len(self.equity_curve) == 0:
            return pd.DataFrame()

        # Convert to returns
        returns = self.equity_curve.pct_change().dropna()

        # Try to create monthly grouping
        if isinstance(returns.index, pd.DatetimeIndex):
            monthly = returns.groupby(pd.Grouper(freq="M")).apply(
                lambda x: (1 + x).prod() - 1
            )
            monthly.index = monthly.index.strftime("%Y-%m")
            return monthly.to_frame("return")

        return pd.DataFrame()

    def save_json(self, path: Union[str, Path]) -> None:
        """
        Save results to JSON file.

        Args:
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare JSON-serializable data
        output = {}
        for key, value in self.results.items():
            if key in ["equity_curve", "trades"]:
                continue
            if isinstance(value, (int, float, str, bool, type(None))):
                output[key] = value
            else:
                output[key] = str(value)

        with open(path, "w") as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"Saved JSON report to {path}")

    def save_csv(self, path: Union[str, Path]) -> None:
        """
        Save trades to CSV file.

        Args:
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if len(self.trades) > 0:
            self.trades.to_csv(path, index=False)
            logger.info(f"Saved trades CSV to {path}")

    def save_html(self, path: Union[str, Path]) -> None:
        """
        Save report to HTML file with embedded charts.

        Args:
            path: Output path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        html_content = self._generate_html_report()

        with open(path, "w") as f:
            f.write(html_content)

        logger.info(f"Saved HTML report to {path}")

    def _generate_html_report(self) -> str:
        """Generate HTML report content."""
        r = self.results

        # Generate equity curve data for chart
        equity_data = list(self.equity_curve.values) if len(self.equity_curve) > 0 else []

        # Generate trade data
        trades_json = "[]"
        if len(self.trades) > 0:
            trades_json = self.trades.to_json(orient="records", date_format="iso")

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        h1, h2, h3 {{
            color: #00d4ff;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #16213e;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #0f3460;
        }}
        .metric-label {{
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #00d4ff;
        }}
        .positive {{ color: #00ff88; }}
        .negative {{ color: #ff4444; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #333;
        }}
        th {{
            background: #16213e;
            color: #00d4ff;
        }}
        .chart-container {{
            background: #16213e;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        canvas {{
            width: 100% !important;
            height: 300px !important;
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>📊 Backtest Report</h1>
    
    <h2>Performance Summary</h2>
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">Total Return</div>
            <div class="metric-value {'positive' if r.get('total_return_pct', 0) >= 0 else 'negative'}">
                {r.get('total_return_pct', 0):.2f}%
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value">{r.get('sharpe_ratio', 0):.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value negative">{r.get('max_drawdown', 0):.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value">{r.get('win_rate', 0):.2f}%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Profit Factor</div>
            <div class="metric-value">{r.get('profit_factor', 0):.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Total Trades</div>
            <div class="metric-value">{r.get('total_trades', 0)}</div>
        </div>
    </div>
    
    <h2>Equity Curve</h2>
    <div class="chart-container">
        <canvas id="equityChart"></canvas>
    </div>
    
    <h2>Detailed Metrics</h2>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Initial Capital</td><td>${r.get('initial_capital', 0):,.2f}</td></tr>
        <tr><td>Final Equity</td><td>${r.get('final_equity', 0):,.2f}</td></tr>
        <tr><td>Sortino Ratio</td><td>{r.get('sortino_ratio', 0):.2f}</td></tr>
        <tr><td>Calmar Ratio</td><td>{r.get('calmar_ratio', 0):.2f}</td></tr>
        <tr><td>Volatility</td><td>{r.get('volatility', 0):.2f}%</td></tr>
        <tr><td>VaR (95%)</td><td>{r.get('var_95', 0):.2f}%</td></tr>
        <tr><td>Avg Trade P&L</td><td>${r.get('avg_trade_pnl', 0):.2f}</td></tr>
        <tr><td>SQN</td><td>{r.get('sqn', 0):.2f}</td></tr>
    </table>
    
    <script>
        const equityData = {equity_data};
        
        new Chart(document.getElementById('equityChart'), {{
            type: 'line',
            data: {{
                labels: equityData.map((_, i) => i),
                datasets: [{{
                    label: 'Equity',
                    data: equityData,
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    fill: true,
                    tension: 0.1
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
                        grid: {{ color: '#333' }},
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

    def save_all(self, directory: Union[str, Path]) -> None:
        """
        Save all report formats to directory.

        Args:
            directory: Output directory
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self.save_json(directory / "results.json")
        self.save_csv(directory / "trades.csv")
        self.save_html(directory / "report.html")

        # Save equity curve
        if len(self.equity_curve) > 0:
            self.equity_curve.to_csv(directory / "equity_curve.csv")

        logger.info(f"Saved all reports to {directory}")

    def print_summary(self) -> None:
        """Print summary to console."""
        print(self.generate_summary())

    @classmethod
    def from_backtest_results(cls, results: Dict[str, Any]) -> "BacktestReport":
        """
        Create report from backtest results.

        Args:
            results: Backtest results dictionary

        Returns:
            BacktestReport instance
        """
        equity_curve = results.get("equity_curve", pd.Series())
        trades = results.get("trades", pd.DataFrame())

        return cls(results, trades, equity_curve)
