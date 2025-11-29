"""
Backtest Analytics Dashboard for MEGA Historical Simulator.

Provides a web-based dashboard for analyzing backtest results.
Runs on localhost:8081 (separate from the main trading dashboard on 8080).
"""

from .data_service import DataService
from .analytics import BacktestAnalytics

# Lazy import for server (requires FastAPI)
def run_dashboard(*args, **kwargs):
    from .server import run_dashboard as _run_dashboard
    return _run_dashboard(*args, **kwargs)


def BacktestDashboard(*args, **kwargs):
    from .server import BacktestDashboard as _BacktestDashboard
    return _BacktestDashboard(*args, **kwargs)


__all__ = ['BacktestDashboard', 'BacktestAnalytics', 'DataService', 'run_dashboard']
