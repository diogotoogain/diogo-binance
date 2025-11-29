"""
Data Service for Backtest Dashboard.

Loads and manages simulation results data for the dashboard.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataService:
    """
    Manages data loading and filtering for the dashboard.
    
    Handles:
    - Loading simulation results (trades, equity curve, metrics)
    - Filtering by date range, time of day, direction, etc.
    - Pagination and sorting
    """
    
    def __init__(self, results_dir: str = "v2/results/mega_sim"):
        """
        Initialize DataService.
        
        Args:
            results_dir: Directory containing simulation results
        """
        self.results_dir = Path(results_dir)
        self._trades_df: Optional[pd.DataFrame] = None
        self._equity_df: Optional[pd.DataFrame] = None
        self._metrics: Dict[str, Any] = {}
        self._drift_events: List[Dict] = []
        self._ol_stats: Dict[str, Any] = {}
        self._config: Dict[str, Any] = {}
    
    def load_results(self, results_dir: Optional[str] = None) -> bool:
        """
        Load simulation results from directory.
        
        Args:
            results_dir: Optional override for results directory
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if results_dir is not None:
            self.results_dir = Path(results_dir)
        
        if not self.results_dir.exists():
            logger.warning(f"Results directory not found: {self.results_dir}")
            return False
        
        try:
            # Load trades
            trades_path = self.results_dir / "trades.csv"
            if trades_path.exists():
                self._trades_df = pd.read_csv(trades_path)
                # Parse datetime columns
                for col in ['entry_time', 'exit_time']:
                    if col in self._trades_df.columns:
                        self._trades_df[col] = pd.to_datetime(self._trades_df[col])
                logger.info(f"Loaded {len(self._trades_df)} trades")
            else:
                self._trades_df = pd.DataFrame()
            
            # Load equity curve
            equity_path = self.results_dir / "equity_curve.csv"
            if equity_path.exists():
                self._equity_df = pd.read_csv(equity_path)
                if 'timestamp' in self._equity_df.columns:
                    self._equity_df['timestamp'] = pd.to_datetime(self._equity_df['timestamp'])
                logger.info(f"Loaded {len(self._equity_df)} equity points")
            else:
                self._equity_df = pd.DataFrame()
            
            # Load metrics
            metrics_path = self.results_dir / "metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    self._metrics = json.load(f)
                logger.info("Loaded metrics")
            
            # Load drift events
            drift_path = self.results_dir / "drift_events.json"
            if drift_path.exists():
                with open(drift_path, 'r') as f:
                    self._drift_events = json.load(f)
                logger.info(f"Loaded {len(self._drift_events)} drift events")
            
            # Load online learning stats
            ol_path = self.results_dir / "online_model_stats.json"
            if ol_path.exists():
                with open(ol_path, 'r') as f:
                    self._ol_stats = json.load(f)
                logger.info("Loaded online learning stats")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load results: {e}")
            return False
    
    def load_from_simulation_result(self, result: Any) -> None:
        """
        Load data directly from SimulationResult object.
        
        Args:
            result: SimulationResult object from HonestSimulator
        """
        # Convert trades to DataFrame
        if hasattr(result, 'trades') and result.trades:
            records = []
            for i, trade in enumerate(result.trades):
                records.append({
                    'trade_id': i + 1,
                    'entry_time': trade.entry_time,
                    'exit_time': trade.exit_time,
                    'direction': trade.direction,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'size': trade.size,
                    'pnl': trade.pnl,
                    'fees': trade.fees,
                    'exit_reason': trade.exit_reason,
                    'entry_step': getattr(trade, 'entry_step', 0),
                    'exit_step': getattr(trade, 'exit_step', 0)
                })
            self._trades_df = pd.DataFrame(records)
            # Add derived columns
            if len(self._trades_df) > 0:
                self._trades_df['duration'] = (
                    self._trades_df['exit_time'] - self._trades_df['entry_time']
                )
                self._trades_df['pnl_pct'] = (
                    self._trades_df['pnl'] / self._trades_df['size'] * 100
                )
                self._trades_df['is_winner'] = self._trades_df['pnl'] > 0
        else:
            self._trades_df = pd.DataFrame()
        
        # Equity curve
        if hasattr(result, 'equity_curve'):
            self._equity_df = pd.DataFrame({'equity': result.equity_curve})
            if hasattr(result, 'timestamps') and result.timestamps:
                # Pad timestamps to match equity length
                n_missing = len(result.equity_curve) - len(result.timestamps)
                if n_missing > 0:
                    timestamps = [None] * n_missing + list(result.timestamps)
                else:
                    timestamps = result.timestamps[-len(result.equity_curve):]
                self._equity_df['timestamp'] = timestamps
        else:
            self._equity_df = pd.DataFrame()
        
        # Metrics
        self._metrics = getattr(result, 'metrics', {})
        
        # Drift events
        self._drift_events = getattr(result, 'drift_events', [])
        
        # Online learning stats
        self._ol_stats = getattr(result, 'online_model_stats', {})
    
    @property
    def trades(self) -> pd.DataFrame:
        """Get trades DataFrame."""
        return self._trades_df if self._trades_df is not None else pd.DataFrame()
    
    @property
    def equity(self) -> pd.DataFrame:
        """Get equity curve DataFrame."""
        return self._equity_df if self._equity_df is not None else pd.DataFrame()
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get metrics dictionary."""
        return self._metrics
    
    @property
    def drift_events(self) -> List[Dict]:
        """Get drift events list."""
        return self._drift_events
    
    @property
    def ol_stats(self) -> Dict[str, Any]:
        """Get online learning statistics."""
        return self._ol_stats
    
    def filter_trades(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        direction: Optional[int] = None,
        result_filter: Optional[str] = None,  # 'winners', 'losers', 'all'
        min_pnl: Optional[float] = None,
        max_pnl: Optional[float] = None,
        hour_start: Optional[int] = None,
        hour_end: Optional[int] = None,
        day_of_week: Optional[List[int]] = None,  # 0=Monday, 6=Sunday
        exit_reason: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Filter trades by various criteria.
        
        Returns filtered DataFrame.
        """
        if self._trades_df is None or len(self._trades_df) == 0:
            return pd.DataFrame()
        
        df = self._trades_df.copy()
        
        # Date range filter
        if start_date is not None and 'entry_time' in df.columns:
            df = df[df['entry_time'] >= start_date]
        if end_date is not None and 'entry_time' in df.columns:
            df = df[df['entry_time'] <= end_date]
        
        # Direction filter
        if direction is not None and 'direction' in df.columns:
            df = df[df['direction'] == direction]
        
        # Result filter
        if result_filter == 'winners' and 'pnl' in df.columns:
            df = df[df['pnl'] > 0]
        elif result_filter == 'losers' and 'pnl' in df.columns:
            df = df[df['pnl'] < 0]
        
        # P&L range filter
        if min_pnl is not None and 'pnl' in df.columns:
            df = df[df['pnl'] >= min_pnl]
        if max_pnl is not None and 'pnl' in df.columns:
            df = df[df['pnl'] <= max_pnl]
        
        # Hour of day filter
        if 'entry_time' in df.columns and df['entry_time'].dtype == 'datetime64[ns]':
            if hour_start is not None:
                df = df[df['entry_time'].dt.hour >= hour_start]
            if hour_end is not None:
                df = df[df['entry_time'].dt.hour <= hour_end]
        
        # Day of week filter
        if day_of_week is not None and 'entry_time' in df.columns:
            if df['entry_time'].dtype == 'datetime64[ns]':
                df = df[df['entry_time'].dt.dayofweek.isin(day_of_week)]
        
        # Exit reason filter
        if exit_reason is not None and 'exit_reason' in df.columns:
            df = df[df['exit_reason'] == exit_reason]
        
        return df
    
    def get_trades_paginated(
        self,
        page: int = 1,
        per_page: int = 25,
        sort_by: str = 'entry_time',
        sort_desc: bool = True,
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, int]:
        """
        Get paginated trades list.
        
        Args:
            page: Page number (1-indexed)
            per_page: Items per page
            sort_by: Column to sort by
            sort_desc: Sort descending if True
            filters: Filter criteria dictionary
            
        Returns:
            Tuple of (filtered DataFrame, total count)
        """
        # Apply filters
        if filters:
            df = self.filter_trades(**filters)
        else:
            df = self.trades.copy()
        
        total = len(df)
        
        # Sort
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=not sort_desc)
        
        # Paginate
        start = (page - 1) * per_page
        end = start + per_page
        df = df.iloc[start:end]
        
        return df, total
    
    def get_date_range(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get date range of trades."""
        if self._trades_df is None or len(self._trades_df) == 0:
            return None, None
        
        if 'entry_time' not in self._trades_df.columns:
            return None, None
        
        return (
            self._trades_df['entry_time'].min(),
            self._trades_df['exit_time'].max() if 'exit_time' in self._trades_df.columns else self._trades_df['entry_time'].max()
        )
    
    def get_equity_downsampled(self, max_points: int = 5000) -> List[Dict[str, Any]]:
        """
        Get downsampled equity curve for charting.
        
        Args:
            max_points: Maximum points to return
            
        Returns:
            List of {timestamp, equity} dicts
        """
        if self._equity_df is None or len(self._equity_df) == 0:
            return []
        
        df = self._equity_df.copy()
        
        if len(df) > max_points:
            step = len(df) // max_points
            indices = np.arange(0, len(df), step)
            if indices[-1] != len(df) - 1:
                indices = np.append(indices, len(df) - 1)
            df = df.iloc[indices]
        
        result = []
        for _, row in df.iterrows():
            result.append({
                'timestamp': row.get('timestamp', '').isoformat() if pd.notna(row.get('timestamp')) else None,
                'equity': float(row['equity']) if 'equity' in row else 0
            })
        
        return result
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for dashboard."""
        stats = dict(self._metrics)
        
        if self._trades_df is not None and len(self._trades_df) > 0:
            df = self._trades_df
            
            # Additional stats
            stats['total_trades'] = len(df)
            stats['winning_trades'] = int((df['pnl'] > 0).sum()) if 'pnl' in df.columns else 0
            stats['losing_trades'] = int((df['pnl'] < 0).sum()) if 'pnl' in df.columns else 0
            stats['breakeven_trades'] = int((df['pnl'] == 0).sum()) if 'pnl' in df.columns else 0
            
            if 'pnl' in df.columns:
                stats['total_pnl'] = float(df['pnl'].sum())
                stats['avg_pnl'] = float(df['pnl'].mean())
                stats['max_win'] = float(df['pnl'].max())
                stats['max_loss'] = float(df['pnl'].min())
                
                winners = df[df['pnl'] > 0]
                losers = df[df['pnl'] < 0]
                stats['avg_win'] = float(winners['pnl'].mean()) if len(winners) > 0 else 0
                stats['avg_loss'] = float(losers['pnl'].mean()) if len(losers) > 0 else 0
            
            if 'direction' in df.columns:
                stats['long_trades'] = int((df['direction'] == 1).sum())
                stats['short_trades'] = int((df['direction'] == -1).sum())
            
            # Date range
            start, end = self.get_date_range()
            if start is not None:
                stats['start_date'] = start.isoformat()
            if end is not None:
                stats['end_date'] = end.isoformat()
        
        # Equity stats
        if self._equity_df is not None and len(self._equity_df) > 0 and 'equity' in self._equity_df.columns:
            equity = self._equity_df['equity'].values
            stats['equity_points'] = len(equity)
            stats['initial_balance'] = float(equity[0])
            stats['final_balance'] = float(equity[-1])
        
        # Drift events
        stats['drift_events_count'] = len(self._drift_events)
        
        return stats
