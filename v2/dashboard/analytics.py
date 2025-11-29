"""
Analytics Module for Backtest Dashboard.

Provides advanced analytics calculations including:
- Performance metrics (Sharpe, Sortino, Calmar, etc.)
- Temporal analysis (by hour, day, month, year)
- Risk analysis (drawdown, VaR, CVaR)
- Sequence analysis (win/loss streaks)
- Monte Carlo simulations
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BacktestAnalytics:
    """
    Advanced analytics for backtest results.
    
    Provides various analysis methods for trades and equity curves.
    """
    
    def __init__(self, data_service: Any):
        """
        Initialize BacktestAnalytics.
        
        Args:
            data_service: DataService instance with loaded data
        """
        self.data_service = data_service
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate all performance metrics."""
        metrics = {}
        
        equity = self.data_service.equity
        trades = self.data_service.trades
        
        if len(equity) > 0 and 'equity' in equity.columns:
            equity_arr = equity['equity'].values
            
            # Returns
            returns = np.diff(equity_arr) / equity_arr[:-1]
            returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
            
            # Total return
            if equity_arr[0] > 0:
                metrics['total_return_pct'] = (equity_arr[-1] - equity_arr[0]) / equity_arr[0] * 100
            
            # Sharpe ratio (annualized, assuming minute data)
            if len(returns) > 0 and np.std(returns) > 0:
                metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(525600)
            else:
                metrics['sharpe_ratio'] = 0.0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and np.std(downside_returns) > 0:
                metrics['sortino_ratio'] = np.mean(returns) / np.std(downside_returns) * np.sqrt(525600)
            else:
                metrics['sortino_ratio'] = 0.0
            
            # Max drawdown
            peak = np.maximum.accumulate(equity_arr)
            drawdown = (equity_arr - peak) / peak
            metrics['max_drawdown'] = abs(np.min(drawdown)) * 100
            
            # Calmar ratio
            if metrics['max_drawdown'] > 0:
                annualized_return = metrics['total_return_pct'] / 100 * (525600 / len(equity_arr))
                metrics['calmar_ratio'] = annualized_return / (metrics['max_drawdown'] / 100)
            else:
                metrics['calmar_ratio'] = 0.0
        
        if len(trades) > 0 and 'pnl' in trades.columns:
            # Win rate
            wins = trades['pnl'] > 0
            metrics['win_rate'] = wins.sum() / len(trades) * 100
            
            # Profit factor
            gross_profit = trades.loc[wins, 'pnl'].sum()
            gross_loss = abs(trades.loc[~wins, 'pnl'].sum())
            if gross_loss > 0:
                metrics['profit_factor'] = gross_profit / gross_loss
            else:
                metrics['profit_factor'] = float('inf') if gross_profit > 0 else 0.0
            
            # Expectancy
            metrics['expectancy'] = trades['pnl'].mean()
        
        return metrics
    
    def analyze_by_period(
        self,
        period: str = 'day'
    ) -> List[Dict[str, Any]]:
        """
        Analyze performance by time period.
        
        Args:
            period: 'day', 'week', 'month', 'year'
            
        Returns:
            List of period summaries
        """
        trades = self.data_service.trades
        
        if len(trades) == 0 or 'entry_time' not in trades.columns:
            return []
        
        df = trades.copy()
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        
        # Create period column
        if period == 'day':
            df['period'] = df['entry_time'].dt.date
        elif period == 'week':
            df['period'] = df['entry_time'].dt.to_period('W').apply(lambda x: x.start_time)
        elif period == 'month':
            df['period'] = df['entry_time'].dt.to_period('M').apply(lambda x: x.start_time)
        elif period == 'year':
            df['period'] = df['entry_time'].dt.year
        else:
            df['period'] = df['entry_time'].dt.date
        
        # Group and aggregate
        grouped = df.groupby('period').agg({
            'pnl': ['sum', 'mean', 'count'],
            'direction': lambda x: (x == 1).sum()  # long count
        }).reset_index()
        
        grouped.columns = ['period', 'total_pnl', 'avg_pnl', 'trades', 'longs']
        grouped['shorts'] = grouped['trades'] - grouped['longs']
        
        # Win rate per period
        win_rates = df.groupby('period').apply(
            lambda x: (x['pnl'] > 0).sum() / len(x) * 100 if len(x) > 0 else 0
        ).reset_index()
        win_rates.columns = ['period', 'win_rate']
        
        grouped = grouped.merge(win_rates, on='period')
        
        result = []
        for _, row in grouped.iterrows():
            result.append({
                'period': str(row['period']),
                'total_pnl': float(row['total_pnl']),
                'avg_pnl': float(row['avg_pnl']),
                'trades': int(row['trades']),
                'longs': int(row['longs']),
                'shorts': int(row['shorts']),
                'win_rate': float(row['win_rate'])
            })
        
        return result
    
    def analyze_by_hour(self) -> List[Dict[str, Any]]:
        """Analyze performance by hour of day (0-23)."""
        trades = self.data_service.trades
        
        if len(trades) == 0 or 'entry_time' not in trades.columns:
            return []
        
        df = trades.copy()
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['hour'] = df['entry_time'].dt.hour
        
        result = []
        for hour in range(24):
            hour_df = df[df['hour'] == hour]
            if len(hour_df) > 0:
                result.append({
                    'hour': hour,
                    'trades': len(hour_df),
                    'total_pnl': float(hour_df['pnl'].sum()),
                    'avg_pnl': float(hour_df['pnl'].mean()),
                    'win_rate': float((hour_df['pnl'] > 0).sum() / len(hour_df) * 100)
                })
            else:
                result.append({
                    'hour': hour,
                    'trades': 0,
                    'total_pnl': 0.0,
                    'avg_pnl': 0.0,
                    'win_rate': 0.0
                })
        
        return result
    
    def analyze_by_day_of_week(self) -> List[Dict[str, Any]]:
        """Analyze performance by day of week."""
        trades = self.data_service.trades
        
        if len(trades) == 0 or 'entry_time' not in trades.columns:
            return []
        
        df = trades.copy()
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['day_of_week'] = df['entry_time'].dt.dayofweek
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        result = []
        for dow in range(7):
            dow_df = df[df['day_of_week'] == dow]
            if len(dow_df) > 0:
                result.append({
                    'day': dow,
                    'day_name': day_names[dow],
                    'trades': len(dow_df),
                    'total_pnl': float(dow_df['pnl'].sum()),
                    'avg_pnl': float(dow_df['pnl'].mean()),
                    'win_rate': float((dow_df['pnl'] > 0).sum() / len(dow_df) * 100)
                })
            else:
                result.append({
                    'day': dow,
                    'day_name': day_names[dow],
                    'trades': 0,
                    'total_pnl': 0.0,
                    'avg_pnl': 0.0,
                    'win_rate': 0.0
                })
        
        return result
    
    def analyze_drawdowns(self, threshold_pct: float = 1.0) -> List[Dict[str, Any]]:
        """
        Analyze drawdown periods.
        
        Args:
            threshold_pct: Minimum drawdown percentage to include
            
        Returns:
            List of drawdown periods with details
        """
        equity = self.data_service.equity
        
        if len(equity) == 0 or 'equity' not in equity.columns:
            return []
        
        equity_arr = equity['equity'].values
        timestamps = equity['timestamp'].values if 'timestamp' in equity.columns else None
        
        # Calculate drawdown
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - peak) / peak * 100
        
        # Find drawdown periods
        drawdowns = []
        in_drawdown = False
        dd_start = 0
        dd_peak_idx = 0
        
        for i in range(len(drawdown)):
            if not in_drawdown and drawdown[i] < -threshold_pct:
                in_drawdown = True
                dd_start = i
                dd_peak_idx = i - 1 if i > 0 else 0
            elif in_drawdown and drawdown[i] >= 0:
                # Recovery
                dd_end = i
                max_dd = np.min(drawdown[dd_start:dd_end])
                max_dd_idx = dd_start + np.argmin(drawdown[dd_start:dd_end])
                
                drawdowns.append({
                    'start_idx': dd_peak_idx,
                    'trough_idx': max_dd_idx,
                    'end_idx': dd_end,
                    'start_time': str(timestamps[dd_peak_idx]) if timestamps is not None else None,
                    'trough_time': str(timestamps[max_dd_idx]) if timestamps is not None else None,
                    'end_time': str(timestamps[dd_end]) if timestamps is not None else None,
                    'max_drawdown_pct': float(abs(max_dd)),
                    'duration_bars': dd_end - dd_peak_idx,
                    'recovery_bars': dd_end - max_dd_idx,
                    'peak_equity': float(equity_arr[dd_peak_idx]),
                    'trough_equity': float(equity_arr[max_dd_idx])
                })
                in_drawdown = False
        
        # Handle ongoing drawdown
        if in_drawdown:
            max_dd = np.min(drawdown[dd_start:])
            max_dd_idx = dd_start + np.argmin(drawdown[dd_start:])
            drawdowns.append({
                'start_idx': dd_peak_idx,
                'trough_idx': max_dd_idx,
                'end_idx': None,
                'start_time': str(timestamps[dd_peak_idx]) if timestamps is not None else None,
                'trough_time': str(timestamps[max_dd_idx]) if timestamps is not None else None,
                'end_time': None,
                'max_drawdown_pct': float(abs(max_dd)),
                'duration_bars': len(equity_arr) - dd_peak_idx,
                'recovery_bars': None,
                'peak_equity': float(equity_arr[dd_peak_idx]),
                'trough_equity': float(equity_arr[max_dd_idx]),
                'ongoing': True
            })
        
        return sorted(drawdowns, key=lambda x: x['max_drawdown_pct'], reverse=True)
    
    def calculate_var(self, confidence: float = 0.95) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR).
        
        Args:
            confidence: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary with VaR and CVaR values
        """
        trades = self.data_service.trades
        
        if len(trades) == 0 or 'pnl' not in trades.columns:
            return {'var_95': 0.0, 'var_99': 0.0, 'cvar_95': 0.0, 'cvar_99': 0.0}
        
        pnl = trades['pnl'].values
        
        # VaR at different confidence levels
        var_95 = np.percentile(pnl, 5)
        var_99 = np.percentile(pnl, 1)
        
        # CVaR (Expected Shortfall)
        cvar_95 = pnl[pnl <= var_95].mean() if len(pnl[pnl <= var_95]) > 0 else var_95
        cvar_99 = pnl[pnl <= var_99].mean() if len(pnl[pnl <= var_99]) > 0 else var_99
        
        return {
            'var_95': float(var_95),
            'var_99': float(var_99),
            'cvar_95': float(cvar_95),
            'cvar_99': float(cvar_99)
        }
    
    def analyze_sequences(self) -> Dict[str, Any]:
        """Analyze win/loss sequences (streaks)."""
        trades = self.data_service.trades
        
        if len(trades) == 0 or 'pnl' not in trades.columns:
            return {
                'max_win_streak': 0,
                'max_loss_streak': 0,
                'current_streak': 0,
                'current_streak_type': 'none',
                'win_streak_distribution': [],
                'loss_streak_distribution': []
            }
        
        # Get sequence of wins/losses
        results = (trades['pnl'] > 0).values
        
        # Calculate streaks
        win_streaks = []
        loss_streaks = []
        current_streak = 0
        current_type = None
        
        for is_win in results:
            if current_type is None:
                current_type = is_win
                current_streak = 1
            elif is_win == current_type:
                current_streak += 1
            else:
                # Streak ended
                if current_type:
                    win_streaks.append(current_streak)
                else:
                    loss_streaks.append(current_streak)
                current_type = is_win
                current_streak = 1
        
        # Add final streak
        if current_type is not None:
            if current_type:
                win_streaks.append(current_streak)
            else:
                loss_streaks.append(current_streak)
        
        # Distribution
        win_dist = self._streak_distribution(win_streaks)
        loss_dist = self._streak_distribution(loss_streaks)
        
        return {
            'max_win_streak': max(win_streaks) if win_streaks else 0,
            'max_loss_streak': max(loss_streaks) if loss_streaks else 0,
            'avg_win_streak': np.mean(win_streaks) if win_streaks else 0,
            'avg_loss_streak': np.mean(loss_streaks) if loss_streaks else 0,
            'current_streak': current_streak,
            'current_streak_type': 'win' if current_type else 'loss' if current_type is False else 'none',
            'win_streak_distribution': win_dist,
            'loss_streak_distribution': loss_dist
        }
    
    def _streak_distribution(self, streaks: List[int]) -> List[Dict[str, int]]:
        """Calculate streak length distribution."""
        if not streaks:
            return []
        
        dist = {}
        for s in streaks:
            dist[s] = dist.get(s, 0) + 1
        
        return [{'length': k, 'count': v} for k, v in sorted(dist.items())]
    
    def monte_carlo_simulation(
        self,
        n_simulations: int = 1000,
        shuffle_trades: bool = True
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation by shuffling trade order.
        
        Args:
            n_simulations: Number of simulations to run
            shuffle_trades: If True, shuffle trade order; if False, sample with replacement
            
        Returns:
            Distribution of possible outcomes
        """
        trades = self.data_service.trades
        metrics = self.data_service.metrics
        
        if len(trades) == 0 or 'pnl' not in trades.columns:
            return {
                'simulations': n_simulations,
                'percentiles': {},
                'current_rank': 0
            }
        
        pnl_values = trades['pnl'].values
        initial_balance = metrics.get('initial_balance', 10000)
        if initial_balance is None:
            initial_balance = 10000
        
        final_balances = []
        max_drawdowns = []
        
        for _ in range(n_simulations):
            if shuffle_trades:
                sim_pnl = np.random.permutation(pnl_values)
            else:
                sim_pnl = np.random.choice(pnl_values, size=len(pnl_values), replace=True)
            
            # Calculate equity curve
            equity = initial_balance + np.cumsum(sim_pnl)
            equity = np.insert(equity, 0, initial_balance)
            
            final_balances.append(equity[-1])
            
            # Max drawdown
            peak = np.maximum.accumulate(equity)
            dd = (equity - peak) / peak
            max_drawdowns.append(abs(np.min(dd)) * 100)
        
        # Percentiles
        percentiles = {
            '5th': np.percentile(final_balances, 5),
            '25th': np.percentile(final_balances, 25),
            '50th': np.percentile(final_balances, 50),
            '75th': np.percentile(final_balances, 75),
            '95th': np.percentile(final_balances, 95)
        }
        
        # Where does actual result fall?
        actual_final = metrics.get('final_balance', initial_balance + pnl_values.sum())
        if actual_final is None:
            actual_final = initial_balance + pnl_values.sum()
        rank = (np.array(final_balances) < actual_final).sum() / n_simulations * 100
        
        return {
            'simulations': n_simulations,
            'percentiles': percentiles,
            'current_rank': float(rank),
            'actual_final': float(actual_final),
            'mean_final': float(np.mean(final_balances)),
            'std_final': float(np.std(final_balances)),
            'mean_max_dd': float(np.mean(max_drawdowns)),
            'histogram': self._create_histogram(final_balances, 20)
        }
    
    def _create_histogram(self, values: List[float], bins: int = 20) -> List[Dict[str, Any]]:
        """Create histogram data for charting."""
        hist, bin_edges = np.histogram(values, bins=bins)
        return [
            {
                'bin_start': float(bin_edges[i]),
                'bin_end': float(bin_edges[i + 1]),
                'count': int(hist[i])
            }
            for i in range(len(hist))
        ]
    
    def compare_vs_buy_hold(self, btc_prices: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Compare strategy vs buy and hold.
        
        Args:
            btc_prices: Optional list of BTC prices (same length as equity curve)
            
        Returns:
            Comparison metrics
        """
        equity = self.data_service.equity
        metrics = self.data_service.metrics
        
        if len(equity) == 0 or 'equity' not in equity.columns:
            return {}
        
        equity_arr = equity['equity'].values
        initial_balance = equity_arr[0]
        
        # Strategy returns
        strategy_return = (equity_arr[-1] - initial_balance) / initial_balance * 100
        
        # If BTC prices not provided, estimate
        if btc_prices is None or len(btc_prices) != len(equity_arr):
            # Can't compare without price data
            return {
                'strategy_return_pct': float(strategy_return),
                'buy_hold_return_pct': None,
                'outperformance_pct': None,
                'strategy_equity': equity_arr.tolist()[-100:] if len(equity_arr) > 100 else equity_arr.tolist(),
            }
        
        btc_arr = np.array(btc_prices)
        buy_hold_return = (btc_arr[-1] - btc_arr[0]) / btc_arr[0] * 100
        
        # Normalized equity curves for comparison
        strategy_normalized = equity_arr / initial_balance * 100
        buy_hold_normalized = btc_arr / btc_arr[0] * 100
        
        return {
            'strategy_return_pct': float(strategy_return),
            'buy_hold_return_pct': float(buy_hold_return),
            'outperformance_pct': float(strategy_return - buy_hold_return),
            'strategy_equity': strategy_normalized.tolist()[-500:] if len(strategy_normalized) > 500 else strategy_normalized.tolist(),
            'buy_hold_equity': buy_hold_normalized.tolist()[-500:] if len(buy_hold_normalized) > 500 else buy_hold_normalized.tolist()
        }
    
    def generate_returns_heatmap(self) -> Dict[str, List[List[float]]]:
        """
        Generate calendar heatmap data for daily returns.
        
        Returns:
            Dictionary with years and their monthly/daily returns
        """
        trades = self.data_service.trades
        
        if len(trades) == 0 or 'entry_time' not in trades.columns:
            return {}
        
        df = trades.copy()
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['date'] = df['entry_time'].dt.date
        
        # Daily P&L
        daily = df.groupby('date')['pnl'].sum().reset_index()
        daily.columns = ['date', 'pnl']
        daily['date'] = pd.to_datetime(daily['date'])
        
        # Group by year
        result = {}
        for year in daily['date'].dt.year.unique():
            year_data = daily[daily['date'].dt.year == year]
            
            # Create 12x31 grid (months x days)
            grid = [[None for _ in range(31)] for _ in range(12)]
            
            for _, row in year_data.iterrows():
                month = row['date'].month - 1  # 0-indexed
                day = row['date'].day - 1  # 0-indexed
                grid[month][day] = float(row['pnl'])
            
            result[str(year)] = grid
        
        return result
    
    def get_rolling_metrics(self, window: int = 30) -> List[Dict[str, Any]]:
        """
        Calculate rolling performance metrics.
        
        Args:
            window: Rolling window size in trades
            
        Returns:
            List of rolling metric snapshots
        """
        trades = self.data_service.trades
        
        if len(trades) < window or 'pnl' not in trades.columns:
            return []
        
        df = trades.copy()
        df = df.sort_values('entry_time' if 'entry_time' in df.columns else df.index)
        
        results = []
        for i in range(window, len(df) + 1):
            window_df = df.iloc[i - window:i]
            
            pnl = window_df['pnl'].values
            win_rate = (pnl > 0).sum() / len(pnl) * 100
            
            # Sharpe-like ratio for trade returns
            if np.std(pnl) > 0:
                sharpe_like = np.mean(pnl) / np.std(pnl) * np.sqrt(252)  # Annualized
            else:
                sharpe_like = 0.0
            
            entry_time = window_df['entry_time'].iloc[-1] if 'entry_time' in window_df.columns else None
            
            results.append({
                'index': i,
                'timestamp': entry_time.isoformat() if entry_time else None,
                'win_rate': float(win_rate),
                'sharpe_like': float(sharpe_like),
                'avg_pnl': float(np.mean(pnl)),
                'total_pnl': float(np.sum(pnl))
            })
        
        return results
