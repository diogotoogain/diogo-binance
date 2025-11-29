/**
 * MEGA Backtest Analytics Dashboard - Main JavaScript
 */

// API Base URL
const API_BASE = '';

// Current page state
let currentPage = 1;
let perPage = 25;
let sortBy = 'entry_time';
let sortDesc = true;
let filters = {};

// Data cache
let summaryData = null;
let equityData = null;

/**
 * Initialize dashboard
 */
document.addEventListener('DOMContentLoaded', () => {
    // Determine current page and initialize
    const path = window.location.pathname;
    
    if (path === '/' || path === '/index.html') {
        initDashboard();
    } else if (path === '/trades') {
        initTradesPage();
    } else if (path === '/performance') {
        initPerformancePage();
    } else if (path === '/temporal') {
        initTemporalPage();
    } else if (path === '/risk') {
        initRiskPage();
    } else if (path === '/model') {
        initModelPage();
    } else if (path === '/compare') {
        initComparePage();
    }
    
    // Setup global event listeners
    setupGlobalListeners();
});

/**
 * Setup global event listeners
 */
function setupGlobalListeners() {
    // Export CSV button
    const exportBtn = document.getElementById('export-csv');
    if (exportBtn) {
        exportBtn.addEventListener('click', exportCSV);
    }
    
    // Refresh button
    const refreshBtn = document.getElementById('refresh-data');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            window.location.reload();
        });
    }
}

/**
 * Fetch data from API
 */
async function fetchAPI(endpoint, params = {}) {
    const url = new URL(API_BASE + endpoint, window.location.origin);
    Object.entries(params).forEach(([key, value]) => {
        if (value !== null && value !== undefined && value !== '') {
            url.searchParams.append(key, value);
        }
    });
    
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return await response.json();
    } catch (error) {
        console.error(`API Error (${endpoint}):`, error);
        return null;
    }
}

/**
 * Format number as currency
 */
function formatCurrency(value, decimals = 2) {
    if (value === null || value === undefined || isNaN(value)) return '--';
    const prefix = value >= 0 ? '$' : '-$';
    return prefix + Math.abs(value).toLocaleString('en-US', { 
        minimumFractionDigits: decimals, 
        maximumFractionDigits: decimals 
    });
}

/**
 * Format number as percentage
 */
function formatPercent(value, decimals = 2) {
    if (value === null || value === undefined || isNaN(value)) return '--';
    const prefix = value >= 0 ? '+' : '';
    return prefix + value.toFixed(decimals) + '%';
}

/**
 * Format number with thousands separator
 */
function formatNumber(value, decimals = 0) {
    if (value === null || value === undefined || isNaN(value)) return '--';
    return value.toLocaleString('en-US', { 
        minimumFractionDigits: decimals, 
        maximumFractionDigits: decimals 
    });
}

/**
 * Format date/time
 */
function formatDateTime(value) {
    if (!value) return '--';
    const date = new Date(value);
    return date.toLocaleString('en-US', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}

/**
 * Apply CSS class based on value sign
 */
function getSignClass(value) {
    if (value > 0) return 'positive';
    if (value < 0) return 'negative';
    return '';
}

// ========== Dashboard Page ==========

async function initDashboard() {
    // Load summary data
    summaryData = await fetchAPI('/api/summary');
    if (summaryData) {
        updateKPIs(summaryData);
        updateSummaryStats(summaryData);
    }
    
    // Load and render equity curve
    equityData = await fetchAPI('/api/equity');
    if (equityData && equityData.length > 0) {
        renderEquityChart(equityData);
    }
}

function updateKPIs(data) {
    const setKPI = (id, value, formatter = formatNumber) => {
        const el = document.getElementById(id);
        if (el) {
            el.textContent = formatter(value);
            el.className = 'kpi-value ' + getSignClass(value);
        }
    };
    
    setKPI('total-return', data.total_return_pct, formatPercent);
    setKPI('sharpe-ratio', data.sharpe_ratio, v => v?.toFixed(2) || '--');
    setKPI('sortino-ratio', data.sortino_ratio, v => v?.toFixed(2) || '--');
    setKPI('max-drawdown', -Math.abs(data.max_drawdown || 0), formatPercent);
    setKPI('win-rate', data.win_rate, formatPercent);
    setKPI('profit-factor', data.profit_factor, v => v?.toFixed(2) || '--');
}

function updateSummaryStats(data) {
    document.getElementById('initial-balance')?.textContent && 
        (document.getElementById('initial-balance').textContent = formatCurrency(data.initial_balance));
    document.getElementById('final-balance')?.textContent && 
        (document.getElementById('final-balance').textContent = formatCurrency(data.final_balance));
    document.getElementById('total-pnl')?.textContent && 
        (document.getElementById('total-pnl').textContent = formatCurrency(data.total_pnl));
    
    document.getElementById('total-trades')?.textContent && 
        (document.getElementById('total-trades').textContent = formatNumber(data.total_trades));
    document.getElementById('winning-trades')?.textContent && 
        (document.getElementById('winning-trades').textContent = formatNumber(data.winning_trades));
    document.getElementById('losing-trades')?.textContent && 
        (document.getElementById('losing-trades').textContent = formatNumber(data.losing_trades));
    
    document.getElementById('start-date')?.textContent && 
        (document.getElementById('start-date').textContent = data.start_date?.split('T')[0] || '--');
    document.getElementById('end-date')?.textContent && 
        (document.getElementById('end-date').textContent = data.end_date?.split('T')[0] || '--');
    document.getElementById('equity-points')?.textContent && 
        (document.getElementById('equity-points').textContent = formatNumber(data.equity_points));
}

// ========== Trades Page ==========

async function initTradesPage() {
    // Setup filter listeners
    document.getElementById('apply-filters')?.addEventListener('click', () => {
        currentPage = 1;
        loadTrades();
    });
    
    document.getElementById('prev-page')?.addEventListener('click', () => {
        if (currentPage > 1) {
            currentPage--;
            loadTrades();
        }
    });
    
    document.getElementById('next-page')?.addEventListener('click', () => {
        currentPage++;
        loadTrades();
    });
    
    // Table header click for sorting
    document.querySelectorAll('#trades-table th[data-sort]').forEach(th => {
        th.addEventListener('click', () => {
            const newSort = th.dataset.sort;
            if (sortBy === newSort) {
                sortDesc = !sortDesc;
            } else {
                sortBy = newSort;
                sortDesc = true;
            }
            loadTrades();
        });
    });
    
    // Load initial data
    loadTrades();
}

async function loadTrades() {
    const direction = document.getElementById('filter-direction')?.value;
    const result = document.getElementById('filter-result')?.value;
    perPage = parseInt(document.getElementById('filter-per-page')?.value || '25');
    
    const params = {
        page: currentPage,
        per_page: perPage,
        sort_by: sortBy,
        sort_desc: sortDesc
    };
    
    if (direction) params.direction = direction;
    if (result) params.result_filter = result;
    
    const data = await fetchAPI('/api/trades', params);
    if (data) {
        renderTradesTable(data.trades);
        updatePagination(data);
        document.getElementById('trades-count').textContent = `${data.total} trades`;
    }
}

function renderTradesTable(trades) {
    const tbody = document.getElementById('trades-tbody');
    if (!tbody) return;
    
    if (!trades || trades.length === 0) {
        tbody.innerHTML = '<tr><td colspan="12" class="loading">No trades found</td></tr>';
        return;
    }
    
    let totalPnl = 0;
    let totalFees = 0;
    
    tbody.innerHTML = trades.map(trade => {
        totalPnl += trade.pnl || 0;
        totalFees += trade.fees || 0;
        
        const dirClass = trade.direction === 1 ? 'direction-long' : 'direction-short';
        const dirText = trade.direction === 1 ? '🟢 LONG' : '🔴 SHORT';
        const pnlClass = getSignClass(trade.pnl);
        
        return `
            <tr>
                <td>${trade.trade_id || '--'}</td>
                <td>${formatDateTime(trade.entry_time)}</td>
                <td>${formatDateTime(trade.exit_time)}</td>
                <td>${trade.duration || '--'}</td>
                <td><span class="${dirClass}">${dirText}</span></td>
                <td>${formatCurrency(trade.entry_price)}</td>
                <td>${formatCurrency(trade.exit_price)}</td>
                <td>${formatCurrency(trade.size)}</td>
                <td class="${pnlClass}">${formatCurrency(trade.pnl)}</td>
                <td class="${pnlClass}">${formatPercent(trade.pnl_pct)}</td>
                <td>${formatCurrency(trade.fees)}</td>
                <td>${trade.exit_reason || '--'}</td>
            </tr>
        `;
    }).join('');
    
    // Update footer
    document.getElementById('total-pnl-footer')?.textContent && 
        (document.getElementById('total-pnl-footer').textContent = formatCurrency(totalPnl));
    document.getElementById('total-fees-footer')?.textContent && 
        (document.getElementById('total-fees-footer').textContent = formatCurrency(totalFees));
    document.getElementById('avg-pnl-footer')?.textContent && 
        (document.getElementById('avg-pnl-footer').textContent = formatPercent(trades.length > 0 ? totalPnl / trades.length : 0));
}

function updatePagination(data) {
    document.getElementById('page-info').textContent = `Page ${data.page} of ${data.pages}`;
    document.getElementById('prev-page').disabled = data.page <= 1;
    document.getElementById('next-page').disabled = data.page >= data.pages;
}

// ========== Performance Page ==========

async function initPerformancePage() {
    // Period selector
    document.querySelectorAll('.period-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.period-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadPeriodData(btn.dataset.period);
        });
    });
    
    // Load initial data
    loadPeriodData('day');
    loadDistributionChart();
    loadHeatmap();
}

async function loadPeriodData(period) {
    const data = await fetchAPI(`/api/analysis/period/${period}`);
    if (data) {
        renderPeriodTable(data);
    }
}

function renderPeriodTable(data) {
    const tbody = document.getElementById('period-tbody');
    if (!tbody) return;
    
    if (!data || data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7">No data available</td></tr>';
        return;
    }
    
    tbody.innerHTML = data.map(row => {
        const pnlClass = getSignClass(row.total_pnl);
        return `
            <tr>
                <td>${row.period}</td>
                <td>${row.trades}</td>
                <td class="${pnlClass}">${formatCurrency(row.total_pnl)}</td>
                <td class="${pnlClass}">${formatCurrency(row.avg_pnl)}</td>
                <td>${formatPercent(row.win_rate)}</td>
                <td>${row.longs}</td>
                <td>${row.shorts}</td>
            </tr>
        `;
    }).join('');
}

async function loadDistributionChart() {
    // Fetch trades for distribution
    const data = await fetchAPI('/api/trades', { per_page: 100 });
    if (data && data.trades) {
        renderDistributionChart(data.trades.map(t => t.pnl));
    }
}

async function loadHeatmap() {
    const data = await fetchAPI('/api/analysis/heatmap');
    if (data) {
        renderHeatmap(data);
    }
}

function renderHeatmap(data) {
    const container = document.getElementById('heatmap-container');
    if (!container) return;
    
    let html = '';
    Object.keys(data).sort().forEach(year => {
        html += `<h4 style="color: var(--text-secondary); margin: 15px 0 5px;">${year}</h4>`;
        const grid = data[year];
        grid.forEach((row, monthIdx) => {
            html += '<div class="heatmap-row">';
            row.forEach((value, dayIdx) => {
                let cellClass = 'heatmap-neutral';
                if (value !== null) {
                    cellClass = value > 0 ? 'heatmap-positive' : value < 0 ? 'heatmap-negative' : 'heatmap-neutral';
                }
                const opacity = value !== null ? Math.min(Math.abs(value) / 100, 1) : 0.3;
                html += `<div class="heatmap-cell ${cellClass}" style="opacity: ${opacity}" title="${value !== null ? formatCurrency(value) : 'No data'}"></div>`;
            });
            html += '</div>';
        });
    });
    
    container.innerHTML = html || '<p class="loading">No heatmap data available</p>';
}

// ========== Temporal Page ==========

async function initTemporalPage() {
    const hourData = await fetchAPI('/api/analysis/hour');
    const dowData = await fetchAPI('/api/analysis/day-of-week');
    const periodData = await fetchAPI('/api/analysis/period/day');
    
    if (hourData) renderHourChart(hourData);
    if (dowData) renderDowChart(dowData);
    if (periodData) {
        renderBestWorstDays(periodData);
    }
}

function renderBestWorstDays(data) {
    const sorted = [...data].sort((a, b) => b.total_pnl - a.total_pnl);
    
    const bestList = document.getElementById('best-days-list');
    const worstList = document.getElementById('worst-days-list');
    
    if (bestList) {
        const best = sorted.slice(0, 5);
        bestList.innerHTML = best.map((d, i) => `
            <div class="stat-row">
                <span class="stat-label">${i + 1}. ${d.period}</span>
                <span class="stat-value positive">${formatCurrency(d.total_pnl)}</span>
            </div>
        `).join('');
    }
    
    if (worstList) {
        const worst = sorted.slice(-5).reverse();
        worstList.innerHTML = worst.map((d, i) => `
            <div class="stat-row">
                <span class="stat-label">${i + 1}. ${d.period}</span>
                <span class="stat-value negative">${formatCurrency(d.total_pnl)}</span>
            </div>
        `).join('');
    }
}

// ========== Risk Page ==========

async function initRiskPage() {
    const varData = await fetchAPI('/api/analysis/var');
    const seqData = await fetchAPI('/api/analysis/sequences');
    const ddData = await fetchAPI('/api/analysis/drawdowns');
    
    if (varData) {
        document.getElementById('var-95').textContent = formatCurrency(varData.var_95);
        document.getElementById('var-99').textContent = formatCurrency(varData.var_99);
        document.getElementById('cvar-95').textContent = formatCurrency(varData.cvar_95);
        document.getElementById('cvar-99').textContent = formatCurrency(varData.cvar_99);
    }
    
    if (seqData) {
        document.getElementById('max-win-streak').textContent = seqData.max_win_streak;
        document.getElementById('max-loss-streak').textContent = seqData.max_loss_streak;
        document.getElementById('current-streak').textContent = 
            `${seqData.current_streak} (${seqData.current_streak_type})`;
        
        renderStreakChart(seqData);
    }
    
    if (ddData) {
        renderDrawdownTable(ddData);
    }
    
    // Load equity for drawdown chart
    const equityData = await fetchAPI('/api/equity');
    if (equityData) {
        renderDrawdownChart(equityData);
    }
}

function renderDrawdownTable(data) {
    const tbody = document.getElementById('drawdown-tbody');
    if (!tbody) return;
    
    if (!data || data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6">No significant drawdowns</td></tr>';
        return;
    }
    
    tbody.innerHTML = data.slice(0, 10).map(dd => `
        <tr>
            <td>${formatDateTime(dd.start_time)}</td>
            <td>${formatDateTime(dd.trough_time)}</td>
            <td>${dd.end_time ? formatDateTime(dd.end_time) : 'Ongoing'}</td>
            <td class="negative">${formatPercent(-dd.max_drawdown_pct)}</td>
            <td>${dd.duration_bars} bars</td>
            <td>${dd.recovery_bars ? dd.recovery_bars + ' bars' : '--'}</td>
        </tr>
    `).join('');
}

// ========== Model Page ==========

async function initModelPage() {
    const olStats = await fetchAPI('/api/ol-stats');
    const driftEvents = await fetchAPI('/api/drift-events');
    const rollingData = await fetchAPI('/api/analysis/rolling');
    
    if (olStats) {
        document.getElementById('samples-seen').textContent = formatNumber(olStats.n_samples_seen);
        document.getElementById('model-accuracy').textContent = 
            olStats.accuracy ? formatPercent(olStats.accuracy * 100) : '--';
    }
    
    if (driftEvents) {
        document.getElementById('drifts-count').textContent = driftEvents.length;
        renderDriftTable(driftEvents);
    }
    
    if (rollingData) {
        renderRollingChart(rollingData);
    }
}

function renderDriftTable(events) {
    const tbody = document.getElementById('drift-tbody');
    if (!tbody) return;
    
    if (!events || events.length === 0) {
        tbody.innerHTML = '<tr><td colspan="3">No drift events detected</td></tr>';
        return;
    }
    
    tbody.innerHTML = events.slice(0, 20).map(event => `
        <tr>
            <td>${event.step}</td>
            <td>${formatDateTime(event.timestamp)}</td>
            <td>${event.metrics?.join(', ') || '--'}</td>
        </tr>
    `).join('');
}

// ========== Compare Page ==========

async function initComparePage() {
    document.getElementById('run-monte-carlo')?.addEventListener('click', runMonteCarlo);
    
    // Load initial data
    const summaryData = await fetchAPI('/api/summary');
    if (summaryData) {
        document.getElementById('strategy-return').textContent = formatPercent(summaryData.total_return_pct);
    }
    
    // Run Monte Carlo
    runMonteCarlo();
}

async function runMonteCarlo() {
    const btn = document.getElementById('run-monte-carlo');
    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Running...';
    }
    
    const data = await fetchAPI('/api/analysis/monte-carlo', { simulations: 1000 });
    
    if (btn) {
        btn.disabled = false;
        btn.textContent = 'Run 1000 Simulations';
    }
    
    if (data) {
        document.getElementById('mc-5th').textContent = formatCurrency(data.percentiles['5th']);
        document.getElementById('mc-50th').textContent = formatCurrency(data.percentiles['50th']);
        document.getElementById('mc-95th').textContent = formatCurrency(data.percentiles['95th']);
        document.getElementById('mc-rank').textContent = formatPercent(data.current_rank);
        
        renderMonteCarloChart(data);
    }
}

// ========== Export ==========

async function exportCSV() {
    const data = await fetchAPI('/api/export/csv');
    if (data && data.content) {
        const blob = new Blob([data.content], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = data.filename || 'trades_export.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    }
}
