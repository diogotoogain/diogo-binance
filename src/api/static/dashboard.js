/**
 * SNAME-MR Trading Bot Dashboard
 * Real-time WebSocket updates with Chart.js
 */

// Global state
let ws = null;
let priceChart = null;
let priceData = [];
let reconnectAttempts = 0;
const maxReconnectAttempts = 10;
const reconnectDelay = 3000;
let currentPriceValue = 0; // Store raw numeric price
let botStartTime = null; // Track bot start time for uptime
let uptimeInterval = null; // Interval for uptime updates
let currentPosition = null; // Store current position data
let lastPingTime = 0; // Track last ping time for latency
let latencyMs = 0; // Current latency in ms
let isPaused = false; // Bot pause state
let soundEnabled = true; // Sound alerts enabled
let darkMode = true; // Dark/Light mode
let logFilter = 'all'; // Log filter: all, BUY, SELL, ERROR, INFO
let equityHistory = []; // Equity curve data
let tradeStats = { // Trade statistics
    bestTrade: { pnl: 0, percent: 0 },
    worstTrade: { pnl: 0, percent: 0 },
    totalGains: 0,
    totalLosses: 0,
    tradeTimes: [],
    tradesPerHour: 0
};
let metaDecisionHistory = []; // Meta-controller decision history

// Note: Initialization is handled at the end of the file
// to ensure all functions are defined before use

/**
 * Initialize Chart.js price chart
 */
let chartType = 'line'; // 'line' or 'candlestick'
let currentTimeframe = '1m';
let entryPrice = 0;
let stopLossPrice = 0;
let takeProfitPrice = 0;

// Horizontal line annotation plugin - registered once
const horizontalLinePlugin = {
    id: 'horizontalLines',
    afterDraw: (chart) => {
        if (!currentPosition || !currentPosition.has_position) return;
        
        const ctx = chart.ctx;
        const yAxis = chart.scales.y;
        const chartArea = chart.chartArea;
        
        // Draw entry price line (yellow)
        if (entryPrice > 0) {
            const yEntry = yAxis.getPixelForValue(entryPrice);
            if (yEntry >= chartArea.top && yEntry <= chartArea.bottom) {
                ctx.save();
                ctx.beginPath();
                ctx.moveTo(chartArea.left, yEntry);
                ctx.lineTo(chartArea.right, yEntry);
                ctx.lineWidth = 2;
                ctx.strokeStyle = '#fcd34d';
                ctx.setLineDash([5, 5]);
                ctx.stroke();
                ctx.fillStyle = '#fcd34d';
                ctx.font = '10px Inter';
                ctx.fillText(`Entry: $${entryPrice.toLocaleString()}`, chartArea.right - 120, yEntry - 5);
                ctx.restore();
            }
        }
        
        // Draw stop loss line (red)
        if (stopLossPrice > 0) {
            const ySL = yAxis.getPixelForValue(stopLossPrice);
            if (ySL >= chartArea.top && ySL <= chartArea.bottom) {
                ctx.save();
                ctx.beginPath();
                ctx.moveTo(chartArea.left, ySL);
                ctx.lineTo(chartArea.right, ySL);
                ctx.lineWidth = 2;
                ctx.strokeStyle = '#ef4444';
                ctx.setLineDash([5, 5]);
                ctx.stroke();
                ctx.fillStyle = '#ef4444';
                ctx.font = '10px Inter';
                ctx.fillText(`SL: $${stopLossPrice.toLocaleString()}`, chartArea.right - 100, ySL - 5);
                ctx.restore();
            }
        }
        
        // Draw take profit line (green)
        if (takeProfitPrice > 0) {
            const yTP = yAxis.getPixelForValue(takeProfitPrice);
            if (yTP >= chartArea.top && yTP <= chartArea.bottom) {
                ctx.save();
                ctx.beginPath();
                ctx.moveTo(chartArea.left, yTP);
                ctx.lineTo(chartArea.right, yTP);
                ctx.lineWidth = 2;
                ctx.strokeStyle = '#22c55e';
                ctx.setLineDash([5, 5]);
                ctx.stroke();
                ctx.fillStyle = '#22c55e';
                ctx.font = '10px Inter';
                ctx.fillText(`TP: $${takeProfitPrice.toLocaleString()}`, chartArea.right - 100, yTP - 5);
                ctx.restore();
            }
        }
    }
};

// Register plugin once
let pluginRegistered = false;

function initChart() {
    const ctx = document.getElementById('priceChart').getContext('2d');
    
    // Register plugin only once
    if (!pluginRegistered) {
        Chart.register(horizontalLinePlugin);
        pluginRegistered = true;
    }
    
    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'BTC/USDT',
                data: [],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 0
            },
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: '#1a1a1a',
                    titleColor: '#fff',
                    bodyColor: '#9ca3af',
                    borderColor: '#3a3a3a',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            return `$${context.parsed.y.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: false
                },
                y: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#6b7280',
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    }
                }
            }
        }
    });
}

/**
 * Initialize equity chart
 */
let equityChart = null;

function initEquityChart() {
    const ctx = document.getElementById('equityChart');
    if (!ctx) return;
    
    equityChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Equity',
                data: [],
                borderColor: '#22c55e',
                backgroundColor: 'rgba(34, 197, 94, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { display: false },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: {
                        color: '#6b7280',
                        callback: (v) => '$' + v.toLocaleString()
                    }
                }
            }
        }
    });
}

/**
 * Update equity chart with new balance data
 */
function updateEquityChart(balance) {
    if (!equityChart || !balance) return;
    
    const totalBalance = balance.total_usdt || 0;
    if (totalBalance <= 0) return;
    
    equityHistory.push({
        time: new Date().toLocaleTimeString(),
        value: totalBalance
    });
    
    // Keep only last 50 points
    if (equityHistory.length > 50) {
        equityHistory.shift();
    }
    
    equityChart.data.labels = equityHistory.map(e => e.time);
    equityChart.data.datasets[0].data = equityHistory.map(e => e.value);
    equityChart.update('none');
    
    // Calculate drawdown
    const peak = Math.max(...equityHistory.map(e => e.value));
    const current = totalBalance;
    const drawdown = peak > 0 ? ((peak - current) / peak) * 100 : 0;
    
    const drawdownEl = document.getElementById('perf-drawdown');
    if (drawdownEl) {
        drawdownEl.textContent = `-${drawdown.toFixed(2)}%`;
    }
}

/**
 * Connect to WebSocket server
 */
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    
    updateConnectionStatus('connecting');
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        reconnectAttempts = 0;
        updateConnectionStatus('connected');
        addLog('INFO', 'Conectado ao servidor');
        showNotification('🔗 Conectado', 'Conexão WebSocket estabelecida', 'success');
    };
    
    ws.onmessage = (event) => {
        try {
            // Calculate latency if this is a pong response
            if (event.data === 'pong' && lastPingTime > 0) {
                latencyMs = Date.now() - lastPingTime;
                updateLatencyDisplay();
                return;
            }
            const message = JSON.parse(event.data);
            handleMessage(message);
        } catch (e) {
            console.error('Error parsing message:', e);
        }
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
        updateConnectionStatus('disconnected');
        addLog('WARNING', 'Desconectado do servidor');
        scheduleReconnect();
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        updateConnectionStatus('error');
    };
}

/**
 * Schedule WebSocket reconnection
 */
function scheduleReconnect() {
    if (reconnectAttempts < maxReconnectAttempts) {
        reconnectAttempts++;
        console.log(`Reconnecting in ${reconnectDelay}ms (attempt ${reconnectAttempts}/${maxReconnectAttempts})`);
        setTimeout(connectWebSocket, reconnectDelay);
    } else {
        addLog('ERROR', 'Falha ao reconectar após múltiplas tentativas');
    }
}

/**
 * Update connection status indicator
 */
function updateConnectionStatus(status) {
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    const indicator = document.getElementById('connection-indicator');
    
    switch (status) {
        case 'connected':
            dot.className = 'status-dot status-live pulse-green';
            text.textContent = 'LIVE';
            text.className = 'text-sm text-green-400';
            if (indicator) {
                indicator.className = 'connection-indicator connected';
                indicator.title = 'Conectado';
            }
            break;
        case 'connecting':
            dot.className = 'status-dot status-offline';
            text.textContent = 'Conectando...';
            text.className = 'text-sm text-yellow-400';
            if (indicator) {
                indicator.className = 'connection-indicator reconnecting';
                indicator.title = 'Reconectando...';
            }
            break;
        case 'disconnected':
        case 'error':
            dot.className = 'status-dot status-offline pulse-red';
            text.textContent = 'Offline';
            text.className = 'text-sm text-red-400';
            if (indicator) {
                indicator.className = 'connection-indicator disconnected';
                indicator.title = 'Desconectado';
            }
            break;
    }
}

/**
 * Update latency display
 */
function updateLatencyDisplay() {
    const latencyEl = document.getElementById('latency-display');
    if (latencyEl) {
        latencyEl.textContent = `${latencyMs}ms`;
        if (latencyMs < 100) {
            latencyEl.className = 'text-green-400';
        } else if (latencyMs < 300) {
            latencyEl.className = 'text-yellow-400';
        } else {
            latencyEl.className = 'text-red-400';
        }
    }
}

/**
 * Update last refresh timestamp
 */
function updateLastRefresh() {
    const refreshEl = document.getElementById('last-refresh');
    if (refreshEl) {
        refreshEl.textContent = new Date().toLocaleTimeString();
    }
}

/**
 * Handle incoming WebSocket messages
 */
function handleMessage(message) {
    const { type, data } = message;
    
    switch (type) {
        case 'initial':
            handleInitialData(data);
            break;
        case 'price':
            updatePrice(data);
            break;
        case 'signal':
            handleSignal(data);
            break;
        case 'position':
            updatePosition(data);
            break;
        case 'liquidation':
            addLiquidation(data);
            break;
        case 'refresh':
            handleRefresh(data);
            break;
        case 'ping':
            ws.send('ping');
            break;
    }
}

/**
 * Handle initial data load
 */
function handleInitialData(data) {
    // Price
    if (data.price) {
        currentPriceValue = data.price.current || 0;
        document.getElementById('current-price').textContent = formatCurrency(currentPriceValue);
        updatePriceChange(data.price.change_24h);
        document.getElementById('volume-24h').textContent = formatCurrency(data.price.volume_24h || 0);
        
        // Load price history into chart
        if (data.price.history && data.price.history.length > 0) {
            priceData = data.price.history;
            updateChart();
        }
    }
    
    // Balance
    if (data.balance) {
        updateBalance(data.balance);
    }
    
    // Position
    if (data.position) {
        updatePosition(data.position);
    }
    
    // Strategies
    if (data.strategies) {
        updateStrategies(data.strategies);
    }
    
    // Meta Controller
    if (data.meta_controller) {
        updateMetaController(data.meta_controller);
    }
    
    // Trades
    if (data.trades) {
        updateTrades(data.trades);
    }
    
    // Liquidations
    if (data.liquidations) {
        updateLiquidations(data.liquidations);
    }
    
    // Logs
    if (data.logs && data.logs.length > 0) {
        const container = document.getElementById('logs-container');
        container.innerHTML = '';
        data.logs.forEach(log => addLogEntry(log));
    }
    
    // System - Start uptime timer
    if (data.system) {
        // Parse the uptime string and start the timer
        const uptimeStr = data.system.uptime || '0:00:00';
        botStartTime = parseUptimeToStartTime(uptimeStr);
        startUptimeTimer();
        
        if (data.system.demo_mode) {
            document.getElementById('testnet-badge').classList.remove('hidden');
        }
    }
    
    // Update last refresh
    updateLastRefresh();
}

/**
 * Parse uptime string to calculate start time
 */
function parseUptimeToStartTime(uptimeStr) {
    const parts = uptimeStr.split(':').map(Number);
    let seconds = 0;
    if (parts.length === 3) {
        seconds = parts[0] * 3600 + parts[1] * 60 + parts[2];
    } else if (parts.length === 2) {
        seconds = parts[0] * 60 + parts[1];
    }
    return new Date(Date.now() - seconds * 1000);
}

/**
 * Start uptime timer - updates every second
 */
function startUptimeTimer() {
    if (uptimeInterval) {
        clearInterval(uptimeInterval);
    }
    
    function updateUptime() {
        if (!botStartTime) return;
        
        const now = new Date();
        const diff = now - botStartTime;
        
        const hours = Math.floor(diff / 3600000);
        const minutes = Math.floor((diff % 3600000) / 60000);
        const seconds = Math.floor((diff % 60000) / 1000);
        
        const formatted = `${hours}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        document.getElementById('uptime').textContent = `Uptime: ${formatted}`;
    }
    
    updateUptime();
    uptimeInterval = setInterval(updateUptime, 1000);
}

/**
 * Handle refresh data
 */
function handleRefresh(data) {
    if (data.balance) updateBalance(data.balance);
    if (data.position) updatePosition(data.position);
    if (data.strategies) updateStrategies(data.strategies);
    if (data.meta_controller) updateMetaController(data.meta_controller);
    if (data.volume_24h !== undefined) {
        document.getElementById('volume-24h').textContent = formatCurrency(data.volume_24h);
    }
    updateLastRefresh();
}

/**
 * Update price display
 */
function updatePrice(data) {
    currentPriceValue = data.current || 0;
    document.getElementById('current-price').textContent = formatCurrency(currentPriceValue);
    updatePriceChange(data.change_24h);
    
    // Add to chart data
    priceData.push({
        time: data.timestamp,
        price: data.current
    });
    
    // Keep only last 100 points
    if (priceData.length > 100) {
        priceData.shift();
    }
    
    updateChart();
}

/**
 * Update price change indicator
 */
function updatePriceChange(change) {
    const el = document.getElementById('price-change');
    const formatted = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
    el.textContent = formatted;
    
    if (change > 0) {
        el.className = 'text-sm text-green-400 pb-1';
    } else if (change < 0) {
        el.className = 'text-sm text-red-400 pb-1';
    } else {
        el.className = 'text-sm text-gray-500 pb-1';
    }
}

/**
 * Update Chart.js with new data
 */
function updateChart() {
    if (!priceChart || priceData.length === 0) return;
    
    const labels = priceData.map(d => {
        const date = new Date(d.time);
        return date.toLocaleTimeString();
    });
    
    const prices = priceData.map(d => d.price);
    
    priceChart.data.labels = labels;
    priceChart.data.datasets[0].data = prices;
    priceChart.update('none');
}

/**
 * Update balance display
 */
function updateBalance(balance) {
    document.getElementById('total-balance').textContent = formatCurrency(balance.total_usdt);
    document.getElementById('available-balance').textContent = formatCurrency(balance.available);
    document.getElementById('margin-balance').textContent = formatCurrency(balance.margin_balance);
    
    const unrealizedPnl = balance.unrealized_pnl || 0;
    const unrealizedEl = document.getElementById('unrealized-pnl');
    unrealizedEl.textContent = formatCurrency(unrealizedPnl, true);
    unrealizedEl.className = `text-sm font-medium ${unrealizedPnl >= 0 ? 'text-green-400' : 'text-red-400'}`;
    
    // Update equity chart
    updateEquityChart(balance);
}

/**
 * Update position display
 */
function updatePosition(position) {
    const noPosition = document.getElementById('no-position');
    const hasPosition = document.getElementById('has-position');
    
    // Store current position globally
    currentPosition = position;
    
    if (!position || !position.has_position) {
        noPosition.classList.remove('hidden');
        hasPosition.classList.add('hidden');
        entryPrice = 0;
        stopLossPrice = 0;
        takeProfitPrice = 0;
        return;
    }
    
    noPosition.classList.add('hidden');
    hasPosition.classList.remove('hidden');
    
    const sideEl = document.getElementById('position-side');
    sideEl.textContent = position.side;
    if (position.side === 'LONG') {
        sideEl.className = 'px-3 py-1 rounded-full text-sm font-medium bg-green-500/20 text-green-400';
    } else {
        sideEl.className = 'px-3 py-1 rounded-full text-sm font-medium bg-red-500/20 text-red-400';
    }
    
    document.getElementById('position-qty').textContent = position.quantity.toFixed(4);
    document.getElementById('entry-price').textContent = formatCurrency(position.entry_price);
    
    // Store prices for chart lines
    entryPrice = position.entry_price || 0;
    stopLossPrice = position.stop_loss || 0;
    takeProfitPrice = position.take_profit || 0;
    
    const pnl = position.unrealized_pnl || 0;
    const pnlPercent = position.pnl_percent || 0;
    const pnlEl = document.getElementById('position-pnl');
    pnlEl.textContent = `${formatCurrency(pnl, true)} (${pnlPercent >= 0 ? '+' : ''}${pnlPercent.toFixed(2)}%)`;
    pnlEl.className = `font-medium ${pnl >= 0 ? 'text-green-400' : 'text-red-400'}`;
    
    // Update Stop Loss with distance
    const slEl = document.getElementById('stop-loss');
    if (position.stop_loss && position.stop_loss > 0) {
        const slDistance = ((position.stop_loss - currentPriceValue) / currentPriceValue) * 100;
        slEl.innerHTML = `${formatCurrency(position.stop_loss)} <span class="text-xs">(${slDistance.toFixed(2)}%)</span>`;
    } else {
        slEl.textContent = '$0.00';
    }
    
    // Update Take Profit with distance
    const tpEl = document.getElementById('take-profit');
    if (position.take_profit && position.take_profit > 0) {
        const tpDistance = ((position.take_profit - currentPriceValue) / currentPriceValue) * 100;
        tpEl.innerHTML = `${formatCurrency(position.take_profit)} <span class="text-xs">(+${tpDistance.toFixed(2)}%)</span>`;
    } else {
        tpEl.textContent = '$0.00';
    }
    
    // Update position progress bar
    updatePositionProgressBar(position);
    
    // Update chart with new price lines
    if (priceChart) {
        priceChart.update('none');
    }
}

/**
 * Update position progress bar visualization
 */
function updatePositionProgressBar(position) {
    const progressContainer = document.getElementById('position-progress-container');
    if (!progressContainer) return;
    
    const sl = position.stop_loss || 0;
    const tp = position.take_profit || 0;
    const current = currentPriceValue;
    const entry = position.entry_price || 0;
    
    if (sl === 0 || tp === 0 || current === 0) {
        progressContainer.innerHTML = '<div class="text-center text-gray-500 text-xs">Aguardando SL/TP</div>';
        return;
    }
    
    // Calculate progress (0% = at SL, 100% = at TP)
    const range = tp - sl;
    const progress = ((current - sl) / range) * 100;
    const clampedProgress = Math.max(0, Math.min(100, progress));
    
    progressContainer.innerHTML = `
        <div class="flex items-center gap-2 text-xs mb-1">
            <span class="text-red-400">SL</span>
            <div class="flex-1 h-2 bg-gray-700 rounded-full relative overflow-visible">
                <div class="absolute inset-0 bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 rounded-full opacity-30"></div>
                <div class="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-blue-500 rounded-full border-2 border-white shadow-lg transition-all duration-300" style="left: calc(${clampedProgress}% - 6px);"></div>
            </div>
            <span class="text-green-400">TP</span>
        </div>
        <div class="text-center text-xs text-gray-500">
            ${progress < 50 ? '↙️ Próximo do SL' : progress > 50 ? '↗️ Próximo do TP' : '⬆️ No centro'}
        </div>
    `;
}

/**
 * Update strategies display
 */
function updateStrategies(strategies) {
    // VPIN
    if (strategies.VPINDetector) {
        const vpin = strategies.VPINDetector.vpin || 0;
        document.querySelector('.vpin-value').textContent = `${(vpin * 100).toFixed(0)}%`;
        document.querySelector('.vpin-progress').style.width = `${vpin * 100}%`;
        
        // Color based on threshold
        const progressEl = document.querySelector('.vpin-progress');
        if (vpin >= 0.7) {
            progressEl.className = 'vpin-progress progress-fill bg-red-500';
        } else if (vpin >= 0.3) {
            progressEl.className = 'vpin-progress progress-fill bg-yellow-500';
        } else {
            progressEl.className = 'vpin-progress progress-fill bg-blue-500';
        }
    }
    
    // OBI
    if (strategies.OBI) {
        const obi = strategies.OBI.last_obi || 0;
        document.querySelector('.obi-value').textContent = obi.toFixed(3);
        
        // OBI ranges from -1 to 1, map to 0-100% for progress bar
        const obiPercent = ((obi + 1) / 2) * 100;
        const obiProgress = document.querySelector('.obi-progress');
        
        if (obi > 0.15) {
            obiProgress.style.width = `${obiPercent - 50}%`;
            obiProgress.style.marginLeft = '50%';
            obiProgress.className = 'obi-progress progress-fill bg-green-500';
            document.querySelector('.obi-indicator').textContent = '🟢';
        } else if (obi < -0.15) {
            obiProgress.style.width = `${50 - obiPercent}%`;
            obiProgress.style.marginLeft = `${obiPercent}%`;
            obiProgress.className = 'obi-progress progress-fill bg-red-500';
            document.querySelector('.obi-indicator').textContent = '🔴';
        } else {
            obiProgress.style.width = '10%';
            obiProgress.style.marginLeft = '45%';
            obiProgress.className = 'obi-progress progress-fill bg-gray-500';
            document.querySelector('.obi-indicator').textContent = '⚪';
        }
    }
    
    // Flow Imbalance
    if (strategies.FlowImbalance) {
        const ratio = (strategies.FlowImbalance.current_ratio || 0.5) * 100;
        document.querySelector('.flow-value').textContent = `${ratio.toFixed(0)}%`;
        document.querySelector('.flow-progress').style.width = `${ratio}%`;
        
        const flowProgress = document.querySelector('.flow-progress');
        if (ratio >= 60) {
            flowProgress.className = 'flow-progress progress-fill bg-green-500';
        } else if (ratio <= 40) {
            flowProgress.className = 'flow-progress progress-fill bg-red-500';
        } else {
            flowProgress.className = 'flow-progress progress-fill bg-orange-500';
        }
    }
    
    // Rolling VWAP
    if (strategies.RollingVWAP) {
        const vwap = strategies.RollingVWAP.current_vwap || 0;
        document.querySelector('.vwap-value').textContent = formatCurrency(vwap);
        
        // Compare with current price using stored value
        const vwapIndicator = document.querySelector('.vwap-indicator');
        
        if (currentPriceValue > 0 && vwap > 0) {
            const diff = ((currentPriceValue - vwap) / vwap) * 100;
            if (diff > 0.5) {
                vwapIndicator.textContent = `Acima (+${diff.toFixed(2)}%)`;
                vwapIndicator.className = 'vwap-indicator text-green-400';
            } else if (diff < -0.5) {
                vwapIndicator.textContent = `Abaixo (${diff.toFixed(2)}%)`;
                vwapIndicator.className = 'vwap-indicator text-red-400';
            } else {
                vwapIndicator.textContent = 'Neutro';
                vwapIndicator.className = 'vwap-indicator text-gray-400';
            }
        }
    }
    
    // Cascade Liquidation
    if (strategies.CascadeLiquidation) {
        const liqs = strategies.CascadeLiquidation.liquidations_in_window || 0;
        document.querySelector('.cascade-value').textContent = `${liqs} liq`;
        
        if (liqs >= 3) {
            document.querySelector('.cascade-last').textContent = 'Cascata detectada!';
            document.querySelector('.cascade-last').className = 'cascade-last text-red-400';
        } else {
            document.querySelector('.cascade-last').textContent = 'Nenhuma';
            document.querySelector('.cascade-last').className = 'cascade-last';
        }
    }
}

/**
 * Update meta-controller display
 */
function updateMetaController(meta) {
    const decisionEl = document.getElementById('meta-decision');
    const iconEl = decisionEl.querySelector('.decision-icon');
    const textEl = decisionEl.querySelector('.decision-text');
    
    const decision = meta.decision || 'HOLD';
    textEl.textContent = decision;
    
    // Track decision in history
    if (meta.decision && meta.decision !== 'HOLD') {
        addMetaDecision(meta);
    }
    
    switch (decision) {
        case 'BUY':
            iconEl.textContent = '🟢';
            decisionEl.className = 'inline-flex items-center gap-2 px-6 py-3 rounded-full bg-green-500/20 border border-green-500/30';
            textEl.className = 'decision-text text-xl font-bold text-green-400';
            break;
        case 'SELL':
            iconEl.textContent = '🔴';
            decisionEl.className = 'inline-flex items-center gap-2 px-6 py-3 rounded-full bg-red-500/20 border border-red-500/30';
            textEl.className = 'decision-text text-xl font-bold text-red-400';
            break;
        default:
            iconEl.textContent = '⚪';
            decisionEl.className = 'inline-flex items-center gap-2 px-6 py-3 rounded-full bg-gray-800';
            textEl.className = 'decision-text text-xl font-bold text-gray-400';
    }
    
    document.getElementById('meta-confidence').textContent = meta.confidence || '-';
    document.getElementById('meta-agreeing').textContent = meta.agreeing_count || 0;
    document.getElementById('meta-score').textContent = (meta.score || 0).toFixed(2);
    document.getElementById('meta-decisions').textContent = meta.total_decisions || 0;
    
    // Update decision history display
    updateMetaDecisionHistory();
}

/**
 * Add a meta-controller decision to history
 */
function addMetaDecision(meta) {
    const decision = {
        timestamp: new Date().toLocaleTimeString(),
        decision: meta.decision,
        confidence: meta.confidence,
        score: meta.score,
        agreeing: meta.agreeing_count
    };
    
    metaDecisionHistory.unshift(decision);
    
    // Keep only last 10 decisions
    if (metaDecisionHistory.length > 10) {
        metaDecisionHistory.pop();
    }
}

/**
 * Update meta-controller decision history display
 */
function updateMetaDecisionHistory() {
    const container = document.getElementById('meta-decision-history');
    if (!container) return;
    
    if (metaDecisionHistory.length === 0) {
        container.innerHTML = '<p class="text-gray-500 text-xs text-center py-2">Nenhuma decisão ainda</p>';
        return;
    }
    
    container.innerHTML = metaDecisionHistory.map(d => {
        const color = d.decision === 'BUY' ? 'text-green-400' : d.decision === 'SELL' ? 'text-red-400' : 'text-gray-400';
        const icon = d.decision === 'BUY' ? '🟢' : d.decision === 'SELL' ? '🔴' : '⚪';
        return `
            <div class="flex justify-between items-center py-1 px-2 bg-gray-800/50 rounded text-xs">
                <span class="text-gray-500">${d.timestamp}</span>
                <span class="${color}">${icon} ${d.decision}</span>
                <span class="text-gray-500">Score: ${(d.score || 0).toFixed(2)}</span>
            </div>
        `;
    }).join('');
}

/**
 * Handle incoming signal
 */
function handleSignal(signal) {
    addLog('INFO', `[${signal.strategy}] ${signal.action}: ${signal.reason || ''}`);
    
    // Show notification for important signals
    if (signal.action === 'BUY' || signal.action === 'SELL') {
        showNotification(
            `📊 ${signal.action}`,
            `${signal.strategy}: ${signal.reason || 'Sinal executado'}`,
            signal.action === 'BUY' ? 'success' : 'error'
        );
        playAlertSound(signal.action);
    }
}

/**
 * Update trades display
 */
function updateTrades(trades) {
    const container = document.getElementById('trades-container');
    
    // Calculate trade statistics
    let totalPnl = 0;
    let wins = 0;
    let losses = 0;
    let totalGains = 0;
    let totalLosses = 0;
    let bestTrade = { pnl: 0, percent: 0 };
    let worstTrade = { pnl: 0, percent: 0 };
    
    if (trades.history && trades.history.length > 0) {
        trades.history.forEach(trade => {
            const pnl = trade.pnl || 0;
            totalPnl += pnl;
            
            if (pnl >= 0) {
                wins++;
                totalGains += pnl;
                if (pnl > bestTrade.pnl) {
                    bestTrade = { pnl: pnl, percent: trade.pnl_percent || 0 };
                }
            } else {
                losses++;
                totalLosses += Math.abs(pnl);
                if (pnl < worstTrade.pnl) {
                    worstTrade = { pnl: pnl, percent: trade.pnl_percent || 0 };
                }
            }
        });
    }
    
    // Update trade stats
    tradeStats.totalGains = totalGains;
    tradeStats.totalLosses = totalLosses;
    tradeStats.bestTrade = bestTrade;
    tradeStats.worstTrade = worstTrade;
    
    // Calculate win rate
    const totalTrades = wins + losses;
    const winRate = totalTrades > 0 ? (wins / totalTrades) * 100 : 0;
    
    // Calculate profit factor
    const profitFactor = totalLosses > 0 ? totalGains / totalLosses : totalGains > 0 ? Infinity : 0;
    
    if (!trades.history || trades.history.length === 0) {
        container.innerHTML = '<p class="text-gray-500 text-center py-4">Nenhum trade ainda</p>';
    } else {
        container.innerHTML = trades.history.map((trade, index) => {
            const pnl = trade.pnl || 0;
            const isWin = pnl >= 0;
            return `
                <div class="flex justify-between items-center py-2 px-3 rounded bg-gray-800/50">
                    <div class="flex items-center gap-2">
                        <span class="text-gray-500">#${trades.history.length - index}</span>
                        <span class="${trade.side === 'LONG' ? 'text-green-400' : 'text-red-400'}">${trade.side}</span>
                    </div>
                    <span class="${isWin ? 'text-green-400' : 'text-red-400'}">${formatCurrency(pnl, true)} ${isWin ? '✅' : '❌'}</span>
                </div>
            `;
        }).join('');
    }
    
    // Update stats display
    document.getElementById('win-rate').textContent = `${winRate.toFixed(1)}%`;
    document.getElementById('total-pnl').textContent = formatCurrency(totalPnl, true);
    document.getElementById('total-pnl').className = `font-medium ${totalPnl >= 0 ? 'text-green-400' : 'text-red-400'}`;
    
    // Update performance metrics if elements exist
    updatePerformanceMetrics(winRate, profitFactor, bestTrade, worstTrade, totalTrades);
}

/**
 * Update liquidations display
 */
function updateLiquidations(liquidations) {
    const container = document.getElementById('liquidations-container');
    
    if (!liquidations.recent || liquidations.recent.length === 0) {
        container.innerHTML = '<p class="text-gray-500 text-center py-4">Nenhuma liquidação detectada</p>';
    } else {
        container.innerHTML = liquidations.recent.map(liq => {
            const isLong = liq.side === 'SELL'; // SELL means longs being liquidated
            const emoji = isLong ? '🔴' : '🟢';
            const time = new Date(liq.timestamp).toLocaleTimeString();
            return `
                <div class="flex justify-between items-center py-2 px-3 rounded bg-gray-800/50">
                    <div class="flex items-center gap-2">
                        <span>${emoji}</span>
                        <span class="${isLong ? 'text-red-400' : 'text-green-400'}">${isLong ? 'LONG' : 'SHORT'}</span>
                        <span class="text-xs text-gray-500">${time}</span>
                    </div>
                    <span class="text-gray-300">${formatCompact(liq.amount_usd)}</span>
                </div>
            `;
        }).join('');
    }
    
    document.getElementById('total-liq-volume').textContent = formatCompact(liquidations.total_volume || 0);
}

/**
 * Add a liquidation to the display
 */
function addLiquidation(liq) {
    const container = document.getElementById('liquidations-container');
    
    // Remove "no liquidations" message if present
    const noLiqs = container.querySelector('p');
    if (noLiqs) noLiqs.remove();
    
    const isLong = liq.side === 'SELL';
    const emoji = isLong ? '🔴' : '🟢';
    const time = new Date(liq.timestamp).toLocaleTimeString();
    
    const html = `
        <div class="flex justify-between items-center py-2 px-3 rounded bg-gray-800/50 animate-fade-in">
            <div class="flex items-center gap-2">
                <span>${emoji}</span>
                <span class="${isLong ? 'text-red-400' : 'text-green-400'}">${isLong ? 'LONG' : 'SHORT'}</span>
                <span class="text-xs text-gray-500">${time}</span>
            </div>
            <span class="text-gray-300">${formatCompact(liq.amount_usd)}</span>
        </div>
    `;
    
    container.insertAdjacentHTML('afterbegin', html);
    
    // Keep only last 10
    while (container.children.length > 10) {
        container.removeChild(container.lastChild);
    }
}

/**
 * Add log entry
 */
function addLog(level, message) {
    const log = {
        time: new Date().toLocaleTimeString(),
        level: level,
        message: message
    };
    addLogEntry(log);
}

/**
 * Add log entry to container
 */
function addLogEntry(log) {
    const container = document.getElementById('logs-container');
    
    // Remove "waiting for logs" message if present
    const waiting = container.querySelector('p');
    if (waiting && waiting.textContent === 'Aguardando logs...') {
        waiting.remove();
    }
    
    // Determine styling based on log level and content
    let levelClass = 'log-info';
    let bgClass = '';
    const message = log.message || '';
    
    if (log.level === 'ERROR') {
        levelClass = 'log-error';
        bgClass = 'bg-red-500/10';
    } else if (log.level === 'WARNING') {
        levelClass = 'log-warning';
        bgClass = 'bg-yellow-500/10';
    } else if (message.toLowerCase().includes('buy')) {
        levelClass = 'text-green-400';
    } else if (message.toLowerCase().includes('sell')) {
        levelClass = 'text-red-400';
    }
    
    const html = `
        <div class="flex gap-2 ${levelClass} ${bgClass} py-1 px-2 rounded" data-level="${log.level}">
            <span class="text-gray-500 shrink-0">${log.time}</span>
            <span class="break-all">${log.message}</span>
        </div>
    `;
    
    container.insertAdjacentHTML('beforeend', html);
    
    // Keep only last 50 logs
    while (container.children.length > 50) {
        container.removeChild(container.firstChild);
    }
    
    // Auto-scroll to bottom
    container.scrollTop = container.scrollHeight;
    
    // Apply current filter
    if (logFilter !== 'all') {
        filterLogs(logFilter);
    }
}

/**
 * Format currency
 */
function formatCurrency(value, showSign = false) {
    if (value === null || value === undefined || isNaN(value)) {
        return '$0.00';
    }
    
    const formatted = Math.abs(value).toLocaleString('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
    
    if (showSign) {
        return value >= 0 ? `+$${formatted}` : `-$${formatted}`;
    }
    
    return `$${formatted}`;
}

/**
 * Format large numbers compactly
 */
function formatCompact(value) {
    if (value >= 1000000) {
        return `$${(value / 1000000).toFixed(2)}M`;
    } else if (value >= 1000) {
        return `$${(value / 1000).toFixed(1)}k`;
    }
    return `$${value.toFixed(0)}`;
}

/**
 * Format timestamp to time string
 */
function formatTime(timestamp) {
    if (!timestamp) return '-';
    const date = new Date(timestamp);
    return date.toLocaleString();
}

/**
 * Initialize tabs functionality
 */
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;
            
            // Update button states
            tabButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Update content visibility
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            document.getElementById(`tab-${tabId}`).classList.add('active');
            
            // Load data for the tab if needed
            loadTabData(tabId);
        });
    });
}

/**
 * Load data for a specific tab
 */
async function loadTabData(tabId) {
    switch(tabId) {
        case 'positions':
            await loadPositions();
            break;
        case 'open-orders':
            await loadOpenOrders();
            break;
        case 'order-history':
            await loadOrderHistory();
            break;
        case 'trade-history':
            await loadTradeHistory();
            break;
        case 'transactions':
            await loadTransactions();
            break;
        case 'assets':
            await loadAssets();
            break;
    }
}

/**
 * Load positions data
 */
async function loadPositions() {
    try {
        const response = await fetch('/api/positions');
        const positions = await response.json();
        
        const tbody = document.getElementById('positions-table-body');
        document.getElementById('positions-count').textContent = positions.length;
        
        if (positions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center text-gray-500 py-8">No open positions</td></tr>';
            return;
        }
        
        tbody.innerHTML = positions.map(pos => {
            const pnlClass = pos.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400';
            const sideClass = pos.side === 'LONG' ? 'text-green-400' : 'text-red-400';
            return `
                <tr>
                    <td class="font-medium">${pos.symbol}</td>
                    <td class="${sideClass}">${pos.side} ${pos.size}</td>
                    <td>${formatCurrency(pos.entry_price)}</td>
                    <td>${formatCurrency(pos.mark_price)}</td>
                    <td class="text-red-400">${formatCurrency(pos.liquidation_price)}</td>
                    <td class="${pnlClass}">${formatCurrency(pos.unrealized_pnl, true)}</td>
                    <td class="${pnlClass}">${pos.roe_percent >= 0 ? '+' : ''}${pos.roe_percent.toFixed(2)}%</td>
                </tr>
            `;
        }).join('');
    } catch (error) {
        console.error('Error loading positions:', error);
    }
}

/**
 * Load open orders data
 */
async function loadOpenOrders() {
    try {
        const response = await fetch('/api/open-orders');
        const orders = await response.json();
        
        const tbody = document.getElementById('open-orders-table-body');
        document.getElementById('orders-count').textContent = orders.length;
        
        if (orders.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center text-gray-500 py-8">No open orders</td></tr>';
            return;
        }
        
        tbody.innerHTML = orders.map(order => {
            const sideClass = order.side === 'BUY' ? 'text-green-400' : 'text-red-400';
            const price = order.stop_price > 0 ? order.stop_price : order.price;
            return `
                <tr>
                    <td class="text-gray-500">${formatTime(order.time)}</td>
                    <td class="font-medium">${order.symbol}</td>
                    <td>${order.type}</td>
                    <td class="${sideClass}">${order.side}</td>
                    <td>${formatCurrency(price)}</td>
                    <td>${order.quantity}</td>
                    <td><span class="px-2 py-1 bg-yellow-500/20 text-yellow-400 rounded text-xs">${order.status}</span></td>
                </tr>
            `;
        }).join('');
    } catch (error) {
        console.error('Error loading open orders:', error);
    }
}

/**
 * Load order history data
 */
async function loadOrderHistory() {
    try {
        const response = await fetch('/api/order-history');
        const orders = await response.json();
        
        const tbody = document.getElementById('order-history-table-body');
        
        if (orders.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center text-gray-500 py-8">No order history</td></tr>';
            return;
        }
        
        tbody.innerHTML = orders.slice(0, 50).map(order => {
            const sideClass = order.side === 'BUY' ? 'text-green-400' : 'text-red-400';
            let statusClass = 'bg-gray-500/20 text-gray-400';
            if (order.status === 'FILLED') statusClass = 'bg-green-500/20 text-green-400';
            if (order.status === 'CANCELED') statusClass = 'bg-red-500/20 text-red-400';
            
            return `
                <tr>
                    <td class="text-gray-500">${formatTime(order.update_time)}</td>
                    <td class="font-medium">${order.symbol}</td>
                    <td>${order.type}</td>
                    <td class="${sideClass}">${order.side}</td>
                    <td>${formatCurrency(order.avg_price)}</td>
                    <td>${order.executed_qty}/${order.quantity}</td>
                    <td><span class="px-2 py-1 ${statusClass} rounded text-xs">${order.status}</span></td>
                </tr>
            `;
        }).join('');
    } catch (error) {
        console.error('Error loading order history:', error);
    }
}

/**
 * Load trade history data
 */
async function loadTradeHistory() {
    try {
        const response = await fetch('/api/trade-history');
        const trades = await response.json();
        
        const tbody = document.getElementById('trade-history-table-body');
        
        if (trades.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-center text-gray-500 py-8">No trade history</td></tr>';
            return;
        }
        
        tbody.innerHTML = trades.slice(0, 50).map(trade => {
            const sideClass = trade.side === 'BUY' ? 'text-green-400' : 'text-red-400';
            const pnlClass = trade.realized_pnl >= 0 ? 'text-green-400' : 'text-red-400';
            
            return `
                <tr>
                    <td class="text-gray-500">${formatTime(trade.time)}</td>
                    <td class="font-medium">${trade.symbol}</td>
                    <td class="${sideClass}">${trade.side}</td>
                    <td>${formatCurrency(trade.price)}</td>
                    <td>${trade.qty}</td>
                    <td class="${pnlClass}">${formatCurrency(trade.realized_pnl, true)}</td>
                    <td class="text-gray-500">${trade.commission.toFixed(6)} ${trade.commission_asset}</td>
                </tr>
            `;
        }).join('');
    } catch (error) {
        console.error('Error loading trade history:', error);
    }
}

/**
 * Load transactions data
 */
async function loadTransactions() {
    try {
        const response = await fetch('/api/transactions');
        const transactions = await response.json();
        
        const tbody = document.getElementById('transactions-table-body');
        
        if (transactions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center text-gray-500 py-8">No transactions</td></tr>';
            return;
        }
        
        tbody.innerHTML = transactions.slice(0, 50).map(tx => {
            const incomeClass = tx.income >= 0 ? 'text-green-400' : 'text-red-400';
            
            return `
                <tr>
                    <td class="text-gray-500">${formatTime(tx.time)}</td>
                    <td class="font-medium">${tx.symbol || '-'}</td>
                    <td>${tx.type}</td>
                    <td class="${incomeClass}">${tx.income >= 0 ? '+' : ''}${tx.income.toFixed(8)}</td>
                    <td>${tx.asset}</td>
                </tr>
            `;
        }).join('');
    } catch (error) {
        console.error('Error loading transactions:', error);
    }
}

/**
 * Load assets data
 */
async function loadAssets() {
    try {
        const response = await fetch('/api/assets');
        const assets = await response.json();
        
        const tbody = document.getElementById('assets-table-body');
        
        if (assets.length === 0) {
            tbody.innerHTML = '<tr><td colspan="4" class="text-center text-gray-500 py-8">No assets</td></tr>';
            return;
        }
        
        tbody.innerHTML = assets.map(asset => {
            const pnlClass = asset.cross_unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400';
            
            // Format based on asset type
            const decimals = asset.asset === 'USDT' ? 2 : 8;
            const balanceFormatted = formatAssetValue(asset.balance, asset.asset);
            const availableFormatted = formatAssetValue(asset.available, asset.asset);
            const pnlFormatted = formatAssetValue(asset.cross_unrealized_pnl, asset.asset);
            
            return `
                <tr>
                    <td class="font-medium">${asset.asset}</td>
                    <td>${balanceFormatted}</td>
                    <td>${availableFormatted}</td>
                    <td class="${pnlClass}">${asset.cross_unrealized_pnl >= 0 ? '+' : ''}${pnlFormatted}</td>
                </tr>
            `;
        }).join('');
    } catch (error) {
        console.error('Error loading assets:', error);
    }
}

/**
 * Format asset value based on asset type
 */
function formatAssetValue(value, asset) {
    if (asset === 'USDT') {
        return value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    } else if (asset === 'BTC') {
        return value.toFixed(8);
    }
    return value.toFixed(8);
}

/**
 * Update performance metrics display
 */
function updatePerformanceMetrics(winRate, profitFactor, bestTrade, worstTrade, totalTrades) {
    // Update performance cards if they exist
    const winRateEl = document.getElementById('perf-win-rate');
    if (winRateEl) winRateEl.textContent = `${winRate.toFixed(1)}%`;
    
    const profitFactorEl = document.getElementById('perf-profit-factor');
    if (profitFactorEl) {
        profitFactorEl.textContent = profitFactor === Infinity ? '∞' : profitFactor.toFixed(2);
    }
    
    const bestTradeEl = document.getElementById('perf-best-trade');
    if (bestTradeEl) {
        bestTradeEl.textContent = formatCurrency(bestTrade.pnl, true);
        bestTradeEl.className = 'mini-stat-value text-green-400';
    }
    
    const worstTradeEl = document.getElementById('perf-worst-trade');
    if (worstTradeEl) {
        worstTradeEl.textContent = formatCurrency(worstTrade.pnl, true);
        worstTradeEl.className = 'mini-stat-value text-red-400';
    }
    
    // Update total trades
    const totalTradesEl = document.getElementById('perf-total-trades');
    if (totalTradesEl) totalTradesEl.textContent = totalTrades;
    
    // Calculate trades per hour
    if (botStartTime && totalTrades > 0) {
        const hoursRunning = (Date.now() - botStartTime) / 3600000;
        const tradesPerHour = hoursRunning > 0 ? totalTrades / hoursRunning : 0;
        const tradesPerHourEl = document.getElementById('perf-trades-per-hour');
        if (tradesPerHourEl) tradesPerHourEl.textContent = `${tradesPerHour.toFixed(1)}/h`;
    }
}

/**
 * Show notification popup
 */
function showNotification(title, message, type = 'info') {
    const container = document.getElementById('notification-container');
    if (!container) return;
    
    const bgColor = type === 'success' ? 'bg-green-500/90' : 
                    type === 'error' ? 'bg-red-500/90' : 
                    type === 'warning' ? 'bg-yellow-500/90' : 'bg-blue-500/90';
    
    const notification = document.createElement('div');
    notification.className = `${bgColor} text-white px-4 py-3 rounded-lg shadow-lg mb-2 animate-fade-in flex items-center gap-3`;
    notification.innerHTML = `
        <div>
            <div class="font-semibold">${title}</div>
            <div class="text-sm opacity-90">${message}</div>
        </div>
        <button class="ml-auto text-white/80 hover:text-white" onclick="this.parentElement.remove()">✕</button>
    `;
    
    container.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.classList.add('opacity-0', 'transition-opacity', 'duration-300');
            setTimeout(() => notification.remove(), 300);
        }
    }, 5000);
}

// Shared AudioContext for sound alerts
let audioContext = null;

/**
 * Play alert sound
 */
function playAlertSound(type = 'default') {
    if (!soundEnabled) return;
    
    try {
        // Create or reuse AudioContext
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        
        // Resume context if suspended (due to autoplay policy)
        if (audioContext.state === 'suspended') {
            audioContext.resume();
        }
        
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        // Different sounds for different types
        if (type === 'BUY') {
            oscillator.frequency.value = 880; // Higher pitch for buy
        } else if (type === 'SELL') {
            oscillator.frequency.value = 440; // Lower pitch for sell
        } else {
            oscillator.frequency.value = 660;
        }
        
        oscillator.type = 'sine';
        gainNode.gain.value = 0.1;
        
        oscillator.start();
        oscillator.stop(audioContext.currentTime + 0.15);
    } catch (e) {
        console.log('Audio not supported:', e.message);
    }
}

/**
 * Toggle sound
 */
function toggleSound() {
    soundEnabled = !soundEnabled;
    const btn = document.getElementById('sound-toggle');
    if (btn) {
        btn.textContent = soundEnabled ? '🔊' : '🔇';
        btn.title = soundEnabled ? 'Som ativado' : 'Som desativado';
    }
    try {
        localStorage.setItem('soundEnabled', soundEnabled);
    } catch (e) {
        console.log('localStorage not available');
    }
}

/**
 * Toggle dark/light mode
 */
function toggleTheme() {
    darkMode = !darkMode;
    document.documentElement.classList.toggle('dark', darkMode);
    document.documentElement.classList.toggle('light', !darkMode);
    
    const btn = document.getElementById('theme-toggle');
    if (btn) {
        btn.textContent = darkMode ? '🌙' : '☀️';
    }
    
    // Update body background
    document.body.style.backgroundColor = darkMode ? '#0f0f0f' : '#f5f5f5';
    
    try {
        localStorage.setItem('darkMode', darkMode);
    } catch (e) {
        console.log('localStorage not available');
    }
}

/**
 * Export logs to CSV
 */
function exportLogsCSV() {
    try {
        const container = document.getElementById('logs-container');
        const logs = container.querySelectorAll('div');
        
        let csv = 'Time,Level,Message\n';
        logs.forEach(log => {
            const spans = log.querySelectorAll('span');
            if (spans.length >= 2) {
                const time = spans[0].textContent;
                const message = spans[1].textContent.replace(/,/g, ';').replace(/"/g, '""');
                const level = log.classList.contains('log-error') ? 'ERROR' : 
                             log.classList.contains('log-warning') ? 'WARNING' : 'INFO';
                csv += `${time},${level},"${message}"\n`;
            }
        });
        
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `bot-logs-${new Date().toISOString().split('T')[0]}.csv`;
        a.click();
        URL.revokeObjectURL(url);
        
        showNotification('📥 Exportado', 'Logs exportados com sucesso!', 'success');
    } catch (e) {
        console.error('Error exporting logs:', e);
        showNotification('❌ Erro', 'Falha ao exportar logs', 'error');
    }
}

/**
 * Toggle bot pause state
 */
async function toggleBotPause() {
    try {
        const response = await fetch('/api/pause', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ paused: !isPaused })
        });
        
        if (response.ok) {
            isPaused = !isPaused;
            updatePauseButton();
            showNotification(
                isPaused ? '⏸️ Pausado' : '▶️ Retomado',
                isPaused ? 'Bot pausado' : 'Bot retomado',
                isPaused ? 'warning' : 'success'
            );
        }
    } catch (error) {
        console.error('Error toggling pause:', error);
    }
}

/**
 * Update pause button state
 */
function updatePauseButton() {
    const btn = document.getElementById('pause-toggle');
    if (btn) {
        btn.textContent = isPaused ? '▶️' : '⏸️';
        btn.title = isPaused ? 'Retomar Bot' : 'Pausar Bot';
        btn.className = isPaused ? 
            'px-3 py-1 bg-green-500/20 text-green-400 rounded-lg hover:bg-green-500/30' :
            'px-3 py-1 bg-yellow-500/20 text-yellow-400 rounded-lg hover:bg-yellow-500/30';
    }
    
    const statusEl = document.getElementById('bot-status');
    if (statusEl) {
        statusEl.textContent = isPaused ? 'Pausado' : 'Executando';
        statusEl.className = isPaused ? 'text-yellow-400' : 'text-green-400';
    }
}

/**
 * Filter logs by type
 */
function filterLogs(filter) {
    logFilter = filter;
    const container = document.getElementById('logs-container');
    const logs = container.querySelectorAll('div');
    
    logs.forEach(log => {
        const message = log.textContent.toLowerCase();
        let show = true;
        
        if (filter !== 'all') {
            if (filter === 'BUY') show = message.includes('buy');
            else if (filter === 'SELL') show = message.includes('sell');
            else if (filter === 'ERROR') show = log.classList.contains('log-error');
            else if (filter === 'INFO') show = log.classList.contains('log-info');
        }
        
        log.style.display = show ? '' : 'none';
    });
    
    // Update filter buttons
    document.querySelectorAll('.log-filter-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.filter === filter);
    });
}

/**
 * Initialize event listeners and load saved preferences
 */
function initPreferences() {
    try {
        // Load saved preferences
        const savedSound = localStorage.getItem('soundEnabled');
        if (savedSound !== null) {
            soundEnabled = savedSound === 'true';
        }
        
        const savedDarkMode = localStorage.getItem('darkMode');
        if (savedDarkMode !== null) {
            darkMode = savedDarkMode === 'true';
        }
    } catch (e) {
        console.log('localStorage not available');
    }
    
    // Apply saved preferences
    if (!darkMode) {
        document.documentElement.classList.remove('dark');
        document.documentElement.classList.add('light');
        document.body.style.backgroundColor = '#f5f5f5';
    }
    
    // Update UI elements
    const soundBtn = document.getElementById('sound-toggle');
    if (soundBtn) soundBtn.textContent = soundEnabled ? '🔊' : '🔇';
    
    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) themeBtn.textContent = darkMode ? '🌙' : '☀️';
}

// Initialize tabs on page load
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    initEquityChart();
    initTabs();
    initPreferences();
    connectWebSocket();
    
    // Load initial tab data
    loadPositions();
    
    // Initialize timeframe selector
    initTimeframeSelector();
    
    // Initialize chart type toggle
    initChartTypeToggle();
    
    // Initialize log filter
    initLogFilter();
});

/**
 * Initialize timeframe selector
 */
function initTimeframeSelector() {
    const btns = document.querySelectorAll('.timeframe-btn');
    btns.forEach(btn => {
        btn.addEventListener('click', () => {
            btns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentTimeframe = btn.dataset.timeframe;
            // Note: Timeframe change would require backend support
            showNotification('⏱️ Timeframe', `Alterado para ${currentTimeframe}`, 'info');
        });
    });
}

/**
 * Initialize chart type toggle
 */
function initChartTypeToggle() {
    const btns = document.querySelectorAll('.chart-type-btn');
    btns.forEach(btn => {
        btn.addEventListener('click', () => {
            btns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            chartType = btn.dataset.type;
            updateChart();
        });
    });
}

/**
 * Initialize log filter
 */
function initLogFilter() {
    const btns = document.querySelectorAll('.log-filter-btn');
    btns.forEach(btn => {
        btn.addEventListener('click', () => {
            filterLogs(btn.dataset.filter);
        });
    });
}

// Periodic refresh for tabs
setInterval(() => {
    // Refresh the active tab data
    const activeTab = document.querySelector('.tab-btn.active');
    if (activeTab) {
        loadTabData(activeTab.dataset.tab);
    }
}, 10000);

// Periodic ping for latency measurement
setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        lastPingTime = Date.now();
        ws.send('ping');
    }
}, 5000);
