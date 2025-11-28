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

// Note: Initialization is handled at the end of the file
// to ensure all functions are defined before use

/**
 * Initialize Chart.js price chart
 */
function initChart() {
    const ctx = document.getElementById('priceChart').getContext('2d');
    
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
    };
    
    ws.onmessage = (event) => {
        try {
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
    
    switch (status) {
        case 'connected':
            dot.className = 'status-dot status-live pulse-green';
            text.textContent = 'LIVE';
            text.className = 'text-sm text-green-400';
            break;
        case 'connecting':
            dot.className = 'status-dot status-offline';
            text.textContent = 'Conectando...';
            text.className = 'text-sm text-yellow-400';
            break;
        case 'disconnected':
        case 'error':
            dot.className = 'status-dot status-offline pulse-red';
            text.textContent = 'Offline';
            text.className = 'text-sm text-red-400';
            break;
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
    
    // System
    if (data.system) {
        document.getElementById('uptime').textContent = `Uptime: ${data.system.uptime}`;
        if (data.system.testnet) {
            document.getElementById('testnet-badge').classList.remove('hidden');
        }
    }
}

/**
 * Handle refresh data
 */
function handleRefresh(data) {
    if (data.balance) updateBalance(data.balance);
    if (data.position) updatePosition(data.position);
    if (data.strategies) updateStrategies(data.strategies);
    if (data.meta_controller) updateMetaController(data.meta_controller);
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
}

/**
 * Update position display
 */
function updatePosition(position) {
    const noPosition = document.getElementById('no-position');
    const hasPosition = document.getElementById('has-position');
    
    if (!position || !position.has_position) {
        noPosition.classList.remove('hidden');
        hasPosition.classList.add('hidden');
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
    
    const pnl = position.unrealized_pnl || 0;
    const pnlPercent = position.pnl_percent || 0;
    const pnlEl = document.getElementById('position-pnl');
    pnlEl.textContent = `${formatCurrency(pnl, true)} (${pnlPercent >= 0 ? '+' : ''}${pnlPercent.toFixed(2)}%)`;
    pnlEl.className = `font-medium ${pnl >= 0 ? 'text-green-400' : 'text-red-400'}`;
    
    if (position.stop_loss) {
        document.getElementById('stop-loss').textContent = formatCurrency(position.stop_loss);
    }
    if (position.take_profit) {
        document.getElementById('take-profit').textContent = formatCurrency(position.take_profit);
    }
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
}

/**
 * Handle incoming signal
 */
function handleSignal(signal) {
    addLog('INFO', `[${signal.strategy}] ${signal.action}: ${signal.reason || ''}`);
}

/**
 * Update trades display
 */
function updateTrades(trades) {
    const container = document.getElementById('trades-container');
    
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
    
    document.getElementById('win-rate').textContent = `${(trades.win_rate || 0).toFixed(1)}%`;
    document.getElementById('total-pnl').textContent = formatCurrency(trades.total_pnl || 0, true);
    document.getElementById('total-pnl').className = `font-medium ${(trades.total_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`;
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
    
    const levelClass = log.level === 'ERROR' ? 'log-error' : (log.level === 'WARNING' ? 'log-warning' : 'log-info');
    
    const html = `
        <div class="flex gap-2 ${levelClass}">
            <span class="text-gray-500">${log.time}</span>
            <span>${log.message}</span>
        </div>
    `;
    
    container.insertAdjacentHTML('beforeend', html);
    
    // Keep only last 50 logs
    while (container.children.length > 50) {
        container.removeChild(container.firstChild);
    }
    
    // Auto-scroll to bottom
    container.scrollTop = container.scrollHeight;
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
            
            return `
                <tr>
                    <td class="font-medium">${asset.asset}</td>
                    <td>${asset.balance.toFixed(8)}</td>
                    <td>${asset.available.toFixed(8)}</td>
                    <td class="${pnlClass}">${asset.cross_unrealized_pnl >= 0 ? '+' : ''}${asset.cross_unrealized_pnl.toFixed(8)}</td>
                </tr>
            `;
        }).join('');
    } catch (error) {
        console.error('Error loading assets:', error);
    }
}

// Initialize tabs on page load
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    initTabs();
    connectWebSocket();
    
    // Load initial tab data
    loadPositions();
});

// Periodic refresh for tabs
setInterval(() => {
    // Refresh the active tab data
    const activeTab = document.querySelector('.tab-btn.active');
    if (activeTab) {
        loadTabData(activeTab.dataset.tab);
    }
}, 10000);

// Periodic uptime update
setInterval(() => {
    // Request status update
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send('ping');
    }
}, 30000);
