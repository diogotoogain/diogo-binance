/**
 * MEGA Backtest Analytics Dashboard - Chart Functions
 * Uses Chart.js for visualization
 */

// Chart instances
let equityChart = null;
let distributionChart = null;
let hourChart = null;
let dowChart = null;
let drawdownChart = null;
let streakChart = null;
let rollingChart = null;
let monteCarloChart = null;
let compareChart = null;

// Chart default options
const chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            display: false
        }
    },
    scales: {
        x: {
            grid: {
                color: 'rgba(255, 255, 255, 0.05)'
            },
            ticks: {
                color: '#9ca3af'
            }
        },
        y: {
            grid: {
                color: 'rgba(255, 255, 255, 0.05)'
            },
            ticks: {
                color: '#9ca3af'
            }
        }
    }
};

/**
 * Render Equity Curve Chart
 */
function renderEquityChart(data) {
    const ctx = document.getElementById('equity-chart');
    if (!ctx) return;
    
    if (equityChart) {
        equityChart.destroy();
    }
    
    const labels = data.map((d, i) => i);
    const values = data.map(d => d.equity);
    
    equityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Equity',
                data: values,
                borderColor: '#00d4ff',
                backgroundColor: 'rgba(0, 212, 255, 0.1)',
                fill: true,
                tension: 0.1,
                pointRadius: 0,
                borderWidth: 2
            }]
        },
        options: {
            ...chartDefaults,
            scales: {
                ...chartDefaults.scales,
                x: {
                    ...chartDefaults.scales.x,
                    display: false
                },
                y: {
                    ...chartDefaults.scales.y,
                    ticks: {
                        ...chartDefaults.scales.y.ticks,
                        callback: (value) => '$' + value.toLocaleString()
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: (context) => '$' + context.parsed.y.toLocaleString()
                    }
                }
            }
        }
    });
}

/**
 * Render Returns Distribution Chart
 */
function renderDistributionChart(pnlValues) {
    const ctx = document.getElementById('distribution-chart');
    if (!ctx) return;
    
    if (distributionChart) {
        distributionChart.destroy();
    }
    
    // Create histogram bins
    const min = Math.min(...pnlValues);
    const max = Math.max(...pnlValues);
    const binCount = 20;
    const binSize = (max - min) / binCount;
    
    const bins = Array(binCount).fill(0);
    const binLabels = [];
    
    for (let i = 0; i < binCount; i++) {
        const binStart = min + i * binSize;
        binLabels.push(binStart.toFixed(0));
    }
    
    pnlValues.forEach(val => {
        const binIndex = Math.min(Math.floor((val - min) / binSize), binCount - 1);
        bins[binIndex]++;
    });
    
    distributionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: binLabels,
            datasets: [{
                label: 'Frequency',
                data: bins,
                backgroundColor: bins.map((_, i) => {
                    const binStart = min + i * binSize;
                    return binStart >= 0 ? 'rgba(0, 255, 136, 0.6)' : 'rgba(255, 68, 68, 0.6)';
                }),
                borderColor: bins.map((_, i) => {
                    const binStart = min + i * binSize;
                    return binStart >= 0 ? '#00ff88' : '#ff4444';
                }),
                borderWidth: 1
            }]
        },
        options: {
            ...chartDefaults,
            scales: {
                x: {
                    ...chartDefaults.scales.x,
                    title: {
                        display: true,
                        text: 'P&L ($)',
                        color: '#9ca3af'
                    }
                },
                y: {
                    ...chartDefaults.scales.y,
                    title: {
                        display: true,
                        text: 'Frequency',
                        color: '#9ca3af'
                    }
                }
            }
        }
    });
}

/**
 * Render Hour of Day Chart
 */
function renderHourChart(data) {
    const ctx = document.getElementById('hour-chart');
    if (!ctx) return;
    
    if (hourChart) {
        hourChart.destroy();
    }
    
    const labels = data.map(d => d.hour + ':00');
    const values = data.map(d => d.avg_pnl);
    
    hourChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Avg P&L',
                data: values,
                backgroundColor: values.map(v => v >= 0 ? 'rgba(0, 255, 136, 0.6)' : 'rgba(255, 68, 68, 0.6)'),
                borderColor: values.map(v => v >= 0 ? '#00ff88' : '#ff4444'),
                borderWidth: 1
            }]
        },
        options: {
            ...chartDefaults,
            scales: {
                y: {
                    ...chartDefaults.scales.y,
                    ticks: {
                        ...chartDefaults.scales.y.ticks,
                        callback: (value) => '$' + value.toFixed(2)
                    }
                }
            }
        }
    });
}

/**
 * Render Day of Week Chart
 */
function renderDowChart(data) {
    const ctx = document.getElementById('dow-chart');
    if (!ctx) return;
    
    if (dowChart) {
        dowChart.destroy();
    }
    
    const labels = data.map(d => d.day_name);
    const values = data.map(d => d.avg_pnl);
    
    dowChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Avg P&L',
                data: values,
                backgroundColor: values.map(v => v >= 0 ? 'rgba(0, 255, 136, 0.6)' : 'rgba(255, 68, 68, 0.6)'),
                borderColor: values.map(v => v >= 0 ? '#00ff88' : '#ff4444'),
                borderWidth: 1
            }]
        },
        options: chartDefaults
    });
}

/**
 * Render Drawdown Chart (Underwater Curve)
 */
function renderDrawdownChart(equityData) {
    const ctx = document.getElementById('drawdown-chart');
    if (!ctx) return;
    
    if (drawdownChart) {
        drawdownChart.destroy();
    }
    
    // Calculate drawdown from equity
    const equity = equityData.map(d => d.equity);
    let peak = equity[0];
    const drawdown = equity.map(e => {
        if (e > peak) peak = e;
        return ((e - peak) / peak) * 100;
    });
    
    const labels = drawdown.map((_, i) => i);
    
    drawdownChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Drawdown %',
                data: drawdown,
                borderColor: '#ff4444',
                backgroundColor: 'rgba(255, 68, 68, 0.2)',
                fill: true,
                tension: 0.1,
                pointRadius: 0,
                borderWidth: 2
            }]
        },
        options: {
            ...chartDefaults,
            scales: {
                x: {
                    ...chartDefaults.scales.x,
                    display: false
                },
                y: {
                    ...chartDefaults.scales.y,
                    ticks: {
                        ...chartDefaults.scales.y.ticks,
                        callback: (value) => value.toFixed(1) + '%'
                    }
                }
            }
        }
    });
}

/**
 * Render Streak Distribution Chart
 */
function renderStreakChart(data) {
    const ctx = document.getElementById('streak-chart');
    if (!ctx) return;
    
    if (streakChart) {
        streakChart.destroy();
    }
    
    const winDist = data.win_streak_distribution || [];
    const lossDist = data.loss_streak_distribution || [];
    
    // Combine into single chart
    const maxLength = Math.max(
        ...winDist.map(d => d.length),
        ...lossDist.map(d => d.length),
        1
    );
    
    const labels = Array.from({length: maxLength}, (_, i) => i + 1);
    const winData = labels.map(l => {
        const found = winDist.find(d => d.length === l);
        return found ? found.count : 0;
    });
    const lossData = labels.map(l => {
        const found = lossDist.find(d => d.length === l);
        return found ? found.count : 0;
    });
    
    streakChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Win Streaks',
                    data: winData,
                    backgroundColor: 'rgba(0, 255, 136, 0.6)',
                    borderColor: '#00ff88',
                    borderWidth: 1
                },
                {
                    label: 'Loss Streaks',
                    data: lossData,
                    backgroundColor: 'rgba(255, 68, 68, 0.6)',
                    borderColor: '#ff4444',
                    borderWidth: 1
                }
            ]
        },
        options: {
            ...chartDefaults,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#9ca3af'
                    }
                }
            },
            scales: {
                x: {
                    ...chartDefaults.scales.x,
                    title: {
                        display: true,
                        text: 'Streak Length',
                        color: '#9ca3af'
                    }
                }
            }
        }
    });
}

/**
 * Render Rolling Metrics Chart
 */
function renderRollingChart(data) {
    const ctx = document.getElementById('rolling-chart');
    if (!ctx) return;
    
    if (rollingChart) {
        rollingChart.destroy();
    }
    
    const labels = data.map(d => d.index);
    const winRates = data.map(d => d.win_rate);
    const sharpes = data.map(d => d.sharpe_like);
    
    rollingChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Win Rate %',
                    data: winRates,
                    borderColor: '#00d4ff',
                    backgroundColor: 'transparent',
                    tension: 0.1,
                    pointRadius: 0,
                    yAxisID: 'y'
                },
                {
                    label: 'Sharpe-like',
                    data: sharpes,
                    borderColor: '#fbbf24',
                    backgroundColor: 'transparent',
                    tension: 0.1,
                    pointRadius: 0,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            ...chartDefaults,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#9ca3af'
                    }
                }
            },
            scales: {
                x: {
                    ...chartDefaults.scales.x,
                    display: false
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    grid: chartDefaults.scales.y.grid,
                    ticks: {
                        color: '#00d4ff',
                        callback: (value) => value.toFixed(0) + '%'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    grid: {
                        drawOnChartArea: false
                    },
                    ticks: {
                        color: '#fbbf24'
                    }
                }
            }
        }
    });
}

/**
 * Render Monte Carlo Simulation Chart
 */
function renderMonteCarloChart(data) {
    const ctx = document.getElementById('monte-carlo-chart');
    if (!ctx) return;
    
    if (monteCarloChart) {
        monteCarloChart.destroy();
    }
    
    const histogram = data.histogram || [];
    const labels = histogram.map(h => h.bin_start.toFixed(0));
    const values = histogram.map(h => h.count);
    
    monteCarloChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Frequency',
                data: values,
                backgroundColor: 'rgba(0, 212, 255, 0.6)',
                borderColor: '#00d4ff',
                borderWidth: 1
            }]
        },
        options: {
            ...chartDefaults,
            scales: {
                x: {
                    ...chartDefaults.scales.x,
                    title: {
                        display: true,
                        text: 'Final Balance ($)',
                        color: '#9ca3af'
                    }
                },
                y: {
                    ...chartDefaults.scales.y,
                    title: {
                        display: true,
                        text: 'Simulations',
                        color: '#9ca3af'
                    }
                }
            },
            plugins: {
                annotation: {
                    annotations: {
                        actualLine: {
                            type: 'line',
                            xMin: data.actual_final,
                            xMax: data.actual_final,
                            borderColor: '#00ff88',
                            borderWidth: 2,
                            label: {
                                display: true,
                                content: 'Your Result',
                                position: 'start'
                            }
                        }
                    }
                }
            }
        }
    });
}

/**
 * Render Strategy vs Buy & Hold Comparison Chart
 */
function renderCompareChart(strategyData, buyHoldData) {
    const ctx = document.getElementById('compare-chart');
    if (!ctx) return;
    
    if (compareChart) {
        compareChart.destroy();
    }
    
    const labels = strategyData.map((_, i) => i);
    
    compareChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Strategy',
                    data: strategyData,
                    borderColor: '#00d4ff',
                    backgroundColor: 'transparent',
                    tension: 0.1,
                    pointRadius: 0,
                    borderWidth: 2
                },
                {
                    label: 'Buy & Hold',
                    data: buyHoldData,
                    borderColor: '#fbbf24',
                    backgroundColor: 'transparent',
                    tension: 0.1,
                    pointRadius: 0,
                    borderWidth: 2
                }
            ]
        },
        options: {
            ...chartDefaults,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#9ca3af'
                    }
                }
            },
            scales: {
                x: {
                    ...chartDefaults.scales.x,
                    display: false
                },
                y: {
                    ...chartDefaults.scales.y,
                    ticks: {
                        ...chartDefaults.scales.y.ticks,
                        callback: (value) => value.toFixed(0) + '%'
                    }
                }
            }
        }
    });
}
