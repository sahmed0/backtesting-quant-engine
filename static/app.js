// --- Fullscreen & Mobile Optimisation Logic ---
async function toggleFullscreen(wrapperId, btnElement) {
  const wrapper = document.getElementById(wrapperId);
  const icon = btnElement.querySelector('i');

  if (!document.fullscreenElement) {
    try {
      await wrapper.requestFullscreen();
      icon.classList.replace('fa-expand', 'fa-compress');
      // Force Landscape on mobile if supported
      if (screen.orientation && screen.orientation.lock) {
        try { await screen.orientation.lock('landscape'); } catch (e) { console.log('Orientation lock unavailable'); }
      }
    } catch (err) {
      console.error(`Error entering fullscreen: ${err.message}`);
    }
  } else {
    document.exitFullscreen();
  }
}

document.addEventListener('fullscreenchange', () => {
  if (!document.fullscreenElement) {
    // 1. Reset icons
    document.querySelectorAll('.fa-compress').forEach(icon => {
      icon.classList.replace('fa-compress', 'fa-expand');
    });

    // 2. Unlock orientation
    if (screen.orientation && screen.orientation.unlock) {
      screen.orientation.unlock();
    }

    // 3. Force Chart.js to snap back to the restored container size
    // Add a small delay so the DOM has time to recalculate the wrapper's width first
    setTimeout(() => {
      if (equityChartInstance) {
        equityChartInstance.resize();
        equityChartInstance.update('none'); // 'none' skips animation for a clean snap
      }
      if (priceChartInstance) {
        priceChartInstance.resize();
        priceChartInstance.update('none');
      }
    }, 100);
  }
});

// --- UI State Management ---
const csvUpload = document.getElementById('csv-upload');
const tickerSelect = document.getElementById('ticker-select');
const clearBtn = document.getElementById('clear-upload');
const runBtn = document.getElementById('run-btn');
const analyseBtn = document.getElementById('analyse-btn');
const statusDiv = document.getElementById('status');

// Global Chart.js Font override to match Theme
Chart.defaults.font.family = "'Chakra Petch', sans-serif";
Chart.defaults.color = '#475569';

function updateUIState() {
  const hasUpload = csvUpload.files.length > 0;
  const hasSelection = tickerSelect.value !== "";
  const hasData = hasUpload || hasSelection;

  tickerSelect.disabled = hasUpload;
  clearBtn.style.display = hasUpload ? 'flex' : 'none';
  runBtn.disabled = !hasData;
  analyseBtn.disabled = !hasData;

  // While either job is in flight, keep both buttons disabled so a second
  // run can't interleave with the first on the single Pyodide thread.
  const busy = runBtn.innerText.includes('Running') || analyseBtn.innerText.includes('Analysing');
  if (busy) {
    runBtn.disabled = true;
    analyseBtn.disabled = true;
  }
}

csvUpload.addEventListener('change', updateUIState);
tickerSelect.addEventListener('change', updateUIState);

// --- Strategy Parameter Visibility ---
const strategySelect = document.getElementById('strategy-select');
const smaParams = document.getElementById('sma-params');
const ouParams = document.getElementById('ou-params');

function updateStrategyParams() {
  const isOU = strategySelect.value === 'ou';
  smaParams.style.display = isOU ? 'none' : 'block';
  ouParams.style.display = isOU ? 'block' : 'none';
}

strategySelect.addEventListener('change', updateStrategyParams);
updateStrategyParams();

clearBtn.addEventListener('click', function () {
  csvUpload.value = '';
  updateUIState();
});

// --- Chart Generation Logic ---
let equityChartInstance = null;
let priceChartInstance = null;

// Helper to colour code text elements based on positive/negative values
function formatMetricColor(elementId, valueStr) {
  const el = document.getElementById(elementId);
  const val = parseFloat(valueStr.replace(/[^0-9.-]+/g, ""));
  if (val > 0) {
    el.className = 'metric-value text-success';
  } else if (val < 0) {
    el.className = 'metric-value text-danger';
  } else {
    el.className = 'metric-value';
  }
}

window.updateCharts = function (timestampsJSON, equityJSON, pricesJSON, tradesJSON, benchmarkJSON) {
  document.getElementById('charts-card').style.display = 'block';
  statusDiv.innerHTML = '<i class="fa-solid fa-check text-success"></i>';

  const timestamps = JSON.parse(timestampsJSON);
  const equity = JSON.parse(equityJSON);
  const prices = JSON.parse(pricesJSON);
  const trades = JSON.parse(tradesJSON);
  const benchmark = benchmarkJSON ? JSON.parse(benchmarkJSON) : [];

  // Dynamically check metrics to apply color coding
  formatMetricColor('val-return', document.getElementById('val-return').innerText);
  formatMetricColor('val-cagr', document.getElementById('val-cagr').innerText);
  formatMetricColor('val-alpha', document.getElementById('val-alpha').innerText);

  // Charts use a real time axis: every series is an {x: epoch-ms, y} point
  // array, so duplicate calendar dates stay distinct and no locale date
  // strings are ever built.
  const equityPoints = timestamps.map((ts, i) => ({ x: ts * 1000, y: equity[i] }));
  const pricePoints = timestamps.map((ts, i) => ({ x: ts * 1000, y: prices[i] }));
  const benchmarkPoints = benchmark.length
    ? timestamps.map((ts, i) => ({ x: ts * 1000, y: benchmark[i] }))
    : [];

  const buyData = [];
  const sellData = [];

  trades.forEach(trade => {
    const point = { x: trade.timestamp * 1000, y: trade.price, quantity: trade.quantity };
    if (trade.direction === 'LONG') {
      buyData.push(point);
    } else if (trade.direction === 'SHORT' || trade.direction === 'EXIT') {
      sellData.push(point);
    }
  });

  // --- 1. Equity Chart with Canvas Gradient ---
  const ctxEquity = document.getElementById('equityChart').getContext('2d');

  // Create Gradient for area under the curve
  let equityGradient = ctxEquity.createLinearGradient(0, 0, 0, 400);
  equityGradient.addColorStop(0, 'rgba(255, 83, 137, 0.4)');
  equityGradient.addColorStop(1, 'rgba(255, 83, 137, 0.0)');

  if (equityChartInstance) equityChartInstance.destroy();
  equityChartInstance = new Chart(ctxEquity, {
    type: 'line',
    data: {
      datasets: [
        {
          label: 'Portfolio Equity ($)',
          data: equityPoints,
          borderColor: '#ff0051',
          backgroundColor: equityGradient,
          borderWidth: 2,
          fill: true,
          pointRadius: 0,
          pointHoverRadius: 8,
          tension: 0.3
        },
        {
          label: 'Buy & Hold ($)',
          data: benchmarkPoints,
          borderColor: '#9333ea',
          borderWidth: 1.5,
          borderDash: [6, 4],
          fill: false,
          pointRadius: 0,
          pointHoverRadius: 8,
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: true },
        zoom: {
          pan: {
            enabled: true, mode: 'x',
            onPanStart: () => { document.getElementById('resetEquityZoom').style.display = 'flex'; }
          },
          zoom: {
            wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x',
            onZoomStart: () => { document.getElementById('resetEquityZoom').style.display = 'flex'; }
          }
        }
      },
      scales: {
        x: { type: 'time', time: { tooltipFormat: 'MMM d, yyyy' }, display: true, grid: { display: false }, ticks: { maxTicksLimit: 8 } },
        y: { display: true, border: { dash: [4, 4] }, grid: { color: 'rgba(0,0,0,0.05)' } }
      }
    }
  });

  document.getElementById('resetEquityZoom').onclick = () => {
    equityChartInstance.resetZoom();
    document.getElementById('resetEquityZoom').style.display = 'none';
  };

  // --- 2. Price Chart ---
  const ctxPrice = document.getElementById('priceChart').getContext('2d');
  if (priceChartInstance) priceChartInstance.destroy();
  priceChartInstance = new Chart(ctxPrice, {
    type: 'line',
    data: {
      datasets: [
        {
          type: 'line',
          label: 'Asset Price ($)',
          data: pricePoints,
          borderColor: '#334155',
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 8,
          fill: false,
          order: 2,
          tension: 0.1
        },
        {
          type: 'scatter',
          label: 'Buy Signal',
          data: buyData,
          backgroundColor: '#10b981',
          borderColor: '#ffffff',
          borderWidth: 2,
          pointStyle: 'circle',
          pointRadius: 10,
          order: 1
        },
        {
          type: 'scatter',
          label: 'Sell/Exit Signal',
          data: sellData,
          backgroundColor: '#ef4444',
          borderColor: '#ffffff',
          borderWidth: 2,
          pointStyle: 'triangle',
          rotation: 180,
          pointRadius: 10,
          order: 1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { position: 'top', labels: { usePointStyle: true, boxWidth: 8, padding: 10 } },
        zoom: {
          pan: {
            enabled: true, mode: 'x',
            onPanStart: () => { document.getElementById('resetPriceZoom').style.display = 'flex'; }
          },
          zoom: {
            wheel: { enabled: true }, pinch: { enabled: true }, mode: 'x',
            onZoomStart: () => { document.getElementById('resetPriceZoom').style.display = 'flex'; }
          }
        },
        tooltip: {
          callbacks: {
            label: function (context) {
              if (context.dataset.type === 'scatter') {
                const t = context.raw;
                return `${context.dataset.label}: $${t.y.toFixed(2)} (Qty: ${t.quantity})`;
              }
              return `Price: $${context.raw.y.toFixed(2)}`;
            }
          }
        }
      },
      scales: {
        x: { type: 'time', time: { tooltipFormat: 'MMM d, yyyy' }, display: true, grid: { display: false }, ticks: { maxTicksLimit: 8 } },
        y: { display: true, border: { dash: [4, 4] }, grid: { color: 'rgba(0,0,0,0.05)' } }
      }
    }
  });

  document.getElementById('resetPriceZoom').onclick = () => {
    priceChartInstance.resetZoom();
    document.getElementById('resetPriceZoom').style.display = 'none';
  };
};

// --- Overfitting Lab heatmaps ---
// Maps a Sharpe ratio to a diverging colour on a shared scale: pale near
// zero, vivid green for strong positive, vivid red for negative.
function sharpeColor(s, scale) {
  if (s === null || s === undefined) return null;
  const t = Math.max(-1, Math.min(1, s / scale));
  const hue = t >= 0 ? 150 : 350;
  const light = 92 - 50 * Math.abs(t); // 92% (pale) -> 42% (vivid)
  return `hsl(${hue}, 65%, ${light}%)`;
}

function renderHeatmap(containerId, grid, shortWins, longWins, pickCell, bestCell, scale) {
  const container = document.getElementById(containerId);
  container.style.gridTemplateColumns = `46px repeat(${longWins.length}, 1fr)`;

  let html = '<div class="hm-corner">short&nbsp;\\&nbsp;long</div>';
  longWins.forEach(l => { html += `<div class="hm-axis">${l}</div>`; });

  grid.forEach((row, i) => {
    html += `<div class="hm-axis">${shortWins[i]}</div>`;
    row.forEach((val, j) => {
      let cls = 'hm-cell';
      if (pickCell && pickCell[0] === i && pickCell[1] === j) cls += ' hm-pick';
      if (bestCell && bestCell[0] === i && bestCell[1] === j) cls += ' hm-best';
      if (val === null || val === undefined) {
        html += `<div class="${cls} hm-invalid">-</div>`;
      } else {
        const bg = sharpeColor(val, scale);
        html += `<div class="${cls}" style="background:${bg}">${val.toFixed(2)}<small>Sharpe</small></div>`;
      }
    });
  });
  container.innerHTML = html;
}

window.updateHeatmaps = function (payloadJSON) {
  const p = JSON.parse(payloadJSON);
  document.getElementById('overfitting-card').style.display = 'block';
  statusDiv.innerHTML = '<i class="fa-solid fa-check text-success"></i>';

  // Shared colour scale across both panels so they are directly comparable.
  let maxAbs = 0.5;
  [].concat(...p.is_sharpe, ...p.oos_sharpe).forEach(v => {
    if (v !== null && v !== undefined) maxAbs = Math.max(maxAbs, Math.abs(v));
  });

  // In-sample panel: ring the cell you'd pick. Out-of-sample panel: ring the
  // SAME cell (so the eye tracks where the pick landed) plus mark the cell
  // that was actually best out of sample.
  renderHeatmap('heatmap-is', p.is_sharpe, p.short_windows, p.long_windows, p.is_best, null, maxAbs);
  renderHeatmap('heatmap-oos', p.oos_sharpe, p.short_windows, p.long_windows, p.is_best, p.oos_best, maxAbs);

  const [ps, pl] = p.is_best_params;
  const [bs, bl] = p.oos_best_params;
  const isSh = p.is_best_is_sharpe, oosSh = p.is_best_oos_sharpe;
  const rankTxt = p.is_best_oos_rank
    ? `#${p.is_best_oos_rank} of ${p.num_cells}`
    : 'unranked';
  const survived = p.is_best_oos_rank && p.is_best_oos_rank <= 2;

  document.getElementById('of-verdict').innerHTML =
    `In sample you would pick <strong>SMA(${ps}, ${pl})</strong> - the brightest ` +
    `cell, Sharpe <strong>${isSh.toFixed(2)}</strong>. Run those exact parameters ` +
    `on data they never saw and they score <strong>${oosSh.toFixed(2)}</strong>, ` +
    `ranking <strong>${rankTxt}</strong> out of sample. The genuinely best ` +
    `out-of-sample pair was <strong>SMA(${bs}, ${bl})</strong> ` +
    `(Sharpe ${p.oos_best_oos_sharpe.toFixed(2)}). ` +
    (survived
      ? `Here the in-sample choice happened to hold up - a robust sign.`
      : `The bright in-sample spike does not survive: that collapse, and the ` +
        `fact the winning cell moves, is curve over-fitting made visible.`);

  if (p.dsr !== undefined && p.dsr !== null) {
    const dsrClass = p.dsr >= 0.95 ? 'dsr-good' : 'dsr-bad';
    document.getElementById('of-verdict').innerHTML +=
      ` After accounting for having tried <strong>${p.dsr_n_trials}</strong> ` +
      `configurations, the Deflated Sharpe Ratio of the in-sample pick is ` +
      `<span class="${dsrClass}">${p.dsr.toFixed(2)}</span> - the probability ` +
      `its edge is real rather than selection luck. Below 0.95, treat it as ` +
      `indistinguishable from noise.`;
  }
};
