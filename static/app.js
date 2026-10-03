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
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.color = '#8b96a5';

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

// --- Data Source Tabs (Preset Ticker / Upload CSV) ---
const tabButtons = document.querySelectorAll('.tab-btn');
const panelTicker = document.getElementById('panel-ticker');
const panelUpload = document.getElementById('panel-upload');

tabButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    if (btn.classList.contains('active')) return;

    tabButtons.forEach(b => {
      b.classList.remove('active');
      b.setAttribute('aria-selected', 'false');
    });
    btn.classList.add('active');
    btn.setAttribute('aria-selected', 'true');

    const showUpload = btn.id === 'tab-btn-upload';
    panelTicker.style.display = showUpload ? 'none' : 'block';
    panelUpload.style.display = showUpload ? 'block' : 'none';

    // The tabs represent a single data-source choice: switching tabs clears
    // whichever source just went out of view so the two stay mutually exclusive.
    if (showUpload) {
      tickerSelect.value = '';
    } else if (csvUpload.files.length > 0) {
      csvUpload.value = '';
    }
    updateUIState();
  });
});

// --- Motion helpers (count-up, staggered reveal) ---
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// Animates a metric box's text from 0 up to whatever value is already
// sitting in the element (set synchronously by web_main.py just before this
// runs), preserving its exact formatting (decimals, %, "days", etc).
function animateMetricValue(el, duration = 700) {
  const targetText = el.textContent.trim();
  if (prefersReducedMotion) return;

  const match = targetText.match(/^(-?[\d,]+\.?\d*)(.*)$/);
  if (!match) return; // e.g. the "-" placeholder before any run

  const targetNum = parseFloat(match[1].replace(/,/g, ''));
  if (Number.isNaN(targetNum)) return;

  const suffix = match[2];
  const decimals = (match[1].split('.')[1] || '').length;
  const startTime = performance.now();

  function frame(now) {
    const t = Math.min(1, (now - startTime) / duration);
    const eased = 1 - Math.pow(1 - t, 3); // easeOutCubic
    el.textContent = (targetNum * eased).toFixed(decimals) + suffix;
    if (t < 1) {
      requestAnimationFrame(frame);
    } else {
      el.textContent = targetText; // snap to the exact original string
    }
  }
  requestAnimationFrame(frame);
}

// Restarts the CSS reveal-in animation on a set of elements, staggered by
// index, so results feel like they land rather than snap into place.
function reveal(elements, staggerMs = 40) {
  if (prefersReducedMotion) return;
  elements.forEach((el, i) => {
    if (!el) return;
    el.classList.remove('reveal-in');
    void el.offsetWidth; // force reflow so the animation restarts
    el.style.animationDelay = `${i * staggerMs}ms`;
    el.classList.add('reveal-in');
  });
}

// --- Loading skeleton: shown while the first backtest of a session runs ---
const chartsSkeleton = document.getElementById('charts-skeleton');

new MutationObserver(() => {
  const isRunning = runBtn.disabled && runBtn.innerText.includes('Running');
  if (isRunning && document.getElementById('charts-card').style.display !== 'block') {
    chartsSkeleton.style.display = 'block';
  } else {
    chartsSkeleton.style.display = 'none';
  }
}).observe(runBtn, { attributes: true, attributeFilter: ['disabled'] });

// --- Chart Generation Logic ---
let equityChartInstance = null;
let priceChartInstance = null;
let drawdownChartInstance = null;

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

const HEADLINE_METRIC_IDS = ['val-return', 'val-sharpe', 'val-drawdown', 'val-cagr'];
const SECONDARY_METRIC_IDS = ['val-winrate', 'val-alpha', 'val-inforatio', 'val-calmar', 'val-trades', 'val-duration'];

window.updateCharts = function (timestampsJSON, equityJSON, pricesJSON, tradesJSON, benchmarkJSON) {
  document.getElementById('charts-card').style.display = 'block';
  chartsSkeleton.style.display = 'none';
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

  // Count the headline/secondary metric values up from zero, and stagger the
  // metric boxes and chart panels in, so a completed run feels like it lands.
  HEADLINE_METRIC_IDS.concat(SECONDARY_METRIC_IDS).forEach(id => animateMetricValue(document.getElementById(id)));
  reveal(document.querySelectorAll('.metric-box'));
  reveal([
    document.getElementById('wrap-equity'),
    document.getElementById('wrap-drawdown'),
    document.getElementById('wrap-price'),
    document.querySelector('.table-container')
  ], 90);

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

  // Pair each entry with its exit so the price chart can draw a connecting
  // segment coloured by whether that round-trip made or lost money. A single
  // dataset is used with a null point breaking the line between pairs, so
  // this stays cheap even with hundreds of trades.
  const pairData = [];
  const pairWin = [];
  let openTrade = null;
  trades.forEach(trade => {
    if (trade.direction === 'LONG' || trade.direction === 'SHORT') {
      openTrade = trade;
    } else if (trade.direction === 'EXIT' && openTrade) {
      const won = openTrade.direction === 'LONG'
        ? trade.price >= openTrade.price
        : trade.price <= openTrade.price;
      pairData.push({ x: openTrade.timestamp * 1000, y: openTrade.price });
      pairWin.push(won);
      pairData.push({ x: trade.timestamp * 1000, y: trade.price });
      pairWin.push(won);
      pairData.push({ x: trade.timestamp * 1000, y: null });
      pairWin.push(won);
      openTrade = null;
    }
  });

  // --- 1. Equity Chart with Canvas Gradient ---
  const ctxEquity = document.getElementById('equityChart').getContext('2d');

  // Create Gradient for area under the curve
  let equityGradient = ctxEquity.createLinearGradient(0, 0, 0, 400);
  equityGradient.addColorStop(0, 'rgba(59, 130, 246, 0.35)');
  equityGradient.addColorStop(1, 'rgba(59, 130, 246, 0.0)');

  if (equityChartInstance) equityChartInstance.destroy();
  equityChartInstance = new Chart(ctxEquity, {
    type: 'line',
    data: {
      datasets: [
        {
          label: 'Portfolio Equity ($)',
          data: equityPoints,
          borderColor: '#3b82f6',
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
          borderColor: '#a78bfa',
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
        y: { display: true, border: { dash: [4, 4], color: '#232a36' }, grid: { color: 'rgba(255,255,255,0.05)' } }
      }
    }
  });

  document.getElementById('resetEquityZoom').onclick = () => {
    equityChartInstance.resetZoom();
    document.getElementById('resetEquityZoom').style.display = 'none';
  };

  // --- Drawdown (underwater) chart, sharing the equity chart's time range ---
  let peakEquity = -Infinity;
  const drawdownPoints = equityPoints.map(pt => {
    peakEquity = Math.max(peakEquity, pt.y);
    const pct = peakEquity > 0 ? ((pt.y - peakEquity) / peakEquity) * 100 : 0;
    return { x: pt.x, y: pct };
  });

  const ctxDrawdown = document.getElementById('drawdownChart').getContext('2d');
  let drawdownGradient = ctxDrawdown.createLinearGradient(0, 0, 0, 130);
  drawdownGradient.addColorStop(0, 'rgba(248, 113, 113, 0.05)');
  drawdownGradient.addColorStop(1, 'rgba(248, 113, 113, 0.45)');

  if (drawdownChartInstance) drawdownChartInstance.destroy();
  drawdownChartInstance = new Chart(ctxDrawdown, {
    type: 'line',
    data: {
      datasets: [{
        label: 'Drawdown',
        data: drawdownPoints,
        borderColor: '#f87171',
        backgroundColor: drawdownGradient,
        borderWidth: 1.5,
        fill: true,
        pointRadius: 0,
        pointHoverRadius: 6,
        tension: 0.15
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => `Drawdown: ${ctx.raw.y.toFixed(2)}%` } }
      },
      scales: {
        x: { type: 'time', display: false },
        y: {
          display: true, max: 0,
          border: { dash: [4, 4], color: '#232a36' },
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: { callback: v => `${v}%`, maxTicksLimit: 3 }
        }
      }
    }
  });

  // --- 2. Price Chart ---
  const ctxPrice = document.getElementById('priceChart').getContext('2d');
  if (priceChartInstance) priceChartInstance.destroy();
  priceChartInstance = new Chart(ctxPrice, {
    type: 'line',
    data: {
      datasets: [
        {
          type: 'line',
          label: 'Round-trip P&L',
          data: pairData,
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 0,
          fill: false,
          spanGaps: false,
          order: 3,
          segment: {
            borderColor: ctx => {
              const won = pairWin[ctx.p0DataIndex];
              if (won === undefined) return 'transparent';
              return won ? 'rgba(52, 211, 153, 0.55)' : 'rgba(248, 113, 113, 0.55)';
            }
          }
        },
        {
          type: 'line',
          label: 'Asset Price ($)',
          data: pricePoints,
          borderColor: '#c9d1d9',
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
          backgroundColor: '#34d399',
          borderColor: '#0b0e14',
          borderWidth: 2,
          pointStyle: 'circle',
          pointRadius: 10,
          order: 1
        },
        {
          type: 'scatter',
          label: 'Sell/Exit Signal',
          data: sellData,
          backgroundColor: '#f87171',
          borderColor: '#0b0e14',
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
        legend: {
          position: 'top',
          labels: {
            usePointStyle: true, boxWidth: 8, padding: 10,
            filter: item => item.text !== 'Round-trip P&L'
          }
        },
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
          filter: item => item.dataset.label !== 'Round-trip P&L',
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
        y: { display: true, border: { dash: [4, 4], color: '#232a36' }, grid: { color: 'rgba(255,255,255,0.05)' } }
      }
    }
  });

  document.getElementById('resetPriceZoom').onclick = () => {
    priceChartInstance.resetZoom();
    document.getElementById('resetPriceZoom').style.display = 'none';
  };
};

// --- Overfitting Lab heatmaps ---
// Maps a Sharpe ratio to a diverging colour on a shared scale: muted slate
// near zero, vivid green for strong positive, vivid red for negative. Tuned
// to sit clearly above the dark page background while staying legible under
// the fixed dark cell text at every point on the scale.
function sharpeColor(s, scale) {
  if (s === null || s === undefined) return null;
  const t = Math.max(-1, Math.min(1, s / scale));
  const a = Math.abs(t);
  const hue = t >= 0 ? 150 : 355;
  const sat = 12 + 55 * a;   // 12% (muted slate) -> 67% (vivid) at extremes
  const light = 55 - 10 * a; // 55% (muted) -> 45% (vivid) at extremes
  return `hsl(${hue}, ${sat}%, ${light}%)`;
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

  reveal(document.querySelectorAll('.heatmap-panel'), 120);

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

  if (p.dsr !== null && p.dsr !== undefined) {
    const cls = v => (v >= 0.95 ? 'dsr-good' : 'dsr-bad');
    document.getElementById('of-verdict').innerHTML +=
      ` The in-sample pick was chosen from <strong>${p.dsr_n_trials}</strong> settings. ` +
      `Its Deflated Sharpe Ratio is <span class="${cls(p.dsr)}">${p.dsr.toFixed(3)}</span> ` +
      `counting all ${p.dsr_n_trials} as separate tries, and ` +
      `<span class="${cls(p.dsr_eff)}">${p.dsr_eff.toFixed(3)}</span> counting them as about ` +
      `${p.n_eff.toFixed(1)} independent tries (their returns have an average correlation of ` +
      `${p.mean_corr.toFixed(2)}). That is the probability the in-sample Sharpe is not just the ` +
      `luckiest of the tries; below 0.95, treat it as noise. It does not say whether these exact ` +
      `settings keep working, which is what the out-of-sample rank above measures.`;
  }

  reveal([document.getElementById('of-verdict')], 0);
};
