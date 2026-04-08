/* AFL Multi Builder - Frontend Application */

const API_BASE = '/api/v1';
let pipelineResults = null;
let backtestResults = null;

// ── Tabs ──────────────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    const target = `tab-${tab.dataset.tab}`;
    document.getElementById(target).classList.add('active');
  });
});

// ── Button handlers ───────────────────────────────────────────────────────
document.getElementById('btn-run-pipeline').addEventListener('click', runPipeline);
document.getElementById('btn-run-training').addEventListener('click', runTraining);
document.getElementById('btn-run-backtest').addEventListener('click', runBacktest);
document.getElementById('btn-summary').addEventListener('click', loadSummary);

// ── Status helpers ────────────────────────────────────────────────────────
function setStatus(msg, type = 'default') {
  const bar = document.getElementById('status-bar');
  bar.className = `status-bar ${type}`;
  bar.innerHTML = type === 'loading'
    ? `<span class="spinner"></span>${msg}`
    : msg;
}

function setButtons(disabled) {
  document.querySelectorAll('.btn').forEach(b => b.disabled = disabled);
}

// ── API calls ─────────────────────────────────────────────────────────────
async function apiPost(path, body = null) {
  const opts = { method: 'POST', headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(`${API_BASE}${path}`, opts);
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || r.statusText);
  }
  return r.json();
}

async function apiGet(path) {
  const r = await fetch(`${API_BASE}${path}`);
  if (!r.ok) throw new Error(r.statusText);
  return r.json();
}

// ── Pipeline ──────────────────────────────────────────────────────────────
async function runPipeline() {
  setButtons(true);
  setStatus('Running prediction pipeline...', 'loading');
  try {
    const result = await apiPost('/pipeline/run');
    pipelineResults = result;
    renderLegs(result.value_legs, 'legs-container');
    renderLegs(result.safe_legs, 'safe-legs-container');
    renderMultis(result.value_multis, 'value-multis-container');
    renderMultis(result.safe_multis, 'safe-multis-container');
    renderMultis(result.same_game_multis, 'same-game-container');
    setStatus(
      `✓ Pipeline complete in ${result.elapsed_seconds.toFixed(1)}s · ` +
      `${result.n_candidate_legs} candidate legs · ` +
      `${result.value_multis.length} value multis`,
      'success'
    );
  } catch (e) {
    setStatus(`✗ Pipeline error: ${e.message}`, 'error');
  } finally {
    setButtons(false);
  }
}

async function runTraining() {
  setButtons(true);
  setStatus('Training models (this may take a moment)...', 'loading');
  try {
    const result = await apiPost('/training/run');
    const mm = result.match_model || {};
    const pm = result.player_disposals_model || {};
    const mmBrier = mm.test_metrics?.calibrated_brier ?? 'N/A';
    setStatus(
      `✓ Training complete · Match model Brier: ${mmBrier} · ` +
      `Run ID: ${result.run_id}`,
      'success'
    );
  } catch (e) {
    setStatus(`✗ Training error: ${e.message}`, 'error');
  } finally {
    setButtons(false);
  }
}

async function runBacktest() {
  setButtons(true);
  setStatus('Running walk-forward backtest...', 'loading');
  try {
    const result = await apiPost('/backtest/run');
    backtestResults = result;
    renderBacktest(result);
    // Switch to backtest tab
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    document.querySelector('[data-tab="backtest"]').classList.add('active');
    document.getElementById('tab-backtest').classList.add('active');
    const m = result.overall_metrics || {};
    setStatus(
      `✓ Backtest complete · Brier: ${m.brier_score?.toFixed(4)} · ` +
      `Accuracy: ${(m.accuracy * 100).toFixed(1)}%`,
      'success'
    );
  } catch (e) {
    setStatus(`✗ Backtest error: ${e.message}`, 'error');
  } finally {
    setButtons(false);
  }
}

async function loadSummary() {
  setButtons(true);
  setStatus('Loading summary...', 'loading');
  try {
    const r = await apiGet('/reports/summary');
    const ds = r.data_summary;
    setStatus(
      `Fixtures: ${ds.completed_fixtures} completed, ${ds.upcoming_fixtures} upcoming · ` +
      `Players: ${ds.total_players} · Models: ${r.models_available.join(', ') || 'none trained'}`,
      'success'
    );
  } catch (e) {
    setStatus(`✗ ${e.message}`, 'error');
  } finally {
    setButtons(false);
  }
}

// ── Leg filters ───────────────────────────────────────────────────────────
function applyLegFilters() {
  if (!pipelineResults) return;
  const minEv = parseFloat(document.getElementById('filter-ev').value) || 0;
  const market = document.getElementById('filter-market').value;
  let legs = pipelineResults.value_legs.filter(l => l.ev >= minEv);
  if (market) legs = legs.filter(l => l.market_type === market);
  renderLegs(legs, 'legs-container');
}

// ── Renderers ─────────────────────────────────────────────────────────────
function renderLegs(legs, containerId) {
  const c = document.getElementById(containerId);
  if (!legs || legs.length === 0) {
    c.innerHTML = '<p class="placeholder">No legs match current criteria.</p>';
    return;
  }
  c.innerHTML = legs.map(leg => legCard(leg)).join('');
}

function legCard(leg) {
  const ev = leg.ev;
  const evClass = ev >= 0.05 ? 'positive' : ev >= 0 ? 'neutral' : 'negative';
  const evStr = (ev >= 0 ? '+' : '') + (ev * 100).toFixed(1) + '%';
  const prob = leg.model_probability;
  const conf = leg.confidence_score;
  const confLabel = conf >= 65 ? 'high' : conf >= 40 ? 'medium' : 'low';
  const cardClass = ev >= 0.05 ? 'positive-ev' : conf >= 65 ? 'high-confidence' : '';

  return `
    <div class="leg-card ${cardClass}">
      <div class="leg-header">
        <div>
          <div class="leg-selection">${formatSelection(leg.selection)}</div>
          <div class="leg-market">${formatMarket(leg.market_type)} · Fixture ${leg.fixture_id}</div>
        </div>
        <div class="leg-odds">$${leg.decimal_odds.toFixed(2)}</div>
      </div>
      <div class="prob-bar-container">
        <div class="prob-bar-label">
          <span>Model ${(prob * 100).toFixed(1)}%</span>
        </div>
        <div class="prob-bar"><div class="prob-bar-fill" style="width:${prob*100}%"></div></div>
      </div>
      <div class="leg-metrics">
        <div class="metric">
          <div class="metric-label">EV</div>
          <div class="metric-value ${evClass}">${evStr}</div>
        </div>
        <div class="metric">
          <div class="metric-label">Prob</div>
          <div class="metric-value neutral">${(prob*100).toFixed(1)}%</div>
        </div>
        <div class="metric">
          <div class="metric-label">Confidence</div>
          <div class="metric-value neutral">
            <span class="confidence-badge ${confLabel}">${conf.toFixed(0)}</span>
          </div>
        </div>
      </div>
      <div class="leg-explanation">${leg.explanation}</div>
    </div>
  `;
}

function renderMultis(multis, containerId) {
  const c = document.getElementById(containerId);
  if (!multis || multis.length === 0) {
    c.innerHTML = '<p class="placeholder">No multis available.</p>';
    return;
  }
  c.innerHTML = multis.map(m => multiCard(m)).join('');
}

function multiCard(m) {
  const evStr = (m.ev >= 0 ? '+' : '') + (m.ev * 100).toFixed(1) + '%';
  const evClass = m.ev >= 0.05 ? 'positive' : m.ev >= 0 ? 'neutral' : 'negative';
  const corrClass = m.correlation_label;
  const cardCorrClass = `${corrClass}-corr`;

  return `
    <div class="multi-card ${cardCorrClass}">
      <div class="multi-header">
        <div>
          <div class="multi-title">${m.n_legs}-Leg ${capitalize(m.multi_type)} Multi</div>
          <div class="multi-subtitle">ID: ${m.multi_id}</div>
        </div>
        <div class="multi-combined-odds">$${m.combined_odds.toFixed(2)}</div>
      </div>
      <div class="multi-metrics">
        <div class="metric">
          <div class="metric-label">EV</div>
          <div class="metric-value ${evClass}">${evStr}</div>
        </div>
        <div class="metric">
          <div class="metric-label">Hit Prob</div>
          <div class="metric-value neutral">${(m.adjusted_probability * 100).toFixed(1)}%</div>
        </div>
        <div class="metric">
          <div class="metric-label">Correlation</div>
          <div class="metric-value neutral"><span class="corr-badge ${corrClass}">${corrClass}</span></div>
        </div>
        <div class="metric">
          <div class="metric-label">Risk</div>
          <div class="metric-value neutral">${m.risk_score.toFixed(0)}/100</div>
        </div>
      </div>
      <div class="multi-explanation">${m.explanation}</div>
    </div>
  `;
}

function renderBacktest(result) {
  const c = document.getElementById('backtest-container');
  if (result.status !== 'success') {
    c.innerHTML = `<p class="placeholder">Backtest status: ${result.status}</p>`;
    return;
  }
  const m = result.overall_metrics || {};
  const cal = result.calibration_report || {};
  const clv = result.clv_analysis || {};
  const roi = result.roi_by_edge_threshold || [];

  let roiRows = '';
  roi.forEach(r => {
    const roiStr = (r.roi >= 0 ? '+' : '') + (r.roi * 100).toFixed(1) + '%';
    roiRows += `
      <tr>
        <td>${(r.edge_threshold * 100).toFixed(0)}%+</td>
        <td>${r.n_bets}</td>
        <td>${(r.hit_rate * 100).toFixed(1)}%</td>
        <td style="color:${r.roi >= 0 ? '#22c55e' : '#ef4444'}">${roiStr}</td>
        <td>${r.avg_odds.toFixed(2)}</td>
        <td>$${r.total_staked.toFixed(0)}</td>
      </tr>
    `;
  });

  c.innerHTML = `
    <div class="backtest-report">
      <h3 style="margin-bottom:16px;font-size:16px;">Walk-Forward Backtest Results</h3>
      <div style="font-size:12px;color:var(--c-text-muted);margin-bottom:14px;">
        Run ID: ${result.run_id} · ${result.n_total_predictions} predictions · ${result.n_folds} folds
      </div>
      <div class="backtest-grid">
        <div class="stat-box">
          <div class="stat-box-label">Brier Score</div>
          <div class="stat-box-value">${(m.brier_score || 0).toFixed(4)}</div>
          <div class="stat-box-desc">0=perfect · 0.25=baseline</div>
        </div>
        <div class="stat-box">
          <div class="stat-box-label">Log Loss</div>
          <div class="stat-box-value">${(m.log_loss || 0).toFixed(4)}</div>
          <div class="stat-box-desc">Lower is better</div>
        </div>
        <div class="stat-box">
          <div class="stat-box-label">Accuracy</div>
          <div class="stat-box-value">${((m.accuracy || 0) * 100).toFixed(1)}%</div>
          <div class="stat-box-desc">Win/loss prediction</div>
        </div>
        <div class="stat-box">
          <div class="stat-box-label">Cal. Improvement</div>
          <div class="stat-box-value">${((cal.brier_improvement || 0) * 10000).toFixed(0)}</div>
          <div class="stat-box-desc">Brier pts (×10000)</div>
        </div>
        ${clv.avg_model_vs_market !== undefined ? `
        <div class="stat-box">
          <div class="stat-box-label">Avg CLV</div>
          <div class="stat-box-value" style="color:${(clv.avg_model_vs_market||0) >= 0 ? '#22c55e' : '#ef4444'}">
            ${((clv.avg_model_vs_market || 0) >= 0 ? '+' : '')}${((clv.avg_model_vs_market || 0) * 100).toFixed(1)}%
          </div>
          <div class="stat-box-desc">Model vs market</div>
        </div>` : ''}
      </div>
      ${roiRows ? `
        <h4 style="margin:20px 0 8px;font-size:13px;color:var(--c-text-muted);">Simulated ROI by Edge Threshold</h4>
        <table class="roi-table">
          <thead><tr><th>Min Edge</th><th>Bets</th><th>Hit Rate</th><th>ROI</th><th>Avg Odds</th><th>Staked</th></tr></thead>
          <tbody>${roiRows}</tbody>
        </table>
      ` : ''}
    </div>
  `;
}

// ── Formatters ────────────────────────────────────────────────────────────
function formatSelection(sel) {
  const map = {
    'home_win': '🏠 Home Win',
    'away_win': '✈ Away Win',
  };
  if (map[sel]) return map[sel];
  if (sel.includes('over_')) return `⬆ Over ${sel.split('over_')[1]}`;
  if (sel.includes('under_')) return `⬇ Under ${sel.split('under_')[1]}`;
  if (sel.includes('_over_')) {
    const parts = sel.split('_over_');
    return `Player ${parts[0].replace('player_','')} Over ${parts[1]}`;
  }
  if (sel.includes('_under_')) {
    const parts = sel.split('_under_');
    return `Player ${parts[0].replace('player_','')} Under ${parts[1]}`;
  }
  return sel.replace(/_/g, ' ');
}

function formatMarket(mt) {
  const map = {
    'head_to_head': 'Head to Head',
    'line': 'Line',
    'total': 'Total',
    'player_disposals': 'Player Disposals',
  };
  return map[mt] || mt;
}

function capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1).replace(/_/g, ' ');
}

// ── Init: load summary on page load ──────────────────────────────────────
window.addEventListener('load', () => {
  apiGet('/reports/summary').then(r => {
    const ds = r.data_summary;
    setStatus(
      `Ready · ${ds.completed_fixtures} completed fixtures · ` +
      `${ds.upcoming_fixtures} upcoming · ` +
      `Models: ${r.models_available.length > 0 ? r.models_available.join(', ') : 'none trained yet'}`,
      'default'
    );
  }).catch(() => setStatus('API ready', 'default'));
});
