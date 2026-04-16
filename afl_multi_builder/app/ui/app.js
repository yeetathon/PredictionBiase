/* AFL Prediction Intelligence — Frontend Application
 * Live-data system. No demo mode. Fails clearly if data is unavailable.
 */

const API_BASE = '/api/v1';
let pipelineResults = null;
let backtestResults = null;
let systemStatus = null;

// ── Tab navigation ────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
  });
});

// ── Button handlers ───────────────────────────────────────────────────────
document.getElementById('btn-run-pipeline').addEventListener('click', runPipeline);
document.getElementById('btn-run-training').addEventListener('click', runTraining);
document.getElementById('btn-run-backtest').addEventListener('click', runBacktest);
document.getElementById('btn-preflight').addEventListener('click', runPreflight);
document.getElementById('btn-refresh-health').addEventListener('click', loadSystemHealth);

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

// ── API helpers ───────────────────────────────────────────────────────────
async function apiPost(path, body = null) {
  const opts = { method: 'POST', headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(`${API_BASE}${path}`, opts);
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    // Surface preflight failure details clearly
    if (err.detail && typeof err.detail === 'object' && err.detail.failed_checks) {
      const fc = err.detail.failed_checks.map(c => `${c.name}: ${c.detail}`).join('\n');
      throw new Error(`${err.detail.message || 'Preflight failed'}\n${fc}`);
    }
    throw new Error(
      typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail)
    );
  }
  return r.json();
}

async function apiGet(path) {
  const r = await fetch(`${API_BASE}${path}`);
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(typeof err.detail === 'string' ? err.detail : r.statusText);
  }
  return r.json();
}

// ── System health ─────────────────────────────────────────────────────────
async function loadSystemHealth() {
  const btn = document.getElementById('btn-refresh-health');
  btn.disabled = true;
  btn.textContent = '...';

  try {
    const status = await apiGet('/system/status');
    systemStatus = status;
    renderSystemHealth(status);
  } catch (e) {
    renderHealthError(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = '↻ Refresh';
  }
}

function setHealthDot(id, state) {
  // state: 'ok' | 'warn' | 'error' | 'unknown'
  const item = document.getElementById(id);
  if (!item) return;
  const dot = item.querySelector('.health-dot');
  if (!dot) return;
  dot.className = `health-dot dot-${state}`;
}

function setHealthDetail(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function renderSystemHealth(status) {
  // Sportradar
  if (status.sportradar.configured) {
    setHealthDot('health-sportradar', 'ok');
    const sid = status.sportradar.season_id || 'no season ID';
    const quota = status.sportradar.quota;
    const quotaStr = quota ? ` · ${quota.used || 0}/${quota.total || '?'} calls` : '';
    setHealthDetail('hd-sportradar', `Connected${quotaStr} · Season: ${sid}`);
  } else {
    setHealthDot('health-sportradar', 'error');
    setHealthDetail('hd-sportradar', 'NOT CONFIGURED — set SPORTRADAR_API_KEY');
  }

  // Odds API
  if (status.odds_api.configured) {
    const rem = status.odds_api.remaining_requests;
    const remStr = rem !== undefined ? ` · ${rem} requests remaining` : '';
    setHealthDot('health-odds', 'ok');
    setHealthDetail('hd-odds', `Connected${remStr}`);
  } else {
    setHealthDot('health-odds', 'warn');
    setHealthDetail('hd-odds', 'Not configured — edge vs market disabled');
  }

  // Fixtures (from summary)
  apiGet('/reports/summary').then(r => {
    const ds = r.data_summary || {};
    const n = ds.upcoming_fixtures || 0;
    setHealthDot('health-fixtures', n > 0 ? 'ok' : 'warn');
    setHealthDetail('hd-fixtures', `${n} upcoming · ${ds.completed_fixtures || 0} completed`);
  }).catch(() => {
    setHealthDot('health-fixtures', 'unknown');
    setHealthDetail('hd-fixtures', 'Could not load');
  });

  // Models
  const models = status.models || [];
  if (models.length > 0) {
    setHealthDot('health-models', 'ok');
    setHealthDetail('hd-models', models.join(', '));
  } else {
    setHealthDot('health-models', 'warn');
    setHealthDetail('hd-models', 'No trained models — run Train Models');
  }

  // Data freshness
  const fresh = status.data_freshness || {};
  if (fresh.fresh === false) {
    setHealthDot('health-cache', 'warn');
    setHealthDetail('hd-cache', `Stale (${fresh.most_recent_age_hours}h old, limit ${fresh.ttl_hours}h)`);
  } else if (fresh.cache_files > 0) {
    setHealthDot('health-cache', 'ok');
    setHealthDetail('hd-cache', `Fresh (${fresh.most_recent_age_hours}h old · ${fresh.cache_files} files)`);
  } else {
    setHealthDot('health-cache', 'ok');
    setHealthDetail('hd-cache', 'No cache yet — will fetch on first run');
  }

  // Supported/disabled markets
  renderMarketStatus(status.supported_markets || [], status.disabled_markets || []);
}

function renderMarketStatus(supported, disabled) {
  const el = document.getElementById('health-markets');
  if (!el) return;
  let html = '<div class="market-status-row">';
  for (const m of supported) {
    html += `<span class="market-tag market-ok" title="${m.reason}">${m.market} ✓</span>`;
  }
  for (const m of disabled) {
    html += `<span class="market-tag market-disabled" title="${m.reason}">${m.market} ✗</span>`;
  }
  html += '</div>';
  el.innerHTML = html;
}

function renderHealthError(msg) {
  ['sportradar', 'odds', 'fixtures', 'models', 'cache'].forEach(k => {
    setHealthDot(`health-${k}`, 'unknown');
    setHealthDetail(`hd-${k}`, '—');
  });
  setStatus(`Health check failed: ${msg}`, 'error');
}

// ── Preflight ─────────────────────────────────────────────────────────────
async function runPreflight() {
  setButtons(true);
  setStatus('Running preflight checks...', 'loading');
  try {
    const report = await apiGet('/preflight');
    renderPreflightResult(report);
  } catch (e) {
    setStatus(`✗ Preflight error: ${e.message}`, 'error');
  } finally {
    setButtons(false);
  }
}

function renderPreflightResult(report) {
  const checks = report.checks || [];
  const failed = checks.filter(c => !c.passed && c.required);
  const warned = checks.filter(c => !c.passed && !c.required);
  const passed = checks.filter(c => c.passed);

  if (report.passed) {
    setStatus(
      `✓ Preflight passed (${passed.length} checks OK${warned.length ? `, ${warned.length} warnings` : ''})`,
      'success'
    );
  } else {
    const lines = failed.map(c => `${c.name}: ${c.detail}`).join(' | ');
    setStatus(`✗ Preflight FAILED — ${failed.length} required check(s): ${lines}`, 'error');
  }

  // Show stage log with preflight details
  const panel = document.getElementById('stage-log-panel');
  const log = document.getElementById('stage-log');
  panel.classList.remove('hidden');
  log.innerHTML = checks.map(c => {
    const icon = c.passed ? '✓' : (c.required ? '✗' : '⚠');
    const cls = c.passed ? 'stage-ok' : (c.required ? 'stage-fail' : 'stage-warn');
    return `<div class="stage-row ${cls}">
      <span class="stage-icon">${icon}</span>
      <span class="stage-name">${c.name}</span>
      <span class="stage-detail">${c.detail}</span>
      ${!c.passed && c.fix ? `<span class="stage-fix">Fix: ${c.fix}</span>` : ''}
    </div>`;
  }).join('');
}

// ── Pipeline ──────────────────────────────────────────────────────────────
async function runPipeline() {
  setButtons(true);
  setStatus('Running prediction pipeline...', 'loading');
  clearResults();
  try {
    const result = await apiPost('/pipeline/run');
    pipelineResults = result;

    // Show stage log
    renderStageLog(result.pipeline_stages);

    // Show/hide no-bet banner
    if (!result.has_bets) {
      showNoBetBanner(result.no_bet_reason, result.quality_thresholds);
    } else {
      hideNoBetBanner();
    }

    // Show filter panel and populate game filter
    populateGameFilter(result.value_legs || []);
    document.getElementById('filter-panel').classList.remove('hidden');

    // Render tabs
    renderLegs(result.value_legs, 'legs-container');
    renderLegs(result.safe_legs, 'safe-legs-container');
    renderMultis(result.value_multis, 'value-multis-container', result.no_bet_reason);
    renderMultis(result.safe_multis, 'safe-multis-container', result.no_bet_reason);
    renderMultis(result.same_game_multis, 'same-game-container', result.no_bet_reason);

    const src = result.data_source || 'Live';
    const odds = result.odds_source || 'unknown';
    const nQ = result.n_quality_legs ?? '?';
    const nC = result.n_candidate_legs ?? '?';
    const qualityStr = `${nQ}/${nC} legs passed quality gates`;
    setStatus(
      `${result.has_bets ? '✓' : '⚠'} Pipeline complete in ${result.elapsed_seconds.toFixed(1)}s · ` +
      `${result.n_fixtures} fixtures · ${qualityStr} · Data: ${src} · Odds: ${odds}`,
      result.has_bets ? 'success' : 'warn'
    );

    // Auto-refresh health
    loadSystemHealth();
  } catch (e) {
    const msg = e.message.includes('\n')
      ? e.message.split('\n').slice(0, 3).join(' | ')
      : e.message;
    setStatus(`✗ Pipeline failed: ${msg}`, 'error');
  } finally {
    setButtons(false);
  }
}

function clearResults() {
  ['legs-container', 'safe-legs-container', 'value-multis-container',
   'safe-multis-container', 'same-game-container'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = '<p class="placeholder">Loading...</p>';
  });
}

function renderStageLog(stages) {
  if (!stages) return;
  const panel = document.getElementById('stage-log-panel');
  const log = document.getElementById('stage-log');
  panel.classList.remove('hidden');

  const nGen = stages.legs_generated || 0;
  const nQ = stages.legs_passed_quality_gate ?? stages.legs_ranked ?? 0;
  const qualityOk = nQ > 0;
  const qualityVal = `${nQ} / ${nGen} (${nGen > 0 ? Math.round(nQ / nGen * 100) : 0}% pass rate)`;

  const entries = [
    { label: 'Preflight', value: stages.preflight || 'passed', ok: stages.preflight !== 'failed' },
    { label: 'Fixtures loaded', value: stages.fixtures_loaded, ok: stages.fixtures_loaded > 0 },
    { label: 'Features built', value: (stages.features_built || 0) + ' rows', ok: true },
    { label: 'Legs generated', value: nGen, ok: nGen > 0 },
    { label: 'Passed quality gates', value: qualityVal, ok: qualityOk },
    { label: 'Multis built', value: stages.multis_built || 0, ok: true },
  ];
  log.innerHTML = entries.map(e => `
    <div class="stage-row ${e.ok ? 'stage-ok' : 'stage-warn'}">
      <span class="stage-icon">${e.ok ? '✓' : '⚠'}</span>
      <span class="stage-name">${e.label}</span>
      <span class="stage-detail">${e.value}</span>
    </div>
  `).join('');
}

function showNoBetBanner(reason, thresholds) {
  let banner = document.getElementById('no-bet-banner');
  if (!banner) {
    banner = document.createElement('div');
    banner.id = 'no-bet-banner';
    banner.className = 'no-bet-banner';
    // Insert before filter-panel
    const filterPanel = document.getElementById('filter-panel');
    filterPanel.parentNode.insertBefore(banner, filterPanel);
  }
  let thresholdHtml = '';
  if (thresholds) {
    const items = [
      ['Min prob', thresholds.min_prob != null ? (thresholds.min_prob * 100).toFixed(0) + '%' : null],
      ['Min edge', thresholds.min_edge != null ? (thresholds.min_edge * 100).toFixed(0) + '%' : null],
      ['Min EV', thresholds.min_ev != null ? (thresholds.min_ev * 100).toFixed(0) + '%' : null],
      ['Min confidence', thresholds.min_confidence != null ? thresholds.min_confidence : null],
      ['Max odds', thresholds.max_odds != null ? '$' + thresholds.max_odds.toFixed(2) : null],
    ].filter(([, v]) => v != null);
    if (items.length) {
      thresholdHtml = '<div class="no-bet-thresholds">Active gates: ' +
        items.map(([k, v]) => `<span>${k}: ${v}</span>`).join(' · ') +
        '</div>';
    }
  }
  banner.innerHTML = `
    <div class="no-bet-icon">⚠</div>
    <div class="no-bet-content">
      <div class="no-bet-title">No high-confidence multis found for this slate</div>
      <div class="no-bet-reason">${reason || 'Insufficient legs cleared all quality gates.'}</div>
      ${thresholdHtml}
    </div>
  `;
  banner.classList.remove('hidden');
}

function hideNoBetBanner() {
  const banner = document.getElementById('no-bet-banner');
  if (banner) banner.classList.add('hidden');
}

async function runTraining() {
  setButtons(true);
  setStatus('Training models (this may take a moment)...', 'loading');
  try {
    const result = await apiPost('/training/run');
    const mm = result.match_model || {};
    const mmBrier = mm.test_metrics?.calibrated_brier ?? 'N/A';
    setStatus(
      `✓ Training complete · Match model Brier: ${mmBrier} · Run ID: ${result.run_id}`,
      'success'
    );
    loadSystemHealth();
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
      `✓ Backtest complete · Brier: ${m.brier_score?.toFixed(4)} · Accuracy: ${(m.accuracy * 100).toFixed(1)}%`,
      'success'
    );
  } catch (e) {
    setStatus(`✗ Backtest error: ${e.message}`, 'error');
  } finally {
    setButtons(false);
  }
}

// ── Filters ───────────────────────────────────────────────────────────────
function populateGameFilter(legs) {
  const sel = document.getElementById('filter-game');
  const games = [...new Set(legs.map(l => l.match_label).filter(Boolean))].sort();
  sel.innerHTML = '<option value="">All Games</option>' +
    games.map(g => `<option value="${g}">${g}</option>`).join('');
}

function applyFilters() {
  if (!pipelineResults) return;
  const game = document.getElementById('filter-game').value;
  const market = document.getElementById('filter-market').value;
  const minEv = parseFloat(document.getElementById('filter-ev').value) || 0;
  const conf = document.getElementById('filter-confidence').value;

  function filterLegs(legs) {
    return (legs || []).filter(l => {
      if (game && l.match_label !== game) return false;
      if (market && l.market_type !== market) return false;
      if (l.ev < minEv) return false;
      if (conf === 'High' && l.confidence_label !== 'High') return false;
      if (conf === 'Medium' && !['High', 'Medium'].includes(l.confidence_label)) return false;
      return true;
    });
  }

  renderLegs(filterLegs(pipelineResults.value_legs), 'legs-container');
  renderLegs(filterLegs(pipelineResults.safe_legs), 'safe-legs-container');
}

function clearFilters() {
  document.getElementById('filter-game').value = '';
  document.getElementById('filter-market').value = '';
  document.getElementById('filter-ev').value = '0.02';
  document.getElementById('filter-confidence').value = '';
  if (pipelineResults) {
    renderLegs(pipelineResults.value_legs, 'legs-container');
    renderLegs(pipelineResults.safe_legs, 'safe-legs-container');
  }
}

// ── Renderers ─────────────────────────────────────────────────────────────
function renderLegs(legs, containerId) {
  const c = document.getElementById(containerId);
  if (!legs || legs.length === 0) {
    c.innerHTML = '<p class="placeholder">No legs match current criteria.</p>';
    return;
  }
  // Group legs by game (match_label)
  const byGame = {};
  legs.forEach(leg => {
    const key = leg.match_label || `Fixture ${leg.fixture_id}`;
    if (!byGame[key]) byGame[key] = { startTime: leg.start_time, legs: [] };
    byGame[key].legs.push(leg);
  });

  let html = '';
  for (const [game, group] of Object.entries(byGame)) {
    html += `<div class="game-group">
      <div class="game-group-header">
        <span class="game-title">${game}</span>
        ${group.startTime ? `<span class="game-time">${formatTime(group.startTime)}</span>` : ''}
      </div>
      <div class="game-legs">
        ${group.legs.map(leg => legCard(leg)).join('')}
      </div>
    </div>`;
  }
  c.innerHTML = html;
}

function legCard(leg) {
  const ev = leg.ev;
  const evClass = ev >= 0.05 ? 'positive' : ev >= 0.01 ? 'neutral' : 'negative';
  const evStr = (ev >= 0 ? '+' : '') + (ev * 100).toFixed(1) + '%';
  const prob = leg.model_probability;
  const mktProb = leg.market_probability;
  const edge = leg.edge;
  const edgeStr = (edge >= 0 ? '+' : '') + (edge * 100).toFixed(1) + '%';
  const conf = leg.confidence_score;
  const confLabel = leg.confidence_label || (conf >= 65 ? 'High' : conf >= 40 ? 'Medium' : 'Low');
  const confCls = confLabel.toLowerCase();
  const cardClass = ev >= 0.05 ? 'positive-ev' : confLabel === 'High' ? 'high-confidence' : '';

  const impliedStr = mktProb ? (mktProb * 100).toFixed(1) + '%' : '—';
  const bookmaker = leg.bookmaker || '';

  // Signal agreement indicator
  const sigAgree = leg.signal_agreement || 0;
  const sigAgreeStr = sigAgree > 0 ? (sigAgree * 100).toFixed(0) + '%' : '—';
  const sigAgreeCls = sigAgree >= 0.75 ? 'positive' : sigAgree >= 0.45 ? 'neutral' : 'negative';

  // Signal breakdown chips
  const sigBreakdown = leg.signal_breakdown || {};
  const sigChips = Object.entries(sigBreakdown).map(([name, p]) =>
    `<span class="sig-chip" title="${name}">${name.slice(0,3).toUpperCase()} ${(p*100).toFixed(0)}%</span>`
  ).join('');

  // Top factors
  const topFactors = (leg.top_factors || []).slice(0, 2);
  const factorsHtml = topFactors.length
    ? `<div class="top-factors">${topFactors.map(f => `<div class="factor-row">• ${f}</div>`).join('')}</div>`
    : '';

  return `
    <div class="leg-card ${cardClass}">
      <div class="leg-header">
        <div class="leg-left">
          <div class="leg-selection">${leg.selection_label || formatSelection(leg.selection)}</div>
          <div class="leg-meta">
            <span class="leg-market-label">${leg.market_label || formatMarket(leg.market_type)}</span>
            ${leg.player_name ? `<span class="leg-player">${leg.player_name}</span>` : ''}
            ${leg.team_name && !leg.player_name ? `<span class="leg-team">${leg.team_name}</span>` : ''}
          </div>
        </div>
        <div class="leg-right">
          <div class="leg-odds-value">${leg.decimal_odds.toFixed(2)}</div>
          ${bookmaker ? `<div class="leg-bookmaker">${bookmaker}</div>` : ''}
        </div>
      </div>

      <div class="prob-comparison">
        <div class="prob-item">
          <span class="prob-label">Consensus</span>
          <span class="prob-value model-prob">${(prob * 100).toFixed(1)}%</span>
        </div>
        <div class="prob-arrow">→</div>
        <div class="prob-item">
          <span class="prob-label">Implied</span>
          <span class="prob-value market-prob">${impliedStr}</span>
        </div>
        <div class="prob-bar-wrap">
          <div class="prob-bar">
            <div class="prob-bar-fill model" style="width:${prob*100}%"></div>
            ${mktProb ? `<div class="prob-bar-fill market" style="width:${mktProb*100}%"></div>` : ''}
          </div>
        </div>
      </div>

      <div class="leg-metrics">
        <div class="metric">
          <div class="metric-label">EV</div>
          <div class="metric-value ${evClass}">${evStr}</div>
        </div>
        <div class="metric">
          <div class="metric-label">Edge</div>
          <div class="metric-value ${edge >= 0 ? 'positive' : 'negative'}">${edgeStr}</div>
        </div>
        <div class="metric">
          <div class="metric-label">Confidence</div>
          <div class="metric-value"><span class="confidence-badge ${confCls}">${confLabel}</span></div>
        </div>
        <div class="metric">
          <div class="metric-label">Signal Agree</div>
          <div class="metric-value ${sigAgreeCls}">${sigAgreeStr}</div>
        </div>
      </div>

      ${sigChips ? `<div class="sig-breakdown">${sigChips}</div>` : ''}
      ${factorsHtml}
      <div class="leg-explanation">${leg.explanation}</div>
    </div>
  `;
}

function renderMultis(multis, containerId, noBetReason) {
  const c = document.getElementById(containerId);
  if (!multis || multis.length === 0) {
    const msg = noBetReason
      ? `No high-confidence multis. ${noBetReason}`
      : 'No multis available.';
    c.innerHTML = `<p class="placeholder">${msg}</p>`;
    return;
  }
  c.innerHTML = multis.map(m => multiCard(m)).join('');
}

function multiCard(m) {
  const evStr = (m.ev >= 0 ? '+' : '') + (m.ev * 100).toFixed(1) + '%';
  const evClass = m.ev >= 0.05 ? 'positive' : m.ev >= 0 ? 'neutral' : 'negative';
  const corrClass = m.correlation_label || 'low';
  const hitPct = (m.adjusted_probability * 100).toFixed(1) + '%';

  // Group legs by game
  const legsByGame = {};
  (m.legs || []).forEach(leg => {
    const game = leg.match_label || `Leg ${leg.leg_id}`;
    if (!legsByGame[game]) legsByGame[game] = [];
    legsByGame[game].push(leg);
  });

  const legGroupHtml = Object.entries(legsByGame).map(([game, legs]) => `
    <div class="multi-game-group">
      <div class="multi-game-label">${game}</div>
      ${legs.map(leg => `
        <div class="multi-leg-row">
          <span class="multi-leg-sel">${leg.selection_label || leg.leg_id}</span>
          <span class="multi-leg-odds">${leg.decimal_odds ? '$' + leg.decimal_odds.toFixed(2) : ''}</span>
          <span class="confidence-badge ${(leg.confidence_label||'').toLowerCase()}">${leg.confidence_label || ''}</span>
        </div>
      `).join('')}
    </div>
  `).join('');

  // Risk/correlation notes
  const corrNote = corrClass === 'high' || corrClass === 'extreme'
    ? `<div class="corr-note warn">High correlation detected — legs move together</div>`
    : corrClass === 'medium'
    ? `<div class="corr-note info">Moderate correlation — some shared dependency</div>`
    : `<div class="corr-note ok">Low correlation — legs are largely independent</div>`;

  return `
    <div class="multi-card multi-corr-${corrClass}">
      <div class="multi-header">
        <div>
          <div class="multi-title">${m.n_legs}-Leg ${capitalize(m.multi_type)} Multi</div>
          <div class="multi-id">ID: ${m.multi_id}</div>
        </div>
        <div class="multi-odds-block">
          <div class="multi-combined-odds">$${m.combined_odds.toFixed(2)}</div>
          <div class="multi-odds-label">Combined</div>
        </div>
      </div>

      <div class="multi-metrics">
        <div class="metric">
          <div class="metric-label">EV</div>
          <div class="metric-value ${evClass}">${evStr}</div>
        </div>
        <div class="metric">
          <div class="metric-label">Hit Prob</div>
          <div class="metric-value neutral">${hitPct}</div>
        </div>
        <div class="metric">
          <div class="metric-label">Correlation</div>
          <div class="metric-value"><span class="corr-badge ${corrClass}">${corrClass}</span></div>
        </div>
        <div class="metric">
          <div class="metric-label">Risk</div>
          <div class="metric-value neutral">${m.risk_score.toFixed(0)}/100</div>
        </div>
      </div>

      ${corrNote}

      <div class="multi-legs-section">
        <div class="multi-legs-title">Legs (${m.n_legs})</div>
        ${legGroupHtml || (m.leg_ids || []).map(id => `<div class="multi-leg-row"><span class="multi-leg-sel">${id}</span></div>`).join('')}
      </div>

      <div class="multi-explanation">${m.explanation}</div>
    </div>
  `;
}

function renderBacktest(result) {
  const c = document.getElementById('backtest-container');
  if (result.status !== 'success') {
    c.innerHTML = `<div class="backtest-report"><p class="placeholder">Backtest status: ${result.status}</p></div>`;
    return;
  }
  const m = result.overall_metrics || {};
  const cal = result.calibration_report || {};
  const clv = result.clv_analysis || {};
  const roi = result.roi_by_edge_threshold || [];

  let roiRows = '';
  roi.forEach(r => {
    const roiStr = (r.roi >= 0 ? '+' : '') + (r.roi * 100).toFixed(1) + '%';
    roiRows += `<tr>
      <td>${(r.edge_threshold * 100).toFixed(0)}%+</td>
      <td>${r.n_bets}</td>
      <td>${(r.hit_rate * 100).toFixed(1)}%</td>
      <td style="color:${r.roi >= 0 ? '#22c55e' : '#ef4444'}">${roiStr}</td>
      <td>${r.avg_odds.toFixed(2)}</td>
      <td>$${r.total_staked.toFixed(0)}</td>
    </tr>`;
  });

  c.innerHTML = `
    <div class="backtest-report">
      <h3>Walk-Forward Backtest Results</h3>
      <div class="backtest-meta">Run ID: ${result.run_id} · ${result.n_total_predictions} predictions · ${result.n_folds} folds</div>
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
          <div class="stat-box-label">Calibration</div>
          <div class="stat-box-value">${((cal.brier_improvement || 0) * 10000).toFixed(0)}</div>
          <div class="stat-box-desc">Brier improvement (×10000)</div>
        </div>
        ${clv.avg_model_vs_market !== undefined ? `
        <div class="stat-box">
          <div class="stat-box-label">Avg CLV</div>
          <div class="stat-box-value" style="color:${(clv.avg_model_vs_market||0) >= 0 ? '#22c55e' : '#ef4444'}">
            ${((clv.avg_model_vs_market||0) >= 0 ? '+' : '')}${((clv.avg_model_vs_market||0)*100).toFixed(1)}%
          </div>
          <div class="stat-box-desc">Model vs market</div>
        </div>` : ''}
      </div>
      ${roiRows ? `
        <h4 class="roi-heading">Simulated ROI by Edge Threshold</h4>
        <table class="roi-table">
          <thead><tr><th>Min Edge</th><th>Bets</th><th>Hit Rate</th><th>ROI</th><th>Avg Odds</th><th>Staked</th></tr></thead>
          <tbody>${roiRows}</tbody>
        </table>
      ` : ''}
    </div>
  `;
}

// ── Formatters ────────────────────────────────────────────────────────────
function formatTime(raw) {
  if (!raw) return '';
  try {
    const d = new Date(raw);
    if (isNaN(d.getTime())) return raw;
    return d.toLocaleDateString('en-AU', { weekday: 'short', day: 'numeric', month: 'short' }) +
      ' ' + d.toLocaleTimeString('en-AU', { hour: '2-digit', minute: '2-digit' });
  } catch { return raw; }
}

function formatSelection(sel) {
  const map = {
    'home_win': 'Home Win',
    'away_win': 'Away Win',
  };
  if (map[sel]) return map[sel];
  if (sel.includes('over_')) return `Over ${sel.split('over_')[1]}`;
  if (sel.includes('under_')) return `Under ${sel.split('under_')[1]}`;
  if (sel.includes('_over_')) {
    const parts = sel.split('_over_');
    return `${parts[0].replace('player_', 'Player ')} Over ${parts[1]}`;
  }
  if (sel.includes('_under_')) {
    const parts = sel.split('_under_');
    return `${parts[0].replace('player_', 'Player ')} Under ${parts[1]}`;
  }
  return sel.replace(/_/g, ' ');
}

function formatMarket(mt) {
  const map = {
    'head_to_head': 'Head-to-Head',
    'line': 'Line',
    'total': 'Total',
    'player_disposals': 'Player Disposals',
    'bookmaker_odds': 'Bookmaker Odds',
  };
  return map[mt] || mt.replace(/_/g, ' ');
}

function capitalize(str) {
  return str.charAt(0).toUpperCase() + str.slice(1).replace(/_/g, ' ');
}

// ── Init: load system health on page load ─────────────────────────────────
window.addEventListener('load', () => {
  loadSystemHealth();
  setStatus('Ready — run preflight to verify data sources, then run the pipeline.', 'default');
});
