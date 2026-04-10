/**
 * AFL Multi Builder — Dashboard Application
 *
 * Workflow:
 *   1. Load weekend AFL fixtures from /api/v1/fixtures/weekend
 *   2. User ticks games, picks style, picks quantity
 *   3. POST /api/v1/generate → get job_id
 *   4. Poll /api/v1/generate/{job_id}/status every 1s
 *   5. Show stage-by-stage progress
 *   6. Render final multis as rich cards
 */

'use strict';

const API = '/api/v1';

// ── State ─────────────────────────────────────────────────────────────────────
let selectedFixtureIds = new Set();
let selectedStyle = 'balanced';
let selectedQty = 1;
let pollTimer = null;
let currentJobId = null;
let fixtureData = [];       // array of fixture objects from API
let styleData = {};         // style profiles from API
let jobStartTime = null;

// ── Init ──────────────────────────────────────────────────────────────────────
window.addEventListener('load', () => {
  checkApiStatus();
  loadWeekendFixtures();
  wireQuantityButtons();
});

// ── API helpers ───────────────────────────────────────────────────────────────
async function apiGet(path) {
  const r = await fetch(`${API}${path}`);
  if (!r.ok) {
    const body = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(body.detail || r.statusText);
  }
  return r.json();
}

async function apiPost(path, body) {
  const r = await fetch(`${API}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const b = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(b.detail || r.statusText);
  }
  return r.json();
}

// ── API status indicator ──────────────────────────────────────────────────────
async function checkApiStatus() {
  try {
    await apiGet('/health');
    document.getElementById('api-status-dot').className = 'status-dot dot-ok';
    document.getElementById('api-status-text').textContent = 'API Connected';
  } catch {
    document.getElementById('api-status-dot').className = 'status-dot dot-err';
    document.getElementById('api-status-text').textContent = 'API Offline';
  }
}

// ── Load weekend fixtures ─────────────────────────────────────────────────────
async function loadWeekendFixtures() {
  showLoadingScreen('Fetching upcoming AFL fixtures...');
  try {
    const data = await apiGet('/fixtures/weekend');
    fixtureData = data.fixtures || [];
    styleData = data.styles || {};

    if (fixtureData.length === 0) {
      showErrorScreen('No upcoming AFL fixtures found. The season may be between rounds.');
      return;
    }

    // Round label
    const roundLabel = `Season ${data.season} · Round ${data.round} · ${data.n_fixtures} Game${data.n_fixtures !== 1 ? 's' : ''}`;
    document.getElementById('round-label').textContent = roundLabel;

    // Default: select all fixtures
    selectedFixtureIds = new Set(fixtureData.map(f => f.fixture_id));

    renderFixtures(fixtureData);
    renderStylePicker(styleData);
    updateSelectionCount();
    updateGenerateButton();

    document.getElementById('footer-data-mode').textContent =
      data.fixtures[0]?.status === 'upcoming' ? 'live · sportradar' : 'demo';

    hideLoadingScreen();
  } catch (err) {
    showErrorScreen(`Failed to load fixtures: ${err.message}`);
  }
}

// ── Fixture rendering ─────────────────────────────────────────────────────────
function renderFixtures(fixtures) {
  const grid = document.getElementById('fixture-grid');
  grid.innerHTML = fixtures.map(f => fixtureCard(f)).join('');

  // Wire checkbox events
  grid.querySelectorAll('.fixture-checkbox').forEach(cb => {
    cb.addEventListener('change', () => {
      const id = parseInt(cb.dataset.id);
      if (cb.checked) {
        selectedFixtureIds.add(id);
      } else {
        selectedFixtureIds.delete(id);
      }
      // Update card visual
      cb.closest('.fixture-card').classList.toggle('fixture-selected', cb.checked);
      updateSelectionCount();
      updateGenerateButton();
    });
  });
}

function fixtureCard(f) {
  const checked = selectedFixtureIds.has(f.fixture_id) ? 'checked' : '';
  const selClass = checked ? 'fixture-selected' : '';
  const time = f.display_time || `${f.date} ${f.time}`.trim() || 'TBC';
  return `
    <label class="fixture-card ${selClass}" for="fx-${f.fixture_id}">
      <input
        type="checkbox"
        class="fixture-checkbox visually-hidden"
        id="fx-${f.fixture_id}"
        data-id="${f.fixture_id}"
        ${checked}
      />
      <div class="fixture-check-indicator">
        <div class="check-icon">✓</div>
      </div>
      <div class="fixture-teams">
        <div class="fixture-home">${escHtml(f.home_team)}</div>
        <div class="fixture-vs">vs</div>
        <div class="fixture-away">${escHtml(f.away_team)}</div>
      </div>
      <div class="fixture-meta">
        <div class="fixture-time">${escHtml(time)}</div>
        <div class="fixture-venue">${escHtml(f.venue)}</div>
      </div>
    </label>
  `;
}

function selectAllFixtures() {
  selectedFixtureIds = new Set(fixtureData.map(f => f.fixture_id));
  document.querySelectorAll('.fixture-checkbox').forEach(cb => {
    cb.checked = true;
    cb.closest('.fixture-card').classList.add('fixture-selected');
  });
  updateSelectionCount();
  updateGenerateButton();
}

function deselectAllFixtures() {
  selectedFixtureIds.clear();
  document.querySelectorAll('.fixture-checkbox').forEach(cb => {
    cb.checked = false;
    cb.closest('.fixture-card').classList.remove('fixture-selected');
  });
  updateSelectionCount();
  updateGenerateButton();
}

function updateSelectionCount() {
  const n = selectedFixtureIds.size;
  document.getElementById('selection-count').textContent =
    n === 0 ? 'No games selected'
    : n === 1 ? '1 game selected'
    : `${n} games selected`;
}

// ── Style picker ──────────────────────────────────────────────────────────────
function renderStylePicker(styles) {
  const grid = document.getElementById('style-grid');
  const order = ['safest', 'balanced', 'value', 'aggressive', 'longshot'];
  grid.innerHTML = order.map(key => {
    const s = styles[key] || {};
    const active = key === selectedStyle ? 'style-active' : '';
    return `
      <button class="style-card ${active}" data-style="${key}" onclick="selectStyle('${key}')">
        <div class="style-icon">${escHtml(s.icon || '')}</div>
        <div class="style-label">${escHtml(s.label || key)}</div>
        <div class="style-risk">${escHtml(s.risk_label || '')}</div>
        <div class="style-desc">${escHtml(s.description || '')}</div>
      </button>
    `;
  }).join('');
}

function selectStyle(key) {
  selectedStyle = key;
  document.querySelectorAll('.style-card').forEach(el => {
    el.classList.toggle('style-active', el.dataset.style === key);
  });
}

// ── Quantity picker ───────────────────────────────────────────────────────────
function wireQuantityButtons() {
  document.querySelectorAll('.qty-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const qty = parseInt(btn.dataset.qty);
      selectedQty = qty;
      document.querySelectorAll('.qty-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('qty-custom').value = '';
    });
  });

  document.getElementById('qty-custom').addEventListener('input', e => {
    const v = parseInt(e.target.value);
    if (v >= 1 && v <= 20) {
      selectedQty = v;
      document.querySelectorAll('.qty-btn').forEach(b => b.classList.remove('active'));
    }
  });
}

// ── Generate button state ─────────────────────────────────────────────────────
function updateGenerateButton() {
  const btn = document.getElementById('btn-generate');
  const hint = document.getElementById('generate-hint');
  const hasSelection = selectedFixtureIds.size > 0;
  btn.disabled = !hasSelection;
  hint.textContent = hasSelection
    ? `${selectedFixtureIds.size} game${selectedFixtureIds.size !== 1 ? 's' : ''} selected · ${selectedStyle} style · ${selectedQty} multi${selectedQty !== 1 ? 's' : ''}`
    : 'Select at least 1 game to continue';
  hint.classList.toggle('hint-ready', hasSelection);
}

// ── Generate workflow ─────────────────────────────────────────────────────────
async function startGenerate() {
  if (selectedFixtureIds.size === 0) return;

  // Lock UI
  document.getElementById('btn-generate').disabled = true;
  document.getElementById('section-results').classList.add('hidden');

  // Show progress section
  const progressSection = document.getElementById('section-progress');
  progressSection.classList.remove('hidden');
  progressSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

  document.getElementById('progress-summary').textContent = 'Submitting analysis request...';
  document.getElementById('progress-elapsed').textContent = '';

  jobStartTime = Date.now();

  try {
    const { job_id } = await apiPost('/generate', {
      fixture_ids: Array.from(selectedFixtureIds),
      multi_style: selectedStyle,
      n_multis: selectedQty,
    });
    currentJobId = job_id;
    startPolling(job_id);
  } catch (err) {
    document.getElementById('progress-summary').textContent = `Error: ${err.message}`;
    document.getElementById('btn-generate').disabled = false;
  }
}

// ── Polling ───────────────────────────────────────────────────────────────────
function startPolling(jobId) {
  clearInterval(pollTimer);
  pollTimer = setInterval(() => pollJobStatus(jobId), 1000);
}

async function pollJobStatus(jobId) {
  try {
    const data = await apiGet(`/generate/${jobId}/status`);
    renderProgress(data);

    if (data.status === 'complete') {
      clearInterval(pollTimer);
      renderResults(data.result);
      document.getElementById('btn-generate').disabled = false;
    } else if (data.status === 'error') {
      clearInterval(pollTimer);
      document.getElementById('progress-summary').textContent =
        `Analysis failed: ${data.error || 'Unknown error'}`;
      document.getElementById('btn-generate').disabled = false;
    }
  } catch (err) {
    // Network blip — keep polling
    console.warn('Poll error:', err.message);
  }
}

// ── Progress rendering ────────────────────────────────────────────────────────
function renderProgress(data) {
  const elapsed = ((Date.now() - jobStartTime) / 1000).toFixed(1);
  document.getElementById('progress-elapsed').textContent = `${elapsed}s`;

  // Summary line
  if (data.status === 'running' || data.status === 'pending') {
    const runningStage = (data.stages || []).find(s => s.status === 'running');
    document.getElementById('progress-summary').textContent =
      runningStage ? runningStage.label : 'Initialising...';
  } else if (data.status === 'complete') {
    document.getElementById('progress-summary').textContent =
      `Complete · ${data.n_candidate_legs} legs analysed · ${data.n_multis_generated} multi${data.n_multis_generated !== 1 ? 's' : ''} generated`;
  }

  // Stage list
  const stagesEl = document.getElementById('progress-stages');
  stagesEl.innerHTML = (data.stages || []).map(s => stageRow(s)).join('');
}

function stageRow(s) {
  let icon, cls;
  switch (s.status) {
    case 'done':    icon = '✓'; cls = 'stage-done';    break;
    case 'running': icon = '';  cls = 'stage-running';  break;
    case 'error':   icon = '✗'; cls = 'stage-error';    break;
    default:        icon = '';  cls = 'stage-pending';  break;
  }
  const spinner = s.status === 'running'
    ? '<span class="stage-spinner"></span>'
    : `<span class="stage-icon-text">${icon}</span>`;

  return `
    <div class="stage-row ${cls}">
      <div class="stage-status-col">${spinner}</div>
      <div class="stage-label-col">
        <span class="stage-label">${escHtml(s.label)}</span>
        ${s.detail ? `<span class="stage-detail">${escHtml(s.detail)}</span>` : ''}
      </div>
    </div>
  `;
}

// ── Results rendering ─────────────────────────────────────────────────────────
function renderResults(result) {
  if (!result) return;

  const section = document.getElementById('section-results');
  section.classList.remove('hidden');

  const n = result.n_generated;
  const requested = result.n_requested;
  const styleLabel = result.style_label || result.multi_style;
  const elapsed = result.elapsed_seconds;

  let summaryText = `${n} ${styleLabel} multi${n !== 1 ? 's' : ''} generated`;
  if (n < requested) summaryText += ` (${requested} requested — ${result.n_qualifying_legs} qualifying legs found)`;
  summaryText += ` · ${result.n_candidate_legs} candidates analysed · ${elapsed}s`;
  document.getElementById('results-summary').textContent = summaryText;

  const grid = document.getElementById('results-grid');
  if (!result.multis || result.multis.length === 0) {
    grid.innerHTML = `
      <div class="no-results">
        <div class="no-results-icon">📊</div>
        <div class="no-results-title">No multis could be constructed</div>
        <div class="no-results-sub">
          Try selecting more games, choosing a less restrictive style,
          or reducing the requested number of multis.
        </div>
      </div>`;
    return;
  }

  grid.innerHTML = result.multis.map((m, i) => multiCard(m, i + 1)).join('');

  // Smooth scroll to results
  setTimeout(() => {
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 100);
}

function multiCard(m, idx) {
  const ev = m.ev;
  const evStr = (ev >= 0 ? '+' : '') + (ev * 100).toFixed(1) + '%';
  const evClass = ev >= 0.05 ? 'ev-pos' : ev >= 0 ? 'ev-neu' : 'ev-neg';
  const corrClass = `corr-${m.correlation_label}`;
  const prob = m.adjusted_probability;
  const probPct = (prob * 100).toFixed(1);

  const legsHtml = (m.legs || []).map(leg => legRow(leg)).join('');

  return `
    <div class="multi-card ${corrClass}">
      <div class="multi-card-header">
        <div class="multi-card-title-row">
          <div class="multi-index">#${idx}</div>
          <div class="multi-title">
            <span class="multi-n-legs">${m.n_legs}-Leg</span>
            <span class="multi-style-badge style-badge-${m.style}">${escHtml(m.style_label)}</span>
            <span class="multi-type-badge">${capitalize(m.multi_type)}</span>
          </div>
        </div>
        <div class="multi-odds-display">
          <div class="multi-odds-value">$${m.combined_odds.toFixed(2)}</div>
          <div class="multi-odds-label">Combined Odds</div>
        </div>
      </div>

      <div class="multi-metrics-row">
        <div class="metric-pill">
          <div class="metric-pill-label">EV</div>
          <div class="metric-pill-value ${evClass}">${evStr}</div>
        </div>
        <div class="metric-pill">
          <div class="metric-pill-label">Hit Prob</div>
          <div class="metric-pill-value">${probPct}%</div>
        </div>
        <div class="metric-pill">
          <div class="metric-pill-label">Correlation</div>
          <div class="metric-pill-value">
            <span class="corr-badge corr-badge-${m.correlation_label}">${capitalize(m.correlation_label)}</span>
          </div>
        </div>
        <div class="metric-pill">
          <div class="metric-pill-label">Risk</div>
          <div class="metric-pill-value risk-val">${escHtml(m.risk_label)}</div>
        </div>
      </div>

      <div class="multi-legs-list">
        ${legsHtml}
      </div>

      <div class="multi-explanation">${escHtml(m.explanation)}</div>
    </div>
  `;
}

function legRow(leg) {
  const ev = leg.ev || 0;
  const evStr = (ev >= 0 ? '+' : '') + (ev * 100).toFixed(1) + '%';
  const evClass = ev >= 0.04 ? 'ev-pos' : ev >= 0 ? 'ev-neu' : 'ev-neg';
  const prob = leg.model_probability || 0;
  const conf = leg.confidence_score || 0;
  const confCls = conf >= 65 ? 'conf-high' : conf >= 45 ? 'conf-med' : 'conf-low';

  return `
    <div class="leg-row">
      <div class="leg-row-left">
        <div class="leg-row-match">${escHtml(leg.match_label || '')}</div>
        <div class="leg-row-selection">${escHtml(leg.selection_label || '')}</div>
        <div class="leg-row-market">${escHtml(leg.market_label || '')}</div>
      </div>
      <div class="leg-row-right">
        <div class="leg-row-odds">$${(leg.decimal_odds || 0).toFixed(2)}</div>
        <div class="leg-row-metrics">
          <span class="leg-metric">Model ${(prob * 100).toFixed(1)}%</span>
          <span class="leg-metric ${evClass}">EV ${evStr}</span>
          <span class="conf-badge ${confCls}">${conf >= 65 ? 'High' : conf >= 45 ? 'Med' : 'Low'} conf</span>
        </div>
      </div>
    </div>
  `;
}

// ── Reset ─────────────────────────────────────────────────────────────────────
function resetWorkflow() {
  clearInterval(pollTimer);
  currentJobId = null;
  jobStartTime = null;

  document.getElementById('section-progress').classList.add('hidden');
  document.getElementById('section-results').classList.add('hidden');
  document.getElementById('btn-generate').disabled = selectedFixtureIds.size === 0;

  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Screen helpers ────────────────────────────────────────────────────────────
function showLoadingScreen(msg) {
  document.getElementById('loading-detail').textContent = msg;
  document.getElementById('loading-screen').classList.remove('hidden');
  document.getElementById('error-screen').classList.add('hidden');
  document.getElementById('workflow').classList.add('hidden');
}

function hideLoadingScreen() {
  document.getElementById('loading-screen').classList.add('hidden');
  document.getElementById('workflow').classList.remove('hidden');
}

function showErrorScreen(msg) {
  document.getElementById('loading-screen').classList.add('hidden');
  document.getElementById('error-screen').classList.remove('hidden');
  document.getElementById('error-detail').textContent = msg;
  document.getElementById('workflow').classList.add('hidden');
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function escHtml(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function capitalize(str) {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1).replace(/_/g, ' ');
}
