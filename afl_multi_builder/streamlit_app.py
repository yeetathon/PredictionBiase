"""
AFL Prediction Intelligence Dashboard
Run:  streamlit run streamlit_app.py
"""
import sys
import time
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AFL Prediction Dashboard",
    page_icon="🏉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global font */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Metric cards */
.metric-card {
    background: #1e2130;
    border-radius: 12px;
    padding: 18px 22px;
    border-left: 4px solid #4f8ef7;
    margin-bottom: 10px;
}
.metric-card .label { font-size: 12px; color: #9aa0b4; text-transform: uppercase; letter-spacing: 0.08em; }
.metric-card .value { font-size: 28px; font-weight: 700; color: #ffffff; margin-top: 4px; }
.metric-card .sub   { font-size: 13px; color: #6b7a99; margin-top: 2px; }

/* Leg cards */
.leg-card {
    background: #1a1d2b;
    border: 1px solid #2a2f45;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
    position: relative;
}
.leg-card:hover { border-color: #4f8ef7; }

.leg-match    { font-size: 13px; color: #8892a8; margin-bottom: 4px; }
.leg-pick     { font-size: 19px; font-weight: 700; color: #ffffff; margin-bottom: 10px; }
.leg-badges   { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
.badge { padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }
.badge-odds   { background: #243044; color: #7eb8f7; }
.badge-model  { background: #1f3327; color: #5dba8c; }
.badge-edge   { background: #2e2020; color: #e07070; }
.badge-edge.pos { background: #1f3327; color: #5dba8c; }
.badge-conf-High   { background: #1f3327; color: #5dba8c; }
.badge-conf-Medium { background: #2a2820; color: #d4a84b; }
.badge-conf-Low    { background: #2e2020; color: #c06060; }
.leg-explain  { font-size: 13px; color: #8892a8; line-height: 1.5; border-top: 1px solid #2a2f45; padding-top: 10px; margin-top: 4px; }

/* Multi cards */
.multi-card {
    background: #1a1d2b;
    border: 1px solid #2a2f45;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 14px;
}
.multi-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; }
.multi-title  { font-size: 17px; font-weight: 700; color: #ffffff; }
.multi-odds   { font-size: 24px; font-weight: 700; color: #4f8ef7; }
.multi-legs-list { padding: 10px 0; border-top: 1px solid #2a2f45; }
.multi-leg-row  { font-size: 13px; color: #9aa0b4; padding: 4px 0; }
.multi-leg-row strong { color: #d0d5e5; }
.multi-explain { font-size: 12px; color: #6b7a99; margin-top: 8px; }

/* Status badge */
.status-demo { background:#1f2d44; color:#7eb8f7; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.status-live { background:#1f3327; color:#5dba8c; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.status-cache{ background:#2a2820; color:#d4a84b; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }

/* Section headers */
.section-header { font-size: 20px; font-weight: 700; color: #e0e4f0; margin: 8px 0 16px 0; }

/* Empty state */
.empty-state { text-align:center; padding:40px; color:#6b7a99; font-size:15px; }

/* Divider */
.hdivider { border:none; border-top:1px solid #2a2f45; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)


# ── Cached resources ──────────────────────────────────────────────────────

@st.cache_resource
def get_pipeline():
    from app.services.pipeline import PredictionPipeline
    return PredictionPipeline()

@st.cache_resource
def get_training_service():
    from app.services.training import TrainingService
    return TrainingService()

@st.cache_resource
def get_backtest_service():
    from app.services.backtest import WalkForwardBacktester
    return WalkForwardBacktester(min_train_samples=20, step_size=5)

@st.cache_resource
def get_loader():
    from app.data_ingestion.loader import DataLoader
    return DataLoader()


# ── Helpers ───────────────────────────────────────────────────────────────

def ev_color(ev: float) -> str:
    return "5dba8c" if ev >= 0 else "e07070"

def format_ev(ev: float) -> str:
    return f"{ev*100:+.1f}%"

def format_prob(p: float) -> str:
    return f"{p:.1%}"

def format_odds(o: float) -> str:
    return f"${o:.2f}"

def data_source_badge(label: str) -> str:
    cls = "status-live" if "Live" in label else ("status-cache" if "Cache" in label else "status-demo")
    return f'<span class="{cls}">{label}</span>'

def render_leg_card(leg: dict):
    match   = leg.get("match_label", "Unknown Match")
    pick    = leg.get("selection_label", leg.get("selection", ""))
    odds    = format_odds(leg.get("decimal_odds", 0))
    model   = format_prob(leg.get("model_probability", 0))
    market  = format_prob(leg.get("market_probability", 0))
    edge    = leg.get("edge", 0)
    ev      = leg.get("ev", 0)
    conf    = leg.get("confidence_label", "Low")
    explain = leg.get("explanation", "")
    start   = leg.get("start_time", "")
    market_label = leg.get("market_label", "")
    player  = leg.get("player_name", "")

    edge_cls = "pos" if edge >= 0 else ""
    match_str = f"{match}"
    if start:
        match_str += f" · {start}"
    if market_label:
        match_str += f" · {market_label}"
    if player:
        match_str += f" · {player}"

    st.markdown(f"""
    <div class="leg-card">
        <div class="leg-match">{match_str}</div>
        <div class="leg-pick">{pick}</div>
        <div class="leg-badges">
            <span class="badge badge-odds">Odds {odds}</span>
            <span class="badge badge-model">Model {model}</span>
            <span class="badge badge-odds">Market {market}</span>
            <span class="badge badge-edge {edge_cls}">Edge {edge:+.1%}</span>
            <span class="badge badge-edge {edge_cls}">EV {format_ev(ev)}</span>
            <span class="badge badge-conf-{conf}">{conf} Conf.</span>
        </div>
        <div class="leg-explain">{explain}</div>
    </div>
    """, unsafe_allow_html=True)

def render_multi_card(multi: dict, idx: int):
    legs = multi.get("legs", [])
    n    = multi.get("n_legs", len(legs))
    odds = format_odds(multi.get("combined_odds", 0))
    hit  = format_prob(multi.get("adjusted_probability", 0))
    ev   = multi.get("ev", 0)
    risk = multi.get("risk_score", 50)
    corr = multi.get("correlation_label", "").title()
    mtype = multi.get("multi_type", "").replace("_", " ").title()
    explain = multi.get("explanation", "")

    legs_html = ""
    for l in legs:
        sel = l.get("selection_label", "")
        match = l.get("match_label", "")
        lodds = l.get("decimal_odds")
        lconf = l.get("confidence_label", "")
        legs_html += f'<div class="multi-leg-row">→ <strong>{sel}</strong> ({match}) @ {format_odds(lodds) if lodds else "?"} · {lconf}</div>'

    st.markdown(f"""
    <div class="multi-card">
        <div class="multi-header">
            <div>
                <div class="multi-title">{n}-Leg {mtype}</div>
                <div style="font-size:13px;color:#8892a8;margin-top:3px">Hit Prob {hit} · EV {format_ev(ev)} · Risk {risk:.0f}/100 · Corr {corr}</div>
            </div>
            <div class="multi-odds">{odds}</div>
        </div>
        <div class="multi-legs-list">{legs_html}</div>
        <div class="multi-explain">{explain}</div>
    </div>
    """, unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────
if "pipeline_result" not in st.session_state:
    st.session_state["pipeline_result"] = None
if "auto_mode" not in st.session_state:
    st.session_state["auto_mode"] = False
if "last_updated" not in st.session_state:
    st.session_state["last_updated"] = None
if "training_result" not in st.session_state:
    st.session_state["training_result"] = None
if "backtest_result" not in st.session_state:
    st.session_state["backtest_result"] = None
if "auto_interval" not in st.session_state:
    st.session_state["auto_interval"] = 15


def run_pipeline():
    """Execute pipeline and store result in session state."""
    import os
    os.makedirs("data/models", exist_ok=True)
    pipeline = get_pipeline()
    result = pipeline.run()
    st.session_state["pipeline_result"] = result
    st.session_state["last_updated"] = datetime.now().strftime("%H:%M:%S")
    return result


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏉 AFL Prediction Dashboard")
    st.caption("Calibrated predictions · Edge detection · Smart multis")
    st.divider()

    # ── Data source status ──────────────────────────────────────
    from app.core.config import settings

    if not settings.is_afl_data_configured:
        st.error(
            "**AFL_DATA_AUTHKEY not configured.**\n\n"
            "Add it to your `.env` file:\n```\nAFL_DATA_AUTHKEY=your_key_here\n```\n"
            "Then restart the app. Live API data is required — demo mode has been removed."
        )
        st.stop()

    mode = settings.effective_data_mode
    if mode == "live":
        ds_label = "AFL Data Sports Group API (Live)"
    elif mode == "cache":
        ds_label = "AFL Data Sports Group API (Cached)"
    else:
        ds_label = "AFL Data Sports Group API"

    pr = st.session_state.get("pipeline_result")
    if pr:
        ds_label = pr.get("data_source", ds_label)

    st.markdown(f"**Data Source:** {data_source_badge(ds_label)}", unsafe_allow_html=True)
    st.divider()

    # ── Controls ───────────────────────────────────────────────
    st.subheader("Controls")

    col_run, col_auto = st.columns([2, 1])
    with col_run:
        run_btn = st.button("▶ Run Pipeline", use_container_width=True, type="primary")
    with col_auto:
        auto = st.toggle("Auto", value=st.session_state["auto_mode"])
        st.session_state["auto_mode"] = auto

    if run_btn:
        with st.spinner("Running prediction pipeline..."):
            try:
                result = run_pipeline()
                st.success(f"✓ {result['n_candidate_legs']} legs generated ({result['data_source']})")
            except Exception as e:
                err_msg = str(e)
                if "No upcoming AFL fixtures" in err_msg or "NoUpcomingFixtures" in err_msg:
                    st.warning(f"No upcoming fixtures: {err_msg}")
                else:
                    st.error(f"Pipeline error: {err_msg}")

    if st.session_state["auto_mode"]:
        interval = st.slider("Refresh interval (min)", 5, 60, st.session_state["auto_interval"])
        st.session_state["auto_interval"] = interval
        last = st.session_state.get("last_updated", "Never")
        st.caption(f"Auto refresh every {interval} min · Last: {last}")

    st.divider()

    # ── Model actions ──────────────────────────────────────────
    with st.expander("Model Actions", expanded=False):
        if st.button("⚙ Train Models", use_container_width=True):
            with st.spinner("Training..."):
                try:
                    svc = get_training_service()
                    r = svc.run_all()
                    st.session_state["training_result"] = r
                    st.success("Models trained")
                except Exception as e:
                    st.error(f"{e}")

        if st.button("📊 Run Backtest", use_container_width=True):
            with st.spinner("Running backtest..."):
                try:
                    bt = get_backtest_service()
                    r = bt.run()
                    st.session_state["backtest_result"] = r
                    if r.get("status") == "success":
                        m = r["overall_metrics"]
                        st.success(f"Brier {m['brier_score']:.4f} · Acc {m['accuracy']:.1%}")
                    else:
                        st.warning(r.get("status"))
                except Exception as e:
                    st.error(f"{e}")

    st.divider()

    # ── Filters ────────────────────────────────────────────────
    st.subheader("Filters")
    min_ev_pct = st.slider("Min EV (%)", -10, 30, 2)
    min_ev = min_ev_pct / 100
    min_conf = st.selectbox("Min Confidence", ["Any", "Low", "Medium", "High"], index=0)
    mkt_opts = ["All", "Head-to-Head", "Player Disposals", "Line", "Total"]
    mkt_filter = st.selectbox("Market Type", mkt_opts, index=0)
    min_legs = st.selectbox("Min Multi Legs", [2, 3, 4], index=0)
    max_legs = st.selectbox("Max Multi Legs", [2, 3, 4], index=2)
    max_corr = st.slider("Max Multi Correlation", 0.1, 1.0, 0.7, step=0.05)

    st.divider()

    # ── Summary stats ──────────────────────────────────────────
    if pr:
        st.subheader("Session Summary")
        st.metric("Fixtures", pr.get("n_fixtures", 0))
        st.metric("Candidate Legs", pr.get("n_candidate_legs", 0))
        st.caption(f"Run {pr.get('run_id','')} · {pr.get('elapsed_seconds',0):.1f}s")
    else:
        loader = get_loader()
        try:
            fixtures = loader.load_fixtures_df()
            if not fixtures.empty:
                col1, col2 = st.columns(2)
                col1.metric("Completed", len(fixtures[fixtures["status"] == "completed"]))
                col2.metric("Upcoming", len(fixtures[fixtures["status"] == "upcoming"]))
        except Exception:
            pass


# ── Filter helpers ────────────────────────────────────────────────────────
_CONF_ORDER = {"High": 3, "Medium": 2, "Low": 1, "Any": 0}

def filter_legs(legs):
    out = []
    for l in legs:
        if l.get("ev", 0) < min_ev:
            continue
        if min_conf != "Any":
            if _CONF_ORDER.get(l.get("confidence_label", "Low"), 0) < _CONF_ORDER.get(min_conf, 0):
                continue
        if mkt_filter != "All":
            if l.get("market_label", "") != mkt_filter:
                continue
        out.append(l)
    return out

def filter_multis(multis):
    out = []
    for m in multis:
        if m.get("ev", 0) < min_ev:
            continue
        if m.get("correlation_score", 1) > max_corr:
            continue
        n = m.get("n_legs", 0)
        if not (min_legs <= n <= max_legs):
            continue
        out.append(m)
    return out


# ── Main area ─────────────────────────────────────────────────────────────
# Header row
hcol1, hcol2 = st.columns([3, 1])
with hcol1:
    st.markdown("# AFL Prediction Intelligence Dashboard")
with hcol2:
    last = st.session_state.get("last_updated")
    if last:
        st.markdown(f"<div style='text-align:right;color:#8892a8;font-size:13px;padding-top:20px'>Last updated: {last}</div>", unsafe_allow_html=True)

if not pr:
    st.info("👆 Click **Run Pipeline** in the sidebar to generate predictions.")

pr = st.session_state.get("pipeline_result")

tab_value, tab_safe, tab_multis, tab_sgm, tab_edge, tab_insights, tab_system = st.tabs([
    "🔥 Value Picks",
    "🛡 Safe Picks",
    "📦 Multi Builder",
    "🎲 Same-Game",
    "📡 Edge Intelligence",
    "📊 Model Insights",
    "⚙ System",
])


# ── Tab 1: Value Picks ────────────────────────────────────────────────────
with tab_value:
    st.markdown('<div class="section-header">🔥 Top Value Picks</div>', unsafe_allow_html=True)
    st.caption("Ranked by Expected Value (EV). Positive EV indicates statistical edge over market pricing.")

    if pr:
        legs = filter_legs(pr.get("value_legs", []))

        # Sort controls
        sort_by = st.selectbox("Sort by", ["EV", "Edge", "Model Probability", "Confidence", "Odds"],
                                key="sort_value")
        sort_map = {
            "EV": "ev", "Edge": "edge", "Model Probability": "model_probability",
            "Confidence": "confidence_score", "Odds": "decimal_odds"
        }
        legs = sorted(legs, key=lambda x: x.get(sort_map[sort_by], 0), reverse=True)

        # Team filter
        teams = sorted({l.get("home_team_name", "") for l in legs} |
                       {l.get("away_team_name", "") for l in legs} - {""})
        team_filter = st.selectbox("Filter by team", ["All teams"] + teams, key="team_value")
        if team_filter != "All teams":
            legs = [l for l in legs if team_filter in (l.get("home_team_name",""), l.get("away_team_name",""))]

        if legs:
            st.caption(f"Showing {len(legs)} legs")
            for leg in legs:
                render_leg_card(leg)
        else:
            st.markdown('<div class="empty-state">No picks pass current filters.<br>Try lowering Min EV or changing filters.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state">Run the pipeline to generate value picks.</div>', unsafe_allow_html=True)


# ── Tab 2: Safe Picks ─────────────────────────────────────────────────────
with tab_safe:
    st.markdown('<div class="section-header">🛡 Safe Picks</div>', unsafe_allow_html=True)
    st.caption("High-probability selections with confirmed edge. Suitable for lower-risk multi building.")

    if pr:
        legs = filter_legs(pr.get("safe_legs", []))
        legs = sorted(legs, key=lambda x: x.get("model_probability", 0), reverse=True)

        if legs:
            st.caption(f"Showing {len(legs)} legs")
            for leg in legs:
                render_leg_card(leg)
        else:
            st.markdown('<div class="empty-state">No safe picks pass current filters.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state">Run the pipeline first.</div>', unsafe_allow_html=True)


# ── Tab 3: Multi Builder ──────────────────────────────────────────────────
with tab_multis:
    st.markdown('<div class="section-header">📦 Multi Builder</div>', unsafe_allow_html=True)
    st.caption("Multi combinations with correlation penalty applied — not naïve independent multiplication.")

    if pr:
        multis = filter_multis(pr.get("value_multis", []))
        safe_multis = filter_multis(pr.get("safe_multis", []))

        sub_tab_v, sub_tab_s = st.tabs(["Value Multis", "Safe Multis"])

        with sub_tab_v:
            multis_sorted = sorted(multis, key=lambda x: x.get("ev", 0), reverse=True)
            if multis_sorted:
                st.caption(f"{len(multis_sorted)} multis")
                for i, m in enumerate(multis_sorted):
                    render_multi_card(m, i)
            else:
                st.markdown('<div class="empty-state">No value multis pass current filters.</div>', unsafe_allow_html=True)

        with sub_tab_s:
            safe_sorted = sorted(safe_multis, key=lambda x: x.get("adjusted_probability", 0), reverse=True)
            if safe_sorted:
                st.caption(f"{len(safe_sorted)} multis")
                for i, m in enumerate(safe_sorted):
                    render_multi_card(m, i)
            else:
                st.markdown('<div class="empty-state">No safe multis pass current filters.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state">Run the pipeline first.</div>', unsafe_allow_html=True)


# ── Tab 4: Same-Game Multis ───────────────────────────────────────────────
with tab_sgm:
    st.markdown('<div class="section-header">🎲 Same-Game Multis</div>', unsafe_allow_html=True)
    st.caption("All legs from a single match. Higher correlation — correlation penalty is heavier but game-specific analysis is sharper.")

    if pr:
        sgm = filter_multis(pr.get("same_game_multis", []))

        # Group by match
        by_match: dict = {}
        for m in sgm:
            legs = m.get("legs", [])
            match = legs[0].get("match_label", "Unknown") if legs else "Unknown"
            by_match.setdefault(match, []).append(m)

        if by_match:
            for match_lbl, match_multis in by_match.items():
                st.markdown(f"**{match_lbl}**")
                for i, m in enumerate(sorted(match_multis, key=lambda x: x.get("ev", 0), reverse=True)):
                    render_multi_card(m, i)
                st.divider()
        else:
            st.markdown('<div class="empty-state">No same-game multis available for current filters.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="empty-state">Run the pipeline first.</div>', unsafe_allow_html=True)


# ── Tab 5: Edge Intelligence ─────────────────────────────────────────────
with tab_edge:
    st.markdown('<div class="section-header">📡 Edge Intelligence</div>', unsafe_allow_html=True)
    st.caption(
        "Scraped signals from AFL news, RSS feeds, and official announcements. "
        "Updates every 30 minutes. Injuries, late outs, suspensions, and team news."
    )

    _SEV_COLOR = {
        "critical": ("#ff4b4b", "#2d0000"),
        "high":     ("#ff8c42", "#2d1500"),
        "medium":   ("#f5c518", "#2a1f00"),
        "low":      ("#5dba8c", "#0f2a1e"),
    }
    _TYPE_EMOJI = {
        "injury":       "🤕",
        "late_out":     "⚠️",
        "suspension":   "🚫",
        "selection":    "📋",
        "team_news":    "📰",
        "weather_alert":"🌧️",
        "breaking":     "🔴",
    }

    def render_signal_card(sig):
        sev = getattr(sig, "severity", "low")
        fg, bg = _SEV_COLOR.get(sev, ("#9aa0b4", "#1a1d2b"))
        stype = getattr(sig, "signal_type", "team_news")
        emoji = _TYPE_EMOJI.get(stype, "📰")
        team = getattr(sig, "team_name", "") or "—"
        player = getattr(sig, "player_name", "")
        headline = getattr(sig, "headline", "")
        summary = getattr(sig, "summary", "")
        source = getattr(sig, "source_name", "")
        pub = getattr(sig, "published_at", None)
        pub_str = pub.strftime("%d %b %H:%M") if pub else ""
        stale = getattr(sig, "is_stale", False)
        stale_str = " · ⏰ STALE" if stale else ""

        st.markdown(f"""
        <div style="background:{bg};border:1px solid {fg}33;border-left:4px solid {fg};
                    border-radius:10px;padding:14px 16px;margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <span style="font-size:15px;font-weight:700;color:#e0e4f0;">{emoji} {headline}</span>
                <span style="background:{fg}22;color:{fg};padding:3px 10px;border-radius:20px;
                             font-size:11px;font-weight:700;text-transform:uppercase;">{sev}{stale_str}</span>
            </div>
            <div style="font-size:12px;color:#9aa0b4;margin-bottom:6px;">
                <strong style="color:#d0d5e5;">{team}</strong>
                {"· " + player if player else ""}
                · <em>{stype.replace("_"," ").title()}</em>
                · {source} · {pub_str}
            </div>
            <div style="font-size:13px;color:#b0b8cc;line-height:1.5;">{summary}</div>
        </div>
        """, unsafe_allow_html=True)

    loader = get_loader()
    if loader.edge is None:
        st.info(
            "Edge intelligence scraping is **disabled**. "
            "Set `ENABLE_SCRAPING=true` in your `.env` file to activate it."
        )
    else:
        col_refresh, col_filter_sev, col_filter_type, col_filter_team = st.columns([1, 1, 1, 2])
        with col_refresh:
            refresh_edge = st.button("🔄 Refresh Signals", use_container_width=True)

        if refresh_edge or "edge_signals" not in st.session_state:
            with st.spinner("Fetching edge signals..."):
                try:
                    signals = loader.load_all_edge_signals()
                    st.session_state["edge_signals"] = signals
                    st.session_state["edge_signals_ts"] = datetime.now().strftime("%H:%M:%S")
                except Exception as e:
                    st.warning(f"Edge intelligence fetch failed: {e}")
                    st.session_state["edge_signals"] = []

        signals = st.session_state.get("edge_signals", [])
        edge_ts = st.session_state.get("edge_signals_ts", "")

        if edge_ts:
            st.caption(f"Last fetched: {edge_ts} · {len(signals)} signals")

        with col_filter_sev:
            sev_filter = st.selectbox("Severity", ["All", "Critical", "High", "Medium", "Low"],
                                      key="edge_sev")
        with col_filter_type:
            type_opts = ["All", "Injury", "Late Out", "Suspension", "Selection", "Team News", "Breaking"]
            type_filter = st.selectbox("Type", type_opts, key="edge_type")
        with col_filter_team:
            all_teams_edge = sorted({getattr(s, "team_name", "") for s in signals} - {""})
            team_filter_edge = st.selectbox("Team", ["All"] + all_teams_edge, key="edge_team")

        # Apply filters
        filtered_signals = []
        for s in signals:
            if sev_filter != "All" and getattr(s, "severity", "").title() != sev_filter:
                continue
            if type_filter != "All":
                stype_display = getattr(s, "signal_type", "").replace("_", " ").title()
                if stype_display != type_filter:
                    continue
            if team_filter_edge != "All" and getattr(s, "team_name", "") != team_filter_edge:
                continue
            filtered_signals.append(s)

        # Sort: critical first, then by published_at desc
        _sev_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        filtered_signals.sort(
            key=lambda s: (
                _sev_rank.get(getattr(s, "severity", "low"), 3),
                -(getattr(s, "published_at", datetime.min).timestamp()
                  if hasattr(getattr(s, "published_at", None), "timestamp") else 0)
            )
        )

        if filtered_signals:
            st.caption(f"Showing {len(filtered_signals)} signals")
            for sig in filtered_signals:
                render_signal_card(sig)
        elif signals:
            st.markdown('<div class="empty-state">No signals match current filters.</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="empty-state">No signals found. Click Refresh Signals to fetch latest AFL news.</div>',
                unsafe_allow_html=True,
            )


# ── Tab 6: Model Insights ─────────────────────────────────────────────────
with tab_insights:
    st.markdown('<div class="section-header">📊 Model Insights</div>', unsafe_allow_html=True)

    bt = st.session_state.get("backtest_result")
    tr = st.session_state.get("training_result")

    if bt and bt.get("status") == "success":
        metrics = bt["overall_metrics"]
        cal = bt.get("calibration_report", {})
        clv = bt.get("clv_analysis", {})

        # Key metrics row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Brier Score", f"{metrics['brier_score']:.4f}",
                  help="Lower is better. 0.25 = random, 0.0 = perfect.")
        c2.metric("Log Loss", f"{metrics['log_loss']:.4f}")
        c3.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        c4.metric("Calibration Gain",
                  f"{cal.get('brier_improvement', 0)*100:.2f} pts")

        st.divider()

        # Calibration quality
        if cal:
            st.subheader("Calibration Quality")
            ec_before = cal.get("expected_calibration_error_before", None)
            ec_after  = cal.get("expected_calibration_error_after", None)
            if ec_before and ec_after:
                col1, col2 = st.columns(2)
                col1.metric("ECE Before Calibration", f"{ec_before:.4f}")
                col2.metric("ECE After Calibration", f"{ec_after:.4f}", delta=f"{ec_after-ec_before:+.4f}")
            st.caption(cal.get("interpretation", ""))

        # CLV
        if clv.get("avg_model_vs_market") is not None:
            st.divider()
            st.subheader("Closing Line Value (CLV)")
            col1, col2, col3 = st.columns(3)
            col1.metric("Model vs Market", f"{clv['avg_model_vs_market']:+.1%}")
            col2.metric("% Positive Edge", f"{clv['pct_positive_edge']:.1%}")
            col3.metric("Interpretation", clv.get("edge_category", "—"))
            st.info(clv.get("interpretation", ""))

        # ROI table
        roi_data = bt.get("roi_by_edge_threshold", [])
        if roi_data:
            st.divider()
            st.subheader("Simulated ROI by Edge Threshold")
            roi_df = pd.DataFrame(roi_data)
            roi_df["edge_threshold"] = roi_df["edge_threshold"].map(lambda x: f"{x:.0%}")
            roi_df["roi"]            = roi_df["roi"].map(lambda x: f"{x:+.1%}")
            roi_df["hit_rate"]       = roi_df["hit_rate"].map(lambda x: f"{x:.1%}")
            st.dataframe(
                roi_df[["edge_threshold", "n_bets", "hit_rate", "roi", "avg_odds"]].rename(columns={
                    "edge_threshold": "Min Edge", "n_bets": "# Bets",
                    "hit_rate": "Hit Rate", "roi": "ROI", "avg_odds": "Avg Odds",
                }),
                use_container_width=True,
                hide_index=True,
            )
    elif bt:
        st.warning(f"Backtest status: {bt.get('status')}. More historical data needed.")
    else:
        st.info("Click **Run Backtest** (in the Model Actions expander in the sidebar) to see model performance metrics.")

    # Training metadata
    if tr:
        st.divider()
        st.subheader("Last Training Run")
        for model_name, result in tr.items():
            if isinstance(result, dict):
                with st.expander(f"Model: {model_name}"):
                    col1, col2 = st.columns(2)
                    col1.metric("Brier (Train)", f"{result.get('brier_score_train', 0):.4f}")
                    col2.metric("Brier (Val)", f"{result.get('brier_score_val', 0):.4f}")
                    if result.get("n_train_samples"):
                        st.caption(f"Trained on {result['n_train_samples']} samples · Validated on {result.get('n_val_samples', 0)}")


# ── Tab 7: System ─────────────────────────────────────────────────────────
with tab_system:
    st.markdown('<div class="section-header">⚙ System Status & Data Sources</div>', unsafe_allow_html=True)

    # ── Data source transparency ──────────────────────────────────────────
    st.subheader("API Data Sources")

    def _status_dot(ok: bool) -> str:
        return "🟢" if ok else "🔴"

    # AFL Data Sports Group
    with st.expander(f"{_status_dot(settings.is_afl_data_configured)} AFL Data Sports Group API (Primary)", expanded=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Status", "Configured" if settings.is_afl_data_configured else "Not configured")
        col2.metric("Mode", settings.effective_data_mode.title())
        col3.metric("Call Rate", "Unlimited")
        if settings.is_afl_data_configured:
            key_preview = settings.afl_data_authkey[:6] + "..." if settings.afl_data_authkey else ""
            st.caption(
                f"Auth key: {key_preview} · "
                f"Competition ID: {settings.afl_data_competition_id} · "
                f"Base URL: {settings.afl_data_base_url}"
            )
            st.success("AFL Advanced Pack active: fixtures, live scores, player stats, box scores, advanced stats")
        else:
            st.error("AFL_DATA_AUTHKEY not set in .env")

    # Odds API
    with st.expander(f"{_status_dot(settings.is_odds_api_configured)} The Odds API (Real bookmaker odds)"):
        col1, col2 = st.columns(2)
        col1.metric("Status", "Configured" if settings.is_odds_api_configured else "Not configured")
        col2.metric("Sport", settings.odds_api_sport)
        if settings.is_odds_api_configured:
            col3, col4 = st.columns(2)
            col3.metric("Cache TTL", "15 min")
            col4.metric("Region", "AU")
            # Try to get quota from loader's odds provider
            try:
                ldr = get_loader()
                if hasattr(ldr.odds, "quota_status"):
                    qs = ldr.odds.quota_status
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Requests Remaining", qs.get("requests_remaining", "—"))
                    c2.metric("Requests Used", qs.get("requests_used", "—"))
                    age = qs.get("cache_age_seconds")
                    c3.metric("Cache Age", f"{age}s" if age is not None else "Not fetched")
            except Exception:
                pass
            bm = settings.odds_api_bookmakers
            st.caption(f"Bookmakers: {bm}")
        else:
            st.warning("ODDS_API_KEY not set — using synthetic odds estimates only")

    # Edge Intelligence
    with st.expander(f"{_status_dot(settings.enable_scraping)} Edge Intelligence (Web scraping)"):
        col1, col2 = st.columns(2)
        col1.metric("Status", "Enabled" if settings.enable_scraping else "Disabled")
        col2.metric("Cache TTL", f"{settings.scrape_cache_ttl_minutes} min")
        if settings.enable_scraping:
            n_signals = len(st.session_state.get("edge_signals", []))
            edge_ts = st.session_state.get("edge_signals_ts", "Not fetched")
            c1, c2 = st.columns(2)
            c1.metric("Cached Signals", n_signals)
            c2.metric("Last Fetched", edge_ts)
            st.caption("Sources: AFL.com.au RSS, Fox Sports RSS, club news pages")
        else:
            st.info("Set ENABLE_SCRAPING=true in .env to activate edge intelligence.")

    # No secondary data source — AFL Data Sports Group API is the sole AFL provider

    st.divider()

    # ── Environment ───────────────────────────────────────────────────────
    st.subheader("Environment")
    st.code(f"""
DATA_MODE={settings.data_mode}
EFFECTIVE_MODE={settings.effective_data_mode}
AFL_DATA_CONFIGURED={settings.is_afl_data_configured}
ODDS_API_CONFIGURED={settings.is_odds_api_configured}
EDGE_SCRAPING_ENABLED={settings.enable_scraping}
ARTIFACTS_DIR={settings.artifacts_dir}
    """, language="ini")

    st.divider()
    st.subheader("Last Pipeline Run")
    if pr:
        st.json({
            "run_id": pr.get("run_id"),
            "timestamp": pr.get("timestamp"),
            "data_source": pr.get("data_source"),
            "n_fixtures": pr.get("n_fixtures"),
            "n_candidate_legs": pr.get("n_candidate_legs"),
            "elapsed_seconds": pr.get("elapsed_seconds"),
        })
    else:
        st.caption("No pipeline run yet — click Run Pipeline in the sidebar.")

    st.divider()
    st.subheader("Continuous Mode")
    st.markdown("""
**Auto Mode** (this UI): Toggle **Auto** in the sidebar to auto-refresh every N minutes.

**Background loop** (terminal):
```bash
# Run once
python scripts/run_loop.py --once

# Continuous loop every 30 minutes
python scripts/run_loop.py --loop --interval 30

# Continuous with forced retrain each cycle
python scripts/run_loop.py --loop --interval 60 --retrain
```
""")


# ── Auto-refresh logic ────────────────────────────────────────────────────
if st.session_state.get("auto_mode"):
    interval_s = st.session_state.get("auto_interval", 15) * 60
    last_ts = st.session_state.get("_last_auto_ts", 0)
    now = time.time()
    if now - last_ts >= interval_s:
        st.session_state["_last_auto_ts"] = now
        try:
            run_pipeline()
        except Exception:
            pass
        st.rerun()


# ── Footer ────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "AFL Prediction Dashboard · "
    "All probabilities are calibrated model estimates, not financial advice. "
    "This system does not place bets."
)
