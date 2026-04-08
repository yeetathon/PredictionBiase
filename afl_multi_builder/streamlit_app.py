"""
AFL Multi Builder - Streamlit UI
Alternative UI using Streamlit for interactive exploration.
Run with: streamlit run streamlit_app.py
"""
import sys
import json
import time
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

st.set_page_config(
    page_title="AFL Multi Builder",
    page_icon="🏉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports (heavy) ──────────────────────────────────────────────────
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


# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏉 AFL Multi Builder")
    st.caption("Calibrated predictions · Edge detection · Correlation-aware multis")
    st.divider()

    st.subheader("Quick Setup")
    if st.button("▶ Run Full Pipeline", use_container_width=True, type="primary"):
        with st.spinner("Running prediction pipeline..."):
            try:
                import os; os.makedirs("data/models", exist_ok=True)
                pipeline = get_pipeline()
                result = pipeline.run()
                st.session_state["pipeline_result"] = result
                st.success(f"✓ {result['n_candidate_legs']} legs generated")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("⚙ Train Models", use_container_width=True):
        with st.spinner("Training models..."):
            try:
                svc = get_training_service()
                result = svc.run_all()
                st.session_state["training_result"] = result
                st.success("Models trained successfully")
            except Exception as e:
                st.error(f"Training error: {e}")

    if st.button("📊 Run Backtest", use_container_width=True):
        with st.spinner("Running walk-forward backtest..."):
            try:
                bt = get_backtest_service()
                result = bt.run()
                st.session_state["backtest_result"] = result
                if result.get("status") == "success":
                    m = result["overall_metrics"]
                    st.success(f"Brier: {m['brier_score']:.4f} · Acc: {m['accuracy']:.1%}")
                else:
                    st.warning(f"Backtest: {result.get('status')}")
            except Exception as e:
                st.error(f"Backtest error: {e}")

    st.divider()
    st.subheader("Filters")
    min_ev = st.slider("Min EV (%)", -10, 30, 2) / 100
    max_corr = st.slider("Max Correlation", 0.1, 1.0, 0.7)
    min_legs = st.selectbox("Min Legs", [2, 3, 4], index=0)
    max_legs = st.selectbox("Max Legs", [2, 3, 4], index=2)

    st.divider()
    loader = get_loader()
    fixtures = loader.load_fixtures_df()
    if not fixtures.empty:
        st.metric("Total Fixtures", len(fixtures))
        completed = fixtures[fixtures["status"] == "completed"]
        upcoming = fixtures[fixtures["status"] == "upcoming"]
        st.metric("Completed", len(completed))
        st.metric("Upcoming", len(upcoming))


# ── Main content ──────────────────────────────────────────────────────────
st.title("AFL Prediction & Multi Builder")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Value Legs", "🛡 Safe Legs", "📈 Value Multis", "🎲 Same-Game", "📊 Backtest"
])

def format_ev(ev):
    s = f"{ev*100:+.1f}%"
    return s

def legs_to_df(legs):
    if not legs:
        return pd.DataFrame()
    rows = []
    for l in legs:
        rows.append({
            "Selection": l["selection"].replace("_", " ").title(),
            "Market": l["market_type"].replace("_", " ").title(),
            "Fixture": l["fixture_id"],
            "Odds": f"${l['decimal_odds']:.2f}",
            "Model Prob": f"{l['model_probability']:.1%}",
            "EV": format_ev(l["ev"]),
            "Confidence": f"{l['confidence_score']:.0f}",
            "Explanation": l["explanation"],
        })
    return pd.DataFrame(rows)

def multis_to_df(multis):
    if not multis:
        return pd.DataFrame()
    rows = []
    for m in multis:
        rows.append({
            "Type": m["multi_type"].replace("_", " ").title(),
            "Legs": m["n_legs"],
            "Comb. Odds": f"${m['combined_odds']:.2f}",
            "Hit Prob": f"{m['adjusted_probability']:.1%}",
            "EV": format_ev(m["ev"]),
            "Correlation": m["correlation_label"].title(),
            "Risk": f"{m['risk_score']:.0f}/100",
        })
    return pd.DataFrame(rows)

pr = st.session_state.get("pipeline_result")

with tab1:
    st.subheader("Top Value Legs")
    st.caption("Ranked by Expected Value (EV). Positive EV legs offer statistical edge over market.")
    if pr:
        legs = [l for l in pr.get("value_legs", []) if l["ev"] >= min_ev]
        df = legs_to_df(legs)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
            # Probability chart
            st.subheader("Model vs Market Probability")
            chart_data = pd.DataFrame([{
                "Leg": l["selection"][:25],
                "Model": l["model_probability"],
            } for l in legs[:10]])
            if not chart_data.empty:
                st.bar_chart(chart_data.set_index("Leg"))
        else:
            st.info("Run the pipeline to see value legs. No legs pass the EV filter yet.")
    else:
        st.info("Click 'Run Full Pipeline' in the sidebar to generate predictions.")

with tab2:
    st.subheader("Safe Legs")
    st.caption("Highest probability legs with minimum edge. Suitable for lower-risk multis.")
    if pr:
        legs = [l for l in pr.get("safe_legs", []) if l["ev"] >= min_ev]
        df = legs_to_df(legs)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No legs pass current filters.")
    else:
        st.info("Run the pipeline first.")

with tab3:
    st.subheader("Value Multis")
    st.caption("Multi combinations ranked by EV. Correlation penalty applied — not naive independent multiplication.")
    if pr:
        multis = [m for m in pr.get("value_multis", [])
                  if m.get("ev", 0) >= min_ev and m.get("correlation_score", 1) <= max_corr
                  and min_legs <= m.get("n_legs", 2) <= max_legs]
        df = multis_to_df(multis)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Show detail for selected multi
            if len(multis) > 0:
                st.subheader("Multi Details")
                selected_idx = st.selectbox("Select multi", range(len(multis)),
                    format_func=lambda i: f"Multi {i+1}: {multis[i]['n_legs']}-leg @ ${multis[i]['combined_odds']:.2f}")
                m = multis[selected_idx]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Combined Odds", f"${m['combined_odds']:.2f}")
                col2.metric("Hit Probability", f"{m['adjusted_probability']:.1%}")
                col3.metric("EV", format_ev(m["ev"]))
                col4.metric("Correlation", m["correlation_label"].title())
                st.info(m["explanation"])
        else:
            st.info("No multis pass current filters.")
    else:
        st.info("Run the pipeline first.")

with tab4:
    st.subheader("Same-Game Multis")
    st.caption("All legs from a single match. Higher correlation but more exciting.")
    if pr:
        sgm = [m for m in pr.get("same_game_multis", [])
               if m.get("ev", 0) >= min_ev]
        df = multis_to_df(sgm)
        if not df.empty:
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No same-game multis available.")
    else:
        st.info("Run the pipeline first.")

with tab5:
    st.subheader("Walk-Forward Backtest Results")
    bt = st.session_state.get("backtest_result")
    if bt and bt.get("status") == "success":
        metrics = bt["overall_metrics"]
        cal = bt.get("calibration_report", {})
        clv = bt.get("clv_analysis", {})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Brier Score", f"{metrics['brier_score']:.4f}",
                    help="0=perfect, 0.25=no-skill baseline. Lower is better.")
        col2.metric("Log Loss", f"{metrics['log_loss']:.4f}")
        col3.metric("Accuracy", f"{metrics['accuracy']:.1%}")
        col4.metric("Calibration Improvement",
                    f"{(cal.get('brier_improvement', 0)*10000):.0f} pts")

        if clv.get("avg_model_vs_market") is not None:
            st.subheader("Closing Line Value Analysis")
            col1, col2 = st.columns(2)
            col1.metric("Avg Model vs Market",
                        f"{clv['avg_model_vs_market']:+.1%}")
            col2.metric("% Positive Edge",
                        f"{clv['pct_positive_edge']:.1%}")
            st.info(clv.get("interpretation", ""))

        roi_data = bt.get("roi_by_edge_threshold", [])
        if roi_data:
            st.subheader("Simulated ROI by Edge Threshold")
            roi_df = pd.DataFrame(roi_data)
            roi_df["edge_threshold"] = roi_df["edge_threshold"].map(lambda x: f"{x:.0%}")
            roi_df["roi"] = roi_df["roi"].map(lambda x: f"{x:+.1%}")
            roi_df["hit_rate"] = roi_df["hit_rate"].map(lambda x: f"{x:.1%}")
            st.dataframe(
                roi_df[["edge_threshold", "n_bets", "hit_rate", "roi", "avg_odds"]].rename(columns={
                    "edge_threshold": "Min Edge", "n_bets": "Bets",
                    "hit_rate": "Hit Rate", "roi": "ROI", "avg_odds": "Avg Odds"
                }),
                use_container_width=True,
                hide_index=True
            )
    elif bt:
        st.warning(f"Backtest status: {bt.get('status')}. More historical data needed for full walk-forward analysis.")
    else:
        st.info("Click 'Run Backtest' in the sidebar.")

# ── Footer ────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "AFL Multi Builder v1.0 · "
    "All probabilities are calibrated model estimates, not guarantees. "
    "This system does not place bets."
)
