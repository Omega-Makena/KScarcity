"""Live policy impact visualization using appended synthetic criticality data."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:  # pragma: no cover
    HAS_PANDAS = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    HAS_PLOTLY = False

PROJECT_ROOT = Path(__file__).resolve().parents[5]
LIVE_DATA_PATH = PROJECT_ROOT / "data" / "synthetic_kenya_policy" / "tweets.csv"

LEVER_EFFECTS = {
    "targeted_subsidy": lambda t: -0.22 * (1.0 - np.exp(-4.0 * t)),
    "tax_relief": lambda t: -0.18 * (1.0 - np.exp(-3.0 * t)),
    "fuel_stabilization": lambda t: -0.12 * (1.0 - np.exp(-5.0 * t)),
    "rate_hike": lambda t: (0.06 * np.exp(-6.0 * t)) - (0.10 * (1.0 - np.exp(-2.0 * t))),
    "public_comms": lambda t: -0.07 * (1.0 - np.exp(-8.0 * t)),
    "security_deescalation": lambda t: -0.10 * (1.0 - np.exp(-6.0 * t)),
}

WINDOW_MAP = {
    "24h": timedelta(hours=24),
    "72h": timedelta(hours=72),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "All": None,
}


@st.cache_data(ttl=45, show_spinner=False)
def _load_live_policy_frame(refresh_nonce: int) -> Tuple["pd.DataFrame", Dict[str, object]]:
    del refresh_nonce
    if not HAS_PANDAS or not LIVE_DATA_PATH.exists():
        return pd.DataFrame() if HAS_PANDAS else None, {}

    frame = pd.read_csv(LIVE_DATA_PATH)
    if frame.empty:
        return frame, {}

    frame["timestamp"] = pd.to_datetime(frame.get("timestamp"), errors="coerce", utc=True)
    frame = frame.dropna(subset=["timestamp"]).copy()
    if frame.empty:
        return frame, {}

    for col in ["threat_score", "escalation_score", "coordination_score", "urgency_rate", "imperative_rate", "policy_severity"]:
        if col not in frame.columns:
            frame[col] = 0.0
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

    frame["location_county"] = frame.get("location_county", "Unknown").fillna("Unknown")
    frame["sector"] = frame.get("topic_cluster", "general").fillna("general")
    frame["policy_event_id"] = frame.get("policy_event_id", "observed").fillna("observed")

    frame["criticality"] = (
        0.45 * frame["threat_score"]
        + 0.25 * frame["escalation_score"]
        + 0.15 * frame["coordination_score"]
        + 0.10 * frame["urgency_rate"]
        + 0.05 * frame["policy_severity"]
    ).clip(0.0, 1.0)

    latest_ts = frame["timestamp"].max()
    latest_cutoff = latest_ts - pd.Timedelta(hours=24)
    frame["is_latest_batch"] = frame["timestamp"] >= latest_cutoff

    file_mtime = datetime.utcfromtimestamp(LIVE_DATA_PATH.stat().st_mtime)
    freshness = {
        "last_ingestion_time": latest_ts.to_pydatetime(),
        "file_mtime": file_mtime,
        "row_count": int(len(frame)),
        "latest_batch_rows": int(frame["is_latest_batch"].sum()),
        "within_24h": bool((datetime.utcnow() - latest_ts.to_pydatetime().replace(tzinfo=None)) <= timedelta(hours=24)),
    }
    return frame, freshness


def _scenario_options(frame: "pd.DataFrame") -> List[str]:
    options = ["Latest Batch (Live)", "All Observed"]
    if "policy_event_id" in frame.columns:
        vals = [str(v) for v in frame["policy_event_id"].dropna().unique().tolist() if str(v).strip() and str(v) != "observed"]
        for val in sorted(vals):
            options.append(f"Observed: {val}")

    for run in st.session_state.get("sim_compare_history", []):
        label = str(run.get("label", "")).strip()
        if label:
            options.append(f"Simulation: {label}")

    if st.session_state.get("whatif_trajectory"):
        options.append("Simulation: What-If Latest")

    # Preserve order while deduping.
    deduped = []
    for opt in options:
        if opt not in deduped:
            deduped.append(opt)
    return deduped


def _scenario_modifier(selected_run: str, n_points: int) -> np.ndarray:
    if n_points <= 0:
        return np.array([])

    if not selected_run.startswith("Simulation:"):
        return np.ones(n_points, dtype=float)

    label = selected_run.replace("Simulation:", "", 1).strip()
    trajectory = None

    if label == "What-If Latest":
        trajectory = st.session_state.get("whatif_trajectory")
    else:
        for run in reversed(st.session_state.get("sim_compare_history", [])):
            if str(run.get("label", "")).strip() == label:
                trajectory = run.get("trajectory")
                break

    if not trajectory or len(trajectory) < 2:
        return np.ones(n_points, dtype=float)

    inflation = np.array([float(f.get("outcomes", {}).get("inflation", 0.0)) for f in trajectory], dtype=float)
    unemployment = np.array([float(f.get("outcomes", {}).get("unemployment", 0.0)) for f in trajectory], dtype=float)
    pressure = inflation + unemployment
    if np.allclose(pressure.std(), 0.0):
        return np.ones(n_points, dtype=float)

    z = (pressure - pressure.mean()) / (pressure.std() + 1e-8)
    scaled = np.clip(1.0 + 0.08 * z, 0.80, 1.20)

    x_old = np.linspace(0.0, 1.0, len(scaled))
    x_new = np.linspace(0.0, 1.0, n_points)
    return np.interp(x_new, x_old, scaled)


def _lever_multiplier(selected_levers: List[str], intensity: float, n_points: int) -> np.ndarray:
    if n_points <= 0:
        return np.array([])
    t = np.linspace(0.0, 1.0, n_points)
    modifier = np.zeros(n_points, dtype=float)
    for lever in selected_levers:
        fn = LEVER_EFFECTS.get(lever)
        if fn is None:
            continue
        modifier += float(intensity) * fn(t)
    return np.clip(1.0 + modifier, 0.70, 1.20)


def render_live_policy_impact(theme) -> None:
    """Render live baseline vs counterfactual trajectories with freshness controls."""
    if not HAS_PANDAS:
        st.info("Policy impact visualization needs pandas in this environment.")
        return

    nonce = int(st.session_state.get("policy_live_refresh_nonce", 0))
    frame, freshness = _load_live_policy_frame(nonce)

    if frame is None or frame.empty:
        st.info("No live policy criticality data found. Run the 24h synthetic generator first.")
        return

    with st.container(border=True):
        top_l, top_r = st.columns([4, 1])
        with top_l:
            st.markdown("#### Policy Impact — Live Criticality")
            st.caption("Baseline vs counterfactual trajectories from the latest appended synthetic criticality stream.")
        with top_r:
            if st.button("Refresh Live Data", key="policy_live_refresh"):
                st.session_state["policy_live_refresh_nonce"] = nonce + 1
                st.rerun()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Last Ingestion", str(freshness.get("last_ingestion_time", "n/a"))[:19])
        m2.metric("Rows", f"{freshness.get('row_count', 0):,}")
        m3.metric("Latest Batch (24h)", f"{freshness.get('latest_batch_rows', 0):,}")
        freshness_label = "Fresh (<=24h)" if freshness.get("within_24h") else "Stale (>24h)"
        m4.metric("Data Freshness", freshness_label)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            window = st.selectbox("Time Window", list(WINDOW_MAP.keys()), index=0, key="policy_live_window")

        county_options = sorted([str(c) for c in frame["location_county"].dropna().unique().tolist()])
        default_counties = county_options[: min(6, len(county_options))] if county_options else []
        with c2:
            counties = st.multiselect("Region/County", county_options, default=default_counties, key="policy_live_county")

        sector_options = sorted([str(s) for s in frame["sector"].dropna().unique().tolist()])
        with c3:
            sectors = st.multiselect("Sector", sector_options, default=sector_options[: min(5, len(sector_options))], key="policy_live_sector")

        with c4:
            selected_run = st.selectbox("Scenario Run", _scenario_options(frame), key="policy_live_run")

        lever_labels = list(LEVER_EFFECTS.keys())
        with c5:
            selected_levers = st.multiselect(
                "Policy Lever(s)",
                lever_labels,
                default=["targeted_subsidy", "fuel_stabilization"],
                key="policy_live_levers",
            )

        intensity = st.slider("Lever Intensity", min_value=0.0, max_value=1.0, value=0.45, step=0.05, key="policy_live_intensity")

        filtered = frame.copy()
        if counties:
            filtered = filtered[filtered["location_county"].isin(counties)]
        if sectors:
            filtered = filtered[filtered["sector"].isin(sectors)]

        now = datetime.utcnow()
        window_delta = WINDOW_MAP[window]
        if window_delta is not None:
            cutoff = now - window_delta
            filtered = filtered[filtered["timestamp"] >= cutoff]

        if selected_run == "Latest Batch (Live)":
            filtered = filtered[filtered["is_latest_batch"] == True]
        elif selected_run.startswith("Observed: "):
            run_id = selected_run.replace("Observed:", "", 1).strip()
            filtered = filtered[filtered["policy_event_id"].astype(str) == run_id]

        if filtered.empty:
            st.warning("No rows matched your filters.")
            return

        freq = "2H" if window in {"24h", "72h"} else "D"
        ts = (
            filtered.set_index("timestamp")["criticality"]
            .resample(freq)
            .mean()
            .dropna()
        )

        if len(ts) < 2:
            # Fallback for sparse windows.
            ts = filtered.sort_values("timestamp").set_index("timestamp")["criticality"]

        baseline = ts.to_numpy(dtype=float)
        timestamps = ts.index.to_list()

        lever_mult = _lever_multiplier(selected_levers, intensity, len(baseline))
        scenario_mult = _scenario_modifier(selected_run, len(baseline))
        counterfactual = np.clip(baseline * lever_mult * scenario_mult, 0.0, 1.0)

        delta_mean = float(counterfactual.mean() - baseline.mean())
        delta_peak = float(counterfactual.max() - baseline.max())
        k1, k2, k3 = st.columns(3)
        k1.metric("Avg Baseline", f"{baseline.mean():.1%}")
        k2.metric("Avg Counterfactual", f"{counterfactual.mean():.1%}", delta=f"{delta_mean:+.1%}")
        k3.metric("Peak Delta", f"{delta_peak:+.1%}")

        if HAS_PLOTLY:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=baseline,
                    mode="lines+markers",
                    name="Baseline",
                    line=dict(color=theme.accent_warning, width=2.5),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=counterfactual,
                    mode="lines+markers",
                    name="Counterfactual",
                    line=dict(color=theme.accent_success, width=2.5),
                    fill="tonexty",
                    fillcolor="rgba(0, 255, 136, 0.10)",
                )
            )

            latest_start = filtered[filtered["is_latest_batch"]]["timestamp"].min()
            if latest_start is not pd.NaT:
                fig.add_vrect(
                    x0=latest_start,
                    x1=max(timestamps),
                    line_width=0,
                    fillcolor="rgba(0, 243, 255, 0.06)",
                    annotation_text="Latest 24h batch",
                    annotation_position="top left",
                )

            fig.update_layout(
                height=380,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=theme.text_primary),
                xaxis_title="Time",
                yaxis_title="Criticality (0-1)",
                yaxis=dict(range=[0, 1]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart(
                {
                    "baseline": baseline,
                    "counterfactual": counterfactual,
                }
            )
