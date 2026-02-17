"""
K-SHIELD: Simulations — Economic Scenario Engine

Matches the card layout pattern of Causal and Terrain:
    section-header -> nav radio (Workspace / Guide) -> data source radio
    -> model configuration expander -> analysis tabs

Integrates:
    - kshiked.simulation.kenya_calibration  (data-driven SFC params)
    - kshiked.simulation.scenario_templates  (9 named scenarios + 8 policies)
    - scarcity.simulation.sfc               (SFC engine)
    - Shared K-SHIELD Dataset infrastructure (cross-card data sharing)
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

logger = logging.getLogger("sentinel.kshield.simulation")

try:
    import pandas as pd
    import numpy as np
    HAS_DATA_STACK = True
except ImportError:
    pd = None
    np = None
    HAS_DATA_STACK = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    go = None
    HAS_PLOTLY = False

try:
    import streamlit.components.v1 as components
    HAS_COMPONENTS = True
except ImportError:
    HAS_COMPONENTS = False

# ── Shared dataset keys (same as causal.py / terrain.py) ─────────────────────
SHARED_DF_KEY = "kshield_shared_df"
SHARED_SOURCE_KEY = "kshield_shared_source"
SHARED_OWNER_KEY = "kshield_shared_owner"


def _set_shared_dataset(df: "pd.DataFrame", source: str, owner: str) -> None:
    if not HAS_DATA_STACK or df is None or df.empty:
        return
    st.session_state[SHARED_DF_KEY] = df.copy(deep=True)
    st.session_state[SHARED_SOURCE_KEY] = source
    st.session_state[SHARED_OWNER_KEY] = owner


def _get_shared_dataset() -> Tuple[Optional["pd.DataFrame"], str]:
    if not HAS_DATA_STACK:
        return None, ""
    candidate = st.session_state.get(SHARED_DF_KEY)
    if isinstance(candidate, pd.DataFrame) and not candidate.empty:
        source = str(st.session_state.get(SHARED_SOURCE_KEY, "Unknown source"))
        owner = str(st.session_state.get(SHARED_OWNER_KEY, "Unknown card"))
        return candidate, f"{source} via {owner}"
    return None, ""


# ── Data loading (reuse same World Bank CSV as causal/terrain) ────────────────

@st.cache_data(ttl=3600, show_spinner="Loading World Bank data ...")
def _load_world_bank_data() -> "pd.DataFrame":
    csv_path = _find_csv()
    if csv_path is None:
        return pd.DataFrame()
    raw = pd.read_csv(csv_path, skiprows=4, encoding="utf-8-sig")
    return _pivot_world_bank(raw)


def _pivot_world_bank(raw: "pd.DataFrame") -> "pd.DataFrame":
    if "Indicator Name" not in raw.columns:
        return pd.DataFrame()
    year_cols = [c for c in raw.columns if c.strip().isdigit()
                 and 1900 <= int(c.strip()) <= 2100]
    if not year_cols:
        return pd.DataFrame()
    melted = raw.melt(
        id_vars=["Indicator Name"], value_vars=year_cols,
        var_name="Year", value_name="Value",
    )
    melted["Year"] = melted["Year"].astype(int)
    melted["Value"] = pd.to_numeric(melted["Value"], errors="coerce")
    pivoted = melted.pivot_table(index="Year", columns="Indicator Name", values="Value")
    pivoted = pivoted.sort_index()
    good_cols = pivoted.columns[pivoted.notna().sum() >= 15]
    pivoted = pivoted[good_cols]
    pivoted = pivoted.interpolate(method="linear", limit=3)
    return pivoted


def _find_csv() -> Optional[str]:
    candidates = [
        Path(__file__).resolve().parents[3] / "data" / "simulation" / "API_KEN_DS2_en_csv_v2_14659.csv",
        Path(os.getcwd()) / "data" / "simulation" / "API_KEN_DS2_en_csv_v2_14659.csv",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _validate_and_load_upload(uploaded_file) -> Tuple[Optional["pd.DataFrame"], Optional[str]]:
    try:
        raw = pd.read_csv(uploaded_file, encoding="utf-8-sig")
    except Exception as e:
        return None, f"Could not parse CSV: {e}"
    if "Indicator Name" in raw.columns:
        year_cols = [c for c in raw.columns if c.strip().isdigit()
                     and 1900 <= int(c.strip()) <= 2100]
        if year_cols:
            df = _pivot_world_bank(raw)
            if df.empty:
                return None, "World Bank format detected but no usable indicator data."
            return df, None
    if len(raw.columns) < 2:
        return None, "CSV must have at least 2 columns."
    df = raw.copy()
    idx_col = df.columns[0]
    try:
        idx_vals = pd.to_numeric(df[idx_col], errors="coerce")
        if idx_vals.notna().sum() > 0.5 * len(idx_vals):
            df[idx_col] = idx_vals
            df = df.set_index(idx_col)
        else:
            try:
                df[idx_col] = pd.to_datetime(df[idx_col])
                df = df.set_index(idx_col)
                df.index = df.index.year
            except Exception:
                return None, f"First column '{idx_col}' must be numeric years or dates."
    except Exception:
        return None, f"Could not parse index column '{idx_col}'."
    df = df.sort_index()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return None, "Need at least 2 numeric data columns."
    df = df[numeric_cols]
    good_cols = df.columns[df.notna().sum() >= 10]
    if len(good_cols) < 2:
        return None, "Not enough data. Each column needs at least 10 non-empty values."
    df = df[good_cols].interpolate(method="linear", limit=3)
    return df, None


# ── Consistent Plotly layout ─────────────────────────────────────────────────

PALETTE = [
    "#00ff88", "#00aaff", "#f5d547", "#ff3366", "#8b5cf6",
    "#14b8a6", "#f97316", "#ec4899", "#a3e635", "#06b6d4",
]


def _base_layout(theme, height=400, **extra):
    layout = dict(
        height=height,
        margin=dict(l=40, r=20, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Mono, monospace", color=theme.text_muted, size=11),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor=theme.border_default,
                   tickfont=dict(color=theme.text_muted)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor=theme.border_default,
                   tickfont=dict(color=theme.text_muted)),
        legend=dict(font=dict(color=theme.text_muted, size=10)),
        hovermode="x unified",
    )
    layout.update(extra)
    return layout


# ── Dynamic dimension discovery ──────────────────────────────────────────────

def _discover_dimensions(trajectory):
    """
    Auto-discover ALL available dimension categories and keys from trajectory
    frames.  Returns {category_name: [sorted list of keys]}.
    Categories scanned: outcomes, channels, flows, sector_balances,
    policy_vector, shock_vector.
    """
    categories = {}
    vector_keys = [
        "outcomes", "channels", "flows", "sector_balances",
        "policy_vector", "shock_vector",
    ]
    for cat in vector_keys:
        keys = sorted({
            k for f in trajectory
            for k in (f.get(cat, {}) if isinstance(f.get(cat), dict) else {})
        })
        if keys:
            categories[cat] = keys
    return categories


def _dim_label(raw: str) -> str:
    """Convert 'gdp_growth' -> 'Gdp Growth' for display."""
    return raw.replace("_", " ").title()


def _extract_dim(dim_str: str, frame: dict) -> float:
    """Extract a value from a frame using 'category::key' notation."""
    cat, key = dim_str.split("::", 1)
    container = frame.get(cat, {})
    if isinstance(container, dict):
        return float(container.get(key, 0))
    return 0.0


def _flat_dim_options(dims: dict) -> list:
    """Flatten discovered dimensions into 'category::key' strings for selectors."""
    options = []
    for cat, keys in dims.items():
        for k in keys:
            options.append(f"{cat}::{k}")
    return options


# ═════════════════════════════════════════════════════════════════════════════
#  GUIDE & TUTORIAL
# ═════════════════════════════════════════════════════════════════════════════

def _render_simulation_guide(theme):
    st.markdown(
        '<div class="section-header">SIMULATION GUIDE &mdash; FULL TUTORIAL</div>',
        unsafe_allow_html=True,
    )
    nav_col, content_col = st.columns([1, 3])
    sections = [
        "1) What this card does",
        "2) Scenarios explained",
        "3) Policy templates",
        "4) How the SFC engine works",
        "5) Reading the results",
        "6) Sensitivity & heatmaps",
        "7) Comparing trajectories",
        "8) Connecting to Causal & Terrain",
        "9) Tips & common mistakes",
    ]
    with nav_col:
        st.markdown("**Guide Navigation**")
        section = st.radio("Jump to", sections, key="sim_guide_section",
                           label_visibility="collapsed")
    with content_col:
        if section == sections[0]:
            st.markdown("""
            This card runs **forward-looking economic simulations** using Kenya's
            calibrated Stock-Flow Consistent (SFC) model.

            You pick a shock scenario (e.g. oil crisis, drought), choose a policy
            response (CBK tightening, fiscal stimulus, etc.), select which outcome
            dimensions to watch, and run the simulation over 20-100 quarters.

            The engine is calibrated from the **same World Bank data** used by the
            Causal and Terrain cards, ensuring consistency across all K-SHIELD analyses.
            """)
        elif section == sections[1]:
            st.markdown("""
            **9 pre-built scenarios** cover Kenya's real risk landscape:

            | Category | Scenario | Key Shocks |
            |----------|----------|------------|
            | Supply | Oil Price Spike (+30%) | supply + FX |
            | Supply | Severe Drought (-20% Agri) | supply + demand |
            | Supply | Food Price Surge (+25%) | supply (ramped) |
            | External | Shilling Depreciation (-15%) | FX |
            | External | Global Recession | demand + FX |
            | External | Foreign Aid Cut (-30%) | fiscal |
            | Fiscal | Sovereign Debt Crisis | fiscal + FX |
            | Combined | Perfect Storm | supply + demand + FX |
            | Fiscal | Government Stimulus Boom | fiscal |

            Each includes a **context narrative** explaining real-world precedents.
            You can also build **custom scenarios** with your own shock magnitudes.
            """)
        elif section == sections[2]:
            st.markdown("""
            **8 policy templates** model real Government / CBK responses:

            - **Do Nothing** — let markets adjust
            - **CBK Tightening** — raise rates +2pp
            - **Aggressive Tightening** — major rate hike + CRR increase
            - **Fiscal Stimulus** — more spending + subsidies
            - **Austerity / IMF Package** — spending cuts + tax hikes
            - **Kenya 2016 Rate Cap** — interest rate cap at 11%
            - **Expansionary Mix** — lower rates + targeted subsidies
            - **Price Controls** — cap fuel + food prices

            Each template pre-fills the monetary and fiscal instrument sliders.
            You can override any slider after selecting a template.
            """)
        elif section == sections[3]:
            st.markdown("""
            The **SFC (Stock-Flow Consistent)** engine models 4 sectors:
            Households, Firms, Government, and Banking.

            Key equations:
            - **Phillips Curve** (New Keynesian): inflation responds to output gap
              with anchoring to prevent runaway spirals
            - **Taylor Rule**: interest rate responds to inflation and output gaps
            - **Fiscal block**: taxes, spending, subsidies, deficit, debt accumulation
            - **Household block**: consumption, savings, welfare
            - **Financial stability**: credit growth, leverage, banking health score

            All parameters are **calibrated from World Bank data** using
            `kenya_calibration.py` — no hardcoded magic numbers.
            """)
        elif section == sections[4]:
            st.markdown("""
            After running, you'll see:

            1. **Impact delta cards** — final value + change for each watched dimension
            2. **Trajectory chart** — time-series of all selected dimensions
            3. **Shock onset marker** — vertical line showing when the shock hits

            Interpretation tips:
            - Green arrows (up for growth, down for inflation) = good outcomes
            - Red arrows = concerning movements
            - The shock onset marker helps you see lag effects
            """)
        elif section == sections[5]:
            st.markdown("""
            The **Sensitivity tab** shows a policy-outcome correlation heatmap:
            - Blue = policy instrument correlated with positive outcome
            - Red = correlated with negative outcome
            - Near zero = low sensitivity

            This helps identify which policy levers have the strongest effect
            on which outcomes, based on the simulation's trajectory data.
            """)
        elif section == sections[6]:
            st.markdown("""
            The **Compare tab** lets you run multiple scenarios back-to-back
            and overlay their trajectories on a single chart. This is useful for:

            - Comparing "do nothing" vs active policy response
            - Testing mild vs aggressive policy actions
            - Checking whether combined shocks are worse than sum of parts
            """)
        elif section == sections[7]:
            st.markdown("""
            All three K-SHIELD cards share the same data infrastructure:

            - **Causal** discovers relationships between indicators
            - **Terrain** maps the policy landscape and stability regions
            - **Simulation** runs forward scenarios with calibrated models

            When you load World Bank data in any card, it's shared via the
            "Shared K-SHIELD Dataset" option in the other cards. The Simulation
            card uses the same World Bank CSV for calibration, ensuring that
            discovered causal links and terrain maps are consistent with
            the simulation parameters.
            """)
        elif section == sections[8]:
            st.markdown("""
            **Tips:**
            - Start with a named scenario before building custom ones
            - Use "Do Nothing" policy first to see the raw shock effect
            - Then compare with an active policy to measure the difference
            - Watch at least 5 dimensions for a holistic view
            - 50 quarters (12.5 years) is usually enough to see full dynamics

            **Common mistakes:**
            - Running too few quarters (< 20) — dynamics haven't played out
            - Ignoring the shock onset — effects lag by 2-4 quarters
            - Comparing scenarios with different step counts
            - Not checking calibration confidence (shown in results header)
            """)


# ═════════════════════════════════════════════════════════════════════════════
#  DATA PROFILE (matches terrain pattern)
# ═════════════════════════════════════════════════════════════════════════════

def _render_data_profile(df: "pd.DataFrame", theme):
    n_rows, n_cols = df.shape
    coverage = f"{df.index.min()}" + " - " + f"{df.index.max()}" if len(df) > 0 else "N/A"
    completeness = f"{df.notna().mean().mean():.0%}"

    st.markdown(f"""
    <div style="display: flex; gap: 2rem; padding: 0.6rem 0; font-size: 0.78rem;
                color: {theme.text_muted}; flex-wrap: wrap;">
        <span>Rows: <b style="color:{theme.text_primary}">{n_rows}</b></span>
        <span>Columns: <b style="color:{theme.text_primary}">{n_cols}</b></span>
        <span>Coverage: <b style="color:{theme.text_primary}">{coverage}</b></span>
        <span>Completeness: <b style="color:{theme.text_primary}">{completeness}</b></span>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SCENARIO CONFIGURATION PANEL
# ═════════════════════════════════════════════════════════════════════════════

def _render_scenario_config(theme, scenario_library, policy_templates,
                            get_scenario_by_id, build_custom_scenario,
                            outcome_dimensions, default_dimensions):
    """
    Model Configuration expander — matches terrain's expander pattern.
    Contains: scenario picker, policy builder, dimension selector, run params.
    Returns (scenario_obj, policy_overrides, selected_dims, steps).
    """
    with st.expander("Scenario Configuration", expanded=True):
        # Row 1: Scenario + Policy + Steps
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:600; "
                        f"font-size:0.8rem; margin-bottom:0.3rem;'>SCENARIO</div>",
                        unsafe_allow_html=True)
            scenario_options = {s.id: f"{s.name}  ({s.category})" for s in scenario_library}
            scenario_options["custom"] = "Custom Scenario"
            selected_id = st.selectbox(
                "Scenario", options=list(scenario_options.keys()),
                format_func=lambda x: scenario_options[x],
                key="sim_scenario_select", label_visibility="collapsed",
            )
            scenario = get_scenario_by_id(selected_id)
            st.session_state["_sim_scenario_id"] = selected_id
            st.session_state["_sim_scenario_obj"] = scenario

        with c2:
            st.markdown(f"<div style='color:{theme.accent_warning}; font-weight:600; "
                        f"font-size:0.8rem; margin-bottom:0.3rem;'>POLICY RESPONSE</div>",
                        unsafe_allow_html=True)
            policy_keys = list(policy_templates.keys())
            suggested = scenario.suggested_policy if scenario else None
            default_idx = 0
            if suggested:
                for i, pk in enumerate(policy_keys):
                    if policy_templates[pk].get("name") == suggested.get("name"):
                        default_idx = i
                        break
            policy_key = st.selectbox(
                "Policy", options=policy_keys,
                format_func=lambda k: policy_templates[k]["name"],
                index=default_idx, key="sim_policy_select", label_visibility="collapsed",
            )
            st.session_state["_sim_policy_key"] = policy_key

        with c3:
            st.markdown(f"<div style='color:{theme.accent_success}; font-weight:600; "
                        f"font-size:0.8rem; margin-bottom:0.3rem;'>SIMULATION</div>",
                        unsafe_allow_html=True)
            steps = st.slider("Quarters", 20, 100, 50, 5, key="sim_steps")

        # Context narrative
        if scenario and scenario.context:
            st.markdown(f"""
            <div style="background: rgba(0,243,255,0.05); border-left: 3px solid {theme.accent_primary};
                        padding: 0.6rem 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0;
                        font-size: 0.8rem; color: {theme.text_muted};">
                {scenario.context}
            </div>
            """, unsafe_allow_html=True)

        # Row 2: Custom shocks (if custom scenario)
        custom_shocks = {}
        if selected_id == "custom":
            st.markdown(f"<div style='color:{theme.text_muted}; font-weight:600; "
                        f"font-size:0.78rem; margin-top:0.5rem;'>CUSTOM SHOCK MAGNITUDES</div>",
                        unsafe_allow_html=True)
            cs1, cs2, cs3, cs4 = st.columns(4)
            with cs1:
                custom_shocks["demand_shock"] = st.slider(
                    "Demand", -0.20, 0.20, 0.0, 0.01, key="cust_demand",
                    help="Negative = contraction, Positive = boom")
            with cs2:
                custom_shocks["supply_shock"] = st.slider(
                    "Supply", -0.20, 0.20, 0.0, 0.01, key="cust_supply",
                    help="Negative = supply disruption")
            with cs3:
                custom_shocks["fiscal_shock"] = st.slider(
                    "Fiscal", -0.15, 0.15, 0.0, 0.01, key="cust_fiscal",
                    help="Positive = stimulus, Negative = austerity")
            with cs4:
                custom_shocks["fx_shock"] = st.slider(
                    "FX Pressure", -0.15, 0.15, 0.0, 0.01, key="cust_fx",
                    help="Positive = depreciation pressure")

            cs5, cs6 = st.columns(2)
            with cs5:
                shock_onset = st.number_input("Shock onset (quarter)", 1, 50, 5, key="cust_onset")
                st.session_state["_sim_onset"] = shock_onset
            with cs6:
                shock_shape = st.selectbox("Shock shape", ["step", "pulse", "ramp", "decay"],
                                           key="cust_shape")
                st.session_state["_sim_shape"] = shock_shape
            st.session_state["_sim_custom_shocks"] = custom_shocks

        # Row 3: Policy instrument overrides
        policy_overrides = {}
        policy_info = policy_templates.get(policy_key, {})
        instruments = policy_info.get("instruments", {})

        if policy_key != "do_nothing" and instruments:
            st.markdown(f"<div style='color:{theme.text_muted}; font-weight:600; "
                        f"font-size:0.78rem; margin-top:0.5rem;'>POLICY INSTRUMENTS</div>",
                        unsafe_allow_html=True)
            col_m, col_f = st.columns(2)
            with col_m:
                st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:600; "
                            f"font-size:0.75rem;'>MONETARY</div>", unsafe_allow_html=True)
                pol_rate = instruments.get("custom_rate")
                policy_overrides["custom_rate"] = st.slider(
                    "Policy Rate (%)", 1.0, 20.0,
                    float(pol_rate) * 100 if pol_rate else 7.0,
                    0.25, key="sim_pol_rate",
                ) / 100.0
                pol_crr = instruments.get("crr")
                policy_overrides["crr"] = st.slider(
                    "Cash Reserve Ratio (%)", 0.0, 15.0,
                    float(pol_crr) * 100 if pol_crr else 5.25,
                    0.25, key="sim_pol_crr",
                ) / 100.0
                rate_cap_on = st.checkbox("Interest Rate Cap",
                                          value="rate_cap" in instruments, key="sim_cap_on")
                if rate_cap_on:
                    pol_cap = instruments.get("rate_cap")
                    policy_overrides["rate_cap"] = st.slider(
                        "Rate Cap (%)", 5.0, 25.0,
                        float(pol_cap) * 100 if pol_cap else 11.0,
                        0.5, key="sim_cap_val",
                    ) / 100.0

            with col_f:
                st.markdown(f"<div style='color:{theme.accent_warning}; font-weight:600; "
                            f"font-size:0.75rem;'>FISCAL</div>", unsafe_allow_html=True)
                pol_tax = instruments.get("custom_tax_rate")
                policy_overrides["custom_tax_rate"] = st.slider(
                    "Tax Rate (%)", 5.0, 30.0,
                    float(pol_tax) * 100 if pol_tax else 15.6,
                    0.5, key="sim_pol_tax",
                ) / 100.0
                pol_spend = instruments.get("custom_spending_ratio")
                policy_overrides["custom_spending_ratio"] = st.slider(
                    "Govt Spending (% GDP)", 5.0, 30.0,
                    float(pol_spend) * 100 if pol_spend else 13.0,
                    0.5, key="sim_pol_spend",
                ) / 100.0
                pol_sub = instruments.get("subsidy_rate")
                policy_overrides["subsidy_rate"] = st.slider(
                    "Subsidies (% GDP)", 0.0, 10.0,
                    float(pol_sub) * 100 if pol_sub else 0.8,
                    0.1, key="sim_pol_subsidy",
                ) / 100.0
                if st.checkbox("Price Controls", value="price_controls" in instruments,
                               key="sim_pc_on"):
                    policy_overrides["price_controls"] = {"fuel": 1.05, "food": 1.03}

            impl_lag = st.number_input("Implementation Lag (quarters)", 0, 10, 0,
                                       key="sim_pol_lag")
            policy_overrides["implementation_lag"] = impl_lag
        else:
            policy_overrides = dict(instruments)

        st.session_state["_sim_policy_overrides"] = policy_overrides

        # Row 4: Dimension selector
        st.markdown(f"<div style='color:{theme.text_muted}; font-weight:600; "
                    f"font-size:0.78rem; margin-top:0.5rem;'>OUTCOME DIMENSIONS TO WATCH</div>",
                    unsafe_allow_html=True)

        categories = {}
        for key, meta in outcome_dimensions.items():
            cat = meta.get("category", "Other")
            categories.setdefault(cat, []).append((key, meta))

        defaults = (scenario.suggested_dimensions if scenario else None) or list(default_dimensions)

        selected_dims = []
        dim_cols = st.columns(min(len(categories), 5))
        for i, (cat, dims) in enumerate(categories.items()):
            with dim_cols[i % len(dim_cols)]:
                st.markdown(f"<div style='color:{theme.text_muted}; font-weight:600; "
                            f"font-size:0.72rem;'>{cat}</div>", unsafe_allow_html=True)
                for key, meta in dims:
                    if st.checkbox(meta["label"], value=key in defaults,
                                   key=f"sim_dim_{key}", help=meta["description"]):
                        selected_dims.append(key)
        if not selected_dims:
            selected_dims = list(default_dimensions)

    return scenario, policy_overrides, selected_dims, steps


# ═════════════════════════════════════════════════════════════════════════════
#  SIMULATION EXECUTION
# ═════════════════════════════════════════════════════════════════════════════

def _run_simulation(theme, SFCEconomy, SFCConfig, calibrate_from_data,
                    build_custom_scenario, selected_dims, steps):
    """Execute button + SFC simulation. Returns True if a new run happened."""
    col_r1, col_r2 = st.columns([3, 1])
    with col_r1:
        # Calibration confidence badge
        calib_cached = st.session_state.get("sim_calibration")
        if calib_cached:
            conf = calib_cached.overall_confidence
            cc = theme.accent_success if conf > 0.6 else theme.accent_warning if conf > 0.3 else theme.accent_danger
            n_data = sum(1 for p in calib_cached.params.values() if p.source == "data")
            st.markdown(f"""
            <div style="font-size:0.75rem; color:{theme.text_muted}; padding-top:0.5rem;">
                Calibration: <span style="color:{cc}; font-weight:600;">{conf:.0%}</span>
                &nbsp;|&nbsp; {n_data}/{len(calib_cached.params)} params from data
            </div>
            """, unsafe_allow_html=True)
    with col_r2:
        run_clicked = st.button("RUN SIMULATION", type="primary", use_container_width=True)

    if run_clicked:
        with st.spinner("Calibrating from data and running simulation..."):
            try:
                scenario_id = st.session_state.get("_sim_scenario_id", "custom")
                scenario = st.session_state.get("_sim_scenario_obj")
                policy_key = st.session_state.get("_sim_policy_key", "do_nothing")
                policy_overrides = st.session_state.get("_sim_policy_overrides", {})

                policy_mode = "custom" if policy_key != "do_nothing" else "off"

                config_overrides = {
                    k: v for k, v in policy_overrides.items()
                    if k in SFCConfig.__dataclass_fields__
                }

                calib = calibrate_from_data(
                    steps=steps, policy_mode=policy_mode, overrides=config_overrides,
                )
                cfg = calib.config

                # Build shock vectors
                if scenario_id == "custom":
                    custom_shocks = st.session_state.get("_sim_custom_shocks", {})
                    cs = build_custom_scenario(
                        name="Custom", shocks=custom_shocks,
                        shock_onset=st.session_state.get("_sim_onset", 5),
                        shock_duration=0,
                        shock_shape=st.session_state.get("_sim_shape", "step"),
                        dimensions=selected_dims,
                    )
                    cfg.shock_vectors = cs.build_shock_vectors(steps)
                elif scenario:
                    cfg.shock_vectors = scenario.build_shock_vectors(steps)

                econ = SFCEconomy(cfg)
                econ.initialize()
                trajectory = econ.run(steps)

                # Store results
                st.session_state["sim_trajectory"] = trajectory
                st.session_state["sim_selected_dims"] = selected_dims
                st.session_state["sim_calibration"] = calib
                st.session_state["sim_steps"] = steps
                st.session_state["sim_state"] = econ

                # Also store for comparison tab
                label = scenario.name if scenario else "Custom"
                history = st.session_state.get("sim_compare_history", [])
                history.append({
                    "label": f"{label} + {policy_key}",
                    "trajectory": trajectory,
                    "dims": selected_dims,
                })
                # Keep last 5 runs
                st.session_state["sim_compare_history"] = history[-5:]

            except Exception as e:
                st.error(f"Simulation error: {e}")
                import traceback
                st.code(traceback.format_exc())

    return run_clicked


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 1: SCENARIO RUNNER — Impact cards + trajectory
# ═════════════════════════════════════════════════════════════════════════════

def _render_scenario_runner_tab(theme, outcome_dimensions, default_dimensions, run_clicked):
    trajectory = st.session_state.get("sim_trajectory")
    if not trajectory or len(trajectory) < 2:
        if not run_clicked:
            st.markdown(f"""
            <div style="text-align:center; padding:3rem; color:{theme.text_muted};">
                <div style="font-size:2rem; margin-bottom:0.5rem; opacity:0.3;">&#9654;</div>
                <div style="font-size:0.9rem;">
                    Configure your scenario above and click <b>RUN SIMULATION</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        return

    sel_dims = st.session_state.get("sim_selected_dims", list(default_dimensions))
    calib = st.session_state.get("sim_calibration")

    # Calibration info
    if calib:
        conf = calib.overall_confidence
        cc = theme.accent_success if conf > 0.6 else theme.accent_warning if conf > 0.3 else theme.accent_danger
        n_data = sum(1 for p in calib.params.values() if p.source == "data")
        st.markdown(f"""
        <div style="font-size:0.75rem; color:{theme.text_muted}; margin-bottom:0.8rem;">
            Engine: <span style="font-weight:600;">PARAMETRIC SFC</span>
            &nbsp;|&nbsp; Calibration: <span style="color:{cc}; font-weight:600;">{conf:.0%}</span>
            &nbsp;|&nbsp; {n_data}/{len(calib.params)} from data
            &nbsp;|&nbsp; {len(trajectory)} frames
        </div>
        """, unsafe_allow_html=True)

    # Impact delta cards
    _render_impact_cards(trajectory, sel_dims, outcome_dimensions, theme)
    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)

    # Time-series trajectory
    if HAS_PLOTLY:
        _render_time_series(trajectory, sel_dims, outcome_dimensions, theme)


def _render_impact_cards(trajectory, sel_dims, outcome_dimensions, theme):
    start = trajectory[0].get("outcomes", {})
    end = trajectory[-1].get("outcomes", {})

    for row_start in range(0, len(sel_dims), 5):
        row_dims = sel_dims[row_start:row_start + 5]
        cols = st.columns(len(row_dims))
        for i, dim_key in enumerate(row_dims):
            meta = outcome_dimensions.get(dim_key, {
                "label": dim_key, "format": ".2f", "higher_is": "better"
            })
            s_val = start.get(dim_key, 0)
            e_val = end.get(dim_key, 0)
            delta = e_val - s_val

            fmt = meta.get("format", ".2f")
            display_val = f"{e_val:{fmt}}"
            display_delta = f"{delta:+{fmt}}"

            is_good = (delta > 0 and meta.get("higher_is") == "better") or \
                      (delta < 0 and meta.get("higher_is") == "worse")
            dc = theme.accent_success if is_good else theme.accent_danger
            arrow = "^" if delta > 0 else "v" if delta < 0 else "-"

            with cols[i]:
                st.markdown(f"""
                <div style="background:{theme.bg_tertiary}; padding:0.8rem; border-radius:10px;
                            border:1px solid {theme.border_default}; text-align:center;">
                    <div style="color:{theme.text_muted}; font-size:0.7rem; font-weight:600;
                                letter-spacing:0.5px; margin-bottom:0.3rem;">
                        {meta.get('label', dim_key).upper()}
                    </div>
                    <div style="color:{theme.text_primary}; font-size:1.3rem; font-weight:700;">
                        {display_val}
                    </div>
                    <div style="color:{dc}; font-size:0.85rem; font-weight:600;">
                        {arrow} {display_delta}
                    </div>
                </div>
                """, unsafe_allow_html=True)


def _render_time_series(trajectory, sel_dims, outcome_dimensions, theme):
    fig = make_subplots(rows=1, cols=1)
    t_vals = [f["t"] for f in trajectory]

    for idx, dim_key in enumerate(sel_dims):
        meta = outcome_dimensions.get(dim_key, {"label": dim_key})
        vals = [f.get("outcomes", {}).get(dim_key, 0) for f in trajectory]

        fmt = meta.get("format", ".2f")
        suffix = " (%)" if "%" in fmt else ""
        if "%" in fmt:
            vals = [v * 100 for v in vals]

        fig.add_trace(go.Scatter(
            x=t_vals, y=vals, mode='lines',
            name=f"{meta.get('label', dim_key)}{suffix}",
            line=dict(color=PALETTE[idx % len(PALETTE)], width=2.5),
            hovertemplate=f"{meta.get('label', dim_key)}: %{{y:.2f}}{suffix}<extra></extra>",
        ))

    # Shock onset marker
    scenario = st.session_state.get("_sim_scenario_obj")
    onset = scenario.shock_onset if scenario else 5
    fig.add_vline(x=onset, line_dash="dash", line_color=theme.accent_danger,
                  annotation_text="Shock", annotation_font_color=theme.accent_danger)

    fig.update_layout(**_base_layout(theme, height=420,
        title=dict(text="Trajectory Over Time", font=dict(color=theme.text_muted, size=13)),
        xaxis=dict(title="Quarter"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor='rgba(0,0,0,0)'),
    ))
    st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 2: SENSITIVITY — Policy-outcome heatmap
# ═════════════════════════════════════════════════════════════════════════════

def _render_sensitivity_tab(theme, outcome_dimensions):
    trajectory = st.session_state.get("sim_trajectory")
    if not trajectory or len(trajectory) < 6:
        st.info("Run a simulation with at least 6 quarters to see sensitivity analysis.")
        return

    if not HAS_PLOTLY:
        st.warning("Plotly required for heatmap visualization.")
        return

    policy_keys = sorted({k for f in trajectory for k in f.get("policy_vector", {})})
    outcome_keys = sorted({k for f in trajectory for k in f.get("outcomes", {})})
    if not policy_keys or not outcome_keys:
        st.info("Policy/outcome vectors missing from simulation frames.")
        return

    impacts = []
    for pol in policy_keys:
        p = np.array([float(f.get("policy_vector", {}).get(pol, 0)) for f in trajectory])
        row = []
        for out in outcome_keys:
            y = np.array([float(f.get("outcomes", {}).get(out, 0)) for f in trajectory])
            if np.allclose(p.std(), 0) or np.allclose(y.std(), 0):
                row.append(0.0)
            else:
                row.append(float(np.corrcoef(p, y)[0, 1]))
        impacts.append(row)

    fig = go.Figure(go.Heatmap(
        z=impacts,
        x=[o.replace("_", " ").title() for o in outcome_keys],
        y=[p.replace("_", " ").title() for p in policy_keys],
        colorscale=[[0, theme.accent_danger], [0.5, '#ffffff'], [1, theme.accent_success]],
        zmin=-1, zmax=1,
        text=[[f"{v:+.2f}" for v in row] for row in impacts],
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(**_base_layout(theme, height=400,
        title=dict(text="Policy-Outcome Sensitivity Matrix",
                   font=dict(color=theme.text_muted, size=13)),
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Plain-English interpretation
    if impacts:
        st.markdown(f"""
        <div style="font-size:0.8rem; color:{theme.text_muted}; padding:0.5rem 0;">
            <b>Reading:</b> Each cell shows the correlation between a policy instrument
            and an outcome dimension over the simulation trajectory.
            Values near <span style="color:{theme.accent_success}">+1.0</span> mean the
            instrument is strongly positively associated;
            <span style="color:{theme.accent_danger}">-1.0</span> means strongly negative.
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 3: STATE CUBE — Dynamic N-D sector view
# ═════════════════════════════════════════════════════════════════════════════

def _render_state_cube_tab(theme, outcome_dimensions):
    trajectory = st.session_state.get("sim_trajectory")
    if not trajectory or len(trajectory) < 4:
        st.info("Run a simulation to see the state cube.")
        return

    if not HAS_PLOTLY:
        st.warning("Plotly required for 3D visualization.")
        return

    dims = _discover_dimensions(trajectory)
    all_opts = _flat_dim_options(dims)
    if len(all_opts) < 3:
        st.info("Need at least 3 discoverable dimensions for a 3D cube.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        x_dim = st.selectbox("X axis", all_opts, index=0, key="cube_x",
                              format_func=lambda s: _dim_label(s.split("::")[-1]))
    with c2:
        y_dim = st.selectbox("Y axis", all_opts,
                              index=min(1, len(all_opts) - 1), key="cube_y",
                              format_func=lambda s: _dim_label(s.split("::")[-1]))
    with c3:
        z_dim = st.selectbox("Z axis", all_opts,
                              index=min(2, len(all_opts) - 1), key="cube_z",
                              format_func=lambda s: _dim_label(s.split("::")[-1]))
    with c4:
        color_opts = ["time"] + all_opts
        color_dim = st.selectbox("Color", color_opts, index=0, key="cube_color",
                                  format_func=lambda s: "Quarter" if s == "time"
                                  else _dim_label(s.split("::")[-1]))

    t_vals = [f.get("t", 0) for f in trajectory]
    x_vals = [_extract_dim(x_dim, f) for f in trajectory]
    y_vals = [_extract_dim(y_dim, f) for f in trajectory]
    z_vals = [_extract_dim(z_dim, f) for f in trajectory]
    c_vals = t_vals if color_dim == "time" else [_extract_dim(color_dim, f) for f in trajectory]
    c_title = "Quarter" if color_dim == "time" else _dim_label(color_dim.split("::")[-1])

    x_label = _dim_label(x_dim.split("::")[-1])
    y_label = _dim_label(y_dim.split("::")[-1])
    z_label = _dim_label(z_dim.split("::")[-1])

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x_vals, y=y_vals, z=z_vals,
        mode='lines+markers',
        marker=dict(size=4, color=c_vals, colorscale='Plasma',
                    colorbar=dict(title=c_title, len=0.6), opacity=0.9),
        line=dict(color=theme.accent_primary, width=3),
        hovertemplate=(
            f"{x_label}: %{{x:.3f}}<br>"
            f"{y_label}: %{{y:.3f}}<br>"
            f"{z_label}: %{{z:.3f}}<extra></extra>"
        ),
        name="State Trajectory",
    ))
    # Start / End markers
    fig.add_trace(go.Scatter3d(
        x=[x_vals[0]], y=[y_vals[0]], z=[z_vals[0]],
        mode='markers', marker=dict(size=10, color=theme.accent_success, symbol='diamond'),
        name='Start', showlegend=True,
    ))
    fig.add_trace(go.Scatter3d(
        x=[x_vals[-1]], y=[y_vals[-1]], z=[z_vals[-1]],
        mode='markers', marker=dict(size=10, color=theme.accent_danger, symbol='diamond'),
        name='End', showlegend=True,
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title=x_label, yaxis_title=y_label, zaxis_title=z_label,
            bgcolor="rgba(0,0,0,0)",
        ),
        **_base_layout(theme, height=550,
            title=dict(text="3D State Cube — Dynamic Sector View",
                       font=dict(color=theme.text_muted, size=13))),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Dynamic summary table for ALL discovered sector_balances
    sector_keys = dims.get("sector_balances", [])
    if sector_keys and len(trajectory) > 1:
        final, initial = trajectory[-1], trajectory[0]
        bal_data = []
        for s in sector_keys:
            init_v = initial.get("sector_balances", {}).get(s, 0)
            fin_v = final.get("sector_balances", {}).get(s, 0)
            bal_data.append({
                "Sector": _dim_label(s),
                "Initial": f"{init_v:.4f}",
                "Final": f"{fin_v:.4f}",
                "Change": f"{fin_v - init_v:+.4f}",
            })
        st.dataframe(pd.DataFrame(bal_data), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 4: COMPARE — overlay multiple runs
# ═════════════════════════════════════════════════════════════════════════════

def _render_compare_tab(theme, outcome_dimensions, default_dimensions):
    history = st.session_state.get("sim_compare_history", [])
    if len(history) < 2:
        st.info("Run at least 2 different scenarios to compare trajectories. "
                "Each run is automatically saved for comparison (last 5 kept).")
        if history:
            st.caption(f"Currently stored: {len(history)}/5 runs — "
                       f"{', '.join(h['label'] for h in history)}")
        return

    if not HAS_PLOTLY:
        st.warning("Plotly required for comparison charts.")
        return

    # Let user pick which dimension to compare
    all_dims = set()
    for h in history:
        all_dims.update(h.get("dims", []))
    all_dims = sorted(all_dims)
    if not all_dims:
        all_dims = list(default_dimensions)

    focus_dim = st.selectbox("Compare dimension", all_dims,
                             format_func=lambda d: outcome_dimensions.get(d, {}).get("label", d),
                             key="sim_compare_dim")
    meta = outcome_dimensions.get(focus_dim, {"label": focus_dim, "format": ".2f"})
    fmt = meta.get("format", ".2f")
    is_pct = "%" in fmt

    fig = go.Figure()
    for i, h in enumerate(history):
        traj = h["trajectory"]
        t_vals = [f["t"] for f in traj]
        vals = [f.get("outcomes", {}).get(focus_dim, 0) for f in traj]
        if is_pct:
            vals = [v * 100 for v in vals]
        fig.add_trace(go.Scatter(
            x=t_vals, y=vals, mode='lines',
            name=h["label"],
            line=dict(color=PALETTE[i % len(PALETTE)], width=2.5),
        ))

    suffix = " (%)" if is_pct else ""
    fig.update_layout(**_base_layout(theme, height=420,
        title=dict(text=f"Comparison: {meta.get('label', focus_dim)}{suffix}",
                   font=dict(color=theme.text_muted, size=13)),
        xaxis=dict(title="Quarter"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor='rgba(0,0,0,0)'),
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    rows = []
    for h in history:
        traj = h["trajectory"]
        if len(traj) >= 2:
            start_v = traj[0].get("outcomes", {}).get(focus_dim, 0)
            end_v = traj[-1].get("outcomes", {}).get(focus_dim, 0)
            mult = 100 if is_pct else 1
            rows.append({
                "Run": h["label"],
                "Start": f"{start_v * mult:{fmt.replace('%', 'f')}}",
                "End": f"{end_v * mult:{fmt.replace('%', 'f')}}",
                "Delta": f"{(end_v - start_v) * mult:+.2f}",
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Clear button
    if st.button("Clear comparison history", key="sim_clear_compare"):
        st.session_state["sim_compare_history"] = []
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 5: DIAGNOSTICS — calibration & engine details (fully dynamic)
# ═════════════════════════════════════════════════════════════════════════════

def _render_diagnostics_tab(theme):
    calib = st.session_state.get("sim_calibration")
    trajectory = st.session_state.get("sim_trajectory")

    if not calib:
        st.info("Run a simulation to see engine diagnostics.")
        return

    # Calibration parameter table
    st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:600; "
                f"font-size:0.85rem; margin-bottom:0.5rem;'>CALIBRATION PARAMETERS</div>",
                unsafe_allow_html=True)

    param_rows = []
    for name, p in sorted(calib.params.items()):
        param_rows.append({
            "Parameter": name,
            "Value": f"{p.value:.4f}" if isinstance(p.value, float) else str(p.value),
            "Source": p.source,
            "Confidence": f"{p.confidence:.0%}",
            "Note": p.note or "",
        })
    st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)

    # Dynamic SFC state summary — auto-discover ALL outcome dimensions
    if trajectory and len(trajectory) > 1:
        st.markdown(f"<div style='color:{theme.accent_warning}; font-weight:600; "
                    f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
                    f"SFC ENGINE SUMMARY</div>", unsafe_allow_html=True)

        final = trajectory[-1]
        outcomes = final.get("outcomes", {})

        # Always show frame count first
        st.metric("Total Frames", len(trajectory))

        # Dynamically render metrics for ALL discovered outcome dimensions
        outcome_keys = sorted(outcomes.keys())
        # Filter out internal breach flags
        outcome_keys = [k for k in outcome_keys if not k.startswith("breach_")]

        # Render in rows of 4
        for row_start in range(0, len(outcome_keys), 4):
            row_keys = outcome_keys[row_start:row_start + 4]
            cols = st.columns(len(row_keys))
            for idx, key in enumerate(row_keys):
                val = outcomes[key]
                with cols[idx]:
                    # Auto-detect format: values in [0,1] that look like rates → show as %
                    if isinstance(val, float) and -1.0 <= val <= 2.0 and key not in ("financial_stability",):
                        st.metric(_dim_label(key), f"{val:.2%}")
                    else:
                        st.metric(_dim_label(key), f"{val:.4f}")

        # Sector balance check (SFC consistency) — dynamic sector discovery
        dims = _discover_dimensions(trajectory)
        sector_keys = dims.get("sector_balances", [])
        if sector_keys:
            all_balances = [
                sum(f.get("sector_balances", {}).get(s, 0) for s in sector_keys)
                for f in trajectory
            ]
            max_imbalance = max(abs(b) for b in all_balances) if all_balances else 0
            bal_ok = max_imbalance < 0.01
            bc = theme.accent_success if bal_ok else theme.accent_danger
            st.markdown(f"""
            <div style="font-size:0.8rem; color:{theme.text_muted}; margin-top:0.5rem;">
                SFC Balance Check: max sector imbalance =
                <span style="color:{bc}; font-weight:600;">{max_imbalance:.6f}</span>
                {'(PASS)' if bal_ok else '(WARNING: large imbalance)'}
            </div>
            """, unsafe_allow_html=True)

        # Channel dynamics summary (also dynamic)
        channel_keys = dims.get("channels", [])
        if channel_keys:
            st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:600; "
                        f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
                        f"CHANNEL DYNAMICS</div>", unsafe_allow_html=True)
            ch_cols = st.columns(len(channel_keys))
            final_channels = final.get("channels", {})
            for i, ck in enumerate(channel_keys):
                with ch_cols[i]:
                    cv = final_channels.get(ck, 0)
                    st.metric(_dim_label(ck), f"{cv:.4f}")


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 6: PHASE EXPLORER — 2D/3D trajectory through any state space
# ═════════════════════════════════════════════════════════════════════════════

def _render_phase_explorer_tab(theme):
    trajectory = st.session_state.get("sim_trajectory")
    if not trajectory or len(trajectory) < 4:
        st.info("Run a simulation to explore phase diagrams.")
        return
    if not HAS_PLOTLY:
        st.warning("Plotly required for phase diagrams.")
        return

    dims = _discover_dimensions(trajectory)
    all_opts = _flat_dim_options(dims)
    if len(all_opts) < 2:
        st.info("Not enough discoverable dimensions for a phase diagram.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        x_dim = st.selectbox("X axis", all_opts, index=0, key="phase_x",
                              format_func=lambda s: _dim_label(s.split("::")[-1]))
    with c2:
        y_dim = st.selectbox("Y axis", all_opts,
                              index=min(1, len(all_opts) - 1), key="phase_y",
                              format_func=lambda s: _dim_label(s.split("::")[-1]))
    with c3:
        z_options = ["(2D — no Z axis)"] + all_opts
        z_dim = st.selectbox("Z axis (optional)", z_options, index=0, key="phase_z",
                              format_func=lambda s: "(2D)" if s.startswith("(")
                              else _dim_label(s.split("::")[-1]))

    t_vals = [f.get("t", 0) for f in trajectory]
    x_vals = [_extract_dim(x_dim, f) for f in trajectory]
    y_vals = [_extract_dim(y_dim, f) for f in trajectory]
    x_label = _dim_label(x_dim.split("::")[-1])
    y_label = _dim_label(y_dim.split("::")[-1])

    is_3d = z_dim and not z_dim.startswith("(")

    if is_3d:
        z_vals = [_extract_dim(z_dim, f) for f in trajectory]
        z_label = _dim_label(z_dim.split("::")[-1])

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='lines+markers',
            marker=dict(size=4, color=t_vals, colorscale='Plasma',
                        colorbar=dict(title="Quarter", len=0.6), opacity=0.9),
            line=dict(color=theme.accent_primary, width=3),
            hovertemplate=(
                f"{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<br>"
                f"{z_label}: %{{z:.3f}}<extra></extra>"
            ),
            name="Trajectory",
        ))
        fig.add_trace(go.Scatter3d(
            x=[x_vals[0]], y=[y_vals[0]], z=[z_vals[0]],
            mode='markers', marker=dict(size=10, color=theme.accent_success, symbol='diamond'),
            name='Start', showlegend=True,
        ))
        fig.add_trace(go.Scatter3d(
            x=[x_vals[-1]], y=[y_vals[-1]], z=[z_vals[-1]],
            mode='markers', marker=dict(size=10, color=theme.accent_danger, symbol='diamond'),
            name='End', showlegend=True,
        ))
        fig.update_layout(
            scene=dict(xaxis_title=x_label, yaxis_title=y_label, zaxis_title=z_label,
                       bgcolor="rgba(0,0,0,0)"),
            **_base_layout(theme, height=560,
                title=dict(text="3D Phase Space Trajectory",
                           font=dict(color=theme.text_muted, size=13))),
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode='lines+markers',
            marker=dict(size=6, color=t_vals, colorscale='Plasma',
                        colorbar=dict(title="Quarter"), opacity=0.8),
            line=dict(color=theme.accent_primary, width=2),
            hovertemplate=(
                f"{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<extra></extra>"
            ),
            name="Trajectory",
        ))
        fig.add_trace(go.Scatter(
            x=[x_vals[0]], y=[y_vals[0]], mode='markers',
            marker=dict(size=12, color=theme.accent_success, symbol='diamond'),
            name='Start', showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=[x_vals[-1]], y=[y_vals[-1]], mode='markers',
            marker=dict(size=12, color=theme.accent_danger, symbol='diamond'),
            name='End', showlegend=True,
        ))
        fig.update_layout(**_base_layout(theme, height=480,
            title=dict(text="2D Phase Diagram",
                       font=dict(color=theme.text_muted, size=13)),
            xaxis=dict(title=x_label), yaxis=dict(title=y_label)))

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">
        <b>Reading:</b>  The path traces how the economy evolves through the chosen
        state space over time.  Colour encodes the quarter — early periods are dark,
        later periods bright.  <span style="color:{theme.accent_success}">&#x25C6;</span>
        = start, <span style="color:{theme.accent_danger}">&#x25C6;</span> = end.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 7: IMPULSE RESPONSE FUNCTIONS — auto-detected deviation from baseline
# ═════════════════════════════════════════════════════════════════════════════

def _render_irf_tab(theme):
    trajectory = st.session_state.get("sim_trajectory")
    if not trajectory or len(trajectory) < 6:
        st.info("Run a simulation with ≥ 6 quarters to see impulse responses.")
        return
    if not HAS_PLOTLY:
        st.warning("Plotly required for IRF charts.")
        return

    # Determine shock onset dynamically
    scenario = st.session_state.get("_sim_scenario_obj")
    onset = scenario.shock_onset if scenario and hasattr(scenario, "shock_onset") else 5

    # Auto-discover outcome dimensions
    outcome_keys = sorted({k for f in trajectory for k in f.get("outcomes", {})
                           if not k.startswith("breach_")})
    if not outcome_keys:
        st.info("No outcome dimensions found in trajectory frames.")
        return

    selected = st.multiselect(
        "Dimensions to display", outcome_keys,
        default=outcome_keys[:min(6, len(outcome_keys))],
        format_func=_dim_label, key="irf_dims",
    )
    if not selected:
        return

    # Baseline = average of pre-shock frames
    pre_shock = [f for f in trajectory if f["t"] < onset]
    if not pre_shock:
        pre_shock = trajectory[:3]

    baseline = {}
    for dim in selected:
        vals = [f.get("outcomes", {}).get(dim, 0) for f in pre_shock]
        baseline[dim] = float(np.mean(vals)) if vals else 0.0

    post_shock = [f for f in trajectory if f["t"] >= onset]
    t_relative = [f["t"] - onset for f in post_shock]

    view_mode = st.radio("View", ["2D Lines", "3D Surface"], horizontal=True, key="irf_view")

    if view_mode == "3D Surface" and len(selected) >= 2:
        z_matrix = []
        for dim in selected:
            base = baseline.get(dim, 0)
            irfs = [(f.get("outcomes", {}).get(dim, 0) - base) for f in post_shock]
            if abs(base) > 1e-8:
                irfs = [v / abs(base) * 100 for v in irfs]
            z_matrix.append(irfs)

        fig = go.Figure(go.Surface(
            x=t_relative,
            y=list(range(len(selected))),
            z=z_matrix,
            colorscale='RdBu_r',
            colorbar=dict(title="% Dev"),
            hovertemplate="t+%{x}<br>Dim idx: %{y}<br>IRF: %{z:.2f}%<extra></extra>",
        ))
        fig.update_layout(
            scene=dict(
                xaxis_title="Quarters After Shock",
                yaxis_title="Dimension",
                zaxis_title="% Deviation from Baseline",
                yaxis=dict(
                    tickvals=list(range(len(selected))),
                    ticktext=[_dim_label(d)[:15] for d in selected],
                ),
                bgcolor="rgba(0,0,0,0)",
            ),
            **_base_layout(theme, height=560,
                title=dict(text="3D Impulse Response Surface",
                           font=dict(color=theme.text_muted, size=13))),
        )
    else:
        fig = make_subplots(rows=1, cols=1)
        for idx, dim in enumerate(selected):
            base = baseline.get(dim, 0)
            irfs = [(f.get("outcomes", {}).get(dim, 0) - base) for f in post_shock]
            if abs(base) > 1e-8:
                irfs = [v / abs(base) * 100 for v in irfs]
            fig.add_trace(go.Scatter(
                x=t_relative, y=irfs, mode='lines',
                name=_dim_label(dim),
                line=dict(color=PALETTE[idx % len(PALETTE)], width=2.5),
            ))
        fig.add_hline(y=0, line_dash="dot", line_color=theme.text_muted)
        fig.update_layout(**_base_layout(theme, height=460,
            title=dict(text="Impulse Response Functions (% deviation from baseline)",
                       font=dict(color=theme.text_muted, size=13)),
            xaxis=dict(title="Quarters After Shock"),
            yaxis=dict(title="% Deviation"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                        bgcolor='rgba(0,0,0,0)')))

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">
        <b>Reading:</b> Each line shows how a dimension deviates from its pre-shock
        baseline (t<{onset}).  Positive = above baseline, negative = below.
        Values are normalised to percentage deviation where possible.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 8: FLOW DYNAMICS — Sankey + waterfall + 3D surface
# ═════════════════════════════════════════════════════════════════════════════

def _render_flow_sankey_tab(theme):
    trajectory = st.session_state.get("sim_trajectory")
    if not trajectory or len(trajectory) < 2:
        st.info("Run a simulation to see flow diagrams.")
        return
    if not HAS_PLOTLY:
        st.warning("Plotly required for flow visualization.")
        return

    flow_keys = sorted({k for f in trajectory for k in f.get("flows", {})})
    if not flow_keys:
        st.info("No flow data found in simulation frames.")
        return

    # Time-step selector
    max_t = len(trajectory) - 1
    t_idx = st.slider("Snapshot quarter", 0, max_t, max_t, key="sankey_t")
    frame = trajectory[t_idx]
    flows = frame.get("flows", {})

    # ── Waterfall bar chart ───────────────────────────────────────────────────
    sorted_flows = sorted(
        [(k, flows.get(k, 0)) for k in flow_keys],
        key=lambda x: abs(x[1]), reverse=True,
    )
    names = [_dim_label(k) for k, _ in sorted_flows]
    values = [v for _, v in sorted_flows]
    colors = [theme.accent_success if v >= 0 else theme.accent_danger for v in values]

    fig_bar = go.Figure(go.Bar(
        x=names, y=values, marker_color=colors,
        text=[f"{v:.4f}" for v in values], textposition='outside',
    ))
    fig_bar.update_layout(**_base_layout(theme, height=400,
        title=dict(text=f"Economic Flows at Quarter {frame.get('t', t_idx)}",
                   font=dict(color=theme.text_muted, size=13)),
        xaxis=dict(title=""), yaxis=dict(title="Flow Magnitude")))
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Sankey diagram ────────────────────────────────────────────────────────
    if len(flow_keys) >= 3:
        all_nodes = ["Inflows", "Outflows"] + [_dim_label(k) for k in flow_keys]
        node_idx = {n: i for i, n in enumerate(all_nodes)}
        sources, targets, values_s, labels_s = [], [], [], []
        for k in flow_keys:
            v = flows.get(k, 0)
            label = _dim_label(k)
            mag = max(abs(v), 1e-8)
            if v >= 0:
                sources.append(node_idx["Inflows"])
                targets.append(node_idx[label])
            else:
                sources.append(node_idx[label])
                targets.append(node_idx["Outflows"])
            values_s.append(mag)
            labels_s.append(f"{label}: {v:.4f}")

        fig_sankey = go.Figure(go.Sankey(
            node=dict(
                label=all_nodes,
                color=[PALETTE[i % len(PALETTE)] for i in range(len(all_nodes))],
                pad=15, thickness=20,
            ),
            link=dict(
                source=sources, target=targets, value=values_s,
                label=labels_s, color="rgba(100,100,100,0.3)",
            ),
        ))
        fig_sankey.update_layout(**_base_layout(theme, height=450,
            title=dict(text="Flow Sankey Diagram",
                       font=dict(color=theme.text_muted, size=13))))
        st.plotly_chart(fig_sankey, use_container_width=True)

    # ── 3D Flow Surface (flows × time) ───────────────────────────────────────
    if st.checkbox("Show 3D Flow Dynamics Surface", key="sankey_3d"):
        z_matrix = []
        t_vals = []
        for fr in trajectory:
            t_vals.append(fr.get("t", 0))
            z_matrix.append([fr.get("flows", {}).get(k, 0) for k in flow_keys])

        fig_3d = go.Figure(go.Surface(
            x=list(range(len(flow_keys))),
            y=t_vals,
            z=z_matrix,
            colorscale='Viridis',
            colorbar=dict(title="Value"),
        ))
        fig_3d.update_layout(
            scene=dict(
                xaxis_title="Flow",
                yaxis_title="Quarter",
                zaxis_title="Magnitude",
                xaxis=dict(
                    tickvals=list(range(len(flow_keys))),
                    ticktext=[_dim_label(k)[:12] for k in flow_keys],
                ),
                bgcolor="rgba(0,0,0,0)",
            ),
            **_base_layout(theme, height=560,
                title=dict(text="3D Flow Dynamics Surface",
                           font=dict(color=theme.text_muted, size=13))),
        )
        st.plotly_chart(fig_3d, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 9: MONTE CARLO — Fan charts with parameter jitter
# ═════════════════════════════════════════════════════════════════════════════

def _render_monte_carlo_tab(theme, SFCEconomy, SFCConfig,
                            calibrate_from_data, build_custom_scenario):
    trajectory = st.session_state.get("sim_trajectory")
    if not trajectory or len(trajectory) < 4:
        st.info("Run a base simulation first, then generate Monte Carlo uncertainty bands.")
        return
    if not HAS_PLOTLY:
        st.warning("Plotly required for fan charts.")
        return

    # Auto-discover outcome dimensions
    outcome_keys = sorted({k for f in trajectory for k in f.get("outcomes", {})
                           if not k.startswith("breach_")})
    if not outcome_keys:
        st.info("No outcome dimensions found.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        focus_dim = st.selectbox("Dimension", outcome_keys,
                                  format_func=_dim_label, key="mc_dim")
    with c2:
        n_runs = st.slider("Number of runs", 10, 200, 50, 10, key="mc_runs")
    with c3:
        jitter_pct = st.slider("Parameter jitter (%)", 1, 25, 10, 1, key="mc_jitter")

    if st.button("Run Monte Carlo", key="mc_go", type="primary"):
        calib = st.session_state.get("sim_calibration")
        steps = st.session_state.get("sim_steps", 50)
        scenario = st.session_state.get("_sim_scenario_obj")

        if not calib:
            st.error("No calibration found. Run base simulation first.")
            return

        progress = st.progress(0, text="Running Monte Carlo simulations...")
        mc_trajectories = []

        for i in range(n_runs):
            try:
                # Jitter each numeric config field
                cfg_dict = {}
                for field_name in SFCConfig.__dataclass_fields__:
                    val = getattr(calib.config, field_name, None)
                    if isinstance(val, (int, float)) and field_name not in ("steps", "dt"):
                        j = 1.0 + np.random.uniform(-jitter_pct / 100, jitter_pct / 100)
                        cfg_dict[field_name] = val * j
                    elif val is not None:
                        cfg_dict[field_name] = val

                cfg = SFCConfig(**{k: v for k, v in cfg_dict.items()
                                   if k in SFCConfig.__dataclass_fields__})
                cfg.steps = steps

                if scenario and hasattr(scenario, 'build_shock_vectors'):
                    cfg.shock_vectors = scenario.build_shock_vectors(steps)

                econ = SFCEconomy(cfg)
                econ.initialize()
                traj = econ.run(steps)
                mc_trajectories.append(traj)
            except Exception:
                pass

            progress.progress((i + 1) / n_runs, text=f"Run {i + 1}/{n_runs}...")

        progress.empty()

        if len(mc_trajectories) < 5:
            st.warning(f"Only {len(mc_trajectories)} successful runs. Try reducing jitter.")
            return

        st.session_state["mc_trajectories"] = mc_trajectories
        st.session_state["mc_focus_dim"] = focus_dim

    # ── Render fan chart if MC results exist ──────────────────────────────────
    mc_trajs = st.session_state.get("mc_trajectories", [])
    if not mc_trajs:
        return

    focus = st.session_state.get("mc_focus_dim", focus_dim)

    max_len = max(len(t) for t in mc_trajs)
    t_vals = list(range(max_len))

    all_vals = []
    for t_idx in range(max_len):
        step_vals = []
        for traj in mc_trajs:
            if t_idx < len(traj):
                step_vals.append(traj[t_idx].get("outcomes", {}).get(focus, 0))
        all_vals.append(step_vals)

    percentiles = [10, 25, 50, 75, 90]
    bands = {p: [float(np.percentile(sv, p)) if sv else 0 for sv in all_vals]
             for p in percentiles}

    view_mode = st.radio("View", ["2D Fan Chart", "3D Uncertainty Surface"],
                         horizontal=True, key="mc_view")

    if view_mode == "3D Uncertainty Surface":
        z_matrix = [bands[p] for p in percentiles]
        fig = go.Figure(go.Surface(
            x=t_vals, y=percentiles, z=z_matrix,
            colorscale='Plasma', colorbar=dict(title="Value"), opacity=0.85,
        ))
        fig.update_layout(
            scene=dict(
                xaxis_title="Quarter", yaxis_title="Percentile",
                zaxis_title=_dim_label(focus),
                bgcolor="rgba(0,0,0,0)",
            ),
            **_base_layout(theme, height=560,
                title=dict(text=f"3D Uncertainty Surface — {_dim_label(focus)}",
                           font=dict(color=theme.text_muted, size=13))),
        )
    else:
        fig = go.Figure()
        band_pairs = [(10, 90, 0.15), (25, 75, 0.25)]
        for lo, hi, opacity in band_pairs:
            fig.add_trace(go.Scatter(
                x=t_vals + t_vals[::-1],
                y=bands[hi] + bands[lo][::-1],
                fill='toself',
                fillcolor=f"rgba(0,170,255,{opacity})",
                line=dict(color='rgba(0,0,0,0)'),
                name=f"{lo}th–{hi}th percentile",
                showlegend=True,
            ))
        fig.add_trace(go.Scatter(
            x=t_vals, y=bands[50], mode='lines',
            name='Median', line=dict(color=theme.accent_primary, width=3),
        ))
        base_vals = [f.get("outcomes", {}).get(focus, 0) for f in trajectory]
        fig.add_trace(go.Scatter(
            x=list(range(len(base_vals))), y=base_vals, mode='lines',
            name='Base Run', line=dict(color=theme.accent_warning, width=2, dash='dash'),
        ))
        fig.update_layout(**_base_layout(theme, height=480,
            title=dict(text=f"Monte Carlo Fan Chart — {_dim_label(focus)} ({len(mc_trajs)} runs)",
                       font=dict(color=theme.text_muted, size=13)),
            xaxis=dict(title="Quarter"),
            yaxis=dict(title=_dim_label(focus)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                        bgcolor='rgba(0,0,0,0)')))

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">
        {len(mc_trajs)} successful runs &nbsp;|&nbsp; Parameter jitter: &plusmn;{jitter_pct}%
        &nbsp;|&nbsp; Bands: 10th–90th and 25th–75th percentiles
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 10: STRESS TEST MATRIX — all scenarios × all outcomes
# ═════════════════════════════════════════════════════════════════════════════

def _render_stress_matrix_tab(theme, scenario_library,
                              SFCEconomy, SFCConfig, calibrate_from_data):
    if not HAS_PLOTLY:
        st.warning("Plotly required for stress test visualization.")
        return

    steps = st.session_state.get("sim_steps", 50)
    policy_key = st.session_state.get("_sim_policy_key", "do_nothing")
    policy_overrides = st.session_state.get("_sim_policy_overrides", {})

    st.markdown(f"""
    <div style="font-size:0.8rem; color:{theme.text_muted}; margin-bottom:0.5rem;">
        Runs <b>every</b> scenario in the library with the currently selected policy
        and shows the impact across <b>all auto-discovered</b> outcome dimensions.
    </div>
    """, unsafe_allow_html=True)

    if st.button("Run All Scenarios Stress Test", key="stress_go", type="primary"):
        progress = st.progress(0, text="Running stress tests...")
        results = {}

        for i, scenario in enumerate(scenario_library):
            try:
                policy_mode = "custom" if policy_key != "do_nothing" else "off"
                config_overrides = {k: v for k, v in policy_overrides.items()
                                    if k in SFCConfig.__dataclass_fields__}
                calib = calibrate_from_data(
                    steps=steps, policy_mode=policy_mode, overrides=config_overrides,
                )
                cfg = calib.config
                cfg.shock_vectors = scenario.build_shock_vectors(steps)

                econ = SFCEconomy(cfg)
                econ.initialize()
                traj = econ.run(steps)

                if traj and len(traj) > 1:
                    final = traj[-1].get("outcomes", {})
                    initial = traj[0].get("outcomes", {})
                    deltas = {k: final.get(k, 0) - initial.get(k, 0) for k in final}
                    results[scenario.name] = {"final": final, "delta": deltas}
            except Exception:
                pass

            progress.progress((i + 1) / len(scenario_library))

        progress.empty()
        st.session_state["stress_results"] = results

    results = st.session_state.get("stress_results", {})
    if not results:
        st.info("Click the button above to generate the stress test matrix.")
        return

    # Auto-discover all outcome dimensions across all scenario runs
    all_dims = sorted({k for r in results.values() for k in r.get("delta", {})
                       if not k.startswith("breach_")})
    scenarios = list(results.keys())

    if not all_dims or not scenarios:
        st.warning("No results to display.")
        return

    z_matrix = []
    text_matrix = []
    for s in scenarios:
        row, text_row = [], []
        for d in all_dims:
            v = results[s].get("delta", {}).get(d, 0)
            row.append(v)
            text_row.append(f"{v:+.3f}")
        z_matrix.append(row)
        text_matrix.append(text_row)

    fig = go.Figure(go.Heatmap(
        z=z_matrix,
        x=[_dim_label(d) for d in all_dims],
        y=scenarios,
        colorscale='RdYlGn',
        text=text_matrix,
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(title="Delta"),
    ))
    fig.update_layout(**_base_layout(theme, height=max(400, len(scenarios) * 55),
        title=dict(text=f"Stress Matrix — {len(scenarios)} scenarios × {len(all_dims)} dimensions",
                   font=dict(color=theme.text_muted, size=13))))
    st.plotly_chart(fig, use_container_width=True)

    # 3D Stress Surface
    if st.checkbox("Show 3D Stress Surface", key="stress_3d"):
        fig_3d = go.Figure(go.Surface(
            x=list(range(len(all_dims))),
            y=list(range(len(scenarios))),
            z=z_matrix,
            colorscale='RdYlGn',
            colorbar=dict(title="Delta"),
        ))
        fig_3d.update_layout(
            scene=dict(
                xaxis_title="Dimension",
                yaxis_title="Scenario",
                zaxis_title="Impact (Delta)",
                xaxis=dict(tickvals=list(range(len(all_dims))),
                           ticktext=[_dim_label(d)[:12] for d in all_dims]),
                yaxis=dict(tickvals=list(range(len(scenarios))),
                           ticktext=[s[:15] for s in scenarios]),
                bgcolor="rgba(0,0,0,0)",
            ),
            **_base_layout(theme, height=560,
                title=dict(text="3D Stress Test Surface",
                           font=dict(color=theme.text_muted, size=13))),
        )
        st.plotly_chart(fig_3d, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 11: PARAMETER RESPONSE SURFACE — 3D sweep
# ═════════════════════════════════════════════════════════════════════════════

def _render_parameter_surface_tab(theme, SFCEconomy, SFCConfig, calibrate_from_data):
    trajectory = st.session_state.get("sim_trajectory")
    if not trajectory or len(trajectory) < 4:
        st.info("Run a base simulation first to enable parameter sweeps.")
        return
    if not HAS_PLOTLY:
        st.warning("Plotly required for 3D surfaces.")
        return

    calib = st.session_state.get("sim_calibration")
    if not calib:
        st.info("No calibration available — run the simulation first.")
        return

    # Auto-discover tunable parameters from SFCConfig
    tunable_params = sorted([
        fn for fn in SFCConfig.__dataclass_fields__
        if isinstance(getattr(calib.config, fn, None), float)
        and fn not in ("steps", "dt")
    ])
    outcome_keys = sorted({k for f in trajectory for k in f.get("outcomes", {})
                           if not k.startswith("breach_")})
    if not tunable_params or not outcome_keys:
        st.info("Need tunable parameters and outcome dimensions.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        param = st.selectbox("Sweep parameter", tunable_params,
                              format_func=_dim_label, key="surf_param")
    with c2:
        outcome = st.selectbox("Outcome dimension", outcome_keys,
                                format_func=_dim_label, key="surf_outcome")
    with c3:
        n_sweep = st.slider("Sweep points", 5, 30, 15, key="surf_n")

    base_val = getattr(calib.config, param, 0.1)
    sweep_range = st.slider(
        f"Range around base ({base_val:.4f})",
        0.1, 2.0, 0.5, 0.1, key="surf_range",
        help="Multiplier: 0.5 → sweep from 50% to 150% of base value",
    )

    if st.button("Generate Response Surface", key="surf_go", type="primary"):
        lo = base_val * (1 - sweep_range)
        hi = base_val * (1 + sweep_range)
        param_vals = np.linspace(max(lo, 1e-6), hi, n_sweep).tolist()

        scenario = st.session_state.get("_sim_scenario_obj")
        steps = st.session_state.get("sim_steps", 50)

        progress = st.progress(0, text="Sweeping parameter space...")
        z_surface = []
        valid_params = []

        for i, pv in enumerate(param_vals):
            try:
                cfg_dict = {}
                for fn in SFCConfig.__dataclass_fields__:
                    cfg_dict[fn] = getattr(calib.config, fn)
                cfg_dict[param] = pv
                cfg_dict["steps"] = steps

                cfg = SFCConfig(**{k: v for k, v in cfg_dict.items()
                                   if k in SFCConfig.__dataclass_fields__})
                if scenario and hasattr(scenario, 'build_shock_vectors'):
                    cfg.shock_vectors = scenario.build_shock_vectors(steps)

                econ = SFCEconomy(cfg)
                econ.initialize()
                traj = econ.run(steps)

                row = [f.get("outcomes", {}).get(outcome, 0) for f in traj]
                z_surface.append(row)
                valid_params.append(pv)
            except Exception:
                pass

            progress.progress((i + 1) / n_sweep)

        progress.empty()
        st.session_state["surf_z"] = z_surface
        st.session_state["surf_params"] = valid_params
        st.session_state["surf_outcome_name"] = outcome
        st.session_state["surf_param_name"] = param

    # ── Render the surface if data exists ─────────────────────────────────────
    z_surface = st.session_state.get("surf_z")
    valid_params = st.session_state.get("surf_params")

    if not z_surface or not valid_params:
        return

    surf_outcome = st.session_state.get("surf_outcome_name", outcome)
    surf_param = st.session_state.get("surf_param_name", param)

    # Normalise row lengths
    max_len = max(len(r) for r in z_surface)
    for i, r in enumerate(z_surface):
        if len(r) < max_len:
            z_surface[i] = r + [r[-1]] * (max_len - len(r))

    fig = go.Figure(go.Surface(
        x=list(range(max_len)),
        y=valid_params,
        z=z_surface,
        colorscale='Viridis',
        colorbar=dict(title=_dim_label(surf_outcome)[:15]),
        opacity=0.9,
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title="Quarter",
            yaxis_title=_dim_label(surf_param),
            zaxis_title=_dim_label(surf_outcome),
            bgcolor="rgba(0,0,0,0)",
        ),
        **_base_layout(theme, height=600,
            title=dict(
                text=f"Parameter Response Surface: {_dim_label(surf_param)} → {_dim_label(surf_outcome)}",
                font=dict(color=theme.text_muted, size=13))),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">
        X = time (quarters), Y = <b>{_dim_label(surf_param)}</b> swept from
        {min(valid_params):.4f} to {max(valid_params):.4f},
        Z = <b>{_dim_label(surf_outcome)}</b>.
        {len(valid_params)} successful parameter values.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN RENDER ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def render_simulation(theme, data=None):
    """
    Render the full simulation card — same pattern as render_causal / render_terrain.
    Called from kshield/page.py router.
    """
    st.markdown(
        '<div class="section-header">'
        'SIMULATIONS &mdash; ECONOMIC SCENARIO ENGINE</div>',
        unsafe_allow_html=True,
    )

    if not HAS_DATA_STACK:
        st.error("Required packages missing: pandas / numpy. Install them for simulation analytics.")
        return

    # ── Nav radio (Workspace vs Guide) ────────────────────────────────────────
    nav_mode = st.radio(
        "Simulation Navigation",
        ["Simulation Workspace", "Guide & Tutorial"],
        horizontal=True,
        key="sim_nav_mode",
    )

    if nav_mode == "Guide & Tutorial":
        _render_simulation_guide(theme)
        return

    # ── Data source radio (matches causal / terrain) ──────────────────────────
    st.markdown("---")
    source = st.radio(
        "Data Source",
        ["World Bank (Kenya)", "Upload your own CSV", "Shared K-SHIELD Dataset"],
        horizontal=True,
        key="sim_data_source",
    )

    df = pd.DataFrame()
    shared_df, shared_meta = _get_shared_dataset()

    if source == "World Bank (Kenya)":
        df = _load_world_bank_data()
        if df.empty:
            st.error(
                "World Bank CSV not found in data/simulation/ or could not be parsed. "
                "Upload a clean CSV to continue."
            )
            return
        _set_shared_dataset(df, "World Bank (Kenya)", "SIMULATION")
    elif source == "Shared K-SHIELD Dataset":
        if shared_df is None or shared_df.empty:
            st.warning("No shared K-SHIELD dataset found yet. "
                       "Load World Bank data or upload a CSV in any K-SHIELD card first.")
            return
        df = shared_df
        st.caption(f"Using shared dataset: {shared_meta}. "
                   f"{len(df.columns)} series across {len(df)} time periods.")
    else:
        uploaded = st.file_uploader("Upload CSV file", type=["csv"], key="sim_upload")
        if uploaded:
            loaded_df, err = _validate_and_load_upload(uploaded)
            if err:
                st.error(err)
                return
            df = loaded_df
            st.session_state["sim_uploaded_df"] = df
            _set_shared_dataset(df, "Uploaded CSV", "SIMULATION")
        elif "sim_uploaded_df" in st.session_state:
            df = st.session_state["sim_uploaded_df"]
        if df.empty:
            return

    # Data profile strip
    _render_data_profile(df, theme)

    # ── Lazy import simulation modules ────────────────────────────────────────
    try:
        from kshiked.simulation.kenya_calibration import (
            calibrate_from_data, OUTCOME_DIMENSIONS, DEFAULT_DIMENSIONS
        )
        from kshiked.simulation.scenario_templates import (
            SCENARIO_LIBRARY, POLICY_TEMPLATES, get_scenario_by_id, build_custom_scenario
        )
        from scarcity.simulation.sfc import SFCEconomy, SFCConfig
    except ImportError as e:
        st.error(f"Simulation engine modules not available: {e}")
        return

    # ── Scenario Configuration expander ───────────────────────────────────────
    scenario, policy_overrides, selected_dims, steps = _render_scenario_config(
        theme, SCENARIO_LIBRARY, POLICY_TEMPLATES,
        get_scenario_by_id, build_custom_scenario,
        OUTCOME_DIMENSIONS, DEFAULT_DIMENSIONS,
    )

    # ── Run button ────────────────────────────────────────────────────────────
    run_clicked = _run_simulation(
        theme, SFCEconomy, SFCConfig,
        calibrate_from_data, build_custom_scenario,
        selected_dims, steps,
    )

    # ── Analysis Tabs ───────────────────────────────────────────────────────
    tabs = st.tabs([
        "Scenario Runner",       # 0
        "Sensitivity Matrix",    # 1
        "3D State Cube",         # 2
        "Compare Runs",          # 3
        "Phase Explorer",        # 4
        "Impulse Response",      # 5
        "Flow Dynamics",         # 6
        "Monte Carlo",           # 7
        "Stress Matrix",         # 8
        "Parameter Surface",     # 9
        "Diagnostics",           # 10
    ])

    with tabs[0]:
        _render_scenario_runner_tab(theme, OUTCOME_DIMENSIONS, DEFAULT_DIMENSIONS, run_clicked)
    with tabs[1]:
        _render_sensitivity_tab(theme, OUTCOME_DIMENSIONS)
    with tabs[2]:
        _render_state_cube_tab(theme, OUTCOME_DIMENSIONS)
    with tabs[3]:
        _render_compare_tab(theme, OUTCOME_DIMENSIONS, DEFAULT_DIMENSIONS)
    with tabs[4]:
        _render_phase_explorer_tab(theme)
    with tabs[5]:
        _render_irf_tab(theme)
    with tabs[6]:
        _render_flow_sankey_tab(theme)
    with tabs[7]:
        _render_monte_carlo_tab(theme, SFCEconomy, SFCConfig,
                                calibrate_from_data, build_custom_scenario)
    with tabs[8]:
        _render_stress_matrix_tab(theme, SCENARIO_LIBRARY,
                                  SFCEconomy, SFCConfig, calibrate_from_data)
    with tabs[9]:
        _render_parameter_surface_tab(theme, SFCEconomy, SFCConfig, calibrate_from_data)
    with tabs[10]:
        _render_diagnostics_tab(theme)
