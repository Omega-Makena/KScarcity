"""
What-If Simulation Workbench

User-driven economic scenario exploration for the SENTINEL dashboard.
Replaces the old Scenario Platform with a guided 3-step flow:
    1. Pick a shock (named scenario or custom)
    2. Build a policy response (template or custom instruments)
    3. Choose dimensions to watch → Run → View results

Architecture:
    kshiked.core.scarcity_bridge (full scarcity access)
    scarcity.simulation.learned_sfc (learned relationships from data)
    kshiked.simulation.fallback_blender (confidence-weighted blending)
    kshiked.simulation.kenya_calibration (data-driven params)
    kshiked.simulation.scenario_templates (named scenarios + policies)
    scarcity.simulation.sfc (parametric fallback)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import streamlit as st
    import streamlit.components.v1 as components
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import numpy as np
except ImportError:
    np = None


def render_whatif_workbench(data: Any, theme: Any, get_flux_graph_html=None):
    """
    Main entry point for the What-If Workbench tab.

    Args:
        data: DashboardData object
        theme: Dashboard theme object with color properties
        get_flux_graph_html: Optional function from flux_viz for 3D flow animation
    """
    if not HAS_STREAMLIT:
        return

    st.markdown('<div class="section-header">WHAT-IF WORKBENCH</div>', unsafe_allow_html=True)

    # --- Lazy imports (K-SHIELD simulation layer) ---
    try:
        from kshiked.simulation.kenya_calibration import (
            calibrate_from_data, OUTCOME_DIMENSIONS, DEFAULT_DIMENSIONS
        )
        from kshiked.simulation.scenario_templates import (
            SCENARIO_LIBRARY, POLICY_TEMPLATES, get_scenario_by_id, build_custom_scenario
        )
        from scarcity.simulation.sfc import SFCEconomy, SFCConfig
    except ImportError as e:
        st.error(f"Simulation modules not available: {e}")
        return

    # Try loading Learned engine
    has_learned = False
    try:
        from kshiked.core.scarcity_bridge import ScarcityBridge
        from scarcity.simulation.learned_sfc import LearnedSFCEconomy, LearnedSFCConfig
        has_learned = True
    except ImportError:
        pass

    # =====================================================================
    # STEP 1: PICK A SHOCK
    # =====================================================================
    _render_shock_picker(theme, SCENARIO_LIBRARY, get_scenario_by_id)

    # =====================================================================
    # STEP 2: BUILD POLICY RESPONSE
    # =====================================================================
    _render_policy_builder(theme, POLICY_TEMPLATES)

    # =====================================================================
    # STEP 3: CHOOSE DIMENSIONS
    # =====================================================================
    selected_dims = _render_dimension_selector(theme, OUTCOME_DIMENSIONS, DEFAULT_DIMENSIONS)

    # =====================================================================
    # RUN BUTTON
    # =====================================================================
    run_clicked = _render_run_controls(
        theme, SFCEconomy, SFCConfig,
        calibrate_from_data, build_custom_scenario,
        selected_dims, has_learned,
    )

    # =====================================================================
    # RESULTS
    # =====================================================================
    _render_results(
        data, theme,
        OUTCOME_DIMENSIONS, DEFAULT_DIMENSIONS,
        get_flux_graph_html, run_clicked,
    )


# =========================================================================
# Component Functions
# =========================================================================

def _render_shock_picker(theme, scenario_library, get_scenario_by_id):
    """Step 1: Scenario / shock selection."""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {theme.bg_secondary}, {theme.bg_tertiary}); 
                padding: 1.2rem; border-radius: 12px; border: 1px solid {theme.border_default};
                margin-bottom: 1rem;">
        <div style="color: {theme.accent_primary}; font-weight: 700; font-size: 0.85rem; 
                    letter-spacing: 1px; margin-bottom: 0.8rem;">
            STEP 1 — PICK A SHOCK
        </div>
    """, unsafe_allow_html=True)

    scenario_options = {s.id: f"{s.name}  ({s.category})" for s in scenario_library}
    scenario_options["custom"] = "Custom Scenario"

    selected_id = st.selectbox(
        "Scenario",
        options=list(scenario_options.keys()),
        format_func=lambda x: scenario_options[x],
        key="whatif_scenario_select",
    )
    st.session_state["_whatif_scenario_id"] = selected_id

    scenario = get_scenario_by_id(selected_id)
    st.session_state["_whatif_scenario_obj"] = scenario

    # Context narrative
    if scenario and scenario.context:
        st.markdown(f"""
        <div style="background: rgba(0,243,255,0.05); border-left: 3px solid {theme.accent_primary}; 
                    padding: 0.8rem 1rem; margin: 0.5rem 0; border-radius: 0 8px 8px 0;
                    font-size: 0.82rem; color: {theme.text_muted};">
            {scenario.context}
        </div>
        """, unsafe_allow_html=True)

    # Custom shock builder
    if selected_id == "custom":
        col_s1, col_s2 = st.columns(2)
        custom_shocks = {}
        with col_s1:
            custom_shocks["demand_shock"] = st.slider(
                "Demand Shock", -0.20, 0.20, 0.0, 0.01,
                help="Negative = contraction, Positive = boom", key="cust_demand"
            )
            custom_shocks["supply_shock"] = st.slider(
                "Supply Shock", -0.05, 0.20, 0.0, 0.01,
                help="Positive = supply disruption", key="cust_supply"
            )
        with col_s2:
            custom_shocks["fiscal_shock"] = st.slider(
                "Fiscal Shock", -0.10, 0.10, 0.0, 0.01,
                help="Positive = spending boost", key="cust_fiscal"
            )
            custom_shocks["fx_shock"] = st.slider(
                "FX / Rate Shock", -0.05, 0.15, 0.0, 0.01,
                help="Positive = depreciation", key="cust_fx"
            )
        custom_shocks = {k: v for k, v in custom_shocks.items() if abs(v) > 0.001}
        st.session_state["_whatif_custom_shocks"] = custom_shocks

        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            st.session_state["_whatif_onset"] = st.number_input("Onset (quarter)", 1, 45, 5, key="cust_onset")
        with col_t2:
            st.session_state["_whatif_duration"] = st.number_input("Duration (0=permanent)", 0, 40, 0, key="cust_dur")
        with col_t3:
            st.session_state["_whatif_shape"] = st.selectbox("Shape", ["step", "pulse", "ramp", "decay"], key="cust_shape")

    st.markdown("</div>", unsafe_allow_html=True)


def _render_policy_builder(theme, policy_templates):
    """Step 2: Policy response builder."""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {theme.bg_secondary}, {theme.bg_tertiary}); 
                padding: 1.2rem; border-radius: 12px; border: 1px solid {theme.border_default};
                margin-bottom: 1rem;">
        <div style="color: {theme.accent_warning}; font-weight: 700; font-size: 0.85rem; 
                    letter-spacing: 1px; margin-bottom: 0.8rem;">
            STEP 2 — BUILD YOUR POLICY RESPONSE
        </div>
    """, unsafe_allow_html=True)

    policy_options = {k: v["name"] for k, v in policy_templates.items()}
    policy_options["custom_manual"] = "Custom (Manual)"

    # Auto-select scenario's suggested policy
    scenario = st.session_state.get("_whatif_scenario_obj")
    default_idx = 0
    if scenario and scenario.suggested_policy:
        for i, (k, v) in enumerate(policy_templates.items()):
            if v["name"] == scenario.suggested_policy.get("name"):
                default_idx = i
                break

    selected_policy_key = st.selectbox(
        "Policy Template",
        options=list(policy_options.keys()),
        format_func=lambda x: policy_options[x],
        index=default_idx,
        key="whatif_policy_select",
    )
    st.session_state["_whatif_policy_key"] = selected_policy_key

    template = policy_templates.get(selected_policy_key, {})
    instruments = template.get("instruments", {})
    policy_overrides = {}

    customize = selected_policy_key == "custom_manual" or st.checkbox(
        "Customize instruments",
        value=(selected_policy_key == "custom_manual"),
        key="customize_policy",
    )

    if customize:
        col_m, col_f = st.columns(2)

        with col_m:
            st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:600; "
                        f"font-size:0.8rem; margin-bottom:0.5rem;'>MONETARY</div>",
                        unsafe_allow_html=True)

            pol_rate = instruments.get("custom_rate")
            policy_overrides["custom_rate"] = st.slider(
                "Central Bank Rate (%)", 0.0, 20.0,
                float(pol_rate) * 100 if pol_rate else 7.0,
                0.25, key="pol_cbr",
            ) / 100.0

            pol_crr = instruments.get("crr")
            policy_overrides["crr"] = st.slider(
                "Cash Reserve Ratio (%)", 0.0, 15.0,
                float(pol_crr) * 100 if pol_crr else 5.25,
                0.25, key="pol_crr",
            ) / 100.0

            rate_cap_on = st.checkbox(
                "Interest Rate Cap",
                value="rate_cap" in instruments,
                key="pol_rate_cap_on",
            )
            if rate_cap_on:
                pol_cap = instruments.get("rate_cap")
                policy_overrides["rate_cap"] = st.slider(
                    "Rate Cap (%)", 5.0, 25.0,
                    float(pol_cap) * 100 if pol_cap else 11.0,
                    0.5, key="pol_rate_cap_val",
                ) / 100.0

        with col_f:
            st.markdown(f"<div style='color:{theme.accent_warning}; font-weight:600; "
                        f"font-size:0.8rem; margin-bottom:0.5rem;'>FISCAL</div>",
                        unsafe_allow_html=True)

            pol_tax = instruments.get("custom_tax_rate")
            policy_overrides["custom_tax_rate"] = st.slider(
                "Tax Rate (%)", 5.0, 30.0,
                float(pol_tax) * 100 if pol_tax else 15.6,
                0.5, key="pol_tax",
            ) / 100.0

            pol_spend = instruments.get("custom_spending_ratio")
            policy_overrides["custom_spending_ratio"] = st.slider(
                "Govt Spending (% GDP)", 5.0, 30.0,
                float(pol_spend) * 100 if pol_spend else 13.0,
                0.5, key="pol_spend",
            ) / 100.0

            pol_sub = instruments.get("subsidy_rate")
            policy_overrides["subsidy_rate"] = st.slider(
                "Subsidies (% GDP)", 0.0, 10.0,
                float(pol_sub) * 100 if pol_sub else 0.8,
                0.1, key="pol_subsidy",
            ) / 100.0

            if st.checkbox("Price Controls", value="price_controls" in instruments, key="pol_pc_on"):
                policy_overrides["price_controls"] = {"fuel": 1.05, "food": 1.03}

        impl_lag = st.number_input("Implementation Lag (quarters)", 0, 10, 0, key="pol_lag")
        policy_overrides["implementation_lag"] = impl_lag
    else:
        policy_overrides = dict(instruments)

    st.session_state["_whatif_policy_overrides"] = policy_overrides
    st.markdown("</div>", unsafe_allow_html=True)


def _render_dimension_selector(theme, outcome_dimensions, default_dimensions):
    """Step 3: Choose which economic dimensions to watch."""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {theme.bg_secondary}, {theme.bg_tertiary}); 
                padding: 1.2rem; border-radius: 12px; border: 1px solid {theme.border_default};
                margin-bottom: 1rem;">
        <div style="color: {theme.accent_success}; font-weight: 700; font-size: 0.85rem; 
                    letter-spacing: 1px; margin-bottom: 0.8rem;">
            STEP 3 — WHAT DO YOU WANT TO WATCH?
        </div>
    """, unsafe_allow_html=True)

    # Group by category
    categories = {}
    for key, meta in outcome_dimensions.items():
        cat = meta.get("category", "Other")
        categories.setdefault(cat, []).append((key, meta))

    scenario = st.session_state.get("_whatif_scenario_obj")
    defaults = (scenario.suggested_dimensions if scenario else None) or list(default_dimensions)

    selected = []
    cols = st.columns(min(len(categories), 5))
    for i, (cat, dims) in enumerate(categories.items()):
        with cols[i % len(cols)]:
            st.markdown(f"<div style='color:{theme.text_muted}; font-weight:600; "
                        f"font-size:0.72rem;'>{cat}</div>", unsafe_allow_html=True)
            for key, meta in dims:
                if st.checkbox(meta["label"], value=key in defaults,
                               key=f"dim_{key}", help=meta["description"]):
                    selected.append(key)

    if not selected:
        selected = list(default_dimensions)

    st.markdown("</div>", unsafe_allow_html=True)
    return selected


def _render_run_controls(theme, SFCEconomy, SFCConfig, calibrate_from_data,
                         build_custom_scenario, selected_dims, has_learned=False):
    """Run button + simulation execution with learned/parametric toggle."""
    col_r1, col_r2, col_r3 = st.columns([2, 1.5, 1])
    with col_r1:
        steps = st.slider("Simulation Quarters", 20, 100, 50, 5, key="sim_steps")
    with col_r2:
        if has_learned:
            sim_mode = st.radio(
                "Engine",
                ["Learned (Scarcity)", "Parametric (SFC)"],
                horizontal=True,
                key="sim_engine_mode",
                help="Learned: uses discovered economic relationships from data. "
                     "Parametric: uses hardcoded SFC equations as fallback.",
            )
        else:
            sim_mode = "Parametric (SFC)"
            st.caption("Engine: Parametric (learned engine not available)")
    with col_r3:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        run_clicked = st.button("RUN SIMULATION", type="primary", use_container_width=True)

    use_learned = has_learned and sim_mode.startswith("Learned")

    if run_clicked:
        spinner_msg = ("Training on historical data and running learned simulation..."
                       if use_learned else "Calibrating from data and running simulation...")
        with st.spinner(spinner_msg):
            try:
                scenario_id = st.session_state.get("_whatif_scenario_id", "custom")
                scenario = st.session_state.get("_whatif_scenario_obj")
                policy_key = st.session_state.get("_whatif_policy_key", "do_nothing")
                policy_overrides = st.session_state.get("_whatif_policy_overrides", {})

                policy_mode = "custom" if policy_key != "do_nothing" else "off"

                # Filter to valid SFCConfig fields
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
                    custom_shocks = st.session_state.get("_whatif_custom_shocks", {})
                    cs = build_custom_scenario(
                        name="Custom", shocks=custom_shocks,
                        shock_onset=st.session_state.get("_whatif_onset", 5),
                        shock_duration=st.session_state.get("_whatif_duration", 0),
                        shock_shape=st.session_state.get("_whatif_shape", "step"),
                        dimensions=selected_dims,
                    )
                    cfg.shock_vectors = cs.build_shock_vectors(steps)
                elif scenario:
                    cfg.shock_vectors = scenario.build_shock_vectors(steps)

                # Run — Learned or Parametric
                if use_learned:
                    from kshiked.core.scarcity_bridge import ScarcityBridge
                    from scarcity.simulation.learned_sfc import LearnedSFCEconomy

                    # Get or create bridge (cached in session)
                    bridge = st.session_state.get("_scarcity_bridge")
                    if bridge is None:
                        bridge = ScarcityBridge()
                        bridge.train()
                        st.session_state["_scarcity_bridge"] = bridge

                    econ = LearnedSFCEconomy(bridge, cfg)
                    econ.initialize()
                    trajectory = econ.run(steps)

                    # Store bridge metadata
                    st.session_state["whatif_engine_mode"] = "learned"
                    st.session_state["whatif_bridge_report"] = bridge.training_report
                    st.session_state["whatif_confidence_map"] = bridge.get_confidence_map()
                else:
                    econ = SFCEconomy(cfg)
                    econ.initialize()
                    trajectory = econ.run(steps)
                    st.session_state["whatif_engine_mode"] = "parametric"

                # Store results
                st.session_state["whatif_trajectory"] = trajectory
                st.session_state["whatif_selected_dims"] = selected_dims
                st.session_state["whatif_calibration"] = calib
                st.session_state["sim_state"] = econ  # Legacy compat

            except Exception as e:
                st.error(f"Simulation error: {e}")
                import traceback
                st.code(traceback.format_exc())

    return run_clicked


def _render_results(data, theme, outcome_dimensions, default_dimensions,
                    get_flux_graph_html, run_clicked):
    """Render simulation results: cards + time-series + flux + heatmap."""
    trajectory = st.session_state.get("whatif_trajectory")
    if not trajectory or len(trajectory) < 2:
        if not run_clicked:
            st.markdown(f"""
            <div style="text-align:center; padding:3rem; color:{theme.text_muted};">
                <div style="font-size:2rem; margin-bottom:0.5rem; opacity:0.3;">▶</div>
                <div style="font-size:0.9rem;">
                    Configure your scenario above and click <b>RUN SIMULATION</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        return

    sel_dims = st.session_state.get("whatif_selected_dims", list(default_dimensions))
    calib = st.session_state.get("whatif_calibration")

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {theme.bg_secondary}, {theme.bg_tertiary}); 
                padding: 1.2rem; border-radius: 12px; border: 1px solid {theme.accent_primary}40;
                margin-top: 1rem;">
        <div style="color: {theme.accent_primary}; font-weight: 700; font-size: 0.85rem; 
                    letter-spacing: 1px; margin-bottom: 0.8rem;">
            SIMULATION RESULTS
        </div>
    """, unsafe_allow_html=True)

    # Engine info badge
    engine_mode = st.session_state.get("whatif_engine_mode", "parametric")
    bridge_report = st.session_state.get("whatif_bridge_report")

    if engine_mode == "learned" and bridge_report:
        overall_conf = bridge_report.overall_confidence
        cc = theme.accent_success if overall_conf > 0.6 else theme.accent_warning
        st.markdown(f"""
        <div style="font-size:0.75rem; color:{theme.text_muted}; margin-bottom:0.8rem;
                    display:flex; gap:1.5rem; flex-wrap:wrap; align-items:center;">
            <span>Engine: <span style="color:{theme.accent_primary}; font-weight:600;">LEARNED</span></span>
            <span>Hypotheses: <span style="color:{cc}; font-weight:600;">{bridge_report.hypotheses_created}</span></span>
            <span>Data confidence: <span style="color:{cc}; font-weight:600;">{overall_conf:.0%}</span></span>
            <span>Training: <span style="font-weight:500;">{bridge_report.years_fed} years</span></span>
        </div>
        """, unsafe_allow_html=True)

        # Show blend ratio from trajectory
        if trajectory:
            avg_blend = sum(f.get("blend_ratio", 0) for f in trajectory) / len(trajectory)
            bc = theme.accent_success if avg_blend > 0.7 else theme.accent_warning if avg_blend > 0.3 else theme.accent_danger
            st.markdown(f"""
            <div style="font-size:0.75rem; color:{theme.text_muted}; margin-bottom:0.8rem;">
                Avg blend ratio: <span style="color:{bc}; font-weight:600;">{avg_blend:.0%} learned</span>
                / <span style="font-weight:500;">{1-avg_blend:.0%} fallback</span>
                &nbsp;— higher = more data-driven
            </div>
            """, unsafe_allow_html=True)
    elif calib:
        conf = calib.overall_confidence
        cc = theme.accent_success if conf > 0.6 else theme.accent_warning if conf > 0.3 else theme.accent_danger
        n_data = sum(1 for p in calib.params.values() if p.source == "data")
        st.markdown(f"""
        <div style="font-size:0.75rem; color:{theme.text_muted}; margin-bottom:0.8rem;">
            Engine: <span style="font-weight:600;">PARAMETRIC</span>
            &nbsp;|&nbsp; Calibration: <span style="color:{cc}; font-weight:600;">{conf:.0%}</span>
            &nbsp;|&nbsp; {n_data}/{len(calib.params)} from data
        </div>
        """, unsafe_allow_html=True)

    # ----- IMPACT DELTA CARDS -----
    _render_impact_cards(trajectory, sel_dims, outcome_dimensions, theme)

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # ----- TIME-SERIES CHART -----
    if HAS_PLOTLY:
        _render_time_series(trajectory, sel_dims, outcome_dimensions, theme)

    # ----- BOTTOM ROW: FLUX + HEATMAP -----
    col_flux, col_heat = st.columns(2)
    with col_flux:
        st.markdown(f"<div style='color:{theme.text_muted}; font-weight:600; "
                    f"font-size:0.8rem; margin-bottom:0.5rem;'>3D ECONOMIC FLUX</div>",
                    unsafe_allow_html=True)
        if get_flux_graph_html:
            try:
                html = get_flux_graph_html(trajectory, height=380)
                components.html(html, height=380)
            except Exception as e:
                st.warning(f"Flux graph error: {e}")
        else:
            st.info("Flux visualization module not loaded.")

    with col_heat:
        st.markdown(f"<div style='color:{theme.text_muted}; font-weight:600; "
                    f"font-size:0.8rem; margin-bottom:0.5rem;'>POLICY SENSITIVITY</div>",
                    unsafe_allow_html=True)
        _render_inline_heatmap(trajectory, theme)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================================
# Rendering Helpers
# =========================================================================

def _render_impact_cards(trajectory, sel_dims, outcome_dimensions, theme):
    """Render impact delta cards for selected dimensions."""
    start = trajectory[0].get("outcomes", {})
    end = trajectory[-1].get("outcomes", {})

    # Render in rows of up to 5
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
            arrow = "▲" if delta > 0 else "▼" if delta < 0 else "—"

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
    """Multi-line time-series chart for selected dimensions."""
    fig = make_subplots(rows=1, cols=1)
    t_vals = [f["t"] for f in trajectory]

    colors = [
        theme.accent_primary, theme.accent_warning, theme.accent_danger,
        theme.accent_success, "#aa00ff", "#ff6b35", "#00ff88",
        "#ff3399", "#66ccff", "#ffcc00", "#ff6666",
    ]

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
            line=dict(color=colors[idx % len(colors)], width=2.5),
            hovertemplate=f"{meta.get('label', dim_key)}: %{{y:.2f}}{suffix}<extra></extra>",
        ))

    # Shock onset marker
    scenario = st.session_state.get("_whatif_scenario_obj")
    onset = scenario.shock_onset if scenario else 5
    fig.add_vline(
        x=onset, line_dash="dash", line_color=theme.accent_danger,
        annotation_text="Shock", annotation_font_color=theme.accent_danger,
    )

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text="Trajectory Over Time", font=dict(color=theme.text_muted, size=13)),
        xaxis=dict(title="Quarter", gridcolor=theme.border_default, color=theme.text_muted),
        yaxis=dict(gridcolor=theme.border_default, color=theme.text_muted),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(color=theme.text_muted, size=11), bgcolor='rgba(0,0,0,0)',
        ),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_inline_heatmap(trajectory, theme):
    """Policy-outcome correlation heatmap from trajectory data."""
    if not HAS_PLOTLY or not np:
        return

    if len(trajectory) < 6:
        st.info("Need more data points for sensitivity analysis.")
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
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': theme.text_muted},
    )
    st.plotly_chart(fig, use_container_width=True)
