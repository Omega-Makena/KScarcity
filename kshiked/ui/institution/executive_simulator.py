"""
Executive Simulation Engine — Dual-Persona Interface (National Gateway)

Composes the full K-SHIELD simulation stack into the Executive Dashboard
with two modes:

  Strategic — Narrative 4-step crisis analysis for decision-makers
  Econometric — Full K-SHIELD research workbench (15+ analytical tabs)

All 7 Scarcity research engines are exposed:
  ResearchSFCEconomy, Bayesian, Financial Accelerator, Heterogeneous,
  Open Economy, IO Structure, WhatIf
"""

import streamlit as st
import traceback
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Theme bridge — K-SHIELD modules expect this ThemeColors object
from kshiked.ui.theme import DARK_THEME

# ═══════════════════════════════════════════════════════════════════════
#  K-SHIELD SIMULATION MODULE IMPORTS
# ═══════════════════════════════════════════════════════════════════════

_KSHIELD_LOADED = False
_KSHIELD_ERROR = None

try:
    from kshiked.ui.kshield.simulation._shared import (
        st as _st, pd, np, go, make_subplots, HAS_PLOTLY,
        PALETTE, base_layout, load_world_bank_data,
    )
    from kshiked.ui.kshield.simulation.scenario_config import render_scenario_config
    from kshiked.ui.kshield.simulation.run import run_simulation, render_scenario_runner_tab
    from kshiked.ui.kshield.simulation.core_analysis import (
        render_sensitivity_tab, render_state_cube_tab,
        render_compare_tab, render_diagnostics_tab,
    )
    from kshiked.ui.kshield.simulation.advanced import (
        render_phase_explorer_tab, render_irf_tab,
        render_flow_sankey_tab, render_monte_carlo_tab,
        render_stress_matrix_tab,
    )
    from kshiked.ui.kshield.simulation.param_surface import render_parameter_surface_tab
    from kshiked.ui.kshield.simulation.research import (
        render_io_sectors_tab, render_inequality_tab,
        render_financial_tab, render_open_economy_tab,
        render_research_engine_tab,
    )
    _KSHIELD_LOADED = True
except ImportError as e:
    _KSHIELD_ERROR = str(e)

# ═══════════════════════════════════════════════════════════════════════
#  RESEARCH ENGINE IMPORTS
# ═══════════════════════════════════════════════════════════════════════

_ENGINE_LOADED = False
_ENGINE_ERROR = None

try:
    from scarcity.simulation.research_sfc import (
        ResearchSFCEconomy, ResearchSFCConfig, default_kenya_research_config
    )
    from scarcity.simulation.sfc import SFCEconomy, SFCConfig
    from scarcity.simulation.io_structure import default_kenya_io_config
    from scarcity.simulation.heterogeneous import (
        default_kenya_heterogeneous_config, InequalityMetrics, IncomeQuintile
    )
    from scarcity.simulation.financial_accelerator import FinancialAcceleratorConfig
    from scarcity.simulation.open_economy import default_kenya_open_economy_config
    from kshiked.simulation.kenya_calibration import (
        calibrate_from_data, OUTCOME_DIMENSIONS, DEFAULT_DIMENSIONS
    )
    from kshiked.simulation.scenario_templates import (
        SCENARIO_LIBRARY, POLICY_TEMPLATES, get_scenario_by_id,
        build_custom_scenario, SHOCK_REGISTRY, POLICY_INSTRUMENT_REGISTRY,
        SHOCK_SHAPES, merge_shock_vectors, merge_policy_instruments,
    )
    _ENGINE_LOADED = True
except ImportError as e:
    _ENGINE_ERROR = str(e)


# ═══════════════════════════════════════════════════════════════════════
#  STRATEGIC MODE — Crisis Presets
# ═══════════════════════════════════════════════════════════════════════

STRATEGIC_SCENARIOS = {
    "Oil Price Spike": {
        "shocks": {"supply_shock": 0.08, "fx_shock": 0.05},
        "context": "Global oil prices surge 30%. Kenya imports ~100% of petroleum — "
                   "transport, manufacturing, and agriculture all face cost pressure.",
        "duration": 6, "shape": "step",
    },
    "Severe Drought": {
        "shocks": {"supply_shock": 0.12, "demand_shock": -0.05},
        "context": "Failed long rains devastate agriculture (~22% of GDP, ~54% employment). "
                   "The 2016-17 drought cut GDP growth by 1.5pp.",
        "duration": 8, "shape": "pulse",
    },
    "Currency Crisis (-15% KES)": {
        "shocks": {"fx_shock": 0.10},
        "context": "KES loses 15% against USD. With ~68% of public debt in foreign currency, "
                   "this inflates the debt stock directly.",
        "duration": 4, "shape": "ramp",
    },
    "Global Recession": {
        "shocks": {"demand_shock": -0.10, "fx_shock": 0.03},
        "context": "Kenya's exports (tea, flowers, textiles) and diaspora remittances (~$4B/yr) "
                   "collapse. The 2008 GFC cut growth from 7.0% to 1.5%.",
        "duration": 0, "shape": "step",
    },
    "Sovereign Debt Crisis": {
        "shocks": {"fiscal_shock": -0.08, "fx_shock": 0.08},
        "context": "Public debt hit 68% of GDP in 2023. A credit downgrade raises borrowing costs "
                   "and triggers capital flight.",
        "duration": 6, "shape": "ramp",
    },
    "Perfect Storm (Drought+Oil+FX)": {
        "shocks": {"supply_shock": 0.15, "demand_shock": -0.05, "fx_shock": 0.10},
        "context": "Simultaneous drought, oil shock, and currency depreciation. "
                   "In 2011 this pushed inflation to 20% and lost KES 25% of value.",
        "duration": 6, "shape": "step",
    },
}

SEVERITY_MULT = {"Mild": 0.5, "Moderate": 1.0, "Severe": 1.5, "Critical": 2.0}

POLICY_RESPONSES = {
    "No Intervention": {
        "mode": "off", "instruments": {},
        "desc": "Let the system absorb the shock naturally.",
    },
    "Measured Response": {
        "mode": "custom",
        "instruments": {"custom_rate": 0.09, "custom_spending_ratio": 0.15, "subsidy_rate": 0.015},
        "desc": "Moderate rate hike + spending boost + targeted subsidies.",
    },
    "Aggressive Intervention": {
        "mode": "custom",
        "instruments": {"custom_rate": 0.12, "crr": 0.075, "custom_spending_ratio": 0.18, "subsidy_rate": 0.03},
        "desc": "Emergency rate hike, CRR increase, fiscal expansion, and subsidies.",
    },
    "IMF Austerity": {
        "mode": "custom",
        "instruments": {"custom_tax_rate": 0.18, "custom_spending_ratio": 0.10, "subsidy_rate": 0.002},
        "desc": "Spending cuts + tax hikes to reduce the deficit.",
    },
}


# ═══════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def render_executive_simulator():
    """Main entry for Policy Simulator tab in the Executive Dashboard."""

    theme = DARK_THEME  # Bridge to K-SHIELD theme system

    if _KSHIELD_ERROR:
        st.warning(f"K-SHIELD modules: {_KSHIELD_ERROR}")
    if _ENGINE_ERROR:
        st.warning(f"Research engine: {_ENGINE_ERROR}")

    mode = st.radio(
        "Interface Mode",
        ["Strategic Crisis Analysis", "Econometric Research Workbench"],
        horizontal=True, key="exec_sim_mode_v3",
        help="Strategic: narrative-driven 4-step crisis analysis. "
             "Econometric: full K-SHIELD research engine with 15+ analytical tools."
    )
    st.markdown("---")

    if mode.startswith("Strategic"):
        _render_strategic_mode(theme)
    else:
        _render_econometric_mode(theme)


# ═══════════════════════════════════════════════════════════════════════
#  STRATEGIC MODE — 4-Step Crisis Analysis
# ═══════════════════════════════════════════════════════════════════════

def _render_strategic_mode(theme):
    """Narrative-driven scenario analysis using the full Research Engine."""

    if not _ENGINE_LOADED:
        st.error("Research engine not available. Cannot run strategic analysis.")
        return

    # ── STEP 1: WHAT HAPPENED? ───────────────────────────────────────
    st.markdown(f"""
    <div style="background:{theme.bg_card}; padding:1.2rem 1.5rem; border-radius:12px;
                border:1px solid {theme.border_default}; margin-bottom:1rem;">
        <div style="color:{theme.accent_primary}; font-weight:700; font-size:0.85rem;
                    letter-spacing:1px; margin-bottom:0.8rem;">
            STEP 1 — WHAT HAPPENED?</div>
    """, unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns([1.5, 1, 1])
    with sc1:
        scenario_name = st.selectbox(
            "Crisis scenario", list(STRATEGIC_SCENARIOS.keys()), key="strat_sc_v3")
    with sc2:
        severity = st.selectbox("Severity", list(SEVERITY_MULT.keys()), index=1, key="strat_sev_v3")
    with sc3:
        horizon = st.selectbox("Horizon", ["Short (20Q)", "Medium (50Q)", "Long (100Q)"],
                                index=1, key="strat_hz_v3")

    scenario = STRATEGIC_SCENARIOS[scenario_name]
    st.markdown(f"""
        <div style="background:rgba(0,255,136,0.04); border-left:3px solid {theme.accent_primary};
                    padding:0.6rem 1rem; border-radius:0 8px 8px 0; margin:0.4rem 0;
                    font-size:0.82rem; color:{theme.text_secondary};">
            {scenario['context']}</div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── STEP 2: WHAT'S THE RESPONSE? ─────────────────────────────────
    st.markdown(f"""
    <div style="background:{theme.bg_card}; padding:1.2rem 1.5rem; border-radius:12px;
                border:1px solid {theme.border_default}; margin-bottom:1rem;">
        <div style="color:{theme.accent_warning}; font-weight:700; font-size:0.85rem;
                    letter-spacing:1px; margin-bottom:0.8rem;">
            STEP 2 — CHOOSE POLICY RESPONSE</div>
    """, unsafe_allow_html=True)

    response_name = st.radio("Policy:", list(POLICY_RESPONSES.keys()),
                              horizontal=True, key="strat_resp_v3")
    resp = POLICY_RESPONSES[response_name]
    st.caption(resp["desc"])
    st.markdown("</div>", unsafe_allow_html=True)

    # ── RUN ───────────────────────────────────────────────────────────
    run = st.button("RUN FULL 7-TIER SIMULATION", type="primary",
                     use_container_width=True, key="strat_run_v3")

    horizon_map = {"Short (20Q)": 20, "Medium (50Q)": 50, "Long (100Q)": 100}
    steps = horizon_map[horizon]
    sev = SEVERITY_MULT[severity]

    if run:
        with st.spinner("Running Research Economy: Financial + Open Economy + IO + Inequality..."):
            try:
                adjusted_shocks = {k: v * sev for k, v in scenario["shocks"].items()}
                cs = build_custom_scenario(
                    name=scenario_name, shocks=adjusted_shocks,
                    shock_onset=3,
                    shock_duration=scenario.get("duration", 0),
                    shock_shape=scenario.get("shape", "step"),
                )
                shock_vecs = cs.build_shock_vectors(steps)

                results = {}
                # Run 3 futures: No Intervention, Selected Policy, Opposite Extreme
                run_configs = [
                    ("no_intervention", "off", {}),
                    ("selected", resp["mode"], resp["instruments"]),
                ]
                # Add a third comparison if the user didn't pick "No Intervention"
                if response_name != "No Intervention":
                    # Add aggressive as third if user chose measured, or vice versa
                    alt = "Aggressive Intervention" if response_name != "Aggressive Intervention" else "Measured Response"
                    alt_r = POLICY_RESPONSES[alt]
                    run_configs.append(("alternative", alt_r["mode"], alt_r["instruments"]))

                progress = st.progress(0, text="Simulating...")
                for idx, (label, pol_mode, instruments) in enumerate(run_configs):
                    cfg = default_kenya_research_config()
                    cfg.sfc.steps = steps
                    cfg.sfc.shock_vectors = shock_vecs
                    cfg.sfc.policy_mode = pol_mode
                    for k, v in instruments.items():
                        if hasattr(cfg.sfc, k):
                            setattr(cfg.sfc, k, v)

                    econ = ResearchSFCEconomy(cfg)
                    econ.initialize()
                    traj = econ.run(steps)

                    results[label] = {
                        "trajectory": traj,
                        "summary": econ.summary(),
                        "stress": econ.stress_test(
                            npl_shock=0.05 * sev, rate_shock=0.02 * sev,
                            fx_shock=0.10 * sev, deposit_run=0.05 * sev),
                        "vulnerability": econ.external_vulnerability_index(),
                        "financial_stability": econ.financial_stability_index(),
                    }
                    progress.progress((idx + 1) / len(run_configs),
                                      text=f"Completed: {label.replace('_', ' ').title()}")
                progress.empty()

                # Store the "selected" result as the research engine result
                # so K-SHIELD tabs can read from it
                sel = results["selected"]
                st.session_state["sim_research_trajectory"] = sel["trajectory"]
                st.session_state["sim_research_econ"] = sel["econ"]
                st.session_state["sim_research_summary"] = sel["summary"]
                st.session_state["strat_results_v3"] = results
                st.session_state["strat_meta_v3"] = {
                    "scenario": scenario_name, "severity": severity,
                    "response": response_name, "steps": steps,
                }
                st.success(f"✓ Simulation complete: {steps} quarters × {len(run_configs)} scenarios")

            except Exception as e:
                st.error(f"Simulation error: {e}")
                st.code(traceback.format_exc())

    # ── RESULTS ──────────────────────────────────────────────────────
    results = st.session_state.get("strat_results_v3")
    meta = st.session_state.get("strat_meta_v3", {})
    if results:
        _render_strategic_results(theme, results, meta)


def _render_strategic_results(theme, results, meta):
    """Render the 4-step strategic analysis output."""

    sel = results.get("selected", {})
    base = results.get("no_intervention", {})
    alt = results.get("alternative", {})
    summary = sel.get("summary", {})

    if not HAS_PLOTLY:
        st.warning("Plotly required for charts.")
        return

    # ── STEP 3: NATIONAL SYSTEM SCAN ─────────────────────────────────
    st.markdown(f"""
    <div style="background:{theme.bg_card}; padding:1.2rem 1.5rem; border-radius:12px;
                border:1px solid {theme.border_default}; margin-bottom:1rem;">
        <div style="color:{theme.accent_danger}; font-weight:700; font-size:0.85rem;
                    letter-spacing:1px; margin-bottom:0.8rem;">
            STEP 3 — WHAT'S AT STAKE?</div>
    """, unsafe_allow_html=True)

    # System Index Cards
    vuln = sel.get("vulnerability", 0)
    fin_stab = sel.get("financial_stability", 0)
    gdp_g = summary.get("gdp_growth", 0) * 100
    infl = summary.get("inflation", 0) * 100
    unemp = summary.get("unemployment", 0) * 100

    def _color(val, good_thresh, bad_thresh, invert=False):
        if invert:
            return PALETTE[0] if val < good_thresh else PALETTE[2] if val < bad_thresh else PALETTE[3]
        return PALETTE[0] if val > good_thresh else PALETTE[2] if val > bad_thresh else PALETTE[3]

    cards = [
        ("EXT. VULNERABILITY", f"{vuln*100:.0f}%", _color(vuln, 0.3, 0.6, True), "0%=safe 100%=critical"),
        ("FIN. STABILITY", f"{fin_stab*100:.0f}%", _color(fin_stab, 0.6, 0.35), "100%=solid"),
        ("GDP GROWTH", f"{gdp_g:+.1f}%", _color(gdp_g, 2, 0), ""),
        ("INFLATION", f"{infl:.1f}%", _color(infl, 5, 10, True), "CBK target: 5±2.5%"),
        ("UNEMPLOYMENT", f"{unemp:.1f}%", _color(unemp, 6, 10, True), ""),
    ]

    cols = st.columns(len(cards))
    for col, (label, value, color, subtext) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div style="background:{theme.bg_tertiary}; padding:1rem; border-radius:10px;
                        border:1px solid {theme.border_default}; text-align:center;">
                <div style="color:{theme.text_muted}; font-size:0.65rem; font-weight:600;
                            letter-spacing:0.5px;">{label}</div>
                <div style="color:{color}; font-size:1.8rem; font-weight:800;">{value}</div>
                <div style="color:{theme.text_muted}; font-size:0.6rem;">{subtext}</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── STEP 4: THREE FUTURES COMPARISON ─────────────────────────────
    st.markdown(f"""
    <div style="background:{theme.bg_card}; padding:1.2rem 1.5rem; border-radius:12px;
                border:1px solid {theme.border_default}; margin-bottom:1rem;">
        <div style="color:{theme.accent_info}; font-weight:700; font-size:0.85rem;
                    letter-spacing:1px; margin-bottom:0.8rem;">
            STEP 4 — COMPARE FUTURES</div>
    """, unsafe_allow_html=True)

    # Multi-panel trajectory comparison
    dims = ["gdp_growth", "inflation", "unemployment", "household_welfare"]
    dim_labels = ["GDP Growth (%)", "Inflation (%)", "Unemployment (%)", "Household Welfare (%)"]
    positions = [(1,1), (1,2), (2,1), (2,2)]

    fig = make_subplots(rows=2, cols=2, subplot_titles=dim_labels,
                        vertical_spacing=0.12, horizontal_spacing=0.08)

    traces_config = [
        ("no_intervention", "No Intervention", PALETTE[3], "dot"),
        ("selected", meta.get("response", "Selected"), PALETTE[0], "solid"),
    ]
    if alt:
        traces_config.append(("alternative", "Alternative", PALETTE[1], "dash"))

    for dim, (r, c) in zip(dims, positions):
        for key, name, color, dash in traces_config:
            traj = results.get(key, {}).get("trajectory", [])
            vals = [f.get("outcomes", {}).get(dim, 0) * 100 for f in traj]
            fig.add_trace(go.Scatter(
                x=list(range(len(vals))), y=vals, name=name,
                line=dict(color=color, width=2.5 if key == "selected" else 2, dash=dash),
                showlegend=(r == 1 and c == 1), legendgroup=key,
            ), row=r, col=c)

    fig.update_layout(**base_layout(theme, height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="center", x=0.5)))
    fig.update_xaxes(title_text="Quarter", gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(title_text="%", gridcolor="rgba(255,255,255,0.05)")
    st.plotly_chart(fig, use_container_width=True)

    # ── STRESS TEST RESULTS ──────────────────────────────────────────
    stress = sel.get("stress", {})
    fin_stress = stress.get("financial", {})
    ext_stress = stress.get("external", {})
    dist_stress = stress.get("distributional", {})

    if fin_stress or ext_stress or dist_stress:
        st.markdown(f"""
        <div style="color:{theme.accent_warning}; font-weight:600; font-size:0.82rem;
                    margin:0.5rem 0;">STRESS TEST: {meta.get('scenario', '')} ({meta.get('severity', '')})</div>
        """, unsafe_allow_html=True)

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            if fin_stress:
                pre_car = fin_stress.get("pre_car", 0) * 100
                post_car = fin_stress.get("post_car", 0) * 100
                breach = fin_stress.get("car_breach", False)
                bc = PALETTE[3] if breach else PALETTE[0]
                st.markdown(f"""
                <div style="background:{theme.bg_tertiary}; padding:1rem; border-radius:10px;
                            border:1px solid {theme.border_default};">
                    <div style="color:{theme.text_muted}; font-size:0.7rem; font-weight:700;">🏦 BANKING</div>
                    <div style="font-size:0.82rem; margin-top:0.4rem; color:{theme.text_secondary};">
                        CAR: {pre_car:.1f}% → <b style="color:{bc};">{post_car:.1f}%</b><br/>
                        Breach: <b style="color:{bc};">{"YES ⚠️" if breach else "NO ✓"}</b><br/>
                        Shortfall: <b>KES {fin_stress.get('capital_shortfall', 0):.0f}B</b>
                    </div>
                </div>""", unsafe_allow_html=True)

        with sc2:
            if ext_stress:
                reserves = ext_stress.get("reserve_adequacy", 0)
                rc = PALETTE[0] if reserves > 4 else PALETTE[2] if reserves > 3 else PALETTE[3]
                st.markdown(f"""
                <div style="background:{theme.bg_tertiary}; padding:1rem; border-radius:10px;
                            border:1px solid {theme.border_default};">
                    <div style="color:{theme.text_muted}; font-size:0.7rem; font-weight:700;">🌍 EXTERNAL</div>
                    <div style="font-size:0.82rem; margin-top:0.4rem; color:{theme.text_secondary};">
                        REER: {ext_stress.get('pre_reer', 100):.1f} → {ext_stress.get('post_reer', 100):.1f}<br/>
                        Trade: {ext_stress.get('post_trade_balance', 0):.1f}<br/>
                        Reserves: <b style="color:{rc};">{reserves:.1f} months</b>
                    </div>
                </div>""", unsafe_allow_html=True)

        with sc3:
            if dist_stress:
                st.markdown(f"""
                <div style="background:{theme.bg_tertiary}; padding:1rem; border-radius:10px;
                            border:1px solid {theme.border_default};">
                    <div style="color:{theme.text_muted}; font-size:0.7rem; font-weight:700;">👥 WHO GETS HURT?</div>
                    <div style="font-size:0.82rem; margin-top:0.4rem; color:{theme.text_secondary};">
                        Poorest 20%: KES {dist_stress.get('q1_income', 0):.1f}<br/>
                        Richest 20%: KES {dist_stress.get('q5_income', 0):.1f}<br/>
                        Rate shock on Q1: <b style="color:{PALETTE[3]};">-{dist_stress.get('rate_shock_q1_impact', 0):.2f}</b>
                    </div>
                </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── DEEP-DIVE TABS (Reuse K-SHIELD Research Modules) ─────────────
    if _KSHIELD_LOADED:
        st.markdown(f"""
        <div style="color:{theme.accent_primary}; font-weight:700; font-size:0.85rem;
                    letter-spacing:1px; margin:1rem 0 0.5rem;">
            DEEP-DIVE ANALYSIS</div>
        """, unsafe_allow_html=True)

        deep_tabs = st.tabs([
            "IO Sectors", "Inequality", "Financial", "Open Economy",
            "Research Radar",
        ])
        with deep_tabs[0]:
            render_io_sectors_tab(theme)
        with deep_tabs[1]:
            render_inequality_tab(theme)
        with deep_tabs[2]:
            render_financial_tab(theme)
        with deep_tabs[3]:
            render_open_economy_tab(theme)
        with deep_tabs[4]:
            render_research_engine_tab(theme, SFCEconomy, SFCConfig, calibrate_from_data)


# ═══════════════════════════════════════════════════════════════════════
#  ECONOMETRIC MODE — Full K-SHIELD Workbench
# ═══════════════════════════════════════════════════════════════════════

def _render_econometric_mode(theme):
    """Full K-SHIELD simulation workbench ported to executive dashboard."""

    if not _KSHIELD_LOADED or not _ENGINE_LOADED:
        st.error("K-SHIELD simulation modules or Research Engine not available.")
        if _KSHIELD_ERROR:
            st.code(_KSHIELD_ERROR)
        if _ENGINE_ERROR:
            st.code(_ENGINE_ERROR)
        return

    # ── DATA SOURCE ──────────────────────────────────────────────────
    df = load_world_bank_data()
    if df is not None and not df.empty:
        st.markdown(f"""
        <div style="font-size:0.7rem; color:{theme.text_muted}; padding:0 0 0.5rem;">
            Data: World Bank Kenya | {df.shape[0]} rows | {df.shape[1]} vars |
            {df.index.min()}-{df.index.max()}</div>
        """, unsafe_allow_html=True)

    # ── CATEGORY NAVIGATION ──────────────────────────────────────────
    category = st.radio(
        "nav",
        ["Setup & Run", "Core Analysis", "Advanced", "Research Modules"],
        horizontal=True, key="econ_category_v3", label_visibility="collapsed",
    )
    st.markdown(f"<div style='border-top:1px solid {theme.border_default}; "
                f"margin:-0.5rem 0 0.8rem;'></div>", unsafe_allow_html=True)

    # ── CATEGORY 1: SETUP & RUN ──────────────────────────────────────
    if category == "Setup & Run":
        scenario_cfg = render_scenario_config(
            theme, SCENARIO_LIBRARY, POLICY_TEMPLATES,
            get_scenario_by_id, build_custom_scenario,
            OUTCOME_DIMENSIONS, DEFAULT_DIMENSIONS,
            SHOCK_REGISTRY, POLICY_INSTRUMENT_REGISTRY, SHOCK_SHAPES,
        )
        run_clicked = run_simulation(
            theme, SFCEconomy, SFCConfig,
            calibrate_from_data, scenario_cfg,
            merge_shock_vectors, merge_policy_instruments,
        )

        setup_tabs = st.tabs(["Results", "Research Engine"])
        with setup_tabs[0]:
            render_scenario_runner_tab(theme, OUTCOME_DIMENSIONS, DEFAULT_DIMENSIONS, run_clicked)
        with setup_tabs[1]:
            render_research_engine_tab(theme, SFCEconomy, SFCConfig, calibrate_from_data)

    # ── CATEGORY 2: CORE ANALYSIS ────────────────────────────────────
    elif category == "Core Analysis":
        if not st.session_state.get("sim_trajectory"):
            st.info("Run a simulation in **Setup & Run** first.")
            return
        core_tabs = st.tabs([
            "Sensitivity", "3D State Cube", "Scenario Compare",
            "Phase Diagram", "Impulse Response", "Flow Dynamics",
        ])
        with core_tabs[0]:
            render_sensitivity_tab(theme, OUTCOME_DIMENSIONS)
        with core_tabs[1]:
            render_state_cube_tab(theme, OUTCOME_DIMENSIONS)
        with core_tabs[2]:
            render_compare_tab(theme, OUTCOME_DIMENSIONS, DEFAULT_DIMENSIONS)
        with core_tabs[3]:
            render_phase_explorer_tab(theme)
        with core_tabs[4]:
            render_irf_tab(theme)
        with core_tabs[5]:
            render_flow_sankey_tab(theme)

    # ── CATEGORY 3: ADVANCED ─────────────────────────────────────────
    elif category == "Advanced":
        if not st.session_state.get("sim_trajectory"):
            st.info("Run a simulation in **Setup & Run** first.")
            return
        adv_tabs = st.tabs([
            "Monte Carlo", "Stress Matrix", "Parameter Surface", "Diagnostics",
        ])
        with adv_tabs[0]:
            render_monte_carlo_tab(theme, SFCEconomy, SFCConfig,
                                    calibrate_from_data, build_custom_scenario)
        with adv_tabs[1]:
            render_stress_matrix_tab(theme, SCENARIO_LIBRARY,
                                      SFCEconomy, SFCConfig, calibrate_from_data)
        with adv_tabs[2]:
            render_parameter_surface_tab(theme, SFCEconomy, SFCConfig, calibrate_from_data)
        with adv_tabs[3]:
            render_diagnostics_tab(theme)

    # ── CATEGORY 4: RESEARCH MODULES ─────────────────────────────────
    elif category == "Research Modules":
        res_tabs = st.tabs([
            "IO Sectors", "Inequality", "Financial", "Open Economy",
        ])
        with res_tabs[0]:
            render_io_sectors_tab(theme)
        with res_tabs[1]:
            render_inequality_tab(theme)
        with res_tabs[2]:
            render_financial_tab(theme)
        with res_tabs[3]:
            render_open_economy_tab(theme)
