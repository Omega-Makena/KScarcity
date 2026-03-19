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
# K-SHIELD SIMULATION MODULE IMPORTS
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
# RESEARCH ENGINE IMPORTS
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
# STRATEGIC MODE — Crisis Presets
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
# BOOTSTRAP HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _run_bootstrap_bundle(base_sfc_cfg, shock_vecs, steps, n_runs=12, jitter_pct=8):
  """Run n_runs jittered SFCEconomy simulations for confidence bands."""
  import numpy as np
  if not _ENGINE_LOADED:
    return []
  bundles = []
  for _ in range(n_runs):
    try:
      cfg_dict = {}
      for field_name in SFCConfig.__dataclass_fields__:
        val = getattr(base_sfc_cfg, field_name, None)
        if isinstance(val, (int, float)) and field_name not in ("steps", "dt"):
          j = 1.0 + np.random.uniform(-jitter_pct / 100, jitter_pct / 100)
          cfg_dict[field_name] = val * j
        elif val is not None:
          cfg_dict[field_name] = val
      cfg = SFCConfig(**{k: v for k, v in cfg_dict.items()
                         if k in SFCConfig.__dataclass_fields__})
      cfg.steps = steps
      cfg.shock_vectors = shock_vecs
      econ = SFCEconomy(cfg)
      econ.initialize()
      bundles.append(econ.run(steps))
    except Exception:
      pass
  return bundles


def _extract_bands(bundles, dim, pcts=(25, 75)):
  """Extract per-timepoint percentile bands for an outcome dimension (already ×100)."""
  import numpy as np
  if not bundles:
    return {}
  T = max(len(b) for b in bundles)
  arr = np.array([
    [b[t].get("outcomes", {}).get(dim, 0) * 100 if t < len(b) else 0
     for t in range(T)]
    for b in bundles
  ])
  return {p: np.percentile(arr, p, axis=0).tolist() for p in pcts}


# Per-scenario band fill colours (semi-transparent)
_BAND_RGBA = {
  "no_intervention": "rgba(255,80,80,0.13)",
  "selected":        "rgba(0,255,136,0.13)",
  "alternative":     "rgba(80,180,255,0.13)",
}


# ═══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

def render_executive_simulator():
  """Main entry for Policy Simulator tab in the Executive Dashboard."""

  theme = DARK_THEME # Bridge to K-SHIELD theme system

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
# STRATEGIC MODE — 4-Step Crisis Analysis
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
            "econ": econ,
            "bootstrap": _run_bootstrap_bundle(cfg.sfc, shock_vecs, steps,
                                               n_runs=12, jitter_pct=8),
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
        st.success(f"Simulation complete: {steps} quarters × {len(run_configs)} scenarios")

      except Exception as e:
        st.error(f"Simulation error: {e}")
        st.code(traceback.format_exc())

  # ── RESULTS ──────────────────────────────────────────────────────
  results = st.session_state.get("strat_results_v3")
  meta = st.session_state.get("strat_meta_v3", {})
  if results:
    _render_strategic_results(theme, results, meta)


def _render_distributional_panel(theme, results, traces_config):
  """Render 'Who Bears the Cost?' distributional impact section."""
  import numpy as np

  # Check inequality data is present in at least one trajectory
  has_ineq = any(
    any(f.get("inequality") for f in results.get(k, {}).get("trajectory", []))
    for k, *_ in traces_config
  )
  if not has_ineq or not HAS_PLOTLY:
    return

  st.markdown(f"""
  <div style="background:{theme.bg_card}; padding:1.2rem 1.5rem; border-radius:12px;
        border:1px solid {theme.border_default}; margin:1rem 0;">
    <div style="color:{theme.accent_warning}; font-weight:700; font-size:0.85rem;
          letter-spacing:1px; margin-bottom:0.8rem;">
      WHO BEARS THE COST?</div>
    <div style="font-size:0.78rem; color:{theme.text_muted}; margin-bottom:0.8rem;">
      How does each policy choice affect different income groups over the simulation horizon.
    </div>
  """, unsafe_allow_html=True)

  quintile_keys = ["q1_bottom_20", "q2_lower_20", "q3_middle_20", "q4_upper_20", "q5_top_20"]
  q_labels = ["Bottom 20%", "Lower 20%", "Middle 20%", "Upper 20%", "Top 20%"]

  # ── Panel 1: Q1 vs Q5 income trajectories ──────────────────────────────
  col_left, col_right = st.columns(2)

  with col_left:
    fig_q = go.Figure()
    line_styles = {"no_intervention": "dot", "selected": "solid", "alternative": "dash"}
    for key, name, color, _ in traces_config:
      traj = results.get(key, {}).get("trajectory", [])
      t_vals = list(range(len(traj)))
      q1_vals = [f.get("inequality", {}).get("quintile_incomes", {}).get("q1_bottom_20", 0) for f in traj]
      q5_vals = [f.get("inequality", {}).get("quintile_incomes", {}).get("q5_top_20", 0) for f in traj]
      dash = line_styles.get(key, "solid")
      fig_q.add_trace(go.Scatter(
        x=t_vals, y=q1_vals, name=f"Q1 · {name}",
        line=dict(color=PALETTE[3], width=1.5, dash=dash),
        legendgroup=f"q1_{key}",
      ))
      fig_q.add_trace(go.Scatter(
        x=t_vals, y=q5_vals, name=f"Q5 · {name}",
        line=dict(color=color, width=2, dash=dash),
        legendgroup=f"q5_{key}",
      ))
    fig_q.update_layout(**base_layout(theme, height=300,
      title=dict(text="Bottom 20% vs Top 20% Income",
                 font=dict(color=theme.text_muted, size=12)),
      xaxis=dict(title="Quarter"), yaxis=dict(title="Income"),
      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                  bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
    ))
    st.plotly_chart(fig_q, use_container_width=True)

  with col_right:
    # Pro-poor growth bar: Q1 vs Q5 growth % per scenario
    bar_names, q1_growths, q5_growths, bar_colors = [], [], [], []
    for key, name, color, _ in traces_config:
      traj = results.get(key, {}).get("trajectory", [])
      if len(traj) < 2:
        continue
      first_ineq = next((f.get("inequality", {}).get("quintile_incomes", {}) for f in traj if f.get("inequality")), {})
      last_ineq = next((f.get("inequality", {}).get("quintile_incomes", {}) for f in reversed(traj) if f.get("inequality")), {})
      q1_s = first_ineq.get("q1_bottom_20", 1) or 1
      q5_s = first_ineq.get("q5_top_20", 1) or 1
      q1_g = (last_ineq.get("q1_bottom_20", q1_s) - q1_s) / abs(q1_s) * 100
      q5_g = (last_ineq.get("q5_top_20", q5_s) - q5_s) / abs(q5_s) * 100
      bar_names.append(name)
      q1_growths.append(q1_g)
      q5_growths.append(q5_g)
      bar_colors.append(color)

    if bar_names:
      fig_bar = go.Figure()
      fig_bar.add_trace(go.Bar(
        name="Bottom 20% growth", x=bar_names, y=q1_growths,
        marker_color=PALETTE[3], text=[f"{v:+.1f}%" for v in q1_growths],
        textposition="outside",
      ))
      fig_bar.add_trace(go.Bar(
        name="Top 20% growth", x=bar_names, y=q5_growths,
        marker_color=PALETTE[1], text=[f"{v:+.1f}%" for v in q5_growths],
        textposition="outside",
      ))
      fig_bar.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_width=1)
      fig_bar.update_layout(**base_layout(theme, height=300,
        title=dict(text="Income Growth by Policy (Q1 vs Q5)",
                   font=dict(color=theme.text_muted, size=12)),
        barmode="group", xaxis=dict(title="Policy"),
        yaxis=dict(title="Growth (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
      ))
      st.plotly_chart(fig_bar, use_container_width=True)

  # ── Panel 2: Gini trajectory ────────────────────────────────────────────
  fig_gini = go.Figure()
  for key, name, color, _ in traces_config:
    traj = results.get(key, {}).get("trajectory", [])
    gini_vals = [f.get("inequality", {}).get("gini", None) for f in traj]
    valid = [(t, g) for t, g in enumerate(gini_vals) if g is not None]
    if valid:
      t_g, g_g = zip(*valid)
      fig_gini.add_trace(go.Scatter(
        x=list(t_g), y=list(g_g), name=name,
        line=dict(color=color, width=2),
        fill='tozeroy',
        fillcolor=_BAND_RGBA.get(key, "rgba(128,128,128,0.08)"),
      ))
  # Kenya benchmark
  if results:
    max_t = max(len(results.get(k, {}).get("trajectory", [])) for k, *_ in traces_config)
    if max_t > 0:
      fig_gini.add_hline(y=0.408, line_dash="dot",
                         line_color="rgba(255,255,255,0.4)", line_width=1,
                         annotation_text="Kenya benchmark 0.408",
                         annotation_position="bottom right",
                         annotation_font=dict(color=theme.text_muted, size=10))
  fig_gini.update_layout(**base_layout(theme, height=240,
    title=dict(text="Gini Coefficient — All Scenarios",
               font=dict(color=theme.text_muted, size=12)),
    xaxis=dict(title="Quarter"), yaxis=dict(title="Gini", range=[0, 0.7]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
  ))
  st.plotly_chart(fig_gini, use_container_width=True)

  # ── Panel 3: Verdict cards ──────────────────────────────────────────────
  verdict_cols = st.columns(len(traces_config))
  for col, (key, name, color, _) in zip(verdict_cols, traces_config):
    traj = results.get(key, {}).get("trajectory", [])
    first_ineq = next((f.get("inequality", {}).get("quintile_incomes", {}) for f in traj if f.get("inequality")), {})
    last_ineq = next((f.get("inequality", {}).get("quintile_incomes", {}) for f in reversed(traj) if f.get("inequality")), {})
    first_gini = next((f.get("inequality", {}).get("gini") for f in traj if f.get("inequality")), None)
    last_gini = next((f.get("inequality", {}).get("gini") for f in reversed(traj) if f.get("inequality")), None)

    if first_ineq and last_ineq:
      q1_s = first_ineq.get("q1_bottom_20", 1) or 1
      q5_s = first_ineq.get("q5_top_20", 1) or 1
      q1_g = (last_ineq.get("q1_bottom_20", q1_s) - q1_s) / abs(q1_s) * 100
      q5_g = (last_ineq.get("q5_top_20", q5_s) - q5_s) / abs(q5_s) * 100

      diff = q1_g - q5_g
      if diff > 0.5:
        verdict, v_color = "PRO-POOR", PALETTE[0]
      elif diff < -0.5:
        verdict, v_color = "REGRESSIVE", PALETTE[3]
      else:
        verdict, v_color = "NEUTRAL", PALETTE[2]

      gini_dir = ""
      if first_gini is not None and last_gini is not None:
        delta_g = last_gini - first_gini
        gini_dir = "↑ worsens" if delta_g > 0.005 else "↓ improves" if delta_g < -0.005 else "→ stable"

      with col:
        st.markdown(f"""
        <div style="background:{theme.bg_tertiary}; padding:1rem; border-radius:10px;
              border:1px solid {theme.border_default}; text-align:center;">
          <div style="color:{theme.text_muted}; font-size:0.65rem; font-weight:600;
                letter-spacing:0.5px; margin-bottom:0.3rem;">{name.upper()}</div>
          <div style="color:{v_color}; font-size:1.1rem; font-weight:800;
                margin-bottom:0.4rem;">{verdict}</div>
          <div style="color:{theme.text_secondary}; font-size:0.72rem; line-height:1.6;">
            Bottom 20%: <b>{q1_g:+.1f}%</b><br/>
            Top 20%: <b>{q5_g:+.1f}%</b><br/>
            Gini: <span style="color:{PALETTE[2] if 'stable' in gini_dir else PALETTE[0] if 'improves' in gini_dir else PALETTE[3]};">{gini_dir}</span>
          </div>
        </div>""", unsafe_allow_html=True)

  st.markdown("</div>", unsafe_allow_html=True)


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
    ("VULNERABILITY", f"{vuln*100:.0f}%", _color(vuln, 0.3, 0.6, True), "Risk of external currency/debt crisis"),
    ("STABILITY", f"{fin_stab*100:.0f}%", _color(fin_stab, 0.6, 0.35), "Banking sector resilience"),
    ("GROWTH", f"{gdp_g:+.1f}%", _color(gdp_g, 2, 0), "Economic expansion/contraction"),
    ("PRICES", f"{infl:.1f}%", _color(infl, 5, 10, True), "Cost of living increase"),
    ("JOBS", f"{unemp:.1f}%", _color(unemp, 6, 10, True), "Unemployment rate"),
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
      
  # Add plain language summary for the system scan
  system_health_text = []
  if gdp_g < 0:
    system_health_text.append("The economy will likely fall into a **recession**.")
  if infl > 8:
    system_health_text.append("Citizens will face **severe cost-of-living spikes**.")
  if unemp > 8:
    system_health_text.append("**Significant job losses** are expected.")
  if vuln > 0.6:
    system_health_text.append("The nation is at high risk of a **currency or debt crisis**.")
  if fin_stab < 0.35:
    system_health_text.append("The banking sector is dangerously **fragile and prone to failure**.")
    
  if not system_health_text:
    system_health_text.append("The economic foundations remain generally stable under this scenario.")
    
  st.markdown(f"""
  <div style="background:#F8FAFC; border-left:4px solid {theme.accent_primary}; padding:12px 16px; margin-top:10px; border-radius:0 4px 4px 0;">
    <strong>Plain-Language Verdict:</strong> {" ".join(system_health_text)}
  </div>
  """, unsafe_allow_html=True)
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
    # First pass: add bootstrap confidence bands (behind the main lines)
    for key, name, color, dash in traces_config:
      bundles = results.get(key, {}).get("bootstrap", [])
      bands = _extract_bands(bundles, dim) if bundles else {}
      if 25 in bands and 75 in bands:
        t_b = list(range(len(bands[25])))
        fill_color = _BAND_RGBA.get(key, "rgba(128,128,128,0.12)")
        fig.add_trace(go.Scatter(
          x=t_b + t_b[::-1],
          y=bands[75] + bands[25][::-1],
          fill='toself', fillcolor=fill_color,
          line=dict(color='rgba(0,0,0,0)'),
          showlegend=False, hoverinfo='skip', legendgroup=key,
        ), row=r, col=c)
    # Second pass: main deterministic lines on top
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

  has_bands = any(results.get(k, {}).get("bootstrap") for k, *_ in traces_config)
  if has_bands:
    st.markdown(f"""
    <div style="font-size:0.72rem; color:{theme.text_muted}; margin:-0.5rem 0 0.8rem;">
      Shaded bands = 25th–75th percentile across 12 parameter-jittered runs
    </div>""", unsafe_allow_html=True)

  # Simple plain-language table comparing the three futures at the end of the horizon
  st.markdown("#### The Three Futures — Executive Summary")
  st.write("Where does the nation stand at the end of this timeline?")
  
  comparisons = []
  for key, name, color, dash in traces_config:
    traj = results.get(key, {}).get("trajectory", [])
    if traj:
      final_state = traj[-1].get("outcomes", {})
      f_gdp = final_state.get("gdp_growth", 0) * 100
      f_infl = final_state.get("inflation", 0) * 100
      f_unemp = final_state.get("unemployment", 0) * 100
      
      summary_statement = []
      if f_gdp > 2 and f_infl < 6 and f_unemp < 8:
        summary_statement.append("Strong recovery")
      elif f_gdp < 0:
        summary_statement.append("Deep recession")
      else:
        summary_statement.append("Stagnation / Slow grind")
        
      if f_infl > 8:
        summary_statement.append("with painful inflation")
      elif f_unemp > 10:
        summary_statement.append("with mass unemployment")
        
      comparisons.append({
        "Policy Choice": name,
        "Final Economic State": " ".join(summary_statement),
        "Growth": f"{f_gdp:+.1f}%",
        "Inflation": f"{f_infl:.1f}%",
        "Joblessness": f"{f_unemp:.1f}%"
      })
      
  if comparisons:
    import pandas as pd
    st.dataframe(pd.DataFrame(comparisons), use_container_width=True, hide_index=True)

  # ── WHO BEARS THE COST? ──────────────────────────────────────────
  _render_distributional_panel(theme, results, traces_config)

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
          <div style="color:{theme.text_muted}; font-size:0.7rem; font-weight:700;"> BANKING SYSTEM</div>
          <div style="font-size:0.82rem; margin-top:0.4rem; color:{theme.text_secondary};">
            Capital Ratio: {pre_car:.1f}% → <b style="color:{bc};">{post_car:.1f}%</b><br/>
            Systemic Failure Risk: <b style="color:{bc};">{"HIGH " if breach else "LOW "}</b><br/>
            Capital Shortfall: <b>KES {fin_stress.get('capital_shortfall', 0):.0f}B</b><br/><br/>
            <i>{'Banks fail regulatory minimums; bailouts may be required.' if breach else 'Banks absorb the shock without failing.'}</i>
          </div>
        </div>""", unsafe_allow_html=True)

    with sc2:
      if ext_stress:
        reserves = ext_stress.get("reserve_adequacy", 0)
        rc = PALETTE[0] if reserves > 4 else PALETTE[2] if reserves > 3 else PALETTE[3]
        st.markdown(f"""
        <div style="background:{theme.bg_tertiary}; padding:1rem; border-radius:10px;
              border:1px solid {theme.border_default};">
          <div style="color:{theme.text_muted}; font-size:0.7rem; font-weight:700;"> EXTERNAL</div>
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
          <div style="color:{theme.text_muted}; font-size:0.7rem; font-weight:700;"> SOCIAL IMPACT</div>
          <div style="font-size:0.82rem; margin-top:0.4rem; color:{theme.text_secondary};">
            Poorest 20%: KES {dist_stress.get('q1_income', 0):.1f}<br/>
            Richest 20%: KES {dist_stress.get('q5_income', 0):.1f}<br/>
            Pain on Poorest: <b style="color:{PALETTE[3]};">{(dist_stress.get('rate_shock_q1_impact', 0) * 100):.1f}% drop</b><br/><br/>
            <i>Measures the disproportionate economic pain felt by the most vulnerable citizens.</i>
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
      "Research Radar", "Causal Estimands",
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
    with deep_tabs[5]:
      _render_causal_estimands_deep(theme)


# ═══════════════════════════════════════════════════════════════════════
# ECONOMETRIC MODE — Full K-SHIELD Workbench
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
      "IO Sectors", "Inequality", "Financial", "Open Economy", "Causal Estimands"
    ])
    with res_tabs[0]:
      render_io_sectors_tab(theme)
    with res_tabs[1]:
      render_inequality_tab(theme)
    with res_tabs[2]:
      render_financial_tab(theme)
    with res_tabs[3]:
      render_open_economy_tab(theme)
    with res_tabs[4]:
      _render_causal_estimands_deep(theme)
    # with res_tabs[4]:
    #   from kshiked.ui.institution.backend.research_engine import ResearchEngine, EngineContext
    #   from kshiked.ui.institution.research_components import render_research_engine_panel
    #
    #   ctx = EngineContext(role="executive", user_id=st.session_state.get('username'), sector_id=None)
    #   engine = ResearchEngine(context=ctx)
    #   render_research_engine_panel(engine, theme)


# ═══════════════════════════════════════════════════════════════════════
# CAUSAL ESTIMANDS — Deep visualisation (2D + 3D)
# ═══════════════════════════════════════════════════════════════════════

def _render_causal_estimands_deep(theme):
  """
  Detailed causal estimand analysis with 2D and 3D Plotly visualisations.

  Panels
  ──────
  1. Variable picker   — choose treatment, outcome, optional confounder
  2. 2D: ATE multi-pair  — bar chart of ATE for top-N treatment→outcome pairs
  3. 2D: Confidence bands — point estimate + 95 % CI error bars
  4. 2D: Dose-response  — binned treatment vs conditional mean outcome
  5. 3D: ATE surface   — ATE over a grid of (treatment quantile, window size)
               computed via rolling-window OLS (fast, no DoWhy needed)
  6. Structural estimates — optional DoWhy/EconML panel via render_causal_evidence_panel
  """
  if not HAS_PLOTLY:
    st.warning("Plotly is required for causal visualisations.")
    return

  try:
    import numpy as _np
    import pandas as _pd2
  except ImportError:
    st.error("numpy and pandas are required.")
    return

  df = load_world_bank_data()
  if df is None or df.empty:
    st.info("World Bank data not loaded. Go to **Setup & Run** and select 'World Bank (Kenya)' as data source.")
    return

  num_cols = [c for c in df.columns if df[c].notna().sum() >= 15]
  if len(num_cols) < 2:
    st.info("Not enough numeric columns with sufficient observations.")
    return

  # ── VARIABLE SELECTION ────────────────────────────────────────────
  st.markdown(
    f'<div style="color:{theme.accent_primary}; font-weight:700; font-size:0.85rem; '
    f'letter-spacing:1px; margin-bottom:0.6rem;">CAUSAL ESTIMANDS — VARIABLE SELECTION</div>',
    unsafe_allow_html=True,
  )

  # Smart defaults: prefer well-known economic pairs
  _priority_t = [c for c in num_cols if any(k in c.lower() for k in ("inflation", "interest", "gdp", "tax", "fuel"))]
  _priority_o = [c for c in num_cols if any(k in c.lower() for k in ("unemployment", "poverty", "growth", "gini", "consumption"))]
  default_t_idx = num_cols.index(_priority_t[0]) if _priority_t else 0
  default_o_idx = next((i for i, c in enumerate(num_cols) if c != num_cols[default_t_idx] and (not _priority_o or c in _priority_o)), 1)

  vc1, vc2, vc3 = st.columns(3)
  with vc1:
    treatment = st.selectbox("Treatment (cause)", num_cols, index=default_t_idx, key="ced_treatment")
  with vc2:
    out_opts = [c for c in num_cols if c != treatment]
    outcome = st.selectbox("Outcome (effect)", out_opts, index=min(default_o_idx, len(out_opts) - 1), key="ced_outcome")
  with vc3:
    conf_opts = ["(none)"] + [c for c in num_cols if c not in (treatment, outcome)]
    confounder = st.selectbox("Confounder (control)", conf_opts, index=0, key="ced_confounder")
    confounder = None if confounder == "(none)" else confounder

  n_pairs = st.slider("Top-N treatment→outcome pairs for multi-pair chart", 3, 12, 6, key="ced_npairs")

  st.markdown(f'<div style="border-top:1px solid {theme.border_default}; margin:0.4rem 0 0.8rem;"></div>', unsafe_allow_html=True)

  # ── SHARED DATA PREP ─────────────────────────────────────────────
  ctrl_cols = [confounder] if confounder else []
  work_cols = [treatment, outcome] + ctrl_cols
  clean = df[work_cols].dropna()

  if len(clean) < 10:
    st.warning(f"Only {len(clean)} complete rows — need at least 10.")
    return

  # Fast OLS-based ATE: coefficient on treatment in linear regression
  def _ols_ate(df_, t_col, y_col, ctrl=None):
    """Return (coef, std_err, n) from OLS y ~ t [+ ctrl]."""
    import numpy as _n
    X = df_[[t_col] + (ctrl or [])].copy()
    X.insert(0, "_const", 1.0)
    y = df_[y_col].values
    try:
      beta, *_ = _n.linalg.lstsq(X.values, y, rcond=None)
      resid = y - X.values @ beta
      n, k = X.shape
      s2 = (resid @ resid) / max(n - k, 1)
      XtX_inv = _n.linalg.pinv(X.values.T @ X.values)
      se = _n.sqrt(_n.maximum(0, _n.diag(XtX_inv) * s2))
      return float(beta[1]), float(se[1]), int(n)
    except Exception:
      return 0.0, 0.0, 0

  # ── PANEL 1 — MULTI-PAIR ATE BAR CHART (2D) ──────────────────────
  st.markdown(
    f'<div style="color:{theme.text_secondary}; font-weight:600; font-size:0.82rem; '
    f'margin-bottom:0.4rem;">① MULTI-PAIR AVERAGE TREATMENT EFFECTS</div>',
    unsafe_allow_html=True,
  )
  st.caption(
    "OLS slope of outcome on treatment (controlling for the selected confounder). "
    "Positive = treatment increases outcome. Bars are coloured by direction."
  )

  pair_results = []
  for t_col in num_cols[:20]:    # scan up to 20 treatments
    for y_col in num_cols[:20]:
      if t_col == y_col:
        continue
      _d = df[[t_col, y_col] + ctrl_cols].dropna()
      if len(_d) < 10:
        continue
      coef, se, n = _ols_ate(_d, t_col, y_col, ctrl_cols)
      pair_results.append({"Treatment": t_col[:35], "Outcome": y_col[:35],
                 "ATE": coef, "SE": se, "N": n})
    if len(pair_results) >= n_pairs * 4:
      break

  if pair_results:
    pair_df = _pd2.DataFrame(pair_results)
    pair_df["abs_ATE"] = pair_df["ATE"].abs()
    top_pairs = pair_df.nlargest(n_pairs, "abs_ATE")
    top_pairs["Label"] = top_pairs["Treatment"].str[:20] + " → " + top_pairs["Outcome"].str[:20]
    colors = [PALETTE[0] if v >= 0 else PALETTE[3] for v in top_pairs["ATE"]]

    fig_bar = go.Figure(go.Bar(
      x=top_pairs["Label"], y=top_pairs["ATE"],
      error_y=dict(type="data", array=top_pairs["SE"].tolist(), visible=True,
             color="rgba(255,255,255,0.4)", thickness=1.5, width=4),
      marker_color=colors,
      text=[f"{v:+.3f}" for v in top_pairs["ATE"]],
      textposition="outside",
      hovertemplate="<b>%{x}</b><br>ATE: %{y:+.4f}<br>SE: %{customdata:.4f}<extra></extra>",
      customdata=top_pairs["SE"].tolist(),
    ))
    fig_bar.update_layout(**base_layout(theme, height=360,
      title=dict(text=f"Top {n_pairs} Causal Effects (OLS ATE ± 1 SE)",
            font=dict(color=theme.text_muted, size=13)),
      xaxis=dict(tickangle=-28, title="Treatment → Outcome"),
      yaxis=dict(title="ATE (coefficient)", zeroline=True,
            zerolinecolor=theme.border_default, zerolinewidth=1),
    ))
    st.plotly_chart(fig_bar, use_container_width=True)

  # ── PANEL 2 — SELECTED PAIR: CI ERROR-BAR PLOT (2D) ──────────────
  st.markdown(
    f'<div style="color:{theme.text_secondary}; font-weight:600; font-size:0.82rem; '
    f'margin-top:0.6rem; margin-bottom:0.4rem;">② CONFIDENCE INTERVAL VISUALISATION</div>',
    unsafe_allow_html=True,
  )
  st.caption(
    f"Point estimate and 95 % confidence interval for each estimand on the selected pair. "
    f"Uses OLS (ATE/ATT/ATC) and subgroup means (CATE by decade)."
  )

  coef_ate, se_ate, n_obs = _ols_ate(clean, treatment, outcome, ctrl_cols)
  ci_half = 1.96 * se_ate

  # Split treated / untreated at median for ATT / ATC
  med_t = clean[treatment].median()
  treated_mask = clean[treatment] >= med_t
  coef_att, se_att, _ = _ols_ate(clean[treated_mask], treatment, outcome, ctrl_cols)
  coef_atc, se_atc, _ = _ols_ate(clean[~treated_mask], treatment, outcome, ctrl_cols)

  estimand_labels = ["ATE", "ATT (above median)", "ATC (below median)"]
  estimand_vals = [coef_ate, coef_att, coef_atc]
  estimand_se = [se_ate, se_att, se_atc]
  estimand_colors = [PALETTE[1], PALETTE[0], PALETTE[4]]

  fig_ci = go.Figure()
  for i, (lbl, val, se, col) in enumerate(zip(estimand_labels, estimand_vals, estimand_se, estimand_colors)):
    fig_ci.add_trace(go.Scatter(
      x=[val], y=[lbl], mode="markers",
      marker=dict(color=col, size=12, symbol="circle"),
      error_x=dict(type="data", array=[1.96 * se], arrayminus=[1.96 * se],
             color=col, thickness=2.5, width=8),
      name=lbl,
      hovertemplate=f"<b>{lbl}</b><br>Estimate: %{{x:+.4f}}<br>95% CI: [{val - 1.96*se:+.4f}, {val + 1.96*se:+.4f}]<extra></extra>",
    ))
  fig_ci.add_vline(x=0, line_dash="dash", line_color=theme.text_muted, line_width=1)
  fig_ci.update_layout(**base_layout(theme, height=280,
    title=dict(text=f"Estimand Comparison: {treatment[:28]} → {outcome[:28]}",
          font=dict(color=theme.text_muted, size=13)),
    xaxis=dict(title="Causal Effect (coefficient)"),
    yaxis=dict(title=""),
    showlegend=False,
  ))
  st.plotly_chart(fig_ci, use_container_width=True)

  # ── PANEL 3 — DOSE-RESPONSE CURVE (2D) ───────────────────────────
  st.markdown(
    f'<div style="color:{theme.accent_secondary}; font-weight:600; font-size:0.82rem; '
    f'margin-top:0.6rem; margin-bottom:0.4rem;">③ DOSE-RESPONSE CURVE</div>',
    unsafe_allow_html=True,
  )
  st.caption(
    "Conditional mean of the outcome at each level of the treatment (binned into deciles). "
    "Non-linearity here signals heterogeneous effects."
  )

  n_bins = min(10, len(clean) // 3)
  if n_bins >= 3:
    clean2 = clean.copy()
    clean2["_bin"] = _pd2.qcut(clean2[treatment], q=n_bins, duplicates="drop")
    bin_stats = clean2.groupby("_bin", observed=False)[outcome].agg(["mean", "sem"]).reset_index()
    bin_mid = [float(iv.mid) for iv in bin_stats["_bin"]]

    fig_dr = go.Figure()
    fig_dr.add_trace(go.Scatter(
      x=bin_mid, y=bin_stats["mean"],
      mode="lines+markers",
      line=dict(color=PALETTE[0], width=3),
      marker=dict(size=7),
      name="Conditional mean",
      hovertemplate="Treatment ≈ %{x:.3f}<br>E[Y|T=%{x:.3f}] = %{y:.4f}<extra></extra>",
    ))
    # Shaded SEM band
    upper = (bin_stats["mean"] + bin_stats["sem"]).tolist()
    lower = (bin_stats["mean"] - bin_stats["sem"]).tolist()
    fig_dr.add_trace(go.Scatter(
      x=bin_mid + bin_mid[::-1],
      y=upper + lower[::-1],
      fill="toself",
      fillcolor=f"{PALETTE[0]}22",
      line=dict(color="rgba(0,0,0,0)"),
      hoverinfo="skip",
      showlegend=False,
    ))
    # Linear fit overlay
    _x = _np.array(bin_mid)
    _y = bin_stats["mean"].values
    _m, _b = _np.polyfit(_x, _y, 1)
    fig_dr.add_trace(go.Scatter(
      x=bin_mid,
      y=(_m * _x + _b).tolist(),
      mode="lines",
      line=dict(color=PALETTE[2], width=2, dash="dash"),
      name=f"Linear fit (slope={_m:+.4f})",
    ))
    fig_dr.update_layout(**base_layout(theme, height=320,
      title=dict(text=f"Dose-Response: {treatment[:28]} → {outcome[:28]}",
            font=dict(color=theme.text_muted, size=13)),
      xaxis=dict(title=f"Treatment level ({treatment[:20]})"),
      yaxis=dict(title=f"Outcome ({outcome[:20]})"),
      legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
    ))
    st.plotly_chart(fig_dr, use_container_width=True)

  # ── PANEL 4 — 3D ATE SURFACE ──────────────────────────────────────
  st.markdown(
    f'<div style="color:{theme.accent_secondary}; font-weight:600; font-size:0.82rem; '
    f'margin-top:0.6rem; margin-bottom:0.4rem;">④ 3D ATE SURFACE</div>',
    unsafe_allow_html=True,
  )
  st.caption(
    "ATE computed across a grid of (treatment quantile threshold, rolling window size). "
    "Each cell is the OLS slope for observations where treatment ≥ that quantile, "
    "estimated over that many most-recent years. Reveals how the causal effect "
    "changes with treatment intensity and estimation sample."
  )

  q_steps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
  w_steps = [10, 15, 20, 25, 30]
  z_surface = []
  for w in w_steps:
    row = []
    for q in q_steps:
      thresh = clean[treatment].quantile(q)
      sub = clean[clean[treatment] >= thresh].tail(w)
      if len(sub) < 5:
        row.append(float("nan"))
        continue
      c_, _, _ = _ols_ate(sub, treatment, outcome, ctrl_cols)
      row.append(c_)
    z_surface.append(row)

  _z = _np.array(z_surface, dtype=float)
  _z_min, _z_max = _np.nanmin(_z), _np.nanmax(_z)

  colorscale = [
    [0.0, "#ff3366"],
    [0.3, "#f5d547"],
    [0.5, "rgba(255,255,255,0.1)"],
    [0.7, "#00aaff"],
    [1.0, "#00ff88"],
  ]

  fig_surf = go.Figure(go.Surface(
    x=[f"Q{int(q*100)}" for q in q_steps],
    y=[f"W{w}yr" for w in w_steps],
    z=_z.tolist(),
    colorscale=colorscale,
    cmin=_z_min, cmax=_z_max,
    contours=dict(
      z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
    ),
    hovertemplate=(
      "Treatment quantile: %{x}<br>"
      "Window: %{y}<br>"
      "ATE: %{z:.4f}<extra></extra>"
    ),
    colorbar=dict(title="ATE", len=0.6, thickness=12,
           tickfont=dict(color=theme.text_muted, size=10)),
  ))
  fig_surf.update_layout(
    height=520,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=theme.text_muted, family="IBM Plex Sans, sans-serif"),
    scene=dict(
      xaxis=dict(title="Treatment quantile", backgroundcolor="rgba(0,0,0,0)",
            gridcolor=theme.border_default, color=theme.text_muted),
      yaxis=dict(title="Sample window (years)", backgroundcolor="rgba(0,0,0,0)",
            gridcolor=theme.border_default, color=theme.text_muted),
      zaxis=dict(title="ATE", backgroundcolor="rgba(0,0,0,0)",
            gridcolor=theme.border_default, color=theme.text_muted,
            zerolinecolor=theme.border_default),
      bgcolor="rgba(0,0,0,0)",
      camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
    ),
    title=dict(text=f"ATE Surface: {treatment[:25]} → {outcome[:25]}",
          font=dict(color=theme.text_muted, size=13)),
    margin=dict(l=0, r=0, t=40, b=0),
  )
  st.plotly_chart(fig_surf, use_container_width=True)

  # ── PANEL 5 — 3D HETEROGENEITY SCATTER ───────────────────────────
  if confounder:
    st.markdown(
      f'<div style="color:{theme.accent_secondary}; font-weight:600; font-size:0.82rem; '
      f'margin-top:0.6rem; margin-bottom:0.4rem;">⑤ 3D HETEROGENEITY SCATTER</div>',
      unsafe_allow_html=True,
    )
    st.caption(
      f"Each observation as a point in (treatment, confounder, outcome) space. "
      f"Colour encodes the outcome level. Reveals interaction effects and subgroup patterns."
    )
    fig_sc3 = go.Figure(go.Scatter3d(
      x=clean[treatment].tolist(),
      y=clean[confounder].tolist(),
      z=clean[outcome].tolist(),
      mode="markers",
      marker=dict(
        size=5,
        color=clean[outcome].tolist(),
        colorscale="Viridis",
        opacity=0.85,
        colorbar=dict(title=outcome[:18], len=0.5, thickness=10,
               tickfont=dict(color=theme.text_muted, size=9)),
        line=dict(width=0),
      ),
      hovertemplate=(
        f"{treatment[:18]}: %{{x:.3f}}<br>"
        f"{confounder[:18]}: %{{y:.3f}}<br>"
        f"{outcome[:18]}: %{{z:.3f}}<extra></extra>"
      ),
    ))
    fig_sc3.update_layout(
      height=500,
      paper_bgcolor="rgba(0,0,0,0)",
      font=dict(color=theme.text_muted, family="IBM Plex Sans, sans-serif"),
      scene=dict(
        xaxis=dict(title=treatment[:22], backgroundcolor="rgba(0,0,0,0)",
              gridcolor=theme.border_default, color=theme.text_muted),
        yaxis=dict(title=confounder[:22], backgroundcolor="rgba(0,0,0,0)",
              gridcolor=theme.border_default, color=theme.text_muted),
        zaxis=dict(title=outcome[:22], backgroundcolor="rgba(0,0,0,0)",
              gridcolor=theme.border_default, color=theme.text_muted),
        bgcolor="rgba(0,0,0,0)",
        camera=dict(eye=dict(x=1.5, y=-1.5, z=1.0)),
      ),
      title=dict(
        text=f"Heterogeneity: {treatment[:20]} × {confounder[:20]} → {outcome[:20]}",
        font=dict(color=theme.text_muted, size=13),
      ),
      margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_sc3, use_container_width=True)

  # ── PANEL 6 — STRUCTURAL CAUSAL ESTIMATES (DoWhy / EconML) ───────
  st.markdown(
    f'<div style="color:{theme.accent_secondary}; font-weight:600; font-size:0.82rem; '
    f'margin-top:0.6rem; margin-bottom:0.4rem;">⑥ STRUCTURAL CAUSAL ESTIMATES</div>',
    unsafe_allow_html=True,
  )
  st.caption(
    "Structural estimate using the Scarcity causal engine (DoWhy backdoor + optional EconML CATE). "
    "Slower than the OLS panels above but accounts for confounding via a causal graph."
  )
  with st.expander("Run structural estimate (DoWhy)", expanded=False):
    try:
      from kshiked.ui.kshield.causal import render_causal_evidence_panel
      render_causal_evidence_panel(
        df=clean,
        treatment=treatment,
        outcome=outcome,
        confounders=ctrl_cols or None,
        theme=theme,
        key_prefix="ced_structural",
      )
    except Exception as _e:
      st.caption(f"Structural estimate unavailable: {_e}")
