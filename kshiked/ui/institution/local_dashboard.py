import streamlit as st
import sys
import os
import pandas as pd
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
  sys.path.insert(0, project_root)

from kshiked.ui.institution.backend.auth import enforce_role
from kshiked.ui.institution.backend.models import Role
from kshiked.ui.institution.backend.ontology import OntologyEnforcer
from kshiked.ui.institution.backend.delta_sync import DeltaSyncManager
from kshiked.ui.institution.backend.auto_pipeline import AutoPipeline
from kshiked.ui.institution.backend.database import get_connection
from kshiked.ui.theme import LIGHT_THEME
from kshiked.ui.institution.style import inject_enterprise_theme
from kshiked.ui.institution.backend.messaging import SecureMessaging
from kshiked.ui.institution.backend.data_sharing import DataSharingManager
from kshiked.ui.institution.collab_room import render_collab_room
from kshiked.ui.institution.backend.analytics_engine import compute_cost_of_delay_kes_b
from kshiked.ui.institution.unified_report_export import render_unified_report_export
from kshiked.ui.kshield.causal.view import (
  _render_granger_section,
  _render_causal_network,
  _render_cross_corr
)
from kshiked.ui.institution.backend.report_narrator import (
  narrate_composite_scores,
  narrate_severity,
  narrate_shock_vector,
  narrate_anomaly_detection,
  narrate_trend_analysis,
  narrate_propagation_chain,
  narrate_anomaly_chart_stats,
  narrate_spatial_hotspot_summary,
  narrate_correlation_findings,
  narrate_forecast_direction,
  narrate_forecast_overview,
  narrate_causal_relationship_summary,
  narrate_risk_propagation_overview,
)


def _get_institution_name(inst_id: int) -> str:
  try:
    with get_connection() as conn:
      c = conn.cursor()
      c.execute("SELECT name FROM institutions WHERE id = ?", (inst_id,))
      row = c.fetchone()
      if row:
        return row['name']
  except Exception:
    pass
  return f"Institution {inst_id}"


def _get_basket_name(basket_id: int) -> str:
  try:
    with get_connection() as conn:
      c = conn.cursor()
      c.execute("SELECT name FROM baskets WHERE id = ?", (basket_id,))
      row = c.fetchone()
      if row:
        return row['name']
  except Exception:
    pass
  return f"Sector {basket_id}"


def _apply_plotly_numeric_font(fig):
  fig.update_layout(
    font=dict(family="IBM Plex Sans, sans-serif"),
    hoverlabel=dict(font=dict(family="IBM Plex Mono, monospace")),
  )
  for axis_name in [
    "xaxis", "xaxis2", "xaxis3", "xaxis4", "xaxis5",
    "yaxis", "yaxis2", "yaxis3", "yaxis4", "yaxis5",
  ]:
    axis = getattr(fig.layout, axis_name, None)
    if axis is not None:
      axis.tickfont = dict(family="IBM Plex Mono, monospace")
  return fig

def _guide(method_html: str, interp_html: str, rec_html: str) -> None:
  """Collapsible 3-section analysis guide (method / interpretation / recommendation)."""
  _AC = "#0066cc"   # LIGHT_THEME accent_primary
  _AW = "#ffc107"   # accent_warning
  _AS = "#28a745"   # accent_success
  _TM = "#6c757d"   # text_muted
  with st.expander("Analysis Guide & Interpretation", expanded=False):
    st.markdown(
      f"<div style='font-size:0.82rem; line-height:1.65;'>"
      f"<div style='color:{_AC}; font-weight:700; font-size:0.72rem; "
      f"letter-spacing:0.08em; margin-bottom:0.35rem;'>WHAT IS THIS ANALYSIS?</div>"
      f"<div style='color:{_TM}; margin-bottom:0.9rem;'>{method_html}</div>"
      f"<div style='color:{_AW}; font-weight:700; font-size:0.72rem; "
      f"letter-spacing:0.08em; margin-bottom:0.35rem;'>INTERPRETATION</div>"
      f"<div style='color:{_TM}; margin-bottom:0.9rem;'>{interp_html}</div>"
      f"<div style='color:{_AS}; font-weight:700; font-size:0.72rem; "
      f"letter-spacing:0.08em; margin-bottom:0.35rem;'>RECOMMENDATION</div>"
      f"<div style='color:{_TM};'>{rec_html}</div>"
      f"</div>",
      unsafe_allow_html=True,
    )


def render(active_section: str = "Data Intake", use_enterprise_theme: bool = True):
  enforce_role(Role.INSTITUTION.value)
  if use_enterprise_theme:
    inject_enterprise_theme()
    st.markdown(
      """
      <style>
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap');
        [data-testid="stMetricValue"],
        [data-testid="stMetricDelta"],
        .spoke-num,
        .stNumberInput input,
        input[type="number"],
        .js-plotly-plot .xtick text,
        .js-plotly-plot .ytick text,
        .js-plotly-plot .hovertext {
          font-family: 'IBM Plex Mono', monospace !important;
          font-variant-numeric: tabular-nums;
        }
      </style>
      """,
      unsafe_allow_html=True,
    )

  inst_id = st.session_state.get('institution_id')
  basket_id = st.session_state.get('basket_id')
  fl_mode = st.session_state.get('fl_mode_enabled', False)

  # Human-readable institution and sector names
  inst_name = _get_institution_name(inst_id) if inst_id else st.session_state.get('username', 'Institution')
  basket_name = _get_basket_name(basket_id) if basket_id else f"Sector {basket_id}"

  _section_descriptions = {
    "Data Intake": "Upload your institution data, validate required schema fields, and prepare clean local datasets.",
    "Signal Analysis": "Run the full intelligence stack and generate narrative insights, trends, and threat indicators.",
    "Granger Causality": "Test temporal lead-lag influence between numeric indicators.",
    "Causal Network": "Visualize directional system dependencies detected from your data.",
    "Cross-Correlations": "Inspect pairwise movement patterns between indicators.",
    "Effect Estimation": "Estimate direct causal effect sizes for selected cause and effect variables.",
    "Active Projects": "Track ongoing operations and shared milestone execution across sectors.",
    "Inbox": "Receive directives and exchange secure operational messages.",
    "Collaboration Room": "Coordinate planning and response with other sector entities.",
    "Model Configuration": "Set local privacy and modeling controls before running sensitive analyses.",
    "FL Training Log": "Review federated submissions, statuses, and prior Mode B training events.",
  }

  st.markdown(f"<h2 class='k-page-title'>{active_section}</h2>", unsafe_allow_html=True)
  st.markdown(
    f"<p class='k-page-desc'>{_section_descriptions.get(active_section, 'Institution analytics workspace.')}</p>",
    unsafe_allow_html=True,
  )

  pipeline_result = st.session_state.get('pipeline_result')
  local_df = st.session_state.get('local_df')
  if pipeline_result is not None and hasattr(pipeline_result, 'composite'):
    comp = getattr(pipeline_result, 'composite', {}) or {}
    severity_seed = float(comp.get('B_Impact', 0.0) or 0.0)
  else:
    severity_seed = 0.0
  if severity_seed <= 0.0:
    severity_seed = 5.0 if active_section == "Signal Analysis" else 3.0

  spoke_cost_snapshot = compute_cost_of_delay_kes_b(severity=severity_seed, projection_steps=4)
  render_unified_report_export(
    dashboard_name="Spoke Dashboard",
    section_name=active_section,
    metrics={
      "institution": str(inst_name),
      "sector": str(basket_name),
      "has_uploaded_data": bool(local_df is not None),
      "rows_in_local_dataset": int(len(local_df)) if isinstance(local_df, pd.DataFrame) else 0,
      "pipeline_result_available": bool(pipeline_result is not None),
      "severity_signal": round(float(severity_seed), 2),
    },
    highlights=[
      "This report is written for non-technical readers and policy decision support.",
      "Use it to communicate local risk posture and urgency to sector command.",
      f"Current section focus: {active_section}.",
    ],
    interpretations=[
      "Higher local severity indicates increased probability of escalation to sector level.",
      "If threat level rises, prioritize early interventions and update sector command quickly.",
      "Use causal and trend outputs to explain why the signal is changing, not just that it changed.",
    ],
    cost_delay=spoke_cost_snapshot,
    tables={
      "local_dataset": local_df if isinstance(local_df, pd.DataFrame) else pd.DataFrame(),
      "relationships": pd.DataFrame(getattr(pipeline_result, 'relationships', []) or []) if pipeline_result is not None else pd.DataFrame(),
      "trend_signals": pd.DataFrame(getattr(pipeline_result, 'trend_signals', []) or []) if pipeline_result is not None else pd.DataFrame(),
      "propagation_chains": pd.DataFrame(getattr(pipeline_result, 'propagation_chains', []) or []) if pipeline_result is not None else pd.DataFrame(),
      "spatial_hotspots": pd.DataFrame(getattr(pipeline_result, 'spatial_hotspots', []) or []) if pipeline_result is not None else pd.DataFrame(),
      "forecast_matrix": pd.DataFrame(getattr(pipeline_result, 'forecast_matrix', []) or []) if pipeline_result is not None else pd.DataFrame(),
    },
    evidence={
      "institution_id": inst_id,
      "basket_id": basket_id,
      "threat_level": getattr(pipeline_result, 'threat_level', '') if pipeline_result is not None else '',
    },
    key_prefix="spoke_unified_report",
  )

  # Fetch all baskets for cross-sector references (cached per session – schemas change rarely)
  _baskets_cache_key = f"_all_baskets_cache"
  if _baskets_cache_key not in st.session_state:
    with get_connection() as conn:
      c = conn.cursor()
      c.execute("SELECT id, name FROM baskets")
      st.session_state[_baskets_cache_key] = {r['id']: r['name'] for r in c.fetchall()}
  all_baskets = st.session_state[_baskets_cache_key]

  # Fetch schemas early so they are available across all tabs (cached per basket per session)
  _schema_cache_key = f"_basket_schema_{basket_id}"
  _custom_schema_cache_key = f"_custom_schemas_{basket_id}"
  if _schema_cache_key not in st.session_state:
    st.session_state[_schema_cache_key] = OntologyEnforcer.get_basket_schema(basket_id)
  from kshiked.ui.institution.backend.schema_manager import SchemaManager
  if _custom_schema_cache_key not in st.session_state:
    st.session_state[_custom_schema_cache_key] = SchemaManager.get_schemas(basket_id)
  schema = st.session_state[_schema_cache_key]
  custom_schemas = st.session_state[_custom_schema_cache_key]
  latest_custom_schema = custom_schemas[0] if custom_schemas else None

  if active_section == "Data Intake":
    st.markdown("### Data Preparation")
    st.write("Upload your institution's data. It will be validated against your sector's expected format. The data is processed locally and never transmitted anywhere.")

    if schema or latest_custom_schema:
      with st.expander("View required column format for this sector"):
        if schema:
          st.write("**Base Sector Ontology:**")
          st.json(schema)
        if latest_custom_schema:
          st.write(f"**Admin Defined Schema: {latest_custom_schema['schema_name']}**")
          st.json(latest_custom_schema['fields'])
    else:
      st.error("No reporting schema is defined for your sector. Contact your sector admin to define one before uploading data.")
      st.stop()

    uploaded_file = st.file_uploader(
      "Upload local data (CSV)",
      type="csv",
      help="Processed entirely in this session. Never transmitted."
    )
    
    if uploaded_file is not None:
      df = pd.read_csv(uploaded_file)
      st.write("Local Data Preview:")
      st.dataframe(df.head(3))
      
      is_valid_base, message_base = OntologyEnforcer.validate_dataset_signature(basket_id, list(df.columns)) if schema else (True, "No base ontology")
      
      is_valid_custom = True
      message_custom = ""
      if latest_custom_schema:
        is_valid_custom, message_custom = SchemaManager.validate_dataframe(latest_custom_schema['fields'], df)
      
      if is_valid_base and is_valid_custom:
        st.success("Data format validated.")
        st.session_state['local_df'] = df
      else:
        errors = []
        if not is_valid_base: errors.append(f"Base Ontology Error: {message_base}")
        if not is_valid_custom: errors.append(f"Custom Schema Error: {message_custom}")
        st.error("Format mismatch — " + " | ".join(errors))
        st.session_state['local_df'] = None

  if active_section == "Signal Analysis":
    st.markdown("### Signal Analysis")
    st.caption("10-module intelligence pipeline. Upload data in the **Data Intake** tab first, then press **Run Analysis**.")

    if st.session_state.get('local_df') is None:
      st.warning("Upload and validate your data in the **Data Intake** tab first.")
    else:
      # ── SENSITIVITY CLASSIFICATION ──────────────────────────────────────
      sensitivity = st.selectbox(
        "Data Sensitivity Level",
        ["Public", "Restricted", "Confidential"],
        key="data_sensitivity",
        help=(
          "Public: aggregated statistics may be shared. "
          "Restricted: only composite scores are forwarded — no raw rows. "
          "Confidential: Federated Learning Mode B is required — raw data never leaves this node."
        )
      )

      if sensitivity == "Confidential" and not fl_mode:
        st.error(
          "**Confidential data requires Federated Learning (Mode B).** \n"
          "Enable **Mode B** in **Model Configuration**. "
          "In Mode B, only mathematical summaries (model gradients) are shared with your sector admin — "
          "your raw data never leaves this node."
        )
        st.info(
          "Why? Confidential datasets may contain personally identifiable, commercially sensitive, or "
          "operationally classified information. Running standard analysis would transmit composite scores "
          "that could allow reconstruction of the underlying data. Mode B prevents this."
        )
        st.stop()
      elif sensitivity == "Confidential" and fl_mode:
        st.success(
          "Mode B active — your raw data stays on this node. "
          "Only mathematical weight summaries are shared upward."
        )
      elif sensitivity == "Restricted":
        st.info(
          "Restricted: only composite scores, anomaly severity, and anonymised relationship patterns "
          "will be included in the report to your sector admin. No raw rows are transmitted."
        )

      if st.button("Run Full Analysis", type="primary", use_container_width=True):
        with st.spinner("Running 10-module intelligence pipeline — anomaly detection, trend analysis, threat assessment, SFC model, propagation mapping..."):
          result = AutoPipeline.run(st.session_state['local_df'])
          st.session_state['pipeline_result'] = result

      res = st.session_state.get('pipeline_result')
      if res is not None:
        # ── HEADER: narrative + composite scores ────────────────────────
        _threat_colors = {"CRITICAL": "#BB0000", "HIGH": "#E05000", "ELEVATED": "#F59E0B", "GUARDED": "#2563EB", "LOW": "#006600"}
        _tc = _threat_colors.get(res.threat_level, "#006600")
        st.markdown(
          f'<div style="background:rgba(0,0,0,0.03); border-left:4px solid {_tc}; padding:1rem; margin-bottom:0.5rem; border-radius:0 8px 8px 0;">'
          f'<div style="color:{_tc}; font-weight:bold; margin-bottom:0.4rem; font-size:0.85rem;">'
          f'ANALYSIS SUMMARY — Threat Level: {res.threat_level}</div>'
          f'<div style="font-size:0.88rem; color:#1F2937;">{res.narrative}</div></div>',
          unsafe_allow_html=True
        )
        c = res.composite
        _m1, _m2, _m3, _m4, _m5 = st.columns(5)
        _m1.metric("Peak Anomaly", f"{res.peak_score:.2f}")
        _m2.metric("Detection", f"{c.get('A_Detection', 0)}/10")
        _m3.metric("Impact", f"{c.get('B_Impact', 0)}/10")
        _m4.metric("Certainty", f"{c.get('C_Certainty', 0)}/10")
        _m5.metric("Threat Level", res.threat_level)

        if res.priority_alerts:
          for alert in res.priority_alerts:
            st.warning(alert)

        st.write("---")

        # ── PLAIN-LANGUAGE EXPLANATION ────────────────────────────────
        st.markdown(
          f'<div style="background:#F0FDF4; border-left:4px solid #10B981; padding:12px 16px; '
          f'border-radius:0 8px 8px 0; margin-bottom:1rem; font-size:0.9rem; line-height:1.6;">'
          f'<strong>What does this mean?</strong><br>'
          f'{narrate_composite_scores(c)}<br><br>'
          f'{narrate_severity(res.peak_score)}</div>',
          unsafe_allow_html=True,
        )

        # ── 1. ANOMALY DETECTION ────────────────────────────────────────
        with st.expander("1. Anomaly Detection — Did something unusual happen?", expanded=True):
          # Identify which specific column deviated most at the peak moment
          _anom_df = st.session_state.get("local_df")
          _peak_col = ""
          _col_deviations: dict = {}
          if isinstance(_anom_df, pd.DataFrame) and res.peak_index >= 0:
            _num_df_a = _anom_df.select_dtypes(include=[np.number])
            for _col in _num_df_a.columns:
              _s = _num_df_a[_col].dropna()
              if len(_s) < 3:
                continue
              _std = float(_s.std())
              if _std < 1e-9:
                continue
              _idx = min(res.peak_index, len(_num_df_a) - 1)
              _z = abs(float(_num_df_a[_col].iloc[_idx]) - float(_s.mean())) / _std
              _col_deviations[_col] = round(_z, 3)
            if _col_deviations:
              _peak_col = max(_col_deviations, key=_col_deviations.get)

          # Plain-language summary — now referencing actual column names
          st.markdown(narrate_anomaly_detection(
            peak_score=res.peak_score,
            structural_breaks=res.structural_breaks,
            peak_column=_peak_col,
            peak_index=res.peak_index,
            col_deviations=_col_deviations,
          ))

          if res.anomaly_scores:
            # Overall anomaly score timeline
            _anom_fig = go.Figure()
            _anom_fig.add_trace(go.Scatter(
              y=res.anomaly_scores, mode="lines",
              line=dict(color="#E05000", width=1.8),
              name="Anomaly Score",
              hovertemplate="Record %{x}: score %{y:.3f}<extra></extra>",
            ))
            # Mark peak
            if 0 <= res.peak_index < len(res.anomaly_scores):
              _anom_fig.add_trace(go.Scatter(
                x=[res.peak_index], y=[res.anomaly_scores[res.peak_index]],
                mode="markers", marker=dict(color="#BB0000", size=10, symbol="x"),
                name=f"Peak ({_peak_col})" if _peak_col else "Peak",
                hovertemplate=f"Peak at record {res.peak_index}: score {res.peak_score:.3f}<extra></extra>",
              ))
            # Mark structural breaks
            for _br in (res.structural_breaks or [])[:20]:
              _anom_fig.add_vline(x=int(_br), line_width=1, line_dash="dot",
                                  line_color="#2563EB", opacity=0.5)
            _anom_fig.update_layout(
              height=240,
              margin=dict(l=8, r=8, t=8, b=8),
              xaxis_title="Record",
              yaxis_title="RRCF Anomaly Score",
              paper_bgcolor="rgba(0,0,0,0)",
              plot_bgcolor="rgba(0,0,0,0)",
              legend=dict(orientation="h", y=1.08),
            )
            _apply_plotly_numeric_font(_anom_fig)
            st.plotly_chart(_anom_fig, use_container_width=True)
            st.caption(
              narrate_anomaly_chart_stats(
                anomaly_scores=res.anomaly_scores,
                peak_score=res.peak_score,
                peak_index=res.peak_index,
                structural_breaks=res.structural_breaks,
              )
            )

            # Per-variable deviation bar at the peak moment
            if _col_deviations:
              st.markdown(f"**Variable deviation at record {res.peak_index}** (z-score — how many standard deviations from each variable's mean):")
              _dev_sorted = sorted(_col_deviations.items(), key=lambda x: x[1], reverse=True)
              _bar_cols = [d[0] for d in _dev_sorted]
              _bar_vals = [d[1] for d in _dev_sorted]
              _bar_colors = ["#BB0000" if v >= 2.0 else "#E05000" if v >= 1.0 else "#F59E0B" if v >= 0.5 else "#6B7280"
                             for v in _bar_vals]
              _dev_fig = go.Figure(go.Bar(
                x=_bar_cols, y=_bar_vals,
                marker_color=_bar_colors,
                text=[f"{v:.2f}" for v in _bar_vals],
                textposition="outside",
                hovertemplate="%{x}: z=%{y:.2f}<extra></extra>",
              ))
              _dev_fig.add_hline(y=2.0, line_dash="dash", line_color="#BB0000",
                                 annotation_text="Severe threshold (z=2)", annotation_position="top right")
              _dev_fig.add_hline(y=1.0, line_dash="dot", line_color="#E05000",
                                 annotation_text="Moderate threshold (z=1)", annotation_position="top right")
              _dev_fig.update_layout(
                height=280,
                margin=dict(l=8, r=8, t=32, b=8),
                xaxis_title="Variable",
                yaxis_title="Deviation (z-score)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
              )
              _apply_plotly_numeric_font(_dev_fig)
              st.plotly_chart(_dev_fig, use_container_width=True)
          else:
            st.info("No anomaly scores computed — check that your data has numeric columns.")

          try:
            _g_peak = res.peak_score
            _g_breaks = len(res.structural_breaks or [])
            _g_top_dev = _dev_sorted[0] if _col_deviations else None
            _g_top_str = f"<b>{_g_top_dev[0]}</b> (z = {_g_top_dev[1]:.2f})" if _g_top_dev else "no specific variable"
            _g_sev = "critical" if _g_peak >= 4.0 else "elevated" if _g_peak >= 2.0 else "low"
          except Exception:
            _g_peak, _g_breaks, _g_top_str, _g_sev = 0, 0, "unknown", "low"
          _guide(
            method_html=(
              "The <b>RRCF Anomaly Score</b> is computed by Robust Random Cut Forest — an unsupervised algorithm that "
              "measures how isolated a data point is within a random forest of cuts. Higher scores indicate observations "
              "that are harder to isolate, meaning they are statistical outliers relative to the overall pattern. "
              "<b>Structural breaks</b> are detected by a CUSUM-style change-point test that identifies when the "
              "underlying data-generating process shifts. The <b>variable deviation bar</b> shows the z-score of each "
              "variable at the peak anomaly moment — how many standard deviations it was from its historical mean."
            ),
            interp_html=(
              f"Peak anomaly score: <b>{_g_peak:.2f}</b> — severity level: <b>{_g_sev}</b>. "
              f"<b>{_g_breaks}</b> structural break(s) detected, indicating regime shift(s) in the data pattern. "
              f"The most deviant variable at the anomaly peak is {_g_top_str}. "
              "Red bars (z ≥ 2) are statistically severe; orange (z ≥ 1) are moderate deviations worth tracking."
            ),
            rec_html=(
              f"Focus immediate investigation on {_g_top_str} — it is the primary contributor to the anomaly. "
              "Cross-reference with external events at the structural break timestamps. "
              "If peak score exceeds 2.0, escalate to your sector admin using the button below."
            ),
          )

        # ── 2. TEMPORAL TREND ANALYSIS ──────────────────────────────────
        with st.expander("2. Temporal Trend Analysis — Are things getting better or worse?"):
          if res.trend_signals:
            # Plain-language summary first
            st.markdown(narrate_trend_analysis(res.trend_signals))
            _source_df = st.session_state.get("local_df")
            _trend_cols_ranked = [
              t.get("column")
              for t in sorted(
                res.trend_signals,
                key=lambda x: abs(float(x.get("growth_rate", 0.0))),
                reverse=True,
              )
              if t.get("column")
            ]
            if isinstance(_source_df, pd.DataFrame):
              _plot_cols = []
              for _col in _trend_cols_ranked:
                if _col in _source_df.columns and _col not in _plot_cols:
                  _plot_cols.append(_col)
                if len(_plot_cols) >= 4:
                  break

              if _plot_cols:
                _trend_plot_df = _source_df[_plot_cols].apply(pd.to_numeric, errors="coerce").dropna(how="all")
                if not _trend_plot_df.empty:
                  _x_axis = list(range(len(_trend_plot_df)))
                  _trend_fig = go.Figure()
                  for _col in _plot_cols:
                    _series = _trend_plot_df[_col].tolist()
                    if not any(pd.notna(_series)):
                      continue
                    _trend_fig.add_trace(
                      go.Scatter(
                        x=_x_axis,
                        y=_series,
                        mode="lines",
                        name=_col,
                        line=dict(width=2),
                      )
                    )

                  for _break_idx in (res.structural_breaks or [])[:20]:
                    if 0 <= int(_break_idx) < len(_trend_plot_df):
                      _trend_fig.add_vline(
                        x=int(_break_idx),
                        line_width=1,
                        line_dash="dot",
                        line_color="#BB0000",
                        opacity=0.55,
                      )

                  _trend_fig.update_layout(
                    height=330,
                    margin=dict(l=8, r=8, t=8, b=8),
                    xaxis_title="Record Sequence",
                    yaxis_title="Observed Value",
                    legend_title_text="Variables",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                  )
                  _apply_plotly_numeric_font(_trend_fig)
                  st.plotly_chart(_trend_fig, use_container_width=True)
                  st.caption(
                    f"Temporal trajectory for the top {len(_plot_cols)} most-changing variables. "
                    "Dotted vertical markers indicate structural break points."
                  )
            _dir_icons = {"acceleration": "", "deceleration": "", "stable": ""}
            _vol_icons = {"increasing volatility": "", "decreasing volatility": "", "stable": ""}
            _rows = []
            for t in res.trend_signals:
              d = t.get("direction", "stable")
              v = t.get("volatility", "stable")
              gr = t.get("growth_rate", 0.0)
              _rows.append({
                "Variable": t.get("column", ""),
                "Trend": f"{_dir_icons.get(d, '')} {d.title()}",
                "Volatility": f"{_vol_icons.get(v, '')} {v.title()}",
                "Growth Rate": f"{gr:+.2%}",
                "Early Mean": t.get("early_mean", ""),
                "Late Mean": t.get("late_mean", ""),
              })
            st.dataframe(_rows, use_container_width=True, hide_index=True)
            if res.structural_breaks:
              st.warning(
                f"**{len(res.structural_breaks)} structural break(s)** detected at rows: "
                f"{res.structural_breaks[:15]}. These represent major regime shifts in your data."
              )

            try:
              _g_accel = [t for t in res.trend_signals if t.get("direction") == "acceleration"]
              _g_decel = [t for t in res.trend_signals if t.get("direction") == "deceleration"]
              _g_top_grow = max(res.trend_signals, key=lambda t: abs(float(t.get("growth_rate", 0))), default={})
              _g_top_grow_name = _g_top_grow.get("column", "unknown")
              _g_top_grow_rate = float(_g_top_grow.get("growth_rate", 0))
              _g_breaks_t = len(res.structural_breaks or [])
            except Exception:
              _g_accel, _g_decel, _g_top_grow_name, _g_top_grow_rate, _g_breaks_t = [], [], "unknown", 0, 0
            _guide(
              method_html=(
                "This chart shows the <b>temporal trajectory</b> of the top variables ranked by absolute growth rate. "
                "Each line is the raw observed value over the record sequence. "
                "<b>Structural breaks</b> (dotted vertical lines) mark rows where a CUSUM test detects a regime shift — "
                "a point where the statistical properties (mean or variance) of the data changed significantly. "
                "The trend table below classifies each variable as accelerating, decelerating, or stable, based on "
                "comparing the mean of the first vs last third of the observation window."
              ),
              interp_html=(
                f"<b>{len(_g_accel)}</b> variable(s) accelerating (growing), "
                f"<b>{len(_g_decel)}</b> decelerating (shrinking). "
                f"Fastest-changing: <b>{_g_top_grow_name}</b> at {_g_top_grow_rate:+.2%} growth rate. "
                + (f"<b>{_g_breaks_t} structural break(s)</b> indicate the data pattern shifted — prior trends may not predict future behaviour." if _g_breaks_t else "No structural breaks detected — pattern is statistically stable.")
              ),
              rec_html=(
                f"Track <b>{_g_top_grow_name}</b> closely — its rate of change is most extreme. "
                "Variables with structural breaks should be reported to your sector admin as potential leading indicators of a systemic shift. "
                "Cross-reference accelerating variables with known events (policy changes, external shocks) at those break points."
              ),
            )
          else:
            st.info("Not enough data for trend analysis (need ≥ 6 rows per variable).")

        # ── 3. SPATIAL ANALYSIS ─────────────────────────────────────────
        with st.expander("3. Spatial Analysis"):
          if res.spatial_available and res.spatial_hotspots:
            st.caption(narrate_spatial_hotspot_summary(res.spatial_hotspots))
            _hs_df = [{"Cluster": i+1, "Latitude": h["lat"], "Longitude": h["lon"], "Records": h["count"]}
                 for i, h in enumerate(res.spatial_hotspots)]
            st.dataframe(_hs_df, use_container_width=True, hide_index=True)
            try:
              import plotly.express as _px_map
              _map_df = pd.DataFrame(res.spatial_hotspots)
              fig_map = _px_map.scatter_mapbox(
                _map_df, lat="lat", lon="lon", size="count",
                mapbox_style="open-street-map", zoom=5, height=300
              )
              fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
              _apply_plotly_numeric_font(fig_map)
              st.plotly_chart(fig_map, use_container_width=True)
            except Exception:
              st.caption("Hotspots were identified, but the map preview is not available in this runtime.")
          else:
            st.info(
              "No geographic columns detected. To enable spatial analysis, include columns named "
              "**lat** and **lon** (or **lng**) in your dataset. "
              "The engine will then compute geographic hotspot clusters automatically."
            )

        # ── 4. CAUSAL RELATIONSHIP ANALYSIS ────────────────────────────
        with st.expander("4. Causal Relationship Analysis — What is causing what?"):
          st.markdown(
            narrate_causal_relationship_summary(
              hypotheses_total=res.hypotheses_total,
              hypotheses_active=res.hypotheses_active,
              overall_confidence=res.overall_confidence,
              relationship_summary=res.relationship_summary,
              knowledge_graph=res.knowledge_graph,
            )
          )

          # Human-readable relationship labels (matches AutoPipeline._REL_LABELS)
          _REL_LABELS = {
            "causal":         "causally drives",
            "correlational":  "moves together with",
            "temporal_lag":   "predicts (with a time lag)",
            "equilibrium":    "stays in balance with",
            "functional":     "shows a deterministic dependency on",
            "compositional":  "is composed of",
            "competitive":    "competes against",
            "synergistic":    "amplifies",
            "probabilistic":  "statistically predicts",
            "structural":     "is structurally linked to",
            "mediating":      "mediates the effect on",
            "moderating":     "moderates the relationship between",
            "graph":          "is graph-connected to",
            "similarity":     "is similar in behaviour to",
            "logical":        "logically implies changes in",
          }

          _display_relationships = []
          _seen_pairs = set()
          for rel in sorted(res.relationships or [], key=lambda r: float(r.get("confidence", 0.0)), reverse=True):
            vars_ = rel.get("variables", [])
            if len(vars_) < 2:
              continue
            pair_key = tuple(sorted((str(vars_[0]), str(vars_[1]))))
            if pair_key in _seen_pairs:
              continue
            _seen_pairs.add(pair_key)
            _confidence = float(rel.get("confidence", 0.0))
            _raw_type = str(rel.get("rel_type", ""))
            # Use the plain-English label; fall back to cleaned raw type
            _rel_label = _REL_LABELS.get(_raw_type) or _raw_type.replace("_", " ") or "is related to"
            _conf_band = "high confidence" if _confidence >= 0.75 else "moderate confidence" if _confidence >= 0.45 else "low confidence"
            _display_relationships.append(
              (vars_[0], _rel_label, vars_[1], _confidence, _conf_band)
            )
            if len(_display_relationships) >= 10:
              break

          if _display_relationships:
            st.markdown("**Discovered relationships (ranked by confidence):**")
            for _v0, _label, _v1, _conf, _band in _display_relationships:
              st.markdown(
                f"- **{_v0}** {_label} **{_v1}** — {_conf:.0%} ({_band})"
              )
          elif res.relationship_summary:
            st.markdown("**Relationship summary from discovery engine:**")
            for sentence in res.relationship_summary[:8]:
              st.markdown(f"- {sentence}")

          # Causal network graph — shows directed links between variables
          if res.knowledge_graph:
            st.markdown("**Causal dependency network** — arrows show which variable influences which:")
            try:
              import networkx as nx
              _G = nx.DiGraph()
              for _edge in res.knowledge_graph[:40]:
                _src = _edge.get("source") or (_edge.get("variables", ["?", "?"])[0])
                _tgt = _edge.get("target") or (_edge.get("variables", ["?", "?"])[-1])
                _w = float(_edge.get("confidence", 0.5))
                _G.add_edge(str(_src), str(_tgt), weight=_w)

              _pos = nx.spring_layout(_G, seed=42, k=2.0)

              _edge_x, _edge_y, _edge_hover = [], [], []
              for _u, _v, _data in _G.edges(data=True):
                x0, y0 = _pos[_u]; x1, y1 = _pos[_v]
                _edge_x += [x0, x1, None]
                _edge_y += [y0, y1, None]
                _label = _REL_LABELS.get(
                  next((r.get("rel_type","") for r in (res.relationships or [])
                        if set([_u, _v]).issubset(set(r.get("variables", [])))), ""),
                  "related to"
                )
                _edge_hover.append(f"{_u} → {_v} ({_label}, {_data.get('weight', 0):.0%})")

              _node_x = [_pos[n][0] for n in _G.nodes()]
              _node_y = [_pos[n][1] for n in _G.nodes()]
              _node_labels = list(_G.nodes())
              _in_degree = dict(_G.in_degree())
              _node_sizes = [12 + _in_degree.get(n, 0) * 6 for n in _G.nodes()]
              _node_colors = ["#BB0000" if _in_degree.get(n, 0) >= 2 else "#006600" if _G.out_degree(n) >= 2 else "#2563EB"
                              for n in _G.nodes()]

              _fig_kg = go.Figure()
              _fig_kg.add_trace(go.Scatter(
                x=_edge_x, y=_edge_y, mode="lines",
                line=dict(width=1, color="#aaa"), hoverinfo="none",
              ))
              _fig_kg.add_trace(go.Scatter(
                x=_node_x, y=_node_y,
                mode="markers+text",
                text=_node_labels,
                textposition="top center",
                textfont=dict(size=11),
                marker=dict(size=_node_sizes, color=_node_colors),
                hovertemplate=[
                  f"<b>{n}</b><br>drives {_G.out_degree(n)} variable(s)<br>driven by {_in_degree.get(n, 0)} variable(s)<extra></extra>"
                  for n in _G.nodes()
                ],
              ))
              _fig_kg.update_layout(
                height=380,
                margin=dict(l=0, r=0, t=8, b=0),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
              )
              _apply_plotly_numeric_font(_fig_kg)
              st.plotly_chart(_fig_kg, use_container_width=True)
              st.caption(
                "Node colour: red = highly influenced (many incoming links), "
                "green = strong driver (many outgoing links), blue = peer node. "
                "Node size grows with number of incoming influences."
              )
            except ImportError:
              st.caption("Install networkx to display the causal network graph: `pip install networkx`")
            except Exception as _kg_err:
              st.caption(f"Network graph could not be rendered: {_kg_err}")

          if not res.relationship_summary and not res.knowledge_graph and not _display_relationships:
            st.info("No relationships discovered yet — the engine needs more data rows to learn patterns.")

          try:
            _g_n_rels = len(_display_relationships)
            _g_n_high = sum(1 for _, _, _, c, _ in _display_relationships if c >= 0.75)
            _g_top_rel = _display_relationships[0] if _display_relationships else None
            _g_top_str = f"<b>{_g_top_rel[0]}</b> {_g_top_rel[1]} <b>{_g_top_rel[2]}</b> ({_g_top_rel[3]:.0%})" if _g_top_rel else "none yet"
            _g_conf = float(res.overall_confidence or 0.0)
          except Exception:
            _g_n_rels, _g_n_high, _g_top_str, _g_conf = 0, 0, "none yet", 0.0
          _guide(
            method_html=(
              "The <b>causal relationship analysis</b> runs the Scarcity Discovery Engine — an online machine learning "
              "system that processes each data row as a stream and builds a knowledge graph of statistical relationships "
              "between variables. It discovers multiple relationship types: <i>causal</i> (one variable drives another), "
              "<i>correlational</i> (co-movement without direction), <i>temporal_lag</i> (leading indicators), "
              "<i>equilibrium</i> (self-correcting pairs), and more. "
              "The <b>network graph</b> shows directed edges: <span style='color:#BB0000;'>red nodes</span> are most "
              "influenced (many incoming arrows), <span style='color:#006600;'>green nodes</span> are strong drivers "
              "(many outgoing arrows). Node size scales with incoming influence."
            ),
            interp_html=(
              f"<b>{_g_n_rels}</b> relationship(s) discovered; <b>{_g_n_high}</b> with high confidence (≥75%). "
              f"Strongest: {_g_top_str}. "
              f"Overall engine confidence: <b>{_g_conf:.0%}</b>. "
              "Green (driver) nodes are variables that appear to be upstream causes. "
              "Red (influenced) nodes are downstream outcomes — they respond to changes in the drivers."
            ),
            rec_html=(
              "Policy actions should target <b>green (driver) nodes</b> — intervening upstream is more effective than "
              "treating downstream symptoms. "
              "Relationships with ≥75% confidence are reliable enough to act on. "
              "Moderate-confidence (45–75%) relationships are hypotheses — corroborate with domain knowledge before escalating."
            ),
          )

        # ── 5. CROSS-SECTOR CORRELATION ─────────────────────────────────
        with st.expander("5. Cross-Sector Correlation — Which variables move together?"):
          _num_df = st.session_state['local_df'].select_dtypes(include=[np.number])
          if _num_df.shape[1] >= 2:
            _corr = _num_df.corr()
            fig_corr = px.imshow(_corr, color_continuous_scale="RdBu", zmin=-1, zmax=1)
            fig_corr.update_layout(height=380, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
            _apply_plotly_numeric_font(fig_corr)
            st.plotly_chart(fig_corr, use_container_width=True)
            # Highlight strongest non-diagonal pairs
            _strong = []
            cols_c = list(_corr.columns)
            for i, c1 in enumerate(cols_c):
              for j, c2 in enumerate(cols_c):
                if i < j:
                  v = _corr.loc[c1, c2]
                  if abs(v) > 0.6:
                    _strong.append({"Variable A": c1, "Variable B": c2, "Correlation": round(float(v), 3)})
            st.caption(narrate_correlation_findings(_strong, len(cols_c)))
            if _strong:
              st.write("**Strong inter-variable dependencies (|r| > 0.6):**")
              st.dataframe(sorted(_strong, key=lambda x: -abs(x["Correlation"])), use_container_width=True, hide_index=True)

            try:
              _g_n_strong = len(_strong)
              _g_top_corr = max(_strong, key=lambda x: abs(x["Correlation"]), default=None)
              _g_top_corr_str = (f"<b>{_g_top_corr['Variable A']}</b> ↔ <b>{_g_top_corr['Variable B']}</b> "
                                 f"(r = {_g_top_corr['Correlation']:+.3f})") if _g_top_corr else "none"
              _g_n_neg = sum(1 for s in _strong if s["Correlation"] < 0)
              _g_n_pos = _g_n_strong - _g_n_neg
            except Exception:
              _g_n_strong, _g_top_corr_str, _g_n_neg, _g_n_pos = 0, "none", 0, 0
            _guide(
              method_html=(
                "The <b>Pearson correlation matrix</b> measures linear co-movement between every pair of numeric "
                "variables. Values range from −1 (perfect inverse) to +1 (perfect co-movement). "
                "The heatmap uses a red–blue diverging scale: <b>deep red</b> = strong positive correlation, "
                "<b>deep blue</b> = strong negative correlation, white = no linear relationship. "
                "The diagonal is always 1.0 (each variable with itself)."
              ),
              interp_html=(
                f"<b>{_g_n_strong}</b> strong pair(s) (|r| > 0.6) found: "
                f"{_g_n_pos} positive, {_g_n_neg} negative. "
                f"Strongest: {_g_top_corr_str}. "
                "Highly correlated pairs move together and may share a common cause or one may be a lagged version of the other. "
                "Negative correlations indicate inverse dynamics — when one rises, the other falls."
              ),
              rec_html=(
                "Use strongly correlated pairs as <b>leading indicators</b>: if data on one variable is delayed, "
                "the other can serve as a proxy. "
                "Strong correlations do not imply causation — validate causal direction in the Causal Relationship tab. "
                "Near-perfect correlation (|r| > 0.95) may indicate redundant columns; consider removing one for cleaner analysis."
              ),
            )
          else:
            st.info("Need at least 2 numeric variables for correlation analysis.")

        # ── 6. DATA INTELLIGENCE SUMMARY ────────────────────────────────
        _util_df = st.session_state.get('local_df')
        _util_num_df = _util_df.select_dtypes(include=[np.number]) if isinstance(_util_df, pd.DataFrame) else pd.DataFrame()
        with st.expander("6. Data Intelligence Summary — How reliable is this analysis?"):
          if not _util_num_df.empty:
            _n_rows = len(_util_num_df)
            _n_cols = len(_util_num_df.columns)

            # ── Per-column data quality ──────────────────────────────────
            st.markdown("**Column-level data completeness and variability:**")
            _col_stats = []
            for _c in _util_num_df.columns:
              _s = _util_num_df[_c].dropna()
              _coverage_pct = len(_s) / max(1, _n_rows) * 100
              _cv = float(_s.std() / (_s.mean() + 1e-9)) if len(_s) > 1 else 0.0
              _trend_entry = next((t for t in (res.trend_signals or []) if t.get("column") == _c), None)
              _direction = _trend_entry.get("direction", "stable").title() if _trend_entry else "—"
              _col_stats.append({
                "Column": _c,
                "Coverage": f"{_coverage_pct:.0f}%",
                "Min": round(float(_s.min()), 3) if len(_s) else "—",
                "Max": round(float(_s.max()), 3) if len(_s) else "—",
                "Mean": round(float(_s.mean()), 3) if len(_s) else "—",
                "Variability (CV)": f"{abs(_cv):.2f}",
                "Trend": _direction,
              })
            st.dataframe(_col_stats, use_container_width=True, hide_index=True)

            # ── Overall readiness indicators ─────────────────────────────
            st.markdown("**Analysis readiness indicators:**")

            _total_cells = max(1, _n_rows * _n_cols)
            _overall_coverage = float(_util_num_df.notna().sum().sum()) / _total_cells
            _low_coverage_cols = [row["Column"] for row in _col_stats if float(row["Coverage"].rstrip("%")) < 70]

            _anomaly_rate = (len(res.structural_breaks or []) / max(1, _n_rows)) * 100
            _accel_count = sum(1 for t in (res.trend_signals or []) if t.get("direction") == "acceleration")
            _decel_count = sum(1 for t in (res.trend_signals or []) if t.get("direction") == "deceleration")
            _stable_count = len(res.trend_signals or []) - _accel_count - _decel_count

            def _readiness_bar(label, value_pct, note=""):
              _color = "#006600" if value_pct >= 80 else ("#F59E0B" if value_pct >= 50 else "#BB0000")
              _pct = int(min(100, max(0, value_pct)))
              _note_html = f"  <span style='font-size:0.78rem; color:#6B7280;'>{note}</span>" if note else ""
              st.markdown(
                f'<div style="margin-bottom:0.55rem;">'
                f'<span style="font-size:0.85rem; font-weight:600;">{label}</span> '
                f'<span style="font-size:0.85rem; color:{_color};">{_pct}%</span>'
                f'{_note_html}<br>'
                f'<div style="background:#eee; border-radius:4px; height:8px;">'
                f'<div style="background:{_color}; width:{_pct}%; height:8px; border-radius:4px;"></div>'
                f'</div></div>', unsafe_allow_html=True
              )

            _readiness_bar(
              "Overall data completeness",
              _overall_coverage * 100,
              f"{_n_rows} rows × {_n_cols} numeric columns"
              + (f" | Low-coverage: {', '.join(_low_coverage_cols[:3])}" if _low_coverage_cols else ""),
            )
            _readiness_bar(
              "Pattern stability (no structural breaks)",
              max(0.0, 100.0 - _anomaly_rate),
              f"{len(res.structural_breaks or [])} break point(s) detected — rows where the data pattern shifted significantly",
            )
            _readiness_bar(
              "Causal model confidence",
              min(100.0, float(res.overall_confidence or 0.0) * 100),
              f"How certain the relationship discovery engine is about the patterns it found",
            )
            if res.forecast_matrix and res.variance_matrix:
              _avg_var_flat = float(np.mean([abs(float(v)) for row in res.variance_matrix for v in row])) if res.variance_matrix else 0.0
              _fc_conf = max(0.0, min(100.0, 100.0 / (1.0 + _avg_var_flat)))
              _readiness_bar(
                "Forecast certainty",
                _fc_conf,
                "Lower forecast variance = more reliable projections",
              )

            # ── Trend summary ─────────────────────────────────────────────
            st.markdown(
              f"**Variable trend summary:** "
              f"{_accel_count} accelerating (growing), "
              f"{_decel_count} decelerating (shrinking), "
              f"{_stable_count} stable."
            )
            if _accel_count > _decel_count:
              st.success(f"More variables are growing than declining — overall sector trajectory is expanding.")
            elif _decel_count > _accel_count:
              st.warning(f"More variables are declining than growing — monitor closely for deterioration.")
            else:
              st.info("Growth and decline are balanced across variables.")

            if _low_coverage_cols:
              st.warning(
                f"**Data gaps detected** in: {', '.join(_low_coverage_cols)}. "
                "Columns with less than 70% coverage may produce unreliable causal and forecast results. "
                "Consider filling missing values or excluding these columns."
              )
          else:
            st.info("Upload a dataset with numeric columns to see the data intelligence summary.")

        # ── 7. RISK PROPAGATION ANALYSIS ────────────────────────────────
        with st.expander("7. Risk Propagation — Could this trigger a chain reaction?"):
          st.markdown(
            narrate_risk_propagation_overview(
              propagation_chains=res.propagation_chains,
              peak_score=res.peak_score,
              anomaly_present=bool(res.anomaly_scores),
            )
          )
          if res.propagation_chains:
            for chain_info in res.propagation_chains:
              st.markdown(narrate_propagation_chain(chain_info))
          else:
            st.info("No propagation chain output is available from this run.")

        # ── 8. FORECASTING ──────────────────────────────────────────────
        with st.expander("8. Forecasting — What might happen next?"):
          _series_names = [str(c).replace('_', ' ').title() for c in (res.columns or [])]
          st.markdown(
            narrate_forecast_overview(
              series_names=_series_names,
              forecast_matrix=res.forecast_matrix or [],
              variance_matrix=res.variance_matrix or [],
            )
          )
          if res.forecast_matrix and res.columns:
            _cols_f = res.columns[:min(4, len(res.columns))]
            _plot_cols = st.columns(min(2, len(_cols_f)))
            _local_hist = st.session_state['local_df']
            _num_dims = len(res.forecast_matrix[0]) if res.forecast_matrix else 0
            for _k, _cn in enumerate(_cols_f):
              if _k >= _num_dims:
                break
              with _plot_cols[_k % 2]:
                _hy = _local_hist[_cn].values[-80:] if _cn in _local_hist.columns else []
                _hx = list(range(len(_hy)))
                _ym = [step[_k] for step in res.forecast_matrix]
                _yv = [step[_k] for step in res.variance_matrix] if res.variance_matrix else [0] * len(_ym)
                _fx = list(range(len(_hy), len(_hy) + len(_ym)))
                _upper = np.array(_ym) + np.sqrt(np.abs(_yv))
                _lower = np.array(_ym) - np.sqrt(np.abs(_yv))
                _fig_f = go.Figure()
                _fig_f.add_scatter(x=_hx, y=_hy, mode='lines', line=dict(color='#888', width=2), name='Actual')
                _fig_f.add_scatter(x=_fx, y=_upper.tolist(), mode='lines', line=dict(width=0), showlegend=False)
                _fig_f.add_scatter(x=_fx, y=_lower.tolist(), mode='lines', fill='tonexty', fillcolor='rgba(0,180,180,0.2)', line=dict(width=0), showlegend=False)
                _fig_f.add_scatter(x=_fx, y=_ym, mode='lines', line=dict(color='#00AAAA', width=2), name='Forecast')
                _fig_f.update_layout(height=220, margin=dict(l=0,r=0,t=24,b=0),
                           title=_cn.replace('_', ' ').title(),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
                _apply_plotly_numeric_font(_fig_f)
                st.plotly_chart(_fig_f, use_container_width=True)
                st.caption(
                  narrate_forecast_direction(
                    series_name=_cn.replace('_', ' ').title(),
                    history=list(_hy),
                    forecast=list(_ym),
                    variance=list(_yv),
                  )
                )
          if res.forecast_matrix and res.columns:
            try:
              _g_steps = len(res.forecast_matrix)
              _g_avg_var = float(np.mean([abs(float(v)) for row in (res.variance_matrix or []) for v in row])) if res.variance_matrix else 0.0
              _g_fc_conf = max(0.0, min(100.0, 100.0 / (1.0 + _g_avg_var)))
              _g_conf_label = "high" if _g_fc_conf >= 75 else "moderate" if _g_fc_conf >= 50 else "low"
              _g_top_fc_col = str(res.columns[0]).replace("_", " ").title() if res.columns else "unknown"
            except Exception:
              _g_steps, _g_fc_conf, _g_conf_label, _g_top_fc_col = 0, 0, "low", "unknown"
            _guide(
              method_html=(
                "The <b>VARX (Vector AutoRegression with eXogenous inputs)</b> model forecasts all numeric variables "
                "jointly, capturing interdependencies between them. It learns from the historical pattern up to the last "
                "observation and projects forward. The <b>cyan band</b> is the ±1 standard deviation confidence interval "
                "(derived from the forecast variance matrix). Narrower bands = higher certainty; wider bands = the model "
                "is less sure, usually when data is scarce, noisy, or has had structural breaks."
              ),
              interp_html=(
                f"Forecast horizon: <b>{_g_steps} step(s)</b> ahead. "
                f"Average forecast variance: <b>{_g_avg_var:.3f}</b> — confidence level: <b>{_g_conf_label}</b> ({_g_fc_conf:.0f}%). "
                f"Leading variable in display: <b>{_g_top_fc_col}</b>. "
                "Wide confidence bands indicate the model is extrapolating into uncertain territory — treat projections as directional, not precise."
              ),
              rec_html=(
                "Use the forecast <b>direction</b> (rising vs falling trend line) to guide near-term decisions. "
                "Do not act on exact forecast values when confidence is low — instead, plan for the range shown by the band. "
                "If structural breaks were detected, the forecast is trained on a shifted regime; consider reducing the history window "
                "to the post-break period for more relevant projections."
              ),
            )
          else:
            st.info("Forecasting requires the VARX engine (needs scarcity.engine.forecasting). Run more data rows for best results.")

        # ── SEND TO SECTOR ADMIN ─────────────────────────────────────────
        st.write("---")
        if res.peak_score > 2.0:
          st.error(
            f"Critical anomaly detected (score: {res.peak_score:.2f}, threat level: {res.threat_level}). "
            "An automatic report has been sent to your sector admin."
          )
          _insight = {
            "incident_type": "SEVERE_STRUCTURAL_DRIFT",
            "detection_engine": res.engine_used,
            "severity_score": round(res.peak_score, 2),
            "composite_scores": res.composite,
            "threat_level": res.threat_level,
            "sensitivity": st.session_state.get('data_sensitivity', 'Public'),
            "time_index": res.peak_index,
            "structural_breaks": res.structural_breaks[:10],
            "relationships_found": len(res.relationships),
            "overall_confidence": round(res.overall_confidence, 3),
            "top_relationship": res.relationship_summary[0] if res.relationship_summary else "",
            "spoke_interpretation": res.narrative,
            "local_weights": [r.get('confidence', 0.5) for r in res.relationships[:10]],
          }
          _sync_id = DeltaSyncManager.queue_insight(inst_id, basket_id, _insight)
          st.success(
            f"Anomaly report sent to your sector admin (Report ID: {_sync_id}). "
            "They will review and decide whether to escalate nationally."
          )
        elif res.peak_score > 0:
          if st.button("Send this analysis to your sector admin", use_container_width=True):
            _insight = {
              "incident_type": "MANUAL_REPORT",
              "detection_engine": res.engine_used,
              "severity_score": round(res.peak_score, 2),
              "composite_scores": res.composite,
              "threat_level": res.threat_level,
              "sensitivity": st.session_state.get('data_sensitivity', 'Public'),
              "relationships_found": len(res.relationships),
              "overall_confidence": round(res.overall_confidence, 3),
              "spoke_interpretation": res.narrative,
              "local_weights": [r.get('confidence', 0.5) for r in res.relationships[:10]],
            }
            _sync_id = DeltaSyncManager.queue_insight(inst_id, basket_id, _insight)
            st.success(f"Report sent (Report ID: {_sync_id}).")

  _causal_views = {"Granger Causality", "Causal Network", "Cross-Correlations"}
  if active_section in _causal_views:
    if st.session_state.get('local_df') is not None:
      causal_df = st.session_state['local_df']
      # Use ALL numeric columns from the uploaded data.
      # Schema required_columns are used as a priority ordering only — columns not
      # in the schema but present in the data are still included at the end.
      _all_numeric = [c for c in causal_df.columns if pd.api.types.is_numeric_dtype(causal_df[c])]
      _schema_cols = schema.get("required_columns", []) if schema else []
      _schema_priority = [c for c in _schema_cols if c in _all_numeric]
      _extra_cols = [c for c in _all_numeric if c not in _schema_priority]
      causal_cols = _schema_priority + _extra_cols

      if not causal_cols:
        st.warning("No numeric columns found in the uploaded data for causal analysis.")
      elif active_section == "Granger Causality":
        _render_granger_section(causal_df, causal_cols, LIGHT_THEME)
      elif active_section == "Causal Network":
        _render_causal_network(causal_df, causal_cols, LIGHT_THEME)
      elif active_section == "Cross-Correlations":
        _render_cross_corr(causal_df, causal_cols, LIGHT_THEME)
    else:
      st.warning("Upload data first to run causal analysis.")

  if active_section == "Effect Estimation":
    st.markdown("### Effect Estimation")
    st.write(
      "Estimate the **direct causal effect** of one indicator on another — going beyond "
      "correlation and Granger tests to structural causal inference."
    )
    _est_df = st.session_state.get('local_df')
    if _est_df is None or _est_df.empty:
      st.warning("Upload your institution’s data in the **Data Intake** tab to enable effect estimation.")
    else:
      _num_cols = [
        c for c in _est_df.columns
        if pd.api.types.is_numeric_dtype(_est_df[c]) and _est_df[c].notna().sum() >= 10
      ]
      if len(_num_cols) < 2:
        st.info("At least two numeric columns with sufficient observations are needed.")
      else:
        from kshiked.ui.kshield.causal import render_causal_evidence_panel
        _ec1, _ec2 = st.columns(2)
        with _ec1:
          _est_treatment = st.selectbox("Cause", _num_cols, key="spoke_ce_treatment")
        with _ec2:
          _est_out_opts = [c for c in _num_cols if c != _est_treatment]
          _est_outcome = st.selectbox("Effect", _est_out_opts, key="spoke_ce_outcome")
        if _est_treatment and _est_outcome:
          render_causal_evidence_panel(
            df=_est_df,
            treatment=_est_treatment,
            outcome=_est_outcome,
            theme=LIGHT_THEME,
            key_prefix="spoke_ce",
          )

  if active_section == "Active Projects":
    with st.container(border=True):
      st.markdown("### Operational Projects")
      st.write("Track active cross-sector projects, view shared goals, and update milestones assigned to your sector.")
      from kshiked.ui.institution.backend.project_manager import ProjectManager
      active_projects = ProjectManager.get_active_projects(basket_id)
      
      if not active_projects:
        st.success("No active operational projects demand your attention at this time.")
      else:
        for proj in active_projects:
          with st.expander(f"WAR ROOM: {proj['title']} (Severity: {proj['severity']})", expanded=True):
            project_data = ProjectManager.get_project_details(proj['id'])
            
            st.write("---")
            st.markdown("#### Structured Project Health")
            from kshiked.ui.institution.project_components import render_project_overview
            render_project_overview(project_data, "Spoke", basket_id, all_baskets)

  if active_section == "Inbox":
    with st.container(border=True):
      st.markdown("### Communications")
      st.write("Send messages to your sector admin or read incoming directives.")

      msg_col1, msg_col2 = st.columns([1, 1])
      with msg_col1:
        st.write("**Send a message to your sector admin**")
        msg_content = st.text_area(
          "Message",
          height=100,
          placeholder="Describe an anomaly, request clarification, or flag a data quality issue...",
          key="spoke_msg_out"
        )
        if st.button("Send to Sector Admin", type="primary", use_container_width=True):
          if msg_content:
            SecureMessaging.send_message(
              sender_role=Role.INSTITUTION.value,
              sender_id=st.session_state.get('username'),
              receiver_role=Role.BASKET_ADMIN.value,
              receiver_id=str(basket_id),
              content=msg_content
            )
            st.success("Dispatch routed successfully.")
          else:
            st.error("Cannot dispatch an empty payload.")
      
      with msg_col2:
        st.write("**Incoming directives**")
        
        # Fetch Downward Directives (Benchmarks/Rules from Exec/Admin)
        directives = DataSharingManager.get_directives_for_spoke(inst_id)
        if directives:
          st.markdown("##### Official Mandates & Benchmarks")
          for d in directives:
            req_ack = " (Requires Action)" if not d['is_acknowledged'] and d['requires_ack'] else ""
            bg = "#FEF2F2" if not d['is_acknowledged'] else "#F3F4F6"
            border = "#DC2626" if not d['is_acknowledged'] else "#9CA3AF"
            
            with st.container():
              st.markdown(f"""
              <div style="background:{bg}; border-left:4px solid {border}; padding:10px 14px; border-radius:4px; margin-bottom:10px;">
                <strong>[{d['directive_type']}] Priority: {d['priority']}</strong>{req_ack}<br/>
                <span style="font-size:0.9rem;">{d['content']}</span>
              </div>
              """, unsafe_allow_html=True)
              if not d['is_acknowledged'] and d['requires_ack']:
                if st.button("Acknowledge Target", key=f"ack_dir_{d['id']}"):
                  DataSharingManager.acknowledge_directive(d['id'], f"Spoke_{inst_id}")
                  st.success("Target acknowledged. Telemetry thresholds updated.")
                  st.rerun()
          st.write("---")

        # Regular Messages
        st.markdown("##### General Messages")
        inbox = SecureMessaging.get_inbox(Role.INSTITUTION.value, st.session_state.get('username'))
        if not inbox and not directives:
          st.caption("No pending directives or messages.")
        elif inbox:
          for msg in inbox:
            sender_badge = "National Command" if msg['sender_role'] == Role.EXECUTIVE.value else "Sector Admin"
            with st.expander(f"{sender_badge} | {msg['timestamp']} {'(NEW)' if not msg['is_read'] else ''}"):
              st.write(msg['content'])
              if not msg['is_read']:
                if st.button("Acknowledge", key=f"ack_{msg['id']}"):
                  SecureMessaging.mark_read(msg['id'])
                  st.rerun()

  if active_section == "Collaboration Room":
    with get_connection() as _conn_cr:
      _c_cr = _conn_cr.cursor()
      _c_cr.execute("SELECT id, name FROM baskets")
      _all_baskets_cr = {r['id']: r['name'] for r in _c_cr.fetchall()}
    render_collab_room(
      role=Role.INSTITUTION.value,
      basket_id=basket_id,
      username=st.session_state.get('username', 'spoke'),
      all_baskets=_all_baskets_cr,
    )

  if active_section == "Model Configuration":
    st.markdown("### Local Model and Privacy Settings")
    st.write(
      "Configure analysis sensitivity and federated reporting behavior before running signal processing. "
      "These controls apply only to your institution workspace."
    )

    sensitivity_default = st.session_state.get("data_sensitivity", "Restricted")
    sensitivity = st.selectbox(
      "Default Sensitivity for New Analyses",
      ["Public", "Restricted", "Confidential"],
      index=["Public", "Restricted", "Confidential"].index(sensitivity_default) if sensitivity_default in ["Public", "Restricted", "Confidential"] else 1,
      key="model_cfg_sensitivity",
    )
    st.session_state["data_sensitivity"] = sensitivity

    mode_b = st.toggle(
      "Enable Federated Learning (Mode B)",
      value=st.session_state.get('fl_mode_enabled', False),
      help="Mode A shares governance summaries. Mode B shares model gradients only; raw rows stay local.",
      key="model_cfg_fl_mode",
    )
    st.session_state['fl_mode_enabled'] = mode_b

    if sensitivity == "Confidential" and not mode_b:
      st.error("Confidential sensitivity requires Mode B. Enable Federated Learning to proceed with confidential workloads.")
    elif mode_b:
      st.success("Mode B enabled. Local data remains on this node and only weight summaries are shared.")
    else:
      st.info("Mode A active. Governance summaries are available for sector oversight.")

  if active_section == "FL Training Log":
      st.markdown("### FL Training Log (Mode B)")
      st.info(
        "**Federated Learning is active.** When an anomaly is detected above the threshold, your institution "
        "automatically submits a mathematical summary (model weights) to your sector admin. "
        "Raw data never leaves this node."
      )
      st.write("---")

      historical_syncs = DeltaSyncManager.get_historical_syncs(inst_id)
      if not historical_syncs:
        st.caption("No training submissions yet. Run Signal Analysis with data uploaded to trigger your first submission.")
      else:
        st.write(f"**{len(historical_syncs)} submission(s) on record.**")
        for s in reversed(historical_syncs[-20:]):
          ts = pd.to_datetime(s.get('created_at', 0), unit='s').strftime('%Y-%m-%d %H:%M') if s.get('created_at') else '—'
          status = s.get('status', 'PENDING')
          status_color = "#006600" if status == 'PROCESSED' else "#F59E0B" if status == 'PENDING' else "#BB0000"
          payload = s.get('payload', {})
          scores = payload.get('composite_scores', {})
          with st.expander(f"{ts} | Status: {status} | Detection: {scores.get('A_Detection', '—')} Impact: {scores.get('B_Impact', '—')}"):
            st.write(f"**Severity score at submission:** {payload.get('severity_score', '—')}")
            st.write(f"**Incident type:** {payload.get('incident_type', '—')}")
            if scores:
              m1, m2, m3 = st.columns(3)
              m1.metric("Detection", f"{scores.get('A_Detection', 0):.1f}/10")
              m2.metric("Impact", f"{scores.get('B_Impact', 0):.1f}/10")
              m3.metric("Certainty", f"{scores.get('C_Certainty', 0):.1f}/10")
            if payload.get('spoke_interpretation'):
              st.caption(f"Local interpretation: {payload['spoke_interpretation'][:300]}")
            st.markdown(f"<span style='color:{status_color}; font-size:0.8rem;'>Sector admin status: **{status}**</span>", unsafe_allow_html=True)

  # with tab_research:
  #   from kshiked.ui.institution.backend.research_engine import ResearchEngine, EngineContext
  #   from kshiked.ui.institution.research_components import render_research_engine_panel
  #   
  #   ctx = EngineContext(role="spoke", user_id=st.session_state.get('username'), sector_id=basket_id)
  #   engine = ResearchEngine(context=ctx)
  #   render_research_engine_panel(engine)
