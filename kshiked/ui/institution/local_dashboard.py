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

  # Fetch all baskets for cross-sector references
  with get_connection() as conn:
    c = conn.cursor()
    c.execute("SELECT id, name FROM baskets")
    all_baskets = {r['id']: r['name'] for r in c.fetchall()}

  # Fetch schemas early so they are available across all tabs
  schema = OntologyEnforcer.get_basket_schema(basket_id)
  from kshiked.ui.institution.backend.schema_manager import SchemaManager
  custom_schemas = SchemaManager.get_schemas(basket_id)
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
          # Plain-language summary first
          st.markdown(narrate_anomaly_detection(res.peak_score, res.structural_breaks))
          if res.anomaly_scores:
            st.line_chart(res.anomaly_scores)
            st.caption(
              narrate_anomaly_chart_stats(
                anomaly_scores=res.anomaly_scores,
                peak_score=res.peak_score,
                peak_index=res.peak_index,
                structural_breaks=res.structural_breaks,
              )
            )
          else:
            st.info("No anomaly scores computed — check that your data has numeric columns.")

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
            _rel_type = str(rel.get("rel_type", "related_to")).replace("_", " ")
            _display_relationships.append(
              f"**{vars_[0]}** { _rel_type } **{vars_[1]}** (confidence: {_confidence:.0%})"
            )
            if len(_display_relationships) >= 8:
              break

          if _display_relationships:
            for sentence in _display_relationships:
              st.markdown(f"- {sentence}")
          elif res.relationship_summary:
            for sentence in res.relationship_summary[:8]:
              st.markdown(f"- {sentence}")
          if res.knowledge_graph:
            st.write("**Causal Network Graph**")
            try:
              import networkx as nx
              G = nx.DiGraph()
              for edge in res.knowledge_graph[:40]:
                src = edge.get('source') or (edge.get('variables', ['?', '?'])[0])
                tgt = edge.get('target') or (edge.get('variables', ['?', '?'])[-1])
                G.add_edge(src, tgt, weight=edge.get('confidence', 0.5))
              pos = nx.spring_layout(G, seed=42)
              edge_x, edge_y = [], []
              for u, v in G.edges():
                x0, y0 = pos[u]; x1, y1 = pos[v]
                edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
              fig_kg = go.Figure()
              fig_kg.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1, color='#aaa'), hoverinfo='none'))
              fig_kg.add_trace(go.Scatter(
                x=[pos[n][0] for n in G.nodes()], y=[pos[n][1] for n in G.nodes()],
                mode='markers+text', text=list(G.nodes()), textposition='top center',
                marker=dict(size=10, color='#006600'), hoverinfo='text'
              ))
              fig_kg.update_layout(height=340, margin=dict(l=0,r=0,t=0,b=0), showlegend=False,
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
              _apply_plotly_numeric_font(fig_kg)
              st.plotly_chart(fig_kg, use_container_width=True)
            except Exception:
              st.caption("Install networkx for causal network graph: pip install networkx")
          if not res.relationship_summary and not res.knowledge_graph:
            st.info("No relationships discovered yet — the engine needs more data rows to learn patterns.")

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
          else:
            st.info("Need at least 2 numeric variables for correlation analysis.")

        # ── 6. RESOURCE UTILIZATION (SECTOR-SPECIFIC) ───────────────────
        _util_df = st.session_state.get('local_df')
        _util_num_df = _util_df.select_dtypes(include=[np.number]) if isinstance(_util_df, pd.DataFrame) else pd.DataFrame()
        with st.expander("6. Resource Utilization Analysis"):
          if not _util_num_df.empty:
            def _clip01(value):
              try:
                return max(0.0, min(1.0, float(value)))
              except Exception:
                return 0.0

            def _quantile_normalize(value, reference):
              _v = _clip01(value)
              _ref = []
              for _r in (reference or []):
                try:
                  _ref.append(_clip01(_r))
                except Exception:
                  continue
              if len(_ref) < 5:
                return _v
              _q10 = float(np.quantile(_ref, 0.10))
              _q90 = float(np.quantile(_ref, 0.90))
              if _q90 <= _q10 + 1e-9:
                return _v
              return _clip01((_v - _q10) / (_q90 - _q10))

            _n_rows = len(_util_num_df)
            _n_cols = len(_util_num_df.columns)
            _total_cells = max(1, _n_rows * _n_cols)
            _coverage = float(_util_num_df.notna().sum().sum()) / _total_cells
            _signal_stability = 1.0 / (1.0 + max(0.0, float(res.peak_score)))
            _regime_consistency = max(0.0, 1.0 - (len(res.structural_breaks or []) / max(1, _n_rows)))
            _trend_total = max(1, len(res.trend_signals or []))
            _trend_accel = sum(1 for t in (res.trend_signals or []) if t.get("direction") == "acceleration")
            _trend_decel = sum(1 for t in (res.trend_signals or []) if t.get("direction") == "deceleration")
            _operational_balance = max(0.0, 1.0 - (abs(_trend_accel - _trend_decel) / _trend_total))
            _model_confidence = max(0.0, min(1.0, float(res.overall_confidence or 0.0)))

            _avg_var = None
            if res.variance_matrix:
              _vals = []
              for _row in res.variance_matrix:
                for _v in _row:
                  try:
                    _vals.append(abs(float(_v)))
                  except Exception:
                    continue
              if _vals:
                _avg_var = sum(_vals) / len(_vals)
            _forecast_confidence = (1.0 / (1.0 + _avg_var)) if _avg_var is not None else None

            _coverage_ref = [
              float(_util_num_df[c].notna().mean())
              for c in _util_num_df.columns
            ]
            _stability_ref = [
              1.0 / (1.0 + max(0.0, float(s)))
              for s in (res.anomaly_scores or [])
            ]

            _valid_breaks = sorted(set(
              int(b) for b in (res.structural_breaks or [])
              if isinstance(b, (int, float)) and 0 <= int(b) <= _n_rows
            ))
            _regime_ref = []
            if _valid_breaks:
              _pts = [0] + _valid_breaks + [_n_rows]
              for _i in range(len(_pts) - 1):
                _gap = max(0, _pts[_i + 1] - _pts[_i])
                _regime_ref.append(_gap / max(1, _n_rows))

            _trend_ref = [
              1.0 / (1.0 + abs(float(t.get("growth_rate", 0.0))) * 10.0)
              for t in (res.trend_signals or [])
            ]

            _confidence_ref = [float(v) for v in (res.confidence_map or {}).values()]
            if not _confidence_ref:
              _confidence_ref = [
                float(r.get("confidence", 0.0))
                for r in (res.relationships or [])
              ]

            _forecast_ref = []
            if res.variance_matrix:
              for _row in res.variance_matrix:
                _row_vals = []
                for _v in _row:
                  try:
                    _row_vals.append(abs(float(_v)))
                  except Exception:
                    continue
                if _row_vals:
                  _forecast_ref.append(1.0 / (1.0 + (sum(_row_vals) / len(_row_vals))))

            _metrics = {
              "Data Coverage": _quantile_normalize(_coverage, _coverage_ref),
              "Signal Stability": _quantile_normalize(_signal_stability, _stability_ref),
              "Regime Consistency": _quantile_normalize(_regime_consistency, _regime_ref),
              "Operational Balance": _quantile_normalize(_operational_balance, _trend_ref),
              "Model Confidence": _quantile_normalize(_model_confidence, _confidence_ref),
            }
            if _forecast_confidence is not None:
              _metrics["Forecast Confidence"] = _quantile_normalize(_forecast_confidence, _forecast_ref)

            _base_weights = {
              "Data Coverage": 0.22,
              "Signal Stability": 0.20,
              "Regime Consistency": 0.16,
              "Operational Balance": 0.16,
              "Model Confidence": 0.16,
              "Forecast Confidence": 0.10,
            }

            _sample_quality = min(1.0, _n_rows / 80.0) * min(1.0, _n_cols / 8.0)
            _trend_quality = min(1.0, len(res.trend_signals or []) / max(1, _n_cols))
            _forecast_quality = 1.0 if "Forecast Confidence" in _metrics else 0.0
            _weights = {}
            for _name in _metrics:
              _w = _base_weights.get(_name, 0.0)
              if _name in {"Signal Stability", "Regime Consistency"}:
                _w *= max(0.35, _sample_quality)
              if _name == "Operational Balance":
                _w *= max(0.35, _trend_quality)
              if _name == "Forecast Confidence":
                _w *= _forecast_quality
              _weights[_name] = _w

            _w_total = sum(_weights.values())
            if _w_total <= 1e-9:
              _weights = {k: 1.0 / len(_metrics) for k in _metrics}
            else:
              _weights = {k: (v / _w_total) for k, v in _weights.items()}

            _util_score = sum(_metrics[k] * _weights[k] for k in _metrics)
            _util_status = "Healthy" if _util_score >= 0.66 else "Watch" if _util_score >= 0.4 else "Constrained"

            _weight_rows = []
            for _name, _val in _metrics.items():
              _w = float(_weights.get(_name, 0.0))
              _weight_rows.append({
                "Metric": _name,
                "Normalized Score": round(float(_val), 3),
                "Adaptive Weight": round(_w, 3),
                "Contribution": round(float(_val) * _w, 3),
              })
            _weight_rows = sorted(_weight_rows, key=lambda x: x["Contribution"], reverse=True)

            st.caption(
              f"Sector resource utilization is derived from this sector's uploaded data ({_n_rows} rows, {_n_cols} numeric variables) "
              "and model outputs from anomaly, trend, relationship, and forecast modules."
            )
            st.markdown(
              f"**Sector Utilization Score:** {_util_score:.2f} ({_util_status})"
            )
            st.caption("Computation rule: Sector Utilization Score = sum(Normalized Score x Adaptive Weight).")

            def _util_bar(label, value, fmt=".1%"):
              color = "#006600" if value > 0.66 else ("#F59E0B" if value > 0.4 else "#BB0000")
              pct = int(min(100, max(0, value * 100)))
              st.markdown(
                f'<div style="margin-bottom:0.5rem;">'
                f'<span style="font-size:0.85rem; font-weight:600;">{label}</span> '
                f'<span style="font-size:0.85rem; color:{color};">{value:{fmt}}</span><br>'
                f'<div style="background:#eee; border-radius:4px; height:8px;">'
                f'<div style="background:{color}; width:{pct}%; height:8px; border-radius:4px;"></div>'
                f'</div></div>', unsafe_allow_html=True
              )

            for _name, _val in _metrics.items():
              _util_bar(_name, _val)

            st.write("**Score Composition**")
            st.dataframe(_weight_rows, use_container_width=True, hide_index=True)

            st.caption("Scores are quantile-normalized from this run and weighted adaptively by available evidence quality.")
          else:
            st.info("Resource utilization requires numeric sector columns in the uploaded dataset.")

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
    if st.session_state.get('local_df') is not None and schema is not None:
      causal_df = st.session_state['local_df']
      _raw_cols = schema.get("required_columns", list(causal_df.columns))
      causal_cols = [c for c in _raw_cols if c in causal_df.columns and pd.api.types.is_numeric_dtype(causal_df[c])]

      if active_section == "Granger Causality":
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
