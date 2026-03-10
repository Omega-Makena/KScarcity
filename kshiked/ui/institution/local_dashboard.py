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
from kshiked.ui.institution.collab_room import render_collab_room
from kshiked.ui.kshield.causal.view import (
    _render_granger_section,
    _render_causal_network,
    _render_cross_corr
)
from kshiked.ui.institution.backend.report_narrator import (
    narrate_composite_scores,
    narrate_severity,
    narrate_threat_level,
    narrate_shock_vector,
    narrate_economic_state,
    narrate_anomaly_detection,
    narrate_trend_analysis,
    narrate_propagation_chain,
    get_threat_index_explanation,
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

def render():
    enforce_role(Role.INSTITUTION.value)
    inject_enterprise_theme()

    inst_id = st.session_state.get('institution_id')
    basket_id = st.session_state.get('basket_id')
    fl_mode = st.session_state.get('fl_mode_enabled', False)

    # Human-readable institution and sector names
    inst_name = _get_institution_name(inst_id) if inst_id else st.session_state.get('username', 'Institution')
    basket_name = _get_basket_name(basket_id) if basket_id else f"Sector {basket_id}"

    st.markdown(f"<h2 style='text-align: center; color: #1F2937;'>{inst_name}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='text-align: center; color: #006600;'>Reporting to: {basket_name} &nbsp;|&nbsp; User: {st.session_state.get('username')}</h5>", unsafe_allow_html=True)
    st.write("---")

    # Fetch schemas early so they are available across all tabs
    schema = OntologyEnforcer.get_basket_schema(basket_id)
    from kshiked.ui.institution.backend.schema_manager import SchemaManager
    custom_schemas = SchemaManager.get_schemas(basket_id)
    latest_custom_schema = custom_schemas[0] if custom_schemas else None

    _base_tabs = ["Data Intake", "Signal Analysis", "Granger Causality", "Causal Network", "Cross-Correlations", "Active Projects", "Inbox", "Collaboration Room"]
    if fl_mode:
        _base_tabs.append("FL Training Log")
    _all_spoke_tabs = st.tabs(_base_tabs)
    tab1         = _all_spoke_tabs[0]
    tab2         = _all_spoke_tabs[1]
    tab_granger  = _all_spoke_tabs[2]
    tab_network  = _all_spoke_tabs[3]
    tab_cross    = _all_spoke_tabs[4]
    tab_projects = _all_spoke_tabs[5]
    tab3         = _all_spoke_tabs[6]
    tab_collab   = _all_spoke_tabs[7]
    tab_fl_log   = _all_spoke_tabs[8] if fl_mode else None
    
    with tab1:
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

    with tab2:
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
                    "**Confidential data requires Federated Learning (Mode B).**  \n"
                    "Enable **Mode B** using the toggle in the sidebar. "
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
                    st.write("")
                    st.caption(
                        "**Technical detail:** RRCF streaming anomaly detection. Each row is scored in real time. "
                        "Score > 2.0 = structural shift. Score > 1.0 = moderate deviation."
                    )
                    if res.anomaly_scores:
                        st.line_chart(res.anomaly_scores)
                        st.caption(f"Peak: **{res.peak_score:.2f}** at row {res.peak_index}. "
                                   f"Structural breaks detected at rows: {res.structural_breaks[:10] if res.structural_breaks else 'none'}.")
                    else:
                        st.info("No anomaly scores computed — check that your data has numeric columns.")

                # ── 2. TEMPORAL TREND ANALYSIS ──────────────────────────────────
                with st.expander("2. Temporal Trend Analysis — Are things getting better or worse?"):
                    if res.trend_signals:
                        # Plain-language summary first
                        st.markdown(narrate_trend_analysis(res.trend_signals))
                        st.write("")
                        st.caption(
                            "**Technical detail:** Compares first half vs second half of your data per variable. "
                            "Detects acceleration, deceleration, volatility increase, and structural breaks."
                        )
                        _dir_icons = {"acceleration": "📈", "deceleration": "📉", "stable": "➡️"}
                        _vol_icons = {"increasing volatility": "⚡", "decreasing volatility": "🔇", "stable": "〰️"}
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
                        st.caption("Geographic hotspot detection. Columns named 'lat'/'lon' were automatically detected.")
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
                            st.plotly_chart(fig_map, use_container_width=True)
                        except Exception:
                            st.caption("Geographic hotspots identified (map requires plotly.express with mapbox).")
                    else:
                        st.info(
                            "No geographic columns detected. To enable spatial analysis, include columns named "
                            "**lat** and **lon** (or **lng**) in your dataset. "
                            "The engine will then compute geographic hotspot clusters automatically."
                        )
                        st.caption(
                            "Supported clustering methods (when spatial data is provided): "
                            "grid-based density estimation, DBSCAN-style hotspot detection, Moran's I spatial autocorrelation."
                        )

                # ── 4. CAUSAL RELATIONSHIP ANALYSIS ────────────────────────────
                with st.expander("4. Causal Relationship Analysis — What is causing what?"):
                    st.markdown(
                        f"The system tested **{res.hypotheses_total} possible cause-and-effect relationships** "
                        f"in your data and found **{res.hypotheses_active} patterns** that appear to be real "
                        f"(confidence: {res.overall_confidence:.0%}). "
                        "These are statistically detected patterns — they suggest connections between variables "
                        "but don't prove one causes the other."
                    )
                    if res.relationship_summary:
                        for sentence in res.relationship_summary:
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
                            st.plotly_chart(fig_kg, use_container_width=True)
                        except Exception:
                            st.caption("Install networkx for causal network graph: pip install networkx")
                    if not res.relationship_summary and not res.knowledge_graph:
                        st.info("No relationships discovered yet — the engine needs more data rows to learn patterns.")

                # ── 5. CROSS-SECTOR CORRELATION ─────────────────────────────────
                with st.expander("5. Cross-Sector Correlation — Which variables move together?"):
                    st.markdown(
                        "This shows how your data variables relate to each other. "
                        "**Red** = they tend to rise and fall together. "
                        "**Blue** = when one goes up, the other goes down. "
                        "Strong connections (above 0.6) are listed below the map."
                    )
                    _num_df = st.session_state['local_df'].select_dtypes(include=[np.number])
                    if _num_df.shape[1] >= 2:
                        _corr = _num_df.corr()
                        fig_corr = px.imshow(_corr, color_continuous_scale="RdBu", zmin=-1, zmax=1)
                        fig_corr.update_layout(height=380, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)")
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
                        if _strong:
                            st.write("**Strong inter-variable dependencies (|r| > 0.6):**")
                            st.dataframe(sorted(_strong, key=lambda x: -abs(x["Correlation"])), use_container_width=True, hide_index=True)
                    else:
                        st.info("Need at least 2 numeric variables for correlation analysis.")

                # ── 6. SENTIMENT & SOCIAL SIGNALS ───────────────────────────────
                with st.expander("6. Public Safety & Social Signal Monitor"):
                    st.markdown(
                        "These 8 indices measure different dimensions of national stability — from public opinion "
                        "to economic resilience to security conditions. Each one tracks a specific type of risk."
                    )
                    if res.threat_report:
                        _tr = res.threat_report
                        _level_bg = {"CRITICAL": "#BB0000", "HIGH": "#E05000", "ELEVATED": "#F59E0B", "GUARDED": "#2563EB", "LOW": "#006600"}
                        _lbg = _level_bg.get(_tr.get("overall_threat_level", "LOW"), "#006600")

                        # Plain-language threat level explanation
                        st.markdown(narrate_threat_level(_tr.get("overall_threat_level", "LOW")))

                        st.markdown(
                            f'<div style="background:{_lbg}22; border:2px solid {_lbg}; border-radius:8px; padding:0.8rem; margin-bottom:1rem; text-align:center;">'
                            f'<span style="color:{_lbg}; font-size:1.4rem; font-weight:bold;">Overall Threat: {_tr.get("overall_threat_level")}</span>'
                            f'</div>', unsafe_allow_html=True
                        )
                        _indices = _tr.get("indices", {})
                        _index_labels = {
                            "polarization": "Polarization Index (PI)",
                            "legitimacy_erosion": "Legitimacy Erosion (LEI)",
                            "mobilization_readiness": "Mobilization Readiness (MRS)",
                            "elite_cohesion": "Elite Cohesion (ECI)",
                            "information_warfare": "Information Warfare (IWI)",
                            "security_friction": "Security Friction (SFI)",
                            "economic_cascade": "Economic Cascade Risk (ECR)",
                            "ethnic_tension": "Ethnic Tension Matrix (ETM)",
                        }
                        _ia_cols = st.columns(2)
                        for _i, (key, label) in enumerate(_index_labels.items()):
                            _idx_data = _indices.get(key, {})
                            _val = _idx_data.get("value") or _idx_data.get("avg_tension", 0.0)
                            _sev = _idx_data.get("severity", "LOW")
                            _sev_color = _level_bg.get(_sev, "#888")
                            with _ia_cols[_i % 2]:
                                explanation = get_threat_index_explanation(key)
                                st.markdown(
                                    f'<div style="border-left:3px solid {_sev_color}; padding:0.3rem 0.6rem; margin-bottom:0.4rem;">'
                                    f'<b style="font-size:0.85rem;">{label}</b><br>'
                                    f'<span style="font-size:1.1rem; color:{_sev_color};">{_val:.2f}</span>'
                                    f'<span style="font-size:0.75rem; color:{_sev_color}; margin-left:0.4rem;">{_sev}</span>'
                                    f'</div>', unsafe_allow_html=True
                                )
                                if explanation:
                                    st.caption(explanation)
                        _alerts = _tr.get("priority_alerts", [])
                        if _alerts:
                            st.write("**Priority Alerts:**")
                            for a in _alerts:
                                st.error(a)
                        else:
                            st.success("No priority alerts at current threat level.")
                    else:
                        st.info("Threat index computation is not available — ensure kshiked.pulse is installed.")

                # ── 7. POLICY IMPACT ANALYSIS ───────────────────────────────────
                with st.expander("7. National Economic Health — What's the economy doing?"):
                    st.markdown(
                        "This uses a mathematical model of the Kenyan economy to assess overall health. "
                        "It simulates how households, businesses, banks, and government interact."
                    )
                    if res.economic_state:
                        _es = res.economic_state
                        # Plain-language economic narrative first
                        st.markdown(
                            f'<div style="background:#F0F9FF; border-left:4px solid #3B82F6; padding:10px 14px; '
                            f'border-radius:0 6px 6px 0; margin:8px 0;">'
                            f'{narrate_economic_state(_es)}</div>',
                            unsafe_allow_html=True,
                        )
                        st.write("")
                        _p1, _p2, _p3, _p4 = st.columns(4)
                        _p1.metric("GDP Growth", f"{_es.get('gdp_growth', 0):.2%}")
                        _p2.metric("Inflation", f"{_es.get('inflation', 0):.2%}")
                        _p3.metric("Unemployment", f"{_es.get('unemployment', 0):.2%}")
                        _p4.metric("Interest Rate", f"{_es.get('interest_rate', 0):.2%}")
                        _p5, _p6, _p7, _p8 = st.columns(4)
                        _p5.metric("Output Gap", f"{_es.get('output_gap', 0):+.3f}")
                        _p6.metric("Credit Spread", f"{_es.get('credit_spread', 0):.2%}")
                        _p7.metric("Govt Debt", f"{_es.get('government_debt', 0):.1f}")
                        _p8.metric("HH Net Worth", f"{_es.get('household_net_worth', 0):.1f}")
                        st.caption(
                            "GDP, debt, and net worth are indexed (100 = baseline). "
                            "Rates are annualised. Output gap = (actual − potential) / potential."
                        )
                    else:
                        st.info("SFC model not available — ensure scarcity.simulation is installed.")

                # ── 8. RESOURCE UTILIZATION ─────────────────────────────────────
                with st.expander("8. Resource Utilization Analysis"):
                    st.caption(
                        "Derived from the SFC model outcomes: how efficiently resources are deployed "
                        "across households, firms, and government. Identifies underutilized or overstressed sectors."
                    )
                    if res.economic_state:
                        _es8 = res.economic_state
                        def _util_bar(label, value, low_is_bad=True, fmt=".1%"):
                            color = "#006600" if value > 0.5 else ("#F59E0B" if value > 0.25 else "#BB0000")
                            if not low_is_bad:
                                color = "#BB0000" if value > 0.75 else ("#F59E0B" if value > 0.5 else "#006600")
                            pct = int(min(100, max(0, value * 100)))
                            st.markdown(
                                f'<div style="margin-bottom:0.5rem;">'
                                f'<span style="font-size:0.85rem; font-weight:600;">{label}</span> '
                                f'<span style="font-size:0.85rem; color:{color};">{value:{fmt}}</span><br>'
                                f'<div style="background:#eee; border-radius:4px; height:8px;">'
                                f'<div style="background:{color}; width:{pct}%; height:8px; border-radius:4px;"></div>'
                                f'</div></div>', unsafe_allow_html=True
                            )
                        _util_bar("Financial Stability", _es8.get("financial_stability", 0.5))
                        _util_bar("Household Welfare (C/GDP)", _es8.get("household_welfare", 0.5))
                        _util_bar("Investment Ratio (I/GDP)", _es8.get("investment_ratio", 0.2))
                        _util_bar("Fiscal Space", max(0.0, _es8.get("fiscal_space", 0.0) + 0.5))
                        _util_bar("Savings Rate", _es8.get("savings_rate", 0.2))
                        _util_bar("Cost of Living Index (inverted)", max(0, 1 - (_es8.get("cost_of_living_index", 1.0) - 1.0) * 5), low_is_bad=True, fmt=".0%")
                        st.caption("All values are from the SFC model baseline run (20 simulation steps).")
                    else:
                        st.info("SFC model not available — resource utilization metrics not computable.")

                # ── 9. RISK PROPAGATION ANALYSIS ────────────────────────────────
                with st.expander("9. Risk Propagation — Could this trigger a chain reaction?"):
                    st.markdown(
                        "When something goes wrong in one part of the economy, it can cause problems elsewhere — "
                        "like dominoes falling. This section maps those potential chain reactions."
                    )
                    if res.propagation_chains:
                        for chain_info in res.propagation_chains:
                            st.markdown(narrate_propagation_chain(chain_info))
                    elif res.anomaly_scores and res.peak_score > 0.5:
                        st.info(
                            f"An anomaly was detected (peak score: {res.peak_score:.2f}). "
                            "Risk propagation mapping requires the discovery engine to find causal pathways first. "
                            "Provide more data rows to enable automatic cascade detection."
                        )
                    else:
                        st.success("No significant risk propagation signals detected at this anomaly level.")
                    # Always show the general model
                    st.write("**Systemic risk propagation model:**")
                    st.caption(
                        "Banking shock → credit tightening → construction slowdown → employment drop → social unrest signals. "
                        "Agricultural shock → food price spike → household stress → political instability risk. "
                        "The discovery engine maps your specific data's propagation pathways."
                    )

                # ── 10. FORECASTING ──────────────────────────────────────────────
                with st.expander("10. Forecasting — What might happen next?"):
                    st.markdown(
                        "Based on your data's historical patterns, the system projects where each variable is heading. "
                        "The **teal line** is the forecast. The **shaded area** shows the uncertainty — "
                        "wider bands mean less certainty about the direction."
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
                                st.plotly_chart(_fig_f, use_container_width=True)
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

    if st.session_state.get('local_df') is not None and schema is not None:
        causal_df = st.session_state['local_df']
        causal_cols = schema.get("required_columns", list(causal_df.columns))

        with tab_granger:
            _render_granger_section(causal_df, causal_cols, LIGHT_THEME)

        with tab_network:
            _render_causal_network(causal_df, causal_cols, LIGHT_THEME)

        with tab_cross:
            _render_cross_corr(causal_df, causal_cols, LIGHT_THEME)
    else:
        with tab_granger:
            st.warning("Upload data first to run causal analysis.")
        with tab_network:
            st.warning("Upload data first to run causal analysis.")
        with tab_cross:
            st.warning("Upload data first to run causal analysis.")

    with tab_projects:
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

    with tab3:
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
                inbox = SecureMessaging.get_inbox(Role.INSTITUTION.value, st.session_state.get('username'))
                if not inbox:
                    st.caption("No messages from your admin or national command.")
                else:
                    for msg in inbox:
                        sender_badge = "National Command" if msg['sender_role'] == Role.EXECUTIVE.value else "Sector Admin"
                        with st.expander(f"{sender_badge} | {msg['timestamp']} {'(NEW)' if not msg['is_read'] else ''}"):
                            st.write(msg['content'])
                            if not msg['is_read']:
                                if st.button("Acknowledge", key=f"ack_{msg['id']}"):
                                    SecureMessaging.mark_read(msg['id'])
                                    st.rerun()

    with tab_collab:
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

    if fl_mode and tab_fl_log is not None:
        with tab_fl_log:
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
                    with st.expander(f"{ts}  |  Status: {status}  |  Detection: {scores.get('A_Detection', '—')}  Impact: {scores.get('B_Impact', '—')}"):
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
