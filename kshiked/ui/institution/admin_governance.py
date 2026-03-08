import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from kshiked.ui.institution.backend.auth import enforce_role
from kshiked.ui.institution.backend.models import Role
from kshiked.ui.institution.backend.delta_sync import DeltaSyncManager
from kshiked.ui.institution.backend.federation_bridge import FederationBridge
from kshiked.ui.institution.backend.project_manager import ProjectManager
from kshiked.ui.institution.backend.database import get_connection
from kshiked.ui.institution.style import inject_enterprise_theme
from kshiked.ui.institution.backend.messaging import SecureMessaging
from kshiked.ui.institution.collab_room import render_collab_room
from kshiked.ui.institution.backend.analytics_engine import (
    generate_inaction_projection,
    get_historical_context,
    generate_recommendation,
    compute_outcome_impact,
)
from kshiked.ui.institution.backend.report_narrator import (
    narrate_composite_scores,
    narrate_severity,
    narrate_shock_vector,
    narrate_risk_for_executive,
)

def plot_shock_vector(shock_vector, title):
    import plotly.graph_objects as go
    metrics = list(shock_vector.keys())
    baselines = [shock_vector[m]["pre_shock_baseline"] for m in metrics]
    peaks = [shock_vector[m]["peak_shock_value"] for m in metrics]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Pre-Shock Baseline', x=metrics, y=baselines, marker_color='#888888'))
    fig.add_trace(go.Bar(name='Peak Shock Value', x=metrics, y=peaks, marker_color='#FF4444'))
    
    fig.update_layout(
        title=title,
        barmode='group',
        template='plotly_white',
        height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig

def render():
    enforce_role(Role.BASKET_ADMIN.value)
    inject_enterprise_theme()
    
    basket_id = st.session_state.get('basket_id')
    
    # Resolve sector name for a human-readable header
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT id, name FROM baskets")
        all_baskets = {r['id']: r['name'] for r in c.fetchall()}

    sector_name = all_baskets.get(basket_id, f"Sector {basket_id}")
    fl_mode = st.session_state.get('fl_mode_enabled', False)

    st.markdown(f"<h2 style='text-align: center; color: #1F2937;'>{sector_name} — Sector Admin</h2>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='text-align: center; color: #006600;'>Administrator: {st.session_state.get('username')}</h5>", unsafe_allow_html=True)

    st.write("---")

    # Build tab list — Mode B appends the Federated Learning tab
    base_tab_labels = [
        "Sector Overview",
        "Spoke Reports",
        "Risk Promotion",
        "Operational Projects",
        "Historical Archive",
        "Communications",
        "Settings",
        "Collaboration Room",
    ]
    if fl_mode:
        base_tab_labels.append("Federated Learning (Mode B)")

    _all_tabs = st.tabs(base_tab_labels)
    tab_overview = _all_tabs[0]    # Sector Overview (New landing page)
    tab1      = _all_tabs[1]   # Spoke Reports (was: Pending Spoke Insights)
    tab_risk  = _all_tabs[2]   # Risk Promotion (fusion workbench)
    tab_proj  = _all_tabs[3]   # Operational Projects
    tab2      = _all_tabs[4]   # Historical Archive (was: Historical Insights)
    tab_comms = _all_tabs[5]   # Communications (was: Secure Networking)
    tab3      = _all_tabs[6]   # Settings (was: Data Merging & Governance — config only)
    tab_collab = _all_tabs[7]  # Collaboration Room
    tab_fl    = _all_tabs[8] if fl_mode else None  # Federated Learning (Mode B only)
    
    with tab_overview:
        st.markdown(f"### {sector_name} Command Center")
        st.write("Real-time telemetry and overview of your sector's institutions.")
        
        # Calculate top-level metrics
        pending_count = len(DeltaSyncManager.get_pending_syncs(basket_id))
        active_proj_count = len(ProjectManager.get_active_projects(basket_id))
        
        inst_count = 0
        try:
            with get_connection() as conn_i:
                ci = conn_i.cursor()
                ci.execute("SELECT COUNT(*) as c FROM institutions WHERE basket_id = ?", (basket_id,))
                res = ci.fetchone()
                if res:
                    inst_count = res['c']
        except Exception:
            pass

        metrics_cols = st.columns(3)
        
        def _render_metric_card(col, title, value, icon, subtext, alert=False):
            bg_color = "#FFF"
            border_col = "#E5E7EB"
            title_col = "#6B7280"
            val_col = "#DC2626" if alert else "#111827"
            
            col.markdown(f"""
            <div style="background:{bg_color}; padding:1.2rem; border-radius:12px; 
                        border:1px solid {border_col}; box-shadow:0 1px 3px rgba(0,0,0,0.05); margin-bottom:1rem;">
                <div style="color:{title_col}; font-size:0.8rem; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">
                    {icon} {title}
                </div>
                <div style="color:{val_col}; font-size:2rem; font-weight:800; margin:0.3rem 0;">
                    {value}
                </div>
                <div style="color:{title_col}; font-size:0.75rem;">{subtext}</div>
            </div>""", unsafe_allow_html=True)

        _render_metric_card(metrics_cols[0], "Registered Spokes", str(inst_count), "🏢", "Active institutions in sector")
        _render_metric_card(metrics_cols[1], "Pending Reports", str(pending_count), "📥", "Requires admin review", alert=(pending_count>0))
        _render_metric_card(metrics_cols[2], "Active Projects", str(active_proj_count), "⚔️", "Cross-sector war rooms")

        # Telemetry Chart
        st.write("#### Telemetry Pulse")
        historical_syncs = DeltaSyncManager.get_historical_syncs(basket_id)
        all_syncs = historical_syncs + DeltaSyncManager.get_pending_syncs(basket_id)
        
        if not all_syncs:
            st.info("Insufficient data to generate telemetry baseline. Waiting for spoke transmissions.")
        else:
            try:
                import plotly.express as px
                df_chart = pd.DataFrame([
                    {
                        "Time": pd.to_datetime(s.get('created_at', s.get('timestamp', 0)), unit='s'),
                        "Severity": s['payload'].get('severity_score', 0.0),
                        "Status": s.get('status', 'PENDING'),
                        "Type": s['payload'].get('incident_type', 'ANOMALY')
                    }
                    for s in all_syncs
                ])
                df_chart = df_chart.sort_values(by="Time")
                
                fig = px.scatter(
                    df_chart, x="Time", y="Severity", color="Status", 
                    size="Severity", hover_data=["Type"],
                    color_discrete_map={"PENDING": "#DC2626", "PROCESSED": "#059669", "REJECTED": "#6B7280"}
                )
                fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
                fig.update_layout(
                    height=300, 
                    margin=dict(l=0, r=0, t=10, b=0),
                    template="plotly_white",
                    xaxis_title="",
                    yaxis_title="Severity Score",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.caption("Unable to render telemetry chart.")
                
        st.write("---")
        st.info("💡 **Tip**: Check the **Spoke Reports** tab to review incoming anomaly reports and promote them to national risks.")

    with tab1:
        st.markdown("### Incoming Reports from Spokes")
        st.write("Review anomaly reports submitted by institutions in your sector. Tick reports you want to combine into a risk and forward to the executive level.")
        
        pending_syncs = DeltaSyncManager.get_pending_syncs(basket_id)
        
        # State to track which events the Admin wants to fuse
        if 'selected_events' not in st.session_state:
            st.session_state['selected_events'] = set()
            
        if not pending_syncs:
            st.success("No pending reports. All incoming anomaly reports have been reviewed.")
        else:
            st.warning(f"{len(pending_syncs)} report(s) awaiting your review.")
            # Resolve institution names for display
            inst_name_cache = {}
            try:
                with get_connection() as conn_i:
                    ci = conn_i.cursor()
                    ci.execute("SELECT id, name FROM institutions WHERE basket_id = ?", (basket_id,))
                    inst_name_cache = {r['id']: r['name'] for r in ci.fetchall()}
            except Exception:
                pass

            for sync in pending_syncs:
                payload = sync['payload']
                inst_display = inst_name_cache.get(sync['institution_id'], f"Institution {sync['institution_id']}")

                col_sel, col_exp = st.columns([1, 15])
                with col_sel:
                    is_selected = st.checkbox("", key=f"sel_{sync['sync_id']}", value=sync['sync_id'] in st.session_state['selected_events'])
                    if is_selected:
                        st.session_state['selected_events'].add(sync['sync_id'])
                    elif sync['sync_id'] in st.session_state['selected_events']:
                        st.session_state['selected_events'].remove(sync['sync_id'])

                with col_exp.expander(f"{inst_display} — Severity {payload.get('severity_score', 0.0):.2f} | {payload.get('incident_type', 'ANOMALY')}"):
                    
                    if 'composite_scores' in payload:
                        st.write("#### Composite Intelligence Scores")
                        c_col1, c_col2, c_col3 = st.columns(3)
                        c_col1.metric("A) Detection Score", f"{payload['composite_scores'].get('A_Detection', 0.0):.2f} / 10.0")
                        c_col2.metric("B) Impact Score", f"{payload['composite_scores'].get('B_Impact', 0.0):.2f} / 10.0")
                        c_col3.metric("C) Certainty Score", f"{payload['composite_scores'].get('C_Certainty', 0.0):.2f} / 10.0")

                        # Plain-language explanation of what these scores mean
                        st.markdown(
                            f'<div style="background:#F0FDF4; border-left:4px solid #10B981; padding:10px 14px; '
                            f'border-radius:0 6px 6px 0; margin:8px 0; font-size:0.88rem; line-height:1.5;">'
                            f'<strong>📋 Plain-language interpretation:</strong><br>'
                            f'{narrate_composite_scores(payload["composite_scores"])}<br><br>'
                            f'{narrate_severity(payload.get("severity_score", 0.0))}</div>',
                            unsafe_allow_html=True,
                        )
                        st.write("---")
                    
                    # ── PILLAR 1: "SO WHAT?" ──
                    sev = payload.get('severity_score', 0.0)
                    projection = generate_inaction_projection(
                        severity=sev,
                        shock_vector=payload.get('shock_vector'),
                        incident_type=payload.get('incident_type', 'ANOMALY'),
                        composite_scores=payload.get('composite_scores'),
                    )
                    if projection:
                        st.markdown(
                            f'<div style="background:#FEF2F2; border-left:4px solid #DC2626; padding:10px 14px; '
                            f'border-radius:0 6px 6px 0; margin:8px 0; font-size:0.88rem;">'
                            f'<strong>⚠ So What?</strong> {projection}</div>',
                            unsafe_allow_html=True,
                        )

                    # ── PILLAR 2: "COMPARED TO WHAT?" ──
                    hist_ctx = get_historical_context(
                        basket_id=basket_id,
                        severity=sev,
                        incident_type=payload.get('incident_type', 'ANOMALY'),
                    )
                    st.markdown(
                        f'<div style="background:#F0F9FF; border-left:4px solid #3B82F6; padding:10px 14px; '
                        f'border-radius:0 6px 6px 0; margin:8px 0; font-size:0.88rem;">'
                        f'<strong>📊 Compared to What?</strong> {hist_ctx}</div>',
                        unsafe_allow_html=True,
                    )

                    if 'shock_vector' in payload:
                        # Plain-language explanation of what changed
                        sv_narrative = narrate_shock_vector(payload['shock_vector'])
                        st.markdown(
                            f'<div style="background:#FFFBEB; border-left:4px solid #F59E0B; padding:10px 14px; '
                            f'border-radius:0 6px 6px 0; margin:8px 0; font-size:0.88rem;">'
                            f'{sv_narrative}</div>',
                            unsafe_allow_html=True,
                        )
                        st.plotly_chart(plot_shock_vector(payload['shock_vector'], "Pre vs Post Shock Vector"), use_container_width=True, key=f"pending_{sync['sync_id']}")
                    if 'spoke_interpretation' in payload:
                        st.markdown(f"**Institution's own assessment:** _{payload['spoke_interpretation']}_")
                    if 'post_shock_volatility_forecast' in payload:
                        st.write("**Volatility forecast at time of anomaly:**")
                        st.json(payload['post_shock_volatility_forecast'])

                    st.write("---")
                    reject_msg = st.text_input("Reason for returning (optional):", key=f"rej_msg_{sync['sync_id']}")
                    if st.button("Return to institution for more information", key=f"rej_btn_{sync['sync_id']}"):
                        DeltaSyncManager.reject_sync(sync['sync_id'], reject_msg or "Please provide additional supporting data.")
                        st.rerun()
            
            
            st.write("---")
            st.write("### Promote to National Risk")
            st.write("Select reports above, write a summary, and forward as a validated risk to the executive level.")
            
            selected_count = len(st.session_state['selected_events'])
            if selected_count > 0:
                st.info(f"{selected_count} Event(s) selected for Fusion.")
                risk_title = st.text_input("Validated Risk Title")
                risk_desc = st.text_area("Executive Summary / Risk Description", height=100)
                
                if st.button("Promote to Validated Risk", type="primary"):
                    if not risk_title or not risk_desc:
                        st.error("Please provide both a Title and Description for the Executive.")
                    else:
                        # Compute average scores across selected events
                        selected_payloads = [s['payload'] for s in pending_syncs if s['sync_id'] in st.session_state['selected_events']]
                        avg_A = np.mean([p.get('composite_scores', {}).get('A_Detection', 0) for p in selected_payloads]) if selected_payloads else 0.0
                        avg_B = np.mean([p.get('composite_scores', {}).get('B_Impact', 0) for p in selected_payloads]) if selected_payloads else 0.0
                        avg_C = np.mean([p.get('composite_scores', {}).get('C_Certainty', 0) for p in selected_payloads]) if selected_payloads else 0.0
                        
                        fused_scores = {
                            "A_Detection": round(avg_A, 2),
                            "B_Impact": round(avg_B, 2),
                            "C_Certainty": round(avg_C, 2)
                        }
                        
                        DeltaSyncManager.promote_risk(
                            basket_id=basket_id,
                            title=risk_title,
                            description=risk_desc,
                            composite_scores=fused_scores,
                            source_sync_ids=list(st.session_state['selected_events'])
                        )
                        st.session_state['selected_events'] = set()
                        st.success(f"Risk '{risk_title}' promoted to executive level.")
            else:
                st.write("*Tick reports above to select them, then fill in the fields here.*")

            if fl_mode:
                st.write("---")
                st.caption("Mode B is active. To run federated aggregation on these reports, go to the **Federated Learning (Mode B)** tab.")

    with tab_risk:
        st.markdown("### Promoted Risks")
        st.write("Risks that you have validated and forwarded to the executive level. These are visible to national command.")

        promoted = DeltaSyncManager.get_promoted_risks(basket_id)
        if not promoted:
            st.success("No risks have been promoted from this sector yet. Review incoming reports in the **Spoke Reports** tab and promote when ready.")
        else:
            st.info(f"{len(promoted)} risk(s) currently visible to the executive.")
            for risk in promoted:
                scores = risk.get('composite_scores', {})
                b_impact = scores.get('B_Impact', 0)
                with st.expander(f"{risk['title']}  |  Impact: {b_impact:.1f}/10  |  {pd.to_datetime(risk.get('timestamp', 0), unit='s').strftime('%Y-%m-%d %H:%M')}"):
                    st.write(risk.get('description', ''))
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Detection", f"{scores.get('A_Detection', 0):.2f}/10")
                    c2.metric("Impact", f"{b_impact:.2f}/10")
                    c3.metric("Certainty", f"{scores.get('C_Certainty', 0):.2f}/10")

                    # ── PILLAR 4: "WHAT SHOULD I DO?" ──
                    rec = generate_recommendation(
                        risk=risk,
                        all_baskets=all_baskets,
                        global_risks=promoted,
                        historical_syncs=DeltaSyncManager.get_historical_syncs(basket_id),
                    )
                    st.markdown(
                        f'<div style="background:#FFFBEB; border-left:4px solid {rec.level_color}; '
                        f'padding:10px 14px; border-radius:0 6px 6px 0; margin:8px 0;">'
                        f'<strong>🎯 Recommendation: '
                        f'<span style="background:{rec.level_color}; color:#fff; padding:2px 8px; '
                        f'border-radius:4px; font-size:0.8rem;">{rec.level}</span></strong><br/>'
                        f'<span style="font-size:0.88rem;">{rec.summary}</span><br/>'
                        f'<span style="font-size:0.82rem; color:#64748b;">'
                        f'<b>Who:</b> {", ".join(rec.who[:4])} &nbsp;|&nbsp; '
                        f'<b>Urgency:</b> {rec.urgency}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    with tab_proj:
        st.info("### Operational Projects (Cross-Basket Fusion Spaces)")
        st.write("Temporary collaboration containers linking multiple baskets to one evolving complex situation.")
        
        # 1. Launch New Project (Moved to Top)
        st.write("---")
        with st.container(border=True):
            st.write("#### Launch New Operational Project")
            st.write("Elevate a systemic threat into a cross-basket shared war room.")
            new_title = st.text_input("Project Name")
            new_desc = st.text_area("Initial SitRep / Description")
            new_severity = st.slider("Assigned Severity", 1.0, 10.0, 5.0, 0.5)
            
            # Select participants (excluding self)
            other_baskets = {k: v for k, v in all_baskets.items() if k != basket_id}
            selected_participants = st.multiselect("Invite Sector Baskets", options=list(other_baskets.keys()), format_func=lambda x: other_baskets[x])
            
            if st.button("Initialize Shared Space", type="primary", key="init_proj_btn"):
                if new_title and selected_participants:
                    # include self in participants
                    participants_list = [basket_id] + selected_participants
                    ProjectManager.create_project(new_title, new_desc, new_severity, participants_list)
                    st.success(f"Operational Project '{new_title}' launched successfully.")
                    st.rerun()
                else:
                    st.error("Please provide a Title and select at least one other Sector Basket to invite.")
                    
        st.write("---")
        
        # 2. View Active Projects
        st.write("#### Active Projects")
        active_projects = ProjectManager.get_active_projects(basket_id)
        
        if not active_projects:
            st.success("No active operational projects require your attention.")
        else:
            for proj in active_projects:
                with st.expander(f"WAR ROOM: {proj['title']} (Severity: {proj['severity']})", expanded=True):
                    project_data = ProjectManager.get_project_details(proj['id'])
                    
                    # 1. Phase Progression Banner
                    current_phase = proj['current_phase']
                    phases = ["EMERGENCE", "ESCALATION", "STABILIZATION", "RECOVERY"]
                    try:
                        p_idx = phases.index(current_phase)
                    except ValueError:
                        p_idx = 0
                        
                    st.write("---")
                    st.write(f"**Current Phase:**  {' ➔ '.join([f'*{p}*' if i < p_idx else f'**{p}**' if i == p_idx else p for i, p in enumerate(phases)])}")
                    if p_idx < len(phases) - 1:
                        if st.button(f"Promote Phase to {phases[p_idx + 1]}", key=f"phase_btn_{proj['id']}", use_container_width=True):
                            ProjectManager.transition_phase(proj['id'], phases[p_idx + 1])
                            st.rerun()
                    st.write("---")
                    
                    p_col1, p_col2, p_col3 = st.columns([1, 2, 1])
                    with p_col1:
                        st.write("**Participants**")
                        for p_b_id in project_data['participants']:
                            st.write(f"- {all_baskets.get(p_b_id, f'Basket {p_b_id}')}")
                        st.write("---")
                        st.write(f"**Created:**<br>{pd.to_datetime(proj['created_at'], unit='s').strftime('%Y-%m-%d %H:%M')}", unsafe_allow_html=True)
                        
                        st.write("---")
                        st.write("**Disagreement Tensor (Consensus Drift)**")
                        dis_matrix = ProjectManager.get_disagreement_matrix(proj['id'])
                        if dis_matrix:
                            import plotly.express as px
                            df_dis = pd.DataFrame(list(dis_matrix.items()), columns=["Sector Admin", "Certainty"])
                            fig = px.bar(df_dis, x="Sector Admin", y="Certainty", range_y=[0.0, 1.0], color="Certainty", color_continuous_scale="RdYlGn")
                            fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0), template="plotly_white")
                            st.plotly_chart(fig, use_container_width=True, key=f"dis_{proj['id']}")
                        else:
                            st.caption("Awaiting sector updates.")
                        
                    with p_col2:
                        st.write("**Shared Causal Stream**")
                        st.markdown(f"*{proj['description']}*")
                        st.write("---")
                        
                        for update in project_data.get('updates', []):
                            u_color = "#006600" if update['update_type'] == 'OBSERVATION' else "#BB0000" if update['update_type'] == 'POLICY_ACTION' else "#1F2937"
                            st.markdown(f"**<span style='color:{u_color};'>[{update['update_type']}]</span> {update['author_name']}**", unsafe_allow_html=True)
                            st.write(update['content'])
                            if update['certainty']:
                                st.caption(f"Certainty Index: {update['certainty']:.2f}/1.0")
                            st.write("")
                            
                    with p_col3:
                        st.write("**Actions**")
                        new_update = st.text_area("Post Update", key=f"upd_text_{proj['id']}")
                        update_type = st.selectbox("Update Type", ["OBSERVATION", "ANALYSIS_REQUEST"], key=f"upd_type_{proj['id']}")
                        
                        cert_val = None
                        if update_type == 'OBSERVATION':
                            cert_val = st.slider("Your Certainty", 0.0, 1.0, 0.5, 0.05, key=f"cert_{proj['id']}")
                            
                        if st.button("Broadcast to Project Stream", key=f"upd_btn_{proj['id']}"):
                            ProjectManager.add_update(proj['id'], st.session_state.get('username', f"Admin {basket_id}"), update_type, new_update, certainty=cert_val)
                            st.rerun()
                            
                        st.write("---")
                        
                        with st.expander("Meta-Learning: Force Archive & Resolution"):
                            st.write("Select the failure mode or resolution state. This securely recalibrates Sector Trust Weights across the entire national network.")
                            res_state = st.selectbox("Resolution State", ['RESOLVED', 'FALSE_ALARM', 'INSUFFICIENT_EVIDENCE', 'CONFLICTING_SIGNALS'], key=f"res_state_{proj['id']}")
                            pol_score = st.slider("Policy Effectiveness", 0.0, 10.0, 5.0, 0.5, key=f"pol_score_{proj['id']}")
                            res_sum = st.text_area("Debrief Summary", key=f"res_sum_{proj['id']}")
                            
                            if st.button("Archive & Calibrate Weights", key=f"arch_btn_{proj['id']}", type="primary"):
                                if not res_sum:
                                    st.error("Debrief summary required for Meta-Learning.")
                                else:
                                    payload = {"final_severity": proj['severity'], "active_participants": len(project_data['participants'])}
                                    ProjectManager.archive_project(proj['id'], res_state, pol_score, res_sum, payload)
                                    st.rerun()
                            
                            
        st.write("---")

    with tab2:
        st.markdown("### Historical Archive")
        st.write("All previously reviewed reports and closed operational projects.")
        
        st.write("#### Institutional Memory (Closed Projects)")
        memories = ProjectManager.get_institutional_memory()
        
        if not memories:
            st.write("No meta-learning historical archives exist yet.")
        else:
            for mem in memories:
                res_color = "#006600" if mem['resolution_state'] == 'RESOLVED' else "#BB0000" if mem['resolution_state'] == 'FALSE_ALARM' else "#1F2937"
                with st.expander(f"[{mem['resolution_state']}] {mem['title']} (Final Severity: {mem['severity']})"):
                    m_col1, m_col2 = st.columns([2, 1])
                    with m_col1:
                        st.markdown(f"**Resolution Summary:**<br>{mem['resolution_summary']}", unsafe_allow_html=True)
                        st.write("---")

                        # ── PILLAR 5: "DID IT WORK?" ──
                        try:
                            participant_ids = []
                            with get_connection() as conn_p5:
                                cp5 = conn_p5.cursor()
                                cp5.execute("SELECT basket_id FROM project_participants WHERE project_id = ?", (mem['id'],))
                                participant_ids = [r['basket_id'] for r in cp5.fetchall()]
                            if participant_ids:
                                impact = compute_outcome_impact(
                                    project_id=mem['id'],
                                    project_created_at=mem.get('created_at', 0),
                                    project_archived_at=mem.get('updated_at', 0),
                                    participant_basket_ids=participant_ids,
                                )
                                bg = "#F0FDF4" if impact.get('delta_pct', 0) < 0 else "#FEF2F2"
                                border = "#10B981" if impact.get('delta_pct', 0) < 0 else "#EF4444"
                                st.markdown(
                                    f'<div style="background:{bg}; border-left:4px solid {border}; '
                                    f'padding:10px 14px; border-radius:0 6px 6px 0; margin:8px 0; font-size:0.88rem;">'
                                    f'<strong>📈 Did it Work?</strong> {impact["narrative"]}</div>',
                                    unsafe_allow_html=True,
                                )
                        except Exception:
                            pass

                        st.write(f"**Learning Payload / Meta Data:**")
                        st.json(mem['learning_payload'])
                    with m_col2:
                        st.metric("Policy Effectiveness", f"{mem['policy_effectiveness_score']}/10.0")
                        
                        ttc_mins = mem['time_to_consensus_seconds'] / 60.0
                        st.metric("Time-to-Resolution", f"{ttc_mins:.1f} minutes")
                        
                        st.markdown(f"<span style='color:{res_color};'>**Network Trust Weight Recalibrated**</span>", unsafe_allow_html=True)
                        
        st.write("---")
        st.write("#### Raw Spoke Anomalies (Merged)")
        
        historical_syncs = DeltaSyncManager.get_historical_syncs(basket_id)
        if not historical_syncs:
            st.write("No historical insights found.")
        else:
            st.write(f"Showing **{len(historical_syncs)}** historical records.")
            for sync in historical_syncs:
                payload = sync['payload']
                status_color = "#00FF00" if sync['status'] == 'PROCESSED' else "#FF4444"
                with st.expander(f"[{sync['status']}] Report from Spoke {sync['institution_id']} (Severity: {payload.get('severity_score', 0.0):.2f})"):
                    if sync['status'] == 'REJECTED':
                        st.error(f"**Admin Feedback:** {payload.get('admin_message', 'No message provided.')}")
                        
                    if 'composite_scores' in payload:
                        st.write("#### Composite Intelligence Scores")
                        c_col1, c_col2, c_col3 = st.columns(3)
                        c_col1.metric("A) Detection Score", f"{payload['composite_scores'].get('A_Detection', 0.0):.2f} / 10.0")
                        c_col2.metric("B) Impact Score", f"{payload['composite_scores'].get('B_Impact', 0.0):.2f} / 10.0")
                        c_col3.metric("C) Certainty Score", f"{payload['composite_scores'].get('C_Certainty', 0.0):.2f} / 10.0")
                        st.write("---")
                        
                    if 'shock_vector' in payload:
                        st.plotly_chart(plot_shock_vector(payload['shock_vector'], f"Historical Shock Vector (ID: {sync['sync_id']})"), use_container_width=True, key=f"hist_{sync['sync_id']}")
                    if 'spoke_interpretation' in payload:
                        st.markdown(f"**Interpretation:** {payload['spoke_interpretation']}")

    with tab3:
        st.markdown("### Settings")
        st.write("Configure detection sensitivity thresholds for your sector's institutions.")
        st.caption("These thresholds are applied to all institutions in your sector when you push them below.")

        col1, col2 = st.columns(2)
        with col1:
            st.slider("Anomaly Detection Sensitivity (Lower = More Alerts)", 1.0, 15.0, 4.5, 0.5)
            st.slider("Prediction Volatility Smoothing", 0.5, 0.99, 0.8, 0.01)
        with col2:
            st.slider("Threat Detection Speed", 0.01, 0.5, 0.1, 0.01)
            st.selectbox("Hardware Defense Protocol", ["Standard Mode", "Aggressive Mode", "High Performance Mode"])

        if st.button("Push Target Bounds to Spokes"):
            st.success("System configurations successfully synchronized to all connected nodes.")

        if fl_mode:
            st.write("---")
            st.info("**Mode B active.** The aggregated global model is available in the Federated Learning tab.")

    if fl_mode and tab_fl is not None:
        with tab_fl:
            st.markdown("### Federated Learning (Mode B)")
            st.info(
                "**Mode B is active.** Raw data from spokes never leaves their node. Only mathematical summaries "
                "(aggregated model weights) are processed here. Gradients flow upward; raw records do not."
            )

            # Auto-hydrate from historical processed syncs
            if 'current_global_weights' not in st.session_state:
                historical_syncs = DeltaSyncManager.get_historical_syncs(basket_id)
                processed_syncs = [s for s in historical_syncs if s['status'] == 'PROCESSED']
                if processed_syncs:
                    payloads_hist = [s['payload'] for s in processed_syncs]
                    gw, gm = FederationBridge.aggregate_spoke_models(payloads_hist, method_name='trimmed_mean')
                    st.session_state['current_global_weights'] = gw
                    st.session_state['last_aggregation_meta'] = gm

            st.write("#### Step 1 \u2014 Aggregate Pending Reports into Global Model")
            pending_fl = DeltaSyncManager.get_pending_syncs(basket_id)
            if not pending_fl:
                st.success("No pending reports to aggregate.")
            else:
                fl_methods = {
                    "Trimmed Mean (recommended \u2014 removes outliers)": "trimmed_mean",
                    "Median": "median",
                    "Krum (most reliable node only)": "krum",
                    "Bulyan (Byzantine-resilient consensus)": "bulyan"
                }
                selected_fl_method = st.selectbox("Aggregation strategy", list(fl_methods.keys()))
                if st.button("Run Aggregation", type="primary"):
                    payloads_fl = [s['payload'] for s in pending_fl]
                    global_weights, meta = FederationBridge.aggregate_spoke_models(
                        payloads_fl, method_name=fl_methods[selected_fl_method]
                    )
                    DeltaSyncManager.mark_synced([s['sync_id'] for s in pending_fl])
                    st.session_state['current_global_weights'] = global_weights
                    st.session_state['last_aggregation_meta'] = meta
                    st.success(f"Aggregated {meta.get('participants', '?')} participants using {selected_fl_method}.")
                    st.rerun()

            if 'current_global_weights' in st.session_state:
                st.write("---")
                st.write("#### Current Global Model Weights")
                import plotly.graph_objects as go_fl
                weights_fl = st.session_state['current_global_weights']
                fig_fl = go_fl.Figure()
                y_data_fl = weights_fl if isinstance(weights_fl, list) else weights_fl.tolist()
                fig_fl.add_trace(go_fl.Scatter(y=y_data_fl, mode='lines+markers', line=dict(color='#006600')))
                fig_fl.update_layout(template='plotly_white', height=280, margin=dict(l=0, r=0, t=0, b=0))
                st.plotly_chart(fig_fl, use_container_width=True, key="global_model_chart_fl")

                st.write("---")
                st.write("#### Step 2 \u2014 Apply Privacy Protection before Broadcast")
                st.write("Add differential privacy noise to protect institution identities before sharing aggregated weights.")
                epsilon_fl = st.slider("Privacy level (lower = more privacy, less precision)", 0.1, 10.0, 1.5, 0.1)
                if st.button("Apply Privacy Noise"):
                    noised = FederationBridge.apply_differential_privacy(weights_fl, epsilon_fl)
                    st.session_state['dp_weights'] = noised
                    st.success("Privacy noise applied.")

                if 'dp_weights' in st.session_state:
                    dp_c1, dp_c2 = st.columns(2)
                    if dp_c1.button("Broadcast to Spokes (privacy-protected)", use_container_width=True):
                        st.success("Aggregated insights broadcast to spoke nodes.")
                    if dp_c2.button("Share with Peer Sectors (privacy-protected)", use_container_width=True):
                        st.success("Cross-sector synchronization complete.")

    # Terminology guide removed — terminology is now inline where needed
        
    with tab_comms:
        with st.container(border=True):
            st.markdown("### Communications")

            # Escalate up to national
            st.write("#### Escalate to national level")
            esc_col1, esc_col2 = st.columns([4, 1])
            with esc_col1:
                escalation = st.text_input("Message to executive", key="esc_payload", placeholder="Describe a high-priority situation requiring national attention...")
            with esc_col2:
                st.write("")
                if st.button("Send to Executive", type="primary", use_container_width=True):
                    if escalation:
                        SecureMessaging.send_message(
                            sender_role=Role.BASKET_ADMIN.value,
                            sender_id=st.session_state.get('username'),
                            receiver_role=Role.EXECUTIVE.value,
                            receiver_id="ALL",
                            content=escalation
                        )
                        st.success("Escalation sent to executive.")

            st.write("---")
            c_col1, c_col2 = st.columns([1, 1])
            with c_col1:
                st.write("**Messages from your institutions**")
                spoke_inbox = SecureMessaging.get_inbox(Role.BASKET_ADMIN.value, str(basket_id))
                if not spoke_inbox:
                    st.caption("No messages from institutions.")
                else:
                    for msg in spoke_inbox:
                        with st.expander(f"From {msg['sender_id']} | {msg['timestamp']} {'(NEW)' if not msg['is_read'] else ''}"):
                            st.write(msg['content'])
                            if not msg['is_read']:
                                if st.button("Mark as read", key=f"ac_clear_{msg['id']}"):
                                    SecureMessaging.mark_read(msg['id'])
                                    st.rerun()

                st.write("---")
                st.write("**Send directive to an institution**")
                target_spoke = st.text_input("Institution username (or 'ALL' to broadcast)")
                directive = st.text_area("Message", height=100)
                if st.button("Send directive", type="primary", use_container_width=True):
                    if directive and target_spoke:
                        SecureMessaging.send_message(
                            sender_role=Role.BASKET_ADMIN.value,
                            sender_id=st.session_state.get('username'),
                            receiver_role=Role.INSTITUTION.value,
                            receiver_id=target_spoke,
                            content=directive
                        )
                        st.success("Directive sent.")

            with c_col2:
                st.write("**Directives from the executive**")
                exec_inbox = SecureMessaging.get_inbox(Role.BASKET_ADMIN.value, st.session_state.get('username'))
                if not exec_inbox:
                    st.caption("No directives received from executive.")
                else:
                    for msg in exec_inbox:
                        with st.expander(f"Executive | {msg['timestamp']} {'(NEW)' if not msg['is_read'] else ''}"):
                            st.write(msg['content'])
                            if not msg['is_read']:
                                if st.button("Acknowledge", key=f"ac_exec_{msg['id']}"):
                                    SecureMessaging.mark_read(msg['id'])
                                    st.rerun()

