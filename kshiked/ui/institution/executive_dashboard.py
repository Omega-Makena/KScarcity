import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import time

project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from kshiked.ui.institution.backend.auth import enforce_role
from kshiked.ui.institution.backend.models import Role
from kshiked.ui.institution.backend.executive_bridge import ExecutiveBridge
from kshiked.ui.institution.backend.delta_sync import DeltaSyncManager
from kshiked.ui.institution.backend.project_manager import ProjectManager
from kshiked.ui.institution.backend.database import get_connection
from kshiked.ui.institution.style import inject_enterprise_theme
from kshiked.ui.institution.backend.messaging import SecureMessaging

def render():
    enforce_role(Role.EXECUTIVE.value)
    inject_enterprise_theme()
    
    st.markdown("<h2 style='text-align: center; color: #C60C30;'>National Security Command Center</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #006747;'>Strategic Intelligence & Coordinated Response</h3>", unsafe_allow_html=True)
    
    # Fetch all baskets
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT id, name FROM baskets")
        all_baskets = {r['id']: r['name'] for r in c.fetchall()}
        
    st.write("---")
    
    # Global state queries for Top-Level Metrics
    active_projects = ProjectManager.get_active_projects(None)
    global_risks = DeltaSyncManager.get_promoted_risks()
    memories = ProjectManager.get_institutional_memory()
    
    # 7. NATIONAL / ORGANIZATIONAL MAP (Spatial Awareness)
    with st.container(border=True):
        st.write("#### National Threat Topography (Kenya)")
        st.write("Geospatial distribution of emerging hotspots across Kenyan counties.")
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Kenyan Counties Data Mock
        counties_data = {
            "County": ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Eldoret", "Garissa", "Nyeri", "Machakos", "Kakamega"],
            "lat": [-1.2921, -4.0435, -0.0917, -0.3031, 0.5143, -0.4532, -0.4167, -1.5177, 0.2827],
            "lon": [36.8219, 39.6682, 34.7680, 36.0800, 35.2698, 39.6461, 36.9500, 37.2634, 34.7519],
        }
        df_geo = pd.DataFrame(counties_data)
        np.random.seed(int(time.time()) % 1000) # dynamic pseudo-random
        df_geo['Stress Index'] = np.random.randint(20, 100, size=len(df_geo))
        
        fig_map = px.scatter_mapbox(
            df_geo, lat="lat", lon="lon", hover_name="County", hover_data=["Stress Index"],
            color="Stress Index", size="Stress Index",
            color_continuous_scale=px.colors.sequential.OrRd, size_max=20, zoom=5.2
        )
        # Use a professional, dark theme Mapbox without needing an API key (carto-darkmatter)
        fig_map.update_layout(
            mapbox_style="carto-darkmatter",
            mapbox_center={"lat": 0.0236, "lon": 37.9062},
            margin={"r":0,"t":0,"l":0,"b":0},
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_map, use_container_width=True, key="exec_map")
    
    # 2. TABBED NAVIGATION Grid
    tab_summaries, tab_risk, tab_projects, tab_history, tab_comms = st.tabs([
        "Sector Summaries",
        "National Risk & Future Simulation", 
        "Active Operational Projects", 
        "Institutional Memory Archive",
        "Command & Control (Comms)"
    ])
    
    with tab_summaries:
        st.info("### Global Sector States")
        st.write("Current aggregated intelligence summaries for all active national baskets.")
        for b_id, b_name in all_baskets.items():
            # Find any active risk for this basket
            b_risks = [r for r in global_risks if r['basket_id'] == b_id]
            status_text = "Stable"
            if b_risks:
                status_text = f"{len(b_risks)} Active Anomalies"
                
            with st.expander(f"Sector: {b_name} | Status: {status_text}"):
                c1, c2 = st.columns([1, 1])
                c1.write("**Operational Overview**")
                c1.write(f"The {b_name} sector is currently reporting a {status_text.lower()} state.")
                
                c2.write("**Recent Intelligence**")
                if b_risks:
                    for r in b_risks:
                        c2.markdown(f"- **{r['title']}**: Impact {r.get('composite_scores', {}).get('B_Impact', 0)}/10")
                else:
                    c2.caption("No significant structural drift detected.")
    
    
    with tab_risk:
        col_risk, col_sim = st.columns([1, 1.2])
        
        with col_risk:
            with st.expander("Key Intelligence Signals", expanded=True):
                st.write("Recent anomalies and shifts across sectors requiring executive attention.")
                
                if not global_risks:
                    st.success("No active systemic signals detected.")
                else:
                    # 2. KEY SIGNALS PANEL (Tabular)
                    signal_data = []
                    for idx, risk in enumerate(global_risks):
                        scores = risk.get('composite_scores', {})
                        b_impact = scores.get('B_Impact', 0)
                        impact_str = "High" if b_impact > 7 else "Medium" if b_impact > 4 else "Low"
                        conf_str = f"{scores.get('C_Certainty', 0.0) / 10.0:.2f}"
                        sector_name = all_baskets.get(risk['basket_id'], f"Sector {risk['basket_id']}")
                        
                        signal_data.append({
                            "Signal": risk['title'],
                            "Impact": impact_str,
                            "Confidence": conf_str,
                            "Time Detected": pd.to_datetime(risk.get('timestamp', time.time()), unit='s').strftime('%Y-%m-%d %H:%M'),
                            "Sector": sector_name
                        })
                    
                    df_signals = pd.DataFrame(signal_data)
                    st.dataframe(df_signals, use_container_width=True, hide_index=True)
                    
                    
                    # 3. CAUSAL EXPLANATION LAYER (Why)
                    with st.expander("Causal Interpretation", expanded=True):
                        
                        # Extracting the first risk as the primary driver to construct a narrative.
                        primary_risk = global_risks[0]
                        cause_text = f"The current <b>{trend.lower()}</b> trend is primarily driven by <b>{primary_risk['title'].lower()}</b> emanating from the <b>{all_baskets.get(primary_risk['basket_id'], 'unknown')}</b> sector."
                        
                        st.markdown(f"> {cause_text}", unsafe_allow_html=True)
                        
                        st.write("**Top Contributing Factors:**")
                        st.markdown(f"- Multi-sector propagation from {all_baskets.get(primary_risk['basket_id'], 'origin sector')}.")
                        st.markdown("- Sustained volatility breaking baseline thresholds.")
                        st.markdown(f"- Certainty Index: ({primary_risk.get('composite_scores', {}).get('C_Certainty', 0.0):.1f}/10)")
                    
                    # 9. ALERT TIMELINE (Chronological Story)
                    with st.expander("Alert Timeline", expanded=True):
                        st.write("Chronology of structural deterioration:")
                        st.markdown(f"- **Day -5:** Sentiment anomaly detected in {all_baskets.get(primary_risk['basket_id'], 'sector')}.")
                        st.markdown("- **Day -3:** Local market volatility increase exceeds bounds.")
                        st.markdown(f"- **Today:** Critical {primary_risk['title'].lower()} signal triggered.")
                    
                    # Priority Recommendations (Principle 6 prototype)
                    with st.expander("Priority Recommendations", expanded=True):
                        st.markdown("1. **Monitor liquidity exposure** in adjoining sectors. (High Confidence)")
                        st.markdown(f"2. **Delay capital reallocation** pending {all_baskets.get(primary_risk['basket_id'], 'sector')} stabilization. (Moderate Confidence)")
                                
        with col_sim:
            with st.expander("Forward Projection (Baseline vs. Risk)", expanded=True):
                st.write("90-Day Outlook if no intervention is authorized.")
                
                # Dynamic Baseline Trajectory
                if not global_risks:
                    df_base, _ = ExecutiveBridge.simulate_policy_shock(0, 0.0, steps=30)
                    st.line_chart(df_base, height=150)
                else:
                    df_base, _ = ExecutiveBridge.simulate_policy_shock(1, 4.0, steps=30)
                    st.line_chart(df_base, height=150)
            
            with st.expander("Policy Action Simulator", expanded=True):
                st.write("Dynamic Scarcity Engine simulation bounding the consequences of interventions.")
                
                sim_c1, sim_c2 = st.columns([1, 1])
                with sim_c1:
                    live_tiers = ExecutiveBridge.get_tiers()
                    target_name = st.selectbox("Intervention Target", live_tiers)
                    target_idx = live_tiers.index(target_name)
                with sim_c2:
                    shock_magnitude = st.slider("Intervention Intensity", -10.0, 10.0, -5.0, 0.5)
                
                if st.button("Project Multi-Sector Outcome", type="primary", use_container_width=True):
                    with st.spinner("Connecting to Scarcity Engine (VARX/GARCH)..."):
                        df_mitigated, df_var = ExecutiveBridge.simulate_policy_shock(target_idx, shock_magnitude, steps=30)
                        
                        end_state = df_mitigated.iloc[-1].mean()
                        base_end_state = df_base.iloc[-1].mean()
                        
                        if end_state < base_end_state:
                            st.success(f"**Expected Outcome:** Threat topology shows systemic cooling.")
                            reduction = ((base_end_state - end_state) / (abs(base_end_state) + 1e-9)) * 100
                            st.metric("Expected Systemic Strain Reduction", f"{reduction:.1f}%")
                        else:
                            st.warning(f"**Expected Outcome:** Intervention compounds systemic strain.")
                            inc = ((end_state - base_end_state) / (abs(base_end_state) + 1e-9)) * 100
                            st.metric("Expected Systemic Strain Increase", f"{inc:.1f}%")
                        
                        st.write("Forecasted Trajectory (Mitigated):")
                        st.line_chart(df_mitigated, height=150)

                    
    with tab_projects:
        with st.container(border=True):
            st.info("#### Initiate National Operational Project")
            st.write("Create a new strategic war-room connecting specific Sector Admins.")
            
            p_c1, p_c2, p_c3 = st.columns([1, 1, 1])
            with p_c1:
                p_title = st.text_input("Project Codename/Title", key="ex_p_title")
                p_sev = st.selectbox("Initial Severity", [1, 2, 3, 4, 5], index=2, key="ex_p_sev")
            with p_c2:
                p_desc = st.text_area("Strategic Objective", key="ex_p_desc", height=100)
            with p_c3:
                p_baskets = st.multiselect("Assign Sector Admins", options=list(all_baskets.keys()), format_func=lambda x: all_baskets[x], key="ex_p_baskets")
                st.write("")
                if st.button("Launch National War Room", type="primary", use_container_width=True, key="ex_btn_launch"):
                    if p_title and p_desc and p_baskets:
                        ProjectManager.create_project(
                            title=p_title,
                            description=p_desc,
                            severity=p_sev,
                            participants=p_baskets
                        )
                        st.success("National Operational Project initiated.")
                        st.rerun()
                    else:
                        st.error("Please fill in all project details.")
                        
        with st.container(border=True):
            st.info("#### Active Operational Projects (War Rooms)")
            st.write("Monitor multi-sector collaboration war rooms. Inject top-down Policy Actions directly into their intelligence streams.")
            
            if not active_projects:
                st.success("No active cross-sector Operational Projects.")
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
                            
                        with st.container(border=True):
                            st.write(f"**Executive Oversight | Phase:**  {' ➔ '.join([f'*{p}*' if i < p_idx else f'**{p}**' if i == p_idx else p for i, p in enumerate(phases)])}")
                        
                        p_col1, p_col2, p_col3 = st.columns([1, 1.5, 1])
                        with p_col1:
                            st.write("**Reporting Sectors**")
                            for p_b_id in project_data['participants']:
                                st.write(f"- {all_baskets.get(p_b_id, f'Basket {p_b_id}')}")
                            st.write("---")
                            st.write(f"**Created:**<br>{pd.to_datetime(proj['created_at'], unit='s').strftime('%Y-%m-%d %H:%M')}", unsafe_allow_html=True)
                            
                            st.write("---")
                            st.write("**Sector Disagreement Metrics**")
                            dis_matrix = ProjectManager.get_disagreement_matrix(proj['id'])
                            if dis_matrix:
                                import plotly.express as px
                                df_dis = pd.DataFrame(list(dis_matrix.items()), columns=["Sector Admin", "Certainty"])
                                fig = px.bar(df_dis, x="Sector Admin", y="Certainty", range_y=[0.0, 1.0], color="Certainty", color_continuous_scale="RdYlGn")
                                fig.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0), template="plotly_white")
                                st.plotly_chart(fig, use_container_width=True, key=f"exec_dis_{proj['id']}")
                            else:
                                st.caption("Awaiting sector updates.")
                            
                        with p_col2:
                            st.write("**Shared Causal Stream**")
                            st.markdown(f"*{proj['description']}*")
                            st.write("---")
                            
                            for update in project_data.get('updates', []):
                                u_color = "#C60C30" if update['update_type'] == 'POLICY_ACTION' else "#006747" if update['update_type'] == 'OBSERVATION' else "#1F2937"
                                st.markdown(f"**<span style='color:{u_color};'>[{update['update_type']}]</span> {update['author_name']}**", unsafe_allow_html=True)
                                st.write(update['content'])
                                if update['certainty']:
                                    st.caption(f"Certainty Index: {update['certainty']:.2f}/1.0")
                                st.write("")
                                
                        with p_col3:
                            with st.container(border=True):
                                st.write("Executive Directives")
                                new_policy = st.text_area("Post Policy Action", key=f"pol_text_{proj['id']}")
                                if st.button("Inject Action to Project Stream", type="primary", key=f"pol_btn_{proj['id']}"):
                                    if new_policy:
                                        ProjectManager.add_update(proj['id'], st.session_state.get('username', "Executive Tier"), 'POLICY_ACTION', new_policy)
                                        st.rerun()
                                    else:
                                        st.error("Text required.")
                                        
                            with st.expander("Executive Override: Force Archive"):
                                st.write("Executive Override: Force resolve this project and execute network-wide trust recalibration.")
                                res_state = st.selectbox("Resolution State", ['RESOLVED', 'FALSE_ALARM', 'INSUFFICIENT_EVIDENCE', 'CONFLICTING_SIGNALS'], key=f"exec_res_{proj['id']}")
                                pol_score = st.slider("National Policy Effectiveness", 0.0, 10.0, 5.0, 0.5, key=f"exec_score_{proj['id']}")
                                res_sum = st.text_area("Executive Debrief", key=f"exec_sum_{proj['id']}")
                                
                                if st.button("Force Archive & Calibrate Network Weights", key=f"exec_arch_{proj['id']}", type="primary"):
                                    if not res_sum:
                                        st.error("Executive debrief summary required.")
                                    else:
                                        payload = {"final_severity": proj['severity'], "active_participants": len(project_data['participants']), "archived_by": "Executive"}
                                        ProjectManager.archive_project(proj['id'], res_state, pol_score, res_sum, payload)
                                        st.rerun()

    with tab_history:
        with st.container(border=True):
            st.info("#### National Institutional Memory")
            st.write("Review all closed Operational Projects and their effect on national intelligence baselines.")
            
            if not memories:
                st.write("No closed projects in the national archive.")
            else:
                for mem in memories:
                    res_color = "#006747" if mem['resolution_state'] == 'RESOLVED' else "#C60C30" if mem['resolution_state'] == 'FALSE_ALARM' else "#1F2937"
                    with st.expander(f"National Archive: {mem['title']} [{mem['resolution_state']}]"):
                        m_col1, m_col2 = st.columns([2, 1])
                        with m_col1:
                            st.markdown(f"**Executive Debrief:**<br>{mem['resolution_summary']}", unsafe_allow_html=True)
                            st.write("---")
                            st.json(mem['learning_payload'])
                        with m_col2:
                            st.metric("Policy Outcome Score", f"{mem['policy_effectiveness_score']}/10.0")
                            ttc_mins = mem['time_to_consensus_seconds'] / 60.0
                            st.metric("Coordination Time", f"{ttc_mins:.1f} minutes")
                            st.markdown(f"<span style='color:{res_color};'>**Network Weights Recalibrated**</span>", unsafe_allow_html=True)
                            
    with tab_comms:
        with st.container(border=True):
            st.warning("### National Command & Control")
            st.write("Intercept escalations from Sector Admins and broadcast policy overrides.")
            
            c1, c2 = st.columns([1, 1])
            with c1:
                with st.container(border=True):
                    st.write("National Escalations (Inbox)")
                    esc_inbox = SecureMessaging.get_inbox(Role.EXECUTIVE.value, "ALL") # Exec receives ALL escalations
                    if not esc_inbox:
                        st.caption("No pending escalations.")
                    else:
                        for msg in esc_inbox:
                            with st.expander(f"From {msg['sender_id']} | {msg['timestamp']} {'(NEW)' if not msg['is_read'] else ''}"):
                                st.write(msg['content'])
                                if not msg['is_read']:
                                    if st.button("Mark Cleared", key=f"ex_clear_{msg['id']}"):
                                        SecureMessaging.mark_read(msg['id'])
                                        st.rerun()

            with c2:
                with st.container(border=True):
                    st.write("Issue National Directives (Downward)")
                    target_level = st.selectbox("Target Hierarchy Level", ["Sector Admins", "Local Institutions (Spokes)", "Global Broadcast"])
                    target_id = st.text_input("Target Node Username (or 'ALL')")
                    directive = st.text_area("Command Payload", height=100)
                    if st.button("Transmit Command", type="primary", use_container_width=True):
                        if directive and target_id:
                            rec_role = Role.BASKET_ADMIN.value
                            if target_level == "Local Institutions (Spokes)":
                                rec_role = Role.INSTITUTION.value
                            elif target_level == "Global Broadcast":
                                rec_role = "ALL_ROLES"
                                target_id = "ALL"
                                
                            if rec_role == "ALL_ROLES":
                                SecureMessaging.send_message(Role.EXECUTIVE.value, st.session_state.get('username', 'Executive node'), Role.BASKET_ADMIN.value, "ALL", directive)
                                SecureMessaging.send_message(Role.EXECUTIVE.value, st.session_state.get('username', 'Executive node'), Role.INSTITUTION.value, "ALL", directive)
                            else:
                                SecureMessaging.send_message(
                                    sender_role=Role.EXECUTIVE.value,
                                    sender_id=st.session_state.get('username', 'Executive node'),
                                    receiver_role=rec_role,
                                    receiver_id=target_id,
                                    content=directive
                                )
                            st.success("Executive Command transmitted to target(s).")
                    


    with st.container(border=True):
        st.write("Privacy Matrix Status: Guaranteed 100% Differential Privacy. No raw citizen data or classified operational logs are exposed to this console.")
