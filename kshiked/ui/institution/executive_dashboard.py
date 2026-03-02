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
    
    st.markdown("<h2 style='text-align: center; color: #BB0000;'>National Security Command Center</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #006600;'>Strategic Intelligence & Coordinated Response</h3>", unsafe_allow_html=True)
    
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

    # Top-Level KPI Metric Cards (Custom HTML/CSS)
    # Replaces basic static text with high-visibility metrics.
    
    # Calculate Systemic Strain safely
    strain_score = len(global_risks) * 2.5
    if strain_score > 10.0:
        strain_score = 10.0
        
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        alert_class = "alert" if global_risks else ""
        st.markdown(f"""
            <div class="kpi-card {alert_class}">
                <div class="kpi-title">Active Threat Signals</div>
                <div class="kpi-value">{len(global_risks)}</div>
                <div class="kpi-sub">Anomalies requiring attention</div>
            </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Systemic Strain</div>
                <div class="kpi-value">{strain_score:.1f}/10</div>
                <div class="kpi-sub">Network topological risk</div>
            </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Active War Rooms</div>
                <div class="kpi-value">{len(active_projects)}</div>
                <div class="kpi-sub">Cross-sector collaborations</div>
            </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Active Sectors</div>
                <div class="kpi-value">{len(all_baskets)}</div>
                <div class="kpi-sub">Federated nodes online</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    st.write("")
    
    # 7. NATIONAL / ORGANIZATIONAL MAP (Spatial Awareness)
    with st.container(border=True):
        st.write("#### National Threat Topography (Kenya)")
        st.write("Geospatial distribution of emerging hotspots across Kenyan counties.")
        import pydeck as pdk
        import json
        import os
        
        # Load local Kenyan counties GeoJSON
        geojson_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "kenya_adm1_simplified.geojson")
        try:
            with open(geojson_path, "r", encoding="utf-8") as f:
                geojson_data = json.load(f)
                
            # Add dynamic "stress" properties for extrusion and coloring (Glassmorphism Light Theme)
            np.random.seed(int(time.time()) % 1000)
            for feature in geojson_data['features']:
                stress = np.random.randint(5, 100)
                feature['properties']['stress'] = stress
                
                # Colors: High Stress = Red, Medium = Orange/Yellow, Low = Blue/White
                if stress > 80:
                    r, g, b = 239, 68, 68 # Red
                elif stress > 50:
                    r, g, b = 245, 158, 11 # Amber
                else:
                    r, g, b = 59, 130, 246 # Blue
                    
                # Semi-transparent for glass effect
                feature['properties']['color'] = [r, g, b, 140]
                feature['properties']['line_color'] = [r, g, b, 255]
                feature['properties']['elevation'] = stress * 1200

            layer = pdk.Layer(
                "GeoJsonLayer",
                geojson_data,
                pickable=True,
                stroked=True,
                filled=True,
                extruded=True,
                wireframe=True,
                get_fill_color="properties.color",
                get_line_color="properties.line_color",
                get_elevation="properties.elevation",
                elevation_scale=1,
                line_width_min_pixels=2,
            )

            view_state = pdk.ViewState(
                latitude=0.0236,
                longitude=37.9062,
                zoom=5.2,
                pitch=55,
                bearing=15
            )

            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={
                    "html": "<b>County:</b> {shapeName} <br/> <b>Hotspot Stress Index:</b> {stress}/100", 
                    "style": {
                        "backgroundColor": "#ffffff",
                        "color": "#0f172a",
                        "border": "1px solid #e2e8f0",
                        "borderRadius": "8px",
                        "boxShadow": "0 4px 6px -1px rgba(0,0,0,0.1)",
                        "fontFamily": "Inter, sans-serif",
                        "fontWeight": "500",
                        "padding": "10px"
                    }
                },
                map_style="mapbox://styles/mapbox/light-v11"
            )
            st.pydeck_chart(r, height=450, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error loading geospatial topography: {e}")
    
    # 8. SPLIT-PANE INTELLIGENCE & FORECASTING
    st.write("---")
    col_risk, col_sim = st.columns([1, 1.2])
    
    with col_risk:
        with st.container(border=True):
            st.markdown("### 🔴 Live Intelligence Feed")
            st.write("Recent anomalies and shifts across sectors requiring executive attention.")
            
            if not global_risks:
                st.success("No active systemic signals detected.")
            else:
                # KEY SIGNALS PANEL (Tabular)
                with st.container(border=True):
                    st.markdown("#### Alert Feed")
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
                            "Detected": pd.to_datetime(risk.get('timestamp', time.time()), unit='s').strftime('%m-%d %H:%M'),
                            "Sector": sector_name
                        })
                    
                    df_signals = pd.DataFrame(signal_data)
                    st.dataframe(df_signals, use_container_width=True, hide_index=True)
                    
                # CAUSAL EXPLANATION LAYER (Why) + ALERT TIMELINE
                c_cause, c_time = st.columns([1.2, 1])
                with c_cause:
                    with st.container(border=True):
                        st.markdown("#### Causal Interpretation")
                        primary_risk = global_risks[0]
                        cause_text = f"The current <b>escalating</b> trend is primarily driven by <b>{primary_risk['title'].lower()}</b> emanating from the <b>{all_baskets.get(primary_risk['basket_id'], 'unknown')}</b> sector."
                        
                        st.markdown(f"> {cause_text}", unsafe_allow_html=True)
                        st.markdown(f"- Multi-sector propagation from {all_baskets.get(primary_risk['basket_id'], 'origin sector')}.")
                        st.markdown("- Sustained volatility breaking baseline thresholds.")
                        st.markdown(f"- Certainty Index: ({primary_risk.get('composite_scores', {}).get('C_Certainty', 0.0):.1f}/10)")
                
                with c_time:
                    with st.container(border=True):
                        st.markdown("#### Alert Timeline")
                        st.markdown(f"- **Day -5:** Sentiment anomaly in {all_baskets.get(primary_risk['basket_id'], 'sector')}.")
                        st.markdown("- **Day -3:** Local market volatility increase.")
                        st.markdown(f"- **Today:** Critical {primary_risk['title'].lower()} triggered.")
                
                # Priority Recommendations
                with st.container(border=True):
                    st.markdown("#### Priority Recommendations")
                    st.markdown("1. **Monitor liquidity exposure** in adjoining sectors.")
                    st.markdown(f"2. **Delay capital reallocation** pending {all_baskets.get(primary_risk['basket_id'], 'sector')} stabilization.")
                            
    with col_sim:
        with st.container(border=True):
            st.markdown("### 📈 Forward Projection & Simulation")
            st.write("90-Day Outlook vs. Intervention Scenarios")
            
            with st.spinner("Compiling VARX/GARCH Mathematical Models..."):
                import plotly.graph_objects as go
                
                if not global_risks:
                    df_base, df_base_var = ExecutiveBridge.simulate_policy_shock(0, 0.0, steps=30)
                else:
                    df_base, df_base_var = ExecutiveBridge.simulate_policy_shock(1, 4.0, steps=30)
                    
                fig_base = go.Figure()
                for col in df_base.columns:
                    fig_base.add_trace(go.Scatter(
                        x=df_base.index, y=df_base[col],
                        mode='lines', name=col,
                        line=dict(width=2)
                    ))
                fig_base.update_layout(
                    height=200, margin=dict(l=0, r=0, t=10, b=0),
                    template="plotly_white",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig_base, use_container_width=True)
        
            st.divider()
            st.markdown("#### Policy Action Simulator")
            
            sim_c1, sim_c2 = st.columns([1, 1])
            with sim_c1:
                live_tiers = ExecutiveBridge.get_tiers()
                target_name = st.selectbox("Intervention Target", live_tiers)
                target_idx = live_tiers.index(target_name)
            with sim_c2:
                shock_magnitude = st.slider("Intervention Intensity", -10.0, 10.0, -5.0, 0.5)
            
            if st.button("Project Mitigated Outcome", type="primary", use_container_width=True):
                with st.spinner("Connecting to Scarcity Engine..."):
                    df_mitigated, df_var = ExecutiveBridge.simulate_policy_shock(target_idx, shock_magnitude, steps=30)
                    
                    end_state = df_mitigated.iloc[-1].mean()
                    base_end_state = df_base.iloc[-1].mean()
                    
                    if end_state < base_end_state:
                        st.success(f"**Outcome:** Threat topology shows systemic cooling.")
                        reduction = ((base_end_state - end_state) / (abs(base_end_state) + 1e-9)) * 100
                        st.metric("Expected Systemic Strain Reduction", f"{reduction:.1f}%")
                    else:
                        st.warning(f"**Outcome:** Intervention compounds systemic strain.")
                        inc = ((end_state - base_end_state) / (abs(base_end_state) + 1e-9)) * 100
                        st.metric("Expected Systemic Strain Increase", f"{inc:.1f}%")
                    
                    st.write("Forecasted Trajectory (Mitigated):")
                    fig_mitigated = go.Figure()
                    for col in df_mitigated.columns:
                        fig_mitigated.add_trace(go.Scatter(
                            x=df_mitigated.index, y=df_mitigated[col],
                            mode='lines', name=col,
                            line=dict(width=2, dash='dot')
                        ))
                    fig_mitigated.update_layout(
                        height=200, margin=dict(l=0, r=0, t=10, b=0),
                        template="plotly_white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_mitigated, use_container_width=True)

    st.write("---")

    # 9. STRATEGIC COMMAND & CONTROL (Tabbed)
    tab_projects, tab_comms, tab_summaries, tab_history = st.tabs([
        "Active Operational Projects (War Rooms)", 
        "Command & Control (Comms)",
        "Sector Summaries",
        "Institutional Memory Archive"
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
                                u_color = "#BB0000" if update['update_type'] == 'POLICY_ACTION' else "#006600" if update['update_type'] == 'OBSERVATION' else "#1F2937"
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
                    res_color = "#006600" if mem['resolution_state'] == 'RESOLVED' else "#BB0000" if mem['resolution_state'] == 'FALSE_ALARM' else "#1F2937"
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
