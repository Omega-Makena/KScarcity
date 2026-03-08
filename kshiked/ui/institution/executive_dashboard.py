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
from kshiked.ui.institution.collab_room import render_collab_room
from kshiked.ui.institution.executive_simulator import render_executive_simulator
from kshiked.ui.institution.backend.analytics_engine import (
    generate_inaction_projection,
    get_historical_context,
    build_county_convergence,
    generate_recommendation,
    compute_outcome_impact,
    get_county_centroid,
)
from kshiked.ui.institution.backend.report_narrator import (
    narrate_risk_for_executive,
    narrate_severity,
)

@st.cache_data(ttl=3600)
def load_and_process_geojson(geojson_path, county_scores=None):
    import json
    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson_data = json.load(f)

    for feature in geojson_data['features']:
        county_name = feature['properties'].get('shapeName', '').lower()

        # Use real data if available, otherwise show zero (honest absence)
        if county_scores and county_name in county_scores:
            stress = int(county_scores[county_name]['score'])
            has_data = True
        else:
            stress = 0
            has_data = False

        feature['properties']['stress'] = stress
        feature['properties']['has_data'] = has_data

        if not has_data:
            r, g, b = 148, 163, 184  # Slate gray — no data
        elif stress > 80:
            r, g, b = 239, 68, 68   # Red
        elif stress > 50:
            r, g, b = 245, 158, 11  # Amber
        elif stress > 20:
            r, g, b = 59, 130, 246  # Blue
        else:
            r, g, b = 16, 185, 129  # Green

        alpha = 140 if has_data else 60
        feature['properties']['color'] = [r, g, b, alpha]
        feature['properties']['line_color'] = [r, g, b, 255]
        feature['properties']['elevation'] = stress * 1200 if has_data else 200

    return geojson_data

@st.cache_data(ttl=300)
def cached_simulate_policy_shock(target_idx, magnitude, steps):
    return ExecutiveBridge.simulate_policy_shock(target_idx, magnitude, steps=steps)

@st.cache_data(ttl=3600)
def get_pulse_data():
    path = os.path.join(project_root, "data", "synthetic_kenya_policy", "tweets.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    cols = ["timestamp", "intent", "topic_cluster", "sentiment_score", "threat_score", "policy_event_id"]
    try:
        df = pd.read_csv(path, usecols=cols).dropna(subset=['intent'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        df.rename(columns={
            'topic_cluster': 'Sector',
            'intent': 'Threat Category',
            'sentiment_score': 'Sentiment',
            'timestamp': 'Timestamp'
        }, inplace=True)
        
        df['Sector'] = df['Sector'].fillna("General Systemic")
        df['Criticality'] = (df['threat_score'] * 8.0) + ((1.0 - df['Sentiment']) * 2.0)
        df['Criticality'] = df['Criticality'].clip(lower=0.0, upper=10.0).round(2)
        
        return df
    except Exception as e:
        st.error(f"Failed to load pulse data: {e}")
        return pd.DataFrame()


def render():
    enforce_role(Role.EXECUTIVE.value)
    inject_enterprise_theme()
    
    st.markdown("<h2 style='text-align: center; color: #1F2937;'>National Executive Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: #BB0000;'>Presidency Intelligence & Coordinated Response</h4>", unsafe_allow_html=True)
    
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

    # Pre-fetch inbox so unread count is available everywhere
    esc_inbox_global = SecureMessaging.get_inbox(Role.EXECUTIVE.value, "ALL")
    unread_escs = sum(1 for m in esc_inbox_global if not m['is_read'])

    # 1. THE 10-TAB ARCHITECTURE
    tab_brief, tab_sectors, tab_map, tab_feed, tab_social, tab_sim, tab_projects, tab_comms, tab_summaries, tab_history, tab_collab = st.tabs([
        "National Briefing",
        "Sector Reports",
        "National Map",
        "Threat Intelligence",
        "Social Signals",
        "Policy Simulator",
        "Active Operations",
        "Command & Control",
        "Sector Summaries",
        "Archive",
        "Collaboration Room",
    ])

    with tab_brief:
        st.markdown("### National Briefing")
        df_social = get_pulse_data()
        avg_sent = df_social['Sentiment'].mean() if not df_social.empty else 0.50
        
        # 1. THE EXECUTIVE BRIEFING (Hero Section)
        strain_score = len(global_risks) * 2.5
        if strain_score > 10.0: strain_score = 10.0
        
        # Dynamic Morning Narrative
        brief_text = f"National Systemic Strain is currently <b>{'Critical' if strain_score > 7 else 'Moderate' if strain_score > 3 else 'Low'}</b> ({strain_score:.1f}/10). "
        if global_risks:
            primary = global_risks[0]
            brief_text += f"The primary driver is an escalating threat involving <i>{primary['title']}</i> within the <b>{all_baskets.get(primary['basket_id'], 'unknown')}</b> sector. "
            if len(global_risks) > 1:
                brief_text += f"There are {len(global_risks) - 1} secondary anomalies detected across the topography. "
            
        if active_projects:
            most_severe_proj = sorted(active_projects, key=lambda x: x['severity'], reverse=True)[0]
            brief_text += f"<br><br>The administration is currently tracking <b>{len(active_projects)} active National Projects</b>. "
            brief_text += f"Immediate executive oversight is required on Project <i>{most_severe_proj['title']}</i> (Severity: {most_severe_proj['severity']}, Phase: {most_severe_proj['current_phase']}). "
        else:
            brief_text += "<br><br>There are currently no active cross-sector Operational Projects requiring executive coordination."
            
        if unread_escs > 0:
            brief_text += f" <span style='color: #BB0000; font-weight: bold;'>You have {unread_escs} unread escalations awaiting Command & Control clearance.</span>"
        
        st.markdown(f"""
            <div class="hero-brief">
                <h3>Executive Summary</h3>
                <p>{brief_text}</p>
            </div>
        """, unsafe_allow_html=True)
        
        m1, m2, m3, m4 = st.columns(4)
        with m1: st.metric("Active Threat Signals", len(global_risks))
        with m2: st.metric("Systemic Strain", f"{strain_score:.1f}/10")
        with m3: st.metric("Active Projects", len(active_projects))
        with m4: st.metric("National Sentiment", f"{avg_sent:.2f}")
            
        st.write("---")
        
        # Layout for Visualizations & Top Priorities
        bc1, bc2 = st.columns([1.5, 1])
        
        with bc1:
            st.markdown("#### Strategic Threat Distribution")
            if global_risks:
                import pandas as pd
                import plotly.express as px
                threat_data = []
                for r in global_risks:
                    threat_data.append({
                        "Sector": all_baskets.get(r['basket_id'], 'Unknown'),
                        "Impact": r.get('composite_scores', {}).get('B_Impact', 5),
                        "Title": r['title']
                    })
                df_threats = pd.DataFrame(threat_data)
                fig_bar = px.bar(df_threats, x="Sector", y="Impact", color="Impact", 
                                 title="Cumulative Threat Impact by Sector",
                                 color_continuous_scale="Reds", template="plotly_white")
                fig_bar.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.success("No active threats.")
                
            st.markdown("#### National Projects by Phase")
            if active_projects:
                proj_data = [{"Phase": p['current_phase'], "Title": p['title']} for p in active_projects]
                df_proj = pd.DataFrame(proj_data)
                fig_donut = px.pie(df_proj, names="Phase", hole=0.5, 
                                   title="Operational Projects Distribution",
                                   color_discrete_sequence=px.colors.sequential.Greens_r, template="plotly_white")
                fig_donut.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.success("No active projects.")
                
        with bc2:
            st.markdown("#### Top Priorities")
            st.write("Immediate Attention Required:")
            
            if unread_escs > 0:
                st.markdown(f"""
                <div style="background: #FEF2F2; border-left: 4px solid #EF4444; padding: 12px; margin-bottom: 10px; border-radius: 4px;">
                    <strong>{unread_escs} Unread Escalations</strong>
                    <div style="font-size: 0.8rem; color: #7F1D1D;">Check Command & Control Inbox</div>
                </div>
                """, unsafe_allow_html=True)
                
            if active_projects:
                top_p = sorted(active_projects, key=lambda x: x['severity'], reverse=True)[0]
                st.markdown(f"""
                <div style="background: #FFFBEB; border-left: 4px solid #F59E0B; padding: 12px; margin-bottom: 10px; border-radius: 4px;">
                    <strong>Project: {top_p['title']}</strong>
                    <div style="font-size: 0.8rem; color: #92400E;">Phase: {top_p['current_phase']} | Severity: {top_p['severity']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            if global_risks:
                top_r = sorted(global_risks, key=lambda x: x.get('composite_scores', {}).get('B_Impact', 0), reverse=True)[0]
                st.markdown(f"""
                <div style="background: #F0FDF4; border-left: 4px solid #10B981; padding: 12px; margin-bottom: 10px; border-radius: 4px;">
                    <strong>Signal: {top_r['title']}</strong>
                    <div style="font-size: 0.8rem; color: #065F46;">Sector: {all_baskets.get(top_r['basket_id'], 'unknown')}</div>
                </div>
                """, unsafe_allow_html=True)
            
    # --- SECTOR REPORTS TAB ---
    with tab_sectors:
        st.markdown("### Sector Reports")
        st.write("Validated risks and recent activity reported by each sector admin. This is Mode A governance data — no raw institutional data is shown.")

        fl_mode = st.session_state.get('fl_mode_enabled', False)

        if not global_risks and not active_projects:
            st.success("No active sector reports at this time.")
        else:
            for b_id, b_name in all_baskets.items():
                sector_risks = [r for r in global_risks if r.get('basket_id') == b_id]
                sector_projects = [p for p in active_projects if b_id in (ProjectManager.get_project_details(p['id']).get('participants', []))]
                total_impact = sum(r.get('composite_scores', {}).get('B_Impact', 0) for r in sector_risks)

                with st.expander(f"{b_name}  |  {len(sector_risks)} validated risk(s)  |  Cumulative impact: {total_impact:.1f}", expanded=len(sector_risks) > 0):
                    if not sector_risks:
                        st.caption("No validated risks from this sector.")
                    else:
                        for risk in sector_risks:
                            scores = risk.get('composite_scores', {})
                            impact = scores.get('B_Impact', 0)
                            detection = scores.get('A_Detection', 0)
                            certainty = scores.get('C_Certainty', 0)
                            ts = pd.to_datetime(risk.get('timestamp', 0), unit='s').strftime('%Y-%m-%d %H:%M')
                            sev_color = "#BB0000" if impact > 7 else "#F59E0B" if impact > 4 else "#006600"

                            # Executive-ready narrative paragraph
                            exec_narrative = narrate_risk_for_executive(
                                title=risk.get('title', 'Untitled Risk'),
                                description=risk.get('description', ''),
                                composite_scores=scores,
                                severity=impact,
                                sector_name=b_name,
                                threat_level=risk.get('threat_level', ''),
                            )
                            st.markdown(
                                f'<div style="background:#F8FAFC; border-left:4px solid {sev_color}; padding:12px 16px; '
                                f'border-radius:0 6px 6px 0; margin-bottom:0.5rem; font-size:0.9rem; line-height:1.6;">'
                                f'{exec_narrative}</div>',
                                unsafe_allow_html=True
                            )

                            st.markdown(
                                f'<div style="padding:0.3rem 1rem; margin-bottom:0.3rem; font-size:0.82rem; color:#64748b;">'
                                f'Detection {detection:.1f}/10 &middot; Impact {impact:.1f}/10 &middot; Certainty {certainty:.1f}/10 &nbsp;&nbsp;|&nbsp;&nbsp;{ts}'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                            # ── PILLAR 1: "SO WHAT?" ──
                            projection = generate_inaction_projection(
                                severity=impact,
                                incident_type='PROMOTED_RISK',
                                composite_scores=scores,
                            )
                            if projection:
                                st.markdown(
                                    f'<div style="background:#FEF2F2; border-left:4px solid #DC2626; padding:8px 12px; '
                                    f'border-radius:0 6px 6px 0; margin:0 0 4px 0; font-size:0.84rem;">'
                                    f'<strong>⚠ So What?</strong> {projection}</div>',
                                    unsafe_allow_html=True,
                                )

                            # ── PILLAR 2: "COMPARED TO WHAT?" ──
                            hist_ctx = get_historical_context(
                                basket_id=b_id,
                                severity=impact,
                                incident_type='PROMOTED_RISK',
                            )
                            st.markdown(
                                f'<div style="background:#F0F9FF; border-left:4px solid #3B82F6; padding:8px 12px; '
                                f'border-radius:0 6px 6px 0; margin:0 0 4px 0; font-size:0.84rem;">'
                                f'<strong>📊 Compared to What?</strong> {hist_ctx}</div>',
                                unsafe_allow_html=True,
                            )

                            # ── PILLAR 4: "WHAT SHOULD I DO?" ──
                            rec = generate_recommendation(
                                risk=risk,
                                all_baskets=all_baskets,
                                global_risks=global_risks,
                            )
                            st.markdown(
                                f'<div style="background:#FFFBEB; border-left:4px solid {rec.level_color}; '
                                f'padding:8px 12px; border-radius:0 6px 6px 0; margin:0 0 10px 0;">'
                                f'<strong>🎯 <span style="background:{rec.level_color}; color:#fff; padding:2px 8px; '
                                f'border-radius:4px; font-size:0.78rem;">{rec.level}</span></strong> '
                                f'<span style="font-size:0.84rem;">{rec.summary}</span><br/>'
                                f'<span style="font-size:0.80rem; color:#64748b;">'
                                f'<b>Who:</b> {", ".join(rec.who[:3])} &nbsp;|&nbsp; '
                                f'<b>Urgency:</b> {rec.urgency}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

    with tab_map:
        st.markdown("### National Map")
        st.write("Geospatial distribution of emerging hotspots across Kenyan counties — driven by real promoted risk data.")
        import pydeck as pdk
        import json
        import os

        # ── PILLAR 3: "WHERE EXACTLY?" — Real geographic data ──
        county_scores = build_county_convergence(global_risks, all_baskets)

        if not county_scores:
            st.info(
                "📍 **Geographic metadata unavailable.** No promoted risks contain "
                "county-level geographic data. The map will activate once risks "
                "with spatial metadata are promoted by sector admins."
            )

        # Load local Kenyan counties GeoJSON with real data
        geojson_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "kenya_adm1_simplified.geojson")
        try:
            geojson_data = load_and_process_geojson(geojson_path, county_scores or None)

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

            tooltip_html = (
                "<b>County:</b> {shapeName} <br/> "
                "<b>Convergence Score:</b> {stress}/100 <br/>"
                "<b>Data:</b> {'Real signal data' if {has_data} else 'No data'}"
            )

            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={
                    "html": "<b>County:</b> {shapeName} <br/> <b>Convergence Score:</b> {stress}/100",
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

            # Legend
            if county_scores:
                data_counties = len(county_scores)
                total_risks = sum(c['risk_count'] for c in county_scores.values())
                st.caption(
                    f"🟢 Low (&lt;20) &nbsp; 🔵 Moderate (20-50) &nbsp; 🟠 Elevated (50-80) &nbsp; 🔴 Critical (&gt;80) &nbsp; ⚪ No data \n\n"
                    f"**{data_counties} counties** with signal data from **{total_risks} promoted risks**."
                )
            else:
                st.caption("⚪ All counties shown in grey — no active signal data.")

        except Exception as e:
            st.error(f"Error loading geospatial topography: {e}")
    
    with tab_feed:
        st.markdown("### Threat Intelligence")
        st.write("Validated risks promoted by sector admins. These have been reviewed and assessed at the sector level before reaching this dashboard.")
        if not global_risks:
            st.success("No active systemic signals detected.")
        else:
            # Custom HTML Threat Cards
            html_cards = ""
            for risk in global_risks:
                scores = risk.get('composite_scores', {})
                b_impact = scores.get('B_Impact', 0)
                sev_class = "high" if b_impact > 7 else "medium" if b_impact > 4 else "low"
                sector_name = all_baskets.get(risk['basket_id'], f"Sector {risk['basket_id']}")
                detected = pd.to_datetime(risk.get('timestamp', time.time()), unit='s').strftime('%H:%M')
                
                html_cards += f"""
                <div class="threat-card {sev_class}">
                    <div class="threat-header">
                        <span class="threat-title">{risk['title']}</span>
                        <span class="threat-meta">{detected}</span>
                    </div>
                    <div>
                        <span class="threat-sector">{sector_name}</span>
                        <span style="font-size:0.8rem; color:#64748b; margin-left:8px;">Impact: {b_impact}/10</span>
                    </div>
                </div>
                """
            st.markdown(html_cards, unsafe_allow_html=True)
            
            # Causal text
            primary_risk = global_risks[0]
            st.markdown(f"**Causal Node:** Anomalous volume detected in *{all_baskets.get(primary_risk['basket_id'], 'unknown')}*.")
            
    with tab_social:
        st.markdown("### Social Signal Monitor")
        df_social = get_pulse_data()
        
        if df_social.empty:
            st.warning("No real-time social signals available to map.")
        else:
            # Filters
            sc1, sc2 = st.columns(2)
            with sc1:
                sel_sector = st.multiselect("Filter Sector", df_social['Sector'].unique(), default=df_social['Sector'].unique()[:3] if len(df_social['Sector'].unique()) > 3 else df_social['Sector'].unique())
            with sc2:
                sel_threat = st.multiselect("Filter Threat", df_social['Threat Category'].unique(), default=df_social['Threat Category'].unique()[:3] if len(df_social['Threat Category'].unique()) > 3 else df_social['Threat Category'].unique())
                
            df_filtered = df_social[(df_social['Sector'].isin(sel_sector)) & (df_social['Threat Category'].isin(sel_threat))]
            
            if len(df_filtered) > 3000:
                df_filtered = df_filtered.sample(n=3000, random_state=42)
            
            import plotly.express as px
            if not df_filtered.empty:
                fig = px.scatter(
                    df_filtered, x="Timestamp", y="Criticality", 
                    color="Sector", size="Criticality", hover_data=["Threat Category", "Sentiment"],
                    template="plotly_white", height=300
                )
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig, use_container_width=True)
                
                # Inline brief
                avg_sent = df_filtered['Sentiment'].mean()
                st.caption(f"**National Sentiment Baseline:** {avg_sent:.2f} | Most strained sector: {df_filtered.groupby('Sector')['Criticality'].mean().idxmax()}")
                
    with tab_sim:
        render_executive_simulator()
        
    with tab_projects:
        with st.container(border=True):
            st.markdown("#### Launch National Operational Project")
            st.write("Create a cross-sector coordination project and assign sector admins to it.")
            
            p_c1, p_c2, p_c3 = st.columns([1, 1, 1])
            with p_c1:
                p_title = st.text_input("Project Codename/Title", key="ex_p_title")
                p_sev = st.selectbox("Initial Severity", [1, 2, 3, 4, 5], index=2, key="ex_p_sev")
            with p_c2:
                p_desc = st.text_area("Strategic Objective", key="ex_p_desc", height=100)
            with p_c3:
                p_baskets = st.multiselect("Assign Sector Admins", options=list(all_baskets.keys()), format_func=lambda x: all_baskets[x], key="ex_p_baskets")
                st.write("")
                if st.button("Launch National Operational Project", type="primary", use_container_width=True, key="ex_btn_launch"):
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
            st.info("#### Active Operational Projects")
            st.write("Monitor multi-sector collaboration projects. Inject top-down Policy Actions directly into their intelligence streams.")
            
            if not active_projects:
                st.success("No active cross-sector Operational Projects.")
            else:
                for proj in active_projects:
                    with st.expander(f"ACTIVE PROJECT: {proj['title']} (Severity: {proj['severity']})", expanded=True):
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
            st.markdown("#### National Archive")
            st.write("Closed operations and the outcomes they recorded.")
            
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

                            st.json(mem['learning_payload'])
                        with m_col2:
                            st.metric("Policy Outcome Score", f"{mem['policy_effectiveness_score']}/10.0")
                            ttc_mins = mem['time_to_consensus_seconds'] / 60.0
                            st.metric("Coordination Time", f"{ttc_mins:.1f} minutes")
                            st.markdown(f"<span style='color:{res_color};'>**Network Weights Recalibrated**</span>", unsafe_allow_html=True)
                            
    with tab_comms:
        with st.container(border=True):
            st.markdown("### Command & Control")
            st.write("Read escalations from sector admins and issue directives downward.")
            
            c1, c2 = st.columns([1, 1])
            with c1:
                with st.container(border=True):
                    st.write("National Escalations (Inbox)")
                    esc_inbox = esc_inbox_global
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
                    st.write("Issue directives (Downward)")
                    target_level = st.selectbox("Send to", ["Sector Admins", "Local Institutions", "Broadcast to all"])
                    target_id = st.text_input("Username (or 'ALL' to broadcast)")
                    directive = st.text_area("Command Payload", height=100)
                    if st.button("Transmit Command", type="primary", use_container_width=True):
                        if directive and target_id:
                            rec_role = Role.BASKET_ADMIN.value
                            if target_level == "Local Institutions":
                                rec_role = Role.INSTITUTION.value
                            elif target_level == "Broadcast to all":
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
                    


    with tab_summaries:
        st.markdown("### Sector Summaries")
        st.write("Per-sector snapshot: validated risks, operational project participation, and public sentiment.")
        st.write("---")

        df_social_sum = get_pulse_data()

        if not all_baskets:
            st.info("No sectors registered in the system.")
        else:
            cols_per_row = 2
            basket_list = list(all_baskets.items())
            for row_start in range(0, len(basket_list), cols_per_row):
                row_items = basket_list[row_start:row_start + cols_per_row]
                sum_cols = st.columns(len(row_items))
                for col_idx, (b_id, b_name) in enumerate(row_items):
                    sector_risks = [r for r in global_risks if r.get('basket_id') == b_id]
                    sector_projects = [p for p in active_projects]
                    risk_count = len(sector_risks)
                    proj_count = len(sector_projects)

                    # Aggregate composite impact
                    total_impact = sum(r.get('composite_scores', {}).get('B_Impact', 0) for r in sector_risks)
                    avg_impact = (total_impact / risk_count) if risk_count else 0.0

                    # Social sentiment for this sector (match by sector name in pulse data)
                    if not df_social_sum.empty and 'Sector' in df_social_sum.columns:
                        sector_mask = df_social_sum['Sector'].str.contains(b_name.split()[0], case=False, na=False)
                        sector_df = df_social_sum[sector_mask]
                        avg_sentiment = sector_df['Sentiment'].mean() if not sector_df.empty else df_social_sum['Sentiment'].mean()
                    else:
                        avg_sentiment = None

                    # Color-code card border by severity
                    if risk_count == 0:
                        border_color = "#10B981"  # green — clear
                        status_label = "Clear"
                    elif avg_impact > 7:
                        border_color = "#EF4444"  # red — critical
                        status_label = "Critical"
                    elif avg_impact > 4:
                        border_color = "#F59E0B"  # amber — elevated
                        status_label = "Elevated"
                    else:
                        border_color = "#3B82F6"  # blue — low
                        status_label = "Low"

                    sentiment_str = f"{avg_sentiment:.2f}" if avg_sentiment is not None else "N/A"

                    with sum_cols[col_idx]:
                        st.markdown(
                            f'<div style="border: 2px solid {border_color}; border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem; background: #f8fafc;">'
                            f'<div style="font-weight: bold; font-size: 1rem; color: #1F2937;">{b_name}</div>'
                            f'<div style="margin-top: 0.4rem;">'
                            f'<span style="background:{border_color}; color:#fff; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:bold;">{status_label}</span>'
                            f'</div>'
                            f'<div style="margin-top: 0.6rem; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.3rem;">'
                            f'<div style="text-align:center;"><div style="font-size:1.4rem; font-weight:bold; color:{border_color};">{risk_count}</div><div style="font-size:0.7rem; color:#64748b;">Risks</div></div>'
                            f'<div style="text-align:center;"><div style="font-size:1.4rem; font-weight:bold; color:#1F2937;">{proj_count}</div><div style="font-size:0.7rem; color:#64748b;">Projects</div></div>'
                            f'<div style="text-align:center;"><div style="font-size:1.4rem; font-weight:bold; color:#1F2937;">{sentiment_str}</div><div style="font-size:0.7rem; color:#64748b;">Sentiment</div></div>'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        if sector_risks:
                            top_risk = sorted(sector_risks, key=lambda x: x.get('composite_scores', {}).get('B_Impact', 0), reverse=True)[0]
                            st.caption(f"Top risk: {top_risk['title'][:60]}")

    with tab_collab:
        render_collab_room(
            role=Role.EXECUTIVE.value,
            basket_id=None,
            username=st.session_state.get('username', 'executive'),
            all_baskets=all_baskets,
        )
