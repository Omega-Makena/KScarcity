"""
SENTINEL Command Center Dashboard v2.0

Premium 8-tab dashboard with:
1. Live Threat Map - Kaspersky-inspired real-time visualization
2. Executive Overview - Traffic light, gauge, top threats
3. Signal Intelligence - 15 signals, cascade, heatmap
4. Causal Analysis - Force graph, Granger tests
5. Simulation - 4D economic viz, shocks
6. Escalation Pathways - Decision intelligence
7. Federation - Multi-agency status
8. Operations - County table, alerts, health

Dark theme default, no emojis, premium aesthetics.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

# Ensure ui/ directory is on sys.path for sibling imports (theme, data_connector, etc.)
_UI_DIR = Path(__file__).resolve().parent
if str(_UI_DIR) not in sys.path:
    sys.path.insert(0, str(_UI_DIR))

# Ensure project root is on sys.path for package imports (kshiked, scarcity, etc.)
_PROJECT_ROOT = _UI_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import streamlit as st
    import streamlit.components.v1 as components
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    st = None
    components = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None
    np = None

from theme import DARK_THEME, LIGHT_THEME, THREAT_LEVELS, generate_css, get_plotly_theme
from data_connector import get_dashboard_data, DashboardData

logger = logging.getLogger("sentinel.dashboard")

# Kenya county coordinates for map
KENYA_COUNTIES = {
    "Nairobi": {"lat": -1.2921, "lon": 36.8219},
    "Mombasa": {"lat": -4.0435, "lon": 39.6682},
    "Kisumu": {"lat": -0.0917, "lon": 34.7680},
    "Nakuru": {"lat": -0.3031, "lon": 36.0666},
    "Eldoret": {"lat": 0.5143, "lon": 35.2698},
    "Garissa": {"lat": -0.4533, "lon": 39.6460},
    "Meru": {"lat": 0.0515, "lon": 37.6493},
    "Nyeri": {"lat": -0.4246, "lon": 36.9514},
    "Machakos": {"lat": -1.5177, "lon": 37.2634},
    "Kisii": {"lat": -0.6698, "lon": 34.7660},
    "Turkana": {"lat": 3.1167, "lon": 35.5667},
    "Wajir": {"lat": 1.7500, "lon": 40.0667},
    "Mandera": {"lat": 3.9167, "lon": 41.8500},
    "Kakamega": {"lat": 0.2833, "lon": 34.7500},
    "Bungoma": {"lat": 0.5667, "lon": 34.5500},
}


def render_sentinel_dashboard():
    """Render the complete SENTINEL Command Center dashboard."""
    if not HAS_STREAMLIT:
        print("Error: Streamlit required")
        return
    
    st.set_page_config(
        page_title="SENTINEL Command Center",
        page_icon="S",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Theme state
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True

    # Causal controls
    if "force_causal_retrain" not in st.session_state:
        st.session_state.force_causal_retrain = False
    if "auto_refresh_interval" not in st.session_state:
        st.session_state.auto_refresh_interval = 0

    st.sidebar.markdown("### Causal Settings")
    st.session_state.force_causal_retrain = st.sidebar.checkbox(
        "Force full causal retrain",
        value=st.session_state.force_causal_retrain,
    )
    interval_label = st.sidebar.selectbox(
        "Auto-refresh",
        options=["Off", "30s", "60s", "5m"],
        index=["Off", "30s", "60s", "5m"].index("Off") if st.session_state.auto_refresh_interval == 0 else
              ["Off", "30s", "60s", "5m"].index(
                  "30s" if st.session_state.auto_refresh_interval == 30 else
                  "60s" if st.session_state.auto_refresh_interval == 60 else
                  "5m"
              ),
    )
    interval_map = {"Off": 0, "30s": 30, "60s": 60, "5m": 300}
    st.session_state.auto_refresh_interval = interval_map[interval_label]
    
    # Analysis controls (merged from pulse/dashboard.py)
    _render_analysis_controls()
    
    theme = DARK_THEME if st.session_state.dark_mode else LIGHT_THEME
    st.markdown(generate_css(theme, st.session_state.dark_mode), unsafe_allow_html=True)
    
    # Pre-load card button CSS to prevent flash of unstyled content
    st.markdown(f"""<style>
    div[data-testid="stVerticalBlock"] button {{
        width: 100% !important;
        height: 320px !important;
        margin: 0 auto !important;
        white-space: pre-wrap !important;
        text-align: left !important;
        padding: 1.8rem !important;
        background: linear-gradient(160deg, rgba(31, 51, 29, 0.55), rgba(20, 38, 22, 0.45)) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-top: 1px solid rgba(0, 255, 136, 0.25) !important;
        border-left: 1px solid rgba(0, 255, 136, 0.15) !important;
        border-bottom: 1px solid rgba(0, 0, 0, 0.4) !important;
        border-right: 1px solid rgba(0, 0, 0, 0.3) !important;
        border-radius: 16px !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: flex-start !important;
        justify-content: flex-start !important;
        gap: 0.3rem !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35), 0 0 1px rgba(0, 255, 136, 0.1) !important;
    }}
    div[data-testid="stVerticalBlock"] button p:first-child {{
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        background: linear-gradient(90deg, {theme.text_primary}, {theme.accent_primary}) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
    }}
    div[data-testid="stVerticalBlock"] button p:not(:first-child) {{
        font-size: 0.88rem !important;
        color: {theme.text_secondary} !important;
        font-weight: 400 !important;
        line-height: 1.65 !important;
        opacity: 0.85 !important;
    }}
    div[data-testid="stVerticalBlock"] button:hover {{
        transform: translateY(-5px) !important;
        border-color: rgba(0, 255, 136, 0.35) !important;
        background: linear-gradient(160deg, rgba(31, 51, 29, 0.7), rgba(0, 255, 136, 0.06)) !important;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.4), 0 0 20px rgba(0, 255, 136, 0.12), inset 0 0 15px rgba(0, 255, 136, 0.03) !important;
    }}
    </style>""", unsafe_allow_html=True)
    
    # Auto-refresh if enabled.
    if st.session_state.auto_refresh_interval > 0:
        components.html(
            f"""
            <script>
                const interval = {st.session_state.auto_refresh_interval * 1000};
                setTimeout(() => {{
                    window.parent.postMessage({{isStreamlitMessage: true, type: 'streamlit:rerun'}}, '*');
                }}, interval);
            </script>
            """,
            height=0,
        )

    # Navigation State
    if "current_view" not in st.session_state:
        st.session_state.current_view = "HOME"

    # Global Sidebar Navigation (always visible, including HOME)
    st.sidebar.title("Navigation")
    NAV_OPTIONS = {
        "Home": "HOME",
        "Live Threat Map": "LIVE_MAP",
        "Executive Overview": "EXECUTIVE",
        "Signal Intelligence": "SIGNALS",
        "Causal Analysis": "CAUSAL",
        "K-SHIELD": "KSHIELD",
        "Simulation (Legacy)": "SIMULATION",
        "Escalation Pathways": "ESCALATION",
        "Federation": "FEDERATION",
        "Operations": "OPERATIONS",
        "System Guide": "SYSTEM_GUIDE",
        "Document Intelligence": "DOCS",
    }

    view_to_name = {v: k for k, v in NAV_OPTIONS.items()}
    current_name = view_to_name.get(st.session_state.current_view, "Home")

    # Sync the radio widget key so card-button navigations are respected
    # (Streamlit ignores `index` if the key already exists in session state)
    if st.session_state.get("sb_nav_radio") != current_name:
        st.session_state["sb_nav_radio"] = current_name

    selected_nav = st.sidebar.radio(
        "Go to",
        list(NAV_OPTIONS.keys()),
        key="sb_nav_radio",
    )
    st.session_state.current_view = NAV_OPTIONS[selected_nav]

    # Main Router
    view = st.session_state.current_view
    if view == "HOME":
        _render_home(theme)
        return

    # Only fetch data when needed (not for HOME page)
    data = get_dashboard_data(force_causal=st.session_state.force_causal_retrain)

    # Start document intelligence background job (once per process)
    try:
        from document_intel import get_document_intel
        get_document_intel().start()
    except Exception as exc:
        logger.warning(f"Document intelligence unavailable: {exc}")

    try:
        from flux_viz import get_flux_graph_html
    except ImportError:
        get_flux_graph_html = None
    _render_header(data, theme)

    if view == "LIVE_MAP":
        _render_live_map_tab(data, theme)
    elif view == "EXECUTIVE":
        _render_executive_tab(data, theme)
    elif view == "SIGNALS":
        _render_signals_tab(data, theme)
    elif view == "CAUSAL":
        _render_causal_tab(data, theme)
    elif view == "KSHIELD":
        try:
            from kshield.page import render as render_kshield
            render_kshield(theme, data)
        except Exception as exc:
            st.error(f"K-SHIELD module error: {exc}")
            import traceback
            st.code(traceback.format_exc())
    elif view == "SIMULATION":
        try:
            from whatif_workbench import render_whatif_workbench
            render_whatif_workbench(data, theme, get_flux_graph_html=get_flux_graph_html)
        except ImportError:
            _render_simulation_tab(data, theme)
    elif view == "ESCALATION":
        _render_escalation_tab(data, theme)
    elif view == "FEDERATION":
        _render_federation_tab(data, theme)
    elif view == "OPERATIONS":
        _render_operations_tab(data, theme)
    elif view == "SYSTEM_GUIDE":
        _render_system_guide_tab(theme)
    elif view == "DOCS":
        _render_document_intel_tab(theme)


def _render_header(data: DashboardData, theme):
    """Render dashboard header with WebGL animation."""
    try:
        from animated_header import get_animated_header_html
    except ImportError:
        try:
            from animated_header import get_animated_header_html
        except ImportError:
            get_animated_header_html = None

    # Render Animated Banner
    if get_animated_header_html:
        html_code = get_animated_header_html()
        components.html(html_code, height=120)
    else:
        # Fallback
        st.markdown(f"""
        <div class="sentinel-header">
            <h1>SENTINEL Command Center</h1>
            <p>Strategic National Economic & Threat Intelligence Layer</p>
        </div>
        """, unsafe_allow_html=True)

    # Metrics Row below banner
    level_info = THREAT_LEVELS.get(data.threat_level, THREAT_LEVELS["ELEVATED"])
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
    
    with col2:
        st.markdown(f"""
        <div class="glass-card-sm" style="text-align: center; padding: 10px;">
            <div class="live-label">THREAT LEVEL</div>
            <div style="color: {level_info['color']}; font-size: 1.2rem; font-weight: 700;">
                {level_info['label'].upper()}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="glass-card-sm" style="text-align: center; padding: 10px;">
            <div class="live-label">LAST UPDATE</div>
            <div class="mono" style="font-size: 1rem;">
                {datetime.now().strftime('%H:%M:%S')}
            </div>
        </div>
        """, unsafe_allow_html=True)


# Tab 1: Live Threat Map
def _render_live_map_tab(data: DashboardData, theme):
    """Kaspersky-inspired live threat map."""
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown('<div class="section-header">KENYA THREAT MAP</div>', unsafe_allow_html=True)
        _render_threat_globe(data, theme)
    
    with col2:
        st.markdown('<div class="section-header">LIVE THREATS</div>', unsafe_allow_html=True)
        _render_threat_counter(data, theme)
        
        st.markdown('<div class="section-header" style="margin-top: 1rem;">TOP COUNTIES</div>', unsafe_allow_html=True)
        _render_top_counties(data, theme)


def _render_threat_globe(data: DashboardData, theme):
    """3D interactive globe with threat indicators (Globe.gl)."""
    try:
        from globe_viz import get_globe_html
    except ImportError:
        st.error("Globe visualization component missing")
        return

    counties = data.counties
    
    # Prepare data for globe (pass as list of dicts)
    globe_data = []
    if counties:
        for name, item in counties.items():
            if isinstance(item, dict):
                risk = item.get('risk_score', 0.5)
            else:
                risk = int(item.risk_score * 100) / 100.0 if hasattr(item, 'risk_score') else 0.5
            
            globe_data.append({
                "name": name,
                "risk_score": risk
            })
        
    # Generate HTML
    html_code = get_globe_html(globe_data, height=500)
    
    # Render component
    components.html(html_code, height=500)




def _render_threat_counter(data: DashboardData, theme):
    """Live threat counter display."""
    total_signals = sum(s.count for s in data.signals) if data.signals else 1247
    
    st.markdown(f"""
    <div class="glass-card" style="text-align: center; margin-bottom: 1rem;">
        <div class="live-label">ACTIVE SIGNALS</div>
        <div class="live-counter">{total_signals:,}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Risk breakdown
    high_risk = len([c for c in data.counties.values() if hasattr(c, 'risk_score') and c.risk_score > 0.7])
    moderate_risk = len([c for c in data.counties.values() if hasattr(c, 'risk_score') and 0.4 < c.risk_score <= 0.7])
    
    st.markdown(f"""
    <div class="glass-card-sm">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span>Critical Counties</span>
            <span style="color: {theme.accent_critical}; font-weight: 700;">{high_risk}</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>Elevated Counties</span>
            <span style="color: {theme.accent_warning}; font-weight: 700;">{moderate_risk}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_top_counties(data: DashboardData, theme):
    """Top risk counties list."""
    sorted_counties = sorted(
        [(k, v) for k, v in data.counties.items() if hasattr(v, 'risk_score')],
        key=lambda x: x[1].risk_score,
        reverse=True
    )[:5]
    
    for name, county in sorted_counties:
        color = (
            theme.accent_critical if county.risk_score > 0.7 else
            theme.accent_danger if county.risk_score > 0.5 else
            theme.accent_warning
        )
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid {theme.border_subtle};">
            <span>{name}</span>
            <span style="color: {color}; font-family: 'Space Mono', monospace;">{county.risk_score:.0%}</span>
        </div>
        """, unsafe_allow_html=True)


# Tab 2: Executive Overview
def _render_executive_tab(data: DashboardData, theme):
    """Executive overview with traffic light, gauge, top threats."""
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        _render_traffic_light(data, theme)
    
    with col2:
        _render_escalation_gauge(data, theme)
    
    with col3:
        st.markdown('<div class="section-header">TOP THREATS</div>', unsafe_allow_html=True)
        _render_top_threats(data, theme)
    
    # Unknown-Unknown Detection (Phase 8)
    st.markdown("---")
    _render_unknown_detection(data, theme)
    
    # Competing Hypotheses (Phase 8)
    st.markdown('<div class="section-header">COMPETING HYPOTHESES</div>', unsafe_allow_html=True)
    _render_competing_hypotheses(data, theme)
    
    # Threat Index Gauges (merged from unified_dashboard.py)
    st.markdown("---")
    st.markdown('<div class="section-header">THREAT INDEX MATRIX</div>', unsafe_allow_html=True)
    _render_threat_index_gauges(data, theme)
    
    # Ethnic Tension Heatmap (merged from unified_dashboard.py)
    st.markdown('<div class="section-header">ETHNIC TENSION MATRIX</div>', unsafe_allow_html=True)
    _render_ethnic_tension_heatmap(data, theme)


def _render_traffic_light(data: DashboardData, theme):
    """Traffic light status indicator."""
    level_info = THREAT_LEVELS.get(data.threat_level, THREAT_LEVELS["ELEVATED"])
    
    st.markdown(f"""
    <div class="status-indicator">
        <div class="status-dot" style="background: {level_info['color']}; color: {level_info['color']};"></div>
        <div class="status-label" style="color: {level_info['color']};">{level_info['label']}</div>
        <div class="status-sublabel">National Threat Level</div>
    </div>
    """, unsafe_allow_html=True)


def _render_escalation_gauge(data: DashboardData, theme):
    """Time-to-escalation gauge."""
    if not HAS_PLOTLY:
        return
    
    hours = data.time_to_escalation
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=hours,
        number={'suffix': " hrs", 'font': {'size': 32, 'color': theme.text_primary}},
        title={'text': "Time to Potential Escalation", 'font': {'size': 12, 'color': theme.text_muted}},
        gauge={
            'axis': {'range': [0, 72], 'tickwidth': 1, 'tickcolor': theme.text_muted},
            'bar': {'color': theme.accent_primary},
            'bgcolor': theme.bg_tertiary,
            'steps': [
                {'range': [0, 12], 'color': 'rgba(255,0,68,0.2)'},
                {'range': [12, 24], 'color': 'rgba(255,107,53,0.2)'},
                {'range': [24, 48], 'color': 'rgba(255,204,0,0.2)'},
                {'range': [48, 72], 'color': 'rgba(0,255,136,0.2)'},
            ],
            'threshold': {
                'line': {'color': theme.accent_danger, 'width': 4},
                'thickness': 0.75,
                'value': 24,
            },
        },
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': theme.text_primary},
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_top_threats(data: DashboardData, theme):
    """Top threat cards."""
    theta_data = data.top_threats if hasattr(data, "top_threats") and data.top_threats else []
    
    if not theta_data:
        st.info("No top threats detected.")
        return

    cols = st.columns(3)
    for i, threat in enumerate(theta_data):
        level_info = THREAT_LEVELS.get(threat.get("level", "LOW"), THREAT_LEVELS["ELEVATED"])
        with cols[i]:
            st.markdown(f"""
            <div class="glass-card">
                <h3 style="font-size: 0.85rem; color: {theme.text_muted}; margin: 0 0 0.5rem 0;">{threat.get('title', 'Unknown')}</h3>
                <div style="font-size: 1.5rem; font-weight: 700; color: {level_info['color']};">{threat.get('level', 'LOW')}</div>
                <div style="color: {theme.accent_danger}; font-weight: 600;">{threat.get('change', '')}</div>
                <div style="font-size: 0.8rem; color: {theme.text_muted}; margin-top: 0.5rem;">{threat.get('desc', '')}</div>
            </div>
            """, unsafe_allow_html=True)


def _render_unknown_detection(data: DashboardData, theme):
    """Unknown-unknown detection alert (Phase 8)."""
    unknowns = getattr(data, "unknown_detections", [])
    
    if unknowns:
        lines = []
        for item in unknowns[:3]:
            if isinstance(item, dict):
                msg = item.get("message") or item.get("pattern") or item.get("title") or "Unprecedented signal pattern"
                conf = item.get("confidence")
                if isinstance(conf, (int, float)):
                    lines.append(f"{msg} (confidence {float(conf):.0%})")
                else:
                    lines.append(str(msg))
            else:
                lines.append(str(item))
        st.markdown(
            f"""
            <div class="alert alert-critical">
                <strong>UNPRECEDENTED PATTERN DETECTED</strong><br>
                {'<br>'.join(lines)}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No unknown-unknown patterns detected in current window.")


def _render_competing_hypotheses(data: DashboardData, theme):
    """Competing hypothesis framework (Phase 8)."""
    hypotheses = getattr(data, "competing_hypotheses", [])
    
    if not hypotheses:
        st.info("No competing hypotheses generated.")
        return
    
    cols = st.columns(3)
    for i, hyp in enumerate(hypotheses):
        with cols[i]:
            bar_width = hyp["probability"] * 100
            st.markdown(f"""
            <div class="glass-card-sm">
                <div style="font-weight: 600; margin-bottom: 0.5rem;">{hyp['name']}</div>
                <div style="background: {theme.bg_tertiary}; border-radius: 4px; height: 8px; margin-bottom: 0.5rem;">
                    <div style="background: {theme.accent_primary}; width: {bar_width}%; height: 100%; border-radius: 4px;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
                    <span style="color: {theme.text_muted};">Evidence: {hyp['evidence']}</span>
                    <span style="color: {theme.accent_primary};">{hyp['probability']:.0%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)


# Remaining tabs - continued in next part
def _render_signals_tab(data: DashboardData, theme):
    """Signal intelligence tab."""
    st.markdown('<div class="section-header">SIGNAL INTENSITIES (15 TYPES)</div>', unsafe_allow_html=True)
    
    if HAS_PLOTLY and data.signals:
        _render_signal_gauges(data, theme)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">SIGNAL CASCADE</div>', unsafe_allow_html=True)
        _render_signal_cascade(data, theme)
    with col2:
        st.markdown('<div class="section-header">CO-OCCURRENCE HEATMAP</div>', unsafe_allow_html=True)
        _render_cooccurrence_heatmap(data, theme)
    
    st.markdown("---")
    st.markdown('<div class="section-header">SIGNAL SILENCE DETECTOR</div>', unsafe_allow_html=True)
    _render_signal_silence(data, theme)
    
    # Risk Timeline (merged from pulse/dashboard.py)
    st.markdown("---")
    st.markdown('<div class="section-header">RISK SCORE TIMELINE</div>', unsafe_allow_html=True)
    _render_risk_timeline(data, theme)


def _render_signal_gauges(data: DashboardData, theme):
    """15 signal intensity gauges."""
    signals = data.signals if data.signals else []
    
    if not signals:
        st.info("No signal data available.")
        return
    
    fig = make_subplots(
        rows=3, cols=5,
        specs=[[{"type": "indicator"}] * 5] * 3,
        subplot_titles=[s.name[:15] for s in signals[:15]],
    )
    
    for i, signal in enumerate(signals[:15]):
        row = i // 5 + 1
        col = i % 5 + 1
        
        color = (
            theme.accent_success if signal.intensity < 0.4 else
            theme.accent_warning if signal.intensity < 0.6 else
            theme.accent_danger
        )
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=signal.intensity * 100,
                number={'suffix': "%", 'font': {'size': 12, 'color': theme.text_primary}},
                gauge={
                    'axis': {'range': [0, 100], 'visible': False},
                    'bar': {'color': color},
                    'bgcolor': theme.bg_tertiary,
                },
            ),
            row=row, col=col,
        )
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': theme.text_muted, 'size': 10},
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_signal_cascade(data: DashboardData, theme):
    """Signal cascade Sankey diagram derived from live signal/county data."""
    if not HAS_PLOTLY:
        return

    signals = sorted(data.signals or [], key=lambda s: float(getattr(s, "intensity", 0.0)), reverse=True)[:8]
    counties = sorted(
        [(name, c) for name, c in (data.counties or {}).items()],
        key=lambda kv: float(kv[1].risk_score if hasattr(kv[1], "risk_score") else kv[1].get("risk_score", 0.0)),
        reverse=True,
    )[:5]

    if not signals or not counties:
        st.info("Signal cascade unavailable: need both signal and county risk data.")
        return

    src_nodes = [s.name[:28] for s in signals]
    tgt_nodes = [name for name, _ in counties]
    labels = src_nodes + tgt_nodes

    county_weights = []
    for _, county in counties:
        risk = county.risk_score if hasattr(county, "risk_score") else county.get("risk_score", 0.0)
        county_weights.append(max(0.01, float(risk)))
    weight_sum = sum(county_weights) or 1.0

    src, tgt, val = [], [], []
    for i, sig in enumerate(signals):
        sig_strength = max(0.01, float(sig.intensity))
        for j, w in enumerate(county_weights):
            src.append(i)
            tgt.append(len(src_nodes) + j)
            val.append(sig_strength * (w / weight_sum))

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=labels,
                color=[theme.accent_primary] * len(src_nodes) + [theme.accent_warning] * len(tgt_nodes),
                pad=12,
                thickness=14,
                line=dict(color=theme.border_default, width=1),
            ),
            link=dict(
                source=src,
                target=tgt,
                value=val,
                color="rgba(0,255,136,0.22)",
            ),
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': theme.text_primary, 'size': 10},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_cooccurrence_heatmap(data: DashboardData, theme):
    """Signal co-occurrence heatmap."""
    if not HAS_PLOTLY:
        return
    
    matrix = getattr(data, "cooccurrence_matrix", None)
    
    if not matrix:
        st.info("Co-occurrence data unavailable.")
        return
        
    # Derive labels from actual signal names when available.
    import numpy as np
    z_data = np.array(matrix, dtype=float)
    if z_data.ndim != 2 or z_data.shape[0] == 0 or z_data.shape[1] == 0:
        st.info("Co-occurrence matrix is empty.")
        return

    n = min(z_data.shape[0], z_data.shape[1])
    z_data = z_data[:n, :n]

    signal_labels = [s.name for s in (data.signals or [])]
    labels = signal_labels[:n]
    if len(labels) < n:
        labels.extend([f"S{i+1}" for i in range(len(labels), n)])
    
    fig = go.Figure(go.Heatmap(
        z=z_data,
        x=labels,
        y=labels,
        colorscale=[[0, theme.bg_secondary], [0.5, theme.accent_warning], [1, theme.accent_danger]],
        zmin=0, zmax=1,
        showscale=True,
    ))
    
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': theme.text_muted, 'size': 9},
        xaxis={'tickangle': -45},
        yaxis={'autorange': 'reversed'} # Matrix convention
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_signal_silence(data: DashboardData, theme):
    """Signal silence detector."""
    silence_data = getattr(data, "silence_indicators", [])
    
    if not silence_data:
        st.info("No silence indicators detected.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if HAS_PLOTLY:
            df = pd.DataFrame(silence_data)
            colors = [theme.accent_danger if s > 0.7 else theme.accent_warning if s > 0.4 else theme.accent_success for s in df["Silence"]]
            
            fig = go.Figure(go.Bar(
                x=df["Region"],
                y=df["Silence"],
                marker_color=colors,
                text=[f"{s:.0%}" for s in df["Silence"]],
                textposition='outside',
            ))
            
            fig.add_hline(y=0.5, line_dash="dash", line_color=theme.accent_danger)
            
            fig.update_layout(
                height=250,
                margin=dict(l=40, r=40, t=20, b=40),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': theme.text_muted},
                yaxis_range=[0, 1],
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        df = pd.DataFrame(silence_data)
        if "Silence" in df.columns and "Region" in df.columns:
            top = df.sort_values("Silence", ascending=False).head(3)
            lines = []
            for _, row in top.iterrows():
                lines.append(f"{row['Region']}: {float(row['Silence']):.0%}")
            st.markdown(
                f"""
                <div class="alert alert-warning">
                    <strong>GOING DARK INDICATORS:</strong><br>
                    {'<br>'.join(lines)}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Silence diagnostics available, but schema is incomplete.")


def _render_causal_tab(data: DashboardData, theme):
    """Causal analysis tab — uses World Bank Kenya data directly."""
    from kshield.causal import render_causal
    render_causal(theme)


def _render_simulation_tab(data: DashboardData, theme):
    """
    Professional Scenario Platform Workspace.
    Layout: [Library (1)] [Builder (1)] [Run & Compare (2)]
    """
    st.markdown('<div class="section-header">SCENARIO PLATFORM (v2.0)</div>', unsafe_allow_html=True)
    
    # Initialize Session State
    if "active_scenario" not in st.session_state:
        st.session_state.active_scenario = {
            "name": "New Scenario",
            "shocks": [],
            "policies": [],
            "base_settings": {"steps": 50, "dt": 1.0}
        }
    
    # Connect
    from data_connector import SimulationConnector
    connector = SimulationConnector()
    connector.connect() # lightweight
    
    # 3-Pane Layout
    col_lib, col_build, col_run = st.columns([1, 1, 2])
    
    # --- Pane 1: Scenario Library ---
    with col_lib:
        st.markdown(f"**Library**")
        st.markdown(f"""
        <div style="background-color: {theme.bg_secondary}; padding: 1rem; border-radius: 8px; border: 1px solid {theme.border_default}; height: 100%;">
        """, unsafe_allow_html=True)
        
        if st.button("+ New Scenario", use_container_width=True):
            st.session_state.active_scenario = {
                "name": "New Scenario", 
                "shocks": [], 
                "policies": []
            }
            st.session_state.active_scenario_id = None
            st.rerun()
            
        st.markdown("---")
        scenarios = connector.list_scenarios()
        for s in scenarios:
            if st.button(f"📄 {s.get('name', 'Untitled')}", key=s.get('id'), use_container_width=True):
                loaded = connector.load_scenario(s.get('id'))
                if loaded:
                    st.session_state.active_scenario = loaded.to_dict()
                    st.session_state.active_scenario_id = loaded.id
                    st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Pane 2: Scenario Builder ---
    with col_build:
        st.markdown("**Builder**")
        st.markdown(f"""
        <div style="background-color: {theme.bg_secondary}; padding: 1rem; border-radius: 8px; border: 1px solid {theme.border_default}; height: 100%;">
        """, unsafe_allow_html=True)
        
        scen = st.session_state.active_scenario
        
        # Meta
        new_name = st.text_input("Name", scen.get("name", "New Scenario"))
        scen["name"] = new_name
        
        # Shocks
        st.markdown("##### Shocks")
        if st.button("Add Shock", key="add_shock"):
            scen.setdefault("shocks", []).append({
                "target": "demand_shock", "magnitude": 0.1, 
                "start_time": 5, "duration": 5, "shape": "ramp"
            })
            
        shocks = scen.get("shocks", [])
        for i, shock in enumerate(shocks):
            with st.expander(f"Shock {i+1}: {shock.get('target')}", expanded=True):
                shock["target"] = st.selectbox("Target", ["demand_shock", "supply_shock", "fiscal_shock", "fx_shock"], key=f"s_t_{i}", index=0)
                shock["magnitude"] = st.number_input("Mag", value=shock.get("magnitude", 0.0), key=f"s_m_{i}")
                shock["start_time"] = st.number_input("Start (t)", value=shock.get("start_time", 5), key=f"s_st_{i}")
                shock["duration"] = st.number_input("Duration", value=shock.get("duration", 5), key=f"s_d_{i}")
                shock["shape"] = st.selectbox("Shape", ["step", "ramp", "pulse", "decay"], key=f"s_sh_{i}", index=1)
                
                if st.button("Remove", key=f"rem_s_{i}"):
                    shocks.pop(i)
                    st.rerun()

        # Policies
        st.markdown("##### Policies (Constraints)")
        if st.button("Add Constraint", key="add_pol"):
            scen.setdefault("policies", []).append({
                "name": "Rate Cap", "key": "interest_rate", "max_value": 0.10
            })
            
        policies = scen.get("policies", [])
        for i, pol in enumerate(policies):
            with st.expander(f"Pol {i+1}: {pol.get('name')}"):
                pol["name"] = st.text_input("Name", pol.get("name"), key=f"p_n_{i}")
                pol["key"] = st.selectbox("Metric", ["interest_rate", "gdp_growth", "inflation", "unemployment"], key=f"p_k_{i}")
                pol["max_value"] = st.number_input("Max Cap", value=pol.get("max_value", 0.1), key=f"p_mx_{i}")
        
        # Save
        if st.button("💾 Save Scenario", use_container_width=True):
            saved_id = connector.save_scenario(scen)
            if saved_id:
                st.session_state.active_scenario_id = saved_id
                st.success("Saved!")
                
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Pane 3: Run & Result ---
    with col_run:
        st.markdown("**Simulation & Analysis**")
        
        if st.button("▶ RUN SCENARIO", type="primary", use_container_width=True):
            with st.spinner("Compiling and Running..."):
                try:
                    # Hydrate objects locally properly
                    from scarcity.simulation.scenario import Scenario
                    scen_obj = Scenario.from_dict(st.session_state.active_scenario)
                    result = connector.run_scenario_object(scen_obj)
                    st.session_state.sim_state = result
                except Exception as e:
                    st.error(f"Run Error: {e}")

        # Visualization
        sim_state = st.session_state.get("sim_state")
        if sim_state:
            # KPIS
            traj = sim_state.trajectory
            if traj:
                start = traj[0]["outcomes"]
                end = traj[-1]["outcomes"]
                c1, c2, c3 = st.columns(3)
                c1.metric("GDP Impact", f"{(end['gdp_growth']-start['gdp_growth'])*100:.2f}pp")
                c2.metric("Inflation Impact", f"{(end['inflation']-start['inflation'])*100:.2f}pp")
                c3.metric("Unemployment Impact", f"{(end['unemployment']-start['unemployment'])*100:.2f}pp")
            
            # 3D Plot
            col_flux, col_cube = st.columns(2)
            with col_flux:
                st.markdown("##### 3D Economic Flux Engine")
                try:
                    from flux_viz import get_flux_graph_html as _flux_fn
                except ImportError:
                    _flux_fn = None
                if _flux_fn:
                    # Pass the raw trajectory
                    html = _flux_fn(traj, height=400)
                    components.html(html, height=400)
                else:
                    st.info("Flux Viz module not loaded")
            
            with col_cube:
                st.markdown("##### 4D State Cube")
                _render_4d_simulation(sim_state, theme, selected_shock_key="demand_shock") # Default key for now
            
        st.markdown("---")
        st.markdown('<div class="section-header">POLICY IMPACT SENSITIVITY</div>', unsafe_allow_html=True)
        _render_policy_sensitivity(data, theme)

        st.markdown("---")
        _render_economic_terrain(data, theme)


def _render_4d_simulation(simulation_state, theme, selected_shock_key="demand_shock"):
    """
    Render 4D State Cube: 
    X=Shock, Y=Policy, Z=Outcome, Color=Time
    """
    if not HAS_PLOTLY or not simulation_state or not simulation_state.trajectory:
        st.info("No simulation data available. Run a scenario to visualize.")
        return
    
    trajectory = simulation_state.trajectory
    
    # Select Outcome Dimension
    outcome_key = st.selectbox(
        "Z-Axis Outcome",
        ["GDP Growth", "Inflation", "Unemployment"],
        index=0,
        key="viz_outcome_select"
    )
    outcome_map = {
        "GDP Growth": "gdp_growth",
        "Inflation": "inflation",
        "Unemployment": "unemployment"
    }
    z_key = outcome_map[outcome_key]
    
    # Extract Series from Frames
    try:
        t_vals = [f["t"] for f in trajectory]
        
        # X: Shock Magnitude usually
        x_vals = [f["shock_vector"].get(selected_shock_key, 0.0) for f in trajectory]
        
        # Y: Policy Response (Rate)
        y_vals = [f["policy_vector"].get("policy_rate", 0.0) * 100 for f in trajectory] # %
        
        # Z: Outcome
        z_vals = [f["outcomes"].get(z_key, 0.0) * 100 for f in trajectory] # %
        
    except KeyError as e:
        st.error(f"Data schema mismatch: Missing key {e}")
        return

    # Create 3D Plot
    fig = go.Figure()
    
    # Main Trajectory Trace
    fig.add_trace(go.Scatter3d(
        x=x_vals, y=y_vals, z=z_vals,
        mode='lines+markers',
        marker=dict(
            size=6,
            color=t_vals,
            colorscale='Viridis',
            colorbar=dict(title="Time (t)", thickness=10, x=0.9),
            symbol='circle'
        ),
        line=dict(
            color=theme.accent_primary,
            width=5
        ),
        name='Scenario Path',
        text=[f"t={t}<br>Shock={x:.2f}<br>Rate={y:.2f}%<br>{outcome_key}={z:.2f}%" 
              for t, x, y, z in zip(t_vals, x_vals, y_vals, z_vals)],
        hoverinfo='text'
    ))
    
    # Start Point
    fig.add_trace(go.Scatter3d(
        x=[x_vals[0]], y=[y_vals[0]], z=[z_vals[0]],
        mode='markers',
        marker=dict(size=10, color='white', symbol='diamond'),
        name='Start'
    ))
    
    # End Point
    fig.add_trace(go.Scatter3d(
        x=[x_vals[-1]], y=[y_vals[-1]], z=[z_vals[-1]],
        mode='markers',
        marker=dict(size=10, color=theme.accent_danger, symbol='x'),
        name='End'
    ))
    
    # Layout
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(text=f"State Cube: {selected_shock_key} → Policy → {outcome_key}", font=dict(color=theme.text_muted)),
        paper_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis=dict(title=f'{selected_shock_key} (Force)', backgroundcolor='rgba(0,0,0,0)', gridcolor=theme.border_default, color=theme.text_muted),
            yaxis=dict(title='Policy Rate (%)', backgroundcolor='rgba(0,0,0,0)', gridcolor=theme.border_default, color=theme.text_muted),
            zaxis=dict(title=f'{outcome_key} (%)', backgroundcolor='rgba(0,0,0,0)', gridcolor=theme.border_default, color=theme.text_muted),
            # camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.5))
        ),
        legend=dict(
            x=0, y=1,
            font=dict(color=theme.text_muted),
            bgcolor='rgba(0,0,0,0)'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_policy_sensitivity(data: DashboardData, theme):
    """Policy impact heatmap derived from simulation trajectory."""
    if not HAS_PLOTLY:
        return

    sim_state = st.session_state.get("sim_state") or data.simulation
    trajectory = getattr(sim_state, "trajectory", None)
    if not trajectory or len(trajectory) < 6:
        st.info("Policy sensitivity unavailable: run a scenario to generate trajectory data.")
        return

    policy_keys = sorted({k for frame in trajectory for k in frame.get("policy_vector", {}).keys()})
    outcome_keys = sorted({k for frame in trajectory for k in frame.get("outcomes", {}).keys()})
    if not policy_keys or not outcome_keys:
        st.info("Policy/outcome vectors missing from simulation frames.")
        return

    impacts = []
    for pol in policy_keys:
        p = np.array([float(f.get("policy_vector", {}).get(pol, 0.0)) for f in trajectory], dtype=float)
        row = []
        for out in outcome_keys:
            y = np.array([float(f.get("outcomes", {}).get(out, 0.0)) for f in trajectory], dtype=float)
            if len(p) < 3 or np.allclose(p.std(), 0.0) or np.allclose(y.std(), 0.0):
                row.append(0.0)
            else:
                row.append(float(np.corrcoef(p, y)[0, 1]))
        impacts.append(row)

    fig = go.Figure(go.Heatmap(
        z=impacts,
        x=[o.replace("_", " ").title() for o in outcome_keys],
        y=[p.replace("_", " ").title() for p in policy_keys],
        colorscale=[[0, theme.accent_danger], [0.5, '#ffffff'], [1, theme.accent_success]],
        zmin=-1,
        zmax=1,
        text=[[f"{v:+.2f}" for v in row] for row in impacts],
        texttemplate="%{text}",
        showscale=True,
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': theme.text_muted},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_economic_terrain(data: DashboardData, theme):
    """3D landscape of economic stability."""
    if not HAS_PLOTLY or not np:
        return
        
    # Create meshgrid for terrain
    x = np.linspace(0, 20, 50)  # Inflation
    y = np.linspace(0, 20, 50)  # Unemployment
    X, Y = np.meshgrid(x, y)
    
    # Potential function (Stability Bowl)
    # Target: 5% inflation, 5% unemployment
    # Z = Instability Potential (Height)
    Z = 0.5 * ((X - 5)**2 + (Y - 5)**2)
    
    fig = go.Figure()
    
    # 1. The Terrain Surface
    fig.add_trace(go.Surface(
        z=Z, x=X, y=Y,
        colorscale='Viridis_r', # Valleys are bright/green (stable), peaks dark/purple (unstable)
        opacity=0.8,
        showscale=False,
        name='Stability Landscape'
    ))
    
    # 2. Current State Marker
    if data.simulation:
        curr_infl = getattr(data.simulation, 'inflation', 0.0)
        curr_unemp = getattr(data.simulation, 'unemployment', 0.0)
    else:
        # Default to 0,0 or center if no data, but let's just not show the marker if no data
        return
    
    # Calculate Z for current state
    curr_z = 0.5 * ((curr_infl - 5)**2 + (curr_unemp - 5)**2)
    
    fig.add_trace(go.Scatter3d(
        x=[curr_infl],
        y=[curr_unemp],
        z=[curr_z + 20], # Float well above surface
        mode='markers+text',
        marker=dict(
            size=8,
            color=theme.accent_danger,
            line=dict(width=2, color='white')
        ),
        text=[f"CURRENT<br>Infl: {curr_infl:.1f}%<br>Unemp: {curr_unemp:.1f}%"],
        textposition="top center",
        name='Current State'
    ))
    
    # Drop line
    fig.add_trace(go.Scatter3d(
        x=[curr_infl, curr_infl],
        y=[curr_unemp, curr_unemp],
        z=[curr_z, curr_z + 20],
        mode='lines',
        line=dict(width=2, color=theme.accent_danger, dash='dash'),
        showlegend=False
    ))
    
    # 3. Equilibrium Marker
    fig.add_trace(go.Scatter3d(
        x=[5], y=[5], z=[0],
        mode='markers',
        marker=dict(size=6, color=theme.accent_success, opacity=0.8),
        name='Equilibrium Target'
    ))
    
    fig.update_layout(
        title=dict(text="Stability Phase Space (Valley = Optimal)", font=dict(color=theme.text_muted, size=12)),
        scene=dict(
            xaxis=dict(title='Inflation (%)', backgroundcolor='rgba(0,0,0,0)', gridcolor=theme.border_default, color=theme.text_muted),
            yaxis=dict(title='Unemployment (%)', backgroundcolor='rgba(0,0,0,0)', gridcolor=theme.border_default, color=theme.text_muted),
            zaxis=dict(title='Instability Potential', backgroundcolor='rgba(0,0,0,0)', gridcolor=theme.border_default, color=theme.text_muted),
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.8))
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_escalation_tab(data: DashboardData, theme):
    """Escalation pathways tab (Phase 7)."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-header">ESCALATION PATHWAY TREE</div>', unsafe_allow_html=True)
        _render_escalation_tree(theme)
    
    with col2:
        st.markdown('<div class="section-header">DECISION LATENCY</div>', unsafe_allow_html=True)
        _render_decision_countdown(data, theme)
        
        st.markdown('<div class="section-header" style="margin-top: 1rem;">FRAGILITY INDEX</div>', unsafe_allow_html=True)
        _render_fragility_gauge(theme)


def _render_escalation_tree(theme):
    """Escalation pathway visualization."""
    st.info("Escalation pathway data unavailable.")


def _render_decision_countdown(data: DashboardData, theme):
    """Decision latency countdown."""
    hours = data.time_to_escalation
    
    st.markdown(f"""
    <div class="glass-card" style="text-align: center;">
        <div class="live-label">TIME TO DECISION POINT</div>
        <div class="live-counter" style="color: {theme.accent_warning if hours < 24 else theme.accent_primary};">
            {hours:.0f}h
        </div>
        <div style="color: {theme.text_muted}; font-size: 0.85rem; margin-top: 0.5rem;">
            {'URGENT' if hours < 24 else 'MONITORING'}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_fragility_gauge(theme):
    """Scenario fragility index."""
    if not HAS_PLOTLY:
        return
    
    # Real fragility index calculation needed
    st.info("Fragility index unavailable.")
    return
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=fragility * 100,
        number={'suffix': "%", 'font': {'size': 24, 'color': theme.text_primary}},
        title={'text': "System Brittleness", 'font': {'size': 11, 'color': theme.text_muted}},
        gauge={
            'axis': {'range': [0, 100], 'visible': False},
            'bar': {'color': theme.accent_warning},
            'bgcolor': theme.bg_tertiary,
        },
    ))
    
    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_federation_tab(data: DashboardData, theme):
    """Federation tab."""
    st.markdown('<div class="section-header">AGENCY STATUS</div>', unsafe_allow_html=True)
    
    agencies = getattr(data, "agencies", [])
    
    if not agencies:
        st.info("No agency status reported.")
        st.markdown("---") # Keep layout clean for siblings below
        col1, col2 = st.columns(2)
        with col1:
             st.markdown('<div class="section-header">CONTRIBUTION TIMELINE</div>', unsafe_allow_html=True)
             _render_contribution_timeline(theme)
        with col2:
             st.markdown('<div class="section-header">MODEL CONVERGENCE</div>', unsafe_allow_html=True)
             _render_convergence_chart(theme)
        return
    
    cols = st.columns(5)
    for i, agency in enumerate(agencies):
        with cols[i]:
            status_color = theme.accent_success if agency.status == "active" else theme.accent_warning
            active_class = "active" if agency.status == "active" else ""
            st.markdown(f"""
            <div class="agency-card {active_class}">
                <div class="name">{agency.name}</div>
                <div class="status" style="color: {status_color};">{agency.status.upper()}</div>
                <div style="margin-top: 0.5rem; font-size: 0.8rem;">
                    Contribution: {agency.contribution_score:.0%}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-header">CONTRIBUTION TIMELINE</div>', unsafe_allow_html=True)
        _render_contribution_timeline(theme)
    
    with col2:
        st.markdown('<div class="section-header">MODEL CONVERGENCE</div>', unsafe_allow_html=True)
        _render_convergence_chart(theme)


def _render_contribution_timeline(theme):
    """Agency contribution over time."""
    if not HAS_PLOTLY:
        return
    
    # Real timeline data needed
    st.info("Timeline data unavailable.")
    return
    
    df = pd.DataFrame(data)
    fig = px.line(df, x="Round", y="Contribution", color="Agency")
    
    fig.update_layout(
        height=300,
        margin=dict(l=40, r=40, t=20, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': theme.text_muted},
        yaxis_range=[0, 1],
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_convergence_chart(theme):
    """Model convergence chart."""
    if not HAS_PLOTLY:
        return
    
    # Real convergence data needed
    st.info("Convergence data unavailable.")
    return
    
    fig = go.Figure(go.Scatter(
        x=rounds, y=loss,
        mode='lines+markers',
        fill='tozeroy',
        line=dict(color=theme.accent_primary),
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=40, r=40, t=20, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Federation Round",
        yaxis_title="Model Loss",
        font={'color': theme.text_muted},
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_operations_tab(data: DashboardData, theme):
    """Operations tab."""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="section-header">COUNTY RISK ASSESSMENT</div>', unsafe_allow_html=True)
        _render_county_table(data, theme)
    
    with col2:
        st.markdown('<div class="section-header">RECENT ALERTS</div>', unsafe_allow_html=True)
        _render_alerts(theme)
        
        st.markdown('<div class="section-header" style="margin-top: 1rem;">INFRASTRUCTURE HEALTH</div>', unsafe_allow_html=True)
        _render_infrastructure_health(theme)
    
    # Merged sections from unified_dashboard.py
    st.markdown("---")
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown('<div class="section-header">NETWORK ANALYSIS</div>', unsafe_allow_html=True)
        _render_network_analysis(data, theme)
    
    with col4:
        st.markdown('<div class="section-header">ECONOMIC SATISFACTION INDEX</div>', unsafe_allow_html=True)
        _render_economic_indicators(data, theme)
    
    # System Primitives (merged from pulse/dashboard.py)
    st.markdown("---")
    st.markdown('<div class="section-header">SYSTEM PRIMITIVES</div>', unsafe_allow_html=True)
    _render_system_primitives(data, theme)


def _render_county_table(data: DashboardData, theme):
    """County risk table."""
    if not HAS_PANDAS:
        return
    
    if not hasattr(data, "counties") or not data.counties:
        st.info("No county data available.")
        return
    
    table_data = []
    # If using real data, iterate over data.counties
    for name, county in data.counties.items():
        # Handle dict or obj
        risk = county.get('risk_score', 0) if isinstance(county, dict) else getattr(county, 'risk_score', 0)
        level = "Critical" if risk > 0.7 else "High" if risk > 0.5 else "Moderate" if risk > 0.3 else "Low"
        # Since we don't have signal counts in the county obj yet commonly, use placeholder or 0
        signals_cnt = len(county.get('top_signals', [])) if isinstance(county, dict) else len(getattr(county, 'top_signals', []))
        
        table_data.append({
            "County": name,
            "Risk Score": f"{risk:.0%}",
            "Level": level,
            "Signals": signals_cnt,
        })
    
    df = pd.DataFrame(table_data)
    df = df.sort_values("Risk Score", ascending=False)
    
    st.dataframe(df, use_container_width=True, height=350)


def _render_alerts(theme):
    """Recent alerts with styled priority banners."""
    # Pull alerts from session state if available
    alerts = []
    if "sentinel_alerts" in st.session_state:
        alerts = st.session_state.sentinel_alerts
    
    if not alerts:
        st.info("No active alerts.")
        return
    
    for alert in alerts:
        alert_type = alert.get('type', 'warning')
        if alert_type == 'critical':
            bg, border, color = '#1a0a0a', theme.accent_danger, '#ff6b6b'
        else:
            bg, border, color = '#1a1a0a', theme.accent_warning, '#ffc107'
        
        st.markdown(f"""
        <div style="background: {bg}; border-left: 4px solid {border}; padding: 0.8rem 1rem;
                     border-radius: 4px; margin-bottom: 0.5rem;">
            <div style="color: {color}; font-weight: 600; font-size: 0.85rem;">{alert.get('time', '')}</div>
            <div style="color: {theme.text_primary}; font-size: 0.8rem; margin-top: 0.25rem;">{alert.get('msg', '')}</div>
        </div>
        """, unsafe_allow_html=True)


def _render_infrastructure_health(theme):
    """Infrastructure health monitor (Phase 6)."""
    # Real infrastructure metrics needed
    metrics = []
    if not metrics:
        st.info("Infrastructure health data unavailable.")
        return
    
    for label, value, color in metrics:
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid {theme.border_subtle};">
            <span>{label}</span>
            <span style="color: {color}; font-weight: 600;">{value}</span>
        </div>
        """, unsafe_allow_html=True)


def _render_system_guide_tab(theme):
    """Comprehensive in-app guide for all dashboard modules."""
    st.markdown('<div class="section-header">SYSTEM GUIDE &mdash; COMPLETE WALKTHROUGH</div>', unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:{theme.text_muted}; font-size:0.85rem;">'
        "This guide explains what each part of SENTINEL does, why it exists, "
        "what data it uses, and how to interpret outputs safely.</p>",
        unsafe_allow_html=True,
    )

    sections = [
        "1) System Purpose",
        "2) Data Pipeline",
        "3) Live Threat Map",
        "4) Executive Overview",
        "5) Signal Intelligence",
        "6) Causal Analysis",
        "7) Simulation",
        "8) Escalation Pathways",
        "9) Federation",
        "10) Operations",
        "11) Document Intelligence",
        "12) Reliability & Failure Modes",
        "13) Quick Glossary",
    ]
    selected = st.radio("Guide Sections", sections, key="system_guide_section")

    if selected == "1) System Purpose":
        st.markdown(
            """
            SENTINEL is a decision-support platform for economic risk and social instability monitoring.

            Core job:
            1. Observe incoming signals.
            2. Estimate risk and likely causal paths.
            3. Simulate policy scenarios.
            4. Surface operational alerts and decision windows.

            It is not a single prediction model. It is a multi-module system combining
            monitoring, causal inference, simulation, and coordination views.
            """
        )
    elif selected == "2) Data Pipeline":
        st.markdown(
            """
            Data flow is connector-based:

            1. `PulseConnector`: live signal ingestion and threat metrics.
            2. `ScarcityConnector`: hypothesis discovery and Granger-style causal candidates.
            3. `SimulationConnector`: scenario execution and trajectory outputs.
            4. `FederationConnector`: multi-agency status and contribution snapshots.
            5. `Document Intelligence`: news and local dossier content retrieval.

            Then `get_dashboard_data(...)` aggregates these into one `DashboardData` object,
            which all tabs consume.
            """
        )
    elif selected == "3) Live Threat Map":
        st.markdown(
            """
            What you see:
            - County-level risk markers.
            - Active signal counters.
            - Top-risk counties summary.

            Why it exists:
            - Gives immediate geographic situational awareness.
            - Answers: where is risk concentrating now?

            How to read:
            - Higher county risk score means greater current stress concentration.
            - Counter trends matter more than a single snapshot.
            """
        )
    elif selected == "4) Executive Overview":
        st.markdown(
            """
            What you see:
            - National threat status indicator.
            - Time-to-escalation gauge.
            - Top threat cards.
            - Competing hypothesis panel.

            Why it exists:
            - Fast decision brief for leadership.

            How to read:
            - Treat this as summary, then drill into Signals/Causal/Simulation tabs.
            - High-level status should be validated by underlying evidence panels.
            """
        )
    elif selected == "5) Signal Intelligence":
        st.markdown(
            """
            What you see:
            - Multi-signal intensity gauges.
            - Signal cascade map.
            - Co-occurrence heatmap.
            - Silence detector and risk timeline.

            Why it exists:
            - Detect early shifts in narrative and stress patterns.

            Key interpretation:
            - Co-occurrence shows which signals rise together.
            - Silence/going-dark may indicate migration to less visible channels.
            - Timeline helps separate temporary spikes from persistent trend changes.
            """
        )
    elif selected == "6) Causal Analysis":
        st.markdown(
            """
            Two layers are used:

            1. Granger layer:
            - Tests whether past X helps predict future Y.
            - Useful for directional early-warning relationships.
            - Not absolute proof of real-world cause.

            2. Structural estimation layer (Scarcity):
            - Runs effect types (estimands) like ATE/ATT/ATC/CATE/LATE/mediation.
            - Produces effect direction, magnitude, confidence intervals, and agreement diagnostics.

            Why estimands can fail:
            - Missing required assumptions (instrument for LATE, mediator for mediation).
            - Missing dependencies (for example econml for CATE/ITE).
            - Runtime worker issues in parallel mode.

            Fallback policy:
            - User policy: continue or fail-fast per estimand errors.
            - Runtime fallback: automatic retry in sequential mode if parallel workers fail.
            """
        )
    elif selected == "7) Simulation":
        st.markdown(
            """
            What you see:
            - Scenario builder (shocks and policy constraints).
            - Run output trajectory.
            - 3D/4D path visualization.
            - Policy sensitivity view.

            Why it exists:
            - Answer "what if we apply this shock/policy?" before real-world action.

            How to read:
            - Compare start vs end outcome deltas.
            - Look at trajectory shape, not just final point.
            - Use sensitivity matrix to see which policy levers move which outcomes most.
            """
        )
    elif selected == "8) Escalation Pathways":
        st.markdown(
            """
            What you see:
            - Decision-latency countdown.
            - Fragility/escalation placeholders or computed metrics when available.

            Why it exists:
            - Prioritize response timing under uncertainty.

            How to read:
            - Shorter time-to-decision means less room for delayed intervention.
            - Escalation logic is strongest when corroborated by Signals + Causal + Simulation.
            """
        )
    elif selected == "9) Federation":
        st.markdown(
            """
            What you see:
            - Agency participation and contribution snapshots.
            - Convergence/timeline panels when available.

            Why it exists:
            - Shows multi-agency coordination health.

            How to read:
            - Contribution imbalance can indicate coordination risk.
            - Active status is not enough; contribution quality and timeliness matter.
            """
        )
    elif selected == "10) Operations":
        st.markdown(
            """
            What you see:
            - County risk table.
            - Alerts feed.
            - Network analysis and economic satisfaction panels.

            Why it exists:
            - Converts analytics into actionable operational queue.

            How to read:
            - Prioritize counties by risk and trend.
            - Use alert severity + recency together.
            - Cross-check network and satisfaction metrics for intervention design.
            """
        )
    elif selected == "11) Document Intelligence":
        st.markdown(
            """
            What you see:
            - Live news stream by category.
            - Local dossier browser with extracted documents.

            Why it exists:
            - Provides evidence context and narrative validation for quantitative signals.

            How to read:
            - Use source credibility and recency.
            - Link narrative shifts to signal and causal changes.
            """
        )
    elif selected == "12) Reliability & Failure Modes":
        st.markdown(
            """
            Common failure patterns:
            - Missing optional libraries (causal/plot dependencies).
            - Sparse or misaligned time-series causing weak causal validity.
            - Parallel worker failures in constrained runtimes.
            - Demo fallback data when upstream connectors are unavailable.

            What the system does:
            - Surfaces explicit warnings and stage-level errors.
            - Skips invalid effect types with reasons.
            - Falls back to safer execution mode when possible.

            Analyst rule:
            - Do not trust one chart in isolation.
            - Require consistency across at least Signals + Causal + Simulation.
            """
        )
    elif selected == "13) Quick Glossary":
        st.markdown(
            """
            - `Signal`: measurable indicator extracted from incoming data.
            - `Hypothesis`: candidate relationship discovered by the engine.
            - `Granger causality`: predictive direction test using lagged values.
            - `Estimand`: the exact effect question being estimated.
            - `Heatmap`: color grid where color intensity encodes value strength.
            - `Confounder`: variable that influences both cause and outcome.
            - `Instrument`: proxy variable used for LATE identification.
            - `Mediator`: variable on the path between cause and outcome.
            - `CI (confidence interval)`: plausible effect range estimate.
            - `Fallback`: automatic safer mode used when preferred execution fails.
            """
        )


def _render_document_intel_tab(theme):
    """Render Intelligence: Live News + PDF Dossiers."""
    try:
        from document_intel import get_document_intel
        intel_data = get_document_intel().get_snapshot()
    except Exception as e:
        st.error(f"Intelligence System Offline: {e}")
        return

    # Sub-navigation
    cols = st.columns([1, 1, 4])
    with cols[0]:
        view_mode = st.radio("Source", ["Live News", "Local Dossiers"], label_visibility="collapsed")
    
    st.markdown("---")

    # ==========================================
    # VIEW: LIVE NEWS (NewsAPI)
    # ==========================================
    if view_mode == "Live News":
        news_data = intel_data.get("news", {})
        
        if not news_data:
            st.info("No news data available. Check API quota or connectivity.")
            return

        # Categories
        categories = sorted(list(news_data.keys()))
        selected_cat = st.selectbox("Category", [c.upper() for c in categories])
        
        if selected_cat:
            cat_key = selected_cat.lower()
            articles = news_data.get(cat_key, [])
            
            st.markdown(f"### {selected_cat} ({len(articles)} Articles)")
            
            if not articles:
                st.info(f"No recent articles in {selected_cat}.")
            else:
                for art in articles:
                     # Calculate time ago
                    pub = art.get('published_at', '')
                    time_label = pub
                    try:
                        dt = datetime.fromisoformat(pub.replace('Z', '+00:00'))
                        now = datetime.now(dt.tzinfo)
                        diff = now - dt
                        if diff.days > 0:
                            time_label = f"{diff.days}d ago"
                        elif diff.seconds > 3600:
                            time_label = f"{diff.seconds // 3600}h ago"
                        else:
                            time_label = f"{diff.seconds // 60}m ago"
                    except:
                        pass

                    st.markdown(f"""
                    <div class="glass-card" style="margin-bottom: 0.8rem; padding: 1rem;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: {theme.accent_primary}; font-size: 0.8rem; font-weight: 700;">
                                {art.get('source', 'Unknown').upper()}
                            </span>
                            <span style="color: {theme.text_muted}; font-size: 0.8rem;">
                                {time_label}
                            </span>
                        </div>
                        <div style="font-size: 1.1rem; font-weight: 600; margin: 0.5rem 0;">
                            <a href="{art.get('url')}" target="_blank" style="text-decoration: none; color: {theme.text_primary};">
                                {art.get('title')}
                            </a>
                        </div>
                        <div style="font-size: 0.9rem; color: {theme.text_secondary};">
                            {art.get('description') or ''}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
    # ==========================================
    # VIEW: LOCAL DOSSIERS (PDFs)
    # ==========================================
    elif view_mode == "Local Dossiers":
        # Session State Parsing
        if "dossier_nav" not in st.session_state:
            st.session_state.dossier_nav = {"view": "themes", "category": None, "file": None}

        base_dir = Path("random/content_extracted")
        if not base_dir.exists():
            st.info("No extracted dossiers found in random/content_extracted.")
            return

        # HEADER
        nav = st.session_state.dossier_nav
        header_text = "LOCAL DOSSIERS"
        if nav["view"] == "list":
            header_text += f" / {nav['category'].replace('_', ' ').upper()}"
        elif nav["view"] == "content":
            header_text += f" / {nav['category'].replace('_', ' ').upper()} / READING"
            
        st.markdown(f'<div class="section-header">{header_text}</div>', unsafe_allow_html=True)

        # ---------------------------
        # VIEW 1: THEMES (Categories)
        # ---------------------------
        if nav["view"] == "themes":
            categories = sorted([d.name for d in base_dir.iterdir() if d.is_dir() and d.name != "Uncategorized"])
            
            if not categories:
                st.info("No dossier themes available.")
                return
                
            cols = st.columns(3)
            for idx, category in enumerate(categories):
                cat_name = category.replace("_", " ").upper()
                
                # Count files
                cat_dir = base_dir / category
                file_count = len(list(cat_dir.glob("*.md")))
                
                with cols[idx % 3]:
                    # Use button AS the card
                    label = f"{cat_name}\n\n{file_count} Dossiers"
                    if st.button(label, key=f"cat_{category}", use_container_width=True):
                        st.session_state.dossier_nav = {"view": "list", "category": category, "file": None}
                        st.rerun()

        # ---------------------------
        # VIEW 2: ARTICLE LIST
        # ---------------------------
        elif nav["view"] == "list":
            if st.button("← Back to Themes", key="back_to_themes"):
                st.session_state.dossier_nav = {"view": "themes", "category": None, "file": None}
                st.rerun()

            category = nav["category"]
            st.markdown(f'<div style="margin-bottom: 1rem; color: {theme.accent_info}; font-weight: 600;">THEME: {category.replace("_", " ").upper()}</div>', unsafe_allow_html=True)

            cat_dir = base_dir / category
            files = list(cat_dir.glob("*.md"))
            
            if not files:
                st.info("No articles found in this theme.")
                return
                
            # SORT BY DATE (Modification Time) - Newest First
            files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            for file_path in files:
                file_name = file_path.stem.replace("_", " ").title()
                file_date = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                file_size = f"{file_path.stat().st_size / 1024:.1f} KB"
                
                # Extract summary (first 200 chars, skipping headers #)
                summary = ""
                try:
                    # Read slightly more to ensure we get past headers
                    raw_text = file_path.read_text(encoding="utf-8")[:1000]
                    lines = [l.strip() for l in raw_text.split('\n') if l.strip() and not l.strip().startswith("#") and not l.strip().startswith("|")]
                    if lines:
                        summary = " ".join(lines)[:250] + "..."
                except Exception:
                    summary = "No preview available."

                # Clickable Receipt/Card Row
                label = f"{file_name}\n\n{summary}\n\n{file_date} | {file_size}"
                if st.button(label, key=f"read_{file_path.name}", use_container_width=True):
                    st.session_state.dossier_nav = {"view": "content", "category": category, "file": str(file_path)}
                    st.rerun()

        # ---------------------------
        # VIEW 3: ARTICLE CONTENT
        # ---------------------------
        elif nav["view"] == "content":
            file_path = Path(nav["file"])
            
            col_back, col_title = st.columns([1, 5])
            with col_back:
                if st.button("← Back", key="back_to_list"):
                    st.session_state.dossier_nav["view"] = "list"
                    st.session_state.dossier_nav["file"] = None
                    st.rerun()
            
            try:
                content = file_path.read_text(encoding="utf-8")
                st.markdown(f"""
                <div class="glass-card" style="padding: 2rem; border: 1px solid {theme.accent_primary}; margin-top: 1rem;">
                <div style="font-family: 'Courier New', monospace; color: {theme.accent_primary}; font-size: 0.8rem; margin-bottom: 1rem;">
                    SOURCE: {file_path.name}
                </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(content, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error reading file: {e}")

# =============================================================================
# Merged Render Functions (from unified_dashboard.py + dashboard.py)
# =============================================================================

def _render_threat_index_gauges(data: DashboardData, theme):
    """Render 2×4 grid of all 8 threat index gauges."""
    if not HAS_PLOTLY:
        return
    
    indices = data.threat_indices
    if not indices:
        st.info("Threat index data unavailable.")
        return
    
    SEVERITY_COLORS = {
        "LOW": "#00ff88", "GUARDED": "#7ed957", "MODERATE": "#ffc107",
        "ELEVATED": "#ff9800", "HIGH": "#ff5722", "CRITICAL": "#ff0044",
    }
    
    fig = make_subplots(
        rows=2, cols=4,
        specs=[[{"type": "indicator"}] * 4] * 2,
        subplot_titles=[idx["name"] for idx in indices[:8]],
    )
    
    for i, idx in enumerate(indices[:8]):
        row = i // 4 + 1
        col = i % 4 + 1
        color = SEVERITY_COLORS.get(idx.get("severity", "MODERATE"), "#ffc107")
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=idx["value"] * 100,
                number={'suffix': "%", 'font': {'size': 14, 'color': theme.text_primary}},
                gauge={
                    'axis': {'range': [0, 100], 'visible': False},
                    'bar': {'color': color},
                    'bgcolor': theme.bg_tertiary,
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(0,255,136,0.1)'},
                        {'range': [30, 60], 'color': 'rgba(255,204,0,0.1)'},
                        {'range': [60, 100], 'color': 'rgba(255,0,68,0.1)'},
                    ],
                },
            ),
            row=row, col=col,
        )
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': theme.text_muted, 'size': 10},
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_ethnic_tension_heatmap(data: DashboardData, theme):
    """Render inter-group ethnic tension heatmap."""
    if not HAS_PLOTLY:
        return
    
    et = data.ethnic_tensions
    tensions = et.get("tensions", {})
    highest = et.get("highest_pair")
    
    if not tensions:
        st.info("Ethnic tension data unavailable.")
        return
    
    if highest:
        st.warning(f"Highest tension: **{highest[0]}** ↔ **{highest[1]}**")
    
    # Build symmetric matrix
    groups = set()
    for key in tensions:
        g1, g2 = key.split("-")
        groups.add(g1)
        groups.add(g2)
    groups = sorted(list(groups))
    
    n = len(groups)
    matrix = [[0.0] * n for _ in range(n)]
    for key, val in tensions.items():
        g1, g2 = key.split("-")
        if g1 in groups and g2 in groups:
            i, j = groups.index(g1), groups.index(g2)
            matrix[i][j] = val
            matrix[j][i] = val
    
    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=groups,
        y=groups,
        colorscale=[[0, theme.bg_secondary], [0.5, theme.accent_warning], [1, theme.accent_danger]],
        zmin=0, zmax=1,
        showscale=True,
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': theme.text_muted, 'size': 10},
        yaxis={'autorange': 'reversed'},
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_risk_timeline(data: DashboardData, theme):
    """Render risk score timeline chart."""
    history = data.risk_history
    
    if not history or len(history) < 2:
        st.info("Collecting data for risk timeline... (need at least 2 data points)")
        return
    
    if HAS_PANDAS:
        df = pd.DataFrame(history)
        if 'timestamp' in df.columns:
            df['time'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('time')
        
        cols_to_plot = [c for c in ['overall_risk', 'peak_risk'] if c in df.columns]
        if cols_to_plot:
            st.line_chart(df[cols_to_plot])
        
        if 'signal_count' in df.columns:
            st.bar_chart(df[['signal_count']])


def _render_network_analysis(data: DashboardData, theme):
    """Render actor role distribution pie chart."""
    if not HAS_PLOTLY:
        return
    
    network = data.network_analysis
    roles = network.get("roles", {})
    
    if not roles:
        st.info("Network analysis data unavailable.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = go.Figure(go.Pie(
            labels=list(roles.keys()),
            values=list(roles.values()),
            hole=0.4,
            marker=dict(colors=[
                theme.accent_danger, theme.accent_warning, theme.accent_primary,
                '#9b59b6', theme.accent_success, theme.text_muted,
            ]),
        ))
        fig.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': theme.text_primary, 'size': 11},
            legend={'font': {'size': 10}},
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.metric("Nodes", network.get("node_count", 0))
        st.metric("Edges", network.get("edge_count", 0))
        st.metric("Communities", network.get("community_count", 0))


def _render_economic_indicators(data: DashboardData, theme):
    """Render Economic Satisfaction Index by domain."""
    if not HAS_PLOTLY:
        return
    
    esi = data.esi_indicators
    if not esi:
        st.info("ESI data unavailable.")
        return
    
    colors = [theme.accent_success if v > 0.5 else theme.accent_danger for v in esi.values()]
    
    fig = go.Figure(go.Bar(
        x=list(esi.keys()),
        y=[v * 100 for v in esi.values()],
        marker_color=colors,
        text=[f"{v:.0%}" for v in esi.values()],
        textposition='outside',
        textfont={'color': theme.text_primary},
    ))
    
    fig.add_hline(y=50, line_dash="dash", line_color=theme.text_muted, annotation_text="Threshold")
    
    fig.update_layout(
        height=250,
        yaxis_title="Satisfaction %",
        yaxis_range=[0, 100],
        margin=dict(l=40, r=10, t=10, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': theme.text_muted},
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _render_system_primitives(data: DashboardData, theme):
    """Render system primitives — scarcity, stress, bonds, risk metrics."""
    prims = data.primitives
    if not prims:
        st.info("System primitives unavailable.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**Scarcity Vector** (Aggregate: {prims.get('aggregate_scarcity', 0):.0%})")
        for domain, value in prims.get("scarcity", {}).items():
            color = theme.accent_danger if value > 0.6 else theme.accent_warning if value > 0.3 else theme.accent_success
            st.markdown(f"""
            <div style="margin-bottom: 0.4rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:{theme.text_muted};">
                    <span>{domain.title()}</span>
                    <span style="color:{color}; font-weight:600;">{value:.0%}</span>
                </div>
                <div style="background:{theme.bg_tertiary}; border-radius:4px; height:6px; overflow:hidden;">
                    <div style="background:{color}; width:{value*100}%; height:100%; border-radius:4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"**Actor Stress** (Total: {prims.get('total_stress', 0):.2f})")
        for actor, stress in prims.get("stress", {}).items():
            indicator = "↓" if stress < 0 else "↑" if stress > 0 else "─"
            color = theme.accent_danger if stress > 0.3 else theme.accent_warning if stress > 0 else theme.accent_success
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:0.3rem 0; border-bottom:1px solid {theme.border_subtle};">
                <span style="color:{theme.text_muted}; font-size:0.85rem;">{actor.title()}</span>
                <span style="color:{color}; font-weight:600;">{indicator} {stress:+.2f}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("**Social Cohesion**")
        bonds = prims.get("bonds", {})
        for label, key in [("National", "national_cohesion"), ("Class", "class_solidarity"),
                           ("Regional", "regional_unity"), ("Fragility", "fragility")]:
            val = bonds.get(key, 0)
            st.metric(label, f"{val:.0%}")
        
        st.markdown("---")
        inst = prims.get("instability_index", 0)
        crisis = prims.get("crisis_probability", 0)
        i_color = theme.accent_danger if inst > 0.5 else theme.accent_warning if inst > 0.3 else theme.accent_success
        c_color = theme.accent_danger if crisis > 0.3 else theme.accent_warning if crisis > 0.15 else theme.accent_success
        st.markdown(f"""
        <div class="glass-card" style="text-align:center; padding:1rem;">
            <div style="font-size:0.8rem; color:{theme.text_muted};">Instability Index</div>
            <div style="font-size:1.8rem; font-weight:700; color:{i_color};">{inst:.1%}</div>
            <div style="font-size:0.8rem; color:{theme.text_muted}; margin-top:0.5rem;">Crisis Probability</div>
            <div style="font-size:1.8rem; font-weight:700; color:{c_color};">{crisis:.1%}</div>
        </div>
        """, unsafe_allow_html=True)


def _render_analysis_controls():
    """Sidebar controls for live signal analysis."""
    with st.sidebar:
        st.markdown("### Signal Analysis")
        
        test_text = st.text_area("Test text:", height=80, key="analysis_input",
                                 placeholder="Enter text to analyze for threat signals...")
        
        if st.button("Analyze", key="analyze_btn"):
            if test_text:
                try:
                    from kshiked.pulse.sensor import PulseSensor
                    sensor = PulseSensor()
                    detections = sensor.process_text(test_text)
                    if detections:
                        st.success(f"Detected {len(detections)} signals:")
                        for d in detections:
                            name = d.signal_id.name.replace("_", " ").title()
                            st.write(f"• {name}: {d.intensity:.0%}")
                    else:
                        st.info("No signals detected.")
                except Exception as e:
                    st.error(f"Analysis error: {e}")
            else:
                st.warning("Enter text to analyze.")
        
        st.markdown("---")
        st.markdown("### System Info")
        st.caption("SENTINEL v2.0 • KShield Engine")


# =============================================================================
# Home / Landing Page
# =============================================================================

def _render_home(theme):
    """Render the landing page with hero section and navigation cards."""
    
    # CSS for Hero and Card-Buttons
    hero_css = f"""
    <style>
    @keyframes fadeIn {{
        0% {{ opacity: 0; transform: translateY(20px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes gradient-shift {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    @keyframes glow {{
        0% {{ text-shadow: 0 0 10px {theme.accent_primary}55; }}
        50% {{ text-shadow: 0 0 25px {theme.accent_primary}, 0 0 10px {theme.text_primary}; }}
        100% {{ text-shadow: 0 0 10px {theme.accent_primary}55; }}
    }}
    @keyframes typing {{
        from {{ width: 0 }}
        to {{ width: 100% }}
    }}
    @keyframes flowPath {{
        0% {{ stroke-dashoffset: 2400; opacity: 0; }}
        10% {{ opacity: 1; }}
        90% {{ opacity: 1; }}
        100% {{ stroke-dashoffset: 0; opacity: 0; }}
    }}
    
    .home-wrapper {{
        position: relative;
        width: 100%;
        min-height: 300px;
        overflow: hidden;
    }}
    
    .bg-paths-svg {{
        position: fixed;
        top: 0; left: 0;
        width: 100vw; height: 100vh;
        opacity: 0.4;
        pointer-events: none;
        z-index: 0;
    }}
    
    .bg-path {{
        fill: none;
        stroke-dasharray: 2400;
        stroke-dashoffset: 2400;
        animation: flowPath 10s ease-in-out infinite;
    }}
    
    .hero-container {{
        position: relative;
        z-index: 1;
        text-align: center;
        padding: 1rem 2rem 0.5rem;
        display: flex;
        flex-direction: column;
        align-items: center;
    }}
    
    .hero-title-wrapper {{
        display: inline-block;
        overflow: hidden;
        white-space: nowrap;
        margin: 0 auto;
        border-right: .15em solid {theme.accent_primary};
        animation: 
            typing 2.5s steps(30, end),
            blink-caret .75s step-end infinite;
        width: 100%;
        max-width: 800px;
    }}
    
    @keyframes blink-caret {{
        from, to {{ border-color: transparent }}
        50% {{ border-color: {theme.accent_primary} }}
    }}

    .hero-title {{
        font-family: 'Courier New', monospace;
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, {theme.text_primary}, {theme.accent_primary});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 4px;
    }}
    
    .hero-desc {{
        font-size: 1.5rem;
        color: #E6EAF0;
        font-weight: 300;
        letter-spacing: 2px;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        opacity: 0;
        animation: fadeIn 1.5s ease-out forwards;
        animation-delay: 1.2s;
    }}
    
    .hero-subtitle {{
        opacity: 0;
        font-size: 1.0rem;
        color: {theme.text_muted};
        font-weight: 400;
        letter-spacing: 6px;
        text-transform: uppercase;
        margin-top: 1rem;
        animation: fadeIn 1s ease-out forwards;
        animation-delay: 2.5s; 
    }}



    </style>
    """
    st.markdown(hero_css, unsafe_allow_html=True)
    
    # Hero with Background Paths
    st.markdown(f"""
    <div class="home-wrapper">
        <svg class="bg-paths-svg" viewBox="0 0 1200 600" preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="pg1" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="{theme.accent_primary}" stop-opacity="0"/>
                    <stop offset="50%" stop-color="{theme.accent_primary}" stop-opacity="0.6"/>
                    <stop offset="100%" stop-color="{theme.accent_primary}" stop-opacity="0"/>
                </linearGradient>
                <linearGradient id="pg2" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stop-color="{theme.accent_info}" stop-opacity="0"/>
                    <stop offset="50%" stop-color="{theme.accent_info}" stop-opacity="0.4"/>
                    <stop offset="100%" stop-color="{theme.accent_info}" stop-opacity="0"/>
                </linearGradient>
            </defs>
            <path class="bg-path" d="M-100 300Q200 100,500 250T900 200T1300 350" stroke="url(#pg1)" stroke-width="2.5" style="animation-delay:0s"/>
            <path class="bg-path" d="M-100 400Q300 200,600 350T1000 250T1400 400" stroke="url(#pg1)" stroke-width="2" style="animation-delay:1.5s"/>
            <path class="bg-path" d="M-100 150Q250 350,550 180T950 320T1400 150" stroke="url(#pg2)" stroke-width="2" style="animation-delay:3s"/>
            <path class="bg-path" d="M-100 500Q350 280,650 420T1050 300T1400 500" stroke="url(#pg1)" stroke-width="1.5" style="animation-delay:4.5s"/>
            <path class="bg-path" d="M-100 50Q200 250,500 100T900 280T1300 80" stroke="url(#pg2)" stroke-width="2" style="animation-delay:6s"/>
            <path class="bg-path" d="M-100 250Q400 50,700 280T1100 100T1400 300" stroke="url(#pg1)" stroke-width="1.5" style="animation-delay:2s"/>
        </svg>
        <div class="hero-container">
            <div class="hero-title-wrapper">
                <div class="hero-title">WELCOME TO SENTINEL</div>
            </div>
            <div class="hero-desc">The Autonomous Economic Defense &amp; Simulation Platform</div>
            <div class="hero-subtitle">— Powered by Scarcity —</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation Cards — 2x2 Grid
    cards = [
        ("K-SHIELD", "Run large-scale economic simulations across sectors. Test policy scenarios, model shocks, and evaluate cascading risk using adaptive ABM agents.", "KSHIELD"),
        ("K-PULSE", "Continuously ingest and analyze live signals to detect anomalies. Monitor behavioral shifts and generate early warning intelligence in real-time.", "SIGNALS"),
        ("K-COLLAB", "Enable organizations to collaboratively train models and generate insights using federated learning and secure aggregation.", "FEDERATION"),
        ("K-EDUCATION", "Translate complex security intelligence into clear public knowledge through explainable analytics and accessible awareness dashboards.", "DOCS"),
    ]
    
    def render_card(col, title, desc, target):
        with col:
            label = f"{title}\n\n{desc}"
            if st.button(label, key=f"card_btn_{target}", use_container_width=True):
                st.session_state.current_view = target
                st.rerun()
    
    # Top Row
    _, c1_top, c2_top, _ = st.columns([1, 3, 3, 1])
    render_card(c1_top, *cards[0])
    render_card(c2_top, *cards[1])
    
    # Bottom Row
    _, c1_bot, c2_bot, _ = st.columns([1, 3, 3, 1])
    render_card(c1_bot, *cards[2])
    render_card(c2_bot, *cards[3])
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: {theme.text_muted}; font-size: 0.8rem;">
        SENTINEL v2.0 &bull; STRATEGIC COMMAND &amp; CONTROL
    </div>
    """, unsafe_allow_html=True)


def main():
    """Run the SENTINEL dashboard."""
    render_sentinel_dashboard()


if __name__ == "__main__":
    main()
