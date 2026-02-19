"""
SENTINEL Kenya Threat Map

Kaspersky Cybermap-inspired visualization:
- Dark theme 3D globe centered on Kenya
- Animated threat arcs between counties
- Glowing hotspots
- Real-time threat counters
- Dramatic visual effects

Designed for the SENTINEL Command Center.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

logger = logging.getLogger("sentinel.threatmap")

# =============================================================================
# Kenya Counties Data (47 counties with coordinates)
# =============================================================================

KENYA_COUNTIES = {
    "Nairobi": {"lat": -1.2921, "lon": 36.8219, "region": "Central"},
    "Mombasa": {"lat": -4.0435, "lon": 39.6682, "region": "Coast"},
    "Kisumu": {"lat": -0.0917, "lon": 34.7680, "region": "Western"},
    "Nakuru": {"lat": -0.3031, "lon": 36.0666, "region": "Rift Valley"},
    "Uasin Gishu": {"lat": 0.5143, "lon": 35.2698, "region": "Rift Valley"},
    "Kiambu": {"lat": -1.1714, "lon": 36.8260, "region": "Central"},
    "Garissa": {"lat": -0.4533, "lon": 39.6460, "region": "North Eastern"},
    "Wajir": {"lat": 1.7500, "lon": 40.0573, "region": "North Eastern"},
    "Mandera": {"lat": 3.9370, "lon": 40.9617, "region": "North Eastern"},
    "Meru": {"lat": 0.0515, "lon": 37.6493, "region": "Eastern"},
    "Nyeri": {"lat": -0.4246, "lon": 36.9514, "region": "Central"},
    "Machakos": {"lat": -1.5177, "lon": 37.2634, "region": "Eastern"},
    "Kisii": {"lat": -0.6698, "lon": 34.7660, "region": "Western"},
    "Kakamega": {"lat": 0.2827, "lon": 34.7520, "region": "Western"},
    "Bungoma": {"lat": 0.5635, "lon": 34.5584, "region": "Western"},
    "Turkana": {"lat": 3.1166, "lon": 35.5991, "region": "Rift Valley"},
    "Marsabit": {"lat": 2.3285, "lon": 37.9897, "region": "Eastern"},
    "Isiolo": {"lat": 0.3556, "lon": 38.4831, "region": "Eastern"},
    "Laikipia": {"lat": 0.1520, "lon": 36.7819, "region": "Rift Valley"},
    "Narok": {"lat": -1.0821, "lon": 35.8716, "region": "Rift Valley"},
    "Kajiado": {"lat": -1.8519, "lon": 36.7820, "region": "Rift Valley"},
    "Kericho": {"lat": -0.3689, "lon": 35.2863, "region": "Rift Valley"},
    "Bomet": {"lat": -0.7813, "lon": 35.3428, "region": "Rift Valley"},
    "Migori": {"lat": -1.0634, "lon": 34.4731, "region": "Western"},
    "Homa Bay": {"lat": -0.5273, "lon": 34.4571, "region": "Western"},
    "Siaya": {"lat": 0.0612, "lon": 34.2422, "region": "Western"},
    "Busia": {"lat": 0.4608, "lon": 34.1115, "region": "Western"},
    "Trans Nzoia": {"lat": 1.0567, "lon": 34.9506, "region": "Rift Valley"},
    "Nandi": {"lat": 0.1836, "lon": 35.1269, "region": "Rift Valley"},
    "Baringo": {"lat": 0.4912, "lon": 35.9683, "region": "Rift Valley"},
    "Elgeyo Marakwet": {"lat": 0.7835, "lon": 35.5123, "region": "Rift Valley"},
    "West Pokot": {"lat": 1.6210, "lon": 35.1215, "region": "Rift Valley"},
    "Samburu": {"lat": 1.2150, "lon": 37.0020, "region": "Rift Valley"},
    "Kwale": {"lat": -4.1737, "lon": 39.4521, "region": "Coast"},
    "Kilifi": {"lat": -3.5107, "lon": 39.8499, "region": "Coast"},
    "Tana River": {"lat": -1.8047, "lon": 39.9887, "region": "Coast"},
    "Lamu": {"lat": -2.2686, "lon": 40.9020, "region": "Coast"},
    "Taita Taveta": {"lat": -3.3160, "lon": 38.3671, "region": "Coast"},
    "Embu": {"lat": -0.5275, "lon": 37.4596, "region": "Eastern"},
    "Kitui": {"lat": -1.3673, "lon": 38.0106, "region": "Eastern"},
    "Makueni": {"lat": -1.8039, "lon": 37.6203, "region": "Eastern"},
    "Tharaka Nithi": {"lat": -0.3076, "lon": 37.8800, "region": "Eastern"},
    "Nyandarua": {"lat": -0.1804, "lon": 36.3869, "region": "Central"},
    "Kirinyaga": {"lat": -0.4989, "lon": 37.2824, "region": "Central"},
    "Muranga": {"lat": -0.7839, "lon": 37.1504, "region": "Central"},
    "Nyamira": {"lat": -0.5634, "lon": 34.9340, "region": "Western"},
    "Vihiga": {"lat": 0.0541, "lon": 34.7234, "region": "Western"},
}

# Threat types with colors
THREAT_TYPES = {
    "survival_stress": {"color": "#ff3333", "name": "Survival Stress", "icon": "🔴"},
    "mobilization": {"color": "#f5d547", "name": "Mobilization", "icon": "🟠"},
    "scapegoating": {"color": "#f5d547", "name": "Scapegoating", "icon": "🟡"},
    "grievance": {"color": "#ff6699", "name": "Grievance", "icon": "🩷"},
    "polarization": {"color": "#cc33ff", "name": "Polarization", "icon": "🟣"},
    "delegitimization": {"color": "#3399ff", "name": "Delegitimization", "icon": "🔵"},
}

# =============================================================================
# Dark Theme Styles
# =============================================================================

THREATMAP_CSS = """
<style>
    /* Dark theme for threat map */
    .threatmap-container {
        background: linear-gradient(135deg, #0a0a1a, #1a1a3e);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .threatmap-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .threatmap-title {
        color: #00ff88;
        font-size: 1.5rem;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(0,255,136,0.5);
    }
    
    .threatmap-subtitle {
        color: #666;
        font-size: 0.85rem;
    }
    
    /* Live indicator */
    .live-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #ff3333;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .live-dot {
        width: 10px;
        height: 10px;
        background: #ff3333;
        border-radius: 50%;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }
    
    /* Threat counters */
    .threat-counters {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    
    .counter-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        min-width: 120px;
        text-align: center;
    }
    
    .counter-value {
        font-size: 1.8rem;
        font-weight: 700;
        font-family: 'Courier New', monospace;
    }
    
    .counter-label {
        font-size: 0.7rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Legend */
    .threat-legend {
        display: flex;
        gap: 1rem;
        flex-wrap: wrap;
        margin-top: 1rem;
        padding: 0.75rem;
        background: rgba(0,0,0,0.3);
        border-radius: 8px;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        color: #aaa;
        font-size: 0.75rem;
    }
    
    .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        box-shadow: 0 0 8px currentColor;
    }
    
    /* Stats panel */
    .stats-panel {
        background: rgba(0,0,0,0.4);
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    .stat-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .stat-row:last-child {
        border-bottom: none;
    }
    
    .stat-label {
        color: #888;
        font-size: 0.85rem;
    }
    
    .stat-value {
        color: #fff;
        font-weight: 600;
    }
    
    .stat-value.critical {
        color: #ff3333;
    }
    
    .stat-value.warning {
        color: #f5d547;
    }
    
    .stat-value.normal {
        color: #00ff88;
    }
</style>
"""

# =============================================================================
# Threat Map Visualization
# =============================================================================

def render_kenya_threatmap(
    threat_data: Dict = None,
    show_arcs: bool = True,
    animate: bool = True,
) -> None:
    """Render Kaspersky-style Kenya threat map."""
    if not HAS_STREAMLIT or not HAS_PLOTLY or not HAS_PANDAS:
        print("Error: Requires streamlit, plotly, pandas, numpy")
        return
    
    # Apply dark theme styles
    st.markdown(THREATMAP_CSS, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="threatmap-container">
        <div class="threatmap-header">
            <div>
                <div class="threatmap-title">🛡️ SENTINEL THREAT MAP</div>
                <div class="threatmap-subtitle">Real-time National Security Intelligence</div>
            </div>
            <div class="live-indicator">
                <div class="live-dot"></div>
                LIVE
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if threat_data is None:
        threat_data = {
            "threats": pd.DataFrame(columns=["source", "target", "source_lat", "source_lon", "target_lat", "target_lon", "type", "intensity", "color"]),
            "arcs": [],
            "stats": {k: 0 for k in THREAT_TYPES.keys()}
        }
    
    threats_df = threat_data["threats"]
    
    # Main map
    col1, col2 = st.columns([3, 1])
    
    with col1:
        _render_3d_globe(threats_df, show_arcs)
    
    with col2:
        _render_threat_stats(threats_df)
    
    # Threat counters
    _render_threat_counters(threats_df)
    
    # Legend
    _render_legend()


def _render_3d_globe(threats: pd.DataFrame, show_arcs: bool) -> None:
    """Render 3D globe visualization centered on Kenya."""
    
    fig = go.Figure()
    
    # Add county hotspots
    hotspot_intensity = threats.groupby("source")["intensity"].sum().to_dict()
    
    for county, coords in KENYA_COUNTIES.items():
        intensity = hotspot_intensity.get(county, 0.1)
        size = 8 + intensity * 20
        
        # Glow effect color
        if intensity > 1.5:
            color = "#ff3333"
        elif intensity > 0.8:
            color = "#f5d547"
        else:
            color = "#00ff88"
        
        fig.add_trace(go.Scattergeo(
            lon=[coords["lon"]],
            lat=[coords["lat"]],
            mode='markers',
            marker=dict(
                size=size,
                color=color,
                opacity=0.8,
                line=dict(width=2, color=color),
            ),
            text=f"<b>{county}</b><br>Threat Level: {intensity:.2f}",
            hoverinfo='text',
            showlegend=False,
        ))
        
        # Add glow ring
        fig.add_trace(go.Scattergeo(
            lon=[coords["lon"]],
            lat=[coords["lat"]],
            mode='markers',
            marker=dict(
                size=size * 1.5,
                color=color,
                opacity=0.2,
            ),
            hoverinfo='skip',
            showlegend=False,
        ))
    
    # Add threat arcs
    if show_arcs:
        for _, threat in threats.iterrows():
            # Create arc points
            arc_lon, arc_lat = _create_arc(
                threat["source_lon"], threat["source_lat"],
                threat["target_lon"], threat["target_lat"],
            )
            
            fig.add_trace(go.Scattergeo(
                lon=arc_lon,
                lat=arc_lat,
                mode='lines',
                line=dict(
                    width=threat["intensity"] * 3,
                    color=threat["color"],
                ),
                opacity=0.6,
                hoverinfo='skip',
                showlegend=False,
            ))
    
    # Kenya-focused view with dark theme
    fig.update_geos(
        center=dict(lat=0.5, lon=37.5),
        projection_scale=15,
        scope='africa',
        showland=True,
        landcolor='#1a1a2e',
        showocean=True,
        oceancolor='#0a0a1a',
        showcountries=True,
        countrycolor='#333366',
        countrywidth=1,
        showlakes=True,
        lakecolor='#0a0a1a',
        bgcolor='#0a0a1a',
    )
    
    fig.update_layout(
        height=550,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='#0a0a1a',
        geo=dict(bgcolor='#0a0a1a'),
        showlegend=False,
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _create_arc(lon1: float, lat1: float, lon2: float, lat2: float, n_points: int = 20) -> tuple:
    """Create curved arc between two points."""
    # Great circle arc with elevation
    lons = np.linspace(lon1, lon2, n_points)
    lats = np.linspace(lat1, lat2, n_points)
    
    # Add curvature (parabolic arc in lat/lon space)
    mid = n_points // 2
    arc_height = np.sqrt((lon2 - lon1)**2 + (lat2 - lat1)**2) * 0.15
    
    for i in range(n_points):
        dist_from_mid = abs(i - mid) / mid
        lats[i] += arc_height * (1 - dist_from_mid**2)
    
    return lons.tolist(), lats.tolist()


def _render_threat_stats(threats: pd.DataFrame) -> None:
    """Render threat statistics panel."""
    
    # Calculate stats
    total_threats = len(threats)
    critical_count = len(threats[threats["intensity"] > 0.7])
    top_source = threats.groupby("source").size().idxmax() if len(threats) > 0 else "N/A"
    top_target = threats.groupby("target").size().idxmax() if len(threats) > 0 else "N/A"
    avg_intensity = threats["intensity"].mean() if len(threats) > 0 else 0
    
    st.markdown(f"""
    <div class="stats-panel">
        <div style="color: #00ff88; font-weight: 600; margin-bottom: 0.75rem; font-size: 0.9rem;">
            📊 THREAT ANALYSIS
        </div>
        
        <div class="stat-row">
            <span class="stat-label">Active Threats</span>
            <span class="stat-value critical">{total_threats}</span>
        </div>
        
        <div class="stat-row">
            <span class="stat-label">Critical Level</span>
            <span class="stat-value warning">{critical_count}</span>
        </div>
        
        <div class="stat-row">
            <span class="stat-label">Hottest Source</span>
            <span class="stat-value">{top_source}</span>
        </div>
        
        <div class="stat-row">
            <span class="stat-label">Most Targeted</span>
            <span class="stat-value">{top_target}</span>
        </div>
        
        <div class="stat-row">
            <span class="stat-label">Avg Intensity</span>
            <span class="stat-value {'critical' if avg_intensity > 0.6 else 'warning' if avg_intensity > 0.4 else 'normal'}">{avg_intensity:.0%}</span>
        </div>
        
        <div style="margin-top: 1rem; padding-top: 0.75rem; border-top: 1px solid rgba(255,255,255,0.1);">
            <div class="stat-row">
                <span class="stat-label">National Status</span>
                <span class="stat-value {'critical' if critical_count > 5 else 'warning' if critical_count > 2 else 'normal'}">
                    {'🔴 ELEVATED' if critical_count > 5 else '🟠 GUARDED' if critical_count > 2 else '🟢 STABLE'}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Top threats by type
    st.markdown("""
    <div class="stats-panel" style="margin-top: 0.75rem;">
        <div style="color: #00ff88; font-weight: 600; margin-bottom: 0.5rem; font-size: 0.9rem;">
            ⚡ THREAT TYPES
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    type_counts = threats.groupby("type").size().sort_values(ascending=False)
    for threat_type, count in type_counts.head(5).items():
        info = THREAT_TYPES[threat_type]
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; padding: 0.25rem 0; color: #aaa; font-size: 0.8rem;">
            <span>{info['icon']} {info['name']}</span>
            <span style="color: {info['color']};">{count}</span>
        </div>
        """, unsafe_allow_html=True)


def _render_threat_counters(threats: pd.DataFrame) -> None:
    """Render animated threat counters."""
    
    # Calculate counters
    total = len(threats)
    critical = len(threats[threats["intensity"] > 0.7])
    regions = threats["source"].nunique()
    
    # Simulated real-time counters
    np.random.seed(42)
    today_total = np.random.randint(150, 300)
    blocked = np.random.randint(80, 150)
    
    st.markdown(f"""
    <div class="threat-counters">
        <div class="counter-card">
            <div class="counter-value" style="color: #ff3333;">{total}</div>
            <div class="counter-label">Active Now</div>
        </div>
        <div class="counter-card">
            <div class="counter-value" style="color: #f5d547;">{critical}</div>
            <div class="counter-label">Critical</div>
        </div>
        <div class="counter-card">
            <div class="counter-value" style="color: #00ff88;">{regions}</div>
            <div class="counter-label">Regions</div>
        </div>
        <div class="counter-card">
            <div class="counter-value" style="color: #3399ff;">{today_total}</div>
            <div class="counter-label">Today Total</div>
        </div>
        <div class="counter-card">
            <div class="counter-value" style="color: #00ff88;">{blocked}</div>
            <div class="counter-label">Mitigated</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_legend() -> None:
    """Render threat type legend."""
    
    legend_items = "".join([
        f'<div class="legend-item"><div class="legend-dot" style="background: {info["color"]}; color: {info["color"]};"></div>{info["name"]}</div>'
        for threat_type, info in THREAT_TYPES.items()
    ])
    
    st.markdown(f"""
    <div class="threat-legend">
        {legend_items}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Standalone Runner
# =============================================================================

def main():
    """Run threat map standalone."""
    st.set_page_config(
        page_title="SENTINEL Threat Map",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    
    # Dark background
    st.markdown("""
    <style>
        .stApp { background-color: #0a0a1a; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
    
    # Create empty mock data structure for standalone run
    mock_data = {
        "threats": pd.DataFrame(columns=["lat", "lon", "type", "intensity", "city", "region"]),
        "arcs": [],
        "stats": {k: 0 for k in THREAT_TYPES.keys()}
    }
    
    render_kenya_threatmap(mock_data)
    
    # Auto-refresh hint
    st.markdown("""
    <div style="text-align: center; color: #444; font-size: 0.75rem; margin-top: 2rem;">
        Refresh page for updated threat data • SENTINEL v2.0
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
