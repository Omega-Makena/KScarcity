"""
KShield Pulse - Visualization Module

Provides interactive visualizations for threat analysis:
- Kenya threat heatmap (geographic)
- Threat index gauges/charts
- Signal timeline
- Network graph visualization
- Ethnic tension flow diagram

Uses Plotly for interactive charts and Folium for maps.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

# Handle optional dependencies
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None
    px = None

try:
    import folium
    from folium.plugins import HeatMap
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False
    folium = None

logger = logging.getLogger("kshield.pulse.visualization")


# =============================================================================
# Kenya Geographic Data
# =============================================================================

KENYA_CENTER = (-1.2921, 37.5)  # Center of Kenya
KENYA_ZOOM = 6

# Major cities and counties
KENYA_LOCATIONS = {
    "nairobi": {"lat": -1.2921, "lon": 36.8219, "population": 4400000},
    "mombasa": {"lat": -4.0435, "lon": 39.6682, "population": 1200000},
    "kisumu": {"lat": -0.0917, "lon": 34.7680, "population": 500000},
    "nakuru": {"lat": -0.3031, "lon": 36.0800, "population": 570000},
    "eldoret": {"lat": 0.5143, "lon": 35.2698, "population": 475000},
    "machakos": {"lat": -1.5177, "lon": 37.2634, "population": 150000},
    "meru": {"lat": 0.0500, "lon": 37.6500, "population": 200000},
    "nyeri": {"lat": -0.4197, "lon": 36.9553, "population": 125000},
    "kakamega": {"lat": 0.2827, "lon": 34.7519, "population": 100000},
    "kisii": {"lat": -0.6817, "lon": 34.7667, "population": 130000},
    "garissa": {"lat": -0.4536, "lon": 39.6401, "population": 180000},
    "turkana": {"lat": 3.3122, "lon": 35.5658, "population": 100000},
    "mandera": {"lat": 3.9366, "lon": 41.8670, "population": 90000},
    "wajir": {"lat": 1.7471, "lon": 40.0573, "population": 80000},
    "kitui": {"lat": -1.3667, "lon": 38.0167, "population": 155000},
    "malindi": {"lat": -3.2138, "lon": 40.1169, "population": 120000},
    "naivasha": {"lat": -0.7172, "lon": 36.4328, "population": 180000},
    "thika": {"lat": -1.0334, "lon": 37.0692, "population": 200000},
    "lamu": {"lat": -2.2686, "lon": 40.9020, "population": 25000},
    "isiolo": {"lat": 0.3556, "lon": 37.5833, "population": 45000},
}


# =============================================================================
# Kenya Threat Map
# =============================================================================

def create_kenya_heatmap(
    threat_data: List[Dict],
    title: str = "Kenya Threat Heatmap",
) -> Optional[Any]:
    """
    Create an interactive Kenya threat heatmap using Folium.
    
    Args:
        threat_data: List of {lat, lon, intensity, name, ...} dicts
        title: Map title
        
    Returns:
        Folium map object or None if folium not installed
    """
    if not HAS_FOLIUM:
        logger.warning("folium not installed: pip install folium")
        return None
    
    # Create base map centered on Kenya
    m = folium.Map(
        location=KENYA_CENTER,
        zoom_start=KENYA_ZOOM,
        tiles="cartodbpositron",
    )
    
    # Add title
    title_html = f'''
    <h3 align="center" style="font-size:16px"><b>{title}</b></h3>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Prepare heatmap data
    heat_points = []
    for d in threat_data:
        if d.get("intensity", 0) > 0:
            heat_points.append([
                d["lat"],
                d["lon"],
                d["intensity"],
            ])
    
    # Add heatmap layer
    if heat_points:
        HeatMap(
            heat_points,
            min_opacity=0.3,
            max_zoom=13,
            radius=25,
            blur=15,
            gradient={0.4: 'yellow', 0.65: 'orange', 1: 'red'},
        ).add_to(m)
    
    # Add markers for major locations
    for d in threat_data:
        if d.get("intensity", 0) > 0.3:  # Only show significant threats
            color = "red" if d["intensity"] > 0.7 else "orange" if d["intensity"] > 0.4 else "yellow"
            
            folium.CircleMarker(
                location=[d["lat"], d["lon"]],
                radius=8 + d["intensity"] * 10,
                popup=f"""
                    <b>{d.get('name', 'Unknown').title()}</b><br>
                    Threat Level: {d['intensity']:.2f}<br>
                    Posts: {d.get('posts', 'N/A')}<br>
                    Threats: {d.get('threats', 'N/A')}
                """,
                color=color,
                fill=True,
                fillOpacity=0.7,
            ).add_to(m)
    
    return m


def save_kenya_map(threat_data: List[Dict], filepath: str) -> bool:
    """
    Create and save Kenya threat map to HTML file.
    
    Args:
        threat_data: List of threat data points
        filepath: Output HTML file path
        
    Returns:
        True if successful
    """
    m = create_kenya_heatmap(threat_data)
    if m:
        m.save(filepath)
        logger.info(f"Saved Kenya heatmap to {filepath}")
        return True
    return False


# =============================================================================
# Threat Index Dashboard
# =============================================================================

def create_threat_gauge(
    value: float,
    title: str,
    severity: str = "LOW",
) -> Optional[Any]:
    """
    Create a gauge chart for a threat index.
    
    Args:
        value: Index value [0, 1]
        title: Index name
        severity: Severity level for color
        
    Returns:
        Plotly figure or None
    """
    if not HAS_PLOTLY:
        return None
    
    # Color based on severity
    colors = {
        "LOW": "green",
        "MODERATE": "yellow",
        "ELEVATED": "orange",
        "HIGH": "orangered",
        "CRITICAL": "red",
        "IMMINENT": "darkred",
        "STABLE": "green",
        "NORMAL": "green",
    }
    bar_color = colors.get(severity, "gray")
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16}},
        number={'suffix': "%", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': bar_color},
            'bgcolor': "white",
            'borderwidth': 2,
            'steps': [
                {'range': [0, 20], 'color': 'lightgreen'},
                {'range': [20, 40], 'color': 'lightyellow'},
                {'range': [40, 60], 'color': 'moccasin'},
                {'range': [60, 80], 'color': 'lightsalmon'},
                {'range': [80, 100], 'color': 'lightcoral'},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value * 100,
            },
        },
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig


def create_threat_dashboard(report: Dict) -> Optional[Any]:
    """
    Create a multi-gauge dashboard for all threat indices.
    
    Args:
        report: ThreatIndexReport.to_dict() output
        
    Returns:
        Plotly figure with all gauges
    """
    if not HAS_PLOTLY:
        return None
    
    indices = report.get("indices", {})
    
    # Create subplots (2 rows x 4 cols)
    fig = make_subplots(
        rows=2, cols=4,
        specs=[[{"type": "indicator"}] * 4] * 2,
        subplot_titles=[
            "Polarization", "Legitimacy Erosion", "Mobilization", "Elite Cohesion",
            "Info Warfare", "Security Friction", "Economic Cascade", "Ethnic Tension"
        ],
    )
    
    # Add gauges
    data = [
        ("polarization", 1, 1),
        ("legitimacy_erosion", 1, 2),
        ("mobilization_readiness", 1, 3),
        ("elite_cohesion", 1, 4),
        ("information_warfare", 2, 1),
        ("security_friction", 2, 2),
        ("economic_cascade", 2, 3),
        ("ethnic_tension", 2, 4),
    ]
    
    for idx_name, row, col in data:
        idx = indices.get(idx_name, {})
        value = idx.get("value", idx.get("avg_tension", 0.5))
        severity = idx.get("severity", "LOW")
        
        colors = {
            "LOW": "green", "MODERATE": "yellow", "ELEVATED": "orange",
            "HIGH": "orangered", "CRITICAL": "red", "STABLE": "green",
        }
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=value * 100,
                number={'suffix': "%"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': colors.get(severity, "gray")},
                    'steps': [
                        {'range': [0, 30], 'color': 'lightgreen'},
                        {'range': [30, 60], 'color': 'lightyellow'},
                        {'range': [60, 100], 'color': 'lightcoral'},
                    ],
                },
            ),
            row=row, col=col,
        )
    
    fig.update_layout(
        height=500,
        title_text=f"Threat Level: {report.get('overall_threat_level', 'UNKNOWN')}",
        title_x=0.5,
        title_font_size=20,
    )
    
    return fig


# =============================================================================
# Signal Timeline
# =============================================================================

def create_signal_timeline(
    signals: List[Dict],
    hours: int = 24,
) -> Optional[Any]:
    """
    Create a timeline of signal detections.
    
    Args:
        signals: List of {signal_id, intensity, timestamp} dicts
        hours: Hours to display
        
    Returns:
        Plotly figure
    """
    if not HAS_PLOTLY:
        return None
    
    if not signals:
        return None
    
    # Group by signal type
    by_signal = {}
    for s in signals:
        sig_id = s.get("signal_id", "unknown")
        if sig_id not in by_signal:
            by_signal[sig_id] = {"x": [], "y": []}
        by_signal[sig_id]["x"].append(s.get("timestamp", datetime.utcnow()))
        by_signal[sig_id]["y"].append(s.get("intensity", 0.5))
    
    fig = go.Figure()
    
    for sig_id, data in by_signal.items():
        fig.add_trace(go.Scatter(
            x=data["x"],
            y=data["y"],
            mode="lines+markers",
            name=sig_id,
            line=dict(width=2),
        ))
    
    fig.update_layout(
        title="Signal Detection Timeline",
        xaxis_title="Time",
        yaxis_title="Intensity",
        yaxis=dict(range=[0, 1]),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    
    return fig


# =============================================================================
# Ethnic Tension Chord Diagram
# =============================================================================

def create_ethnic_tension_chord(
    tensions: Dict[str, float],
) -> Optional[Any]:
    """
    Create a chord diagram showing ethnic group tensions.
    
    Args:
        tensions: Dict of "group1-group2" -> tension_value
        
    Returns:
        Plotly figure
    """
    if not HAS_PLOTLY:
        return None
    
    if not tensions:
        return None
    
    # Parse tensions into matrix
    groups = set()
    for key in tensions:
        g1, g2 = key.split("-")
        groups.add(g1)
        groups.add(g2)
    
    groups = sorted(list(groups))
    n = len(groups)
    
    if n < 2:
        return None
    
    # Build adjacency matrix
    matrix = [[0.0] * n for _ in range(n)]
    for key, value in tensions.items():
        g1, g2 = key.split("-")
        i, j = groups.index(g1), groups.index(g2)
        matrix[i][j] = value
        matrix[j][i] = value
    
    # Create heatmap (simpler than chord for now)
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=groups,
        y=groups,
        colorscale='RdYlGn_r',
        zmin=0,
        zmax=1,
        colorbar=dict(title="Tension"),
    ))
    
    fig.update_layout(
        title="Ethnic Group Tension Matrix",
        xaxis_title="Group",
        yaxis_title="Group",
        height=400,
        width=500,
    )
    
    return fig


# =============================================================================
# Network Visualization
# =============================================================================

def create_network_graph(
    nodes: List[Dict],
    edges: List[Dict],
    title: str = "Actor Network",
) -> Optional[Any]:
    """
    Create an interactive network graph visualization.
    
    Args:
        nodes: List of {id, label, role, size} dicts
        edges: List of {source, target, weight} dicts
        title: Graph title
        
    Returns:
        Plotly figure
    """
    if not HAS_PLOTLY:
        return None
    
    if not nodes:
        return None
    
    # Create a simple spring layout manually
    import random
    random.seed(42)
    
    pos = {n["id"]: (random.random() * 2 - 1, random.random() * 2 - 1) for n in nodes}
    
    # Role colors
    role_colors = {
        "mobilizer": "red",
        "broker": "purple",
        "ideologue": "orange",
        "amplifier": "blue",
        "influencer": "green",
        "bot": "gray",
        "unknown": "lightgray",
    }
    
    # Create edge traces
    edge_x = []
    edge_y = []
    for e in edges:
        x0, y0 = pos.get(e["source"], (0, 0))
        x1, y1 = pos.get(e["target"], (0, 0))
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines',
    )
    
    # Create node traces
    node_x = [pos[n["id"]][0] for n in nodes]
    node_y = [pos[n["id"]][1] for n in nodes]
    node_colors = [role_colors.get(n.get("role", "unknown"), "lightgray") for n in nodes]
    node_sizes = [10 + n.get("size", 1) * 5 for n in nodes]
    node_text = [f"{n.get('label', n['id'])}<br>Role: {n.get('role', 'unknown')}" for n in nodes]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[n.get('label', '')[:10] for n in nodes],
        textposition="top center",
        hovertext=node_text,
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=2, color='white'),
        ),
    )
    
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title=title,
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
    )
    
    return fig


# =============================================================================
# HTML Report Generator
# =============================================================================

def generate_html_report(
    threat_report: Dict,
    location_data: Optional[List[Dict]] = None,
    signals: Optional[List[Dict]] = None,
    output_path: str = "threat_report.html",
) -> str:
    """
    Generate a complete HTML threat report with all visualizations.
    
    Args:
        threat_report: ThreatIndexReport.to_dict()
        location_data: Optional location threat data for map
        signals: Optional signal timeline data
        output_path: Output file path
        
    Returns:
        Path to generated HTML file
    """
    html_parts = []
    
    # Header
    html_parts.append(f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>KShield Pulse - Threat Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .header {{ background: linear-gradient(135deg, #1a1a2e, #16213e); color: white; padding: 20px; border-radius: 10px; }}
            .header h1 {{ margin: 0; }}
            .alert-box {{ background: #fff3cd; border: 1px solid #ffc107; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .critical {{ background: #f8d7da; border-color: #f5c6cb; }}
            .section {{ background: white; padding: 20px; margin: 20px 0; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .metric {{ text-align: center; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
            .metric h3 {{ margin: 0; color: #666; }}
            .metric .value {{ font-size: 36px; font-weight: bold; }}
            .low {{ color: green; }}
            .moderate {{ color: orange; }}
            .high {{ color: orangered; }}
            .critical-text {{ color: red; }}
        </style>
    </head>
    <body>
    """)
    
    # Header with overall threat level
    level = threat_report.get("overall_threat_level", "UNKNOWN")
    level_class = "critical-text" if level in ("CRITICAL", "HIGH") else "high" if level == "ELEVATED" else "moderate" if level == "GUARDED" else "low"
    
    html_parts.append(f"""
    <div class="header">
        <h1>🛡️ KShield Pulse - Threat Analysis Report</h1>
        <p>Generated: {threat_report.get('computed_at', datetime.utcnow().isoformat())}</p>
        <h2>Overall Threat Level: <span class="{level_class}">{level}</span></h2>
    </div>
    """)
    
    # Priority Alerts
    alerts = threat_report.get("priority_alerts", [])
    if alerts:
        html_parts.append('<div class="section">')
        html_parts.append('<h2>⚠️ Priority Alerts</h2>')
        for alert in alerts:
            html_parts.append(f'<div class="alert-box critical">{alert}</div>')
        html_parts.append('</div>')
    
    # Index Summary
    indices = threat_report.get("indices", {})
    html_parts.append('<div class="section">')
    html_parts.append('<h2>📊 Threat Indices</h2>')
    html_parts.append('<div class="grid">')
    
    index_names = [
        ("polarization", "Polarization Index"),
        ("legitimacy_erosion", "Legitimacy Erosion"),
        ("mobilization_readiness", "Mobilization Readiness"),
        ("elite_cohesion", "Elite Cohesion"),
        ("information_warfare", "Information Warfare"),
        ("security_friction", "Security Friction"),
        ("economic_cascade", "Economic Cascade Risk"),
        ("ethnic_tension", "Ethnic Tension"),
    ]
    
    for key, name in index_names:
        idx = indices.get(key, {})
        value = idx.get("value", idx.get("avg_tension", 0))
        severity = idx.get("severity", "UNKNOWN")
        sev_class = "critical-text" if severity in ("CRITICAL", "IMMINENT") else "high" if severity in ("HIGH", "ELEVATED") else "moderate" if severity == "MODERATE" else "low"
        
        html_parts.append(f"""
        <div class="metric">
            <h3>{name}</h3>
            <div class="value {sev_class}">{value*100:.0f}%</div>
            <div>{severity}</div>
        </div>
        """)
    
    html_parts.append('</div></div>')
    
    # Footer
    html_parts.append("""
    <div class="section">
        <p style="text-align: center; color: #666;">
            KShield Pulse Engine - National Threat Detection System<br>
            © 2025 KShield Kenya
        </p>
    </div>
    </body>
    </html>
    """)
    
    # Write file
    html_content = "\n".join(html_parts)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"Generated HTML report: {output_path}")
    return output_path


# =============================================================================
# Convenience Functions
# =============================================================================

def visualize_threat_report(report: Dict, output_dir: str = ".") -> Dict[str, str]:
    """
    Generate all visualizations for a threat report.
    
    Args:
        report: ThreatIndexReport.to_dict()
        output_dir: Directory for output files
        
    Returns:
        Dict of visualization_name -> file_path
    """
    import os
    outputs = {}
    
    # HTML Report
    html_path = os.path.join(output_dir, "threat_report.html")
    generate_html_report(report, output_path=html_path)
    outputs["html_report"] = html_path
    
    # Dashboard
    if HAS_PLOTLY:
        fig = create_threat_dashboard(report)
        if fig:
            dashboard_path = os.path.join(output_dir, "threat_dashboard.html")
            fig.write_html(dashboard_path)
            outputs["dashboard"] = dashboard_path
    
    # Ethnic tension heatmap
    ethnic = report.get("indices", {}).get("ethnic_tension", {})
    if HAS_PLOTLY and ethnic:
        # Create sample tensions for visualization
        tensions = {
            "kikuyu-luo": 0.7,
            "kikuyu-kalenjin": 0.5,
            "luo-kalenjin": 0.4,
            "luhya-kalenjin": 0.3,
        }
        fig = create_ethnic_tension_chord(tensions)
        if fig:
            ethnic_path = os.path.join(output_dir, "ethnic_tension.html")
            fig.write_html(ethnic_path)
            outputs["ethnic_tension"] = ethnic_path
    
    return outputs
