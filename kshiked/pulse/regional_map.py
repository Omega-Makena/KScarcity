"""
Kenya Regional Stress Signal Map.

Interactive choropleth map showing stress signals by county.
Uses Folium for map rendering with color-coded risk levels.
"""

from typing import Dict, Optional, Any
import json
from pathlib import Path

try:
    import folium
    from folium.plugins import MarkerCluster
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

from .geo_mapper import KenyaGeoMapper, KENYA_COUNTIES


# Kenya center coordinates and zoom
KENYA_CENTER = (0.0236, 37.9062)
DEFAULT_ZOOM = 6

# Risk level colors
RISK_COLORS = {
    "Low": "#28a745",       # Green
    "Moderate": "#ffc107",  # Yellow
    "High": "#fd7e14",      # Orange
    "Critical": "#dc3545",  # Red
}

# GeoJSON for Kenya counties (simplified inline for 47 counties)
# In production, this would be loaded from file
KENYA_GEOJSON_URL = "https://raw.githubusercontent.com/mikelmaron/kenya-election-data/master/data/counties.geojson"


def get_risk_color(risk_score: float) -> str:
    """Get color based on risk score."""
    if risk_score < 0.3:
        return RISK_COLORS["Low"]
    elif risk_score < 0.5:
        return RISK_COLORS["Moderate"]
    elif risk_score < 0.7:
        return RISK_COLORS["High"]
    else:
        return RISK_COLORS["Critical"]


def create_kenya_map(
    geo_mapper: Optional[KenyaGeoMapper] = None,
    width: str = "100%",
    height: str = "600px",
    use_markers: bool = True,
    use_circles: bool = True,
) -> Optional[Any]:
    """
    Create an interactive Kenya map with stress signal visualization.
    
    Args:
        geo_mapper: KenyaGeoMapper with signal data (creates sample if None)
        width: Map width
        height: Map height
        use_markers: Show marker popups
        use_circles: Show circle markers sized by risk
        
    Returns:
        Folium Map object
    """
    if not FOLIUM_AVAILABLE:
        raise ImportError("Folium not installed. Run: pip install folium")
    
    # Create mapper with sample data if not provided
    if geo_mapper is None:
        geo_mapper = KenyaGeoMapper()
        geo_mapper.inject_sample_data()
    
    # Get county data
    county_data = geo_mapper.get_county_data()
    
    # Create base map
    m = folium.Map(
        location=KENYA_CENTER,
        zoom_start=DEFAULT_ZOOM,
        tiles="cartodbpositron",
        width=width,
        height=height,
    )
    
    # Add county markers
    if use_circles:
        for county, data in county_data.items():
            if data["signal_count"] > 0:
                # Circle radius based on signal count
                radius = max(8, min(30, data["signal_count"] * 5))
                
                # Create popup content
                popup_html = f"""
                <div style="font-family: Arial, sans-serif; min-width: 200px;">
                    <h4 style="margin: 0 0 8px 0; color: #333;">{county} County</h4>
                    <hr style="margin: 4px 0;">
                    <p><b>Risk Level:</b> 
                        <span style="color: {get_risk_color(data['risk_score'])}; font-weight: bold;">
                            {data['risk_level']}
                        </span>
                    </p>
                    <p><b>Risk Score:</b> {data['risk_score']:.2f}</p>
                    <p><b>Signal Count:</b> {data['signal_count']}</p>
                    <p><b>Recent Signals:</b></p>
                    <ul style="margin: 4px 0; padding-left: 20px;">
                        {''.join(f'<li>{s}</li>' for s in data['signals'][-5:])}
                    </ul>
                </div>
                """
                
                folium.CircleMarker(
                    location=[data["lat"], data["lon"]],
                    radius=radius,
                    color=get_risk_color(data["risk_score"]),
                    fill=True,
                    fill_color=get_risk_color(data["risk_score"]),
                    fill_opacity=0.7,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"{county}: {data['risk_level']} ({data['signal_count']} signals)",
                ).add_to(m)
    
    # Add markers for high-risk counties
    if use_markers:
        high_risk = geo_mapper.get_high_risk_counties(0.5)
        for county in high_risk:
            data = county_data[county]
            folium.Marker(
                location=[data["lat"], data["lon"]],
                icon=folium.Icon(
                    color="red" if data["risk_score"] >= 0.7 else "orange",
                    icon="exclamation-triangle",
                    prefix="fa",
                ),
                popup=f"{county}: {data['risk_level']}",
            ).add_to(m)
    
    # Add legend
    legend_html = """
    <div style="
        position: fixed; 
        bottom: 50px; 
        left: 50px; 
        z-index: 1000;
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        font-family: Arial, sans-serif;
    ">
        <h4 style="margin: 0 0 10px 0;">Risk Level</h4>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 20px; height: 20px; background: #28a745; border-radius: 50%; margin-right: 8px;"></div>
            <span>Low (0-30%)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 20px; height: 20px; background: #ffc107; border-radius: 50%; margin-right: 8px;"></div>
            <span>Moderate (30-50%)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 20px; height: 20px; background: #fd7e14; border-radius: 50%; margin-right: 8px;"></div>
            <span>High (50-70%)</span>
        </div>
        <div style="display: flex; align-items: center; margin: 5px 0;">
            <div style="width: 20px; height: 20px; background: #dc3545; border-radius: 50%; margin-right: 8px;"></div>
            <span>Critical (70-100%)</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add title
    title_html = """
    <div style="
        position: fixed; 
        top: 10px; 
        left: 50%; 
        transform: translateX(-50%);
        z-index: 1000;
        background-color: rgba(255,255,255,0.9);
        padding: 10px 20px;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        font-family: Arial, sans-serif;
    ">
        <h3 style="margin: 0; color: #333;">Kenya Stress Signal Map</h3>
        <p style="margin: 5px 0 0 0; color: #666; font-size: 12px;">
            Real-time risk monitoring by county
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m


def save_map(
    geo_mapper: Optional[KenyaGeoMapper] = None,
    output_path: str = "kenya_stress_map.html",
) -> str:
    """
    Save Kenya stress map to HTML file.
    
    Args:
        geo_mapper: KenyaGeoMapper with signal data
        output_path: Path to save HTML file
        
    Returns:
        Path to saved file
    """
    m = create_kenya_map(geo_mapper)
    m.save(output_path)
    return output_path


def get_map_html(geo_mapper: Optional[KenyaGeoMapper] = None) -> str:
    """
    Get Kenya map as HTML string (for Streamlit embedding).
    
    Args:
        geo_mapper: KenyaGeoMapper with signal data
        
    Returns:
        HTML string
    """
    m = create_kenya_map(geo_mapper)
    return m._repr_html_()


# Streamlit integration
def render_in_streamlit(geo_mapper: Optional[KenyaGeoMapper] = None):
    """
    Render map in Streamlit using st.components.v1.html.
    
    Args:
        geo_mapper: KenyaGeoMapper with signal data
    """
    try:
        import streamlit as st
        from streamlit.components.v1 import html
        
        map_html = get_map_html(geo_mapper)
        html(map_html, height=650, scrolling=True)
    except ImportError:
        raise ImportError("Streamlit not installed. Run: pip install streamlit")


if __name__ == "__main__":
    # Demo: Create and save a sample map
    print("Creating Kenya Stress Signal Map...")
    
    mapper = KenyaGeoMapper()
    mapper.inject_sample_data()
    
    output = save_map(mapper, "kenya_stress_map.html")
    print(f"Map saved to: {output}")
    print("Open the file in a browser to view the interactive map.")
