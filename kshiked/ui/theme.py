"""
SENTINEL Dashboard Theme System

Premium dark/light theme configuration with glassmorphism effects,
gradient color schemes, and micro-animations.
"""

from dataclasses import dataclass
from typing import Dict

# =============================================================================
# Theme Configuration
# =============================================================================

@dataclass
class ThemeColors:
    """Color palette for a theme."""
    bg_primary: str
    bg_secondary: str
    bg_tertiary: str
    bg_card: str
    bg_card_hover: str
    text_primary: str
    text_secondary: str
    text_muted: str
    accent_primary: str
    accent_success: str
    accent_warning: str
    accent_danger: str
    accent_critical: str
    accent_info: str
    border_default: str
    border_subtle: str
    shadow: str
    gradient_primary: str
    gradient_danger: str


DARK_THEME = ThemeColors(
    bg_primary="#020a06", # Almost black, very subtle green
    bg_secondary="#1f331d", # Darker Moss (was #395935)
    bg_tertiary="#142113",
    bg_card="rgba(31, 51, 29, 0.4)", # Subtle Glassy Dark Moss
    bg_card_hover="rgba(31, 51, 29, 0.6)",
    text_primary="#ffffff",
    text_secondary="#aabfac",
    text_muted="#6e8a70",
    accent_primary="#00ff88",
    accent_success="#00ff88",
    accent_warning="#f5d547", # Neon Gold
    accent_danger="#ff3366",
    accent_critical="#ff0044",
    accent_info="#00aaff",
    border_default="rgba(255, 255, 255, 0.1)",
    border_subtle="rgba(255, 255, 255, 0.05)",
    shadow="0 8px 32px rgba(0, 0, 0, 0.6)",
    gradient_primary="linear-gradient(135deg, #0d2116 0%, #020a06 100%)", # Deep Forest -> Black
    gradient_danger="linear-gradient(135deg, #ff3366 0%, #ff6b35 100%)",
)


LIGHT_THEME = ThemeColors(
    bg_primary="#f5f7fa",
    bg_secondary="#ffffff",
    bg_tertiary="#eef1f5",
    bg_card="rgba(255, 255, 255, 0.95)",
    bg_card_hover="rgba(245, 247, 250, 1.0)",
    text_primary="#1a1a2e",
    text_secondary="#2d2d44",
    text_muted="#6c757d",
    accent_primary="#0066cc",
    accent_success="#28a745",
    accent_warning="#ffc107",
    accent_danger="#dc3545",
    accent_critical="#c82333",
    accent_info="#17a2b8",
    border_default="rgba(0, 0, 0, 0.1)",
    border_subtle="rgba(0, 0, 0, 0.05)",
    shadow="0 4px 16px rgba(0, 0, 0, 0.08)",
    gradient_primary="linear-gradient(135deg, #0066cc 0%, #17a2b8 100%)",
    gradient_danger="linear-gradient(135deg, #dc3545 0%, #fd7e14 100%)",
)


# =============================================================================
# Threat Level Configuration
# =============================================================================

THREAT_LEVELS = {
    "STABLE": {"color": "#00ff88", "label": "Stable", "bg": "rgba(0, 255, 136, 0.1)"},
    "GUARDED": {"color": "#7ed957", "label": "Guarded", "bg": "rgba(126, 217, 87, 0.1)"},
    "ELEVATED": {"color": "#f5d547", "label": "Elevated", "bg": "rgba(245, 213, 71, 0.1)"},
    "HIGH": {"color": "#ff6b35", "label": "High", "bg": "rgba(255, 107, 53, 0.1)"},
    "CRITICAL": {"color": "#ff0044", "label": "Critical", "bg": "rgba(255, 0, 68, 0.15)"},
}


# =============================================================================
# CSS Generator
# =============================================================================

def generate_css(theme: ThemeColors, is_dark: bool = True) -> str:
    """Generate complete CSS for the dashboard theme."""
    
    mode_class = "dark-mode" if is_dark else "light-mode"
    
    return f"""
<style>
    /* =========================================
       SENTINEL Premium Theme - {'Dark' if is_dark else 'Light'} Mode
       ========================================= */
    
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
    
    /* Root Variables */
    :root {{
        --bg-primary: {theme.bg_primary};
        --bg-secondary: {theme.bg_secondary};
        --bg-tertiary: {theme.bg_tertiary};
        --bg-card: {theme.bg_card};
        --bg-card-hover: {theme.bg_card_hover};
        --text-primary: {theme.text_primary};
        --text-secondary: {theme.text_secondary};
        --text-muted: {theme.text_muted};
        --accent-primary: {theme.accent_primary};
        --accent-success: {theme.accent_success};
        --accent-warning: {theme.accent_warning};
        --accent-danger: {theme.accent_danger};
        --accent-critical: {theme.accent_critical};
        --accent-info: {theme.accent_info};
        --border-default: {theme.border_default};
        --border-subtle: {theme.border_subtle};
        --shadow: {theme.shadow};
        --gradient-primary: {theme.gradient_primary};
        --gradient-danger: {theme.gradient_danger};
    }}
    
    /* Base App Styles */
    .stApp {{
        background: var(--bg-primary);
        color: var(--text-primary);
        font-family: 'Space Mono', 'Courier New', monospace;
    }}
    
    /* Hide Streamlit Branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    
    /* =========================================
       Header Component
       ========================================= */
    
    .sentinel-header {{
        background: {theme.gradient_primary};
        padding: 1.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow);
        position: relative;
        overflow: hidden;
    }}
    
    .sentinel-header::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
        animation: shimmer 3s infinite;
    }}
    
    @keyframes shimmer {{
        0% {{ transform: translateX(-100%); }}
        100% {{ transform: translateX(100%); }}
    }}
    
    .sentinel-header h1 {{
        margin: 0;
        font-size: 1.75rem;
        font-weight: 700;
        color: white;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }}
    
    .sentinel-header p {{
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        color: white;
    }}
    
    /* =========================================
       Glassmorphism Cards
       ========================================= */
    
    .glass-card {{
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border-default);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: var(--shadow);
        transition: all 0.3s ease;
    }}
    
    .glass-card:hover {{
        background: var(--bg-card-hover);
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5);
    }}
    
    .glass-card-sm {{
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid var(--border-default);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: var(--shadow);
        transition: all 0.2s ease;
    }}
    
    /* =========================================
       Metric Cards
       ========================================= */
    
    .metric-card {{
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-default);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }}
    
    .metric-card h3 {{
        font-size: 0.8rem;
        font-weight: 500;
        color: var(--text-muted);
        margin: 0 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    .metric-card .value {{
        font-family: 'Space Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.2;
    }}
    
    .metric-card .delta {{
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }}
    
    .delta-up {{ color: var(--accent-success); }}
    .delta-down {{ color: var(--accent-danger); }}
    .delta-neutral {{ color: var(--text-muted); }}
    
    /* =========================================
       Traffic Light / Status Indicator
       ========================================= */
    
    .status-indicator {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 2rem;
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-default);
        border-radius: 16px;
    }}
    
    .status-dot {{
        width: 80px;
        height: 80px;
        border-radius: 50%;
        margin-bottom: 1rem;
        animation: pulse-glow 2s infinite;
    }}
    
    @keyframes pulse-glow {{
        0%, 100% {{ box-shadow: 0 0 20px currentColor, 0 0 40px currentColor; opacity: 1; }}
        50% {{ box-shadow: 0 0 30px currentColor, 0 0 60px currentColor; opacity: 0.8; }}
    }}
    
    .status-label {{
        font-size: 1.5rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    .status-sublabel {{
        font-size: 0.85rem;
        color: var(--text-muted);
        margin-top: 0.5rem;
    }}
    
    /* =========================================
       Section Headers
       ========================================= */
    
    .section-header {{
        font-size: 1rem;
        font-weight: 600;
        color: var(--text-secondary);
        padding: 0.75rem 0;
        border-bottom: 2px solid var(--border-default);
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    /* =========================================
       Alert Styles
       ========================================= */
    
    .alert {{
        padding: 1rem 1.25rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }}
    
    .alert-critical {{
        background: rgba(255, 0, 68, 0.15);
        border-color: var(--accent-critical);
        color: var(--accent-critical);
    }}
    
    .alert-danger {{
        background: rgba(255, 51, 102, 0.1);
        border-color: var(--accent-danger);
    }}
    
    .alert-warning {{
        background: rgba(255, 204, 0, 0.1);
        border-color: var(--accent-warning);
    }}
    
    .alert-info {{
        background: rgba(0, 170, 255, 0.1);
        border-color: var(--accent-info);
    }}
    
    .alert-success {{
        background: rgba(0, 255, 136, 0.1);
        border-color: var(--accent-success);
    }}
    
    /* =========================================
       Agency Cards
       ========================================= */
    
    .agency-card {{
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-default);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: all 0.2s ease;
    }}
    
    .agency-card.active {{
        border-color: var(--accent-success);
        background: rgba(0, 255, 136, 0.05);
    }}
    
    .agency-card .name {{
        font-weight: 600;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }}
    
    .agency-card .status {{
        font-size: 0.75rem;
        color: var(--text-muted);
    }}
    
    /* =========================================
       Tab Styling Override
       ========================================= */
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
        background: var(--bg-tertiary);
        border-radius: 12px;
        padding: 4px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        color: var(--text-muted);
        border-radius: 8px;
        padding: 0.75rem 1.25rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        background: var(--bg-card);
        color: var(--text-primary);
    }}
    
    .stTabs [aria-selected="true"] {{
        background: var(--bg-card) !important;
        color: var(--accent-primary) !important;
        border-bottom: none !important;
    }}
    
    /* =========================================
       Data Table Styling
       ========================================= */
    
    .stDataFrame {{
        background: var(--bg-card);
        border-radius: 12px;
        overflow: hidden;
    }}
    
    .stDataFrame [data-testid="stDataFrameResizable"] {{
        background: var(--bg-card);
    }}
    
    /* =========================================
       Plotly Chart Container
       ========================================= */
    
    .js-plotly-plot {{
        border-radius: 12px;
        overflow: hidden;
    }}
    
    /* =========================================
       Live Counter Animation
       ========================================= */
    
    .live-counter {{
        font-family: 'Space Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--accent-primary);
        text-shadow: 0 0 20px var(--accent-primary);
    }}
    
    .live-label {{
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    /* =========================================
       Threat Map Overlay
       ========================================= */
    
    .map-overlay {{
        position: absolute;
        top: 1rem;
        left: 1rem;
        background: var(--bg-card);
        backdrop-filter: blur(12px);
        border: 1px solid var(--border-default);
        border-radius: 12px;
        padding: 1rem;
        z-index: 1000;
    }}
    
    .map-legend {{
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }}
    
    .legend-item {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.8rem;
    }}
    
    .legend-dot {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }}
    
    /* =========================================
       Escalation Path Tree
       ========================================= */
    
    .escalation-node {{
        background: var(--bg-card);
        border: 2px solid var(--border-default);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        transition: all 0.2s ease;
    }}
    
    .escalation-node.active {{
        border-color: var(--accent-warning);
        background: rgba(255, 204, 0, 0.1);
    }}
    
    .escalation-node.critical {{
        border-color: var(--accent-danger);
        background: rgba(255, 51, 102, 0.1);
    }}
    
    /* =========================================
       Scrollbar Styling
       ========================================= */
    
    ::-webkit-scrollbar {{
        width: 8px;
        height: 8px;
    }}
    
    ::-webkit-scrollbar-track {{
        background: var(--bg-tertiary);
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb {{
        background: var(--text-muted);
        border-radius: 4px;
    }}
    
    ::-webkit-scrollbar-thumb:hover {{
        background: var(--text-secondary);
    }}
    
    /* =========================================
       Utility Classes
       ========================================= */
    
    .mono {{ font-family: 'Space Mono', monospace; }}
    .text-primary {{ color: var(--text-primary); }}
    .text-secondary {{ color: var(--text-secondary); }}
    .text-muted {{ color: var(--text-muted); }}
    .text-success {{ color: var(--accent-success); }}
    .text-warning {{ color: var(--accent-warning); }}
    .text-danger {{ color: var(--accent-danger); }}
    .text-info {{ color: var(--accent-info); }}
    
    .bg-success {{ background: var(--accent-success); }}
    .bg-warning {{ background: var(--accent-warning); }}
    .bg-danger {{ background: var(--accent-danger); }}
    .bg-critical {{ background: var(--accent-critical); }}
    
    /* =========================================
       Button Styling (Card-Like Buttons)
       ========================================= */
       
    div.stButton > button {{
        width: 100%;
        text-align: left;
        background: var(--bg-card);
        border: 1px solid var(--border-default);
        color: var(--text-primary);
        padding: 1.5rem;
        border-radius: 12px;
        transition: all 0.3s ease;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: var(--shadow);
        height: auto;
        white-space: pre-wrap; /* Allow multiline text */
        line-height: 1.4;
    }}

    div.stButton > button:hover {{
        background: var(--bg-card-hover);
        border-color: var(--accent-primary);
        color: var(--text-primary);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 255, 136, 0.15);
    }}
    
    div.stButton > button:active {{
        box-shadow: none;
        transform: translateY(0);
    }}
    
    div.stButton > button p {{
        font-family: 'Space Mono', monospace;
    }}
    
</style>
"""



def get_plotly_theme(theme: ThemeColors, is_dark: bool = True) -> Dict:
    """Get Plotly layout defaults for the current theme."""
    return {
        "paper_bgcolor": "rgba(0,0,0,0)" if is_dark else theme.bg_secondary,
        "plot_bgcolor": "rgba(0,0,0,0)" if is_dark else theme.bg_tertiary,
        "font": {
            "family": "Space Mono, monospace",
            "color": theme.text_primary,
        },
        "colorway": [
            theme.accent_primary,
            theme.accent_info,
            theme.accent_success,
            theme.accent_warning,
            theme.accent_danger,
        ],
        "xaxis": {
            "gridcolor": theme.border_subtle,
            "linecolor": theme.border_default,
            "tickfont": {"color": theme.text_muted},
        },
        "yaxis": {
            "gridcolor": theme.border_subtle,
            "linecolor": theme.border_default,
            "tickfont": {"color": theme.text_muted},
        },
    }
