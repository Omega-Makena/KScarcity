"""SENTINEL Command Center - main router and entry point."""

from ._shared import (
    st, components, logger,
    DARK_THEME, LIGHT_THEME, generate_css,
    get_dashboard_data, HAS_STREAMLIT,
)
from .live_map import render_live_map_tab
from .executive import render_executive_tab
from .signals import render_signals_tab
from .causal_sim import render_causal_tab, render_simulation_tab
from .escalation import render_escalation_tab
from .federation import render_federation_tab
from .operations import render_operations_tab
from .guide import render_system_guide_tab
from .document_intel import render_document_intel_tab
from .analysis_controls import render_analysis_controls
from .home import render_home
from .policy_chat import render_policy_chat


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
    render_analysis_controls()

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
        "Policy Intelligence": "POLICY_CHAT",
    }

    view_to_name = {v: k for k, v in NAV_OPTIONS.items()}
    current_name = view_to_name.get(st.session_state.current_view, "Home")

    # Sync the radio widget key so card-button navigations are respected
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
        render_home(theme)
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

    if view == "LIVE_MAP":
        render_live_map_tab(data, theme)
    elif view == "EXECUTIVE":
        render_executive_tab(data, theme)
    elif view == "SIGNALS":
        render_signals_tab(data, theme)
    elif view == "CAUSAL":
        render_causal_tab(data, theme)
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
            render_simulation_tab(data, theme)
    elif view == "ESCALATION":
        render_escalation_tab(data, theme)
    elif view == "FEDERATION":
        render_federation_tab(data, theme)
    elif view == "OPERATIONS":
        render_operations_tab(data, theme)
    elif view == "SYSTEM_GUIDE":
        render_system_guide_tab(theme)
    elif view == "DOCS":
        render_document_intel_tab(theme)
    elif view == "POLICY_CHAT":
        render_policy_chat(theme)


def main():
    """Run the SENTINEL dashboard."""
    render_sentinel_dashboard()
