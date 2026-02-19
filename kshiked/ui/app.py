"""
K-Scarcity Dashboard — App Router

Thin router that sets up the page, injects global CSS,
and routes to the appropriate module based on session state.

Run with: streamlit run kshiked/ui/app.py
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

# Ensure ui/ is on the path for relative imports
UI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(UI_DIR))

import streamlit as st
import streamlit.components.v1 as components

from theme import DARK_THEME, LIGHT_THEME, generate_css
from common.css import inject_global_css

logger = logging.getLogger("sentinel.app")


def main():
    """Run the dashboard."""
    st.set_page_config(
        page_title="K-Scarcity",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    
    # Theme state
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True
    
    theme = DARK_THEME if st.session_state.dark_mode else LIGHT_THEME
    
    # Inject all global CSS first (prevents FOUC)
    inject_global_css(theme, st.session_state.dark_mode, generate_css)
    
    # Navigation state is card-driven (no sidebar router).
    if "current_view" not in st.session_state:
        st.session_state.current_view = "HOME"

    # Normalize sub-card routing emitted by landing cards.
    # Example: "kshield_TERRAIN" -> current_view="KSHIELD", kshield_view="TERRAIN"
    cv = str(st.session_state.current_view)
    if cv.startswith("kshield_") and cv != "KSHIELD":
        st.session_state.kshield_view = cv.replace("kshield_", "")
        st.session_state.current_view = "KSHIELD"
        st.rerun()

    view = st.session_state.current_view
    
    # =========================================================================
    # Route to module
    # =========================================================================
    
    if view == "HOME":
        from home.page import render as render_home
        render_home(theme)
    
    elif view == "KSHIELD":
        # Keep K-SHIELD entry fast; sub-pages can lazy-load if needed.
        data = None
        from kshield.page import render as render_kshield
        render_kshield(theme, data)
    
    elif view == "KPULSE":
        # Future: K-PULSE module
        _render_placeholder("K-PULSE", "Signal Intelligence & Early Warning", theme)
    
    elif view == "KCOLLAB":
        # Future: K-COLLAB module
        _render_placeholder("K-COLLAB", "Federated Learning & Collaboration", theme)
    
    elif view == "KEDUCATION":
        # Future: K-EDUCATION module
        _render_placeholder("K-EDUCATION", "Public Knowledge & Explainable Analytics", theme)
    
    else:
        # Legacy tab routing (backwards compatibility with old dashboard)
        _route_legacy(view, theme)


def _load_data():
    """Lazy-load dashboard data."""
    try:
        from kshiked.ui.connector import get_dashboard_data
        
        if "force_causal_retrain" not in st.session_state:
            st.session_state.force_causal_retrain = False
        
        return get_dashboard_data(force_causal=st.session_state.force_causal_retrain)
    except Exception as e:
        logger.warning(f"Data loading failed: {e}")
        return None


def _render_placeholder(title: str, subtitle: str, theme):
    """Render a placeholder for modules not yet extracted."""
    from common.landing import render_landing
    render_landing(
        theme=theme,
        title=title,
        subtitle=subtitle,
        tagline="— Coming Soon —",
        cards=[],
        back_target="HOME",
    )
    
    st.info(f"{title} module is not yet available. Check back soon!")


def _route_legacy(view: str, theme):
    """
    Route to legacy tab views from the old monolith.
    
    This handles direct tab navigation (LIVE_MAP, EXECUTIVE, etc.)
    for backwards compatibility. These will eventually be absorbed
    into their respective card modules.
    """
    try:
        # Fall back to the old monolith for tabs not yet extracted
        from sentinel_dashboard import render_sentinel_dashboard
        render_sentinel_dashboard()
    except Exception as e:
        st.error(f"Legacy route failed: {e}")
        if st.button("← Back to Home"):
            st.session_state.current_view = "HOME"
            st.rerun()


if __name__ == "__main__":
    main()
