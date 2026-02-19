"""
K-SHIELD Module — Entry Point

Auth gate → Landing page with 4 sub-cards → Sub-page routing.

Sub-cards:
- Causal Relationships
- Policy Terrain
- Simulations
- Policy Impact
"""

from __future__ import annotations

import streamlit as st
import logging

# Resolve imports relative to the ui/ directory
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import threading
from common.auth import check_access
from common.landing import render_landing

logger = logging.getLogger("sentinel.kshield.page")

_IMPORT_WARMED = False

def _warm_import_cache():
    """Pre-import heavy modules in a background thread so card clicks are instant."""
    # Use session state to ensure we only warm up once per session
    if st.session_state.get("_kshield_warmed", False):
        return
    st.session_state["_kshield_warmed"] = True

    def _bg_imports():
        try:
            import importlib
            import time
            # These are the heaviest imports — warm them into sys.modules
            # Adding sleeps to yield GIL and prevent UI stutter during landing page render
            modules = [
                "kshiked.ui.kshield.causal",
                "kshiked.ui.kshield.terrain",
                "kshiked.ui.kshield.simulation",
                "kshiked.ui.kshield.impact",
            ]
            for mod in modules:
                try:
                    importlib.import_module(mod)
                    time.sleep(0.1) # Yield GIL
                except Exception as e:
                    # Silent fail, will be caught when user actually clicks card
                    pass
        except Exception:
            pass

    t = threading.Thread(target=_bg_imports, daemon=True)
    t.start()


# K-SHIELD sub-card definitions
KSHIELD_CARDS = [
    (
        "CAUSAL RELATIONSHIPS",
        "Explore discovered causal pathways between economic variables. "
        "View the force-directed network graph, Granger causality test results, "
        "and top relationship rankings.",
        "CAUSAL",
    ),
    (
        "POLICY TERRAIN",
        "Visualize the economic stability landscape as a 3D terrain. "
        "Map inflation vs unemployment against instability potential. "
        "See where the economy currently sits in the phase space.",
        "TERRAIN",
    ),
    (
        "SIMULATIONS",
        "Run full economic simulations across sectors. Design shock scenarios, "
        "set policy constraints, and compare outcomes in the 4D State Cube. "
        "Build and save reproducible scenario libraries.",
        "SIMULATION",
    ),
    (
        "POLICY IMPACT",
        "Measure real-time public sentiment on active policies. "
        "Track economic satisfaction by domain, monitor scarcity vectors, "
        "actor stress levels, and social cohesion metrics.",
        "IMPACT",
    ),
]


def render(theme, data=None):
    """
    Main entry point for the K-SHIELD module.
    
    Args:
        theme: ThemeColors instance
        data: DashboardData instance (loaded lazily when needed)
    """
    # Step 1: Access Gate
    if not check_access("K-SHIELD", theme):
        return
    
    # Step 2: Sub-navigation state
    if "kshield_view" not in st.session_state:
        st.session_state.kshield_view = "LANDING"
    
    view = st.session_state.kshield_view
    
    # Step 3: Route
    if view == "LANDING":
        _render_landing(theme)
    elif view == "CAUSAL":
        _render_causal_page(theme, data)
    elif view == "TERRAIN":
        _render_terrain_page(theme, data)
    elif view == "SIMULATION":
        _render_simulation_page(theme, data)
    elif view == "IMPACT":
        _render_impact_page(theme, data)
    else:
        st.session_state.kshield_view = "LANDING"
        st.rerun()


def _get_kshield_data():
    """Load shared dashboard data only for pages that require it."""
    if "kshield_cached_data" in st.session_state:
        return st.session_state["kshield_cached_data"]
    try:
        from kshiked.ui.connector import get_dashboard_data

        force_causal = bool(st.session_state.get("force_causal_retrain", False))
        data = get_dashboard_data(force_causal=force_causal)
        st.session_state["kshield_cached_data"] = data
        return data
    except Exception as exc:
        logger.warning("K-SHIELD data load failed: %s", exc)
        return None


def _render_landing(theme):
    """K-SHIELD landing page with 4 sub-cards."""
    render_landing(
        theme=theme,
        title="K-SHIELD",
        subtitle="Defense Simulation & Causal Intelligence",
        tagline="— Analyze · Simulate · Predict —",
        cards=KSHIELD_CARDS,
        view_prefix="kshield_",
        back_target="HOME",
    )
    # Pre-import heavy modules in background while user reads the landing page
    _warm_import_cache()


def _back_to_kshield(key_suffix: str):
    """Render back button that returns to K-SHIELD landing."""
    if st.button("← Back to K-SHIELD", key=f"back_{key_suffix}"):
        st.session_state.kshield_view = "LANDING"
        st.rerun()


def _render_causal_page(theme, data):
    """Causal Relationships sub-page — self-contained, uses World Bank data."""
    _back_to_kshield("causal")
    
def _render_causal_page(theme, data):
    """Causal Relationships sub-page — self-contained, uses World Bank data."""
    _back_to_kshield("causal")
    
    try:
        from kshiked.ui.kshield.causal import render_causal
        render_causal(theme)
    except Exception as e:
        st.error(f"Causal module error: {e}")
        logger.exception("Causal page error")


def _render_terrain_page(theme, data):
    """Policy Terrain sub-page."""
    _back_to_kshield("terrain")
    
def _render_terrain_page(theme, data):
    """Policy Terrain sub-page."""
    _back_to_kshield("terrain")
    
    try:
        from kshiked.ui.kshield.terrain import render_terrain
        render_terrain(theme, data)
    except Exception as e:
        st.error(f"Terrain module error: {e}")
        logger.exception("Terrain page error")


def _render_simulation_page(theme, data):
    """Simulations sub-page."""
    _back_to_kshield("simulation")
    
def _render_simulation_page(theme, data):
    """Simulations sub-page."""
    _back_to_kshield("simulation")
    
    try:
        from kshiked.ui.kshield.simulation import render_simulation
        render_simulation(theme, data)
    except Exception as e:
        st.error(f"Simulation module error: {e}")
        logger.exception("Simulation page error")


def _render_impact_page(theme, data):
    """Policy Impact sub-page."""
    _back_to_kshield("impact")
    
def _render_impact_page(theme, data):
    """Policy Impact sub-page."""
    _back_to_kshield("impact")
    
    try:
        # Avoid caching the function object to allow hot-reloading
        from kshiked.ui.kshield.impact import render_impact
        
        if data is None:
            data = _get_kshield_data()
        render_impact(theme, data)
    except Exception as e:
        st.error(f"Impact module error: {e}")
        logger.exception("Impact page error")

