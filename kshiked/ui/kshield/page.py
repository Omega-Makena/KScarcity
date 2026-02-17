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

from common.auth import check_access
from common.landing import render_landing
from common.nav import render_back_button

logger = logging.getLogger("sentinel.kshield.page")


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
        from data_connector import get_dashboard_data

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
    
    # Handle sub-card clicks (the landing page sets current_view to kshield_CAUSAL etc.)
    # We need to intercept and route to kshield_view instead
    cv = st.session_state.get("current_view", "")
    if cv.startswith("kshield_") and cv != "KSHIELD":
        sub = cv.replace("kshield_", "")
        st.session_state.kshield_view = sub
        st.session_state.current_view = "KSHIELD"
        st.rerun()


def _back_to_kshield(key_suffix: str):
    """Render back button that returns to K-SHIELD landing."""
    if st.button("← Back to K-SHIELD", key=f"back_{key_suffix}"):
        st.session_state.kshield_view = "LANDING"
        st.rerun()


def _render_causal_page(theme, data):
    """Causal Relationships sub-page — self-contained, uses World Bank data."""
    _back_to_kshield("causal")
    
    from kshield.causal import render_causal
    render_causal(theme)  # no data arg needed — loads CSV internally


def _render_terrain_page(theme, data):
    """Policy Terrain sub-page."""
    _back_to_kshield("terrain")
    
    from kshield.terrain import render_terrain
    render_terrain(theme, data)


def _render_simulation_page(theme, data):
    """Simulations sub-page."""
    _back_to_kshield("simulation")
    
    from kshield.simulation import render_simulation
    if data is None:
        with st.spinner("Loading simulation data..."):
            data = _get_kshield_data()
    render_simulation(theme, data)


def _render_impact_page(theme, data):
    """Policy Impact sub-page."""
    _back_to_kshield("impact")
    
    from kshield.impact import render_impact
    if data is None:
        with st.spinner("Loading impact data..."):
            data = _get_kshield_data()
    render_impact(theme, data)
