"""
Pulse Engine Dashboard Components

Provides Streamlit-based visualization for:
- Real-time signal intensity
- Signal co-occurrence heatmap
- Risk score timeline
- Alert notifications
- Primitive state display

Usage:
    import streamlit as st
    from kshiked.pulse.dashboard import render_pulse_dashboard
    
    render_pulse_dashboard(sensor, scorer)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np

# Lazy imports for Streamlit to avoid import errors when not using dashboard
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    st = None

from .mapper import SignalID, SIGNAL_CATEGORIES, SignalCategory
from .primitives import PulseState, ResourceDomain, ActorType
from .cooccurrence import RiskScore, RiskScorer, SignalCorrelationMatrix
from .sensor import PulseSensor

logger = logging.getLogger("kshield.pulse.dashboard")


# =============================================================================
# Dashboard State Management
# =============================================================================

def get_dashboard_state() -> Dict[str, Any]:
    """Get or initialize dashboard state in Streamlit session."""
    if not HAS_STREAMLIT:
        return {}
    
    if "pulse_dashboard" not in st.session_state:
        st.session_state.pulse_dashboard = {
            "risk_history": [],
            "signal_history": {},
            "alerts": [],
            "last_update": None,
        }
    
    return st.session_state.pulse_dashboard


def update_dashboard_history(risk_score: RiskScore, max_history: int = 100) -> None:
    """Update dashboard history with new data."""
    state = get_dashboard_state()
    
    # Add risk score to history
    state["risk_history"].append({
        "timestamp": risk_score.timestamp,
        "overall": risk_score.overall,
        "anomaly": risk_score.anomaly_score,
        "trend": risk_score.trend,
        **{f"cat_{k}": v for k, v in risk_score.by_category.items()},
    })
    
    # Trim history
    if len(state["risk_history"]) > max_history:
        state["risk_history"] = state["risk_history"][-max_history:]
    
    # Update signal history
    for signal_id, intensity in risk_score.by_signal.items():
        if signal_id.name not in state["signal_history"]:
            state["signal_history"][signal_id.name] = []
        state["signal_history"][signal_id.name].append({
            "timestamp": risk_score.timestamp,
            "intensity": intensity,
        })
        # Trim
        if len(state["signal_history"][signal_id.name]) > max_history:
            state["signal_history"][signal_id.name] = state["signal_history"][signal_id.name][-max_history:]
    
    state["last_update"] = risk_score.timestamp


# =============================================================================
# Main Dashboard Renderer
# =============================================================================

def render_pulse_dashboard(
    sensor: PulseSensor = None,
    scorer: RiskScorer = None,
    show_controls: bool = True,
) -> None:
    """
    Render the complete Pulse Engine dashboard.
    
    Args:
        sensor: PulseSensor instance
        scorer: RiskScorer instance
        show_controls: Whether to show interactive controls
    """
    if not HAS_STREAMLIT:
        logger.error("Streamlit not installed. Cannot render dashboard.")
        return
    
    st.header("🔴 Pulse Engine - Social Signal Intelligence")
    
    # Top metrics row
    if scorer:
        risk_score = scorer.compute()
        update_dashboard_history(risk_score)
        _render_top_metrics(risk_score)
    else:
        st.info("No scorer provided. Connect a RiskScorer for live metrics.")
        risk_score = None
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Signal Monitor", 
        "🔥 Heatmap", 
        "📈 Timeline",
        "⚡ Primitives"
    ])
    
    with tab1:
        _render_signal_monitor(sensor, risk_score)
    
    with tab2:
        _render_correlation_heatmap(scorer)
    
    with tab3:
        _render_timeline()
    
    with tab4:
        if sensor:
            _render_primitives(sensor.state)
        else:
            st.info("Connect a PulseSensor to view primitive state.")
    
    # Controls sidebar
    if show_controls:
        _render_controls_sidebar(sensor)


# =============================================================================
# Component Renderers
# =============================================================================

def _render_top_metrics(risk_score: RiskScore) -> None:
    """Render top-level risk metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        _metric_card(
            "Overall Risk",
            f"{risk_score.overall:.1%}",
            _risk_color(risk_score.overall),
            delta=risk_score.trend,
        )
    
    with col2:
        _metric_card(
            "Anomaly Score",
            f"{risk_score.anomaly_score:.1%}",
            _risk_color(risk_score.anomaly_score),
        )
    
    with col3:
        top_category = max(risk_score.by_category.items(), key=lambda x: x[1]) if risk_score.by_category else ("None", 0)
        _metric_card(
            "Top Category",
            top_category[0],
            "#ff6b6b",
            subtitle=f"{top_category[1]:.1%} intensity",
        )
    
    with col4:
        _metric_card(
            "Trend",
            risk_score.trend.upper(),
            "#4ecdc4" if risk_score.trend == "falling" else "#ff6b6b" if risk_score.trend == "rising" else "#95a5a6",
        )


def _metric_card(title: str, value: str, color: str, delta: str = None, subtitle: str = None) -> None:
    """Render a styled metric card."""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}22, {color}11);
        border-left: 4px solid {color};
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    ">
        <div style="font-size: 0.8rem; color: #888;">{title}</div>
        <div style="font-size: 1.8rem; font-weight: bold; color: {color};">{value}</div>
        {f'<div style="font-size: 0.7rem; color: #666;">{subtitle}</div>' if subtitle else ''}
        {f'<div style="font-size: 0.7rem; color: #666;">↑ {delta}</div>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)


def _risk_color(value: float) -> str:
    """Get color based on risk level."""
    if value < 0.3:
        return "#2ecc71"  # Green
    elif value < 0.5:
        return "#f39c12"  # Yellow
    elif value < 0.7:
        return "#e67e22"  # Orange
    else:
        return "#e74c3c"  # Red


def _render_signal_monitor(sensor: PulseSensor, risk_score: RiskScore) -> None:
    """Render signal intensity monitor."""
    st.subheader("Signal Intensities")
    
    if not risk_score:
        st.info("No risk data available.")
        return
    
    # Group by category
    for category in SignalCategory:
        signals = [s for s, c in SIGNAL_CATEGORIES.items() if c == category]
        
        if not signals:
            continue
        
        with st.expander(f"📌 {category.name}", expanded=True):
            cols = st.columns(len(signals))
            
            for i, signal_id in enumerate(signals):
                intensity = risk_score.by_signal.get(signal_id, 0.0)
                with cols[i]:
                    _render_signal_gauge(signal_id.name, intensity)


def _render_signal_gauge(name: str, value: float) -> None:
    """Render a small signal intensity gauge."""
    # Short name
    short_name = name.replace("_", " ").title()[:20]
    
    color = _risk_color(value)
    
    st.markdown(f"""
    <div style="text-align: center; padding: 0.5rem;">
        <div style="font-size: 0.7rem; color: #666; margin-bottom: 4px;">{short_name}</div>
        <div style="
            width: 100%;
            height: 8px;
            background: #eee;
            border-radius: 4px;
            overflow: hidden;
        ">
            <div style="
                width: {value * 100}%;
                height: 100%;
                background: {color};
                border-radius: 4px;
            "></div>
        </div>
        <div style="font-size: 0.8rem; font-weight: bold; color: {color};">{value:.0%}</div>
    </div>
    """, unsafe_allow_html=True)


def _render_correlation_heatmap(scorer: RiskScorer) -> None:
    """Render signal correlation heatmap."""
    st.subheader("Signal Co-occurrence Heatmap")
    
    if not scorer or not hasattr(scorer, 'correlation'):
        st.info("Connect a RiskScorer with correlation tracking to view heatmap.")
        return
    
    matrix = scorer.correlation.get_matrix()
    
    if matrix.sum() == 0:
        st.info("No co-occurrence data yet. Process more signals to build correlations.")
        return
    
    # Create labels
    labels = [s.name[:15] for s in SignalID]
    
    # Simple heatmap using markdown table (Streamlit-native)
    st.write("**Correlation Matrix** (Jaccard Similarity)")
    
    # Top correlations
    top_corrs = scorer.correlation.get_top_correlations(5)
    if top_corrs:
        st.write("**Top Correlated Pairs:**")
        for s1, s2, corr in top_corrs:
            st.write(f"- {s1.name} ↔ {s2.name}: {corr:.2f}")


def _render_timeline() -> None:
    """Render risk score timeline."""
    st.subheader("Risk Timeline")
    
    state = get_dashboard_state()
    history = state.get("risk_history", [])
    
    if len(history) < 2:
        st.info("Collecting data for timeline... (need at least 2 data points)")
        return
    
    # Simple line chart
    import pandas as pd
    
    df = pd.DataFrame(history)
    if 'timestamp' in df.columns:
        df['time'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('time')
    
    st.line_chart(df[['overall', 'anomaly']])
    
    # Category breakdown
    cat_cols = [c for c in df.columns if c.startswith('cat_')]
    if cat_cols:
        st.write("**By Category:**")
        st.line_chart(df[cat_cols])


def _render_primitives(state: PulseState) -> None:
    """Render primitive state display."""
    st.subheader("Primitive State")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Scarcity Vector**")
        for domain in ResourceDomain:
            value = state.scarcity.get(domain)
            st.progress(value, text=f"{domain.value}: {value:.0%}")
        
        st.metric("Aggregate Scarcity", f"{state.scarcity.aggregate_score():.1%}")
    
    with col2:
        st.write("**Actor Stress**")
        for actor in ActorType:
            stress = state.stress.get_stress(actor)
            # Stress is [-1, 1], convert to display
            display = (stress + 1) / 2  # Now [0, 1]
            color = "red" if stress < 0 else "green"
            st.progress(display, text=f"{actor.value}: {stress:+.2f}")
        
        st.metric("System Stress", f"{state.stress.total_system_stress():.2f}")
    
    # Bond strength
    st.write("**Social Cohesion**")
    cols = st.columns(4)
    with cols[0]:
        st.metric("National", f"{state.bonds.national_cohesion:.0%}")
    with cols[1]:
        st.metric("Class", f"{state.bonds.class_solidarity:.0%}")
    with cols[2]:
        st.metric("Regional", f"{state.bonds.regional_unity:.0%}")
    with cols[3]:
        st.metric("Fragility", f"{state.bonds.fragility_score():.0%}")
    
    # Risk summary
    state.compute_risk_metrics()
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Instability Index", f"{state.instability_index:.1%}")
    with col2:
        st.metric("Crisis Probability", f"{state.crisis_probability:.1%}")


def _render_controls_sidebar(sensor: PulseSensor) -> None:
    """Render control sidebar."""
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Test input
        st.subheader("Test Signal Detection")
        test_text = st.text_area("Enter test text:", height=100)
        if st.button("Analyze"):
            if sensor and test_text:
                detections = sensor.process_text(test_text)
                if detections:
                    st.success(f"Detected {len(detections)} signals:")
                    for d in detections:
                        st.write(f"- {d.signal_id.name}: {d.intensity:.0%}")
                else:
                    st.info("No signals detected.")
        
        # NLP toggle
        st.subheader("Settings")
        if sensor:
            if st.button("Upgrade to NLP Detectors"):
                sensor.upgrade_to_nlp()
                st.success("Upgraded to NLP detectors!")
        
        # Metrics
        st.subheader("Sensor Metrics")
        if sensor:
            metrics = sensor.get_metrics()
            st.json(metrics)


# =============================================================================
# Standalone Dashboard Runner
# =============================================================================

def run_dashboard(
    sensor: PulseSensor = None,
    scorer: RiskScorer = None,
    port: int = 8501,
) -> None:
    """
    Run the dashboard as a standalone Streamlit app.
    
    Note: This should be called from a separate script, not imported.
    """
    if not HAS_STREAMLIT:
        print("Streamlit not installed. Run: pip install streamlit")
        return
    
    # The actual rendering happens when Streamlit runs the script
    render_pulse_dashboard(sensor, scorer)
