"""
Main Layout and Chat Interface for Policy Intelligence.
"""
from __future__ import annotations
import streamlit as st
import urllib.request

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from .context import init_session_state, handle_user_input
from .metrics import render_sidebar
from .live_policy import render_live_policy_impact

def render_policy_chat_interface(theme, sidebar_enabled=True, show_title=True, data=None):
    """Reusable policy chat interface component."""
    init_session_state()

    # Sidebar controls & Metrics
    if sidebar_enabled:
        render_sidebar(theme, data)

    # Main area header
    if show_title:
        st.markdown(
            '<div class="section-header">POLICY INTELLIGENCE</div>',
            unsafe_allow_html=True,
        )

    # Existing Policy Impact card now includes live baseline/counterfactual trajectories.
    render_live_policy_impact(theme)

    # Ollama status indicator
    _render_status_bar()

    # Chat history
    _render_chat_history(theme)

    # Chat input — always visible at bottom
    user_input = st.chat_input("Ask about any Kenyan policy or bill...")
    if user_input:
        handle_user_input(user_input, theme)

def _render_status_bar():
    """Show a small status indicator for Ollama connectivity."""
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        st.caption("Ollama: **connected** (qwen2.5:3b + nomic-embed-text)")
    except Exception:
        st.warning(
            "Ollama is not reachable at localhost:11434. "
            "Start it with `ollama serve` for live analysis."
        )

def _render_chat_history(theme):
    """Render all chat messages."""
    for msg in st.session_state.policy_chat_messages:
        role = msg["role"]
        content = msg["content"]

        with st.chat_message(role):
            st.markdown(content, unsafe_allow_html=True)

            # Render inline visualizations if present
            metadata = msg.get("metadata", {})
            if metadata.get("type") == "bill_analysis":
                _render_inline_visuals(metadata, theme)

def _render_inline_visuals(metadata: dict, theme):
    """Render county risk chart and provision severity chart from stored metadata."""
    county_risks = metadata.get("county_risks", {})
    provisions_data = metadata.get("provisions_data", [])

    # County risk chart
    if county_risks and HAS_PLOTLY:
        sorted_counties = sorted(
            county_risks.items(), key=lambda x: x[1], reverse=True
        )[:15]
        if sorted_counties:
            counties, risks = zip(*sorted_counties)
            colors = [
                theme.accent_danger if r >= 0.7 else
                theme.accent_warning if r >= 0.4 else
                theme.accent_success
                for r in risks
            ]
            fig = go.Figure(go.Bar(
                x=list(risks),
                y=list(counties),
                orientation="h",
                marker_color=colors,
                text=[f"{r:.0%}" for r in risks],
                textposition="outside",
            ))
            fig.update_layout(
                title="County Risk Assessment",
                xaxis_title="Risk Score",
                yaxis=dict(autorange="reversed"),
                height=400,
                margin=dict(l=10, r=10, t=40, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=theme.text_primary, size=11),
                xaxis=dict(range=[0, 1.1]),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Provision severity chart
    if provisions_data and HAS_PLOTLY:
        labels = [p["clause_id"] for p in provisions_data[:8]]
        severities = [p["severity"] for p in provisions_data[:8]]
        colors = [
            theme.accent_danger if s >= 0.8 else
            theme.accent_warning if s >= 0.5 else
            theme.accent_success
            for s in severities
        ]
        fig = go.Figure(go.Bar(
            x=labels,
            y=severities,
            marker_color=colors,
            text=[f"{s:.0%}" for s in severities],
            textposition="outside",
            ))
        fig.update_layout(
            title="Provision Severity",
            yaxis_title="Severity",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=theme.text_primary, size=11),
            yaxis=dict(range=[0, 1.1]),
        )
        st.plotly_chart(fig, use_container_width=True)
