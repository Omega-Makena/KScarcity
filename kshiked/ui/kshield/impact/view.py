"""
K-SHIELD: Policy Impact Sub-page

Real-time policy sentiment, scarcity vectors, actor stress, and social cohesion.
Extracted from sentinel_dashboard.py.
"""

from __future__ import annotations

import streamlit as st

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def render_impact(theme, data):
    """Render the policy impact page."""

    # Pre-warm Ollama model in the background so it's loaded by the time
    # the user types their first query (eliminates cold-start delay).
    _ensure_ollama_warm()

    # Redesigned: No tabs, Chat is main view, Metrics moved to sidebar (via data arg)
    try:
        # Explicit absolute import to avoid confusion
        from kshiked.ui.kshield.impact.components.layout import render_policy_chat_interface
        # Pass data to the chat interface so it can render metrics in side panel
        render_policy_chat_interface(theme, sidebar_enabled=True, show_title=True, data=data)
    except ImportError as e:
        st.error(f"Policy Chat module could not be loaded: {e}")


def _ensure_ollama_warm():
    """Fire a background warm-up request once per session so the LLM model
    is already loaded into memory before the user sends a query."""
    if st.session_state.get("_ollama_warmed"):
        return
    st.session_state["_ollama_warmed"] = True
    try:
        import threading
        import urllib.request
        import json

        def _warm():
            """Send tiny requests to pre-load both models into memory."""
            try:
                # Warm the generation model (qwen2.5:3b)
                payload = json.dumps({
                    "model": "qwen2.5:3b",
                    "prompt": "hi",
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {"num_predict": 1},
                }).encode()
                req = urllib.request.Request(
                    "http://localhost:11434/api/generate",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=60)
            except Exception:
                pass
            try:
                # Warm the embedding model (nomic-embed-text)
                payload = json.dumps({
                    "model": "nomic-embed-text",
                    "input": "warm up",
                    "keep_alive": "30m",
                }).encode()
                req = urllib.request.Request(
                    "http://localhost:11434/api/embed",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=30)
            except Exception:
                pass

        threading.Thread(target=_warm, daemon=True).start()
    except Exception:
        pass


def _render_system_primitives(data, theme):
    """Render system primitives — scarcity, stress, bonds, risk metrics."""
    prims = data.primitives
    if not prims:
        st.info("System primitives unavailable.")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**Scarcity Vector** (Aggregate: {prims.get('aggregate_scarcity', 0):.0%})")
        for domain, value in prims.get("scarcity", {}).items():
            color = theme.accent_danger if value > 0.6 else theme.accent_warning if value > 0.3 else theme.accent_success
            st.markdown(f"""
            <div style="margin-bottom: 0.4rem;">
                <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:{theme.text_muted};">
                    <span>{domain.title()}</span>
                    <span style="color:{color}; font-weight:600;">{value:.0%}</span>
                </div>
                <div style="background:{theme.bg_tertiary}; border-radius:4px; height:6px; overflow:hidden;">
                    <div style="background:{color}; width:{value*100}%; height:100%; border-radius:4px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"**Actor Stress** (Total: {prims.get('total_stress', 0):.2f})")
        for actor, stress in prims.get("stress", {}).items():
            indicator = "↓" if stress < 0 else "↑" if stress > 0 else "─"
            color = theme.accent_danger if stress > 0.3 else theme.accent_warning if stress > 0 else theme.accent_success
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:0.3rem 0; border-bottom:1px solid {theme.border_subtle};">
                <span style="color:{theme.text_muted}; font-size:0.85rem;">{actor.title()}</span>
                <span style="color:{color}; font-weight:600;">{indicator} {stress:+.2f}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("**Social Cohesion**")
        bonds = prims.get("bonds", {})
        for label, key in [("National", "national_cohesion"), ("Class", "class_solidarity"),
                           ("Regional", "regional_unity"), ("Fragility", "fragility")]:
            val = bonds.get(key, 0)
            st.metric(label, f"{val:.0%}")
        
        st.markdown("---")
        inst = prims.get("instability_index", 0)
        crisis = prims.get("crisis_probability", 0)
        i_color = theme.accent_danger if inst > 0.5 else theme.accent_warning if inst > 0.3 else theme.accent_success
        c_color = theme.accent_danger if crisis > 0.3 else theme.accent_warning if crisis > 0.15 else theme.accent_success
        st.markdown(f"""
        <div class="glass-card" style="text-align:center; padding:1rem;">
            <div style="font-size:0.8rem; color:{theme.text_muted};">Instability Index</div>
            <div style="font-size:1.8rem; font-weight:700; color:{i_color};">{inst:.1%}</div>
            <div style="font-size:0.8rem; color:{theme.text_muted}; margin-top:0.5rem;">Crisis Probability</div>
            <div style="font-size:1.8rem; font-weight:700; color:{c_color};">{crisis:.1%}</div>
        </div>
        """, unsafe_allow_html=True)


def _render_economic_indicators(data, theme):
    """Render Economic Satisfaction Index by domain."""
    if not HAS_PLOTLY:
        return
    
    esi = data.esi_indicators
    if not esi:
        st.info("ESI data unavailable.")
        return
    
    colors = [theme.accent_success if v > 0.5 else theme.accent_danger for v in esi.values()]
    
    fig = go.Figure(go.Bar(
        x=list(esi.keys()),
        y=[v * 100 for v in esi.values()],
        marker_color=colors,
        text=[f"{v:.0%}" for v in esi.values()],
        textposition='outside',
        textfont={'color': theme.text_primary},
    ))
    
    fig.add_hline(y=50, line_dash="dash", line_color=theme.text_muted, annotation_text="Threshold")
    
    fig.update_layout(
        height=250,
        yaxis_title="Satisfaction %",
        yaxis_range=[0, 100],
        margin=dict(l=40, r=10, t=10, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': theme.text_muted},
    )
    
    st.plotly_chart(fig, use_container_width=True)
