"""Escalation Pathways tab."""

from ._shared import st, go, HAS_PLOTLY, DashboardData


def render_escalation_tab(data: DashboardData, theme):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="section-header">ESCALATION PATHWAY TREE</div>', unsafe_allow_html=True)
        _render_escalation_tree(theme)
    with col2:
        st.markdown('<div class="section-header">DECISION LATENCY</div>', unsafe_allow_html=True)
        _render_decision_countdown(data, theme)
        st.markdown('<div class="section-header" style="margin-top: 1rem;">FRAGILITY INDEX</div>', unsafe_allow_html=True)
        _render_fragility_gauge(theme)


def _render_escalation_tree(theme):
    st.info("Escalation pathway data unavailable.")


def _render_decision_countdown(data: DashboardData, theme):
    hours = data.time_to_escalation
    st.markdown(f"""
    <div class="glass-card" style="text-align: center;">
        <div class="live-label">TIME TO DECISION POINT</div>
        <div class="live-counter" style="color: {theme.accent_warning if hours < 24 else theme.accent_primary};">
            {hours:.0f}h
        </div>
        <div style="color: {theme.text_muted}; font-size: 0.85rem; margin-top: 0.5rem;">
            {'URGENT' if hours < 24 else 'MONITORING'}
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_fragility_gauge(theme):
    if not HAS_PLOTLY:
        return
    st.info("Fragility index unavailable.")
    return
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=0,
        number={'suffix': "%", 'font': {'size': 24, 'color': theme.text_primary}},
        title={'text': "System Brittleness", 'font': {'size': 11, 'color': theme.text_muted}},
        gauge={'axis': {'range': [0, 100], 'visible': False}, 'bar': {'color': theme.accent_warning}, 'bgcolor': theme.bg_tertiary},
    ))
    fig.update_layout(height=180, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
