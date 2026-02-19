"""Executive Overview tab."""

from ._shared import st, go, make_subplots, HAS_PLOTLY, DashboardData, THREAT_LEVELS
from .widgets import render_threat_index_gauges, render_ethnic_tension_heatmap


def render_executive_tab(data: DashboardData, theme):
    """Executive overview with traffic light, gauge, top threats."""
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        _render_traffic_light(data, theme)
    with col2:
        _render_escalation_gauge(data, theme)
    with col3:
        st.markdown('<div class="section-header">TOP THREATS</div>', unsafe_allow_html=True)
        _render_top_threats(data, theme)

    st.markdown("---")
    _render_unknown_detection(data, theme)

    st.markdown('<div class="section-header">COMPETING HYPOTHESES</div>', unsafe_allow_html=True)
    _render_competing_hypotheses(data, theme)

    st.markdown("---")
    st.markdown('<div class="section-header">THREAT INDEX MATRIX</div>', unsafe_allow_html=True)
    render_threat_index_gauges(data, theme)

    st.markdown('<div class="section-header">ETHNIC TENSION MATRIX</div>', unsafe_allow_html=True)
    render_ethnic_tension_heatmap(data, theme)


def _render_traffic_light(data: DashboardData, theme):
    level_info = THREAT_LEVELS.get(data.threat_level, THREAT_LEVELS["ELEVATED"])
    st.markdown(f"""
    <div class="status-indicator">
        <div class="status-dot" style="background: {level_info['color']}; color: {level_info['color']};"></div>
        <div class="status-label" style="color: {level_info['color']};">{level_info['label']}</div>
        <div class="status-sublabel">National Threat Level</div>
    </div>
    """, unsafe_allow_html=True)


def _render_escalation_gauge(data: DashboardData, theme):
    if not HAS_PLOTLY:
        return
    hours = data.time_to_escalation
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=hours,
        number={'suffix': " hrs", 'font': {'size': 32, 'color': theme.text_primary}},
        title={'text': "Time to Potential Escalation", 'font': {'size': 12, 'color': theme.text_muted}},
        gauge={
            'axis': {'range': [0, 72], 'tickwidth': 1, 'tickcolor': theme.text_muted},
            'bar': {'color': theme.accent_primary},
            'bgcolor': theme.bg_tertiary,
            'steps': [
                {'range': [0, 12], 'color': 'rgba(255,0,68,0.2)'},
                {'range': [12, 24], 'color': 'rgba(255,107,53,0.2)'},
                {'range': [24, 48], 'color': 'rgba(255,204,0,0.2)'},
                {'range': [48, 72], 'color': 'rgba(0,255,136,0.2)'},
            ],
            'threshold': {
                'line': {'color': theme.accent_danger, 'width': 4},
                'thickness': 0.75,
                'value': 24,
            },
        },
    ))
    fig.update_layout(
        height=280, margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)', font={'color': theme.text_primary},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_top_threats(data: DashboardData, theme):
    theta_data = data.top_threats if hasattr(data, "top_threats") and data.top_threats else []
    if not theta_data:
        st.info("No top threats detected.")
        return
    cols = st.columns(3)
    for i, threat in enumerate(theta_data):
        level_info = THREAT_LEVELS.get(threat.get("level", "LOW"), THREAT_LEVELS["ELEVATED"])
        with cols[i]:
            st.markdown(f"""
            <div class="glass-card">
                <h3 style="font-size: 0.85rem; color: {theme.text_muted}; margin: 0 0 0.5rem 0;">{threat.get('title', 'Unknown')}</h3>
                <div style="font-size: 1.5rem; font-weight: 700; color: {level_info['color']};">{threat.get('level', 'LOW')}</div>
                <div style="color: {theme.accent_danger}; font-weight: 600;">{threat.get('change', '')}</div>
                <div style="font-size: 0.8rem; color: {theme.text_muted}; margin-top: 0.5rem;">{threat.get('desc', '')}</div>
            </div>
            """, unsafe_allow_html=True)


def _render_unknown_detection(data: DashboardData, theme):
    unknowns = getattr(data, "unknown_detections", [])
    if unknowns:
        lines = []
        for item in unknowns[:3]:
            if isinstance(item, dict):
                msg = item.get("message") or item.get("pattern") or item.get("title") or "Unprecedented signal pattern"
                conf = item.get("confidence")
                if isinstance(conf, (int, float)):
                    lines.append(f"{msg} (confidence {float(conf):.0%})")
                else:
                    lines.append(str(msg))
            else:
                lines.append(str(item))
        st.markdown(f"""
        <div class="alert alert-critical">
            <strong>UNPRECEDENTED PATTERN DETECTED</strong><br>
            {'<br>'.join(lines)}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No unknown-unknown patterns detected in current window.")


def _render_competing_hypotheses(data: DashboardData, theme):
    hypotheses = getattr(data, "competing_hypotheses", [])
    if not hypotheses:
        st.info("No competing hypotheses generated.")
        return
    cols = st.columns(3)
    for i, hyp in enumerate(hypotheses):
        with cols[i]:
            bar_width = hyp["probability"] * 100
            st.markdown(f"""
            <div class="glass-card-sm">
                <div style="font-weight: 600; margin-bottom: 0.5rem;">{hyp['name']}</div>
                <div style="background: {theme.bg_tertiary}; border-radius: 4px; height: 8px; margin-bottom: 0.5rem;">
                    <div style="background: {theme.accent_primary}; width: {bar_width}%; height: 100%; border-radius: 4px;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
                    <span style="color: {theme.text_muted};">Evidence: {hyp['evidence']}</span>
                    <span style="color: {theme.accent_primary};">{hyp['probability']:.0%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
