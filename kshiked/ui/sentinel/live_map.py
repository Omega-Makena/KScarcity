"""Live Threat Map tab."""

from ._shared import st, components, DashboardData, THREAT_LEVELS


def render_live_map_tab(data: DashboardData, theme):
    """Kaspersky-inspired live threat map."""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown('<div class="section-header">KENYA THREAT MAP</div>', unsafe_allow_html=True)
        _render_threat_globe(data, theme)

    with col2:
        st.markdown('<div class="section-header">LIVE THREATS</div>', unsafe_allow_html=True)
        _render_threat_counter(data, theme)

        st.markdown('<div class="section-header" style="margin-top: 1rem;">TOP COUNTIES</div>', unsafe_allow_html=True)
        _render_top_counties(data, theme)


def _render_threat_globe(data: DashboardData, theme):
    """3D interactive globe with threat indicators (Globe.gl)."""
    try:
        from globe_viz import get_globe_html
    except ImportError:
        st.error("Globe visualization component missing")
        return

    counties = data.counties

    globe_data = []
    if counties:
        for name, item in counties.items():
            if isinstance(item, dict):
                risk = item.get('risk_score', 0.5)
            else:
                risk = int(item.risk_score * 100) / 100.0 if hasattr(item, 'risk_score') else 0.5
            globe_data.append({"name": name, "risk_score": risk})

    html_code = get_globe_html(globe_data, height=500)
    components.html(html_code, height=500)


def _render_threat_counter(data: DashboardData, theme):
    """Live threat counter display."""
    total_signals = sum(s.count for s in data.signals) if data.signals else 1247

    st.markdown(f"""
    <div class="glass-card" style="text-align: center; margin-bottom: 1rem;">
        <div class="live-label">ACTIVE SIGNALS</div>
        <div class="live-counter">{total_signals:,}</div>
    </div>
    """, unsafe_allow_html=True)

    high_risk = len([c for c in data.counties.values() if hasattr(c, 'risk_score') and c.risk_score > 0.7])
    moderate_risk = len([c for c in data.counties.values() if hasattr(c, 'risk_score') and 0.4 < c.risk_score <= 0.7])

    st.markdown(f"""
    <div class="glass-card-sm">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span>Critical Counties</span>
            <span style="color: {theme.accent_critical}; font-weight: 700;">{high_risk}</span>
        </div>
        <div style="display: flex; justify-content: space-between;">
            <span>Elevated Counties</span>
            <span style="color: {theme.accent_warning}; font-weight: 700;">{moderate_risk}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_top_counties(data: DashboardData, theme):
    """Top risk counties list."""
    sorted_counties = sorted(
        [(k, v) for k, v in data.counties.items() if hasattr(v, 'risk_score')],
        key=lambda x: x[1].risk_score,
        reverse=True,
    )[:5]

    for name, county in sorted_counties:
        color = (
            theme.accent_critical if county.risk_score > 0.7 else
            theme.accent_danger if county.risk_score > 0.5 else
            theme.accent_warning
        )
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid {theme.border_subtle};">
            <span>{name}</span>
            <span style="color: {color}; font-family: 'Space Mono', monospace;">{county.risk_score:.0%}</span>
        </div>
        """, unsafe_allow_html=True)
