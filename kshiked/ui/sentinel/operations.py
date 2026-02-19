"""Operations tab."""

from ._shared import st, go, pd, np, HAS_PLOTLY, HAS_PANDAS, DashboardData
from .widgets import render_network_analysis, render_economic_indicators, render_system_primitives


def render_operations_tab(data: DashboardData, theme):
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<div class="section-header">COUNTY RISK ASSESSMENT</div>', unsafe_allow_html=True)
        _render_county_table(data, theme)
    with col2:
        st.markdown('<div class="section-header">RECENT ALERTS</div>', unsafe_allow_html=True)
        _render_alerts(theme)
        st.markdown('<div class="section-header" style="margin-top: 1rem;">INFRASTRUCTURE HEALTH</div>', unsafe_allow_html=True)
        _render_infrastructure_health(theme)
    st.markdown("---")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="section-header">NETWORK ANALYSIS</div>', unsafe_allow_html=True)
        render_network_analysis(data, theme)
    with col4:
        st.markdown('<div class="section-header">ECONOMIC SATISFACTION INDEX</div>', unsafe_allow_html=True)
        render_economic_indicators(data, theme)
    st.markdown("---")
    st.markdown('<div class="section-header">SYSTEM PRIMITIVES</div>', unsafe_allow_html=True)
    render_system_primitives(data, theme)


def _render_county_table(data: DashboardData, theme):
    if not HAS_PANDAS:
        return
    if not hasattr(data, "counties") or not data.counties:
        st.info("No county data available.")
        return
    table_data = []
    for name, county in data.counties.items():
        risk = county.get('risk_score', 0) if isinstance(county, dict) else getattr(county, 'risk_score', 0)
        level = "Critical" if risk > 0.7 else "High" if risk > 0.5 else "Moderate" if risk > 0.3 else "Low"
        signals_cnt = len(county.get('top_signals', [])) if isinstance(county, dict) else len(getattr(county, 'top_signals', []))
        table_data.append({"County": name, "Risk Score": f"{risk:.0%}", "Level": level, "Signals": signals_cnt})
    df = pd.DataFrame(table_data).sort_values("Risk Score", ascending=False)
    st.dataframe(df, use_container_width=True, height=350)


def _render_alerts(theme):
    alerts = st.session_state.get("sentinel_alerts", [])
    if not alerts:
        st.info("No active alerts.")
        return
    for alert in alerts:
        alert_type = alert.get('type', 'warning')
        if alert_type == 'critical':
            bg, border, color = '#1a0a0a', theme.accent_danger, '#ff6b6b'
        else:
            bg, border, color = '#1a1a0a', theme.accent_warning, '#ffc107'
        st.markdown(f"""
        <div style="background: {bg}; border-left: 4px solid {border}; padding: 0.8rem 1rem;
                     border-radius: 4px; margin-bottom: 0.5rem;">
            <div style="color: {color}; font-weight: 600; font-size: 0.85rem;">{alert.get('time', '')}</div>
            <div style="color: {theme.text_primary}; font-size: 0.8rem; margin-top: 0.25rem;">{alert.get('msg', '')}</div>
        </div>
        """, unsafe_allow_html=True)


def _render_infrastructure_health(theme):
    metrics = []
    if not metrics:
        st.info("Infrastructure health data unavailable.")
        return
    for label, value, color in metrics:
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid {theme.border_subtle};">
            <span>{label}</span>
            <span style="color: {color}; font-weight: 600;">{value}</span>
        </div>
        """, unsafe_allow_html=True)
