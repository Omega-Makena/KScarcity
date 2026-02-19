"""Signal Intelligence tab."""

from ._shared import st, go, np, pd, make_subplots, HAS_PLOTLY, HAS_PANDAS, DashboardData
from .widgets import render_risk_timeline


def render_signals_tab(data: DashboardData, theme):
    """Signal intelligence tab."""
    st.markdown('<div class="section-header">SIGNAL INTENSITIES (15 TYPES)</div>', unsafe_allow_html=True)

    if HAS_PLOTLY and data.signals:
        _render_signal_gauges(data, theme)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">SIGNAL CASCADE</div>', unsafe_allow_html=True)
        _render_signal_cascade(data, theme)
    with col2:
        st.markdown('<div class="section-header">CO-OCCURRENCE HEATMAP</div>', unsafe_allow_html=True)
        _render_cooccurrence_heatmap(data, theme)

    st.markdown("---")
    st.markdown('<div class="section-header">SIGNAL SILENCE DETECTOR</div>', unsafe_allow_html=True)
    _render_signal_silence(data, theme)

    st.markdown("---")
    st.markdown('<div class="section-header">RISK SCORE TIMELINE</div>', unsafe_allow_html=True)
    render_risk_timeline(data, theme)


def _render_signal_gauges(data: DashboardData, theme):
    signals = data.signals if data.signals else []
    if not signals:
        st.info("No signal data available.")
        return
    fig = make_subplots(
        rows=3, cols=5,
        specs=[[{"type": "indicator"}] * 5] * 3,
        subplot_titles=[s.name[:15] for s in signals[:15]],
    )
    for i, signal in enumerate(signals[:15]):
        row = i // 5 + 1
        col = i % 5 + 1
        color = (
            theme.accent_success if signal.intensity < 0.4 else
            theme.accent_warning if signal.intensity < 0.6 else
            theme.accent_danger
        )
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=signal.intensity * 100,
                number={'suffix': "%", 'font': {'size': 12, 'color': theme.text_primary}},
                gauge={
                    'axis': {'range': [0, 100], 'visible': False},
                    'bar': {'color': color},
                    'bgcolor': theme.bg_tertiary,
                },
            ),
            row=row, col=col,
        )
    fig.update_layout(
        height=350, margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)', font={'color': theme.text_muted, 'size': 10},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_signal_cascade(data: DashboardData, theme):
    if not HAS_PLOTLY:
        return
    signals = sorted(data.signals or [], key=lambda s: float(getattr(s, "intensity", 0.0)), reverse=True)[:8]
    counties = sorted(
        [(name, c) for name, c in (data.counties or {}).items()],
        key=lambda kv: float(kv[1].risk_score if hasattr(kv[1], "risk_score") else kv[1].get("risk_score", 0.0)),
        reverse=True,
    )[:5]
    if not signals or not counties:
        st.info("Signal cascade unavailable: need both signal and county risk data.")
        return
    src_nodes = [s.name[:28] for s in signals]
    tgt_nodes = [name for name, _ in counties]
    labels = src_nodes + tgt_nodes
    county_weights = []
    for _, county in counties:
        risk = county.risk_score if hasattr(county, "risk_score") else county.get("risk_score", 0.0)
        county_weights.append(max(0.01, float(risk)))
    weight_sum = sum(county_weights) or 1.0
    src, tgt, val = [], [], []
    for i, sig in enumerate(signals):
        sig_strength = max(0.01, float(sig.intensity))
        for j, w in enumerate(county_weights):
            src.append(i)
            tgt.append(len(src_nodes) + j)
            val.append(sig_strength * (w / weight_sum))
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels, color=[theme.accent_primary]*len(src_nodes) + [theme.accent_warning]*len(tgt_nodes),
                  pad=12, thickness=14, line=dict(color=theme.border_default, width=1)),
        link=dict(source=src, target=tgt, value=val, color="rgba(0,255,136,0.22)"),
    ))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)', font={'color': theme.text_primary, 'size': 10},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_cooccurrence_heatmap(data: DashboardData, theme):
    if not HAS_PLOTLY:
        return
    matrix = getattr(data, "cooccurrence_matrix", None)
    if not matrix:
        st.info("Co-occurrence data unavailable.")
        return
    import numpy as _np
    z_data = _np.array(matrix, dtype=float)
    if z_data.ndim != 2 or z_data.shape[0] == 0 or z_data.shape[1] == 0:
        st.info("Co-occurrence matrix is empty.")
        return
    n = min(z_data.shape[0], z_data.shape[1])
    z_data = z_data[:n, :n]
    signal_labels = [s.name for s in (data.signals or [])]
    labels = signal_labels[:n]
    if len(labels) < n:
        labels.extend([f"S{i+1}" for i in range(len(labels), n)])
    fig = go.Figure(go.Heatmap(
        z=z_data, x=labels, y=labels,
        colorscale=[[0, theme.bg_secondary], [0.5, theme.accent_warning], [1, theme.accent_danger]],
        zmin=0, zmax=1, showscale=True,
    ))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font={'color': theme.text_muted, 'size': 9},
        xaxis={'tickangle': -45}, yaxis={'autorange': 'reversed'},
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_signal_silence(data: DashboardData, theme):
    silence_data = getattr(data, "silence_indicators", [])
    if not silence_data:
        st.info("No silence indicators detected.")
        return
    col1, col2 = st.columns([1, 1])
    with col1:
        if HAS_PLOTLY:
            df = pd.DataFrame(silence_data)
            colors = [theme.accent_danger if s > 0.7 else theme.accent_warning if s > 0.4 else theme.accent_success
                      for s in df["Silence"]]
            fig = go.Figure(go.Bar(
                x=df["Region"], y=df["Silence"], marker_color=colors,
                text=[f"{s:.0%}" for s in df["Silence"]], textposition='outside',
            ))
            fig.add_hline(y=0.5, line_dash="dash", line_color=theme.accent_danger)
            fig.update_layout(
                height=250, margin=dict(l=40, r=40, t=20, b=40),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font={'color': theme.text_muted}, yaxis_range=[0, 1],
            )
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        df = pd.DataFrame(silence_data)
        if "Silence" in df.columns and "Region" in df.columns:
            top = df.sort_values("Silence", ascending=False).head(3)
            lines = [f"{row['Region']}: {float(row['Silence']):.0%}" for _, row in top.iterrows()]
            st.markdown(f"""
            <div class="alert alert-warning">
                <strong>GOING DARK INDICATORS:</strong><br>
                {'<br>'.join(lines)}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Silence diagnostics available, but schema is incomplete.")
