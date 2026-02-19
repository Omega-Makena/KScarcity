"""Reusable widget components shared across sentinel tabs."""

from ._shared import st, go, pd, np, make_subplots, HAS_PLOTLY, HAS_PANDAS, DashboardData


def render_threat_index_gauges(data: DashboardData, theme):
    """Render 2x4 grid of all 8 threat index gauges."""
    if not HAS_PLOTLY:
        return

    indices = data.threat_indices
    if not indices:
        st.info("Threat index data unavailable.")
        return

    SEVERITY_COLORS = {
        "LOW": "#00ff88", "GUARDED": "#7ed957", "MODERATE": "#ffc107",
        "ELEVATED": "#ff9800", "HIGH": "#ff5722", "CRITICAL": "#ff0044",
    }

    fig = make_subplots(
        rows=2, cols=4,
        specs=[[{"type": "indicator"}] * 4] * 2,
        subplot_titles=[idx["name"] for idx in indices[:8]],
    )

    for i, idx in enumerate(indices[:8]):
        row = i // 4 + 1
        col = i % 4 + 1
        color = SEVERITY_COLORS.get(idx.get("severity", "MODERATE"), "#ffc107")

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=idx["value"] * 100,
                number={'suffix': "%", 'font': {'size': 14, 'color': theme.text_primary}},
                gauge={
                    'axis': {'range': [0, 100], 'visible': False},
                    'bar': {'color': color},
                    'bgcolor': theme.bg_tertiary,
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(0,255,136,0.1)'},
                        {'range': [30, 60], 'color': 'rgba(255,204,0,0.1)'},
                        {'range': [60, 100], 'color': 'rgba(255,0,68,0.1)'},
                    ],
                },
            ),
            row=row, col=col,
        )

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': theme.text_muted, 'size': 10},
    )

    st.plotly_chart(fig, use_container_width=True)


def render_ethnic_tension_heatmap(data: DashboardData, theme):
    """Render inter-group ethnic tension heatmap."""
    if not HAS_PLOTLY:
        return

    et = data.ethnic_tensions
    tensions = et.get("tensions", {})
    highest = et.get("highest_pair")

    if not tensions:
        st.info("Ethnic tension data unavailable.")
        return

    if highest:
        st.warning(f"Highest tension: **{highest[0]}** -- **{highest[1]}**")

    # Build symmetric matrix
    groups = set()
    for key in tensions:
        g1, g2 = key.split("-")
        groups.add(g1)
        groups.add(g2)
    groups = sorted(list(groups))

    n = len(groups)
    matrix = [[0.0] * n for _ in range(n)]
    for key, val in tensions.items():
        g1, g2 = key.split("-")
        if g1 in groups and g2 in groups:
            i, j = groups.index(g1), groups.index(g2)
            matrix[i][j] = val
            matrix[j][i] = val

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=groups,
        y=groups,
        colorscale=[[0, theme.bg_secondary], [0.5, theme.accent_warning], [1, theme.accent_danger]],
        zmin=0, zmax=1,
        showscale=True,
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': theme.text_muted, 'size': 10},
        yaxis={'autorange': 'reversed'},
    )

    st.plotly_chart(fig, use_container_width=True)


def render_risk_timeline(data: DashboardData, theme):
    """Render risk score timeline chart."""
    history = data.risk_history

    if not history or len(history) < 2:
        st.info("Collecting data for risk timeline... (need at least 2 data points)")
        return

    if HAS_PANDAS:
        df = pd.DataFrame(history)
        if 'timestamp' in df.columns:
            df['time'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('time')

        cols_to_plot = [c for c in ['overall_risk', 'peak_risk'] if c in df.columns]
        if cols_to_plot:
            st.line_chart(df[cols_to_plot])

        if 'signal_count' in df.columns:
            st.bar_chart(df[['signal_count']])


def render_network_analysis(data: DashboardData, theme):
    """Render actor role distribution pie chart."""
    if not HAS_PLOTLY:
        return

    network = data.network_analysis
    roles = network.get("roles", {})

    if not roles:
        st.info("Network analysis data unavailable.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure(go.Pie(
            labels=list(roles.keys()),
            values=list(roles.values()),
            hole=0.4,
            marker=dict(colors=[
                theme.accent_danger, theme.accent_warning, theme.accent_primary,
                '#9b59b6', theme.accent_success, theme.text_muted,
            ]),
        ))
        fig.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': theme.text_primary, 'size': 11},
            legend={'font': {'size': 10}},
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Nodes", network.get("node_count", 0))
        st.metric("Edges", network.get("edge_count", 0))
        st.metric("Communities", network.get("community_count", 0))


def render_economic_indicators(data: DashboardData, theme):
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


def render_system_primitives(data: DashboardData, theme):
    """Render system primitives -- scarcity, stress, bonds, risk metrics."""
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
            indicator = "down" if stress < 0 else "up" if stress > 0 else "--"
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
