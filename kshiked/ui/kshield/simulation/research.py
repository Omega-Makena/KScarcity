"""Research module tabs: IO Sectors, Inequality, Financial, Open Economy, Research Engine."""
from ._shared import (st, pd, np, go, make_subplots, HAS_DATA_STACK, HAS_PLOTLY, PALETTE, base_layout)


def render_io_sectors_tab(theme):
    """Input-Output disaggregation across 5 Kenya sectors."""
    trajectory = st.session_state.get("sim_research_trajectory")
    research_econ = st.session_state.get("sim_research_econ")

    if not trajectory or not research_econ:
        st.info("Run the **Research Engine** tab first to see IO sector analysis.")
        return
    if not HAS_PLOTLY:
        st.warning("Plotly required for sector charts.")
        return

    io_frames = [f.get("io_sectors", {}) for f in trajectory if f.get("io_sectors")]
    if not io_frames:
        st.warning("IO sector data not available. Enable IO module in Research Engine.")
        return

    sector_names = list(io_frames[0].keys())
    t_vals = list(range(len(io_frames)))

    # ── 1. LEONTIEF MATRIX HEATMAP ───────────────────────────────────────────
    st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:600; "
                f"font-size:0.85rem; margin-bottom:0.5rem;'>LEONTIEF INPUT-OUTPUT MATRIX</div>",
                unsafe_allow_html=True)
    st.caption("Technical coefficients: A[i,j] = fraction of sector j's output used as input by sector i. "
               "Darker = stronger inter-sector dependency.")

    A_matrix = research_econ.io_model.A
    L_matrix = research_econ.io_model.leontief_inverse
    display_names = [n.title() for n in sector_names]

    col_a, col_b = st.columns(2)
    with col_a:
        # Technical coefficient matrix (A)
        fig_a = go.Figure(go.Heatmap(
            z=A_matrix.tolist(),
            x=display_names, y=display_names,
            colorscale=[[0, 'rgba(0,0,0,0)'], [0.5, PALETTE[1]], [1, PALETTE[0]]],
            text=[[f"{v:.3f}" for v in row] for row in A_matrix.tolist()],
            texttemplate="%{text}", textfont=dict(size=11),
            hovertemplate="From %{x} → To %{y}: %{z:.4f}<extra></extra>",
            showscale=False,
        ))
        fig_a.update_layout(**base_layout(theme, height=340,
            title=dict(text="A Matrix (Technical Coefficients)", font=dict(color=theme.text_muted, size=12)),
            xaxis=dict(title="Supplying Sector", side="bottom"),
            yaxis=dict(title="Consuming Sector", autorange="reversed")))
        st.plotly_chart(fig_a, use_container_width=True)

    with col_b:
        # Leontief inverse (I-A)^-1
        fig_l = go.Figure(go.Heatmap(
            z=L_matrix.tolist(),
            x=display_names, y=display_names,
            colorscale=[[0, 'rgba(0,0,0,0)'], [0.5, PALETTE[4]], [1, PALETTE[2]]],
            text=[[f"{v:.3f}" for v in row] for row in L_matrix.tolist()],
            texttemplate="%{text}", textfont=dict(size=11),
            hovertemplate="From %{x} → To %{y}: %{z:.4f}<extra></extra>",
            showscale=False,
        ))
        fig_l.update_layout(**base_layout(theme, height=340,
            title=dict(text="Leontief Inverse (I-A)⁻¹", font=dict(color=theme.text_muted, size=12)),
            xaxis=dict(title="Final Demand", side="bottom"),
            yaxis=dict(title="Gross Output", autorange="reversed")))
        st.plotly_chart(fig_l, use_container_width=True)

    # ── 2. INTER-SECTOR FLOW SANKEY ──────────────────────────────────────────
    st.markdown(f"<div style='color:{theme.accent_warning}; font-weight:600; "
                f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
                f"INTER-SECTOR FLOWS (Intermediate Inputs)</div>",
                unsafe_allow_html=True)

    last_io = io_frames[-1]
    src_nodes, tgt_nodes, flow_vals, flow_labels = [], [], [], []
    node_labels = display_names + [f"{n} (input)" for n in display_names]
    node_colors = PALETTE[:len(sector_names)] + PALETTE[:len(sector_names)]

    for i, from_name in enumerate(sector_names):
        from_output = last_io.get(from_name, {}).get("output", 1)
        for j, to_name in enumerate(sector_names):
            coeff = A_matrix[j, i]
            flow = coeff * from_output
            if flow > 0.01:
                src_nodes.append(i)
                tgt_nodes.append(len(sector_names) + j)
                flow_vals.append(flow)
                flow_labels.append(f"{from_name.title()} → {to_name.title()}: {flow:.2f}")

    if flow_vals:
        fig_sankey = go.Figure(go.Sankey(
            arrangement='snap',
            node=dict(label=node_labels, color=node_colors, pad=15, thickness=18,
                      line=dict(color=theme.border_default, width=0.5)),
            link=dict(source=src_nodes, target=tgt_nodes, value=flow_vals,
                      label=flow_labels,
                      color=[PALETTE[s % len(PALETTE)] + '30' for s in src_nodes]),
        ))
        fig_sankey.update_layout(**base_layout(theme, height=400,
            title=dict(text="Intermediate Input Flows Between Sectors",
                       font=dict(color=theme.text_muted, size=13))))
        st.plotly_chart(fig_sankey, use_container_width=True)

    # ── 3. OUTPUT SHARE EVOLUTION (Stacked Area) ─────────────────────────────
    st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:600; "
                f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
                f"STRUCTURAL CHANGE: SECTOR OUTPUT SHARES</div>",
                unsafe_allow_html=True)

    fig_area = go.Figure()
    for i, name in enumerate(sector_names):
        shares = [f.get(name, {}).get("output_share", 0) * 100 for f in io_frames]
        fig_area.add_trace(go.Scatter(
            x=t_vals, y=shares, mode='lines', name=name.title(),
            line=dict(color=PALETTE[i % len(PALETTE)], width=0),
            stackgroup='one', groupnorm='percent',
            hovertemplate=f"{name.title()}: %{{y:.1f}}%<extra></extra>",
        ))
    fig_area.update_layout(**base_layout(theme, height=380,
        title=dict(text="Sector Output Shares Over Time (Stacked %)",
                   font=dict(color=theme.text_muted, size=13)),
        xaxis=dict(title="Quarter"),
        yaxis=dict(title="Share (%)", range=[0, 100]),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor='rgba(0,0,0,0)')))
    st.plotly_chart(fig_area, use_container_width=True)

    # ── 4. COMPARATIVE ADVANTAGE SCATTER ─────────────────────────────────────
    st.markdown(f"<div style='color:{theme.accent_warning}; font-weight:600; "
                f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
                f"COMPARATIVE ADVANTAGE: OUTPUT vs EMPLOYMENT</div>",
                unsafe_allow_html=True)

    emp_shares_cfg = research_econ.io_cfg.employment_shares or {}
    output_shares_final = [last_io.get(n, {}).get("output_share", 0) * 100 for n in sector_names]
    emp_shares_final = [emp_shares_cfg.get(n, 0.2) * 100 for n in sector_names]
    gross_outputs = [last_io.get(n, {}).get("output", 0) for n in sector_names]

    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=emp_shares_final, y=output_shares_final,
        mode='markers+text',
        marker=dict(size=[max(15, v * 0.6) for v in gross_outputs],
                    color=PALETTE[:len(sector_names)],
                    line=dict(width=2, color='white')),
        text=display_names, textposition='top center',
        textfont=dict(color=theme.text_primary, size=11),
        hovertemplate="%{text}<br>Employment: %{x:.1f}%<br>Output: %{y:.1f}%<extra></extra>",
    ))
    # 45-degree line (output = employment share)
    max_val = max(max(output_shares_final, default=60), max(emp_shares_final, default=60)) + 5
    fig_scatter.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val], mode='lines',
        line=dict(color=theme.text_muted, dash='dash', width=1),
        name='Equal Productivity', showlegend=True,
    ))
    fig_scatter.update_layout(**base_layout(theme, height=400,
        title=dict(text="Sector Efficiency: Above line = high productivity, Below = labor-intensive",
                   font=dict(color=theme.text_muted, size=12)),
        xaxis=dict(title="Employment Share (%)", range=[0, max_val]),
        yaxis=dict(title="Output Share (%)", range=[0, max_val]),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor='rgba(0,0,0,0)')))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── 5. OUTPUT MULTIPLIERS & LINKAGES BAR CHART ───────────────────────────
    st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:600; "
                f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
                f"OUTPUT MULTIPLIERS &amp; LINKAGES</div>",
                unsafe_allow_html=True)

    multipliers = research_econ.io_model.output_multipliers()
    backward = research_econ.io_model.backward_linkages()
    forward = research_econ.io_model.forward_linkages()

    fig_mult = go.Figure()
    fig_mult.add_trace(go.Bar(x=display_names, y=multipliers.tolist(), name="Output Multiplier",
                               marker_color=PALETTE[0], text=[f"{v:.2f}" for v in multipliers],
                               textposition='auto'))
    fig_mult.add_trace(go.Bar(x=display_names, y=backward.tolist(), name="Backward Linkage",
                               marker_color=PALETTE[1], text=[f"{v:.2f}" for v in backward],
                               textposition='auto'))
    fig_mult.add_trace(go.Bar(x=display_names, y=forward.tolist(), name="Forward Linkage",
                               marker_color=PALETTE[4], text=[f"{v:.2f}" for v in forward],
                               textposition='auto'))
    fig_mult.add_hline(y=1.0, line_dash="dash", line_color=theme.text_muted,
                       annotation_text="Average (1.0)")
    fig_mult.update_layout(**base_layout(theme, height=380,
        title=dict(text="Sector Multipliers & Linkage Strength",
                   font=dict(color=theme.text_muted, size=13)),
        barmode='group', yaxis=dict(title="Index Value"),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor='rgba(0,0,0,0)')))
    st.plotly_chart(fig_mult, use_container_width=True)

    st.markdown(f"<div style='font-size:0.75rem; color:{theme.text_muted}; padding:0.3rem 0;'>"
                f"<b>Output Multiplier:</b> total output effect per unit of final demand. "
                f"<b>Backward:</b> how much a sector's demand pulls from all sectors. "
                f"<b>Forward:</b> how much a sector's output feeds into all sectors.</div>",
                unsafe_allow_html=True)


def render_inequality_tab(theme):
    """Distributional analysis across income quintiles."""
    trajectory = st.session_state.get("sim_research_trajectory")
    research_econ = st.session_state.get("sim_research_econ")

    if not trajectory or not research_econ:
        st.info("Run the **Research Engine** tab first to see inequality analysis.")
        return
    if not HAS_PLOTLY:
        st.warning("Plotly required.")
        return

    ineq_frames = [f.get("inequality", {}) for f in trajectory if f.get("inequality")]
    if not ineq_frames:
        st.warning("Inequality data not available. Enable heterogeneous agents in Research Engine.")
        return

    t_vals = list(range(len(ineq_frames)))
    quintile_keys = list(ineq_frames[0].get("quintile_incomes", {}).keys())
    gini_vals = [f.get("gini", 0) for f in ineq_frames]
    palma_vals = [f.get("palma", 0) for f in ineq_frames]

    # ── 1. GINI & PALMA GAUGES ───────────────────────────────────────────────
    st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:600; "
                f"font-size:0.85rem; margin-bottom:0.5rem;'>INEQUALITY INDICATORS</div>",
                unsafe_allow_html=True)

    col_g, col_p = st.columns(2)
    with col_g:
        gini_now = gini_vals[-1]
        # Gauge: 0 = perfect equality, 0.63 = Kenya's historical worst
        fig_gini = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=gini_now,
            number=dict(suffix="", valueformat=".4f", font=dict(color=theme.text_primary)),
            delta=dict(reference=gini_vals[0], valueformat=".4f",
                       increasing=dict(color=PALETTE[3]),
                       decreasing=dict(color=PALETTE[0])),
            gauge=dict(
                axis=dict(range=[0, 0.7], tickfont=dict(color=theme.text_muted, size=9)),
                bar=dict(color=PALETTE[0]),
                bgcolor='rgba(255,255,255,0.05)',
                bordercolor=theme.border_default,
                steps=[
                    dict(range=[0, 0.30], color='rgba(0,255,136,0.1)'),
                    dict(range=[0.30, 0.45], color='rgba(245,213,71,0.1)'),
                    dict(range=[0.45, 0.70], color='rgba(255,51,102,0.1)'),
                ],
                threshold=dict(line=dict(color=PALETTE[3], width=3), thickness=0.8, value=0.408),
            ),
            title=dict(text="Gini Coefficient<br><span style='font-size:0.65rem'>Kenya benchmark: 0.408</span>",
                       font=dict(size=13, color=theme.text_muted)),
        ))
        fig_gini.update_layout(**base_layout(theme, height=280, margin=dict(l=30, r=30, t=60, b=10)))
        st.plotly_chart(fig_gini, use_container_width=True)

    with col_p:
        palma_now = palma_vals[-1]
        fig_palma = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=palma_now,
            number=dict(suffix="", valueformat=".2f", font=dict(color=theme.text_primary)),
            delta=dict(reference=palma_vals[0], valueformat=".2f",
                       increasing=dict(color=PALETTE[3]),
                       decreasing=dict(color=PALETTE[0])),
            gauge=dict(
                axis=dict(range=[0, 5], tickfont=dict(color=theme.text_muted, size=9)),
                bar=dict(color=PALETTE[1]),
                bgcolor='rgba(255,255,255,0.05)',
                bordercolor=theme.border_default,
                steps=[
                    dict(range=[0, 1.5], color='rgba(0,255,136,0.1)'),
                    dict(range=[1.5, 3.0], color='rgba(245,213,71,0.1)'),
                    dict(range=[3.0, 5.0], color='rgba(255,51,102,0.1)'),
                ],
                threshold=dict(line=dict(color=PALETTE[3], width=3), thickness=0.8, value=2.5),
            ),
            title=dict(text="Palma Ratio<br><span style='font-size:0.65rem'>Top 10% / Bottom 40%</span>",
                       font=dict(size=13, color=theme.text_muted)),
        ))
        fig_palma.update_layout(**base_layout(theme, height=280, margin=dict(l=30, r=30, t=60, b=10)))
        st.plotly_chart(fig_palma, use_container_width=True)

    # ── 2. QUINTILE INCOME SHARE STACKED AREA ────────────────────────────────
    st.markdown(f"<div style='color:{theme.accent_warning}; font-weight:600; "
                f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
                f"INCOME DISTRIBUTION: QUINTILE SHARES OVER TIME</div>",
                unsafe_allow_html=True)

    if quintile_keys:
        fig_qarea = go.Figure()
        q_labels = ["Q1 (Poorest 20%)", "Q2 (Lower)", "Q3 (Middle)", "Q4 (Upper)", "Q5 (Richest 20%)"]
        q_colors = [PALETTE[3], PALETTE[6], PALETTE[2], PALETTE[1], PALETTE[0]]

        for i, qk in enumerate(quintile_keys):
            all_incomes_by_step = []
            for f in ineq_frames:
                q_incomes = f.get("quintile_incomes", {})
                total = sum(q_incomes.values()) or 1.0
                all_incomes_by_step.append(q_incomes.get(qk, 0) / total * 100)
            label = q_labels[i] if i < len(q_labels) else qk.upper()
            fig_qarea.add_trace(go.Scatter(
                x=t_vals, y=all_incomes_by_step, mode='lines',
                name=label,
                line=dict(color=q_colors[i % len(q_colors)], width=0),
                stackgroup='one',
                hovertemplate=f"{label}: %{{y:.1f}}%<extra></extra>",
            ))

        fig_qarea.update_layout(**base_layout(theme, height=400,
            title=dict(text="Income Share by Quintile (%)", font=dict(color=theme.text_muted, size=13)),
            xaxis=dict(title="Quarter"),
            yaxis=dict(title="Income Share (%)", range=[0, 100]),
            legend=dict(orientation="h", y=1.08, x=0, bgcolor='rgba(0,0,0,0)')))
        st.plotly_chart(fig_qarea, use_container_width=True)

    # ── 3. PRO-POOR GROWTH BAR CHART ─────────────────────────────────────────
    st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:600; "
                f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
                f"INCOME GROWTH BY QUINTILE (% change start → end)</div>",
                unsafe_allow_html=True)

    if quintile_keys and len(ineq_frames) > 1:
        first_incomes = ineq_frames[0].get("quintile_incomes", {})
        last_incomes = ineq_frames[-1].get("quintile_incomes", {})
        growth_pcts = []
        bar_colors = []

        for qk in quintile_keys:
            start_v = first_incomes.get(qk, 1)
            end_v = last_incomes.get(qk, 1)
            pct = ((end_v - start_v) / max(abs(start_v), 0.01)) * 100
            growth_pcts.append(pct)

        # Color: green if bottom quintiles grow faster than top (pro-poor)
        avg_bottom = np.mean(growth_pcts[:2]) if len(growth_pcts) >= 2 else 0
        avg_top = np.mean(growth_pcts[-2:]) if len(growth_pcts) >= 2 else 0
        is_pro_poor = avg_bottom > avg_top

        for i, pct in enumerate(growth_pcts):
            if i < 2:
                bar_colors.append(PALETTE[0] if pct > avg_top else PALETTE[3])
            elif i >= 3:
                bar_colors.append(PALETTE[0] if pct > 0 else PALETTE[3])
            else:
                bar_colors.append(PALETTE[2])

        q_labels_short = ["Q1\n(Poorest)", "Q2", "Q3", "Q4", "Q5\n(Richest)"]
        fig_ppg = go.Figure(go.Bar(
            x=q_labels_short[:len(growth_pcts)], y=growth_pcts,
            marker_color=bar_colors,
            text=[f"{v:+.1f}%" for v in growth_pcts], textposition='auto',
            hovertemplate="%{x}: %{y:+.1f}%<extra></extra>",
        ))
        fig_ppg.add_hline(y=0, line_color=theme.text_muted, line_dash="dot")
        ppg_label = "<span style='color:" + (PALETTE[0] if is_pro_poor else PALETTE[3]) + "'>" + \
                    ("✓ PRO-POOR" if is_pro_poor else "✗ PRO-RICH") + "</span>"
        fig_ppg.update_layout(**base_layout(theme, height=350,
            title=dict(text=f"Income Growth by Quintile — {ppg_label}",
                       font=dict(color=theme.text_muted, size=13)),
            yaxis=dict(title="Growth (%)")))
        st.plotly_chart(fig_ppg, use_container_width=True)

    # ── 4. INTERACTIVE LORENZ CURVE WITH SLIDER ──────────────────────────────
    st.markdown(f"<div style='color:{theme.accent_warning}; font-weight:600; "
                f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
                f"LORENZ CURVE EVOLUTION</div>",
                unsafe_allow_html=True)

    if quintile_keys and len(ineq_frames) > 1:
        lorenz_t = st.slider("Select quarter for Lorenz curve", 0, len(ineq_frames) - 1,
                             len(ineq_frames) - 1, key="lorenz_t")

        sel_ineq = ineq_frames[lorenz_t]
        q_incomes = sel_ineq.get("quintile_incomes", {})
        sorted_inc = sorted(q_incomes.values())
        total = sum(sorted_inc) or 1
        cum_share = [0.0]
        for v in sorted_inc:
            cum_share.append(cum_share[-1] + v / total)
        pop_share = [i / len(sorted_inc) for i in range(len(sorted_inc) + 1)]

        # Also compute for period 0 for comparison
        q_incomes_0 = ineq_frames[0].get("quintile_incomes", {})
        sorted_inc_0 = sorted(q_incomes_0.values())
        total_0 = sum(sorted_inc_0) or 1
        cum_share_0 = [0.0]
        for v in sorted_inc_0:
            cum_share_0.append(cum_share_0[-1] + v / total_0)

        fig_lorenz = go.Figure()
        # Perfect equality
        fig_lorenz.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
            name='Perfect Equality', line=dict(color=theme.text_muted, dash='dash', width=1)))
        # Period 0 Lorenz
        fig_lorenz.add_trace(go.Scatter(x=pop_share, y=cum_share_0, mode='lines',
            name='Period 0', line=dict(color=PALETTE[2], width=2, dash='dot')))
        # Selected period Lorenz
        fig_lorenz.add_trace(go.Scatter(x=pop_share, y=cum_share, mode='lines+markers',
            name=f'Period {lorenz_t}', line=dict(color=PALETTE[0], width=3),
            marker=dict(size=8, color=PALETTE[0])))
        # Shade inequality area
        fig_lorenz.add_trace(go.Scatter(
            x=pop_share + pop_share[::-1],
            y=cum_share + pop_share[::-1],
            fill='toself', fillcolor='rgba(255,51,102,0.1)',
            line=dict(color='rgba(0,0,0,0)'), name='Inequality Area',
            showlegend=True,
        ))

        gini_t = sel_ineq.get("gini", 0)
        fig_lorenz.update_layout(**base_layout(theme, height=400,
            title=dict(text=f"Lorenz Curve — Quarter {lorenz_t} (Gini={gini_t:.4f})",
                       font=dict(color=theme.text_muted, size=13)),
            xaxis=dict(title="Cumulative Population Share", range=[0, 1]),
            yaxis=dict(title="Cumulative Income Share", range=[0, 1]),
            legend=dict(orientation="h", y=1.08, x=0, bgcolor='rgba(0,0,0,0)')))
        st.plotly_chart(fig_lorenz, use_container_width=True)

    # ── 5. GINI TRAJECTORY LINE ──────────────────────────────────────────────
    st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:600; "
                f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
                f"GINI &amp; PALMA TIME SERIES</div>",
                unsafe_allow_html=True)

    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ts.add_trace(go.Scatter(
        x=t_vals, y=gini_vals, name="Gini",
        line=dict(color=PALETTE[0], width=3),
        fill='tozeroy', fillcolor=PALETTE[0] + '14',
    ), secondary_y=False)
    fig_ts.add_trace(go.Scatter(
        x=t_vals, y=palma_vals, name="Palma Ratio",
        line=dict(color=PALETTE[1], width=3),
    ), secondary_y=True)
    # Kenya benchmark
    fig_ts.add_hline(y=0.408, line_dash="dash", line_color=PALETTE[3],
                     annotation_text="Kenya Gini (0.408)", secondary_y=False)
    fig_ts.update_layout(**base_layout(theme, height=350,
        title=dict(text="Inequality Trajectory", font=dict(color=theme.text_muted, size=13)),
        xaxis=dict(title="Quarter"),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor='rgba(0,0,0,0)')))
    fig_ts.update_yaxes(title_text="Gini", secondary_y=False)
    fig_ts.update_yaxes(title_text="Palma", secondary_y=True)
    st.plotly_chart(fig_ts, use_container_width=True)


def render_financial_tab(theme):
    """Financial accelerator dashboard — NPL, CAR, credit spread, EFP."""
    trajectory = st.session_state.get("sim_research_trajectory")
    research_econ = st.session_state.get("sim_research_econ")

    if not trajectory or not research_econ:
        st.info("Run the **Research Engine** tab first to see financial analysis.")
        return
    if not HAS_PLOTLY:
        st.warning("Plotly required.")
        return

    fin_frames = [f.get("financial", {}) for f in trajectory if f.get("financial")]
    if not fin_frames:
        st.warning("Financial data not available. Enable financial accelerator in Research Engine.")
        return

    t_vals = list(range(len(fin_frames)))

    # ── 1. GAUGE DASHBOARD — NPL & CAR ──────────────────────────────────────
    st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:600; "
                f"font-size:0.85rem; margin-bottom:0.5rem;'>FINANCIAL STABILITY GAUGES</div>",
                unsafe_allow_html=True)

    last_fin = fin_frames[-1]
    npl = last_fin.get("npl_ratio", 0)
    car = last_fin.get("car", 0)
    spread = last_fin.get("credit_spread", 0)
    efp = last_fin.get("efp", 0)

    col_npl, col_car, col_fsi = st.columns(3)
    with col_npl:
        fig_npl_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=npl * 100,
            number=dict(suffix="%", valueformat=".1f", font=dict(color=theme.text_primary)),
            delta=dict(reference=fin_frames[0].get("npl_ratio", npl) * 100,
                       suffix="%", valueformat=".2f",
                       increasing=dict(color=PALETTE[3]),
                       decreasing=dict(color=PALETTE[0])),
            gauge=dict(
                axis=dict(range=[0, 25], ticksuffix="%",
                          tickfont=dict(color=theme.text_muted, size=9)),
                bar=dict(color=PALETTE[3] if npl > 0.10 else PALETTE[2] if npl > 0.05 else PALETTE[0]),
                bgcolor='rgba(255,255,255,0.05)',
                bordercolor=theme.border_default,
                steps=[
                    dict(range=[0, 5], color='rgba(0,255,136,0.08)'),
                    dict(range=[5, 10], color='rgba(245,213,71,0.08)'),
                    dict(range=[10, 25], color='rgba(255,51,102,0.08)'),
                ],
                threshold=dict(line=dict(color=PALETTE[3], width=3), thickness=0.8, value=10),
            ),
            title=dict(text="NPL Ratio<br><span style='font-size:0.6rem'>Threshold: 10%</span>",
                       font=dict(size=12, color=theme.text_muted)),
        ))
        fig_npl_g.update_layout(**base_layout(theme, height=250, margin=dict(l=20, r=20, t=55, b=5)))
        st.plotly_chart(fig_npl_g, use_container_width=True)

    with col_car:
        fig_car_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=car * 100,
            number=dict(suffix="%", valueformat=".1f", font=dict(color=theme.text_primary)),
            delta=dict(reference=fin_frames[0].get("car", car) * 100,
                       suffix="%", valueformat=".2f",
                       increasing=dict(color=PALETTE[0]),
                       decreasing=dict(color=PALETTE[3])),
            gauge=dict(
                axis=dict(range=[0, 30], ticksuffix="%",
                          tickfont=dict(color=theme.text_muted, size=9)),
                bar=dict(color=PALETTE[0] if car > 0.145 else PALETTE[2] if car > 0.10 else PALETTE[3]),
                bgcolor='rgba(255,255,255,0.05)',
                bordercolor=theme.border_default,
                steps=[
                    dict(range=[0, 10], color='rgba(255,51,102,0.08)'),
                    dict(range=[10, 14.5], color='rgba(245,213,71,0.08)'),
                    dict(range=[14.5, 30], color='rgba(0,255,136,0.08)'),
                ],
                threshold=dict(line=dict(color=PALETTE[3], width=3), thickness=0.8, value=14.5),
            ),
            title=dict(text="Capital Adequacy<br><span style='font-size:0.6rem'>CBK Min: 14.5%</span>",
                       font=dict(size=12, color=theme.text_muted)),
        ))
        fig_car_g.update_layout(**base_layout(theme, height=250, margin=dict(l=20, r=20, t=55, b=5)))
        st.plotly_chart(fig_car_g, use_container_width=True)

    with col_fsi:
        fsi = research_econ.financial_stability_index()
        fig_fsi_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fsi * 100,
            number=dict(suffix="/100", valueformat=".0f", font=dict(color=theme.text_primary)),
            gauge=dict(
                axis=dict(range=[0, 100], tickfont=dict(color=theme.text_muted, size=9)),
                bar=dict(color=PALETTE[0] if fsi > 0.6 else PALETTE[2] if fsi > 0.35 else PALETTE[3]),
                bgcolor='rgba(255,255,255,0.05)',
                bordercolor=theme.border_default,
                steps=[
                    dict(range=[0, 35], color='rgba(255,51,102,0.08)'),
                    dict(range=[35, 60], color='rgba(245,213,71,0.08)'),
                    dict(range=[60, 100], color='rgba(0,255,136,0.08)'),
                ],
            ),
            title=dict(text="Financial Stability<br><span style='font-size:0.6rem'>Composite Index</span>",
                       font=dict(size=12, color=theme.text_muted)),
        ))
        fig_fsi_g.update_layout(**base_layout(theme, height=250, margin=dict(l=20, r=20, t=55, b=5)))
        st.plotly_chart(fig_fsi_g, use_container_width=True)

    # ── 2. CREDIT CYCLE SCATTER ──────────────────────────────────────────────
    st.markdown(f"<div style='color:{theme.accent_warning}; font-weight:600; "
                f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
                f"CREDIT CYCLE: CREDIT GROWTH vs GDP GROWTH</div>",
                unsafe_allow_html=True)

    # Compute credit growth from performing loans time series
    perf_loans = [f.get("performing_loans", 0) for f in fin_frames]
    credit_growth = [0.0]
    for i in range(1, len(perf_loans)):
        prev = max(perf_loans[i - 1], 0.01)
        credit_growth.append((perf_loans[i] - prev) / prev * 100)

    gdp_growth_vals = [f.get("outcomes", {}).get("gdp_growth", 0) * 100 for f in trajectory
                       if f.get("financial")]
    # Align lengths
    min_len = min(len(credit_growth), len(gdp_growth_vals))
    credit_growth = credit_growth[:min_len]
    gdp_growth_vals = gdp_growth_vals[:min_len]
    t_color = list(range(min_len))

    fig_cycle = go.Figure()
    fig_cycle.add_trace(go.Scatter(
        x=gdp_growth_vals, y=credit_growth,
        mode='lines+markers',
        marker=dict(size=7, color=t_color, colorscale='Plasma',
                    colorbar=dict(title="Quarter", len=0.6), opacity=0.9),
        line=dict(color=PALETTE[1], width=2),
        hovertemplate="GDP: %{x:.1f}%<br>Credit: %{y:.1f}%<br>Q%{marker.color}<extra></extra>",
        name="Credit Cycle",
    ))
    # Start/End markers
    if min_len > 1:
        fig_cycle.add_trace(go.Scatter(
            x=[gdp_growth_vals[0]], y=[credit_growth[0]], mode='markers',
            marker=dict(size=14, color=PALETTE[0], symbol='diamond'), name='Start'))
        fig_cycle.add_trace(go.Scatter(
            x=[gdp_growth_vals[-1]], y=[credit_growth[-1]], mode='markers',
            marker=dict(size=14, color=PALETTE[3], symbol='diamond'), name='End'))
    fig_cycle.add_hline(y=0, line_dash="dot", line_color=theme.text_muted)
    fig_cycle.add_vline(x=0, line_dash="dot", line_color=theme.text_muted)
    fig_cycle.update_layout(**base_layout(theme, height=400,
        title=dict(text="Credit-GDP Cycle Phase Diagram",
                   font=dict(color=theme.text_muted, size=13)),
        xaxis=dict(title="GDP Growth (%)"), yaxis=dict(title="Credit Growth (%)"),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor='rgba(0,0,0,0)')))
    st.plotly_chart(fig_cycle, use_container_width=True)

    st.markdown(f"<div style='font-size:0.75rem; color:{theme.text_muted};'>"
                f"<b>Reading:</b> Top-right = expansion (credit & GDP growing). "
                f"Bottom-left = contraction. Counter-clockwise rotation = typical cycle.</div>",
                unsafe_allow_html=True)

    # ── 3. BANK BALANCE SHEET WATERFALL ──────────────────────────────────────
    st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:600; "
                f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
                f"BANK BALANCE SHEET DECOMPOSITION</div>",
                unsafe_allow_html=True)

    bank = research_econ.bank
    bs_labels = ["Performing\nLoans", "NPLs", "Gov.\nSecurities", "Reserves",
                 "Total\nAssets", "Deposits", "Tier 1\nCapital", "Tier 2\nCapital",
                 "Total\nLiab+Cap"]
    bs_values = [
        bank.performing_loans, bank.non_performing_loans,
        bank.government_securities, bank.reserves,
        0,  # total
        -bank.deposits, -bank.tier1_capital, -bank.tier2_capital,
        0,  # total
    ]
    bs_measures = ["relative", "relative", "relative", "relative", "total",
                   "relative", "relative", "relative", "total"]

    fig_bs = go.Figure(go.Waterfall(
        x=bs_labels, y=bs_values, measure=bs_measures,
        connector=dict(line=dict(color=theme.border_default, width=1)),
        increasing=dict(marker=dict(color=PALETTE[0])),
        decreasing=dict(marker=dict(color=PALETTE[3])),
        totals=dict(marker=dict(color=PALETTE[1])),
        texttemplate="%{y:.1f}", textposition="outside",
        textfont=dict(size=9),
    ))
    fig_bs.update_layout(**base_layout(theme, height=380,
        title=dict(text="Bank Balance Sheet (Assets positive, Liabilities negative)",
                   font=dict(color=theme.text_muted, size=12)),
        yaxis=dict(title="Value")))
    st.plotly_chart(fig_bs, use_container_width=True)

    # ── 4. NPL / CAR / SPREAD TIME SERIES ───────────────────────────────────
    st.markdown(f"<div style='color:{theme.accent_warning}; font-weight:600; "
                f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
                f"FINANCIAL INDICATORS TRAJECTORY</div>",
                unsafe_allow_html=True)

    npl_vals = [f.get("npl_ratio", 0) * 100 for f in fin_frames]
    car_vals = [f.get("car", 0) * 100 for f in fin_frames]
    spread_vals = [f.get("credit_spread", 0) * 100 for f in fin_frames]
    efp_vals = [f.get("efp", 0) * 100 for f in fin_frames]

    fig_fin_ts = make_subplots(rows=2, cols=2,
        subplot_titles=["NPL Ratio (%)", "Capital Adequacy Ratio (%)",
                        "Credit Spread (%)", "External Finance Premium (%)"],
        horizontal_spacing=0.08, vertical_spacing=0.12)

    for col_idx, (vals, color, row, col) in enumerate([
        (npl_vals, PALETTE[3], 1, 1), (car_vals, PALETTE[0], 1, 2),
        (spread_vals, PALETTE[4], 2, 1), (efp_vals, PALETTE[5], 2, 2),
    ]):
        fig_fin_ts.add_trace(go.Scatter(
            x=t_vals, y=vals, mode='lines', showlegend=False,
            line=dict(color=color, width=2.5),
            fill='tozeroy', fillcolor=color + '14',
        ), row=row, col=col)

    # Add threshold lines
    fig_fin_ts.add_hline(y=10, line_dash="dash", line_color=PALETTE[3],
                         annotation_text="10%", row=1, col=1)
    fig_fin_ts.add_hline(y=14.5, line_dash="dash", line_color=PALETTE[0],
                         annotation_text="14.5%", row=1, col=2)

    fig_fin_ts.update_layout(**base_layout(theme, height=500,
        margin=dict(l=40, r=20, t=40, b=30)))
    for r in range(1, 3):
        for c in range(1, 3):
            fig_fin_ts.update_xaxes(gridcolor='rgba(255,255,255,0.03)', row=r, col=c)
            fig_fin_ts.update_yaxes(gridcolor='rgba(255,255,255,0.03)', row=r, col=c)
    st.plotly_chart(fig_fin_ts, use_container_width=True)


def render_open_economy_tab(theme):
    """External sector dashboard — FX, trade, BoP, reserves."""
    trajectory = st.session_state.get("sim_research_trajectory")
    research_econ = st.session_state.get("sim_research_econ")

    if not trajectory or not research_econ:
        st.info("Run the **Research Engine** tab first to see open economy analysis.")
        return
    if not HAS_PLOTLY:
        st.warning("Plotly required.")
        return

    ext_frames = [f.get("external", {}) for f in trajectory if f.get("external")]
    if not ext_frames:
        st.warning("External sector data not available. Enable open economy in Research Engine.")
        return

    t_vals = list(range(len(ext_frames)))
    last_ext = ext_frames[-1]

    # ── 1. GAUGES ROW — REER, Reserves, Vulnerability ────────────────────────
    st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:600; "
                f"font-size:0.85rem; margin-bottom:0.5rem;'>EXTERNAL SECTOR GAUGES</div>",
                unsafe_allow_html=True)

    reer_now = last_ext.get("reer", 100)
    reserves_now = last_ext.get("reserves_months", 0)
    ext_vuln = research_econ.external_vulnerability_index()

    col_reer, col_res, col_vuln = st.columns(3)
    with col_reer:
        fig_reer_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=reer_now,
            number=dict(valueformat=".1f", font=dict(color=theme.text_primary)),
            delta=dict(reference=ext_frames[0].get("reer", 100), valueformat=".1f",
                       increasing=dict(color=PALETTE[2]),
                       decreasing=dict(color=PALETTE[3])),
            gauge=dict(
                axis=dict(range=[50, 150], tickfont=dict(color=theme.text_muted, size=9)),
                bar=dict(color=PALETTE[1]),
                bgcolor='rgba(255,255,255,0.05)',
                bordercolor=theme.border_default,
                steps=[
                    dict(range=[50, 85], color='rgba(255,51,102,0.08)'),
                    dict(range=[85, 115], color='rgba(0,255,136,0.08)'),
                    dict(range=[115, 150], color='rgba(245,213,71,0.08)'),
                ],
                threshold=dict(line=dict(color=theme.text_muted, width=2), thickness=0.8, value=100),
            ),
            title=dict(text="REER Index<br><span style='font-size:0.6rem'>Equilibrium: 100</span>",
                       font=dict(size=12, color=theme.text_muted)),
        ))
        fig_reer_g.update_layout(**base_layout(theme, height=250, margin=dict(l=20, r=20, t=55, b=5)))
        st.plotly_chart(fig_reer_g, use_container_width=True)

    with col_res:
        fig_res_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=reserves_now,
            number=dict(suffix=" mo", valueformat=".1f", font=dict(color=theme.text_primary)),
            gauge=dict(
                axis=dict(range=[0, 12], ticksuffix=" mo",
                          tickfont=dict(color=theme.text_muted, size=9)),
                bar=dict(color=PALETTE[0] if reserves_now >= 4 else PALETTE[3]),
                bgcolor='rgba(255,255,255,0.05)',
                bordercolor=theme.border_default,
                steps=[
                    dict(range=[0, 3], color='rgba(255,51,102,0.08)'),
                    dict(range=[3, 4], color='rgba(245,213,71,0.08)'),
                    dict(range=[4, 12], color='rgba(0,255,136,0.08)'),
                ],
                threshold=dict(line=dict(color=PALETTE[3], width=3), thickness=0.8, value=4),
            ),
            title=dict(text="Reserve Cover<br><span style='font-size:0.6rem'>CBK Min: 4 months</span>",
                       font=dict(size=12, color=theme.text_muted)),
        ))
        fig_res_g.update_layout(**base_layout(theme, height=250, margin=dict(l=20, r=20, t=55, b=5)))
        st.plotly_chart(fig_res_g, use_container_width=True)

    with col_vuln:
        fig_vuln_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=ext_vuln * 100,
            number=dict(suffix="/100", valueformat=".0f", font=dict(color=theme.text_primary)),
            gauge=dict(
                axis=dict(range=[0, 100], tickfont=dict(color=theme.text_muted, size=9)),
                bar=dict(color=PALETTE[3] if ext_vuln > 0.6 else PALETTE[2] if ext_vuln > 0.35 else PALETTE[0]),
                bgcolor='rgba(255,255,255,0.05)',
                bordercolor=theme.border_default,
                steps=[
                    dict(range=[0, 35], color='rgba(0,255,136,0.08)'),
                    dict(range=[35, 60], color='rgba(245,213,71,0.08)'),
                    dict(range=[60, 100], color='rgba(255,51,102,0.08)'),
                ],
            ),
            title=dict(text="Ext. Vulnerability<br><span style='font-size:0.6rem'>Composite Index</span>",
                       font=dict(size=12, color=theme.text_muted)),
        ))
        fig_vuln_g.update_layout(**base_layout(theme, height=250, margin=dict(l=20, r=20, t=55, b=5)))
        st.plotly_chart(fig_vuln_g, use_container_width=True)

    # ── 2. BALANCE OF PAYMENTS WATERFALL ─────────────────────────────────────
    st.markdown(f"<div style='color:{theme.accent_warning}; font-weight:600; "
                f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
                f"BALANCE OF PAYMENTS DECOMPOSITION</div>",
                unsafe_allow_html=True)

    tb = last_ext.get("trade_balance", 0)
    remit = last_ext.get("remittances", 0)
    ca = last_ext.get("current_account", 0)
    ka = last_ext.get("capital_account", 0)
    inv_income = ca - tb - remit  # residual

    bop_labels = ["Exports\n− Imports", "Remittances", "Investment\nIncome",
                  "Current\nAccount", "Capital\nAccount", "Overall\nBoP"]
    bop_values = [tb, remit, inv_income, 0, ka, 0]
    bop_measures = ["relative", "relative", "relative", "total", "relative", "total"]

    fig_bop = go.Figure(go.Waterfall(
        x=bop_labels, y=bop_values, measure=bop_measures,
        connector=dict(line=dict(color=theme.border_default, width=1)),
        increasing=dict(marker=dict(color=PALETTE[0])),
        decreasing=dict(marker=dict(color=PALETTE[3])),
        totals=dict(marker=dict(color=PALETTE[1])),
        texttemplate="%{y:.2f}", textposition="outside",
        textfont=dict(size=10),
    ))
    fig_bop.update_layout(**base_layout(theme, height=380,
        title=dict(text="BoP Components (Final Period)",
                   font=dict(color=theme.text_muted, size=13)),
        yaxis=dict(title="Value")))
    st.plotly_chart(fig_bop, use_container_width=True)

    # ── 3. TWIN DEFICIT SCATTER ──────────────────────────────────────────────
    st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:600; "
                f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
                f"TWIN DEFICIT ANALYSIS: FISCAL vs CURRENT ACCOUNT</div>",
                unsafe_allow_html=True)

    fiscal_deficit_vals = [
        f.get("outcomes", {}).get("fiscal_deficit_gdp", 0) * 100
        for f in trajectory if f.get("external")
    ]
    ca_gdp_vals = [
        f.get("external", {}).get("current_account", 0) /
        max(f.get("outcomes", {}).get("gdp", 100), 1) * 100
        for f in trajectory if f.get("external")
    ]
    min_twin = min(len(fiscal_deficit_vals), len(ca_gdp_vals))
    fiscal_deficit_vals = fiscal_deficit_vals[:min_twin]
    ca_gdp_vals = ca_gdp_vals[:min_twin]
    t_twin = list(range(min_twin))

    if min_twin > 1:
        fig_twin = go.Figure()
        fig_twin.add_trace(go.Scatter(
            x=fiscal_deficit_vals, y=ca_gdp_vals,
            mode='lines+markers',
            marker=dict(size=7, color=t_twin, colorscale='Plasma',
                        colorbar=dict(title="Quarter", len=0.6), opacity=0.9),
            line=dict(color=PALETTE[4], width=2),
            hovertemplate="Fiscal: %{x:.1f}%<br>CA/GDP: %{y:.1f}%<extra></extra>",
            name="Twin Deficit Path",
        ))
        # Quadrant labels
        fig_twin.add_annotation(x=0.02, y=0.98, xref='paper', yref='paper',
            text="<b>CA Surplus<br>Fiscal Deficit</b>", showarrow=False,
            font=dict(color=PALETTE[2], size=9), align='left')
        fig_twin.add_annotation(x=0.98, y=0.02, xref='paper', yref='paper',
            text="<b>CA Deficit<br>Fiscal Surplus</b>", showarrow=False,
            font=dict(color=PALETTE[2], size=9), align='right')
        fig_twin.add_annotation(x=0.02, y=0.02, xref='paper', yref='paper',
            text="<b style='color:" + PALETTE[3] + "'>TWIN DEFICIT<br>ZONE</b>", showarrow=False,
            font=dict(size=10), align='left')
        fig_twin.add_hline(y=0, line_dash="dot", line_color=theme.text_muted)
        fig_twin.add_vline(x=0, line_dash="dot", line_color=theme.text_muted)
        # Start/End
        fig_twin.add_trace(go.Scatter(
            x=[fiscal_deficit_vals[0]], y=[ca_gdp_vals[0]], mode='markers',
            marker=dict(size=14, color=PALETTE[0], symbol='diamond'), name='Start'))
        fig_twin.add_trace(go.Scatter(
            x=[fiscal_deficit_vals[-1]], y=[ca_gdp_vals[-1]], mode='markers',
            marker=dict(size=14, color=PALETTE[3], symbol='diamond'), name='End'))
        fig_twin.update_layout(**base_layout(theme, height=430,
            title=dict(text="Twin Deficit Path (Fiscal Deficit vs CA/GDP)",
                       font=dict(color=theme.text_muted, size=13)),
            xaxis=dict(title="Fiscal Deficit (% GDP)"),
            yaxis=dict(title="Current Account (% GDP)"),
            legend=dict(orientation="h", y=1.08, x=0, bgcolor='rgba(0,0,0,0)')))
        st.plotly_chart(fig_twin, use_container_width=True)

    # ── 4. REER vs INFLATION TWIN PANEL ──────────────────────────────────────
    st.markdown(f"<div style='color:{theme.accent_warning}; font-weight:600; "
                f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
                f"REER &amp; INFLATION CO-MOVEMENT</div>",
                unsafe_allow_html=True)

    reer_vals = [f.get("reer", 100) for f in ext_frames]
    inflation_vals = [f.get("outcomes", {}).get("inflation", 0) * 100 for f in trajectory
                      if f.get("external")]
    min_ri = min(len(reer_vals), len(inflation_vals))

    fig_ri = make_subplots(specs=[[{"secondary_y": True}]])
    fig_ri.add_trace(go.Scatter(
        x=list(range(min_ri)), y=reer_vals[:min_ri], name="REER",
        line=dict(color=PALETTE[1], width=3),
        fill='tozeroy', fillcolor=PALETTE[1] + '14',
    ), secondary_y=False)
    fig_ri.add_trace(go.Scatter(
        x=list(range(min_ri)), y=inflation_vals[:min_ri], name="Inflation (%)",
        line=dict(color=PALETTE[3], width=2.5),
    ), secondary_y=True)
    fig_ri.add_hline(y=100, line_dash="dash", line_color=theme.text_muted,
                     annotation_text="REER Eq.", secondary_y=False)
    fig_ri.update_layout(**base_layout(theme, height=380,
        title=dict(text="REER & Inflation — Exchange Rate Pass-Through",
                   font=dict(color=theme.text_muted, size=13)),
        xaxis=dict(title="Quarter"),
        legend=dict(orientation="h", y=1.08, x=0, bgcolor='rgba(0,0,0,0)')))
    fig_ri.update_yaxes(title_text="REER Index", secondary_y=False)
    fig_ri.update_yaxes(title_text="Inflation (%)", secondary_y=True)
    st.plotly_chart(fig_ri, use_container_width=True)

    # ── 5. RESERVE ADEQUACY TRAJECTORY ───────────────────────────────────────
    st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:600; "
                f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
                f"RESERVE ADEQUACY TRAJECTORY</div>",
                unsafe_allow_html=True)

    res_vals = [f.get("reserves_months", 0) for f in ext_frames]
    fig_res = go.Figure()
    fig_res.add_trace(go.Scatter(
        x=t_vals, y=res_vals, mode='lines',
        name="Reserves (months)",
        line=dict(color=PALETTE[0], width=3),
        fill='tozeroy',
        fillcolor='rgba(0,255,136,0.08)',
    ))
    # Color the danger zone
    fig_res.add_hrect(y0=0, y1=4, fillcolor='rgba(255,51,102,0.06)',
                      line_width=0, annotation_text="Below CBK minimum",
                      annotation_position="bottom left")
    fig_res.add_hline(y=4.0, line_dash="dash", line_color=PALETTE[3],
                      annotation_text="CBK Minimum (4 months)")
    fig_res.update_layout(**base_layout(theme, height=320,
        title=dict(text="Foreign Reserve Cover Over Time",
                   font=dict(color=theme.text_muted, size=13)),
        xaxis=dict(title="Quarter"), yaxis=dict(title="Months of Import Cover")))
    st.plotly_chart(fig_res, use_container_width=True)


def render_research_engine_tab(theme, SFCEconomy, SFCConfig, calibrate_from_data):
    """Unified research-grade SFC engine with all 7 upgrade modules."""
    st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:600; "
                f"font-size:0.85rem; margin-bottom:0.5rem;'>"
                f"RESEARCH-GRADE SFC ENGINE</div>",
                unsafe_allow_html=True)
    st.caption("Runs the unified economy with: IO Sectors + Heterogeneous Agents + "
               "Financial Accelerator + Open Economy — all feedback loops active.")

    # ── Module toggles ────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        enable_io = st.checkbox("IO Sectors", value=True, key="re_io")
    with c2:
        enable_het = st.checkbox("Heterogeneous Agents", value=True, key="re_het")
    with c3:
        enable_fin = st.checkbox("Financial Accelerator", value=True, key="re_fin")
    with c4:
        enable_open = st.checkbox("Open Economy", value=True, key="re_open")

    c5, c6 = st.columns(2)
    with c5:
        steps = st.slider("Simulation Quarters", 10, 200, 50, 5, key="re_steps")
    with c6:
        seed = st.number_input("Random Seed", value=42, min_value=0, max_value=9999, key="re_seed")

    # ── Run Button ────────────────────────────────────────────────────────────
    if st.button("RUN RESEARCH ENGINE", type="primary", key="re_run",
                 use_container_width=True):
        with st.spinner("Running research-grade SFC with all active modules..."):
            try:
                from scarcity.simulation.research_sfc import (
                    ResearchSFCEconomy, ResearchSFCConfig,
                )
                from scarcity.simulation.io_structure import default_kenya_io_config
                from scarcity.simulation.heterogeneous import default_kenya_heterogeneous_config
                from scarcity.simulation.financial_accelerator import FinancialAcceleratorConfig
                from scarcity.simulation.open_economy import default_kenya_open_economy_config

                calib = st.session_state.get("sim_calibration")
                base_cfg = calib.config if calib else SFCConfig()
                base_cfg.steps = steps

                scenario = st.session_state.get("_sim_scenario_obj")
                if scenario and hasattr(scenario, 'build_shock_vectors'):
                    base_cfg.shock_vectors = scenario.build_shock_vectors(steps)

                research_cfg = ResearchSFCConfig(
                    sfc=base_cfg,
                    io=default_kenya_io_config(),
                    heterogeneous=default_kenya_heterogeneous_config(),
                    financial=FinancialAcceleratorConfig(),
                    open_economy=default_kenya_open_economy_config(),
                    enable_io=enable_io,
                    enable_heterogeneous=enable_het,
                    enable_financial=enable_fin,
                    enable_open_economy=enable_open,
                    seed=int(seed),
                )

                econ = ResearchSFCEconomy(research_cfg)
                econ.initialize(100.0)

                progress = st.progress(0, text="Running research engine...")
                for i in range(steps):
                    econ.step()
                    if (i + 1) % max(1, steps // 20) == 0:
                        progress.progress((i + 1) / steps,
                                          text=f"Quarter {i + 1}/{steps}")
                progress.empty()

                st.session_state["sim_research_trajectory"] = econ.trajectory
                st.session_state["sim_research_econ"] = econ
                st.session_state["sim_research_summary"] = econ.summary()

                st.success(f"Research engine completed: {steps} quarters, "
                           f"{len(econ.trajectory)} frames recorded.")

            except Exception as e:
                st.error(f"Research engine error: {e}")
                import traceback
                st.code(traceback.format_exc())
                return

    # ── Results section ───────────────────────────────────────────────────────
    summary = st.session_state.get("sim_research_summary")
    research_econ = st.session_state.get("sim_research_econ")
    trajectory = st.session_state.get("sim_research_trajectory")
    if not summary or not research_econ:
        return
    if not HAS_PLOTLY:
        st.warning("Plotly required for research charts.")
        return

    st.markdown("---")

    # ── 1. RADAR SPIDER — Module Health ────────────────────────────────────
    st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:600; "
                f"font-size:0.85rem; margin-bottom:0.5rem;'>MULTI-MODULE HEALTH RADAR</div>",
                unsafe_allow_html=True)

    # Compute module scores (0-100)
    fin_stability = research_econ.financial_stability_index() if research_econ.config.enable_financial else 0.5
    ext_vulnerability = 1.0 - research_econ.external_vulnerability_index() if research_econ.config.enable_open_economy else 0.5
    macro_score = max(0, min(1, 1.0 - abs(summary.get("inflation", 0.03) - 0.05) / 0.10))
    growth_score = max(0, min(1, (summary.get("gdp_growth", 0) + 0.05) / 0.15))
    ineq_score = max(0, min(1, 1.0 - summary.get("inequality", {}).get("gini", 0.4)))
    sector_div = 1.0 - max(summary.get("sectors", {}).values(), default=0.5)

    radar_cats = ["Financial\nStability", "External\nResilience", "Macro\nBalance",
                  "Growth\nMomentum", "Equality", "Sector\nDiversity"]
    radar_vals = [fin_stability * 100, ext_vulnerability * 100, macro_score * 100,
                  growth_score * 100, ineq_score * 100, sector_div * 100]
    radar_vals_closed = radar_vals + [radar_vals[0]]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=radar_vals_closed,
        theta=radar_cats + [radar_cats[0]],
        fill='toself',
        fillcolor='rgba(0,255,136,0.15)',
        line=dict(color=PALETTE[0], width=3),
        name='Current State',
        hovertemplate='%{theta}: %{r:.0f}/100<extra></extra>',
    ))
    # Add benchmark ring at 50
    fig_radar.add_trace(go.Scatterpolar(
        r=[50] * (len(radar_cats) + 1),
        theta=radar_cats + [radar_cats[0]],
        line=dict(color=theme.text_muted, width=1, dash='dot'),
        name='Threshold (50)',
        showlegend=True,
    ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(255,255,255,0.08)',
                            tickfont=dict(size=9, color=theme.text_muted)),
            angularaxis=dict(gridcolor='rgba(255,255,255,0.08)',
                             tickfont=dict(size=10, color=theme.text_muted)),
        ),
        **base_layout(theme, height=420,
            title=dict(text="Economy Health Radar", font=dict(color=theme.text_muted, size=13)),
            legend=dict(orientation="h", y=-0.05, x=0.3, bgcolor='rgba(0,0,0,0)')),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── 2. GDP DECOMPOSITION WATERFALL ─────────────────────────────────────
    st.markdown(f"<div style='color:{theme.accent_warning}; font-weight:600; "
                f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
                f"GDP DEMAND DECOMPOSITION: Y = C + I + G + NX</div>",
                unsafe_allow_html=True)

    gdp = summary.get("gdp", 100)
    # Estimate components from engine state
    hh_consumption = float(research_econ.economy.households.assets.get("deposits", 0)) * 0.3
    if not hh_consumption:
        hh_consumption = gdp * 0.60
    investment = float(research_econ.bank.performing_loans * 0.08) if research_econ.config.enable_financial else gdp * 0.18
    gov_spending = gdp * research_econ.economy.config.gov_spending_ratio
    net_exports = float(research_econ.external.trade_balance) if research_econ.config.enable_open_economy else 0.0
    residual = gdp - hh_consumption - investment - gov_spending - net_exports

    wf_labels = ["Consumption (C)", "Investment (I)", "Gov. Spending (G)", "Net Exports (NX)", "Residual", "GDP"]
    wf_values = [hh_consumption, investment, gov_spending, net_exports, residual, 0]
    wf_measures = ["relative", "relative", "relative", "relative", "relative", "total"]

    fig_wf = go.Figure(go.Waterfall(
        x=wf_labels, y=wf_values, measure=wf_measures,
        connector=dict(line=dict(color=theme.border_default, width=1)),
        increasing=dict(marker=dict(color=PALETTE[0])),
        decreasing=dict(marker=dict(color=PALETTE[3])),
        totals=dict(marker=dict(color=PALETTE[1])),
        texttemplate="%{y:.1f}", textposition="outside",
        hovertemplate="%{x}: %{y:.2f}<extra></extra>",
    ))
    fig_wf.update_layout(**base_layout(theme, height=380,
        title=dict(text="GDP Components (Waterfall)", font=dict(color=theme.text_muted, size=13)),
        yaxis=dict(title="Value")))
    st.plotly_chart(fig_wf, use_container_width=True)

    # ── 3. POLICY TRANSMISSION SANKEY ──────────────────────────────────────
    if trajectory and len(trajectory) > 2:
        st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:600; "
                    f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
                    f"POLICY TRANSMISSION MECHANISM</div>",
                    unsafe_allow_html=True)

        # Build transmission flow: Policy Rate → Bank → Credit → Sectors → Households → GDP
        rate = float(research_econ.economy.interest_rate)
        npl = float(research_econ.bank.npl_ratio) if research_econ.config.enable_financial else 0.05
        spread = float(research_econ.bank.credit_spread) if research_econ.config.enable_financial else 0.02
        efp = float(research_econ.bank.external_finance_premium) if research_econ.config.enable_financial else 0.0

        nodes = ["Policy Rate", "Bank Capital", "Credit Supply", "NPL Drag",
                 "Agriculture", "Manufacturing", "Services", "Mining", "Construction",
                 "Q1 (Poorest)", "Q3 (Middle)", "Q5 (Richest)", "GDP Output"]
        node_colors = [PALETTE[1], PALETTE[0], PALETTE[0], PALETTE[3],
                       PALETTE[4], PALETTE[5], PALETTE[6], PALETTE[7], PALETTE[8],
                       PALETTE[3], PALETTE[2], PALETTE[0], PALETTE[1]]

        # Compute flow magnitudes (normalised 0-1 for visual proportionality)
        src, tgt, vals, link_colors = [], [], [], []
        flow_data = [
            (0, 1, rate * 10, PALETTE[1]),          # Rate → Bank
            (0, 3, npl * 10, PALETTE[3]),            # Rate → NPL
            (1, 2, (1 - spread) * 5, PALETTE[0]),    # Bank → Credit
            (3, 2, npl * 5, PALETTE[3]),              # NPL → Credit (drag)
        ]
        # Credit → Sectors (by share)
        sector_shares = summary.get("sectors", {"agriculture": 0.22, "manufacturing": 0.08,
                                                 "services": 0.53, "mining": 0.04, "construction": 0.13})
        for i, (name, share) in enumerate(sector_shares.items()):
            flow_data.append((2, 4 + i, share * 10, PALETTE[4 + i]))
        # Sectors → Households
        for i in range(5):
            flow_data.append((4 + i, 9, 0.4, 'rgba(255,51,102,0.3)'))   # → Q1
            flow_data.append((4 + i, 10, 0.8, 'rgba(245,213,71,0.3)'))  # → Q3
            flow_data.append((4 + i, 11, 1.2, 'rgba(0,255,136,0.3)'))   # → Q5
        # Households → GDP
        for hh_idx, hh_wt in [(9, 2), (10, 4), (11, 6)]:
            flow_data.append((hh_idx, 12, hh_wt, PALETTE[1]))

        for s, t, v, c in flow_data:
            src.append(s); tgt.append(t); vals.append(max(v, 0.01)); link_colors.append(c + '40')

        fig_sankey = go.Figure(go.Sankey(
            arrangement='snap',
            node=dict(label=nodes, color=node_colors, pad=12, thickness=20,
                      line=dict(color=theme.border_default, width=0.5)),
            link=dict(source=src, target=tgt, value=vals, color=link_colors),
        ))
        fig_sankey.update_layout(**base_layout(theme, height=480,
            title=dict(text="Policy Transmission: Rate → Bank → Sectors → Households → GDP",
                       font=dict(color=theme.text_muted, size=13))))
        st.plotly_chart(fig_sankey, use_container_width=True)

    # ── 4. MACRO TRAJECTORY SPARKLINES ROW ────────────────────────────────
    if trajectory and len(trajectory) > 4:
        st.markdown(f"<div style='color:{theme.accent_warning}; font-weight:600; "
                    f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
                    f"MACRO TRAJECTORY SPARKLINES</div>",
                    unsafe_allow_html=True)

        t_vals = list(range(len(trajectory)))
        spark_dims = [
            ("GDP Growth", [f.get("outcomes", {}).get("gdp_growth", 0) * 100 for f in trajectory], "%"),
            ("Inflation", [f.get("outcomes", {}).get("inflation", 0) * 100 for f in trajectory], "%"),
            ("Unemployment", [f.get("outcomes", {}).get("unemployment", 0) * 100 for f in trajectory], "%"),
            ("Interest Rate", [f.get("outcomes", {}).get("interest_rate", 0) * 100 for f in trajectory], "%"),
        ]

        fig_spark = make_subplots(rows=1, cols=4, subplot_titles=[s[0] for s in spark_dims],
                                   horizontal_spacing=0.06)
        for i, (name, vals, unit) in enumerate(spark_dims):
            col = i + 1
            color = PALETTE[i]
            fig_spark.add_trace(go.Scatter(
                x=t_vals, y=vals, mode='lines', name=name,
                line=dict(color=color, width=2.5), showlegend=False,
                fill='tozeroy', fillcolor=color.replace(')', ',0.08)').replace('rgb', 'rgba') if color.startswith('rgb') else color + '14',
            ), row=1, col=col)
            # Current value annotation
            if vals:
                fig_spark.add_annotation(
                    x=t_vals[-1], y=vals[-1], text=f"<b>{vals[-1]:.1f}{unit}</b>",
                    showarrow=False, font=dict(color=color, size=11),
                    xref=f"x{'' if col == 1 else col}", yref=f"y{'' if col == 1 else col}",
                )

        fig_spark.update_layout(**base_layout(theme, height=200,
            margin=dict(l=30, r=10, t=35, b=25)))
        for i in range(1, 5):
            ax_suffix = "" if i == 1 else str(i)
            fig_spark.update_xaxes(showticklabels=False, gridcolor='rgba(255,255,255,0.03)', row=1, col=i)
            fig_spark.update_yaxes(gridcolor='rgba(255,255,255,0.03)', row=1, col=i)
        st.plotly_chart(fig_spark, use_container_width=True)

    # ── 5. STRESS TEST PANEL ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"<div style='color:{theme.accent_danger}; font-weight:600; "
                f"font-size:0.85rem; margin-bottom:0.5rem;'>COMBINED STRESS TEST</div>",
                unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        npl_shock = st.slider("NPL Shock", 0.0, 0.20, 0.05, 0.01, key="re_npl_shock")
    with sc2:
        rate_shock = st.slider("Rate Shock", 0.0, 0.10, 0.02, 0.005, key="re_rate_shock")
    with sc3:
        fx_shock = st.slider("FX Shock", 0.0, 0.30, 0.10, 0.01, key="re_fx_shock")
    with sc4:
        deposit_run = st.slider("Deposit Run", 0.0, 0.20, 0.05, 0.01, key="re_deposit_run")

    if st.button("Run Stress Test", key="re_stress"):
        try:
            stress = research_econ.stress_test(
                npl_shock=npl_shock, rate_shock=rate_shock,
                fx_shock=fx_shock, deposit_run=deposit_run,
            )
            st.session_state["sim_research_stress"] = stress
        except Exception as e:
            st.error(f"Stress test error: {e}")

    stress = st.session_state.get("sim_research_stress")
    if stress and HAS_PLOTLY:
        # Stress results as grouped bar chart
        categories, pre_vals, post_vals = [], [], []

        fin_stress = stress.get("financial", {})
        if fin_stress:
            categories += ["CAR"]
            pre_vals += [fin_stress.get("pre_car", 0) * 100]
            post_vals += [fin_stress.get("post_car", 0) * 100]

        ext_stress = stress.get("external", {})
        if ext_stress:
            categories += ["REER"]
            pre_vals += [ext_stress.get("pre_reer", 100)]
            post_vals += [ext_stress.get("post_reer", 100)]

        dist_stress = stress.get("distributional", {})
        if dist_stress:
            categories += ["Q1 Impact", "Q5 Impact"]
            pre_vals += [0, 0]
            post_vals += [dist_stress.get("rate_shock_q1_impact", 0) * 100,
                          dist_stress.get("rate_shock_q5_impact", 0) * 100]

        if categories:
            fig_stress = go.Figure()
            fig_stress.add_trace(go.Bar(x=categories, y=pre_vals, name="Pre-Shock",
                                        marker_color=PALETTE[0], text=[f"{v:.1f}" for v in pre_vals],
                                        textposition='auto'))
            fig_stress.add_trace(go.Bar(x=categories, y=post_vals, name="Post-Shock",
                                        marker_color=PALETTE[3], text=[f"{v:.1f}" for v in post_vals],
                                        textposition='auto'))
            # Threshold lines
            if fin_stress:
                fig_stress.add_hline(y=14.5, line_dash="dash", line_color=theme.accent_danger,
                                     annotation_text="CBK Min CAR (14.5%)")

            fig_stress.update_layout(**base_layout(theme, height=380,
                title=dict(text="Stress Test: Pre vs Post Shock", font=dict(color=theme.text_muted, size=13)),
                barmode='group',
                legend=dict(orientation="h", y=1.08, x=0, bgcolor='rgba(0,0,0,0)')))
            st.plotly_chart(fig_stress, use_container_width=True)

    # Hint to check other tabs
    st.markdown(f"<div style='font-size:0.75rem; color:{theme.text_muted}; "
                f"margin-top:1rem; text-align:center;'>"
                f"&#8594; See <b>IO Sectors</b>, <b>Inequality</b>, <b>Financial</b>, "
                f"and <b>Open Economy</b> tabs for detailed charts.</div>",
                unsafe_allow_html=True)
