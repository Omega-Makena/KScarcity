"""Advanced analysis tabs: Phase Explorer, IRF, Flow Sankey, Monte Carlo, Stress Matrix, Parameter Surface."""

from ._shared import (st, pd, np, go, make_subplots, HAS_DATA_STACK, HAS_PLOTLY, PALETTE, base_layout, discover_dimensions, flat_dim_options, dim_label, extract_dim)


def render_phase_explorer_tab(theme):
    trajectory = st.session_state.get("sim_trajectory")
    if not trajectory or len(trajectory) < 4:
        st.info("Run a simulation to explore phase diagrams.")
        return
    if not HAS_PLOTLY:
        st.warning("Plotly required for phase diagrams.")
        return

    dims = discover_dimensions(trajectory)
    all_opts = flat_dim_options(dims)
    if len(all_opts) < 2:
        st.info("Not enough discoverable dimensions for a phase diagram.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        x_dim = st.selectbox("X axis", all_opts, index=0, key="phase_x",
                              format_func=lambda s: dim_label(s.split("::")[-1]))
    with c2:
        y_dim = st.selectbox("Y axis", all_opts,
                              index=min(1, len(all_opts) - 1), key="phase_y",
                              format_func=lambda s: dim_label(s.split("::")[-1]))
    with c3:
        z_options = ["(2D — no Z axis)"] + all_opts
        z_dim = st.selectbox("Z axis (optional)", z_options, index=0, key="phase_z",
                              format_func=lambda s: "(2D)" if s.startswith("(")
                              else dim_label(s.split("::")[-1]))

    t_vals = [f.get("t", 0) for f in trajectory]
    x_vals = [extract_dim(x_dim, f) for f in trajectory]
    y_vals = [extract_dim(y_dim, f) for f in trajectory]
    x_label = dim_label(x_dim.split("::")[-1])
    y_label = dim_label(y_dim.split("::")[-1])

    is_3d = z_dim and not z_dim.startswith("(")

    if is_3d:
        z_vals = [extract_dim(z_dim, f) for f in trajectory]
        z_label = dim_label(z_dim.split("::")[-1])

        fig = go.Figure()
        fig.add_trace(go.Scatter3d(
            x=x_vals, y=y_vals, z=z_vals,
            mode='lines+markers',
            marker=dict(size=4, color=t_vals, colorscale='Plasma',
                        colorbar=dict(title="Quarter", len=0.6), opacity=0.9),
            line=dict(color=theme.accent_primary, width=3),
            hovertemplate=(
                f"{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<br>"
                f"{z_label}: %{{z:.3f}}<extra></extra>"
            ),
            name="Trajectory",
        ))
        fig.add_trace(go.Scatter3d(
            x=[x_vals[0]], y=[y_vals[0]], z=[z_vals[0]],
            mode='markers', marker=dict(size=10, color=theme.accent_success, symbol='diamond'),
            name='Start', showlegend=True,
        ))
        fig.add_trace(go.Scatter3d(
            x=[x_vals[-1]], y=[y_vals[-1]], z=[z_vals[-1]],
            mode='markers', marker=dict(size=10, color=theme.accent_danger, symbol='diamond'),
            name='End', showlegend=True,
        ))
        fig.update_layout(
            scene=dict(xaxis_title=x_label, yaxis_title=y_label, zaxis_title=z_label,
                       bgcolor="rgba(0,0,0,0)"),
            **base_layout(theme, height=560,
                title=dict(text="3D Phase Space Trajectory",
                           font=dict(color=theme.text_muted, size=13))),
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode='lines+markers',
            marker=dict(size=6, color=t_vals, colorscale='Plasma',
                        colorbar=dict(title="Quarter"), opacity=0.8),
            line=dict(color=theme.accent_primary, width=2),
            hovertemplate=(
                f"{x_label}: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<extra></extra>"
            ),
            name="Trajectory",
        ))
        fig.add_trace(go.Scatter(
            x=[x_vals[0]], y=[y_vals[0]], mode='markers',
            marker=dict(size=12, color=theme.accent_success, symbol='diamond'),
            name='Start', showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=[x_vals[-1]], y=[y_vals[-1]], mode='markers',
            marker=dict(size=12, color=theme.accent_danger, symbol='diamond'),
            name='End', showlegend=True,
        ))
        fig.update_layout(**base_layout(theme, height=480,
            title=dict(text="2D Phase Diagram",
                       font=dict(color=theme.text_muted, size=13)),
            xaxis=dict(title=x_label), yaxis=dict(title=y_label)))

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">
        <b>Reading:</b>  The path traces how the economy evolves through the chosen
        state space over time.  Colour encodes the quarter — early periods are dark,
        later periods bright.  <span style="color:{theme.accent_success}">&#x25C6;</span>
        = start, <span style="color:{theme.accent_danger}">&#x25C6;</span> = end.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 7: IMPULSE RESPONSE FUNCTIONS — auto-detected deviation from baseline
# ═════════════════════════════════════════════════════════════════════════════

def render_irf_tab(theme):
    trajectory = st.session_state.get("sim_trajectory")
    if not trajectory or len(trajectory) < 6:
        st.info("Run a simulation with ≥ 6 quarters to see impulse responses.")
        return
    if not HAS_PLOTLY:
        st.warning("Plotly required for IRF charts.")
        return

    # Determine shock onset dynamically
    scenario = st.session_state.get("_sim_scenario_obj")
    onset = scenario.shock_onset if scenario and hasattr(scenario, "shock_onset") else 5

    # Auto-discover outcome dimensions
    outcome_keys = sorted({k for f in trajectory for k in f.get("outcomes", {})
                           if not k.startswith("breach_")})
    if not outcome_keys:
        st.info("No outcome dimensions found in trajectory frames.")
        return

    selected = st.multiselect(
        "Dimensions to display", outcome_keys,
        default=outcome_keys[:min(6, len(outcome_keys))],
        format_func=dim_label, key="irf_dims",
    )
    if not selected:
        return

    # Baseline = average of pre-shock frames
    pre_shock = [f for f in trajectory if f["t"] < onset]
    if not pre_shock:
        pre_shock = trajectory[:3]

    baseline = {}
    for dim in selected:
        vals = [f.get("outcomes", {}).get(dim, 0) for f in pre_shock]
        baseline[dim] = float(np.mean(vals)) if vals else 0.0

    post_shock = [f for f in trajectory if f["t"] >= onset]
    t_relative = [f["t"] - onset for f in post_shock]

    view_mode = st.radio("View", ["2D Lines", "3D Surface"], horizontal=True, key="irf_view")

    if view_mode == "3D Surface" and len(selected) >= 2:
        z_matrix = []
        for dim in selected:
            base = baseline.get(dim, 0)
            irfs = [(f.get("outcomes", {}).get(dim, 0) - base) for f in post_shock]
            if abs(base) > 1e-8:
                irfs = [v / abs(base) * 100 for v in irfs]
            z_matrix.append(irfs)

        fig = go.Figure(go.Surface(
            x=t_relative,
            y=list(range(len(selected))),
            z=z_matrix,
            colorscale='RdBu_r',
            colorbar=dict(title="% Dev"),
            hovertemplate="t+%{x}<br>Dim idx: %{y}<br>IRF: %{z:.2f}%<extra></extra>",
        ))
        fig.update_layout(
            scene=dict(
                xaxis_title="Quarters After Shock",
                yaxis_title="Dimension",
                zaxis_title="% Deviation from Baseline",
                yaxis=dict(
                    tickvals=list(range(len(selected))),
                    ticktext=[dim_label(d)[:15] for d in selected],
                ),
                bgcolor="rgba(0,0,0,0)",
            ),
            **base_layout(theme, height=560,
                title=dict(text="3D Impulse Response Surface",
                           font=dict(color=theme.text_muted, size=13))),
        )
    else:
        fig = make_subplots(rows=1, cols=1)
        for idx, dim in enumerate(selected):
            base = baseline.get(dim, 0)
            irfs = [(f.get("outcomes", {}).get(dim, 0) - base) for f in post_shock]
            if abs(base) > 1e-8:
                irfs = [v / abs(base) * 100 for v in irfs]
            fig.add_trace(go.Scatter(
                x=t_relative, y=irfs, mode='lines',
                name=dim_label(dim),
                line=dict(color=PALETTE[idx % len(PALETTE)], width=2.5),
            ))
        fig.add_hline(y=0, line_dash="dot", line_color=theme.text_muted)
        fig.update_layout(**base_layout(theme, height=460,
            title=dict(text="Impulse Response Functions (% deviation from baseline)",
                       font=dict(color=theme.text_muted, size=13)),
            xaxis=dict(title="Quarters After Shock"),
            yaxis=dict(title="% Deviation"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                        bgcolor='rgba(0,0,0,0)')))

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">
        <b>Reading:</b> Each line shows how a dimension deviates from its pre-shock
        baseline (t<{onset}).  Positive = above baseline, negative = below.
        Values are normalised to percentage deviation where possible.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 8: FLOW DYNAMICS — Sankey + waterfall + 3D surface
# ═════════════════════════════════════════════════════════════════════════════

def render_flow_sankey_tab(theme):
    trajectory = st.session_state.get("sim_trajectory")
    if not trajectory or len(trajectory) < 2:
        st.info("Run a simulation to see flow diagrams.")
        return
    if not HAS_PLOTLY:
        st.warning("Plotly required for flow visualization.")
        return

    flow_keys = sorted({k for f in trajectory for k in f.get("flows", {})})
    if not flow_keys:
        st.info("No flow data found in simulation frames.")
        return

    # Time-step selector
    max_t = len(trajectory) - 1
    t_idx = st.slider("Snapshot quarter", 0, max_t, max_t, key="sankey_t")
    frame = trajectory[t_idx]
    flows = frame.get("flows", {})

    # ── Waterfall bar chart ───────────────────────────────────────────────────
    sorted_flows = sorted(
        [(k, flows.get(k, 0)) for k in flow_keys],
        key=lambda x: abs(x[1]), reverse=True,
    )
    names = [dim_label(k) for k, _ in sorted_flows]
    values = [v for _, v in sorted_flows]
    colors = [theme.accent_success if v >= 0 else theme.accent_danger for v in values]

    fig_bar = go.Figure(go.Bar(
        x=names, y=values, marker_color=colors,
        text=[f"{v:.4f}" for v in values], textposition='outside',
    ))
    fig_bar.update_layout(**base_layout(theme, height=400,
        title=dict(text=f"Economic Flows at Quarter {frame.get('t', t_idx)}",
                   font=dict(color=theme.text_muted, size=13)),
        xaxis=dict(title=""), yaxis=dict(title="Flow Magnitude")))
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Sankey diagram ────────────────────────────────────────────────────────
    if len(flow_keys) >= 3:
        all_nodes = ["Inflows", "Outflows"] + [dim_label(k) for k in flow_keys]
        node_idx = {n: i for i, n in enumerate(all_nodes)}
        sources, targets, values_s, labels_s = [], [], [], []
        for k in flow_keys:
            v = flows.get(k, 0)
            label = dim_label(k)
            mag = max(abs(v), 1e-8)
            if v >= 0:
                sources.append(node_idx["Inflows"])
                targets.append(node_idx[label])
            else:
                sources.append(node_idx[label])
                targets.append(node_idx["Outflows"])
            values_s.append(mag)
            labels_s.append(f"{label}: {v:.4f}")

        fig_sankey = go.Figure(go.Sankey(
            node=dict(
                label=all_nodes,
                color=[PALETTE[i % len(PALETTE)] for i in range(len(all_nodes))],
                pad=15, thickness=20,
            ),
            link=dict(
                source=sources, target=targets, value=values_s,
                label=labels_s, color="rgba(100,100,100,0.3)",
            ),
        ))
        fig_sankey.update_layout(**base_layout(theme, height=450,
            title=dict(text="Flow Sankey Diagram",
                       font=dict(color=theme.text_muted, size=13))))
        st.plotly_chart(fig_sankey, use_container_width=True)

    # ── 3D Flow Surface (flows × time) ───────────────────────────────────────
    if st.checkbox("Show 3D Flow Dynamics Surface", key="sankey_3d"):
        z_matrix = []
        t_vals = []
        for fr in trajectory:
            t_vals.append(fr.get("t", 0))
            z_matrix.append([fr.get("flows", {}).get(k, 0) for k in flow_keys])

        fig_3d = go.Figure(go.Surface(
            x=list(range(len(flow_keys))),
            y=t_vals,
            z=z_matrix,
            colorscale='Viridis',
            colorbar=dict(title="Value"),
        ))
        fig_3d.update_layout(
            scene=dict(
                xaxis_title="Flow",
                yaxis_title="Quarter",
                zaxis_title="Magnitude",
                xaxis=dict(
                    tickvals=list(range(len(flow_keys))),
                    ticktext=[dim_label(k)[:12] for k in flow_keys],
                ),
                bgcolor="rgba(0,0,0,0)",
            ),
            **base_layout(theme, height=560,
                title=dict(text="3D Flow Dynamics Surface",
                           font=dict(color=theme.text_muted, size=13))),
        )
        st.plotly_chart(fig_3d, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 9: MONTE CARLO — Fan charts with parameter jitter
# ═════════════════════════════════════════════════════════════════════════════

def render_monte_carlo_tab(theme, SFCEconomy, SFCConfig,
                            calibrate_from_data, build_custom_scenario):
    trajectory = st.session_state.get("sim_trajectory")
    if not trajectory or len(trajectory) < 4:
        st.info("Run a base simulation first, then generate Monte Carlo uncertainty bands.")
        return
    if not HAS_PLOTLY:
        st.warning("Plotly required for fan charts.")
        return

    # Auto-discover outcome dimensions
    outcome_keys = sorted({k for f in trajectory for k in f.get("outcomes", {})
                           if not k.startswith("breach_")})
    if not outcome_keys:
        st.info("No outcome dimensions found.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        focus_dim = st.selectbox("Dimension", outcome_keys,
                                  format_func=dim_label, key="mc_dim")
    with c2:
        n_runs = st.slider("Number of runs", 10, 200, 50, 10, key="mc_runs")
    with c3:
        jitter_pct = st.slider("Parameter jitter (%)", 1, 25, 10, 1, key="mc_jitter")

    if st.button("Run Monte Carlo", key="mc_go", type="primary"):
        calib = st.session_state.get("sim_calibration")
        steps = st.session_state.get("sim_steps", 50)
        scenario = st.session_state.get("_sim_scenario_obj")

        if not calib:
            st.error("No calibration found. Run base simulation first.")
            return

        progress = st.progress(0, text="Running Monte Carlo simulations...")
        mc_trajectories = []

        for i in range(n_runs):
            try:
                # Jitter each numeric config field
                cfg_dict = {}
                for field_name in SFCConfig.__dataclass_fields__:
                    val = getattr(calib.config, field_name, None)
                    if isinstance(val, (int, float)) and field_name not in ("steps", "dt"):
                        j = 1.0 + np.random.uniform(-jitter_pct / 100, jitter_pct / 100)
                        cfg_dict[field_name] = val * j
                    elif val is not None:
                        cfg_dict[field_name] = val

                cfg = SFCConfig(**{k: v for k, v in cfg_dict.items()
                                   if k in SFCConfig.__dataclass_fields__})
                cfg.steps = steps

                if scenario and hasattr(scenario, 'build_shock_vectors'):
                    cfg.shock_vectors = scenario.build_shock_vectors(steps)

                econ = SFCEconomy(cfg)
                econ.initialize()
                traj = econ.run(steps)
                mc_trajectories.append(traj)
            except Exception:
                pass

            progress.progress((i + 1) / n_runs, text=f"Run {i + 1}/{n_runs}...")

        progress.empty()

        if len(mc_trajectories) < 5:
            st.warning(f"Only {len(mc_trajectories)} successful runs. Try reducing jitter.")
            return

        st.session_state["mc_trajectories"] = mc_trajectories
        st.session_state["mc_focus_dim"] = focus_dim

    # ── Render fan chart if MC results exist ──────────────────────────────────
    mc_trajs = st.session_state.get("mc_trajectories", [])
    if not mc_trajs:
        return

    focus = st.session_state.get("mc_focus_dim", focus_dim)

    max_len = max(len(t) for t in mc_trajs)
    t_vals = list(range(max_len))

    all_vals = []
    for t_idx in range(max_len):
        step_vals = []
        for traj in mc_trajs:
            if t_idx < len(traj):
                step_vals.append(traj[t_idx].get("outcomes", {}).get(focus, 0))
        all_vals.append(step_vals)

    percentiles = [10, 25, 50, 75, 90]
    bands = {p: [float(np.percentile(sv, p)) if sv else 0 for sv in all_vals]
             for p in percentiles}

    view_mode = st.radio("View", ["2D Fan Chart", "3D Uncertainty Surface"],
                         horizontal=True, key="mc_view")

    if view_mode == "3D Uncertainty Surface":
        z_matrix = [bands[p] for p in percentiles]
        fig = go.Figure(go.Surface(
            x=t_vals, y=percentiles, z=z_matrix,
            colorscale='Plasma', colorbar=dict(title="Value"), opacity=0.85,
        ))
        fig.update_layout(
            scene=dict(
                xaxis_title="Quarter", yaxis_title="Percentile",
                zaxis_title=dim_label(focus),
                bgcolor="rgba(0,0,0,0)",
            ),
            **base_layout(theme, height=560,
                title=dict(text=f"3D Uncertainty Surface — {dim_label(focus)}",
                           font=dict(color=theme.text_muted, size=13))),
        )
    else:
        fig = go.Figure()
        band_pairs = [(10, 90, 0.15), (25, 75, 0.25)]
        for lo, hi, opacity in band_pairs:
            fig.add_trace(go.Scatter(
                x=t_vals + t_vals[::-1],
                y=bands[hi] + bands[lo][::-1],
                fill='toself',
                fillcolor=f"rgba(0,170,255,{opacity})",
                line=dict(color='rgba(0,0,0,0)'),
                name=f"{lo}th–{hi}th percentile",
                showlegend=True,
            ))
        fig.add_trace(go.Scatter(
            x=t_vals, y=bands[50], mode='lines',
            name='Median', line=dict(color=theme.accent_primary, width=3),
        ))
        base_vals = [f.get("outcomes", {}).get(focus, 0) for f in trajectory]
        fig.add_trace(go.Scatter(
            x=list(range(len(base_vals))), y=base_vals, mode='lines',
            name='Base Run', line=dict(color=theme.accent_warning, width=2, dash='dash'),
        ))
        fig.update_layout(**base_layout(theme, height=480,
            title=dict(text=f"Monte Carlo Fan Chart — {dim_label(focus)} ({len(mc_trajs)} runs)",
                       font=dict(color=theme.text_muted, size=13)),
            xaxis=dict(title="Quarter"),
            yaxis=dict(title=dim_label(focus)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                        bgcolor='rgba(0,0,0,0)')))

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">
        {len(mc_trajs)} successful runs &nbsp;|&nbsp; Parameter jitter: &plusmn;{jitter_pct}%
        &nbsp;|&nbsp; Bands: 10th–90th and 25th–75th percentiles
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 10: STRESS TEST MATRIX — all scenarios × all outcomes
# ═════════════════════════════════════════════════════════════════════════════

def render_stress_matrix_tab(theme, scenario_library,
                              SFCEconomy, SFCConfig, calibrate_from_data):
    if not HAS_PLOTLY:
        st.warning("Plotly required for stress test visualization.")
        return

    steps = st.session_state.get("sim_steps", 50)
    policy_key = st.session_state.get("_sim_policy_key", "do_nothing")
    policy_overrides = st.session_state.get("_sim_policy_overrides", {})

    st.markdown(f"""
    <div style="font-size:0.8rem; color:{theme.text_muted}; margin-bottom:0.5rem;">
        Runs <b>every</b> scenario in the library with the currently selected policy
        and shows the impact across <b>all auto-discovered</b> outcome dimensions.
    </div>
    """, unsafe_allow_html=True)

    if st.button("Run All Scenarios Stress Test", key="stress_go", type="primary"):
        progress = st.progress(0, text="Running stress tests...")
        results = {}

        for i, scenario in enumerate(scenario_library):
            try:
                policy_mode = "custom" if policy_key != "do_nothing" else "off"
                config_overrides = {k: v for k, v in policy_overrides.items()
                                    if k in SFCConfig.__dataclass_fields__}
                calib = calibrate_from_data(
                    steps=steps, policy_mode=policy_mode, overrides=config_overrides,
                )
                cfg = calib.config
                cfg.shock_vectors = scenario.build_shock_vectors(steps)

                econ = SFCEconomy(cfg)
                econ.initialize()
                traj = econ.run(steps)

                if traj and len(traj) > 1:
                    final = traj[-1].get("outcomes", {})
                    initial = traj[0].get("outcomes", {})
                    deltas = {k: final.get(k, 0) - initial.get(k, 0) for k in final}
                    results[scenario.name] = {"final": final, "delta": deltas}
            except Exception:
                pass

            progress.progress((i + 1) / len(scenario_library))

        progress.empty()
        st.session_state["stress_results"] = results

    results = st.session_state.get("stress_results", {})
    if not results:
        st.info("Click the button above to generate the stress test matrix.")
        return

    # Auto-discover all outcome dimensions across all scenario runs
    all_dims = sorted({k for r in results.values() for k in r.get("delta", {})
                       if not k.startswith("breach_")})
    scenarios = list(results.keys())

    if not all_dims or not scenarios:
        st.warning("No results to display.")
        return

    z_matrix = []
    text_matrix = []
    for s in scenarios:
        row, text_row = [], []
        for d in all_dims:
            v = results[s].get("delta", {}).get(d, 0)
            row.append(v)
            text_row.append(f"{v:+.3f}")
        z_matrix.append(row)
        text_matrix.append(text_row)

    fig = go.Figure(go.Heatmap(
        z=z_matrix,
        x=[dim_label(d) for d in all_dims],
        y=scenarios,
        colorscale='RdYlGn',
        text=text_matrix,
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(title="Delta"),
    ))
    fig.update_layout(**base_layout(theme, height=max(400, len(scenarios) * 55),
        title=dict(text=f"Stress Matrix — {len(scenarios)} scenarios × {len(all_dims)} dimensions",
                   font=dict(color=theme.text_muted, size=13))))
    st.plotly_chart(fig, use_container_width=True)

    # 3D Stress Surface
    if st.checkbox("Show 3D Stress Surface", key="stress_3d"):
        fig_3d = go.Figure(go.Surface(
            x=list(range(len(all_dims))),
            y=list(range(len(scenarios))),
            z=z_matrix,
            colorscale='RdYlGn',
            colorbar=dict(title="Delta"),
        ))
        fig_3d.update_layout(
            scene=dict(
                xaxis_title="Dimension",
                yaxis_title="Scenario",
                zaxis_title="Impact (Delta)",
                xaxis=dict(tickvals=list(range(len(all_dims))),
                           ticktext=[dim_label(d)[:12] for d in all_dims]),
                yaxis=dict(tickvals=list(range(len(scenarios))),
                           ticktext=[s[:15] for s in scenarios]),
                bgcolor="rgba(0,0,0,0)",
            ),
            **base_layout(theme, height=560,
                title=dict(text="3D Stress Test Surface",
                           font=dict(color=theme.text_muted, size=13))),
        )
        st.plotly_chart(fig_3d, use_container_width=True)
