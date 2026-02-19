"""Parameter Surface tab."""
from ._shared import (st, pd, np, go, make_subplots, HAS_DATA_STACK, HAS_PLOTLY, PALETTE, base_layout, dim_label)


def render_parameter_surface_tab(theme, SFCEconomy, SFCConfig, calibrate_from_data):
    trajectory = st.session_state.get("sim_trajectory")
    if not trajectory or len(trajectory) < 4:
        st.info("Run a base simulation first to enable parameter sweeps.")
        return
    if not HAS_PLOTLY:
        st.warning("Plotly required for 3D surfaces.")
        return

    calib = st.session_state.get("sim_calibration")
    if not calib:
        st.info("No calibration available — run the simulation first.")
        return

    # Auto-discover tunable parameters from SFCConfig
    tunable_params = sorted([
        fn for fn in SFCConfig.__dataclass_fields__
        if isinstance(getattr(calib.config, fn, None), float)
        and fn not in ("steps", "dt")
    ])
    outcome_keys = sorted({k for f in trajectory for k in f.get("outcomes", {})
                           if not k.startswith("breach_")})
    if not tunable_params or not outcome_keys:
        st.info("Need tunable parameters and outcome dimensions.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        param = st.selectbox("Sweep parameter", tunable_params,
                              format_func=dim_label, key="surf_param")
    with c2:
        outcome = st.selectbox("Outcome dimension", outcome_keys,
                                format_func=dim_label, key="surf_outcome")
    with c3:
        n_sweep = st.slider("Sweep points", 5, 30, 15, key="surf_n")

    base_val = getattr(calib.config, param, 0.1)
    sweep_range = st.slider(
        f"Range around base ({base_val:.4f})",
        0.1, 2.0, 0.5, 0.1, key="surf_range",
        help="Multiplier: 0.5 → sweep from 50% to 150% of base value",
    )

    if st.button("Generate Response Surface", key="surf_go", type="primary"):
        lo = base_val * (1 - sweep_range)
        hi = base_val * (1 + sweep_range)
        param_vals = np.linspace(max(lo, 1e-6), hi, n_sweep).tolist()

        scenario = st.session_state.get("_sim_scenario_obj")
        steps = st.session_state.get("sim_steps", 50)

        progress = st.progress(0, text="Sweeping parameter space...")
        z_surface = []
        valid_params = []

        for i, pv in enumerate(param_vals):
            try:
                cfg_dict = {}
                for fn in SFCConfig.__dataclass_fields__:
                    cfg_dict[fn] = getattr(calib.config, fn)
                cfg_dict[param] = pv
                cfg_dict["steps"] = steps

                cfg = SFCConfig(**{k: v for k, v in cfg_dict.items()
                                   if k in SFCConfig.__dataclass_fields__})
                if scenario and hasattr(scenario, 'build_shock_vectors'):
                    cfg.shock_vectors = scenario.build_shock_vectors(steps)

                econ = SFCEconomy(cfg)
                econ.initialize()
                traj = econ.run(steps)

                row = [f.get("outcomes", {}).get(outcome, 0) for f in traj]
                z_surface.append(row)
                valid_params.append(pv)
            except Exception:
                pass

            progress.progress((i + 1) / n_sweep)

        progress.empty()
        st.session_state["surf_z"] = z_surface
        st.session_state["surf_params"] = valid_params
        st.session_state["surf_outcome_name"] = outcome
        st.session_state["surf_param_name"] = param

    # ── Render the surface if data exists ─────────────────────────────────────
    z_surface = st.session_state.get("surf_z")
    valid_params = st.session_state.get("surf_params")

    if not z_surface or not valid_params:
        return

    surf_outcome = st.session_state.get("surf_outcome_name", outcome)
    surf_param = st.session_state.get("surf_param_name", param)

    # Normalise row lengths
    max_len = max(len(r) for r in z_surface)
    for i, r in enumerate(z_surface):
        if len(r) < max_len:
            z_surface[i] = r + [r[-1]] * (max_len - len(r))

    fig = go.Figure(go.Surface(
        x=list(range(max_len)),
        y=valid_params,
        z=z_surface,
        colorscale='Viridis',
        colorbar=dict(title=dim_label(surf_outcome)[:15]),
        opacity=0.9,
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title="Quarter",
            yaxis_title=dim_label(surf_param),
            zaxis_title=dim_label(surf_outcome),
            bgcolor="rgba(0,0,0,0)",
        ),
        **base_layout(theme, height=600,
            title=dict(
                text=f"Parameter Response Surface: {dim_label(surf_param)} → {dim_label(surf_outcome)}",
                font=dict(color=theme.text_muted, size=13))),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">
        X = time (quarters), Y = <b>{dim_label(surf_param)}</b> swept from
        {min(valid_params):.4f} to {max(valid_params):.4f},
        Z = <b>{dim_label(surf_outcome)}</b>.
        {len(valid_params)} successful parameter values.
    </div>
    """, unsafe_allow_html=True)
