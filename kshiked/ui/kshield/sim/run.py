"""Simulation execution, scenario runner, and time-series rendering."""

from ._shared import st, pd, np, go, make_subplots, HAS_DATA_STACK, HAS_PLOTLY, PALETTE, base_layout, dim_label


def run_simulation(theme, SFCEconomy, SFCConfig, calibrate_from_data,
                    scenario_cfg, merge_shock_vectors_fn, merge_policy_instruments_fn):
    """Execute button + SFC simulation with merged multi-shock/multi-policy.

    Args:
        scenario_cfg: dict from _render_scenario_config with keys:
            selected_scenarios, custom_shocks, selected_policy_keys,
            custom_instruments, selected_dims, steps
        merge_shock_vectors_fn: merge_shock_vectors from scenario_templates
        merge_policy_instruments_fn: merge_policy_instruments from scenario_templates
    """
    selected_scenarios = scenario_cfg["selected_scenarios"]
    custom_shocks = scenario_cfg["custom_shocks"]
    selected_policy_keys = scenario_cfg["selected_policy_keys"]
    custom_instruments = scenario_cfg["custom_instruments"]
    selected_dims = scenario_cfg["selected_dims"]
    steps = scenario_cfg["steps"]

    col_r1, col_r2 = st.columns([3, 1])
    with col_r1:
        # Calibration confidence badge
        calib_cached = st.session_state.get("sim_calibration")
        if calib_cached:
            conf = calib_cached.overall_confidence
            cc = theme.accent_success if conf > 0.6 else theme.accent_warning if conf > 0.3 else theme.accent_danger
            n_data = sum(1 for p in calib_cached.params.values() if p.source == "data")
            st.markdown(f"""
            <div style="font-size:0.75rem; color:{theme.text_muted}; padding-top:0.5rem;">
                Calibration: <span style="color:{cc}; font-weight:600;">{conf:.0%}</span>
                &nbsp;|&nbsp; {n_data}/{len(calib_cached.params)} params from data
            </div>
            """, unsafe_allow_html=True)

        # Show active configuration summary
        summary_parts = []
        if selected_scenarios:
            scenario_names = [s.name for s in selected_scenarios]
            summary_parts.append(f"Shocks: {' + '.join(scenario_names)}")
        if custom_shocks:
            n_custom = len([cs for cs in custom_shocks if abs(cs.get('magnitude', 0)) > 1e-9])
            if n_custom:
                summary_parts.append(f"+{n_custom} custom shock(s)")
        if selected_policy_keys:
            summary_parts.append(f"Policy: {' + '.join(selected_policy_keys)}")
        if custom_instruments:
            summary_parts.append(f"+{len(custom_instruments)} custom instrument(s)")
        if not summary_parts:
            summary_parts.append("No shocks / No policy (baseline)")

        st.markdown(f"<div style='font-size:0.72rem; color:{theme.text_muted};'>"
                    f"{'  |  '.join(summary_parts)}</div>",
                    unsafe_allow_html=True)

    with col_r2:
        run_clicked = st.button("RUN SIMULATION", type="primary", use_container_width=True)

    if run_clicked:
        with st.spinner("Calibrating from data and running simulation..."):
            try:
                # Determine policy mode
                has_policy = bool(selected_policy_keys) or bool(custom_instruments)
                policy_mode = "custom" if has_policy else "off"

                # Merge policy instruments
                merged_instruments = merge_policy_instruments_fn(
                    selected_policy_keys, custom_instruments,
                )
                config_overrides = {
                    k: v for k, v in merged_instruments.items()
                    if k in SFCConfig.__dataclass_fields__
                }

                # Calibrate
                calib = calibrate_from_data(
                    steps=steps, policy_mode=policy_mode, overrides=config_overrides,
                )
                cfg = calib.config

                # Merge shock vectors (additive superposition)
                # Apply user-edited magnitude overrides from sliders
                preset_overrides = st.session_state.get("sim_preset_overrides", {})
                effective_scenarios = []
                for s in selected_scenarios:
                    if s.id in preset_overrides:
                        from copy import copy as _copy
                        s_copy = _copy(s)
                        s_copy.shocks = dict(preset_overrides[s.id])
                        effective_scenarios.append(s_copy)
                    else:
                        effective_scenarios.append(s)

                merged_shocks = merge_shock_vectors_fn(
                    effective_scenarios, custom_shocks, steps,
                )
                if merged_shocks:
                    cfg.shock_vectors = merged_shocks

                # Run
                econ = SFCEconomy(cfg)
                econ.initialize()
                trajectory = econ.run(steps)

                # Store results
                st.session_state["sim_trajectory"] = trajectory
                st.session_state["sim_selected_dims"] = selected_dims
                st.session_state["sim_calibration"] = calib
                st.session_state["sim_steps"] = steps
                st.session_state["sim_state"] = econ

                # Build descriptive label
                label_parts = []
                if selected_scenarios:
                    label_parts.append(" + ".join(s.name for s in selected_scenarios))
                n_custom_shocks = len([cs for cs in custom_shocks if abs(cs.get('magnitude', 0)) > 1e-9])
                if n_custom_shocks:
                    label_parts.append(f"{n_custom_shocks} custom")
                shock_label = " + ".join(label_parts) if label_parts else "Baseline"

                policy_parts = list(selected_policy_keys)
                if custom_instruments:
                    policy_parts.append(f"{len(custom_instruments)} custom")
                policy_label = " + ".join(policy_parts) if policy_parts else "No policy"

                run_label = f"{shock_label} | {policy_label}"

                # Store for comparison tab
                history = st.session_state.get("sim_compare_history", [])
                history.append({
                    "label": run_label,
                    "trajectory": trajectory,
                    "dims": selected_dims,
                })
                st.session_state["sim_compare_history"] = history[-5:]

            except Exception as e:
                st.error(f"Simulation error: {e}")
                import traceback
                st.code(traceback.format_exc())

    return run_clicked


# ═════════════════════════════════════════════════════════════════════════════
#  TAB 1: SCENARIO RUNNER — Impact cards + trajectory
# ═════════════════════════════════════════════════════════════════════════════

def render_scenario_runner_tab(theme, outcome_dimensions, default_dimensions, run_clicked):
    trajectory = st.session_state.get("sim_trajectory")
    if not trajectory or len(trajectory) < 2:
        if not run_clicked:
            st.markdown(f"""
            <div style="text-align:center; padding:3rem; color:{theme.text_muted};">
                <div style="font-size:2rem; margin-bottom:0.5rem; opacity:0.3;">&#9654;</div>
                <div style="font-size:0.9rem;">
                    Configure your scenario above and click <b>RUN SIMULATION</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
        return

    sel_dims = st.session_state.get("sim_selected_dims", list(default_dimensions))
    calib = st.session_state.get("sim_calibration")

    # Calibration info
    if calib:
        conf = calib.overall_confidence
        cc = theme.accent_success if conf > 0.6 else theme.accent_warning if conf > 0.3 else theme.accent_danger
        n_data = sum(1 for p in calib.params.values() if p.source == "data")
        st.markdown(f"""
        <div style="font-size:0.75rem; color:{theme.text_muted}; margin-bottom:0.8rem;">
            Engine: <span style="font-weight:600;">PARAMETRIC SFC</span>
            &nbsp;|&nbsp; Calibration: <span style="color:{cc}; font-weight:600;">{conf:.0%}</span>
            &nbsp;|&nbsp; {n_data}/{len(calib.params)} from data
            &nbsp;|&nbsp; {len(trajectory)} frames
        </div>
        """, unsafe_allow_html=True)

    # Impact summary table (compact)
    _start = trajectory[0].get("outcomes", {})
    _end = trajectory[-1].get("outcomes", {})
    _rows = []
    for dk in sel_dims:
        m = outcome_dimensions.get(dk, {"label": dk, "format": ".2f", "higher_is": "better"})
        sv, ev = _start.get(dk, 0), _end.get(dk, 0)
        d = ev - sv
        fmt = m.get("format", ".2f")
        _rows.append({"Indicator": m.get("label", dk), "Start": f"{sv:{fmt}}",
                       "End": f"{ev:{fmt}}", "Delta": f"{d:+{fmt}}"})
    if _rows:
        import pandas as _pd
        st.dataframe(_pd.DataFrame(_rows), use_container_width=True, hide_index=True)
    st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

    # Time-series trajectory
    if HAS_PLOTLY:
        render_time_series(trajectory, sel_dims, outcome_dimensions, theme)


def render_time_series(trajectory, sel_dims, outcome_dimensions, theme):
    fig = make_subplots(rows=1, cols=1)
    t_vals = [f["t"] for f in trajectory]

    for idx, dim_key in enumerate(sel_dims):
        meta = outcome_dimensions.get(dim_key, {"label": dim_key})
        vals = [f.get("outcomes", {}).get(dim_key, 0) for f in trajectory]

        fmt = meta.get("format", ".2f")
        suffix = " (%)" if "%" in fmt else ""
        if "%" in fmt:
            vals = [v * 100 for v in vals]

        fig.add_trace(go.Scatter(
            x=t_vals, y=vals, mode='lines',
            name=f"{meta.get('label', dim_key)}{suffix}",
            line=dict(color=PALETTE[idx % len(PALETTE)], width=2.5),
            hovertemplate=f"{meta.get('label', dim_key)}: %{{y:.2f}}{suffix}<extra></extra>",
        ))

    # Shock onset marker
    scenario = st.session_state.get("_sim_scenario_obj")
    onset = scenario.shock_onset if scenario else 5
    fig.add_vline(x=onset, line_dash="dash", line_color=theme.accent_danger,
                  annotation_text="Shock", annotation_font_color=theme.accent_danger)

    fig.update_layout(**base_layout(theme, height=420,
        title=dict(text="Trajectory Over Time", font=dict(color=theme.text_muted, size=13)),
        xaxis=dict(title="Quarter"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor='rgba(0,0,0,0)'),
    ))
    st.plotly_chart(fig, use_container_width=True)
