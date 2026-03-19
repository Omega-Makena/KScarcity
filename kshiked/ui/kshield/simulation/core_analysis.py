"""Core analysis tabs: Sensitivity, State Cube, Compare, Diagnostics."""

from ._shared import (st, pd, np, go, make_subplots, HAS_DATA_STACK, HAS_PLOTLY, PALETTE, base_layout, discover_dimensions, flat_dim_options, dim_label, extract_dim)


def render_sensitivity_tab(theme, outcome_dimensions):
  trajectory = st.session_state.get("sim_trajectory")
  if not trajectory or len(trajectory) < 6:
    st.info("Run a simulation with at least 6 quarters to see sensitivity analysis.")
    return

  if not HAS_PLOTLY:
    st.warning("Plotly required for heatmap visualization.")
    return

  policy_keys = sorted({k for f in trajectory for k in f.get("policy_vector", {})})
  outcome_keys = sorted({k for f in trajectory for k in f.get("outcomes", {})})
  if not policy_keys or not outcome_keys:
    st.info("Policy/outcome vectors missing from simulation frames.")
    return

  impacts = []
  for pol in policy_keys:
    p = np.array([float(f.get("policy_vector", {}).get(pol, 0)) for f in trajectory])
    row = []
    for out in outcome_keys:
      y = np.array([float(f.get("outcomes", {}).get(out, 0)) for f in trajectory])
      if np.allclose(p.std(), 0) or np.allclose(y.std(), 0):
        row.append(0.0)
      else:
        row.append(float(np.corrcoef(p, y)[0, 1]))
    impacts.append(row)

  fig = go.Figure(go.Heatmap(
    z=impacts,
    x=[o.replace("_", " ").title() for o in outcome_keys],
    y=[p.replace("_", " ").title() for p in policy_keys],
    colorscale=[[0, theme.accent_danger], [0.5, '#ffffff'], [1, theme.accent_success]],
    zmin=-1, zmax=1,
    text=[[f"{v:+.2f}" for v in row] for row in impacts],
    texttemplate="%{text}",
    showscale=True,
  ))
  fig.update_layout(**base_layout(theme, height=400,
    title=dict(text="Policy-Outcome Sensitivity Matrix",
          font=dict(color=theme.text_muted, size=13)),
  ))
  st.plotly_chart(fig, use_container_width=True)

  # Plain-English interpretation
  if impacts:
    st.markdown(f"""
    <div style="font-size:0.8rem; color:{theme.text_muted}; padding:0.5rem 0;">
      <b>Reading:</b> Each cell shows the correlation between a policy instrument
      and an outcome dimension over the simulation trajectory.
      Values near <span style="color:{theme.accent_success}">+1.0</span> mean the
      instrument is strongly positively associated;
      <span style="color:{theme.accent_danger}">-1.0</span> means strongly negative.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: STATE CUBE — Dynamic N-D sector view
# ═════════════════════════════════════════════════════════════════════════════

def render_state_cube_tab(theme, outcome_dimensions):
  trajectory = st.session_state.get("sim_trajectory")
  if not trajectory or len(trajectory) < 4:
    st.info("Run a simulation to see the state cube.")
    return

  if not HAS_PLOTLY:
    st.warning("Plotly required for 3D visualization.")
    return

  dims = discover_dimensions(trajectory)
  all_opts = flat_dim_options(dims)
  if len(all_opts) < 3:
    st.info("Need at least 3 discoverable dimensions for a 3D cube.")
    return

  c1, c2, c3, c4 = st.columns(4)
  with c1:
    x_dim = st.selectbox("X axis", all_opts, index=0, key="cube_x",
               format_func=lambda s: dim_label(s.split("::")[-1]))
  with c2:
    y_dim = st.selectbox("Y axis", all_opts,
               index=min(1, len(all_opts) - 1), key="cube_y",
               format_func=lambda s: dim_label(s.split("::")[-1]))
  with c3:
    z_dim = st.selectbox("Z axis", all_opts,
               index=min(2, len(all_opts) - 1), key="cube_z",
               format_func=lambda s: dim_label(s.split("::")[-1]))
  with c4:
    color_opts = ["time"] + all_opts
    color_dim = st.selectbox("Color", color_opts, index=0, key="cube_color",
                 format_func=lambda s: "Quarter" if s == "time"
                 else dim_label(s.split("::")[-1]))

  t_vals = [f.get("t", 0) for f in trajectory]
  x_vals = [extract_dim(x_dim, f) for f in trajectory]
  y_vals = [extract_dim(y_dim, f) for f in trajectory]
  z_vals = [extract_dim(z_dim, f) for f in trajectory]
  c_vals = t_vals if color_dim == "time" else [extract_dim(color_dim, f) for f in trajectory]
  c_title = "Quarter" if color_dim == "time" else dim_label(color_dim.split("::")[-1])

  x_label = dim_label(x_dim.split("::")[-1])
  y_label = dim_label(y_dim.split("::")[-1])
  z_label = dim_label(z_dim.split("::")[-1])

  fig = go.Figure()
  fig.add_trace(go.Scatter3d(
    x=x_vals, y=y_vals, z=z_vals,
    mode='lines+markers',
    marker=dict(size=4, color=c_vals, colorscale='Plasma',
          colorbar=dict(title=c_title, len=0.6), opacity=0.9),
    line=dict(color=theme.accent_primary, width=3),
    hovertemplate=(
      f"{x_label}: %{{x:.3f}}<br>"
      f"{y_label}: %{{y:.3f}}<br>"
      f"{z_label}: %{{z:.3f}}<extra></extra>"
    ),
    name="State Trajectory",
  ))
  # Start / End markers
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
    scene=dict(
      xaxis_title=x_label, yaxis_title=y_label, zaxis_title=z_label,
      bgcolor="rgba(0,0,0,0)",
    ),
    **base_layout(theme, height=550,
      title=dict(text="3D State Cube — Dynamic Sector View",
            font=dict(color=theme.text_muted, size=13))),
  )
  st.plotly_chart(fig, use_container_width=True)

  # Dynamic summary table for ALL discovered sector_balances
  sector_keys = dims.get("sector_balances", [])
  if sector_keys and len(trajectory) > 1:
    final, initial = trajectory[-1], trajectory[0]
    bal_data = []
    for s in sector_keys:
      init_v = initial.get("sector_balances", {}).get(s, 0)
      fin_v = final.get("sector_balances", {}).get(s, 0)
      bal_data.append({
        "Sector": dim_label(s),
        "Initial": f"{init_v:.4f}",
        "Final": f"{fin_v:.4f}",
        "Change": f"{fin_v - init_v:+.4f}",
      })
    st.dataframe(pd.DataFrame(bal_data), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: COMPARE — overlay multiple runs
# ═════════════════════════════════════════════════════════════════════════════

def render_compare_tab(theme, outcome_dimensions, default_dimensions):
  history = st.session_state.get("sim_compare_history", [])
  if len(history) < 2:
    st.info("Run at least 2 different scenarios to compare trajectories. "
        "Each run is automatically saved for comparison (last 5 kept).")
    if history:
      st.caption(f"Currently stored: {len(history)}/5 runs — "
            f"{', '.join(h['label'] for h in history)}")
    return

  if not HAS_PLOTLY:
    st.warning("Plotly required for comparison charts.")
    return

  # Let user pick which dimension to compare
  all_dims = set()
  for h in history:
    all_dims.update(h.get("dims", []))
  all_dims = sorted(all_dims)
  if not all_dims:
    all_dims = list(default_dimensions)

  ctrl_c1, ctrl_c2 = st.columns([3, 1])
  with ctrl_c1:
    focus_dim = st.selectbox("Compare dimension", all_dims,
                 format_func=lambda d: outcome_dimensions.get(d, {}).get("label", d),
                 key="sim_compare_dim")
  with ctrl_c2:
    show_bands = st.checkbox("Uncertainty bands", value=False, key="sim_compare_bands",
                             help="Overlay 25th–75th percentile bands from 12 jittered runs")

  meta = outcome_dimensions.get(focus_dim, {"label": focus_dim, "format": ".2f"})
  fmt = meta.get("format", ".2f")
  is_pct = "%" in fmt
  mult = 100 if is_pct else 1

  # Clear stale band caches for other dimensions when focus changes
  prev_dim = st.session_state.get("_compare_bands_dim")
  if prev_dim and prev_dim != focus_dim:
    st.session_state.pop(f"_compare_bands_{prev_dim}", None)
  st.session_state["_compare_bands_dim"] = focus_dim

  # Compute MC bands when requested (cached per dimension in session state)
  band_cache_key = f"_compare_bands_{focus_dim}"
  if show_bands:
    if band_cache_key not in st.session_state:
      calib = st.session_state.get("sim_calibration")
      if calib:
        try:
          from scarcity.simulation.sfc import SFCEconomy, SFCConfig
          steps = st.session_state.get("sim_steps", 50)
          scenario = st.session_state.get("_sim_scenario_obj")
          with st.spinner("Computing uncertainty bands…"):
            bundles = []
            base_cfg = calib.config
            for _ in range(12):
              cfg_dict = {}
              for fn in SFCConfig.__dataclass_fields__:
                val = getattr(base_cfg, fn, None)
                if isinstance(val, (int, float)) and fn not in ("steps", "dt"):
                  j = 1.0 + np.random.uniform(-0.08, 0.08)
                  cfg_dict[fn] = val * j
                elif val is not None:
                  cfg_dict[fn] = val
              try:
                cfg = SFCConfig(**{k: v for k, v in cfg_dict.items()
                                   if k in SFCConfig.__dataclass_fields__})
                cfg.steps = steps
                if scenario and hasattr(scenario, 'build_shock_vectors'):
                  cfg.shock_vectors = scenario.build_shock_vectors(steps)
                econ = SFCEconomy(cfg)
                econ.initialize()
                bundles.append(econ.run(steps))
              except Exception:
                pass
          if len(bundles) >= 5:
            T = max(len(b) for b in bundles)
            arr = np.array([
              [b[t].get("outcomes", {}).get(focus_dim, 0) * mult if t < len(b) else 0
               for t in range(T)]
              for b in bundles
            ])
            st.session_state[band_cache_key] = {
              "p25": np.percentile(arr, 25, axis=0).tolist(),
              "p75": np.percentile(arr, 75, axis=0).tolist(),
            }
        except ImportError:
          pass

  fig = go.Figure()

  # Draw bands first (behind lines)
  if show_bands and band_cache_key in st.session_state:
    bands = st.session_state[band_cache_key]
    p25, p75 = bands["p25"], bands["p75"]
    t_b = list(range(len(p25)))
    fig.add_trace(go.Scatter(
      x=t_b + t_b[::-1], y=p75 + p25[::-1],
      fill='toself', fillcolor="rgba(0,170,255,0.13)",
      line=dict(color='rgba(0,0,0,0)'),
      name="25th–75th pct", showlegend=True, hoverinfo='skip',
    ))

  for i, h in enumerate(history):
    traj = h["trajectory"]
    t_vals = [f["t"] for f in traj]
    vals = [f.get("outcomes", {}).get(focus_dim, 0) * mult for f in traj]
    fig.add_trace(go.Scatter(
      x=t_vals, y=vals, mode='lines',
      name=h["label"],
      line=dict(color=PALETTE[i % len(PALETTE)], width=2.5),
    ))

  suffix = " (%)" if is_pct else ""
  fig.update_layout(**base_layout(theme, height=420,
    title=dict(text=f"Comparison: {meta.get('label', focus_dim)}{suffix}",
          font=dict(color=theme.text_muted, size=13)),
    xaxis=dict(title="Quarter"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
          bgcolor='rgba(0,0,0,0)'),
  ))
  st.plotly_chart(fig, use_container_width=True)

  # Summary table
  rows = []
  for h in history:
    traj = h["trajectory"]
    if len(traj) >= 2:
      start_v = traj[0].get("outcomes", {}).get(focus_dim, 0)
      end_v = traj[-1].get("outcomes", {}).get(focus_dim, 0)
      mult = 100 if is_pct else 1
      rows.append({
        "Run": h["label"],
        "Start": f"{start_v * mult:{fmt.replace('%', 'f')}}",
        "End": f"{end_v * mult:{fmt.replace('%', 'f')}}",
        "Delta": f"{(end_v - start_v) * mult:+.2f}",
      })
  if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

  # Clear button
  if st.button("Clear comparison history", key="sim_clear_compare"):
    st.session_state["sim_compare_history"] = []
    st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5: DIAGNOSTICS — calibration & engine details (fully dynamic)
# ═════════════════════════════════════════════════════════════════════════════

def render_diagnostics_tab(theme):
  calib = st.session_state.get("sim_calibration")
  trajectory = st.session_state.get("sim_trajectory")

  if not calib:
    st.info("Run a simulation to see engine diagnostics.")
    return

  # Calibration parameter table
  st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:600; "
        f"font-size:0.85rem; margin-bottom:0.5rem;'>CALIBRATION PARAMETERS</div>",
        unsafe_allow_html=True)

  param_rows = []
  for name, p in sorted(calib.params.items()):
    param_rows.append({
      "Parameter": name,
      "Value": f"{p.value:.4f}" if isinstance(p.value, float) else str(p.value),
      "Source": p.source,
      "Confidence": f"{p.confidence:.0%}",
      "Note": getattr(p, 'note', '') or "",
    })
  st.dataframe(pd.DataFrame(param_rows), use_container_width=True, hide_index=True)

  # Dynamic SFC state summary — auto-discover ALL outcome dimensions
  if trajectory and len(trajectory) > 1:
    st.markdown(f"<div style='color:{theme.accent_warning}; font-weight:600; "
          f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
          f"SFC ENGINE SUMMARY</div>", unsafe_allow_html=True)

    final = trajectory[-1]
    outcomes = final.get("outcomes", {})

    st.text(f"Total frames: {len(trajectory)}")

    # Dynamically render metrics for ALL discovered outcome dimensions
    outcome_keys = sorted(outcomes.keys())
    # Filter out internal breach flags
    outcome_keys = [k for k in outcome_keys if not k.startswith("breach_")]

    # Render as a compact table
    _metric_rows = []
    for key in outcome_keys:
      val = outcomes[key]
      if isinstance(val, float) and -1.0 <= val <= 2.0 and key not in ("financial_stability",):
        _metric_rows.append({"Indicator": dim_label(key), "Value": f"{val:.2%}"})
      else:
        _metric_rows.append({"Indicator": dim_label(key), "Value": f"{val:.4f}"})
    if _metric_rows:
      st.dataframe(pd.DataFrame(_metric_rows), use_container_width=True, hide_index=True)

    # Sector balance check (SFC consistency) — dynamic sector discovery
    dims = discover_dimensions(trajectory)
    sector_keys = dims.get("sector_balances", [])
    if sector_keys:
      all_balances = [
        sum(f.get("sector_balances", {}).get(s, 0) for s in sector_keys)
        for f in trajectory
      ]
      max_imbalance = max(abs(b) for b in all_balances) if all_balances else 0
      bal_ok = max_imbalance < 0.01
      bc = theme.accent_success if bal_ok else theme.accent_danger
      st.markdown(f"""
      <div style="font-size:0.8rem; color:{theme.text_muted}; margin-top:0.5rem;">
        SFC Balance Check: max sector imbalance =
        <span style="color:{bc}; font-weight:600;">{max_imbalance:.6f}</span>
        {'(PASS)' if bal_ok else '(WARNING: large imbalance)'}
      </div>
      """, unsafe_allow_html=True)

    # Channel dynamics summary (also dynamic)
    channel_keys = dims.get("channels", [])
    if channel_keys:
      st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:600; "
            f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
            f"CHANNEL DYNAMICS</div>", unsafe_allow_html=True)
      _ch_rows = []
      final_channels = final.get("channels", {})
      for ck in channel_keys:
        cv = final_channels.get(ck, 0)
        _ch_rows.append({"Channel": dim_label(ck), "Value": f"{cv:.4f}"})
      if _ch_rows:
        st.dataframe(pd.DataFrame(_ch_rows), use_container_width=True, hide_index=True)
