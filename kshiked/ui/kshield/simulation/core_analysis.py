"""Core analysis tabs: Sensitivity, State Cube, Compare, Diagnostics."""

from ._shared import (st, pd, np, go, make_subplots, HAS_DATA_STACK, HAS_PLOTLY, PALETTE, base_layout, discover_dimensions, flat_dim_options, dim_label, extract_dim)


def _guide(theme, method_html: str, interp_html: str, rec_html: str) -> None:
  """Render a collapsible 3-section analysis guide."""
  with st.expander("Analysis Guide & Interpretation", expanded=False):
    st.markdown(
      f"<div style='font-size:0.82rem; line-height:1.65;'>"
      f"<div style='color:{theme.accent_primary}; font-weight:700; font-size:0.72rem; "
      f"letter-spacing:0.08em; margin-bottom:0.35rem;'>WHAT IS THIS ANALYSIS?</div>"
      f"<div style='color:{theme.text_muted}; margin-bottom:0.9rem;'>{method_html}</div>"
      f"<div style='color:{theme.accent_warning}; font-weight:700; font-size:0.72rem; "
      f"letter-spacing:0.08em; margin-bottom:0.35rem;'>INTERPRETATION</div>"
      f"<div style='color:{theme.text_muted}; margin-bottom:0.9rem;'>{interp_html}</div>"
      f"<div style='color:{theme.accent_success}; font-weight:700; font-size:0.72rem; "
      f"letter-spacing:0.08em; margin-bottom:0.35rem;'>RECOMMENDATION</div>"
      f"<div style='color:{theme.text_muted};'>{rec_html}</div>"
      f"</div>",
      unsafe_allow_html=True,
    )


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

    # ── Sensitivity interpretation ───────────────────────────────────────────
    try:
      _flat_impacts = [(policy_keys[r], outcome_keys[c], impacts[r][c])
                       for r in range(len(policy_keys)) for c in range(len(outcome_keys))]
      _top_pos = max(_flat_impacts, key=lambda x: x[2])
      _top_neg = min(_flat_impacts, key=lambda x: x[2])
      _strong_pos = [(p, o, v) for p, o, v in _flat_impacts if v > 0.6]
      _strong_neg = [(p, o, v) for p, o, v in _flat_impacts if v < -0.6]
      _interp_sens = (
        f"Strongest positive lever: <b>{dim_label(_top_pos[0])}</b> → <b>{dim_label(_top_pos[1])}</b> "
        f"(ρ = {_top_pos[2]:+.2f}). "
        f"Strongest negative lever: <b>{dim_label(_top_neg[0])}</b> → <b>{dim_label(_top_neg[1])}</b> "
        f"(ρ = {_top_neg[2]:+.2f}). "
        + (f"{len(_strong_pos)} strong positive association(s) and {len(_strong_neg)} strong negative association(s) detected (|ρ| > 0.6)."
           if _strong_pos or _strong_neg else
           "No strong associations (|ρ| > 0.6) detected — policy instruments have diffuse effects in this scenario.")
      )
      if _top_pos[2] > 0.5:
        _rec_sens = (
          f"To improve <b>{dim_label(_top_pos[1])}</b>, prioritise expanding "
          f"<b>{dim_label(_top_pos[0])}</b> — it has the strongest positive association in this scenario. "
          + (f"Conversely, be cautious with <b>{dim_label(_top_neg[0])}</b> — increasing it "
             f"is associated with a deterioration in <b>{dim_label(_top_neg[1])}</b> (ρ = {_top_neg[2]:+.2f})."
             if _top_neg[2] < -0.4 else "")
        )
      else:
        _rec_sens = (
          "No instrument has a dominant positive effect on any single outcome. "
          "Policy must rely on combinations of instruments rather than a single lever. "
          "Consider running different scenarios to find conditions under which stronger sensitivities emerge."
        )
    except Exception:
      _interp_sens = "Look for the darkest green cells (strongest positive levers) and darkest red cells (strongest negative associations)."
      _rec_sens = "Pull the green levers; avoid or reverse the red ones relative to your target outcome column."

    _guide(
      theme,
      method_html=(
        "The <b>Policy-Outcome Sensitivity Matrix</b> shows the <b>Pearson correlation</b> between "
        "each policy instrument and each outcome dimension over the full simulation trajectory.<br><br>"
        "Each cell value ρ ranges from −1 to +1:<br>"
        "<ul style='margin:0.3rem 0; padding-left:1.2rem;'>"
        "<li><b>+1.0</b> — instrument and outcome move in perfect lockstep (turning this lever up always improves the outcome)</li>"
        "<li>&nbsp;0.0 — no relationship; this instrument doesn't affect this outcome</li>"
        "<li><b>−1.0</b> — perfect negative correlation (turning this lever up worsens the outcome)</li>"
        "</ul>"
        "Note: correlation ≠ causation. A high correlation can arise because both variables respond to the same underlying shock. "
        "Use the Causal Estimands tab for causal identification."
      ),
      interp_html=_interp_sens,
      rec_html=_rec_sens,
    )


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
  if x_vals and y_vals and z_vals and len(trajectory) > 1:
    _dx = x_vals[-1] - x_vals[0]
    _dy = y_vals[-1] - y_vals[0]
    _dz = z_vals[-1] - z_vals[0]
    st.markdown(f"""
    <div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">
      <b>How to read:</b> Each dot is the economy's state in
      {x_label}–{y_label}–{z_label} space at a given quarter.
      The path from <span style="color:{theme.accent_success};">&#x25C6; Start</span>
      to <span style="color:{theme.accent_danger};">&#x25C6; End</span>
      traces how the system evolved. Colour gradient = time (early = dark, late = bright).
      <br><b>Net movement over simulation:</b>
      {x_label} {_dx:+.3f} &nbsp;|&nbsp;
      {y_label} {_dy:+.3f} &nbsp;|&nbsp;
      {z_label} {_dz:+.3f}.
      Drag to rotate; scroll to zoom.
    </div>
    """, unsafe_allow_html=True)

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

  # ── State Cube interpretation ─────────────────────────────────────────────
  try:
    _n = len(x_vals) - 1
    _seg3 = [(x_vals[i+1]-x_vals[i])**2 + (y_vals[i+1]-y_vals[i])**2 + (z_vals[i+1]-z_vals[i])**2
             for i in range(_n)]
    _path_3d = sum(s**0.5 for s in _seg3)
    _disp_3d = ((x_vals[-1]-x_vals[0])**2 + (y_vals[-1]-y_vals[0])**2 + (z_vals[-1]-z_vals[0])**2)**0.5
    _straight_3d = float(_disp_3d / max(_path_3d, 1e-9))
    _vx = [x_vals[i+1]-x_vals[i] for i in range(_n)]
    _vy = [y_vals[i+1]-y_vals[i] for i in range(_n)]
    _rev_x = sum(1 for i in range(1, len(_vx)) if _vx[i]*_vx[i-1] < 0)
    _rev_y = sum(1 for i in range(1, len(_vy)) if _vy[i]*_vy[i-1] < 0)
    if _straight_3d > 0.8:
      _topo = "monotonic convergence — the economy moves cleanly toward a new equilibrium in all 3 dimensions"
    elif _straight_3d > 0.45:
      _topo = "curved adjustment — oscillatory dynamics present in at least one dimension"
    else:
      _topo = "highly looping trajectory — strong cyclical or oscillatory dynamics; the economy revisits similar states"
    _interp_cube = (
      f"3D path length: {_path_3d:.3f} units. Straightness index: <b>{_straight_3d:.2f}</b>. "
      f"Topology: <b>{_topo}</b>. "
      f"Direction reversals: {_rev_x} on {x_label}, {_rev_y} on {y_label}. "
      f"The colour axis ({c_title}) encodes a 4th dimension — watch for colour banding or clustering, "
      f"which indicates the 4th variable is correlated with the 3D trajectory shape."
    )
    if _straight_3d > 0.7:
      _rec_cube = (
        "Clean convergence detected. The system is well-behaved in all three selected dimensions. "
        "Rotate the cube to check whether the trajectory is flat in any dimension (indicating that "
        "variable is not actively adjusting — it may not be the right choice for the Z or X axis)."
      )
    else:
      _rec_cube = (
        f"Oscillatory dynamics detected ({max(_rev_x, _rev_y)} reversal(s)). "
        "Rotate the 3D cube to identify the plane where oscillation is most pronounced — "
        "the two variables whose axis contains the most curvature are the primary boom-bust pair. "
        "Use the Phase Explorer tab to focus on just those two dimensions for a cleaner 2D analysis."
      )
  except Exception:
    _interp_cube = "Examine the trajectory shape: tight cluster = near equilibrium; long arc = sustained trend; loop = cyclical dynamics."
    _rec_cube = "Use the colour selector to encode a 4th variable — watch for colour banding along the trajectory, which signals correlation with the path direction."

  _guide(
    theme,
    method_html=(
      "The <b>3D State Cube</b> is a multi-dimensional projection of the economy's state space. "
      "Three variables are mapped to X, Y, Z coordinates; colour adds a <b>4th dimension</b> "
      "— making this effectively a <b>4D visualisation</b> of a system with far more than 4 variables.<br><br>"
      "Why a cube instead of time-series lines? Because the economy is a <b>coupled system</b> — "
      "all variables interact simultaneously. The cube shows these couplings directly: "
      "when the economy moves through X-Y-Z space, it reveals whether the variables "
      "move together (correlated adjustment) or independently.<br><br>"
      "The <b>trajectory path</b> on the manifold (the 3D surface the economy is constrained to) "
      "shows the dynamic structure:<br>"
      "<ul style='margin:0.3rem 0; padding-left:1.2rem;'>"
      "<li><b>Tight cluster</b> → near-equilibrium, low volatility</li>"
      "<li><b>Long straight arc</b> → sustained directional trend (structural shift)</li>"
      "<li><b>Looping</b> → cyclical dynamics; the economy repeatedly revisits similar states</li>"
      "<li><b>Chaotic scatter</b> → high sensitivity to initial conditions</li>"
      "</ul>"
      "The sector balance table confirms <b>SFC accounting consistency</b>: all sector balances must "
      "sum near zero (a household saving = a firm borrowing = a bank lending)."
    ),
    interp_html=_interp_cube,
    rec_html=_rec_cube,
  )


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
  if history and len(history) >= 2:
    _end_vals = {h["label"]: h["trajectory"][-1].get("outcomes", {}).get(focus_dim, 0) * mult
                 for h in history if h.get("trajectory")}
    if _end_vals:
      _sorted_runs = sorted(_end_vals.items(), key=lambda x: x[1], reverse=True)
      _leader_lbl, _leader_val = _sorted_runs[0]
      _trailer_lbl, _trailer_val = _sorted_runs[-1]
      _sfx = "%" if is_pct else ""
      st.markdown(f"""
      <div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">
        <b>Reading:</b> Each line is a saved scenario run overlaid for comparison.
        At the final quarter: <b>{_leader_lbl}</b> leads at {_leader_val:.2f}{_sfx};
        <b>{_trailer_lbl}</b> trails at {_trailer_val:.2f}{_sfx}
        (gap = {abs(_leader_val - _trailer_val):.2f}{_sfx}).
        {f"Shaded band = 25th–75th percentile uncertainty range from 12 jittered runs." if show_bands else ""}
      </div>
      """, unsafe_allow_html=True)

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
