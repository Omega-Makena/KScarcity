"""Advanced analysis tabs: Phase Explorer, IRF, Flow Sankey, Monte Carlo, Stress Matrix, Parameter Surface."""

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
    <b>Reading:</b> The path traces how the economy evolves through the chosen
    state space over time. Colour encodes the quarter — early periods are dark,
    later periods bright. <span style="color:{theme.accent_success}">&#x25C6;</span>
    = start, <span style="color:{theme.accent_danger}">&#x25C6;</span> = end.
  </div>
  """, unsafe_allow_html=True)

  # ── Phase Explorer interpretation ─────────────────────────────────────────
  try:
    n_ = len(x_vals) - 1
    _seg = [(x_vals[i+1]-x_vals[i])**2 + (y_vals[i+1]-y_vals[i])**2 for i in range(n_)]
    if is_3d:
      _seg = [_seg[i] + (z_vals[i+1]-z_vals[i])**2 for i in range(n_)]
    _path_len = sum(s**0.5 for s in _seg)
    _dx = x_vals[-1] - x_vals[0]; _dy = y_vals[-1] - y_vals[0]
    _dz = (z_vals[-1] - z_vals[0]) if is_3d else 0
    _disp = (_dx**2 + _dy**2 + _dz**2)**0.5
    _straight = float(_disp / max(_path_len, 1e-9))
    _vx = [x_vals[i+1]-x_vals[i] for i in range(n_)]
    _reversals = sum(1 for i in range(1, len(_vx)) if _vx[i]*_vx[i-1] < 0)
    if _straight > 0.8:
      _shape = "nearly straight — the economy is adjusting <b>monotonically</b> toward a new equilibrium with minimal oscillation"
    elif _straight > 0.45:
      _shape = "moderately curved — <b>some oscillatory adjustment</b> is occurring (common in demand-led models)"
    else:
      _shape = "highly curved or looping — <b>strong oscillatory or cyclical dynamics</b> are present; the economy visits the same region of state space multiple times"
    _rev_note = (f" {_reversals} directional reversal(s) detected on the {x_label} axis — consistent with boom-bust cycling."
                 if _reversals > 1 else "")
    _interp_phase = (
      f"Path straightness index: <b>{_straight:.2f}</b> (1.0 = perfectly straight, 0.0 = closed loop). "
      f"The trajectory is {_shape}.{_rev_note} "
      f"Net displacement: {x_label} {_dx:+.3f}, {y_label} {_dy:+.3f}"
      + (f", {z_label} {_dz:+.3f}" if is_3d else "") + "."
    )
    if _straight > 0.7:
      _rec_phase = (
        "The system is converging cleanly. Policy interventions made at any point will propagate "
        "smoothly without cycle amplification. A good time for structural reforms is when the "
        "trajectory is still far from its endpoint (large displacement remaining)."
      )
    elif _reversals > 2:
      _rec_phase = (
        "Oscillatory dynamics detected. Time policy changes to the troughs of the cycle — "
        "stimulus applied at the bottom of a downswing has maximum stabilising effect. "
        "Avoid large interventions at peaks where they may amplify the next downswing."
      )
    else:
      _rec_phase = (
        "Moderate oscillation present. Monitor trajectory in real-time for inflection points. "
        "Gradual, rule-based policy adjustments are preferable to discrete shocks, as the "
        "system is sensitive to impulse timing."
      )
  except Exception:
    _interp_phase = "Examine the trajectory shape directly: straight = convergent, spiral = oscillatory, loop = persistent cycle."
    _rec_phase = "Use the 3D/2D toggle and rotate the view to identify oscillatory patterns."

  _guide(
    theme,
    method_html=(
      "The <b>phase space</b> is the full set of possible combinations of the economy's state variables. "
      "Instead of plotting each variable against time, this diagram plots variables <i>against each other</i>, "
      "revealing the system's fundamental dynamic structure.<br><br>"
      "The trajectory's <b>shape on the manifold</b> tells you everything:<br>"
      "<ul style='margin:0.3rem 0; padding-left:1.2rem;'>"
      "<li><b>Straight line</b> → monotonic convergence to new equilibrium (clean adjustment)</li>"
      "<li><b>Spiral inward</b> → damped oscillation — boom-bust cycles that gradually fade</li>"
      "<li><b>Closed loop</b> → limit cycle — self-sustaining boom-bust with no convergence</li>"
      "<li><b>Chaotic scatter</b> → sensitive dependence on initial conditions</li>"
      "</ul>"
      "In 3D mode, the economy traces a <b>space curve on a 3-dimensional manifold</b> — a surface "
      "that shows which combinations of state variables are actually reachable. Not all regions of "
      "the cube are visited; the trajectory stays on the manifold determined by the model's structure."
    ),
    interp_html=_interp_phase,
    rec_html=_rec_phase,
  )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 7: IMPULSE RESPONSE FUNCTIONS — auto-detected deviation from baseline
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
    baseline (t<{onset}). Positive = above baseline, negative = below.
    Values are normalised to percentage deviation where possible.
  </div>
  """, unsafe_allow_html=True)

  # ── IRF interpretation ────────────────────────────────────────────────────
  try:
    _peak_devs = {}
    _recovery_qs = {}
    for _dim in selected:
      _base = baseline.get(_dim, 0)
      _devs = [(f.get("outcomes", {}).get(_dim, 0) - _base) for f in post_shock]
      if abs(_base) > 1e-8:
        _devs = [v / abs(_base) * 100 for v in _devs]
      if _devs:
        _pk = max(_devs, key=abs)
        _peak_devs[_dim] = _pk
        _half = abs(_pk) * 0.1
        _rec_q = next((i for i, v in enumerate(_devs) if abs(v) <= _half and i > 0), None)
        _recovery_qs[_dim] = _rec_q
    _most_resp_dim, _most_resp_val = max(_peak_devs.items(), key=lambda x: abs(x[1])) if _peak_devs else (selected[0], 0)
    _rec_q_most = _recovery_qs.get(_most_resp_dim)
    _rec_note = (f"returning to within 10% of baseline by quarter +{_rec_q_most}" if _rec_q_most else
                 "not yet returning to baseline within the simulation horizon")
    _persistent = [d for d, q in _recovery_qs.items() if q is None]
    _persist_note = (f" {len(_persistent)} dimension(s) ({', '.join(dim_label(d) for d in _persistent[:3])}) "
                     "show persistent deviation and do not recover — structural adjustment required." if _persistent else
                     " All selected dimensions eventually return toward baseline.")
    _interp_irf = (
      f"Most responsive dimension: <b>{dim_label(_most_resp_dim)}</b> with a peak deviation of "
      f"<b>{_most_resp_val:+.1f}%</b>, {_rec_note}.{_persist_note}"
    )
    if _persistent:
      _rec_irf = (
        f"The persistent deviation in {', '.join(dim_label(d) for d in _persistent[:2])} signals a <b>structural shift</b> "
        f"— the economy has moved to a new equilibrium, not just a temporary displacement. "
        "Policy should target the new steady-state level, not attempt to restore the pre-shock baseline."
      )
    elif _rec_q_most and _rec_q_most <= 4:
      _rec_irf = (
        "Quick recovery observed. The shock transmits sharply but the economy self-corrects within 4 quarters. "
        "Short-term stabilisation policy (liquidity support, automatic stabilisers) is sufficient — "
        "avoid longer-term structural interventions that could overshoot."
      )
    else:
      _rec_irf = (
        "Slow recovery path detected. The shock has lasting transmission effects. "
        "Consider sustained policy support for the most-affected dimensions until the trajectory "
        "returns to within 10% of baseline. Monitor for second-round effects."
      )
  except Exception:
    _interp_irf = "Examine which lines deviate most from the zero baseline and whether they converge back."
    _rec_irf = "Dimensions that never return to zero require structural policy adjustment, not just short-term stabilisation."

  _guide(
    theme,
    method_html=(
      "An <b>Impulse Response Function (IRF)</b> measures how each economic variable responds to a sudden shock. "
      "The pre-shock average (quarters before shock onset) is used as the baseline. "
      "Each dimension is then tracked as a <b>percentage deviation</b> from that baseline in subsequent quarters.<br><br>"
      "In the <b>3D surface view</b>: X = quarters after shock, Y = each outcome dimension, Z = % deviation. "
      "Colours near zero = muted response; deep red/blue = strong shock transmission. "
      "Rotating the surface lets you compare recovery speeds across all dimensions simultaneously.<br><br>"
      "Key concepts: <b>impact response</b> = peak deviation at or near shock onset; "
      "<b>persistence</b> = how many quarters the deviation lasts; "
      "<b>permanent shift</b> = deviation that never returns to zero (new equilibrium)."
    ),
    interp_html=_interp_irf,
    rec_html=_rec_irf,
  )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 8: FLOW DYNAMICS — Sankey + waterfall + 3D surface
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
  if sorted_flows:
    _top_pos = [(dim_label(k), v) for k, v in sorted_flows if v >= 0][:2]
    _top_neg = [(dim_label(k), v) for k, v in sorted_flows if v < 0][:2]
    _cap_parts = []
    if _top_pos:
      _cap_parts.append("Largest inflows: " + ", ".join(f"{n} (+{v:.4f})" for n, v in _top_pos))
    if _top_neg:
      _cap_parts.append("largest outflows: " + ", ".join(f"{n} ({v:.4f})" for n, v in _top_neg))
    st.markdown(f"""
    <div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">
      <b>Reading:</b> Bars show each flow's size at quarter {frame.get('t', t_idx)}.
      Green = net inflow (resources entering the economy); red = net outflow (resources leaving).
      {' &nbsp;·&nbsp; '.join(_cap_parts) if _cap_parts else ''}
    </div>
    """, unsafe_allow_html=True)

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
    st.markdown(f"""
    <div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">
      <b>Reading:</b> Ribbon width = flow magnitude. Resources entering the economy
      flow from the "Inflows" node outward; those leaving flow toward "Outflows".
      Wider ribbons indicate larger transfers between economic sectors at this quarter.
    </div>
    """, unsafe_allow_html=True)

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
    st.markdown(f"""
    <div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">
      <b>How to read this 3D surface:</b> The X-axis (depth) = individual flow types;
      Y-axis = quarter; Z-axis (height) = magnitude. Peaks show quarters when a flow
      surged. Drag to rotate; scroll to zoom. Colour encodes magnitude (bright = high).
    </div>
    """, unsafe_allow_html=True)

  # ── Flow interpretation ───────────────────────────────────────────────────
  try:
    _net = sum(v for _, v in sorted_flows)
    _pos_total = sum(v for _, v in sorted_flows if v > 0)
    _neg_total = sum(v for _, v in sorted_flows if v < 0)
    _n_pos = sum(1 for _, v in sorted_flows if v > 0)
    _n_neg = sum(1 for _, v in sorted_flows if v < 0)
    _net_sign = "net inflow" if _net > 0 else "net outflow"
    _interp_flow = (
      f"At quarter {frame.get('t', t_idx)}: {_n_pos} inflow(s) totalling <b>{_pos_total:+.4f}</b> "
      f"and {_n_neg} outflow(s) totalling <b>{_neg_total:+.4f}</b>. "
      f"<b>Net balance: {_net:+.4f}</b> ({_net_sign}). "
      + ("The economy is absorbing more resources than it releases — consistent with expansion or investment phase."
         if _net > 0.001 else
         "The economy is releasing more resources than it absorbs — consistent with deleveraging, fiscal drain, or external drain."
         if _net < -0.001 else
         "Flows are near-balanced — the economy is in accounting equilibrium at this snapshot.")
    )
    if abs(_net) < 0.001:
      _rec_flow = (
        "Balanced flows signal accounting consistency. Use the time-slider to check earlier quarters — "
        "if flows were imbalanced in prior periods, the economy has since self-corrected. "
        "Enable the 3D surface to see the full temporal pattern."
      )
    elif _net > 0:
      _rec_flow = (
        "Net inflow detected. Verify the source: if driven by credit expansion or capital inflows, "
        "ensure the accumulation is sustainable (check private credit and current account dimensions). "
        "Inflows funded by debt require careful fiscal monitoring."
      )
    else:
      _rec_flow = (
        "Net outflow detected. Identify the dominant outflow channel and assess whether it is "
        "intentional (e.g. debt repayment, import-led growth) or a vulnerability signal "
        "(e.g. capital flight, fiscal drain). Compare against the Open Economy tab for external sector context."
      )
  except Exception:
    _interp_flow = "Use the waterfall chart to identify the largest inflows and outflows at the selected quarter."
    _rec_flow = "Move the time-slider to observe how the flow structure evolves — sudden changes indicate shock transmission."

  _guide(
    theme,
    method_html=(
      "Economic <b>flows</b> are transfers between sectors — households consuming from firms, "
      "firms paying taxes to government, banks extending credit, the external sector remitting or absorbing funds. "
      "The <b>waterfall bar chart</b> shows each flow's net magnitude at the selected quarter: "
      "green = net inflow to the system, red = net outflow.<br><br>"
      "The <b>Sankey diagram</b> visualises magnitudes as ribbon widths — wider = larger transfer. "
      "All flows originate from 'Inflows' or terminate at 'Outflows' to show directionality.<br><br>"
      "The <b>3D flow surface</b> (optional) adds the time dimension: X = flow type, Y = quarter, "
      "Z = magnitude. A peak on the surface marks a quarter when a specific flow surged — "
      "useful for identifying when shocks transmitted through specific channels."
    ),
    interp_html=_interp_flow,
    rec_html=_rec_flow,
  )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 9: MONTE CARLO — Fan charts with parameter jitter
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

  c1, c2, c3, c4 = st.columns(4)
  with c1:
    focus_dim = st.selectbox("Dimension", outcome_keys,
                 format_func=dim_label, key="mc_dim")
  with c2:
    n_runs = st.slider("Number of runs", 10, 200, 50, 10, key="mc_runs")
  with c3:
    jitter_pct = st.slider("Parameter jitter (%)", 1, 25, 10, 1, key="mc_jitter")
  with c4:
    dist_type = st.selectbox(
      "Sampling distribution",
      ["Gaussian", "Log-Normal", "Uniform"],
      index=0, key="mc_dist",
      help="Gaussian: normal noise centred on base value. "
           "Log-Normal: multiplicative skewed noise. "
           "Uniform: flat ±jitter (original behaviour).",
    )

  # Parameter groups for correlated sampling (same draw shared within each group)
  _PARAM_GROUPS = {
    "propensity": ["consumer_propensity", "import_propensity", "export_propensity"],
    "fiscal": ["spending_ratio", "tax_rate"],
    "monetary": ["interest_rate", "reserve_ratio"],
    "credit": ["loan_default_rate", "credit_expansion_rate"],
  }

  corr_mode = st.checkbox(
    "Correlated group sampling — params in the same economic group move together",
    value=False, key="mc_corr",
    help="When enabled, parameters in the same group (e.g. all propensity params) "
         "share the same random draw so they move in the same direction, "
         "producing more realistic macro uncertainty.",
  )

  if st.button("Run Monte Carlo", key="mc_go", type="primary"):
    calib = st.session_state.get("sim_calibration")
    steps = st.session_state.get("sim_steps", 50)
    scenario = st.session_state.get("_sim_scenario_obj")

    if not calib:
      st.error("No calibration found. Run base simulation first.")
      return

    sigma = jitter_pct / 100.0

    def _draw(rng_state=None):
      """Return a scalar jitter multiplier using the chosen distribution."""
      if dist_type == "Gaussian":
        return 1.0 + np.random.normal(0.0, sigma)
      elif dist_type == "Log-Normal":
        return float(np.exp(np.random.normal(0.0, sigma)))
      else:  # Uniform
        return 1.0 + np.random.uniform(-sigma, sigma)

    progress = st.progress(0, text="Running Monte Carlo simulations...")
    mc_trajectories = []
    # Store per-run param vectors for sensitivity decomposition
    mc_param_draws: list[dict] = []

    numeric_fields = [
      f for f in SFCConfig.__dataclass_fields__
      if isinstance(getattr(calib.config, f, None), (int, float))
      and f not in ("steps", "dt")
    ]

    # Build reverse lookup: field → group name
    field_to_group: dict[str, str] = {}
    for grp, fields in _PARAM_GROUPS.items():
      for fld in fields:
        field_to_group[fld] = grp

    for i in range(n_runs):
      try:
        # One shared draw per group when correlated mode is on
        group_draws: dict[str, float] = {g: _draw() for g in _PARAM_GROUPS} if corr_mode else {}

        cfg_dict = {}
        run_draws: dict[str, float] = {}
        for field_name in SFCConfig.__dataclass_fields__:
          val = getattr(calib.config, field_name, None)
          if isinstance(val, (int, float)) and field_name not in ("steps", "dt"):
            if corr_mode and field_name in field_to_group:
              j = group_draws[field_to_group[field_name]]
            else:
              j = _draw()
            run_draws[field_name] = j
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
        mc_param_draws.append(run_draws)
      except Exception:
        pass

      progress.progress((i + 1) / n_runs, text=f"Run {i + 1}/{n_runs}...")

    progress.empty()

    if len(mc_trajectories) < 5:
      st.warning(f"Only {len(mc_trajectories)} successful runs. Try reducing jitter.")
      return

    st.session_state["mc_trajectories"] = mc_trajectories
    st.session_state["mc_focus_dim"] = focus_dim
    st.session_state["mc_param_draws"] = mc_param_draws
    st.session_state["mc_numeric_fields"] = numeric_fields

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
    {len(mc_trajs)} successful runs &nbsp;|&nbsp; Distribution: {st.session_state.get('mc_dist','Gaussian')}
    &nbsp;|&nbsp; Jitter: &plusmn;{jitter_pct}%
    &nbsp;|&nbsp; Bands: 10th–90th and 25th–75th percentiles
  </div>
  """, unsafe_allow_html=True)

  # ── Monte Carlo interpretation ────────────────────────────────────────────
  try:
    _p10_t = bands[10][-1]; _p90_t = bands[90][-1]; _p50_t = bands[50][-1]
    _base_t = base_vals[-1] if base_vals else _p50_t
    _width = abs(_p90_t - _p10_t)
    _width_pct = _width / max(abs(_p50_t), 1e-9) * 100
    _base_pct = (_base_t - _p10_t) / max(_width, 1e-9)
    if _width_pct < 5:
      _unc = "narrow"; _unc_msg = "parameter uncertainty has minimal effect on this outcome"
    elif _width_pct < 20:
      _unc = "moderate"; _unc_msg = "the outcome is meaningfully sensitive to parameter uncertainty"
    else:
      _unc = "wide"; _unc_msg = "the outcome is highly sensitive — key parameters are poorly constrained"
    if _base_pct > 0.75:
      _pos_msg = "above the 75th percentile — the base calibration is <b>optimistic</b> relative to the uncertainty distribution"
    elif _base_pct < 0.25:
      _pos_msg = "below the 25th percentile — the base calibration is <b>conservative</b>"
    else:
      _pos_msg = "within the central 50% band — the base calibration is <b>representative</b>"
    _interp_mc = (
      f"Across {len(mc_trajs)} runs with ±{jitter_pct}% {st.session_state.get('mc_dist','Gaussian')} jitter: "
      f"the 10th–90th percentile band at the terminal quarter spans "
      f"<b>{_p10_t:.3f} → {_p90_t:.3f}</b> "
      f"(width = {_width:.3f}, or <b>{_width_pct:.0f}% of the median</b>). "
      f"Uncertainty level: <b>{_unc}</b> — {_unc_msg}. "
      f"The base run ends at {_base_t:.3f}, which is {_pos_msg}."
    )
    if _unc == "narrow":
      _rec_mc = (
        "Low uncertainty: the model's output for this dimension is robust to parameter perturbations. "
        "Policy conclusions drawn from the base run can be stated with high confidence. "
        "Consider increasing jitter or testing additional dimensions where uncertainty may be higher."
      )
    elif _unc == "wide":
      _rec_mc = (
        "High uncertainty: expand the Parameter Sensitivity Decomposition below to identify which "
        "parameters drive the spread. Prioritise collecting better empirical data for the top-ranked "
        "parameters. Avoid single-point policy recommendations — present the full fan chart to stakeholders."
      )
    else:
      _rec_mc = (
        "Moderate uncertainty: the base run is a reasonable central estimate, but the fan chart should "
        "accompany any policy recommendation as a risk envelope. "
        "Check the Sensitivity Decomposition below to identify which 2–3 parameters explain most of the spread."
      )
  except Exception:
    _interp_mc = "Run Monte Carlo to see band width and base run position within the uncertainty envelope."
    _rec_mc = "Narrow bands = robust conclusions; wide bands = improve parameter data quality before committing to policy."

  _guide(
    theme,
    method_html=(
      "Monte Carlo simulation runs the model <b>N times</b>, each with randomly perturbed structural parameters, "
      "then summarises the spread of outcomes as percentile bands.<br><br>"
      "<b>What is a jitter?</b> A jitter is a random multiplier applied to each calibrated parameter before a run. "
      "It represents structural uncertainty — the fact that no calibration is exact. "
      f"With σ = {jitter_pct}%, each parameter is drawn from around ×1.0 with spread ±{jitter_pct}%, "
      "so a parameter might be ×0.88 or ×1.14 in any given run.<br><br>"
      "<b>Sampling distributions:</b>"
      "<ul style='margin:0.3rem 0; padding-left:1.2rem;'>"
      "<li><b>Gaussian</b> — symmetric bell curve; most draws are near the base value. Best for additive measurement error.</li>"
      "<li><b>Log-Normal</b> — right-skewed, always positive. Appropriate for multiplicative parameters like propensity scores that cannot go negative.</li>"
      "<li><b>Uniform</b> — all values in [1−σ, 1+σ] equally likely. Maximum entropy; use when the shape of uncertainty is unknown.</li>"
      "</ul>"
      "<b>Reading the fan chart:</b> Inner band (25th–75th pct) = likely range; "
      "outer band (10th–90th pct) = plausible worst-to-best envelope. "
      "The dashed line = your deterministic base run. "
      "The 3D uncertainty surface adds a percentile axis so you can see how the distribution evolves over time."
    ),
    interp_html=_interp_mc,
    rec_html=_rec_mc,
  )

  # ── Parameter Sensitivity Decomposition ───────────────────────────────────
  mc_draws = st.session_state.get("mc_param_draws", [])
  numeric_fields_saved = st.session_state.get("mc_numeric_fields", [])
  if mc_draws and numeric_fields_saved and len(mc_trajs) >= 5:
    with st.expander("Parameter Sensitivity Decomposition", expanded=False):
      st.caption(
        "Rank correlation (Spearman) between each parameter's jitter multiplier and "
        "the terminal outcome value. Parameters with high |correlation| are the primary "
        "drivers of output uncertainty."
      )
      # Extract terminal outcome per run
      terminal_vals = np.array([
        traj[-1].get("outcomes", {}).get(focus, 0) for traj in mc_trajs
      ])
      sensitivities = []
      for fld in numeric_fields_saved:
        draws = np.array([d.get(fld, 1.0) for d in mc_draws])
        if draws.std() < 1e-9:
          continue
        # Spearman rank correlation
        rank_d = np.argsort(np.argsort(draws)).astype(float)
        rank_o = np.argsort(np.argsort(terminal_vals)).astype(float)
        n_ = len(rank_d)
        rho = 1.0 - 6.0 * float(np.sum((rank_d - rank_o) ** 2)) / max(1, n_ * (n_ ** 2 - 1))
        sensitivities.append((fld, rho))

      sensitivities.sort(key=lambda x: abs(x[1]), reverse=True)
      top_n = sensitivities[:12]

      if top_n:
        labels = [dim_label(f) for f, _ in top_n]
        rhos = [r for _, r in top_n]
        colors = [PALETTE[0] if r >= 0 else PALETTE[3] for r in rhos]
        fig_sens = go.Figure(go.Bar(
          x=labels, y=rhos,
          marker_color=colors,
          text=[f"{r:+.3f}" for r in rhos],
          textposition="outside",
          hovertemplate="<b>%{x}</b><br>Spearman ρ: %{y:+.3f}<extra></extra>",
        ))
        fig_sens.update_layout(**base_layout(theme, height=340,
          title=dict(
            text=f"Parameter Sensitivity — {dim_label(focus)} (Spearman ρ)",
            font=dict(color=theme.text_muted, size=12)),
          xaxis=dict(tickangle=-30, title="Parameter"),
          yaxis=dict(title="Spearman ρ", zeroline=True,
                zerolinecolor=theme.border_default, range=[-1.05, 1.05]),
        ))
        st.plotly_chart(fig_sens, use_container_width=True)
        st.markdown(f"""
        <div style="font-size:0.78rem; color:{theme.text_muted};">
          <b>Reading:</b> Green bars = parameter increases raise the outcome;
          red bars = parameter increases lower the outcome.
          ρ close to ±1 = that parameter almost fully determines the uncertainty.
          ρ near 0 = that parameter's uncertainty has negligible impact on the outcome.
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 10: STRESS TEST MATRIX — all scenarios × all outcomes
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
  if z_matrix and scenarios and all_dims:
    _flat = [(scenarios[r], all_dims[c], z_matrix[r][c])
             for r in range(len(scenarios)) for c in range(len(all_dims))]
    _worst_s, _worst_d, _worst_v = min(_flat, key=lambda x: x[2])
    _best_s, _best_d, _best_v = max(_flat, key=lambda x: x[2])
    st.markdown(f"""
    <div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">
      <b>Reading:</b> Each cell = end-minus-start change for a scenario × outcome pair.
      Green = improvement; red = deterioration.
      Worst outcome: <b>{_worst_s}</b> hits <b>{dim_label(_worst_d)}</b> hardest
      ({_worst_v:+.3f}).
      Best outcome: <b>{_best_s}</b> improves <b>{dim_label(_best_d)}</b> most
      ({_best_v:+.3f}).
    </div>
    """, unsafe_allow_html=True)

  # 3D Stress Surface
  # ── Stress Matrix interpretation ──────────────────────────────────────────
  try:
    _all_deltas = [(s, d, z_matrix[si][di])
                   for si, s in enumerate(scenarios)
                   for di, d in enumerate(all_dims)]
    _scenario_damage = {}
    for s in scenarios:
      _neg = [z_matrix[scenarios.index(s)][di] for di, d in enumerate(all_dims)
              if z_matrix[scenarios.index(s)][di] < 0]
      _scenario_damage[s] = sum(_neg)
    _worst_scen = min(_scenario_damage.items(), key=lambda x: x[1])
    _best_scen = max(_scenario_damage.items(), key=lambda x: x[1])
    _dim_avg_impact = {}
    for di, d in enumerate(all_dims):
      _vals = [z_matrix[si][di] for si in range(len(scenarios))]
      _dim_avg_impact[d] = float(sum(_vals) / max(len(_vals), 1))
    _most_vulnerable_dim = min(_dim_avg_impact.items(), key=lambda x: x[1])
    _most_resilient_dim = max(_dim_avg_impact.items(), key=lambda x: x[1])
    _interp_stress = (
      f"Across {len(scenarios)} scenario(s) and {len(all_dims)} outcome dimension(s):<br>"
      f"• <b>Most damaging scenario</b>: {_worst_scen[0]} (total negative impact: {_worst_scen[1]:+.3f})<br>"
      f"• <b>Least damaging scenario</b>: {_best_scen[0]} (total negative impact: {_best_scen[1]:+.3f})<br>"
      f"• <b>Most vulnerable outcome</b>: {dim_label(_most_vulnerable_dim[0])} "
      f"(avg cross-scenario change: {_most_vulnerable_dim[1]:+.3f})<br>"
      f"• <b>Most resilient outcome</b>: {dim_label(_most_resilient_dim[0])} "
      f"(avg cross-scenario change: {_most_resilient_dim[1]:+.3f})"
    )
    _rec_stress = (
      f"Prioritise policy buffers against <b>{_worst_scen[0]}</b> — it inflicts the greatest cumulative damage. "
      f"Pay particular attention to protecting <b>{dim_label(_most_vulnerable_dim[0])}</b>, "
      f"which deteriorates most consistently across scenarios. "
      "Use the 3D surface to visually identify scenario-outcome 'valleys' (deepest red) — these are your highest-priority risk pairs."
    )
  except Exception:
    _interp_stress = "Review the heatmap: deep red cells are your highest-priority scenario-outcome risk pairs."
    _rec_stress = "Design policy buffers around the scenarios that produce the most red cells — particularly for outcomes that are red across multiple scenarios."

  _guide(
    theme,
    method_html=(
      "<b>Stress testing</b> systematically runs <i>every</i> scenario in the library through the model and "
      "records the end-minus-start change for every outcome dimension. This creates a full "
      "<b>scenario × outcome impact matrix</b>.<br><br>"
      "Each cell value = (terminal value − initial value) for a specific crisis scenario hitting a specific outcome. "
      "Green = improvement; red = deterioration. "
      "The matrix answers: <i>'Which outcome is most consistently damaged across all scenarios?'</i> "
      "and <i>'Which scenario causes the most widespread damage?'</i><br><br>"
      "The optional <b>3D stress surface</b> maps scenarios to the Y-axis, outcomes to the X-axis, "
      "and impact to the Z-axis (height). Red valleys = acute vulnerabilities; green peaks = resilient combinations. "
      "Rotating the surface reveals clusters of co-occurring vulnerabilities."
    ),
    interp_html=_interp_stress,
    rec_html=_rec_stress,
  )

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
    st.markdown(f"""
    <div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">
      <b>How to read this 3D stress surface:</b> X-axis = outcome dimensions;
      Y-axis = crisis scenarios; Z-axis (height) = impact delta.
      Red valleys = severe deterioration; green peaks = resilience.
      Drag to rotate and identify which scenario-outcome combinations are most severe.
    </div>
    """, unsafe_allow_html=True)
