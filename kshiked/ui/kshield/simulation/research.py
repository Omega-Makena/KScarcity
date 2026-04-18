"""Research module tabs: IO Sectors, Inequality, Financial, Open Economy, Research Engine."""
from ._shared import (st, pd, np, go, make_subplots, HAS_DATA_STACK, HAS_PLOTLY, PALETTE, base_layout)


def _hex_rgba(hex6: str, alpha: float = 0.08) -> str:
  """Convert #rrggbb hex to rgba(r,g,b,alpha) — Plotly fillcolor rejects 8-char hex."""
  h = hex6.lstrip('#')
  r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
  return f"rgba({r},{g},{b},{alpha})"


def _guide(theme, method_html: str, interp_html: str, rec_html: str) -> None:
  """Collapsible 3-section analysis guide (method / interpretation / recommendation)."""
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
    st.caption("The Leontief inverse shows total production required (direct + indirect) to deliver one unit of final demand. Values > 1.0 indicate strong multiplier effects across the supply chain.")

  try:
    _g_A_max_idx = np.unravel_index(A_matrix.argmax(), A_matrix.shape)
    _g_from = display_names[_g_A_max_idx[1]]
    _g_to = display_names[_g_A_max_idx[0]]
    _g_coeff = float(A_matrix[_g_A_max_idx])
    _g_L_max = float(L_matrix.max())
    _g_L_max_idx = np.unravel_index(L_matrix.argmax(), L_matrix.shape)
    _g_L_from = display_names[_g_L_max_idx[1]]
    _g_L_to = display_names[_g_L_max_idx[0]]
  except Exception:
    _g_from, _g_to, _g_coeff, _g_L_max, _g_L_from, _g_L_to = "?", "?", 0, 0, "?", "?"
  _guide(theme,
    method_html=(
      "The <b>Leontief Input-Output model</b> captures how sectors depend on each other for production. "
      "The <b>A matrix</b> (technical coefficients) shows what fraction of sector j's output is used as "
      "input by sector i — a value of 0.3 means sector i buys 30 cents worth of inputs from sector j per "
      "unit of its own output. The <b>Leontief inverse (I–A)⁻¹</b> extends this to capture indirect effects: "
      "if final demand for sector j rises by 1 unit, how much total gross output does every sector in the "
      "economy need to produce to satisfy it? Values above 1.0 indicate multiplier effects through the supply chain."
    ),
    interp_html=(
      f"Strongest direct dependency: <b>{_g_from}</b> → <b>{_g_to}</b> "
      f"(technical coefficient: <b>{_g_coeff:.3f}</b>). "
      f"Highest Leontief multiplier: <b>{_g_L_from}</b> supplying <b>{_g_L_to}</b> "
      f"(total output requirement: <b>{_g_L_max:.3f}</b>). "
      "Values greater than 2.0 in the inverse indicate strong supply-chain amplification."
    ),
    rec_html=(
      f"A demand stimulus targeting <b>{_g_L_from}</b> will have the largest multiplied "
      "effect through the economy due to its high Leontief coefficient. "
      "Sectors with high backward linkages are ideal targets for investment policy. "
      "Watch for supply bottlenecks in the highest-coefficient cell — disruption there propagates widely."
    ),
  )

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
    def _to_rgba(hex_c, alpha=0.2):
      h = hex_c.lstrip('#')
      return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha})"

    fig_sankey = go.Figure(go.Sankey(
      arrangement='snap',
      node=dict(label=node_labels, color=node_colors, pad=15, thickness=18,
           line=dict(color=theme.border_default, width=0.5)),
      link=dict(source=src_nodes, target=tgt_nodes, value=flow_vals,
           label=flow_labels,
           color=[_to_rgba(PALETTE[s % len(PALETTE)], 0.2) for s in src_nodes]),
    ))
    fig_sankey.update_layout(**base_layout(theme, height=400,
      title=dict(text="Intermediate Input Flows Between Sectors",
            font=dict(color=theme.text_muted, size=13))))
    st.plotly_chart(fig_sankey, use_container_width=True)
    _idx_max = flow_vals.index(max(flow_vals))
    st.markdown(f"""<div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">Wider ribbons = larger intermediate flows. Largest flow: <b>{flow_labels[_idx_max]}</b>. These flows represent industries buying inputs from each other — thicker connections mean stronger inter-sector dependency.</div>""", unsafe_allow_html=True)

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
  st.caption("Each colour band shows a sector's share of total output. A shrinking band means structural decline; a growing band means economic ascent.")

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
  st.caption("Sectors above the dashed diagonal produce more output per worker than their employment share would suggest (high productivity). Sectors below are labor-intensive relative to their output contribution. Bubble size = gross output.")

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

  try:
    _g_top_mult_idx = int(np.argmax(multipliers))
    _g_top_mult = display_names[_g_top_mult_idx]
    _g_mult_val = float(multipliers[_g_top_mult_idx])
    _g_top_bwd_idx = int(np.argmax(backward))
    _g_top_bwd = display_names[_g_top_bwd_idx]
    _g_top_fwd_idx = int(np.argmax(forward))
    _g_top_fwd = display_names[_g_top_fwd_idx]
  except Exception:
    _g_top_mult, _g_mult_val, _g_top_bwd, _g_top_fwd = "?", 0, "?", "?"
  _guide(theme,
    method_html=(
      "<b>Output multipliers</b> (column sum of Leontief inverse) show how much total economy-wide output "
      "is generated when final demand for that sector rises by 1 unit. "
      "<b>Backward linkages</b> measure how much a sector purchases from other sectors (it buys a lot = "
      "high backward linkage = good demand stimulus). "
      "<b>Forward linkages</b> measure how much a sector's output is used by other sectors as input "
      "(it supplies a lot = high forward linkage = strategic infrastructure sector)."
    ),
    interp_html=(
      f"Highest output multiplier: <b>{_g_top_mult}</b> at <b>{_g_mult_val:.2f}</b> — each unit of "
      f"demand stimulus here generates {_g_mult_val:.2f}x in total economy-wide output. "
      f"Strongest backward linkage: <b>{_g_top_bwd}</b> (buys most from other sectors). "
      f"Strongest forward linkage: <b>{_g_top_fwd}</b> (supplies most to other sectors as input)."
    ),
    rec_html=(
      f"Target <b>{_g_top_mult}</b> for demand-side stimulus (highest multiplier). "
      f"Target <b>{_g_top_bwd}</b> sectors for supply chain resilience policy. "
      f"Protect <b>{_g_top_fwd}</b> — disruption to high-forward-linkage sectors cascades widely."
    ),
  )

  # ── 6. SECTOR-VS-SECTOR SHOCK & POLICY CLASSIFIER ──────────────────────
  st.markdown(f"<div style='color:{theme.accent_warning}; font-weight:600; "
        f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
        f"SECTOR CLASSIFICATION: SHOCKS &amp; POLICIES VS BENCHMARK SECTOR</div>",
        unsafe_allow_html=True)

  benchmark_sector = st.selectbox(
    "Benchmark sector",
    options=sector_names,
    format_func=lambda x: str(x).title(),
    key="io_sector_classifier_benchmark",
    help="Classifies every sector's shock risk and policy priority relative to the selected sector.",
  )

  idx_by_sector = {name: i for i, name in enumerate(sector_names)}
  benchmark_idx = idx_by_sector[benchmark_sector]
  shock_sensitivity_cfg = getattr(getattr(research_econ, "io_cfg", None), "shock_sensitivity", {}) or {}

  def _minmax(vals):
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
      return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-9:
      return np.ones_like(arr) * 0.5
    return (arr - lo) / (hi - lo)

  def _norm_key(txt: str) -> str:
    return "".join(ch for ch in str(txt).lower() if ch.isalnum())

  def _shock_class(score: float) -> str:
    if score >= 0.67:
      return "High Spillover Risk"
    if score >= 0.40:
      return "Moderate Spillover Risk"
    return "Contained Spillover Risk"

  def _policy_class(score: float) -> str:
    if score >= 0.70:
      return "Tier 1 Policy Priority"
    if score >= 0.45:
      return "Tier 2 Policy Priority"
    return "Monitor / Stabilize"

  dep_vs_benchmark = []
  spillover_vs_benchmark = []
  sensitivity_avg = []

  for sec in sector_names:
    i = idx_by_sector[sec]
    # Two-way direct dependence in A captures bilateral supply-chain coupling.
    dep_in = float(A_matrix[i, benchmark_idx])
    dep_out = float(A_matrix[benchmark_idx, i])
    dep_vs_benchmark.append(0.5 * (dep_in + dep_out))

    # Two-way Leontief pass-through captures direct + indirect cascade effects.
    spill_in = float(L_matrix[i, benchmark_idx])
    spill_out = float(L_matrix[benchmark_idx, i])
    spillover_vs_benchmark.append(0.5 * (spill_in + spill_out))

    sens = shock_sensitivity_cfg.get(sec, {})
    sensitivity_avg.append(float(np.mean(list(sens.values()))) if sens else 0.0)

  dep_norm = _minmax(dep_vs_benchmark)
  spill_norm = _minmax(spillover_vs_benchmark)
  sens_norm = _minmax(sensitivity_avg)

  mult_norm = _minmax(multipliers)
  bwd_norm = _minmax(backward)
  fwd_norm = _minmax(forward)

  shock_scores = 0.45 * dep_norm + 0.35 * spill_norm + 0.20 * sens_norm
  policy_scores = 0.50 * mult_norm + 0.30 * bwd_norm + 0.20 * fwd_norm

  try:
    from kshiked.simulation.scenario_templates import SHOCK_REGISTRY, POLICY_INSTRUMENT_REGISTRY
  except Exception:
    SHOCK_REGISTRY, POLICY_INSTRUMENT_REGISTRY = {}, {}

  shock_families = {s: set() for s in sector_names}
  for sk, meta in SHOCK_REGISTRY.items():
    sec_norm = _norm_key(meta.get("sector", ""))
    if not sec_norm:
      continue
    label = str(meta.get("label", sk))
    for sec in sector_names:
      io_norm = _norm_key(sec)
      if sec_norm == io_norm:
        shock_families[sec].add(label)
      elif sec_norm == "foodmarkets" and io_norm in {"agriculture", "manufacturing", "services"}:
        shock_families[sec].add(label)
      elif sec_norm == "communications" and io_norm == "services":
        shock_families[sec].add(label)
      elif sec_norm == "displacement" and io_norm in {"services", "security"}:
        shock_families[sec].add(label)

  policy_tools = {s: set() for s in sector_names}
  for pk, meta in POLICY_INSTRUMENT_REGISTRY.items():
    cat_norm = _norm_key(meta.get("category", ""))
    label = str(meta.get("label", pk))
    for sec in sector_names:
      io_norm = _norm_key(sec)
      if cat_norm in {"monetary", "fiscal"}:
        policy_tools[sec].add(label)
      elif cat_norm == io_norm:
        policy_tools[sec].add(label)
      elif cat_norm == "markets" and io_norm in {"agriculture", "manufacturing", "services"}:
        policy_tools[sec].add(label)
      elif cat_norm == "socialprotection" and io_norm in {"services", "health", "security"}:
        policy_tools[sec].add(label)
      elif cat_norm == "communications" and io_norm in {"services", "security"}:
        policy_tools[sec].add(label)

  rows = []
  for i, sec in enumerate(sector_names):
    shocks_for_sec = sorted(shock_families.get(sec, set()))
    tools_for_sec = sorted(policy_tools.get(sec, set()))

    rows.append({
      "Sector": sec.title(),
      f"Interdependence vs {benchmark_sector.title()}": round(float(dep_vs_benchmark[i]), 3),
      f"Spillover vs {benchmark_sector.title()}": round(float(spillover_vs_benchmark[i]), 3),
      "Shock Score": round(float(shock_scores[i]), 3),
      "Shock Class": _shock_class(float(shock_scores[i])),
      "Policy Score": round(float(policy_scores[i]), 3),
      "Policy Class": _policy_class(float(policy_scores[i])),
      "Matched Shock Families": ", ".join(shocks_for_sec[:3]) if shocks_for_sec else "Macro-general",
      "Matched Policy Tools": ", ".join(tools_for_sec[:3]) if tools_for_sec else "Macro mix",
    })

  class_df = pd.DataFrame(rows).sort_values(["Shock Score", "Policy Score"], ascending=False)
  st.dataframe(class_df, use_container_width=True, hide_index=True)

  fig_cls = go.Figure()
  fig_cls.add_trace(go.Bar(
    x=class_df["Sector"],
    y=class_df["Shock Score"],
    name="Shock Score",
    marker_color=PALETTE[3],
    hovertemplate="%{x}<br>Shock score: %{y:.3f}<extra></extra>",
  ))
  fig_cls.add_trace(go.Bar(
    x=class_df["Sector"],
    y=class_df["Policy Score"],
    name="Policy Score",
    marker_color=PALETTE[0],
    hovertemplate="%{x}<br>Policy score: %{y:.3f}<extra></extra>",
  ))
  fig_cls.update_layout(**base_layout(theme, height=360,
    title=dict(
      text=f"Sector Classification Against {benchmark_sector.title()}",
      font=dict(color=theme.text_muted, size=13),
    ),
    barmode="group",
    xaxis=dict(title="Sector"),
    yaxis=dict(title="Score (0-1)", range=[0, 1.0]),
    legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
  ))
  st.plotly_chart(fig_cls, use_container_width=True)

  top_shock = str(class_df.iloc[0]["Sector"]) if not class_df.empty else "N/A"
  top_policy = str(class_df.sort_values("Policy Score", ascending=False).iloc[0]["Sector"]) if not class_df.empty else "N/A"
  st.caption(
    f"Benchmark: {benchmark_sector.title()}. Highest spillover risk sector: {top_shock}. "
    f"Highest policy leverage sector: {top_policy}."
  )


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
    st.markdown(f"""<div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">Gini <b>{gini_now:.4f}</b> {'(above Kenya benchmark 0.408 — inequality is high)' if gini_now > 0.408 else '(below Kenya benchmark — relatively equal)'}. {'Rising' if gini_vals[-1] > gini_vals[0] else 'Falling'} from {gini_vals[0]:.4f} at the start. Higher = more unequal (0 = perfect equality, 1 = maximum inequality).</div>""", unsafe_allow_html=True)

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
    st.markdown(f"""<div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">Palma ratio <b>{palma_now:.2f}</b>: the richest 10% earn {palma_now:.2f}x the income of the poorest 40% combined. {'Rising' if palma_now > palma_vals[0] else 'Falling'} from {palma_vals[0]:.2f} at the start. Values above 2 suggest significant concentration at the top.</div>""", unsafe_allow_html=True)

  try:
    _g_gini_end = float(gini_vals[-1]) if gini_vals else 0
    _g_gini_trend = "worsening" if len(gini_vals) > 1 and gini_vals[-1] > gini_vals[0] else "improving"
    _g_palma_end = float(palma_vals[-1]) if palma_vals else 0
    _g_palma_lbl = "severe" if _g_palma_end > 3.0 else "elevated" if _g_palma_end > 2.5 else "moderate"
  except Exception:
    _g_gini_end, _g_gini_trend, _g_palma_end, _g_palma_lbl = 0, "unknown", 0, "unknown"
  _guide(theme,
    method_html=(
      "The <b>Gini coefficient</b> measures overall income inequality on a 0–1 scale: 0 = perfect equality "
      "(everyone has the same income), 1 = perfect inequality (one person has all income). Kenya's current "
      "real-world Gini is approximately 0.408. "
      "The <b>Palma ratio</b> is the income share of the richest 10% divided by the income share of the "
      "poorest 40%. It is more sensitive than the Gini to what happens at the extremes of the distribution — "
      "Palma = 2.5 means the top 10% earn 2.5× what the bottom 40% earn combined."
    ),
    interp_html=(
      f"Terminal Gini: <b>{_g_gini_end:.3f}</b> (Kenya benchmark: 0.408) — trend is <b>{_g_gini_trend}</b>. "
      f"Terminal Palma: <b>{_g_palma_end:.2f}</b> — inequality level is <b>{_g_palma_lbl}</b>. "
      + ("Gini is above Kenya's benchmark — the simulated policy is worsening inequality." if _g_gini_end > 0.408
         else "Gini is below Kenya's benchmark — the simulated policy improves equality.")
    ),
    rec_html=(
      "Policies that raise the Gini above 0.408 are regressive — they widen the income gap. "
      "Target fiscal policy to reduce the Palma ratio: progressive taxation and direct transfers to the "
      "poorest 40% are the most direct levers. "
      "If Palma exceeds 3.0, social stability risks increase significantly."
    ),
  )

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
    st.caption("Each band shows one income quintile's share of total income over time. A narrowing bottom band (Q1) means the poorest are losing ground; a widening top band (Q5) means concentration at the top.")

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
          ("PRO-POOR" if is_pro_poor else "✗ PRO-RICH") + "</span>"
    fig_ppg.update_layout(**base_layout(theme, height=350,
      title=dict(text=f"Income Growth by Quintile — {ppg_label}",
            font=dict(color=theme.text_muted, size=13)),
      yaxis=dict(title="Growth (%)")))
    st.plotly_chart(fig_ppg, use_container_width=True)
    st.markdown(f"""<div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">{'Bottom quintiles are growing faster than top — pro-poor outcome.' if is_pro_poor else 'Top quintiles are growing faster — regressive outcome.'} Bottom 40% avg growth: <b>{avg_bottom:+.1f}%</b>; Top 40% avg: <b>{avg_top:+.1f}%</b>. Green bars = growing faster than the top; red bars = falling behind.</div>""", unsafe_allow_html=True)

  try:
    q1_vals_g = [f.get("quintile_incomes", {}).get("q1_bottom_20", 0) for f in ineq_frames]
    q5_vals_g = [f.get("quintile_incomes", {}).get("q5_top_20", 0) for f in ineq_frames]
    _g_q1_start, _g_q1_end = float(q1_vals_g[0]) if q1_vals_g else 1, float(q1_vals_g[-1]) if q1_vals_g else 1
    _g_q5_start, _g_q5_end = float(q5_vals_g[0]) if q5_vals_g else 1, float(q5_vals_g[-1]) if q5_vals_g else 1
    _g_q1_growth = (_g_q1_end - _g_q1_start) / max(abs(_g_q1_start), 0.001) * 100
    _g_q5_growth = (_g_q5_end - _g_q5_start) / max(abs(_g_q5_start), 0.001) * 100
    _g_verdict = "pro-poor" if _g_q1_growth > _g_q5_growth else "regressive" if _g_q5_growth > _g_q1_growth + 0.5 else "neutral"
  except Exception:
    _g_q1_growth, _g_q5_growth, _g_verdict = 0, 0, "unknown"
  _guide(theme,
    method_html=(
      "<b>Pro-poor growth</b> occurs when the income of the poorest quintile (Q1, bottom 20%) "
      "grows faster than the richest quintile (Q5, top 20%). This bar chart shows the percentage income "
      "change from simulation start to end for each quintile, allowing comparison of who gained and who lost "
      "from the simulated policy. A 'pro-poor' label means the poorest benefit most; 'regressive' means the "
      "wealthy benefit disproportionately."
    ),
    interp_html=(
      f"Q1 (bottom 20%) income change: <b>{_g_q1_growth:+.1f}%</b>. "
      f"Q5 (top 20%) income change: <b>{_g_q5_growth:+.1f}%</b>. "
      f"Verdict: <b>{_g_verdict.upper()}</b> — "
      + ("the poorest gain more than the richest from this scenario." if _g_verdict == "pro-poor"
         else "the richest gain more, widening the gap." if _g_verdict == "regressive"
         else "growth is distributed neutrally across quintiles.")
    ),
    rec_html=(
      "To shift a regressive outcome to pro-poor: increase the government spending ratio directed at transfers "
      "and public goods (healthcare, education) rather than capital subsidies. "
      "Monitor Q1/Q5 income ratio as a key performance indicator for distributional equity goals."
    ),
  )

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
    st.markdown(f"""<div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">The Lorenz curve shows how income is distributed. At quarter {lorenz_t}, Gini = <b>{gini_t:.4f}</b>. The further the blue curve bows below the dashed equality line, the more unequal the distribution. The red shaded area represents the inequality gap.</div>""", unsafe_allow_html=True)

  # ── 5. GINI TRAJECTORY LINE ──────────────────────────────────────────────
  st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:600; "
        f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
        f"GINI &amp; PALMA TIME SERIES</div>",
        unsafe_allow_html=True)

  fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
  fig_ts.add_trace(go.Scatter(
    x=t_vals, y=gini_vals, name="Gini",
    line=dict(color=PALETTE[0], width=3),
    fill='tozeroy', fillcolor='rgba(0, 255, 136, 0.1)',
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
  st.markdown(f"""<div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">Gini {'rising' if gini_vals[-1] > gini_vals[0] else 'falling'} from {gini_vals[0]:.4f} to {gini_vals[-1]:.4f}; Palma {'rising' if palma_vals[-1] > palma_vals[0] else 'falling'} from {palma_vals[0]:.2f} to {palma_vals[-1]:.2f}. Dashed red line = Kenya benchmark Gini (0.408). Upward trend in both signals widening inequality.</div>""", unsafe_allow_html=True)


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
    st.markdown(f"""<div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">NPL ratio: <b>{npl:.1%}</b> — {'critical: above 10% threshold' if npl > 0.10 else 'elevated: above 5% caution zone' if npl > 0.05 else 'healthy: below 5%'}. Non-performing loans erode bank capital and restrict credit supply to the real economy.</div>""", unsafe_allow_html=True)

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
    st.markdown(f"""<div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">Capital adequacy ratio: <b>{car:.1%}</b> — {'below CBK minimum of 14.5%: systemic risk' if car < 0.145 else 'above minimum but watch closely' if car < 0.20 else 'adequate buffer above regulatory floor'}. Higher CAR = more capacity to absorb loan losses without insolvency.</div>""", unsafe_allow_html=True)

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
    st.markdown(f"""<div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">Financial Stability Index aggregates NPL ratio, capital adequacy, and credit spread into a single score (0–100). Current: <b>{fsi * 100:.0f}/100</b>. Below 35 indicates stress; above 60 indicates resilience. Watch for rapid drops.</div>""", unsafe_allow_html=True)

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
    0, # total
    -bank.deposits, -bank.tier1_capital, -bank.tier2_capital,
    0, # total
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
  st.caption("The waterfall decomposes the banking sector's balance sheet. Green bars = assets; red bars = liabilities. The two total bars show gross assets and gross liabilities+capital — they should balance.")

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
    r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
    fig_fin_ts.add_trace(go.Scatter(
      x=t_vals, y=vals, mode='lines', showlegend=False,
      line=dict(color=color, width=2.5),
      fill='tozeroy', fillcolor=f'rgba({r}, {g}, {b}, 0.08)',
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
  st.caption("Time series of key financial ratios. Rising NPL combined with falling CAR is an early warning of a credit crunch. Watch for sustained divergence between the two panels.")


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
    st.markdown(f"""<div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">Real Effective Exchange Rate (REER): <b>{reer_now:.1f}</b>. Above 100 = currency overvalued vs trading partners (exports become less competitive). Below 100 = undervalued (cheaper exports, pricier imports).</div>""", unsafe_allow_html=True)

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
    st.markdown(f"""<div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">Foreign exchange reserves as months of import cover. Current: <b>{reserves_now:.1f} months</b>. Below 3 months is critical; 3–4 months is the CBK caution zone; above 4 months provides a comfortable buffer against external shocks.</div>""", unsafe_allow_html=True)

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
    st.markdown(f"""<div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">External vulnerability index aggregates exchange rate pressure, reserves adequacy, and current account balance into a single risk score (0–100). Current score: <b>{ext_vuln * 100:.0f}/100</b> — {'high exposure to external shocks' if ext_vuln > 0.6 else 'moderate exposure, monitor closely' if ext_vuln > 0.35 else 'resilient external position'}.</div>""", unsafe_allow_html=True)

  # ── 2. BALANCE OF PAYMENTS WATERFALL ─────────────────────────────────────
  st.markdown(f"<div style='color:{theme.accent_warning}; font-weight:600; "
        f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
        f"BALANCE OF PAYMENTS DECOMPOSITION</div>",
        unsafe_allow_html=True)

  tb = last_ext.get("trade_balance", 0)
  remit = last_ext.get("remittances", 0)
  ca = last_ext.get("current_account", 0)
  ka = last_ext.get("capital_account", 0)
  inv_income = ca - tb - remit # residual

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
  st.caption("Balance of Payments waterfall: green bars are inflows (exports, remittances, capital); red bars are outflows (imports, debt service). The final bar = net BoP — negative signals a current account deficit.")

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
    st.caption("Twin deficit diagram: fiscal deficit (X-axis) vs current account deficit (Y-axis). Points in the lower-left quadrant have both deficits simultaneously — a classic vulnerability signal for emerging economies.")

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
    fill='tozeroy', fillcolor=_hex_rgba(PALETTE[1], 0.08),
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
  st.caption("REER and inflation often move together. Rising inflation erodes competitiveness (pushes REER up). Divergence between the two may signal exchange rate misalignment.")

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
  st.markdown(f"""<div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">Reserve trajectory shows whether the buffer is building or eroding over time. Current cover: <b>{res_vals[-1]:.1f} months</b>. A declining trend toward the CBK minimum of 4 months warrants immediate policy attention.</div>""", unsafe_allow_html=True)


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
  st.markdown(f"""<div style="font-size:0.78rem; color:{theme.text_muted}; padding:0.3rem 0;">The macro health radar shows 6 dimensions simultaneously. A larger filled area = healthier economy overall. Inward spikes identify the weakest dimensions. The dotted ring marks the mid-point threshold (50/100) — any dimension below it warrants attention.</div>""", unsafe_allow_html=True)

  try:
    _g_weakest_dim_idx = int(np.argmin(radar_vals))
    _g_weakest = radar_cats[_g_weakest_dim_idx].replace("\n", " ")
    _g_weakest_val = radar_vals[_g_weakest_dim_idx]
    _g_strongest_idx = int(np.argmax(radar_vals))
    _g_strongest = radar_cats[_g_strongest_idx].replace("\n", " ")
    _g_overall = float(np.mean(radar_vals))
    _g_below_thresh = sum(1 for v in radar_vals if v < 50)
  except Exception:
    _g_weakest, _g_weakest_val, _g_strongest, _g_overall, _g_below_thresh = "?", 0, "?", 0, 0
  _guide(theme,
    method_html=(
      "The <b>Multi-Module Health Radar</b> is a 6-dimensional spider/polar chart that simultaneously shows "
      "the health of each major economic subsystem after the simulation run. Each axis is normalised to 0–100 "
      "where 100 = ideal and 0 = worst case. The six dimensions are: "
      "<b>Financial Stability</b> (bank solvency and credit health), "
      "<b>External Resilience</b> (reserve adequacy and REER stability), "
      "<b>Macro Balance</b> (inflation near target), "
      "<b>Growth Momentum</b> (GDP expansion), "
      "<b>Equality</b> (low Gini), and "
      "<b>Sector Diversity</b> (no single sector dominance). "
      "The dotted ring at 50 is the alert threshold — inward spikes below it identify vulnerabilities."
    ),
    interp_html=(
      f"Overall economy health: <b>{_g_overall:.0f}/100</b>. "
      f"<b>{_g_below_thresh}</b> dimension(s) below alert threshold (50). "
      f"Weakest dimension: <b>{_g_weakest}</b> at <b>{_g_weakest_val:.0f}/100</b>. "
      f"Strongest dimension: <b>{_g_strongest}</b>. "
      + ("Economy is broadly healthy — all dimensions above alert threshold." if _g_below_thresh == 0
         else f"Priority concern: {_g_weakest} is critically weak and needs targeted policy attention.")
    ),
    rec_html=(
      f"Focus policy on <b>{_g_weakest}</b> — it is the binding constraint limiting overall health. "
      "Improving the weakest dimension typically yields the highest marginal gain to overall resilience. "
      "Use the specific module tabs (Financial, Open Economy, Inequality) to drill into the weakest dimensions."
    ),
  )

  # ── 2. GDP DECOMPOSITION WATERFALL ─────────────────────────────────────
  st.markdown(f"<div style='color:{theme.accent_warning}; font-weight:600; "
        f"font-size:0.85rem; margin-top:1rem; margin-bottom:0.5rem;'>"
        f"GDP DEMAND DECOMPOSITION: Y = C + I + G + NX</div>",
        unsafe_allow_html=True)

  gdp = summary.get("gdp", 100)
  # Estimate components from engine state
  # Handle both bare SFCEconomy and wrapped ResearchSFCEconomy
  base_econ = getattr(research_econ, "economy", research_econ)
  
  hh_consumption = float(base_econ.households.assets.get("deposits", 0)) * 0.3
  if not hh_consumption:
    hh_consumption = gdp * 0.60
    
  enable_fin = getattr(research_econ.config, "enable_financial", False)
  investment = float(research_econ.bank.performing_loans * 0.08) if enable_fin and hasattr(research_econ, "bank") else gdp * 0.18
  
  gov_spending = gdp * base_econ.config.spending_ratio
  
  enable_nx = getattr(research_econ.config, "enable_open_economy", False)
  net_exports = float(research_econ.external.trade_balance) if enable_nx and hasattr(research_econ, "external") else 0.0
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
  st.caption("GDP decomposition waterfall: positive bars (green) = components adding to growth; negative bars (red) = drags on output. The final bar is net GDP.")

  try:
    _g_gdp_val = float(gdp)
    _g_c_pct = hh_consumption / _g_gdp_val * 100
    _g_i_pct = investment / _g_gdp_val * 100
    _g_g_pct = gov_spending / _g_gdp_val * 100
    _g_nx_pct = net_exports / _g_gdp_val * 100
    _g_nx_label = "surplus" if net_exports > 0 else "deficit"
    _g_dominant = "Consumption" if _g_c_pct >= _g_i_pct and _g_c_pct >= _g_g_pct else "Investment" if _g_i_pct >= _g_g_pct else "Government"
  except Exception:
    _g_gdp_val, _g_c_pct, _g_i_pct, _g_g_pct, _g_nx_pct, _g_nx_label, _g_dominant = 0, 0, 0, 0, 0, "unknown", "?"
  _guide(theme,
    method_html=(
      "The <b>GDP Demand-Side Decomposition</b> breaks total output (Y) into its expenditure components: "
      "<b>C</b> (household consumption), <b>I</b> (private investment, driven by credit), "
      "<b>G</b> (government spending), and <b>NX</b> (net exports = exports − imports). "
      "This follows the national income identity: Y ≡ C + I + G + NX. "
      "Green waterfall bars add to GDP; red bars are drags. The final bar is total GDP."
    ),
    interp_html=(
      f"GDP = <b>{_g_gdp_val:.1f}</b>. "
      f"Consumption: {_g_c_pct:.1f}%, Investment: {_g_i_pct:.1f}%, "
      f"Government: {_g_g_pct:.1f}%, Net Exports: {_g_nx_pct:+.1f}% ({_g_nx_label}). "
      f"<b>{_g_dominant}</b> is the dominant demand driver. "
      + ("Trade deficit is a drag on GDP — imports exceed exports." if net_exports < 0 else "")
    ),
    rec_html=(
      f"To boost GDP, the most efficient lever is the largest component: {_g_dominant}. "
      "If NX is negative (trade deficit), policies to boost exports or reduce import dependency "
      "will improve GDP. "
      "Monitor the I/GDP ratio — investment below 15% signals under-capitalisation and slowing future growth."
    ),
  )

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
      (0, 1, rate * 10, PALETTE[1]),     # Rate → Bank
      (0, 3, npl * 10, PALETTE[3]),      # Rate → NPL
      (1, 2, (1 - spread) * 5, PALETTE[0]),  # Bank → Credit
      (3, 2, npl * 5, PALETTE[3]),       # NPL → Credit (drag)
    ]
    # Credit → Sectors (by share)
    sector_shares = summary.get("sectors", {"agriculture": 0.22, "manufacturing": 0.08,
                         "services": 0.53, "mining": 0.04, "construction": 0.13})
    for i, (name, share) in enumerate(sector_shares.items()):
      palette_idx = (4 + i) % len(PALETTE)
      flow_data.append((2, 4 + i, share * 10, PALETTE[palette_idx]))
    # Sectors → Households (dynamic quintile weights from inequality module)
    _q_incomes = (summary.get("inequality") or {}).get("quintile_incomes") or {}
    _q1_w = max(0.01, float(_q_incomes.get("q1_bottom_20", 0.4)))
    _q3_w = max(0.01, float(_q_incomes.get("q3_middle_20", 0.8)))
    _q5_w = max(0.01, float(_q_incomes.get("q5_top_20", 1.2)))
    _hh_total = _q1_w + _q3_w + _q5_w
    _q1_norm = _q1_w / _hh_total * 3
    _q3_norm = _q3_w / _hh_total * 3
    _q5_norm = _q5_w / _hh_total * 3
    for i in range(5):
      flow_data.append((4 + i, 9, _q1_norm, 'rgba(255,51,102,0.3)'))  # → Q1
      flow_data.append((4 + i, 10, _q3_norm, 'rgba(245,213,71,0.3)')) # → Q3
      flow_data.append((4 + i, 11, _q5_norm, 'rgba(0,255,136,0.3)'))  # → Q5
    # Households → GDP (proportional to quintile income share)
    for hh_idx, hh_wt in [(9, _q1_norm * 2), (10, _q3_norm * 2), (11, _q5_norm * 2)]:
      flow_data.append((hh_idx, 12, hh_wt, PALETTE[1]))

    for s, t, v, c in flow_data:
      src.append(s); tgt.append(t); vals.append(max(v, 0.01))
      if c.startswith('#'):
        link_colors.append(f"rgba({int(c[1:3], 16)},{int(c[3:5], 16)},{int(c[5:7], 16)},0.4)")
      else:
        link_colors.append(c)

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
    st.caption("This Sankey traces how a central bank rate change transmits through the economy: Rate → Bank capital → Credit supply → Sector investment → Household income → GDP. Wider ribbons = stronger transmission.")

    try:
      _g_rate = float(research_econ.economy.interest_rate) * 100
      _g_npl_v = float(npl) * 100
      _g_spread_v = float(spread) * 100
      _g_q1_share = _q1_norm / (_q1_norm + _q3_norm + _q5_norm) * 100
      _g_q5_share = _q5_norm / (_q1_norm + _q3_norm + _q5_norm) * 100
    except Exception:
      _g_rate, _g_npl_v, _g_spread_v, _g_q1_share, _g_q5_share = 0, 0, 0, 0, 0
    _guide(theme,
      method_html=(
        "The <b>Policy Transmission Sankey</b> traces how a central bank interest rate decision propagates "
        "through the economy. The flow goes: Policy Rate → Bank Capital (affected by rate cost) → "
        "Credit Supply (bank lending capacity, reduced by NPL drag) → Sectors (credit allocated by sector share) → "
        "Household quintiles (income distributed from sector output) → GDP. "
        "Ribbon widths are proportional to flow magnitude — a thick ribbon between Policy Rate and NPL Drag "
        "means credit quality is severely impaired. Household flows are dynamic: they use actual quintile "
        "income data from the heterogeneous agents module."
      ),
      interp_html=(
        f"Policy rate: <b>{_g_rate:.2f}%</b>. NPL ratio: <b>{_g_npl_v:.1f}%</b>. "
        f"Credit spread: <b>{_g_spread_v:.2f}%</b>. "
        f"Q1 (poorest) receives <b>{_g_q1_share:.1f}%</b> of household income flow; "
        f"Q5 (richest) receives <b>{_g_q5_share:.1f}%</b>. "
        + ("High NPL is choking credit supply — rate cuts will have limited pass-through." if _g_npl_v > 10
           else "Credit quality is healthy — rate changes will transmit efficiently to the real economy.")
      ),
      rec_html=(
        "If the NPL ribbon is thick, rate cuts alone won't stimulate investment — address credit quality first. "
        "If Q5 captures a disproportionately large share of income flow, complement monetary policy with "
        "targeted transfers to Q1. "
        "The widest sector ribbon shows which industry receives the most credit — ensure this aligns with development priorities."
      ),
    )

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
        fill='tozeroy', fillcolor=color.replace(')', ',0.08)').replace('rgb', 'rgba') if color.startswith('rgb') else _hex_rgba(color, 0.08),
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
    st.caption("Six macro indicators at a glance. Look for synchronised downturns across multiple panels — that signals a broad recession. Upward trajectories in all four indicate a healthy expansion.")

    try:
      _g_gdp_growth_end = float(trajectory[-1].get("outcomes", {}).get("gdp_growth", 0)) * 100
      _g_inflation_end = float(trajectory[-1].get("outcomes", {}).get("inflation", 0)) * 100
      _g_unemp_end = float(trajectory[-1].get("outcomes", {}).get("unemployment", 0)) * 100
      _g_rate_end = float(trajectory[-1].get("outcomes", {}).get("interest_rate", 0)) * 100
      _g_gdp_trend_vals = [f.get("outcomes", {}).get("gdp_growth", 0) * 100 for f in trajectory]
      _g_expanding = _g_gdp_growth_end > 0
      _g_gdp_accel = len(_g_gdp_trend_vals) > 4 and _g_gdp_trend_vals[-1] > _g_gdp_trend_vals[-4]
    except Exception:
      _g_gdp_growth_end, _g_inflation_end, _g_unemp_end, _g_rate_end, _g_expanding, _g_gdp_accel = 0, 0, 0, 0, False, False
    _guide(theme,
      method_html=(
        "The <b>Macro Trajectory Sparklines</b> show four key macroeconomic indicators over the full simulation "
        "horizon as compact time-series: GDP Growth %, Inflation %, Unemployment %, and Interest Rate %. "
        "These are the 'vitals' of the simulated economy — monitoring all four simultaneously reveals whether "
        "the economy is in expansion, stagflation, deflation, or a healthy Goldilocks zone. "
        "Each panel shows the terminal (final quarter) value as an annotation."
      ),
      interp_html=(
        f"Terminal values — GDP growth: <b>{_g_gdp_growth_end:+.1f}%</b>, "
        f"Inflation: <b>{_g_inflation_end:.1f}%</b>, "
        f"Unemployment: <b>{_g_unemp_end:.1f}%</b>, "
        f"Interest rate: <b>{_g_rate_end:.1f}%</b>. "
        f"Economy is {'<b>expanding</b>' if _g_expanding else '<b>contracting</b>'}. "
        f"GDP growth is {'<b>accelerating</b>' if _g_gdp_accel else '<b>decelerating</b>'} over the last 4 quarters. "
        + ("Inflation above 8% is concerning; consider tightening." if _g_inflation_end > 8 else "")
      ),
      rec_html=(
        "Look for synchronised downturns across all four panels — that signals a broad recession. "
        "Healthy targets: GDP growth 5–7%, inflation 5–6% (Kenya target), unemployment below 10%, "
        "real interest rate positive (rate > inflation). "
        "If inflation and unemployment are both rising simultaneously, the economy is in stagflation — "
        "require structural supply-side reform, not just monetary policy."
      ),
    )

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
      st.caption("Pre- and post-shock comparison. Red bars show deterioration after the stress event; green bars show resilience. Taller red bars identify the economy's most vulnerable pressure points.")

      try:
        _g_car_pre = float(fin_stress.get("pre_car", 0)) * 100 if fin_stress else 0
        _g_car_post = float(fin_stress.get("post_car", 0)) * 100 if fin_stress else 0
        _g_car_drop = _g_car_pre - _g_car_post
        _g_car_ok = _g_car_post >= 14.5
        _g_q1_impact = float(dist_stress.get("rate_shock_q1_impact", 0)) * 100 if dist_stress else 0
        _g_q5_impact = float(dist_stress.get("rate_shock_q5_impact", 0)) * 100 if dist_stress else 0
        _g_regressive_stress = _g_q5_impact > _g_q1_impact
      except Exception:
        _g_car_pre, _g_car_post, _g_car_drop, _g_car_ok, _g_q1_impact, _g_q5_impact, _g_regressive_stress = 0, 0, 0, True, 0, 0, False
      _guide(theme,
        method_html=(
          "The <b>Combined Stress Test</b> applies simultaneous shocks to the economy and measures the impact "
          "on key financial and distributional indicators: "
          "<b>NPL shock</b> = sudden rise in non-performing loans (credit quality deterioration); "
          "<b>Rate shock</b> = sudden interest rate hike (monetary tightening); "
          "<b>FX shock</b> = currency depreciation (import cost inflation); "
          "<b>Deposit run</b> = sudden withdrawal of bank deposits (liquidity crisis). "
          "This is a standard bank stress test methodology (Basel III DFAST framework). "
          "The CAR (Capital Adequacy Ratio) is the primary resilience indicator — CBK minimum is 14.5%."
        ),
        interp_html=(
          f"Post-shock CAR: <b>{_g_car_post:.1f}%</b> (pre: {_g_car_pre:.1f}%, drop: {_g_car_drop:.1f}pp). "
          f"CBK minimum: 14.5% — <b>{'PASSES' if _g_car_ok else 'FAILS'}</b>. "
          f"Distributional impact: Q1 change <b>{_g_q1_impact:+.1f}%</b>, Q5 change <b>{_g_q5_impact:+.1f}%</b>. "
          + ("Stress is <b>regressive</b> — the shock hits the poorest harder." if _g_regressive_stress else "Shock impact is broadly neutral across income groups.")
        ),
        rec_html=(
          f"{'The bank system is resilient to this stress scenario.' if _g_car_ok else f'CRITICAL: Post-shock CAR ({_g_car_post:.1f}%) breaches CBK minimum (14.5%). Recapitalisation required.'} "
          "Test with progressively larger shocks to find the failure threshold. "
          "If stress is regressive, pre-position social safety nets for Q1 households before any policy tightening."
        ),
      )

  # Hint to check other tabs
  st.markdown(f"<div style='font-size:0.75rem; color:{theme.text_muted}; "
        f"margin-top:1rem; text-align:center;'>"
        f"&#8594; See <b>IO Sectors</b>, <b>Inequality</b>, <b>Financial</b>, "
        f"and <b>Open Economy</b> tabs for detailed charts.</div>",
        unsafe_allow_html=True)
