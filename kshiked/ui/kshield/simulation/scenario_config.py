"""Scenario Configuration Panel."""

from ._shared import st, pd, np, go, HAS_DATA_STACK, HAS_PLOTLY, PALETTE, base_layout


def render_scenario_config(theme, scenario_library, policy_templates,
              get_scenario_by_id, build_custom_scenario,
              outcome_dimensions, default_dimensions,
              shock_registry=None, policy_instrument_registry=None,
              shock_shapes=None):
  """
  Model Configuration expander — multi-shock & multi-policy builder.

  Returns a dict with:
    selected_scenarios, custom_shocks, selected_policy_keys,
    custom_instruments, selected_dims, steps
  """
  if shock_registry is None:
    shock_registry = {}
  if policy_instrument_registry is None:
    policy_instrument_registry = {}
  if shock_shapes is None:
    shock_shapes = ["step", "pulse", "ramp", "decay"]

  with st.expander("Scenario Configuration", expanded=True):

    # ─── SECTION 1: PRESET SCENARIO SHOCKS ────────────────────────────
    st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:700; "
          f"font-size:0.85rem; margin-bottom:0.3rem;'>"
          f"SHOCK EVENTS</div>", unsafe_allow_html=True)
    st.caption("Select multiple preset scenarios and/or add custom shocks. "
          "All shocks stack additively — just like real economies.")

    # Preset multi-select
    scenario_options = {s.id: f"{s.name} ({s.category})" for s in scenario_library}
    selected_scenario_ids = st.multiselect(
      "Preset Scenarios (stack multiple)",
      options=list(scenario_options.keys()),
      format_func=lambda x: scenario_options[x],
      key="sim_preset_scenarios",
      help="Each selected scenario adds its shocks on top of the others",
    )
    selected_scenarios = [get_scenario_by_id(sid) for sid in selected_scenario_ids
               if get_scenario_by_id(sid) is not None]

    # Show combined context + EDITABLE shock magnitudes for each preset
    if selected_scenarios:
      contexts = [s.context for s in selected_scenarios if s.context]
      if contexts:
        st.markdown(f"""
        <div style="background: rgba(0,243,255,0.05); border-left: 3px solid {theme.accent_primary};
              padding: 0.5rem 0.8rem; margin: 0.3rem 0; border-radius: 0 8px 8px 0;
              font-size: 0.78rem; color: {theme.text_muted};">
          <b>Combined scenario context:</b><br>
          {'<br>• '.join(contexts)}
        </div>
        """, unsafe_allow_html=True)

      # Editable sliders for each scenario's shocks
      st.markdown(f"<div style='color:{theme.text_muted}; font-weight:600; "
            f"font-size:0.78rem; margin-top:0.5rem; margin-bottom:0.3rem;'>"
            f"ADJUST PRESET MAGNITUDES</div>", unsafe_allow_html=True)

      # Initialize override storage
      if "sim_preset_overrides" not in st.session_state:
        st.session_state["sim_preset_overrides"] = {}
      preset_overrides = st.session_state["sim_preset_overrides"]

      for s in selected_scenarios:
        if s.id not in preset_overrides:
          preset_overrides[s.id] = dict(s.shocks)
        st.markdown(f"<div style='font-size:0.75rem; color:{theme.accent_primary}; "
              f"font-weight:600; margin-top:0.3rem;'>{s.name}</div>",
              unsafe_allow_html=True)
        shock_keys = list(s.shocks.keys())
        cols = st.columns(len(shock_keys)) if shock_keys else []
        for idx, sk in enumerate(shock_keys):
          reg = shock_registry.get(sk, {})
          default_val = s.shocks[sk]
          with cols[idx]:
            new_val = st.slider(
              reg.get("label", sk),
              min_value=float(reg.get("min", -0.30)),
              max_value=float(reg.get("max", 0.30)),
              value=float(preset_overrides[s.id].get(sk, default_val)),
              step=float(reg.get("step", 0.01)),
              key=f"preset_{s.id}_{sk}",
              help=f"Default: {default_val:+.2f}. {reg.get('examples', '')}",
            )
            preset_overrides[s.id][sk] = new_val

      st.session_state["sim_preset_overrides"] = preset_overrides

      # Clean up overrides for deselected scenarios
      for old_id in list(preset_overrides.keys()):
        if old_id not in selected_scenario_ids:
          del preset_overrides[old_id]

      # Show combined shock summary after user edits
      shock_summary = {}
      for s in selected_scenarios:
        overrides = preset_overrides.get(s.id, s.shocks)
        for k, v in overrides.items():
          shock_summary[k] = shock_summary.get(k, 0) + v
      if shock_summary:
        parts = []
        for k, v in shock_summary.items():
          if isinstance(v, dict):
            # Safely handle the bizarre case where v is a dict
            val_str = str(v)
          else:
            val_str = f"{v:+.2f}"
          
          label = str(k)
          if hasattr(shock_registry, "get"):
            reg_entry = shock_registry.get(k)
            if isinstance(reg_entry, dict):
              label = reg_entry.get("label", str(k))
          
          parts.append(f"{label}: <b>{val_str}</b>")
        
        st.markdown(f"<div style='font-size:0.75rem; color:{theme.text_muted};'>"
              f"Combined shocks: {' | '.join(parts)}</div>",
              unsafe_allow_html=True)

    # ─── CUSTOM SHOCKS ────────────────────────────────────────────────
    st.markdown(f"<div style='color:{theme.text_muted}; font-weight:600; "
          f"font-size:0.78rem; margin-top:0.6rem;'>CUSTOM SHOCKS</div>",
          unsafe_allow_html=True)

    # Initialize custom shocks list in session state
    if "sim_custom_shocks" not in st.session_state:
      st.session_state["sim_custom_shocks"] = []

    custom_shocks = list(st.session_state["sim_custom_shocks"])

    # Render existing custom shocks
    shocks_to_remove = []
    
    # ── Grouping logic for Shocks ─────────────────────────────────────
    shock_type_keys = []
    shock_type_labels = {}
    if shock_registry:
      # Group by sector if it exists, otherwise "Macro"
      grouped_shocks = {}
      for k, v in shock_registry.items():
        sector = v.get("sector", "Macro")
        grouped_shocks.setdefault(sector, []).append((k, v.get("label", k)))
      
      # Flatten alphabetically by sector
      for sector in sorted(grouped_shocks.keys()):
        for k, label in sorted(grouped_shocks[sector], key=lambda x: x[1]):
          shock_type_keys.append(k)
          shock_type_labels[k] = f"[{sector}] {label}"
    else:
      shock_type_keys = ["demand_shock", "supply_shock", "fiscal_shock", "fx_shock"]
      shock_type_labels = {k: k for k in shock_type_keys}

    for i, cs in enumerate(custom_shocks):
      cols = st.columns([2, 2, 1.5, 1.5, 1.5, 0.5])
      with cols[0]:
        cs_key = cs.get("key", "demand_shock")
        cs["key"] = st.selectbox(
          "Type", shock_type_keys,
          index=shock_type_keys.index(cs_key) if cs_key in shock_type_keys else 0,
          format_func=lambda k, _labels=shock_type_labels: _labels.get(k, k),
          key=f"cs_type_{i}",
        )
      reg = shock_registry.get(cs["key"], {})
      with cols[1]:
        cs["magnitude"] = st.slider(
          "Magnitude", float(reg.get("min", -0.30)), float(reg.get("max", 0.30)),
          float(cs.get("magnitude", 0.0)), float(reg.get("step", 0.01)),
          key=f"cs_mag_{i}",
          help=reg.get("examples", ""),
        )
      with cols[2]:
        cs["onset"] = st.number_input(
          "Onset Q", 1, 100, int(cs.get("onset", 5)), key=f"cs_onset_{i}",
        )
      with cols[3]:
        cs["duration"] = st.number_input(
          "Duration", 0, 50, int(cs.get("duration", 0)), key=f"cs_dur_{i}",
          help="0 = permanent",
        )
      with cols[4]:
        cs_shape = cs.get("shape", "step")
        cs["shape"] = st.selectbox(
          "Shape", shock_shapes,
          index=shock_shapes.index(cs_shape) if cs_shape in shock_shapes else 0,
          key=f"cs_shape_{i}",
        )
      with cols[5]:
        st.markdown("<div style='padding-top:1.6rem;'></div>", unsafe_allow_html=True)
        if st.button("X", key=f"cs_rm_{i}", help="Remove this shock"):
          shocks_to_remove.append(i)

    # Remove marked shocks
    if shocks_to_remove:
      for idx in sorted(shocks_to_remove, reverse=True):
        if idx < len(custom_shocks):
          custom_shocks.pop(idx)
      st.session_state["sim_custom_shocks"] = custom_shocks
      st.rerun()

    # Add shock button
    if st.button("+ Add Custom Shock", key="sim_add_shock"):
      custom_shocks.append({
        "key": "demand_shock", "magnitude": 0.0,
        "onset": 5, "duration": 0, "shape": "step",
      })
      st.session_state["sim_custom_shocks"] = custom_shocks
      st.rerun()

    # Update session state
    st.session_state["sim_custom_shocks"] = custom_shocks

    # ─── SHOCK OUTLINE (DETAILED) ────────────────────────────────────
    st.markdown(f"<div style='color:{theme.text_muted}; font-weight:600; "
          f"font-size:0.78rem; margin-top:0.65rem;'>SHOCK OUTLINE (DETAILED)</div>",
          unsafe_allow_html=True)

    if HAS_DATA_STACK and pd is not None:
      preset_overrides = st.session_state.get("sim_preset_overrides", {})
      shock_rows = []

      for s in selected_scenarios:
        overrides = preset_overrides.get(s.id, s.shocks)
        for k, v in overrides.items():
          reg = shock_registry.get(k, {}) if hasattr(shock_registry, "get") else {}
          sfc_map = reg.get("sfc_mapping", {}) if isinstance(reg, dict) else {}
          map_txt = ", ".join(
            f"{ch}:{float(w):.2f}" for ch, w in sfc_map.items()
          ) if sfc_map else "direct"
          shock_rows.append({
            "Shock": reg.get("label", k) if isinstance(reg, dict) else str(k),
            "Source": f"Preset: {s.name}",
            "Sector": reg.get("sector", "Macro") if isinstance(reg, dict) else "Macro",
            "Magnitude": round(float(v), 4),
            "Onset": f"Q{int(getattr(s, 'shock_onset', 5))}",
            "Duration": int(getattr(s, "shock_duration", 0) or 0),
            "Shape": str(getattr(s, "shock_shape", "step")),
            "SFC Channels": map_txt,
            "Description": reg.get("description", "") if isinstance(reg, dict) else "",
          })

      for idx_cs, cs in enumerate(custom_shocks, start=1):
        k = str(cs.get("key", "")).strip()
        if not k:
          continue
        reg = shock_registry.get(k, {}) if hasattr(shock_registry, "get") else {}
        sfc_map = reg.get("sfc_mapping", {}) if isinstance(reg, dict) else {}
        map_txt = ", ".join(
          f"{ch}:{float(w):.2f}" for ch, w in sfc_map.items()
        ) if sfc_map else "direct"
        shock_rows.append({
          "Shock": reg.get("label", k) if isinstance(reg, dict) else str(k),
          "Source": f"Custom #{idx_cs}",
          "Sector": reg.get("sector", "Macro") if isinstance(reg, dict) else "Macro",
          "Magnitude": round(float(cs.get("magnitude", 0.0) or 0.0), 4),
          "Onset": f"Q{int(cs.get('onset', 5) or 5)}",
          "Duration": int(cs.get("duration", 0) or 0),
          "Shape": str(cs.get("shape", "step") or "step"),
          "SFC Channels": map_txt,
          "Description": reg.get("description", "") if isinstance(reg, dict) else "",
        })

      if shock_rows:
        shock_df = pd.DataFrame(shock_rows)
        st.dataframe(shock_df, use_container_width=True, hide_index=True)

        summary_df = (
          shock_df.groupby("Shock", dropna=False, as_index=False)
          .agg(
            Net_Magnitude=("Magnitude", "sum"),
            Sources=("Source", "count"),
          )
          .sort_values("Net_Magnitude", key=lambda s: s.abs(), ascending=False)
        )
        summary_df = summary_df.rename(columns={
          "Net_Magnitude": "Net Magnitude",
          "Sources": "Number of Sources",
        })
        st.caption("Shock roll-up: net magnitude and how many sources (preset/custom) contribute to each shock.")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
      else:
        st.info("No shocks configured yet. Add a preset scenario or custom shock to see a detailed outline.")
    else:
      st.info("Install pandas to display the detailed shock outline table.")

    # ─── SECTION 2: POLICY RESPONSES ──────────────────────────────────
    st.markdown("---")
    st.markdown(f"<div style='color:{theme.accent_warning}; font-weight:700; "
          f"font-size:0.85rem; margin-bottom:0.3rem;'>"
          f"POLICY RESPONSES</div>", unsafe_allow_html=True)
    st.caption("Select multiple preset policies and/or add custom instruments. "
          "Later selections override earlier ones for the same instrument.")

    # Preset policy multi-select
    policy_keys_available = [k for k in policy_templates.keys() if k != "do_nothing"]
    selected_policy_keys = st.multiselect(
      "Preset Policies (layer multiple)",
      options=policy_keys_available,
      format_func=lambda k: policy_templates[k]["name"],
      key="sim_preset_policies",
      help="Each policy layers its instruments. Later policies override earlier ones.",
    )

    # Show combined policy instruments
    if selected_policy_keys:
      combined_instruments = {}
      for pk in selected_policy_keys:
        tmpl = policy_templates.get(pk, {})
        for k, v in tmpl.get("instruments", {}).items():
          combined_instruments[k] = v
      if combined_instruments:
        parts = []
        for k, v in combined_instruments.items():
          reg = policy_instrument_registry.get(k, {})
          label = reg.get("label", k)
          scale = reg.get("display_scale", 1)
          if isinstance(v, (int, float)):
            parts.append(f"{label}: <b>{v * scale:.1f}%</b>")
          else:
            parts.append(f"{label}: <b>{v}</b>")
        st.markdown(f"<div style='font-size:0.75rem; color:{theme.text_muted};'>"
              f"Combined policy instruments: {' | '.join(parts)}</div>",
              unsafe_allow_html=True)

    # ─── CUSTOM POLICY INSTRUMENTS ────────────────────────────────────
    st.markdown(f"<div style='color:{theme.text_muted}; font-weight:600; "
          f"font-size:0.78rem; margin-top:0.6rem;'>CUSTOM INSTRUMENTS</div>",
          unsafe_allow_html=True)

    if "sim_custom_instruments" not in st.session_state:
      st.session_state["sim_custom_instruments"] = []

    custom_instruments = list(st.session_state["sim_custom_instruments"])

    # ── Grouping logic for Policy Instruments ─────────────────────────
    inst_type_keys = []
    inst_type_labels = {}
    if policy_instrument_registry:
      grouped_insts = {}
      for k, v in policy_instrument_registry.items():
        cat = v.get("category", "General")
        grouped_insts.setdefault(cat, []).append((k, v.get("label", k)))
        
      for cat in sorted(grouped_insts.keys()):
        for k, label in sorted(grouped_insts[cat], key=lambda x: x[1]):
          inst_type_keys.append(k)
          inst_type_labels[k] = f"[{cat}] {label}"

    insts_to_remove = []
    for i, ci in enumerate(custom_instruments):
      cols = st.columns([3, 4, 0.5])
      with cols[0]:
        ci_key = ci.get("key", "custom_rate")
        ci["key"] = st.selectbox(
          "Instrument", inst_type_keys,
          index=inst_type_keys.index(ci_key) if ci_key in inst_type_keys else 0,
          format_func=lambda k, _labels=inst_type_labels: _labels.get(k, k),
          key=f"ci_type_{i}",
        )
      reg = policy_instrument_registry.get(ci["key"], {})
      scale = reg.get("display_scale", 1)
      with cols[1]:
        raw_val = st.slider(
          f"{reg.get('label', ci['key'])} ({reg.get('unit', '')})",
          float(reg.get("min", 0) * scale),
          float(reg.get("max", 0.25) * scale),
          float(ci.get("value", reg.get("default", 0.07)) * scale),
          float(reg.get("step", 0.01) * scale),
          key=f"ci_val_{i}",
          help=reg.get("description", ""),
        )
        ci["value"] = raw_val / scale if scale != 0 else raw_val
      with cols[2]:
        st.markdown("<div style='padding-top:1.6rem;'></div>", unsafe_allow_html=True)
        if st.button("X", key=f"ci_rm_{i}", help="Remove this instrument"):
          insts_to_remove.append(i)

    if insts_to_remove:
      for idx in sorted(insts_to_remove, reverse=True):
        if idx < len(custom_instruments):
          custom_instruments.pop(idx)
      st.session_state["sim_custom_instruments"] = custom_instruments
      st.rerun()

    if inst_type_keys and st.button("+ Add Custom Instrument", key="sim_add_instrument"):
      custom_instruments.append({
        "key": "custom_rate", "value": 0.07,
      })
      st.session_state["sim_custom_instruments"] = custom_instruments
      st.rerun()

    # Update session state
    st.session_state["sim_custom_instruments"] = custom_instruments

    # ── No-policy fallback
    if not selected_policy_keys and not custom_instruments:
      st.info("No policy selected — simulation will run with **no policy intervention** "
          "(baseline / do-nothing mode).")

    # ─── SECTION 2B: SECTOR RISK PREVIEW ────────────────────────────
    st.markdown("---")
    st.markdown(f"<div style='color:{theme.accent_primary}; font-weight:700; "
          f"font-size:0.85rem; margin-bottom:0.3rem;'>"
          f"SECTOR RISK PREVIEW</div>", unsafe_allow_html=True)

    _srp_tab_preview, _srp_tab_method = st.tabs(["Preview", "Data & Methodology"])

    with _srp_tab_method:
      st.markdown(f"""
<div style="font-size:0.85rem; line-height:1.75; color:{theme.text_secondary};">

<b style="font-size:0.9rem;">Where does the data come from?</b><br>
The sector risk scores are built from two sources:
<ol>
  <li><b>Kenya Input-Output (IO) Table</b> — a matrix that records how much each sector
  of the economy buys from and sells to every other sector. For example, it captures how
  much agriculture relies on manufacturing for inputs such as fertilizer, and how much
  manufacturing relies on agriculture for raw materials. Kenya's IO table is derived from
  national accounts data published by the Kenya National Bureau of Statistics (KNBS) and
  the World Bank's EORA global supply chain database.</li>
  <li><b>Your selected shocks and policies</b> — the magnitudes you set above are combined
  with the IO table to calculate how much pressure each sector is under and how well your
  chosen policy instruments cover it.</li>
</ol>

<b style="font-size:0.9rem;">What is the Leontief inverse?</b><br>
The IO table (matrix <b>A</b>) records only <em>direct</em> linkages — sector X buys
directly from sector Y. But shocks propagate indirectly too: if agriculture is hit, it
buys less from manufacturing, which then buys less from services, and so on.
The <b>Leontief inverse</b> (matrix <b>L = (I − A)⁻¹</b>) solves for the <em>total</em>
output change in every sector per unit of final demand change — capturing all direct and
indirect ripple effects in one calculation. This is the standard tool in IO economics
(Leontief, 1941) for measuring economy-wide interdependence.<br><br>

<b style="font-size:0.9rem;">How is the Shock Exposure score calculated?</b><br>
For each sector the engine computes three sub-scores, then combines them:

<table style="width:100%; border-collapse:collapse; font-size:0.8rem; margin:0.6rem 0;">
  <tr style="border-bottom:1px solid #444;">
    <th style="text-align:left; padding:4px 8px; color:{theme.text_muted};">Sub-score</th>
    <th style="text-align:left; padding:4px 8px; color:{theme.text_muted};">What it measures</th>
    <th style="text-align:center; padding:4px 8px; color:{theme.text_muted};">Weight</th>
  </tr>
  <tr>
    <td style="padding:4px 8px;"><b>Shock Pressure</b></td>
    <td style="padding:4px 8px;">How much of your selected shocks land directly on this
    sector, based on the shock-to-sector mapping in the scenario library</td>
    <td style="text-align:center; padding:4px 8px;">55 %</td>
  </tr>
  <tr>
    <td style="padding:4px 8px;"><b>IO Dependence</b></td>
    <td style="padding:4px 8px;">How tightly the sector is linked to the rest of the economy
    in the raw IO matrix (A) — sectors with many input links amplify shocks faster</td>
    <td style="text-align:center; padding:4px 8px;">25 %</td>
  </tr>
  <tr>
    <td style="padding:4px 8px;"><b>Leontief Spillover</b></td>
    <td style="padding:4px 8px;">How large the total economy-wide ripple is if this sector
    is hit — taken from the Leontief inverse (L), so indirect chains are included</td>
    <td style="text-align:center; padding:4px 8px;">20 %</td>
  </tr>
</table>

All three sub-scores are min-max normalised to [0, 1] before weighting, so the final
<b>Shock Exposure</b> value is also on a 0–1 scale (0 = lowest relative risk, 1 = highest).

<br><b style="font-size:0.9rem;">How is the Policy Coverage score calculated?</b><br>
Each policy instrument you select is mapped to the sectors it affects.
For example, a monetary easing instrument affects all sectors at a base weight of 0.35,
while a targeted agriculture subsidy scores 1.0 for agriculture and lower for others.
The raw policy pressure per sector is min-max normalised, then blended with a
<b>structural reach score</b> (30 %) derived from the Leontief backward and forward
linkage multipliers — sectors with stronger economy-wide linkages benefit more from
broad-based policies.

<br><b style="font-size:0.9rem;">How do I read the Risk Level labels?</b><br>

<table style="width:100%; border-collapse:collapse; font-size:0.8rem; margin:0.6rem 0;">
  <tr style="border-bottom:1px solid #444;">
    <th style="text-align:left; padding:4px 8px; color:{theme.text_muted};">Label</th>
    <th style="text-align:left; padding:4px 8px; color:{theme.text_muted};">Shock Exposure score</th>
    <th style="text-align:left; padding:4px 8px; color:{theme.text_muted};">Meaning</th>
  </tr>
  <tr>
    <td style="padding:4px 8px;"><b>High Spillover Risk</b></td>
    <td style="padding:4px 8px;">≥ 0.67</td>
    <td style="padding:4px 8px;">Sector is heavily exposed; shocks are likely to cascade
    through supply chains into the broader economy</td>
  </tr>
  <tr>
    <td style="padding:4px 8px;"><b>Moderate Spillover Risk</b></td>
    <td style="padding:4px 8px;">0.40 – 0.67</td>
    <td style="padding:4px 8px;">Meaningful exposure; spillovers are possible but
    limited in scope</td>
  </tr>
  <tr>
    <td style="padding:4px 8px;"><b>Contained Spillover</b></td>
    <td style="padding:4px 8px;">&lt; 0.40</td>
    <td style="padding:4px 8px;">Low relative exposure; sector is relatively insulated
    from the selected shocks</td>
  </tr>
</table>

<b style="font-size:0.9rem;">Important caveats</b><br>
This preview is computed <em>before</em> the simulation runs. It is a static, pre-run
diagnostic based on the IO structure and your shock/policy configuration — not a
time-series forecast. The <b>Sector Impact</b> tab (after running) shows the full
dynamic trajectory. Treat this preview as a quick sense-check of where risk is
concentrated, not as a final result.

</div>
""", unsafe_allow_html=True)

    with _srp_tab_preview:
      st.caption(
        "Which sectors are most exposed to your selected shocks, and how well do your policies cover them? "
        "Scores combine Kenya's Leontief IO matrix with your shock magnitudes. "
        "See the Data & Methodology tab for a full explanation."
      )

      if not HAS_DATA_STACK or np is None or pd is None:
        st.info("Install numpy/pandas to enable sector classification preview.")
      else:
        try:
          from scarcity.simulation.io_structure import default_kenya_io_config, LeontiefModel

          io_cfg = default_kenya_io_config()
          sector_names = list((io_cfg.sector_shares or {}).keys())
          if not sector_names:
            raise ValueError("No sector definitions found in IO config.")

          # Use a fixed reference sector — no need to expose this to the user
          benchmark_sector = sector_names[0]

          A = np.asarray(io_cfg.io_matrix, dtype=float)
          if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] != len(sector_names):
            raise ValueError("IO matrix shape does not match configured sectors.")

          L = np.asarray(LeontiefModel(A).leontief_inverse, dtype=float)
          idx = {s: i for i, s in enumerate(sector_names)}
          b = idx[benchmark_sector]

          def _minmax(vals):
            arr = np.asarray(vals, dtype=float)
            if arr.size == 0:
              return arr
            lo = float(np.min(arr))
            hi = float(np.max(arr))
            if hi - lo < 1e-9:
              return np.ones_like(arr) * 0.5
            return (arr - lo) / (hi - lo)

          def _norm_key(txt):
            return "".join(ch for ch in str(txt).lower() if ch.isalnum())

          dep_vs_b = []
          spill_vs_b = []
          for s in sector_names:
            i = idx[s]
            dep_vs_b.append(0.5 * (float(A[i, b]) + float(A[b, i])))
            spill_vs_b.append(0.5 * (float(L[i, b]) + float(L[b, i])))

          dep_norm = _minmax(dep_vs_b)
          spill_norm = _minmax(spill_vs_b)

          # Effective shock magnitudes from selected presets + custom shocks
          effective_shocks = {}
          preset_overrides = st.session_state.get("sim_preset_overrides", {})
          for s in selected_scenarios:
            overrides = preset_overrides.get(s.id, s.shocks)
            for k, v in overrides.items():
              try:
                effective_shocks[k] = effective_shocks.get(k, 0.0) + float(v)
              except Exception:
                continue
          for cs in custom_shocks:
            k = str(cs.get("key", "")).strip()
            if not k:
              continue
            try:
              effective_shocks[k] = effective_shocks.get(k, 0.0) + float(cs.get("magnitude", 0.0) or 0.0)
            except Exception:
              continue

          sens_cfg = io_cfg.shock_sensitivity or {}
          shock_pressure = {s: 0.0 for s in sector_names}
          shock_driver_map = {s: {} for s in sector_names}
          macro_keys = {"demand_shock", "supply_shock", "fiscal_shock", "fx_shock"}

          def _add_driver(sec_name, shock_key, contribution):
            if contribution <= 0:
              return
            prev = shock_driver_map[sec_name].get(shock_key, 0.0)
            shock_driver_map[sec_name][shock_key] = prev + float(contribution)

          for k, mag in effective_shocks.items():
            mag_abs = abs(float(mag))
            if mag_abs < 1e-9:
              continue
            reg = shock_registry.get(k, {})
            sfc_map = reg.get("sfc_mapping", {}) if isinstance(reg, dict) else {}

            # Map direct sector-tagged shocks to their closest IO sector.
            sec_tag = reg.get("sector", "") if isinstance(reg, dict) else ""
            sec_tag_norm = _norm_key(sec_tag)
            for s in sector_names:
              s_norm = _norm_key(s)
              if sec_tag_norm and sec_tag_norm == s_norm:
                direct = 0.50 * mag_abs
                shock_pressure[s] += direct
                _add_driver(s, k, direct)
              elif sec_tag_norm == "foodmarkets" and s_norm in {"agriculture", "manufacturing", "services"}:
                direct = 0.25 * mag_abs
                shock_pressure[s] += direct
                _add_driver(s, k, direct)
              elif sec_tag_norm == "communications" and s_norm == "services":
                direct = 0.30 * mag_abs
                shock_pressure[s] += direct
                _add_driver(s, k, direct)
              elif sec_tag_norm == "displacement" and s_norm in {"services", "security"}:
                direct = 0.20 * mag_abs
                shock_pressure[s] += direct
                _add_driver(s, k, direct)

            if sfc_map:
              for s in sector_names:
                s_sens = sens_cfg.get(s, {})
                mapped = 0.0
                for ch, w in sfc_map.items():
                  mapped += mag_abs * abs(float(w)) * float(s_sens.get(ch, 1.0))
                shock_pressure[s] += mapped
                _add_driver(s, k, mapped)
            elif k in macro_keys:
              for s in sector_names:
                s_sens = sens_cfg.get(s, {})
                mapped = mag_abs * float(s_sens.get(k, 1.0))
                shock_pressure[s] += mapped
                _add_driver(s, k, mapped)

          shock_vec = np.array([shock_pressure[s] for s in sector_names], dtype=float)
          shock_norm = _minmax(shock_vec)

          # Effective policy configuration from selected presets + custom instruments.
          effective_policy = {}
          for pk in selected_policy_keys:
            effective_policy.update(policy_templates.get(pk, {}).get("instruments", {}))
          for ci in custom_instruments:
            k = str(ci.get("key", "")).strip()
            if k:
              effective_policy[k] = ci.get("value")

          policy_pressure = {s: 0.0 for s in sector_names}
          for k, v in effective_policy.items():
            try:
              mag_abs = abs(float(v))
            except Exception:
              continue
            reg = policy_instrument_registry.get(k, {})
            cat_norm = _norm_key(reg.get("category", "General")) if isinstance(reg, dict) else "general"
            for s in sector_names:
              s_norm = _norm_key(s)
              w = 0.0
              if cat_norm in {"general", "monetary", "fiscal"}:
                w = 0.35
              elif cat_norm == s_norm:
                w = 1.00
              elif cat_norm == "markets" and s_norm in {"agriculture", "manufacturing", "services"}:
                w = 0.70
              elif cat_norm == "socialprotection" and s_norm in {"services", "health", "security"}:
                w = 0.60
              elif cat_norm == "communications" and s_norm in {"services", "security"}:
                w = 0.45
              policy_pressure[s] += w * mag_abs

          policy_vec = np.array([policy_pressure[s] for s in sector_names], dtype=float)
          policy_norm = _minmax(policy_vec)

          multipliers = np.asarray(L.sum(axis=0), dtype=float)
          backward = np.asarray(L.sum(axis=0), dtype=float)
          forward = np.asarray(L.sum(axis=1), dtype=float)
          structural = 0.40 * _minmax(multipliers) + 0.35 * _minmax(backward) + 0.25 * _minmax(forward)

          shock_scores = 0.55 * shock_norm + 0.25 * dep_norm + 0.20 * spill_norm
          policy_scores = 0.70 * policy_norm + 0.30 * structural

          def _shock_cls(v):
            if v >= 0.67:
              return "High Spillover Risk"
            if v >= 0.40:
              return "Moderate Spillover Risk"
            return "Contained Spillover"

          def _policy_cls(v):
            if v >= 0.70:
              return "High Coverage"
            if v >= 0.45:
              return "Partial Coverage"
            return "Coverage Gap"

          rows = []
          for i, s in enumerate(sector_names):
            top_driver_items = sorted(
              shock_driver_map.get(s, {}).items(),
              key=lambda kv: kv[1],
              reverse=True,
            )[:3]
            top_driver_text = ", ".join([
              f"{shock_registry.get(kd, {}).get('label', kd)} ({val:.2f})"
              for kd, val in top_driver_items
            ]) if top_driver_items else "None"

            rows.append({
              "Sector": s.title(),
              "Risk Level": _shock_cls(float(shock_scores[i])),
              "Shock Exposure": round(float(shock_scores[i]), 3),
              "Main Shock Drivers": top_driver_text,
              "Policy Coverage": _policy_cls(float(policy_scores[i])),
              "_policy_score": round(float(policy_scores[i]), 3),  # kept for chart only
            })

          class_df = pd.DataFrame(rows).sort_values("Shock Exposure", ascending=False)
          # Display simplified table — hide internal chart column
          display_cols = ["Sector", "Risk Level", "Shock Exposure", "Main Shock Drivers", "Policy Coverage"]
          st.dataframe(class_df[display_cols], use_container_width=True, hide_index=True)

          if HAS_PLOTLY and go is not None:
            fig_cls = go.Figure()
            fig_cls.add_trace(go.Bar(
              x=class_df["Sector"],
              y=class_df["Shock Exposure"],
              name="Shock Exposure",
              marker_color=PALETTE[3],
            ))
            fig_cls.add_trace(go.Bar(
              x=class_df["Sector"],
              y=class_df["_policy_score"],
              name="Policy Coverage",
              marker_color=PALETTE[0],
            ))
            fig_cls.update_layout(**base_layout(theme, height=300,
              title=dict(
                text="Sector Shock Exposure vs Policy Coverage (0 = none, 1 = high)",
                font=dict(color=theme.text_muted, size=13),
              ),
              barmode="group",
              xaxis=dict(title="Sector"),
              yaxis=dict(title="Score (0–1)", range=[0, 1.0]),
              legend=dict(orientation="h", y=1.08, x=0, bgcolor="rgba(0,0,0,0)"),
            ))
            st.plotly_chart(fig_cls, use_container_width=True)
        except Exception as e:
          st.warning(f"Sector classification preview unavailable: {e}")

    # ─── SECTION 3: SIMULATION PARAMS ─────────────────────────────────
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
      st.markdown(f"<div style='color:{theme.accent_success}; font-weight:600; "
            f"font-size:0.8rem; margin-bottom:0.3rem;'>SIMULATION</div>",
            unsafe_allow_html=True)
      steps = st.slider("Quarters", 20, 100, 50, 5, key="sim_steps")

    # ─── SECTION 4: OUTCOME DIMENSIONS ────────────────────────────────
    with c2:
      st.markdown(f"<div style='color:{theme.text_muted}; font-weight:600; "
            f"font-size:0.8rem; margin-bottom:0.3rem;'>OUTCOME DIMENSIONS</div>",
            unsafe_allow_html=True)

    # Collect suggested dims from all selected scenarios
    all_suggested = set()
    for s in selected_scenarios:
      all_suggested.update(s.suggested_dimensions)
    defaults = list(all_suggested) if all_suggested else list(default_dimensions)

    categories = {}
    for key, meta in outcome_dimensions.items():
      cat = meta.get("category", "Other")
      categories.setdefault(cat, []).append((key, meta))

    selected_dims = []
    dim_cols = st.columns(min(len(categories), 5))
    for i, (cat, dims) in enumerate(categories.items()):
      with dim_cols[i % len(dim_cols)]:
        st.markdown(f"<div style='color:{theme.text_muted}; font-weight:600; "
              f"font-size:0.72rem;'>{cat}</div>", unsafe_allow_html=True)
        for key, meta in dims:
          if st.checkbox(meta["label"], value=key in defaults,
                  key=f"sim_dim_{key}", help=meta["description"]):
            selected_dims.append(key)
    if not selected_dims:
      selected_dims = list(default_dimensions)

  # Store scenario obj for IRF tab (use first selected, or None)
  st.session_state["_sim_scenario_obj"] = selected_scenarios[0] if selected_scenarios else None

  return {
    "selected_scenarios": selected_scenarios,
    "custom_shocks": custom_shocks,
    "selected_policy_keys": selected_policy_keys,
    "custom_instruments": custom_instruments,
    "selected_dims": selected_dims,
    "steps": steps,
  }
