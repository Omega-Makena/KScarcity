"""Scenario Configuration Panel."""

from ._shared import st


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
        scenario_options = {s.id: f"{s.name}  ({s.category})" for s in scenario_library}
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
                parts = [f"{shock_registry.get(k, {{}}).get('label', k)}: "
                         f"<b>{v:+.2f}</b>" for k, v in shock_summary.items()]
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
        shock_type_keys = list(shock_registry.keys()) if shock_registry else ["demand_shock", "supply_shock", "fiscal_shock", "fx_shock"]
        shock_type_labels = {k: v.get("label", k) for k, v in shock_registry.items()} if shock_registry else {k: k for k in shock_type_keys}

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

        inst_type_keys = list(policy_instrument_registry.keys()) if policy_instrument_registry else []
        inst_type_labels = {k: v.get("label", k) for k, v in policy_instrument_registry.items()} if policy_instrument_registry else {}

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
