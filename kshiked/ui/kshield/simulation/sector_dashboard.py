"""
Sector Impact Dashboard — renders all four UI components:

  1. Sector Panels       — side-by-side baseline→projected→delta, color-coded
  2. Impact Heatmap      — 6×6 influence grid, editable in Advanced mode
  3. Timeline            — multi-line chart, 3 horizons
  4. Severity/Confidence — per-sector severity bar + confidence indicator

Simplified Mode: summary view, static heatmap, 3 horizon snapshots.
Advanced Mode:   sub-indicator drill-down, editable heatmap, continuous timeline.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ._shared import st, pd, np, go, make_subplots, HAS_PLOTLY, PALETTE, base_layout

from kshiked.simulation.sector_registry import (
    ALL_SECTORS, INFLUENCE_WEIGHTS, SECTOR_COLORS, SECTOR_LABELS,
    SECTOR_SHORT, SECTOR_SUB_INDICATORS, SectorID, SectorState,
)


# ─── Fallback: rebuild Economics sector from raw SFC trajectory ───────────────

def _rebuild_econ_sector_from_trajectory(
    trajectory: List[Dict],
) -> Dict[SectorID, SectorState]:
    """
    Construct a minimal Economics SectorState from SFC trajectory outcomes.
    Used when SectorSimulator.project() fails so the Sector Impact tab always
    shows real data for at least the core macro sector.
    """
    import math

    if len(trajectory) < 2:
        return {}

    from kshiked.simulation.sector_engine import KENYA_BASELINES

    inds = SECTOR_SUB_INDICATORS.get(SectorID.ECONOMICS_FINANCE, [])
    baseline = dict(KENYA_BASELINES.get(SectorID.ECONOMICS_FINANCE, {}))

    t0 = trajectory[0].get("outcomes", {})
    tf = trajectory[-1].get("outcomes", {})

    # Build timeline directly from trajectory frames
    timeline: Dict[str, List[float]] = {}
    for ind in inds:
        series = []
        for frame in trajectory:
            val = frame.get("outcomes", {}).get(ind.key)
            series.append(float(val) if val is not None else baseline.get(ind.key, 0.0))
        timeline[ind.key] = series

    # Terminal delta = last frame value − baseline
    deltas: Dict[str, float] = {}
    projected: Dict[str, float] = {}
    for ind in inds:
        base_val = baseline.get(ind.key, 0.0)
        traj_val = tf.get(ind.key)
        if traj_val is not None:
            delta = float(traj_val) - base_val
        else:
            delta = 0.0
        lo, hi = ind.typical_range
        clamped = max(lo, min(hi, base_val + delta))
        deltas[ind.key] = round(clamped - base_val, 6)
        projected[ind.key] = round(clamped, 6)

    # Severity
    sev_num = sev_den = 0.0
    for ind in inds:
        d = abs(deltas.get(ind.key, 0.0))
        rng = ind.typical_range[1] - ind.typical_range[0]
        if rng < 1e-9:
            continue
        sev_num += (d / rng) * ind.weight
        sev_den += ind.weight
    severity = max(1.0, min(10.0, (sev_num / sev_den if sev_den > 0 else 0.0) * 10.0))

    state = SectorState(
        sector_id=SectorID.ECONOMICS_FINANCE,
        baseline=baseline,
        projected=projected,
        delta=deltas,
        severity=round(severity, 1),
        confidence=60.0,
        sub_indicators=inds,
        direct_effects=list(deltas.keys()),
        induced_effects=[],
        spillover_hints=[],
        model_assumptions={
            "source": "SFC trajectory passthrough (sector engine unavailable)",
            "calibration": "Kenya 2022 World Bank / KNBS baselines",
        },
        timeline=timeline,
    )
    return {SectorID.ECONOMICS_FINANCE: state}


# ─── Color helpers ────────────────────────────────────────────────────────────

def _hex_rgba(hex6: str, alpha: float = 0.15) -> str:
    h = hex6.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _delta_color(delta: float, higher_is_better: bool, theme) -> str:
    if abs(delta) < 1e-9:
        return theme.text_muted
    improving = (delta > 0) if higher_is_better else (delta < 0)
    return theme.accent_success if improving else theme.accent_danger


def _severity_color(severity: float, theme) -> str:
    if severity < 3:
        return theme.accent_success
    if severity < 6:
        return theme.accent_warning
    return theme.accent_danger


# ─── Component 1: Sector Panels ───────────────────────────────────────────────

def render_sector_panels(
    sector_results: Dict[SectorID, SectorState],
    theme,
    advanced: bool = False,
) -> None:
    """
    Render one panel per sector. Simplified: summary metrics.
    Advanced: expandable sub-indicators + confidence intervals.
    """
    if not sector_results:
        st.info("No sector results available. Run a simulation first.")
        return

    sectors = [s for s in ALL_SECTORS if s in sector_results]
    cols = st.columns(min(3, len(sectors)))

    for idx, sid in enumerate(sectors):
        state = sector_results[sid]
        col   = cols[idx % len(cols)]
        color = SECTOR_COLORS[sid]

        with col:
            # Panel header
            net = state.net_impact_score()
            net_color = theme.accent_success if net > 0 else (
                theme.accent_danger if net < 0 else theme.text_muted
            )
            sev_color = _severity_color(state.severity, theme)

            st.markdown(
                f"<div style='border:1px solid {color}33; border-radius:6px; "
                f"padding:0.75rem; margin-bottom:0.5rem;'>"
                f"<div style='color:{color}; font-weight:700; font-size:0.80rem; "
                f"letter-spacing:0.06em;'>{SECTOR_LABELS[sid].upper()}</div>"
                f"<div style='display:flex; gap:1rem; margin-top:0.4rem;'>"
                f"<span style='font-size:0.72rem; color:{sev_color};'>"
                f"Severity {state.severity:.1f}/10</span>"
                f"<span style='font-size:0.72rem; color:{theme.text_muted};'>"
                f"Confidence {state.confidence:.0f}%</span>"
                f"<span style='font-size:0.72rem; color:{net_color}; font-weight:600;'>"
                f"Net: {net:+.2f}</span>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            if advanced:
                # Expandable sub-indicator drill-down
                with st.expander("Sub-indicators", expanded=False):
                    for ind in state.sub_indicators:
                        base_v = state.baseline.get(ind.key, 0.0)
                        proj_v = state.projected.get(ind.key, 0.0)
                        d      = state.delta.get(ind.key, 0.0)
                        dc     = _delta_color(d, ind.higher_is_better, theme)
                        direct_tag = (
                            "<span style='font-size:0.65rem; color:#888;'>[direct]</span>"
                            if ind.key in state.direct_effects else
                            "<span style='font-size:0.65rem; color:#555;'>[induced]</span>"
                        )
                        st.markdown(
                            f"<div style='display:flex; justify-content:space-between; "
                            f"align-items:center; padding:0.2rem 0; "
                            f"border-bottom:1px solid {theme.border_default}33;'>"
                            f"<span style='font-size:0.78rem; color:{theme.text_primary};'>"
                            f"{ind.label} {direct_tag}</span>"
                            f"<span style='font-size:0.78rem;'>"
                            f"<span style='color:{theme.text_muted};'>{base_v:.3f}</span>"
                            f"<span style='color:{theme.text_muted};'> → </span>"
                            f"<span style='color:{theme.text_primary};'>{proj_v:.3f}</span>"
                            f"<span style='color:{dc}; margin-left:0.4rem;'>"
                            f"({d:+.3f})</span>"
                            f"</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                # Model assumptions
                with st.expander("Assumptions", expanded=False):
                    for k, v in state.model_assumptions.items():
                        st.markdown(
                            f"<div style='font-size:0.72rem; color:{theme.text_muted}; "
                            f"margin-bottom:0.2rem;'><b>{k}:</b> {v}</div>",
                            unsafe_allow_html=True,
                        )
            else:
                # Overview: all indicators, most-moved first
                sorted_inds = sorted(
                    state.sub_indicators,
                    key=lambda i: abs(state.delta.get(i.key, 0.0)),
                    reverse=True,
                )
                for ind in sorted_inds:
                    base_v = state.baseline.get(ind.key, 0.0)
                    proj_v = state.projected.get(ind.key, 0.0)
                    d      = state.delta.get(ind.key, 0.0)
                    dc     = _delta_color(d, ind.higher_is_better, theme)
                    # Skip indicators with no movement in overview mode
                    if abs(d) < 1e-6:
                        continue
                    st.markdown(
                        f"<div style='font-size:0.76rem; padding:0.12rem 0; "
                        f"display:flex; justify-content:space-between; "
                        f"border-bottom:1px solid {theme.border_default}22;'>"
                        f"<span style='color:{theme.text_muted};'>{ind.label}</span>"
                        f"<span>"
                        f"<span style='color:{theme.text_muted}; font-size:0.70rem;'>"
                        f"{base_v:.3f} → {proj_v:.3f}</span>"
                        f"<span style='color:{dc}; font-weight:600; margin-left:0.4rem;'>"
                        f"({d:+.3f})</span>"
                        f"</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Spillover hints (collapsed)
            if state.spillover_hints:
                with st.expander("Spillover Risks", expanded=False):
                    for h in state.spillover_hints[:4]:
                        st.markdown(
                            f"<div style='font-size:0.72rem; color:{theme.text_muted}; "
                            f"margin-bottom:0.2rem;'>"
                            f"→ <b>{h['label']}</b>: {h['reason']} "
                            f"(est. {h['magnitude_estimate']:.2f})</div>",
                            unsafe_allow_html=True,
                        )


# ─── Component 2: Impact Heatmap ─────────────────────────────────────────────

def render_impact_heatmap(
    sector_results: Dict[SectorID, SectorState],
    theme,
    advanced: bool = False,
) -> Optional[Dict[SectorID, Dict[SectorID, float]]]:
    """
    6×6 influence heatmap. Cell value = w[row_sector][col_sector] × row_severity.
    Advanced mode: user can override individual weights via number inputs.

    Returns updated weights dict if user modified them (else None).
    """
    if not HAS_PLOTLY:
        st.warning("Plotly not available — heatmap disabled.")
        return None

    sectors = ALL_SECTORS
    labels  = [SECTOR_SHORT[s] for s in sectors]

    # Load weights — editable in advanced mode
    weights_key = "sim_sector_influence_weights"
    if weights_key not in st.session_state:
        st.session_state[weights_key] = {
            from_s: dict(row) for from_s, row in INFLUENCE_WEIGHTS.items()
        }
    current_weights: Dict[SectorID, Dict[SectorID, float]] = st.session_state[weights_key]

    if advanced:
        with st.expander("Edit Influence Weights (Advanced)", expanded=False):
            st.markdown(
                f"<div style='font-size:0.75rem; color:{theme.text_muted}; "
                f"margin-bottom:0.5rem;'>"
                f"Override cross-sector influence weights. "
                f"Values 0–1 (1=max influence).</div>",
                unsafe_allow_html=True,
            )
            changed = False
            w_cols = st.columns(len(sectors))
            for j, to_sid in enumerate(sectors):
                with w_cols[j]:
                    st.markdown(
                        f"<div style='font-size:0.68rem; color:{SECTOR_COLORS[to_sid]}; "
                        f"text-align:center; margin-bottom:0.2rem;'>"
                        f"{SECTOR_SHORT[to_sid]}</div>",
                        unsafe_allow_html=True,
                    )
            for i, from_sid in enumerate(sectors):
                row_cols = st.columns(len(sectors))
                for j, to_sid in enumerate(sectors):
                    with row_cols[j]:
                        if from_sid == to_sid:
                            st.markdown(
                                f"<div style='text-align:center; font-size:0.72rem; "
                                f"color:{theme.text_muted};'>—</div>",
                                unsafe_allow_html=True,
                            )
                        else:
                            current_w = current_weights[from_sid][to_sid]
                            new_w = st.number_input(
                                f"{SECTOR_SHORT[from_sid]}→{SECTOR_SHORT[to_sid]}",
                                min_value=0.0, max_value=1.0,
                                value=float(current_w), step=0.05,
                                key=f"w_{from_sid.value}_{to_sid.value}",
                                label_visibility="collapsed",
                            )
                            if abs(new_w - current_w) > 1e-6:
                                current_weights[from_sid][to_sid] = new_w
                                changed = True
            if changed:
                st.session_state[weights_key] = current_weights

    # Build matrix: cell = w[from][to] × from_severity
    n = len(sectors)
    z     = [[0.0] * n for _ in range(n)]
    text  = [[""] * n for _ in range(n)]

    for i, from_sid in enumerate(sectors):
        sev = (sector_results[from_sid].severity / 10.0
               if from_sid in sector_results else 0.5)
        for j, to_sid in enumerate(sectors):
            w = current_weights.get(from_sid, {}).get(to_sid, 0.0)
            val = w * sev if from_sid != to_sid else 0.0
            z[i][j]    = round(val, 3)
            text[i][j] = f"{val:.2f}" if from_sid != to_sid else "self"

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        colorscale=[
            [0.0, "#1a1a2e"],
            [0.3, "#16213e"],
            [0.6, "#f97316"],
            [1.0, "#ff3366"],
        ],
        zmin=0.0, zmax=0.65,
        showscale=True,
        hovertemplate=(
            "<b>%{y} → %{x}</b><br>"
            "Influence strength: %{z:.3f}<extra></extra>"
        ),
    ))

    fig.update_layout(**base_layout(
        theme, height=420,
        title=dict(
            text="Cross-Sector Influence Heatmap (row → col)",
            font=dict(color=theme.text_muted, size=12),
        ),
        xaxis=dict(title="Affected Sector"),
        yaxis=dict(title="Influencing Sector", autorange="reversed"),
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f"<div style='font-size:0.70rem; color:{theme.text_muted}; margin-top:-0.5rem;'>"
        "Cell = influence_weight × severity of influencing sector. "
        "Warm = stronger amplification. Rows drive columns.</div>",
        unsafe_allow_html=True,
    )

    return current_weights if advanced else None


# ─── Component 3: Timeline ────────────────────────────────────────────────────

def render_sector_timeline(
    sector_results: Dict[SectorID, SectorState],
    theme,
    advanced: bool = False,
) -> None:
    """
    Multi-line chart, one line per sector (using net_impact_score per step).
    Simplified: 3 horizon snapshots (short / medium / long-term).
    Advanced:   continuous trajectory + horizon selector.
    """
    if not HAS_PLOTLY or not sector_results:
        return

    sectors = [s for s in ALL_SECTORS if s in sector_results]
    if not sectors:
        return

    # Determine common timeline length
    n_steps = max(
        (
            len(next(iter(state.timeline.values()), []))
            for state in sector_results.values()
            if state.timeline
        ),
        default=0,
    )
    if n_steps < 2:
        st.info("No timeline data — run a simulation with at least 2 steps.")
        return

    if advanced:
        horizon_opts = {
            "Short-term (0–8 qtrs)":   (0,       min(8, n_steps)),
            "Medium-term (0–40 qtrs)":  (0,       min(40, n_steps)),
            "Long-term (all)":          (0,       n_steps),
        }
        col_h, col_ind = st.columns([2, 2])
        with col_h:
            h_choice = st.selectbox(
                "Horizon", list(horizon_opts.keys()),
                key="sector_tl_horizon",
            )
        with col_ind:
            ind_opts = ["Net impact score"] + [
                ind.label
                for ind in SECTOR_SUB_INDICATORS.get(sectors[0], [])
            ]
            ind_choice = st.selectbox(
                "Indicator", ind_opts,
                key="sector_tl_indicator",
            )
        t_start, t_end = horizon_opts[h_choice]
    else:
        t_start, t_end = 0, n_steps
        ind_choice = "Net impact score"

    fig = go.Figure()

    for idx, sid in enumerate(sectors):
        state = sector_results[sid]
        color = SECTOR_COLORS[sid]
        label = SECTOR_LABELS[sid]

        if ind_choice == "Net impact score":
            # Compute net impact score at each step from timelines
            inds_data = state.timeline
            inds_defs = SECTOR_SUB_INDICATORS.get(sid, [])
            series = []
            for t in range(t_start, t_end):
                step_score = 0.0
                step_w     = 0.0
                for ind in inds_defs:
                    tl = inds_data.get(ind.key, [])
                    if len(tl) > t:
                        base = state.baseline.get(ind.key, 0.0)
                        rng  = ind.typical_range[1] - ind.typical_range[0]
                        if rng < 1e-9:
                            continue
                        d    = tl[t] - base
                        norm = d / rng
                        sign = norm if ind.higher_is_better else -norm
                        step_score += sign * ind.weight
                        step_w     += ind.weight
                series.append(step_score / step_w if step_w > 0 else 0.0)
            y_title = "Net Impact Score"
        else:
            # Find the matching sub-indicator
            ind_match = next(
                (i for i in SECTOR_SUB_INDICATORS.get(sid, []) if i.label == ind_choice),
                None,
            )
            if ind_match is None:
                series = [0.0] * (t_end - t_start)
                y_title = ind_choice
            else:
                tl     = state.timeline.get(ind_match.key, [])
                series = [tl[t] if t < len(tl) else 0.0 for t in range(t_start, t_end)]
                y_title = f"{ind_match.label} ({ind_match.unit})"

        x_vals = list(range(t_start, t_end))

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=series,
            mode="lines",
            name=label,
            line=dict(color=color, width=2.5),
            hovertemplate=f"{label}: %{{y:.3f}}<extra></extra>",
        ))

        if not advanced:
            # Fill between zero and the series
            fig.add_trace(go.Scatter(
                x=x_vals + x_vals[::-1],
                y=series + [0.0] * len(series),
                fill="toself",
                fillcolor=_hex_rgba(color, 0.06),
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))

    # Horizon bands for simplified mode
    if not advanced:
        for band_start, band_end, band_label in [
            (0,           min(8,  n_steps), "Short-term"),
            (min(8,  n_steps), min(40, n_steps), "Medium-term"),
            (min(40, n_steps), n_steps,           "Long-term"),
        ]:
            if band_end > band_start:
                fig.add_vrect(
                    x0=band_start, x1=band_end,
                    fillcolor="rgba(255,255,255,0.02)",
                    line_width=0,
                    annotation_text=band_label,
                    annotation_position="top left",
                    annotation_font=dict(size=9, color=theme.text_muted),
                )

    fig.add_hline(y=0, line_dash="dot", line_color=theme.text_muted,
                  line_width=1, opacity=0.4)

    fig.update_layout(**base_layout(
        theme, height=380,
        title=dict(
            text=f"Sector Timeline — {ind_choice}",
            font=dict(color=theme.text_muted, size=12),
        ),
        xaxis=dict(title="Quarter"),
        yaxis=dict(title=y_title),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
        ),
    ))
    st.plotly_chart(fig, use_container_width=True)

    # 3 horizon snapshots (always shown)
    _render_horizon_snapshots(sector_results, theme, n_steps)


def _render_horizon_snapshots(
    sector_results: Dict[SectorID, SectorState],
    theme,
    n_steps: int,
) -> None:
    horizons = [
        ("Short-term",  min(8,        n_steps) - 1),
        ("Medium-term", min(40,       n_steps) - 1),
        ("Long-term",   min(n_steps,  n_steps) - 1),
    ]
    hcols = st.columns(3)
    for (h_label, t_idx), hcol in zip(horizons, hcols):
        with hcol:
            st.markdown(
                f"<div style='font-size:0.72rem; color:{theme.text_muted}; "
                f"font-weight:700; margin-bottom:0.3rem;'>{h_label.upper()}</div>",
                unsafe_allow_html=True,
            )
            for sid in ALL_SECTORS:
                if sid not in sector_results:
                    continue
                state = sector_results[sid]
                color = SECTOR_COLORS[sid]
                inds  = SECTOR_SUB_INDICATORS.get(sid, [])
                score = 0.0
                sw    = 0.0
                for ind in inds:
                    tl = state.timeline.get(ind.key, [])
                    if len(tl) > t_idx:
                        base = state.baseline.get(ind.key, 0.0)
                        rng  = ind.typical_range[1] - ind.typical_range[0]
                        if rng < 1e-9:
                            continue
                        d    = tl[t_idx] - base
                        norm = d / rng
                        sign = norm if ind.higher_is_better else -norm
                        score += sign * ind.weight
                        sw    += ind.weight
                net = score / sw if sw > 0 else 0.0
                nc  = theme.accent_success if net > 0 else (
                    theme.accent_danger if net < 0 else theme.text_muted
                )
                st.markdown(
                    f"<div style='display:flex; justify-content:space-between; "
                    f"font-size:0.75rem; padding:0.1rem 0;'>"
                    f"<span style='color:{color};'>{SECTOR_SHORT[sid]}</span>"
                    f"<span style='color:{nc}; font-weight:600;'>{net:+.3f}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ─── Component 4: Severity & Confidence ──────────────────────────────────────

def render_severity_confidence(
    sector_results: Dict[SectorID, SectorState],
    theme,
    advanced: bool = False,
) -> None:
    """
    Bar chart of severity (1–10) per sector + confidence badges.
    Advanced: shows the assumption/mechanism driving each confidence score.
    """
    if not HAS_PLOTLY or not sector_results:
        return

    sectors = [s for s in ALL_SECTORS if s in sector_results]
    if not sectors:
        return

    labels     = [SECTOR_SHORT[s] for s in sectors]
    severities = [sector_results[s].severity    for s in sectors]
    confs      = [sector_results[s].confidence  for s in sectors]
    colors_hex = [SECTOR_COLORS[s]              for s in sectors]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Severity (1–10)", "Confidence (%)"],
        horizontal_spacing=0.10,
    )

    fig.add_trace(
        go.Bar(
            x=labels, y=severities,
            marker=dict(
                color=[_hex_rgba(c, 0.85) for c in colors_hex],
                line=dict(color=colors_hex, width=1.5),
            ),
            hovertemplate="<b>%{x}</b><br>Severity: %{y:.1f}/10<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )

    fig.add_trace(
        go.Bar(
            x=labels, y=confs,
            marker=dict(
                color=[_hex_rgba(c, 0.65) for c in colors_hex],
                line=dict(color=colors_hex, width=1.5),
            ),
            hovertemplate="<b>%{x}</b><br>Confidence: %{y:.0f}%<extra></extra>",
            showlegend=False,
        ),
        row=1, col=2,
    )

    # Severity reference lines
    for y_ref, label_ref in [(3, "Low"), (6, "Moderate"), (9, "High")]:
        fig.add_hline(
            y=y_ref, row=1, col=1,
            line_dash="dot", line_color=theme.text_muted,
            line_width=1, opacity=0.4,
            annotation_text=label_ref,
            annotation_font=dict(size=8, color=theme.text_muted),
        )

    fig.update_layout(**base_layout(
        theme, height=320,
        title=dict(
            text="Severity & Confidence per Sector",
            font=dict(color=theme.text_muted, size=12),
        ),
    ))
    fig.update_yaxes(range=[0, 10.5], row=1, col=1)
    fig.update_yaxes(range=[0, 100],  row=1, col=2)

    st.plotly_chart(fig, use_container_width=True)

    if advanced:
        # Full methodology per sector
        with st.expander("Confidence Methodology", expanded=False):
            for sid in sectors:
                state  = sector_results[sid]
                color  = SECTOR_COLORS[sid]
                assump = state.model_assumptions

                n_direct  = len(state.direct_effects)
                n_induced = len(state.induced_effects)
                n_total   = n_direct + n_induced
                dir_ratio = n_direct / n_total if n_total > 0 else 0.5

                st.markdown(
                    f"<div style='border-left:3px solid {color}; "
                    f"padding:0.4rem 0.6rem; margin-bottom:0.6rem;'>"
                    f"<div style='color:{color}; font-weight:700; font-size:0.78rem;'>"
                    f"{SECTOR_LABELS[sid]}</div>"
                    f"<div style='font-size:0.72rem; color:{theme.text_muted}; "
                    f"margin-top:0.25rem;'>"
                    f"Confidence: <b>{state.confidence:.0f}%</b><br>"
                    f"Direct indicators: {n_direct} | Induced: {n_induced} | "
                    f"Directness ratio: {dir_ratio:.0%}<br>"
                    + "".join(
                        f"<b>{k}</b>: {v}<br>"
                        for k, v in assump.items()
                        if not k.startswith("conflict_")
                    )
                    + "</div></div>",
                    unsafe_allow_html=True,
                )

    # Policy conflicts (if any)
    conflicts = {
        k: v
        for sid, state in sector_results.items()
        for k, v in state.model_assumptions.items()
        if k.startswith("conflict_")
    }
    if conflicts:
        st.warning(
            f"{len(conflicts)} policy conflict(s) detected. "
            "Check Sector Panels > Assumptions for details."
        )


# ─── Top-level renderer ───────────────────────────────────────────────────────

def render_sector_impact(theme) -> None:
    """
    Main entry point — called from view.py under the 'Sector Impact' category.
    Loads sector_results from session state and renders all four components.
    """
    sector_results: Optional[Dict[SectorID, SectorState]] = (
        st.session_state.get("sim_sector_results")
    )

    if not sector_results:
        trajectory = st.session_state.get("sim_trajectory")
        if not trajectory:
            # No simulation run at all
            st.markdown(
                f"<div style='text-align:center; padding:3rem; color:{theme.text_muted};'>"
                "<div style='font-size:2rem; opacity:0.3; margin-bottom:0.5rem;'>"
                "&#9698;</div>"
                "<div style='font-size:0.9rem;'>"
                "Run a simulation in <b>Setup &amp; Run</b> first to see multi-sector impact."
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            if st.button("Go to Setup & Run", key="_sector_goto_setup", type="primary"):
                st.session_state["sim_category"] = "Setup & Run"
                st.rerun()
            return

        # Simulation ran but sector engine failed — rebuild from SFC trajectory
        st.warning(
            "The multi-sector projection engine encountered an error. "
            "Showing Economics sector data derived directly from the SFC trajectory. "
            "Re-run the simulation to attempt full 6-sector projection."
        )
        sector_results = _rebuild_econ_sector_from_trajectory(trajectory)
        if not sector_results:
            st.info("Unable to derive sector data from trajectory.")
            return

    # ── Top bar: mode toggle + data provenance note ───────────────────────────
    hdr_l, hdr_r = st.columns([4, 2])
    with hdr_l:
        # Show which shocks/policies drove this run
        _traj = st.session_state.get("sim_trajectory", [])
        _calib = st.session_state.get("sim_calibration")
        _n_frames = len(_traj)
        _n_data = (
            sum(1 for p in _calib.params.values() if p.source == "data")
            if _calib else 0
        )
        _n_total = len(_calib.params) if _calib else 0
        _conf = f"{_calib.overall_confidence:.0%}" if _calib else "—"
        st.markdown(
            f"<div style='font-size:0.72rem; color:{theme.text_muted}; padding-top:0.4rem;'>"
            f"Macro calibration: <b>{_conf}</b> ({_n_data}/{_n_total} from data) &nbsp;|&nbsp; "
            f"{_n_frames} simulation frames &nbsp;|&nbsp; "
            f"Baselines: Kenya 2022 (World Bank / KNBS)</div>",
            unsafe_allow_html=True,
        )
    with hdr_r:
        mode = st.radio(
            "Display mode",
            ["Overview", "Advanced"],
            horizontal=True,
            key="sector_impact_mode",
            label_visibility="collapsed",
        )
    advanced = mode == "Advanced"

    st.markdown(
        f"<div style='border-top:1px solid {theme.border_default}; "
        f"margin:0.3rem 0 0.8rem;'></div>",
        unsafe_allow_html=True,
    )

    # ── Persistent 6-sector summary strip (always visible) ────────────────────
    sectors_present = [s for s in ALL_SECTORS if s in sector_results]
    if sectors_present:
        summary_cols = st.columns(len(sectors_present))
        for col, sid in zip(summary_cols, sectors_present):
            state = sector_results[sid]
            color = SECTOR_COLORS[sid]
            net   = state.net_impact_score()
            sev   = state.severity
            sev_c = _severity_color(sev, theme)
            net_c = (theme.accent_success if net > 0.02 else
                     theme.accent_danger  if net < -0.02 else theme.text_muted)
            with col:
                st.markdown(
                    f"<div style='border:1px solid {color}44; border-radius:5px; "
                    f"padding:0.4rem 0.5rem; text-align:center; margin-bottom:0.6rem;'>"
                    f"<div style='color:{color}; font-size:0.68rem; font-weight:700; "
                    f"letter-spacing:0.05em;'>{SECTOR_SHORT[sid]}</div>"
                    f"<div style='color:{sev_c}; font-size:0.70rem; margin-top:0.15rem;'>"
                    f"Sev {sev:.1f}/10</div>"
                    f"<div style='color:{net_c}; font-size:0.78rem; font-weight:600;'>"
                    f"{net:+.3f}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── Three-tab detailed view ────────────────────────────────────────────────
    tabs = st.tabs(["Sector Indicators", "Timeline", "Influence & Diagnostics"])

    with tabs[0]:
        render_sector_panels(sector_results, theme, advanced=advanced)
        if advanced:
            st.markdown(
                f"<div style='border-top:1px solid {theme.border_default}; "
                f"margin:1rem 0 0.5rem;'></div>",
                unsafe_allow_html=True,
            )
            render_severity_confidence(sector_results, theme, advanced=True)

    with tabs[1]:
        render_sector_timeline(sector_results, theme, advanced=advanced)

    with tabs[2]:
        render_impact_heatmap(sector_results, theme, advanced=advanced)
        st.markdown(
            f"<div style='border-top:1px solid {theme.border_default}; "
            f"margin:1rem 0 0.5rem;'></div>",
            unsafe_allow_html=True,
        )
        render_severity_confidence(sector_results, theme, advanced=advanced)
