"""
Shared Collaboration Room
=========================

Rendered inside all three dashboards (spoke / sector admin / executive) as a
common "Collaboration Room" tab.  Every role sees the same project spaces but
with authority-gated controls:

    INSTITUTION  (spoke)   → post OBSERVATION + attach pipeline analysis
    BASKET_ADMIN (admin)   → post OBSERVATION + ANALYSIS_REQUEST + promote phase
    EXECUTIVE              → all of the above + POLICY_ACTION + force archive

Projects visible to each role:
    spoke   → projects where their basket_id is a participant
    admin   → projects where their basket_id is a participant
    executive → all active projects

Analysis results posted to the stream include the full 10-module pipeline output,
rendered visually inline for every viewer.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from kshiked.ui.institution.backend.project_manager import ProjectManager
from kshiked.ui.institution.backend.models import Role


# ── colour palette shared with dashboards ────────────────────────────────────
_THREAT_COLORS = {
    "CRITICAL": "#BB0000",
    "HIGH": "#E05000",
    "ELEVATED": "#F59E0B",
    "GUARDED": "#2563EB",
    "LOW": "#006600",
}
_UPDATE_COLORS = {
    "OBSERVATION": "#006600",
    "ANALYSIS_REQUEST": "#2563EB",
    "POLICY_ACTION": "#BB0000",
    "ANALYSIS_RESULT": "#007B7B",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _phase_banner(current_phase: str) -> None:
    phases = ["EMERGENCE", "ESCALATION", "STABILIZATION", "RECOVERY"]
    try:
        idx = phases.index(current_phase)
    except ValueError:
        idx = 0
    parts = []
    for i, p in enumerate(phases):
        if i < idx:
            parts.append(f"<span style='color:#aaa;'>~~{p}~~</span>")
        elif i == idx:
            parts.append(f"<span style='background:#1F2937;color:#fff;padding:0 6px;border-radius:4px;font-weight:bold;'>{p}</span>")
        else:
            parts.append(f"<span style='color:#aaa;'>{p}</span>")
    st.markdown(" ➔ ".join(parts), unsafe_allow_html=True)


def _render_analysis_result(content_json: str) -> None:
    """Render a stored pipeline result inline in the stream."""
    try:
        data = json.loads(content_json)
    except Exception:
        st.caption(content_json[:300])
        return

    threat_level = data.get("threat_level", "LOW")
    tc = _THREAT_COLORS.get(threat_level, "#006600")

    # Header
    st.markdown(
        f'<div style="background:{tc}22; border:1px solid {tc}; border-radius:6px; '
        f'padding:0.5rem 0.8rem; margin-bottom:0.5rem;">'
        f'<span style="color:{tc}; font-weight:bold;">Threat Level: {threat_level}</span>'
        f'&nbsp;&nbsp;·&nbsp;&nbsp;'
        f'<span style="color:#555; font-size:0.85rem;">{data.get("narrative", "")[:200]}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Composite scores
    comp = data.get("composite", {})
    if comp:
        _ca, _cb, _cc = st.columns(3)
        _ca.metric("Detection", f"{comp.get('A_Detection', 0)}/10")
        _cb.metric("Impact", f"{comp.get('B_Impact', 0)}/10")
        _cc.metric("Certainty", f"{comp.get('C_Certainty', 0)}/10")

    # Priority alerts
    for alert in data.get("priority_alerts", [])[:3]:
        st.warning(alert)

    # Trend table
    trends = data.get("trend_signals", [])
    if trends:
        _dir_icons = {"acceleration": "📈", "deceleration": "📉", "stable": "➡️"}
        with st.expander("Temporal Trends", expanded=False):
            rows = [{"Variable": t.get("column", ""), "Trend": f"{_dir_icons.get(t.get('direction','stable'),'')} {t.get('direction','').title()}",
                     "Volatility": t.get("volatility", ""), "Growth Rate": f"{t.get('growth_rate', 0):+.2%}"}
                    for t in trends]
            st.dataframe(rows, use_container_width=True, hide_index=True)

    # 8 threat indices
    threat_report = data.get("threat_report")
    if threat_report:
        indices = threat_report.get("indices", {})
        _index_labels = {
            "polarization": "Polarization (PI)",
            "legitimacy_erosion": "Legitimacy Erosion (LEI)",
            "mobilization_readiness": "Mobilization Readiness (MRS)",
            "elite_cohesion": "Elite Cohesion (ECI)",
            "information_warfare": "Information Warfare (IWI)",
            "security_friction": "Security Friction (SFI)",
            "economic_cascade": "Economic Cascade (ECR)",
            "ethnic_tension": "Ethnic Tension (ETM)",
        }
        with st.expander("8 Threat Indices (Pulse Engine)", expanded=False):
            _cols = st.columns(2)
            for _i, (key, label) in enumerate(_index_labels.items()):
                _d = indices.get(key, {})
                _v = _d.get("value") or _d.get("avg_tension", 0.0)
                _s = _d.get("severity", "LOW")
                _sc = _THREAT_COLORS.get(_s, "#888")
                with _cols[_i % 2]:
                    st.markdown(
                        f'<div style="border-left:3px solid {_sc}; padding:0.2rem 0.5rem; margin-bottom:0.3rem;">'
                        f'<span style="font-size:0.8rem; font-weight:600;">{label}</span> '
                        f'<span style="color:{_sc}; font-size:1rem;">{_v:.2f}</span> '
                        f'<span style="color:{_sc}; font-size:0.75rem;">{_s}</span></div>',
                        unsafe_allow_html=True
                    )

    # SFC economic state
    econ = data.get("economic_state")
    if econ:
        with st.expander("Economic State (SFC Model)", expanded=False):
            _e1, _e2, _e3, _e4 = st.columns(4)
            _e1.metric("GDP Growth", f"{econ.get('gdp_growth', 0):.2%}")
            _e2.metric("Inflation", f"{econ.get('inflation', 0):.2%}")
            _e3.metric("Unemployment", f"{econ.get('unemployment', 0):.2%}")
            _e4.metric("Interest Rate", f"{econ.get('interest_rate', 0):.2%}")

    # Relationships
    rels = data.get("relationship_summary", [])
    if rels:
        with st.expander(f"Discovered Relationships ({len(rels)})", expanded=False):
            for r in rels[:8]:
                st.markdown(f"- {r}")

    # Propagation
    prop = data.get("propagation_chains", [])
    if prop:
        with st.expander("Risk Propagation Chain", expanded=False):
            for c in prop:
                st.markdown(
                    f'<div style="background:#FFF8E7; border-left:3px solid #F59E0B; padding:0.4rem 0.7rem; border-radius:0 4px 4px 0;">'
                    f'<b>{c.get("description", "")}</b> — impact: {c.get("estimated_impact", 0):.0%}</div>',
                    unsafe_allow_html=True
                )

    st.caption(f"Engine: {data.get('engine_used', '—')} · Elapsed: {data.get('elapsed_ms', 0):.0f} ms · Confidence: {data.get('overall_confidence', 0):.0%}")


def _render_stream(updates: List[Dict[str, Any]]) -> None:
    """Render the shared causal stream of updates."""
    if not updates:
        st.caption("No contributions to this project yet. Be the first to post an observation.")
        return

    for upd in updates:
        utype = upd.get("update_type", "OBSERVATION")
        color = _UPDATE_COLORS.get(utype, "#555")
        author = upd.get("author_name", "Unknown")
        ts_raw = upd.get("timestamp", 0)
        ts_str = pd.to_datetime(ts_raw, unit='s').strftime('%Y-%m-%d %H:%M') if ts_raw else "—"
        content = upd.get("content", "")

        with st.container():
            st.markdown(
                f'<div style="border-left:3px solid {color}; padding:0.3rem 0.7rem; margin-bottom:0.6rem;">'
                f'<span style="color:{color}; font-weight:bold; font-size:0.8rem;">[{utype}]</span> '
                f'<span style="font-weight:600; font-size:0.85rem;">{author}</span> '
                f'<span style="color:#aaa; font-size:0.75rem;">· {ts_str}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

            if utype == "ANALYSIS_RESULT":
                _render_analysis_result(content)
            else:
                st.write(content)
                cert = upd.get("certainty")
                if cert is not None:
                    st.caption(f"Certainty: {cert:.2f}")

            st.write("")


# ── Main render function ──────────────────────────────────────────────────────

def render_collab_room(
    role: str,
    basket_id: Optional[int],
    username: str,
    all_baskets: Dict[int, str],
) -> None:
    """
    Render the Collaboration Room.

    Args:
        role:        The current user's role value (Role.INSTITUTION.value, etc.)
        basket_id:   Basket ID for spoke/admin, None for executive.
        username:    Current user's username.
        all_baskets: Dict {basket_id: basket_name} for display.
    """
    is_spoke     = role == Role.INSTITUTION.value
    is_admin     = role == Role.BASKET_ADMIN.value
    is_executive = role == Role.EXECUTIVE.value

    st.markdown("### Collaboration Room")
    st.caption(
        "Real-time shared workspace for active operational projects. "
        "All participants — spokes, sector admins, and the executive — see the same stream. "
        "Each role has appropriate contribution rights."
    )

    # ── Role badge ────────────────────────────────────────────────────────────
    _role_label = {"institution": "Spoke Institution", "basket_admin": "Sector Admin", "executive": "National Executive"}.get(role, role)
    _role_color = {"institution": "#006600", "basket_admin": "#1F2937", "executive": "#BB0000"}.get(role, "#555")
    st.markdown(
        f'<div style="display:inline-block; background:{_role_color}22; border:1px solid {_role_color}; '
        f'border-radius:4px; padding:0.2rem 0.6rem; font-size:0.8rem; color:{_role_color}; margin-bottom:0.8rem;">'
        f'You are viewing as: <b>{_role_label}</b></div>',
        unsafe_allow_html=True
    )
    st.write("---")

    # ── Query visible projects ────────────────────────────────────────────────
    query_basket = None if is_executive else basket_id
    projects = ProjectManager.get_active_projects(query_basket)

    if not projects:
        if is_executive:
            st.info("No active operational projects. Launch one from the **Active Operations** tab.")
        elif is_admin:
            st.info("Your sector is not a participant in any active project. Use the **Operational Projects** tab to create one.")
        else:
            st.info(
                "Your institution is not attached to any active cross-sector project. "
                "Your sector admin can include your sector in a project from the **Operational Projects** tab."
            )
        return

    # ── Pipeline result attach (spoke only) ──────────────────────────────────
    pipeline_res = st.session_state.get('pipeline_result')
    attach_options = {p['id']: p['title'] for p in projects}

    if is_spoke and pipeline_res is not None:
        st.markdown("#### Attach Your Analysis to a Project")
        st.caption("Share your 10-module analysis results directly into a project's shared stream.")
        _proj_choice = st.selectbox(
            "Select project to attach analysis to",
            options=list(attach_options.keys()),
            format_func=lambda x: attach_options[x],
            key="collab_attach_proj"
        )
        if st.button("Attach Analysis to Project Stream", type="primary", key="collab_attach_btn"):
            r = pipeline_res
            analysis_payload = {
                "narrative": r.narrative,
                "threat_level": r.threat_level,
                "priority_alerts": r.priority_alerts,
                "composite": r.composite,
                "trend_signals": r.trend_signals[:10],
                "structural_breaks": r.structural_breaks[:10],
                "relationship_summary": r.relationship_summary[:8],
                "threat_report": r.threat_report,
                "economic_state": r.economic_state,
                "propagation_chains": r.propagation_chains,
                "overall_confidence": r.overall_confidence,
                "engine_used": r.engine_used,
                "elapsed_ms": r.elapsed_ms,
                "sensitivity": st.session_state.get('data_sensitivity', 'Public'),
            }
            ProjectManager.add_update(
                project_id=_proj_choice,
                author_name=username,
                update_type="ANALYSIS_RESULT",
                content=json.dumps(analysis_payload),
                certainty=r.overall_confidence,
            )
            st.success(f"Analysis attached to project '{attach_options[_proj_choice]}' and visible to all participants.")
            st.rerun()
        st.write("---")

    # ── Print each project ────────────────────────────────────────────────────
    for proj in projects:
        p_id = proj['id']
        phase = proj.get('current_phase', 'EMERGENCE')
        severity = proj.get('severity', 1)
        sev_color = "#BB0000" if severity >= 4 else "#F59E0B" if severity >= 2.5 else "#006600"

        with st.expander(
            f"{'🔴' if severity >= 4 else '🟠' if severity >= 2.5 else '🟢'}  "
            f"{proj['title']}  —  Phase: {phase}  —  Severity: {severity}",
            expanded=(len(projects) == 1)
        ):
            _phase_banner(phase)
            st.write("")

            project_data = ProjectManager.get_project_details(p_id)
            participants = project_data.get('participants', [])
            updates = project_data.get('updates', [])

            _left, _center, _right = st.columns([1, 2.5, 1.2])

            # ── LEFT: participants & metadata ─────────────────────────────
            with _left:
                st.markdown("**Participants**")
                for b in participants:
                    is_self = (b == basket_id)
                    label = all_baskets.get(b, f"Sector {b}")
                    if is_self:
                        st.markdown(f"- **{label} ← you**")
                    else:
                        st.markdown(f"- {label}")
                st.write("")
                st.caption(f"Created: {pd.to_datetime(proj['created_at'], unit='s').strftime('%Y-%m-%d %H:%M')}")
                st.caption(f"Updates: {len(updates)}")
                st.write("")

                # Disagreement / consensus
                dis = ProjectManager.get_disagreement_matrix(p_id)
                if dis:
                    st.markdown("**Sector Certainty**")
                    df_dis = pd.DataFrame(list(dis.items()), columns=["Sector", "Certainty"])
                    fig_dis = px.bar(df_dis, x="Sector", y="Certainty", range_y=[0, 1],
                                     color="Certainty", color_continuous_scale="RdYlGn",
                                     height=150)
                    fig_dis.update_layout(margin=dict(l=0, r=0, t=0, b=30), showlegend=False,
                                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_dis, use_container_width=True, key=f"collab_dis_{p_id}_{role}")

            # ── CENTRE: shared causal stream ──────────────────────────────
            with _center:
                st.markdown("**Shared Activity Stream**")
                _render_stream(updates)

            # ── RIGHT: post contribution ──────────────────────────────────
            with _right:
                st.markdown("**Post to Stream**")

                # Determine available update types by role
                if is_executive:
                    _types = ["OBSERVATION", "ANALYSIS_REQUEST", "POLICY_ACTION"]
                elif is_admin:
                    _types = ["OBSERVATION", "ANALYSIS_REQUEST"]
                else:  # spoke
                    _types = ["OBSERVATION"]

                _utype = st.selectbox("Type", _types, key=f"collab_utype_{p_id}")
                _ucontent = st.text_area(
                    "Message",
                    height=120,
                    placeholder="Describe what you observed, request analysis, or post a policy directive...",
                    key=f"collab_text_{p_id}"
                )
                _cert = None
                if _utype == "OBSERVATION":
                    _cert = st.slider("Your Certainty", 0.0, 1.0, 0.5, 0.05, key=f"collab_cert_{p_id}")

                if st.button("Post to Project Stream", type="primary", key=f"collab_post_{p_id}", use_container_width=True):
                    if _ucontent.strip():
                        ProjectManager.add_update(
                            project_id=p_id,
                            author_name=username,
                            update_type=_utype,
                            content=_ucontent,
                            certainty=_cert,
                        )
                        st.rerun()
                    else:
                        st.error("Cannot post an empty message.")

                st.write("---")

                # Phase controls — admin and executive only
                if is_admin or is_executive:
                    phases = ["EMERGENCE", "ESCALATION", "STABILIZATION", "RECOVERY"]
                    try:
                        p_idx = phases.index(phase)
                    except ValueError:
                        p_idx = 0
                    if p_idx < len(phases) - 1:
                        if st.button(
                            f"Advance → {phases[p_idx + 1]}",
                            key=f"collab_phase_{p_id}",
                            use_container_width=True
                        ):
                            ProjectManager.transition_phase(p_id, phases[p_idx + 1])
                            st.rerun()

                # Force archive — executive only
                if is_executive:
                    with st.expander("Force Archive"):
                        _res_state = st.selectbox(
                            "Resolution", ['RESOLVED', 'FALSE_ALARM', 'INSUFFICIENT_EVIDENCE', 'CONFLICTING_SIGNALS'],
                            key=f"collab_res_{p_id}"
                        )
                        _pol_score = st.slider("Policy Effectiveness", 0.0, 10.0, 5.0, 0.5, key=f"collab_pscore_{p_id}")
                        _debrief = st.text_area("Debrief", key=f"collab_debrief_{p_id}")
                        if st.button("Archive & Calibrate", type="primary", key=f"collab_arch_{p_id}", use_container_width=True):
                            if _debrief:
                                ProjectManager.archive_project(
                                    p_id, _res_state, _pol_score, _debrief,
                                    {"archived_from": "collab_room", "participants": len(participants)}
                                )
                                st.rerun()
                            else:
                                st.error("Debrief summary required.")

    # ── Closed projects (archive preview) ────────────────────────────────────
    memories = ProjectManager.get_institutional_memory()
    closed = [m for m in memories if m.get('project_id')]
    if closed:
        st.write("---")
        with st.expander(f"Closed Projects in Archive ({len(closed)})", expanded=False):
            for mem in closed[:10]:
                eff = mem.get('policy_effectiveness_score', 0)
                eff_color = "#006600" if eff >= 7 else "#F59E0B" if eff >= 4 else "#BB0000"
                st.markdown(
                    f'<div style="border-left:3px solid {eff_color}; padding:0.3rem 0.6rem; margin-bottom:0.4rem;">'
                    f'<b style="font-size:0.85rem;">Project #{mem.get("project_id")}</b> — '
                    f'{mem.get("resolution_state", "—")} — '
                    f'Effectiveness: <span style="color:{eff_color};">{eff:.1f}/10</span><br>'
                    f'<span style="font-size:0.8rem; color:#555;">{mem.get("resolution_summary", "")[:200]}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
