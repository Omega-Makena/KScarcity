import streamlit as st
import sys
import os
import re
import pandas as pd
import numpy as np
import json
from pathlib import Path
import time
from html import escape

project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
  sys.path.insert(0, project_root)

from kshiked.ui.institution.backend.auth import enforce_role, logout_user
from kshiked.ui.institution.backend.models import Role
from kshiked.ui.institution.backend.executive_bridge import ExecutiveBridge
from kshiked.ui.institution.backend.delta_sync import DeltaSyncManager
from kshiked.ui.institution.backend.project_manager import ProjectManager
from kshiked.ui.institution.backend.database import get_connection
from kshiked.ui.institution.style import inject_enterprise_theme, get_base64_of_bin_file
from kshiked.ui.institution.backend.messaging import SecureMessaging
from kshiked.ui.institution.collab_room import render_collab_room
from kshiked.ui.institution.executive_simulator import render_executive_simulator
from kshiked.ui.institution.backend.data_sharing import DataSharingManager
from kshiked.ui.institution.backend.analytics_engine import (
  generate_inaction_projection,
  get_historical_context,
  build_county_convergence,
  generate_recommendation,
  compute_outcome_impact,
  get_county_centroid,
)
from kshiked.ui.institution.backend.report_narrator import (
  narrate_risk_for_executive,
  narrate_severity,
  get_threat_index_explanation,
)
from kshiked.ui.institution.backend.model_quality import build_quality_assurance_snapshot
from kshiked.ui.institution.shared_sidebar import render_shared_sidebar
from kshiked.ui.theme import LIGHT_THEME as theme


EXEC_COLORS = {
  "black": "#1a1a1a",
  "red": "#BB0000",
  "red_light": "#F9EDED",
  "green": "#006600",
  "green_light": "#EAF3E0",
  "white": "#FFFFFF",
  "surface": "#F8F7F5",
  "border": "rgba(26,26,26,0.12)",
  "border_strong": "rgba(26,26,26,0.25)",
  "text_muted": "#6B6B6B",
  "text_faint": "#9E9E9E",
}


EXEC_NAV_ITEMS = [
  ("intelligence", "National Briefing", "national-briefing", EXEC_COLORS["red"], None),
  ("intelligence", "Threat Intelligence", "threat-intelligence", EXEC_COLORS["red"], None),
  ("intelligence", "Social Signals", "social-signals", EXEC_COLORS["red"], None),
  ("intelligence", "National Map", "national-map", EXEC_COLORS["red"], None),
  ("sectors", "Sector Reports", "sector-reports", EXEC_COLORS["green"], "pending"),
  ("sectors", "Sector Summaries", "sector-summaries", EXEC_COLORS["green"], None),
  ("command", "Active Operations", "active-operations", EXEC_COLORS["black"], None),
  ("command", "Command & Control", "command-control", EXEC_COLORS["black"], None),
  ("command", "Policy Simulator", "policy-simulator", EXEC_COLORS["black"], None),
  ("command", "Collaboration Room", "collaboration-room", EXEC_COLORS["black"], None),
  ("command", "Archive", "archive", EXEC_COLORS["black"], None),
]


EXEC_NAV_KEY_TO_SECTION = {
  "national-briefing": "National Briefing",
  "threat-intelligence": "Threat Intelligence",
  "social-signals": "Social Signals",
  "national-map": "National Map",
  "sector-reports": "Sector Reports",
  "sector-summaries": "Sector Summaries",
  "active-operations": "Active Operations",
  "command-control": "Command & Control",
  "policy-simulator": "Policy Simulator",
  "collaboration-room": "Collaboration Room",
  "archive": "Archive",
}

AGENCY_APPROVALS_FILE = Path(__file__).resolve().parent / "agency_approvals.json"
AGENCY_APPROVAL_AUDIT_FILE = Path(__file__).resolve().parent / "agency_approval_audit.jsonl"


def _load_agency_pending_requests() -> list[dict]:
  if not AGENCY_APPROVALS_FILE.exists():
    return []
  try:
    with open(AGENCY_APPROVALS_FILE, "r", encoding="utf-8") as f:
      data = json.load(f)
    pending = data.get("pending", {}) if isinstance(data, dict) else {}
    rows = []
    for node_id, payload in pending.items():
      rows.append({
        "node_id": str(node_id),
        "institution_name": str(payload.get("institution_name", "")),
        "domain": str(payload.get("domain", "")),
        "requested_at": float(payload.get("requested_at", 0.0) or 0.0),
      })
    return sorted(rows, key=lambda r: r.get("requested_at", 0.0), reverse=True)
  except Exception:
    return []


def _load_agency_approval_audit(limit: int = 25) -> list[dict]:
  if not AGENCY_APPROVAL_AUDIT_FILE.exists():
    return []
  rows = []
  try:
    with open(AGENCY_APPROVAL_AUDIT_FILE, "r", encoding="utf-8") as f:
      for line in f:
        line = str(line).strip()
        if not line:
          continue
        try:
          event = json.loads(line)
        except json.JSONDecodeError:
          continue
        rows.append({
          "timestamp": float(event.get("timestamp", 0.0) or 0.0),
          "action": str(event.get("action", "")),
          "node_id": str(event.get("node_id", "")),
          "actor": str(event.get("actor", "")),
          "domain": str((event.get("details") or {}).get("domain", "")),
        })
  except Exception:
    return []

  rows = sorted(rows, key=lambda r: r.get("timestamp", 0.0), reverse=True)
  return rows[: max(1, int(limit))]


def _render_agency_onboarding_snapshot() -> None:
  pending_rows = _load_agency_pending_requests()
  audit_rows = _load_agency_approval_audit(limit=20)

  with st.expander("Agency Onboarding Queue", expanded=False):
    domain_options = sorted({str(r.get("domain", "")) for r in pending_rows if str(r.get("domain", ""))})
    selected_domain = st.selectbox(
      "Filter Domain",
      options=["All"] + domain_options,
      key="exec_onboarding_domain_filter",
    )

    if selected_domain != "All":
      pending_rows = [r for r in pending_rows if str(r.get("domain", "")) == selected_domain]
      audit_rows = [r for r in audit_rows if str(r.get("domain", "")) == selected_domain]

    c1, c2, c3 = st.columns(3)
    c1.metric("Pending Requests", len(pending_rows))
    approved_24h = sum(
      1
      for r in audit_rows
      if r.get("action") == "APPROVED" and (time.time() - float(r.get("timestamp", 0.0))) <= 86400
    )
    rejected_24h = sum(
      1
      for r in audit_rows
      if r.get("action") == "REJECTED" and (time.time() - float(r.get("timestamp", 0.0))) <= 86400
    )
    c2.metric("Approved (24h)", approved_24h)
    c3.metric("Rejected (24h)", rejected_24h)

    if pending_rows:
      df_pending = pd.DataFrame(pending_rows)
      if "requested_at" in df_pending.columns:
        df_pending["requested_at"] = pd.to_datetime(df_pending["requested_at"], unit="s", errors="coerce")
      st.dataframe(df_pending, use_container_width=True, hide_index=True)
      st.download_button(
        "Export Pending CSV",
        data=df_pending.to_csv(index=False).encode("utf-8"),
        file_name="agency_pending_requests.csv",
        mime="text/csv",
        key="exec_export_pending_agencies_csv",
      )
    else:
      st.caption("No pending agency onboarding requests.")

    st.markdown("##### Recent Approval Audit")
    if audit_rows:
      df_audit = pd.DataFrame(audit_rows)
      if "timestamp" in df_audit.columns:
        df_audit["timestamp"] = pd.to_datetime(df_audit["timestamp"], unit="s", errors="coerce")
      st.dataframe(df_audit, use_container_width=True, hide_index=True)
      st.download_button(
        "Export Audit CSV",
        data=df_audit.to_csv(index=False).encode("utf-8"),
        file_name="agency_approval_audit.csv",
        mime="text/csv",
        key="exec_export_agency_audit_csv",
      )
    else:
      st.caption("No approval audit actions yet.")


def _render_executive_assurance_snapshot(snapshot: dict | None = None) -> None:
  if snapshot is None:
    snapshot = build_quality_assurance_snapshot()
  overall = snapshot.get("overall_assurance", {}) if isinstance(snapshot, dict) else {}
  metric = snapshot.get("metric_credibility", {}) if isinstance(snapshot, dict) else {}
  robustness = snapshot.get("robustness", {}) if isinstance(snapshot, dict) else {}
  traceability = snapshot.get("traceability", {}) if isinstance(snapshot, dict) else {}
  deployability = snapshot.get("deployment_realism", {}) if isinstance(snapshot, dict) else {}
  drg_allocator = deployability.get("dynamic_resource_allocator", {}) if isinstance(deployability.get("dynamic_resource_allocator"), dict) else {}
  summary_rows = snapshot.get("summary_rows", []) if isinstance(snapshot.get("summary_rows"), list) else []

  with st.expander("Model Assurance Snapshot", expanded=False):
    st.caption("Credibility vs baseline, robustness under drift/partial failure, and audit traceability.")

    light = str(overall.get("traffic_light", "amber")).lower()
    light_color = "#006600" if light == "green" else "#b54708" if light == "amber" else "#BB0000"
    light_label = "GREEN" if light == "green" else "AMBER" if light == "amber" else "RED"
    note = str(overall.get("note", ""))
    st.markdown(
      (
        f"<div style='background:#FFFFFF; border:0.5px solid rgba(26,26,26,0.12); border-left:4px solid {light_color}; "
        f"border-radius:8px; padding:10px 12px; margin-bottom:8px;'>"
        f"<div style='font-size:0.78rem; color:#6B6B6B; text-transform:uppercase;'>Overall Assurance Verdict</div>"
        f"<div style='font-size:1.02rem; font-weight:600; color:{light_color}; margin-top:2px;'>"
        f"{light_label} | {float(overall.get('score', 0.0) or 0.0) * 100:.0f}%"
        f"</div>"
        f"<div style='font-size:0.84rem; color:#4B5563; margin-top:4px;'>{escape(note)}</div>"
        f"</div>"
      ),
      unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
      "Metric Credibility",
      f"{float(metric.get('score', 0.0) or 0.0) * 100:.0f}%",
      str(metric.get("band", "n/a")),
    )
    c2.metric(
      "Robustness",
      f"{float(robustness.get('score', 0.0) or 0.0) * 100:.0f}%",
      str(robustness.get("band", "n/a")),
    )
    c3.metric(
      "Traceability",
      f"{float(traceability.get('score', 0.0) or 0.0) * 100:.0f}%",
      str(traceability.get("band", "n/a")),
    )
    c4.metric(
      "Deployment Realism",
      f"{float(deployability.get('score', 0.0) or 0.0) * 100:.0f}%",
      str(deployability.get("band", "n/a")),
    )
    c5.metric(
      "DRG Allocator",
      f"{float(drg_allocator.get('score', 0.0) or 0.0) * 100:.0f}%",
      str(drg_allocator.get("activity_score", "n/a")),
    )

    with st.expander("Why this score?", expanded=False):
      st.caption("Formal scoring equations and weighted contributions used in this assurance verdict.")
      overall_formula = str(overall.get("formula") or "")
      if overall_formula:
        st.code(overall_formula, language="text")

      export_payload = {
        "overall_assurance": overall,
        "metric_credibility": metric,
        "robustness": robustness,
        "traceability": traceability,
        "deployment_realism": deployability,
        "summary_rows": summary_rows,
      }
      ex1, ex2, ex3 = st.columns(3)
      ex1.download_button(
        "Export Explainability JSON",
        data=json.dumps(export_payload, ensure_ascii=True, indent=2).encode("utf-8"),
        file_name="assurance_explainability_executive.json",
        mime="application/json",
        key="exec_assurance_explainability_json_export",
      )

      components = overall.get("components", []) if isinstance(overall.get("components"), list) else []
      if components:
        df_components = pd.DataFrame(components)
        if "weight" in df_components.columns:
          df_components["weight_pct"] = df_components["weight"].astype(float) * 100.0
        if "score" in df_components.columns:
          df_components["score_pct"] = df_components["score"].astype(float) * 100.0
        if "contribution" in df_components.columns:
          df_components["contribution_pct"] = df_components["contribution"].astype(float) * 100.0
        st.dataframe(df_components, use_container_width=True, hide_index=True)
        ex2.download_button(
          "Export Components CSV",
          data=df_components.to_csv(index=False).encode("utf-8"),
          file_name="assurance_components_executive.csv",
          mime="text/csv",
          key="exec_assurance_components_csv_export",
        )

      def _render_breakdown(title: str, obj: dict) -> None:
        st.markdown(f"##### {title}")
        formula = str(obj.get("formula") or "")
        if formula:
          st.code(formula, language="text")
        parts = obj.get("score_breakdown", {}) if isinstance(obj.get("score_breakdown"), dict) else {}
        if parts:
          rows = []
          for name, payload in parts.items():
            if not isinstance(payload, dict):
              continue
            weight = float(payload.get("weight", 0.0) or 0.0)
            value = float(payload.get("value", 0.0) or 0.0)
            rows.append(
              {
                "signal": str(name),
                "weight": weight,
                "value": value,
                "contribution": weight * value,
                "formula": str(payload.get("formula") or ""),
              }
            )
          if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

      _render_breakdown("Metric Credibility Formula", metric)
      _render_breakdown("Robustness Formula", robustness)
      _render_breakdown("Deployment Realism Formula", deployability)

      breakdown_rows = []
      for criterion_name, criterion_payload in (
        ("metric_credibility", metric),
        ("robustness", robustness),
        ("deployment_realism", deployability),
        ("traceability", traceability),
      ):
        parts = criterion_payload.get("score_breakdown", {}) if isinstance(criterion_payload.get("score_breakdown"), dict) else {}
        if not parts and criterion_name == "traceability":
          parts = criterion_payload.get("transparency_breakdown", {}) if isinstance(criterion_payload.get("transparency_breakdown"), dict) else {}
        for signal_name, payload in parts.items():
          if not isinstance(payload, dict):
            continue
          w = float(payload.get("weight", 0.0) or 0.0)
          v = float(payload.get("value", 0.0) or 0.0)
          breakdown_rows.append(
            {
              "criterion": criterion_name,
              "signal": str(signal_name),
              "weight": w,
              "value": v,
              "contribution": w * v,
              "raw_count": int(float(payload.get("raw_count", 0) or 0)),
              "formula": str(payload.get("formula") or ""),
            }
          )
      if breakdown_rows:
        ex3.download_button(
          "Export Signal Breakdown CSV",
          data=pd.DataFrame(breakdown_rows).to_csv(index=False).encode("utf-8"),
          file_name="assurance_signal_breakdown_executive.csv",
          mime="text/csv",
          key="exec_assurance_breakdown_csv_export",
        )

    baseline_rows = metric.get("baseline_rows", []) if isinstance(metric.get("baseline_rows"), list) else []
    if baseline_rows:
      st.markdown("##### Baseline Deltas")
      st.dataframe(pd.DataFrame(baseline_rows), use_container_width=True, hide_index=True)

    r1, r2, r3 = st.columns(3)
    r1.metric("Drift Detect Rate", f"{float(robustness.get('detect_rate', 0.0) or 0.0):.2f}")
    r2.metric("Drift Split Rate", f"{float(robustness.get('split_rate', 0.0) or 0.0):.2f}")
    r3.metric("Fallback Signals", int(robustness.get("fallback_signals", 0) or 0))

    t1, t2 = st.columns(2)
    t1.metric("Override Events", int(traceability.get("override_events", 0) or 0))
    t2.metric("Decision Artifacts", int(traceability.get("decision_artifacts", 0) or 0))

    docs = traceability.get("documentation_paths", []) if isinstance(traceability.get("documentation_paths"), list) else []
    if docs:
      st.caption("Audit and explanation documentation")
      st.code("\n".join(docs), language="text")

    transparency_breakdown = traceability.get("transparency_breakdown", {}) if isinstance(traceability.get("transparency_breakdown"), dict) else {}
    if transparency_breakdown:
      st.markdown("##### Transparency Breakdown")
      rows = []
      for name, payload in transparency_breakdown.items():
        if not isinstance(payload, dict):
          continue
        weight = float(payload.get("weight", 0.0) or 0.0)
        value = float(payload.get("value", 0.0) or 0.0)
        rows.append(
          {
            "signal": str(name),
            "weight": weight,
            "value": value,
            "contribution": weight * value,
            "raw_count": int(float(payload.get("raw_count", 0) or 0)),
            "formula": str(payload.get("formula") or ""),
          }
        )
      if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    recent_overrides = traceability.get("recent_override_samples", []) if isinstance(traceability.get("recent_override_samples"), list) else []
    if recent_overrides:
      st.markdown("##### Recent Override Decisions")
      df_overrides = pd.DataFrame(recent_overrides)
      if "created_at" in df_overrides.columns:
        df_overrides["created_at"] = pd.to_datetime(df_overrides["created_at"], unit="s", errors="coerce")
      st.dataframe(df_overrides, use_container_width=True, hide_index=True)

    assumptions = deployability.get("assumptions", []) if isinstance(deployability.get("assumptions"), list) else []
    if assumptions:
      st.caption("Deployment assumptions")
      st.code("\n".join(assumptions), language="text")

    if drg_allocator:
      st.markdown("##### Dynamic Resource Allocator Signals")
      d1, d2, d3, d4 = st.columns(4)
      d1.metric("Readiness", f"{float(drg_allocator.get('readiness_score', 0.0) or 0.0) * 100:.0f}%")
      d2.metric("Activity", f"{float(drg_allocator.get('activity_score', 0.0) or 0.0) * 100:.0f}%")
      d3.metric("Runtime Files", int(drg_allocator.get("runtime_activity_files", 0) or 0))
      d4.metric("Keyword Hits", int(drg_allocator.get("runtime_keyword_hits", 0) or 0))

    if summary_rows:
      st.markdown("##### Assurance Criteria Summary")
      st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


@st.cache_data(show_spinner=False)
def _get_gok_logo_b64() -> str:
  logo_path = Path(project_root) / "GOK.png"
  if not logo_path.exists():
    return ""
  return get_base64_of_bin_file(str(logo_path))


def _inject_executive_shell_css():
  logo_b64 = _get_gok_logo_b64()
  if logo_b64:
    background_css = f"""
      .stApp {{
        background-image: url("data:image/png;base64,{logo_b64}") !important;
        background-size: 360px auto !important;
        background-repeat: no-repeat !important;
        background-position: center calc(50% + 92px) !important;
        background-color: {EXEC_COLORS['surface']} !important;
      }}

      [data-testid="stAppViewContainer"] {{
        background-color: rgba(248, 247, 245, 0.94) !important;
      }}
    """
  else:
    background_css = f"""
      .stApp {{
        background: {EXEC_COLORS['surface']} !important;
      }}
    """

  st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');

      html, body, [class*="css"], .stApp {{
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 400 !important;
      }}

      [data-testid="stMetricValue"],
      [data-testid="stMetricDelta"],
      .exec-strain-value,
      .exec-num,
      .stNumberInput input,
      input[type="number"],
      .js-plotly-plot .xtick text,
      .js-plotly-plot .ytick text,
      .js-plotly-plot .hovertext {{
        font-family: 'IBM Plex Mono', monospace !important;
        font-variant-numeric: tabular-nums;
      }}

      {background_css}

      .main .block-container {{
        padding-top: 20px !important;
        padding-left: 24px !important;
        padding-right: 24px !important;
        padding-bottom: 24px !important;
      }}

      [data-testid="stSidebar"] {{
        background: {EXEC_COLORS['white']} !important;
        width: 210px !important;
        min-width: 210px !important;
        max-width: 210px !important;
        border-right: 0.5px solid {EXEC_COLORS['border']} !important;
      }}

      [data-testid="stSidebar"] > div:first-child {{
        background: {EXEC_COLORS['white']} !important;
      }}

      [data-testid="stSidebar"] .block-container {{
        padding-top: 0.75rem !important;
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-bottom: 0.75rem !important;
      }}

      [data-testid="stSidebar"] .stButton > button {{
        width: 100% !important;
        text-align: left !important;
        border-radius: 8px !important;
        border: 0.5px solid {EXEC_COLORS['border_strong']} !important;
        background: {EXEC_COLORS['white']} !important;
        color: {EXEC_COLORS['black']} !important;
        box-shadow: none !important;
        font-size: 13px !important;
        font-weight: 400 !important;
        letter-spacing: 0.1px !important;
        padding: 0.42rem 0.55rem !important;
        min-height: 34px !important;
      }}

      [data-testid="stSidebar"] .stButton > button:hover {{
        border-color: {EXEC_COLORS['red']} !important;
        color: {EXEC_COLORS['black']} !important;
        background: {EXEC_COLORS['red_light']} !important;
      }}

      [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        background: {EXEC_COLORS['red_light']} !important;
        color: {EXEC_COLORS['red']} !important;
        font-weight: 500 !important;
        border-color: {EXEC_COLORS['border']} !important;
        border-left: 3px solid {EXEC_COLORS['red']} !important;
      }}

      [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
        color: {EXEC_COLORS['red']} !important;
        border-left: 3px solid {EXEC_COLORS['red']} !important;
      }}

      [data-testid="stSidebar"] .stButton:last-of-type > button {{
        border: none !important;
        background: transparent !important;
        color: {EXEC_COLORS['text_faint']} !important;
        font-size: 11px !important;
        padding: 2px 0 !important;
        min-height: auto !important;
      }}

      [data-testid="stSidebar"] .stButton:last-of-type > button:hover {{
        background: transparent !important;
        color: {EXEC_COLORS['red']} !important;
      }}

      .exec-topbar {{
        background: {EXEC_COLORS['black']};
        border: 0.5px solid {EXEC_COLORS['border_strong']};
        border-radius: 12px;
        min-height: 56px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        padding: 0 14px;
        margin-bottom: 10px;
      }}

      .exec-topbar-left {{
        display: flex;
        align-items: center;
        gap: 6px;
        min-width: 72px;
      }}

      .exec-topbar-dot {{
        width: 5px;
        height: 5px;
        border-radius: 50%;
        display: inline-block;
      }}

      .exec-topbar-center {{
        color: {EXEC_COLORS['white']};
        flex: 1;
        text-align: center;
        font-size: 15px;
        font-weight: 500;
        line-height: 1.25;
      }}

      .exec-topbar-actions {{
        display: flex;
        align-items: center;
        gap: 8px;
      }}

      .exec-btn {{
        border-radius: 8px;
        padding: 5px 10px;
        font-size: 12px;
        font-weight: 500;
        min-width: 72px;
        text-align: center;
        background: transparent;
      }}

      .exec-btn-stop {{
        border: 0.5px solid rgba(187,0,0,0.6);
        color: {EXEC_COLORS['red']};
      }}

      .exec-btn-deploy {{
        border: 0.5px solid {EXEC_COLORS['white']};
        color: {EXEC_COLORS['white']};
      }}

      .exec-command-header {{
        background: {EXEC_COLORS['black']};
        border: 0.5px solid {EXEC_COLORS['border_strong']};
        border-left: 3px solid {EXEC_COLORS['red']};
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 14px;
      }}

      .exec-command-label {{
        color: {EXEC_COLORS['text_faint']};
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-weight: 500;
      }}

      .exec-command-title {{
        color: {EXEC_COLORS['white']};
        font-size: 22px;
        font-weight: 500;
        margin: 3px 0;
      }}

      .exec-command-subtitle {{
        color: #F3C9C9;
        font-size: 13px;
      }}

      .exec-strain-wrap {{
        margin-top: 10px;
        border: 0.5px solid {EXEC_COLORS['border_strong']};
        border-radius: 8px;
        padding: 8px 10px;
        background: rgba(255,255,255,0.03);
      }}

      .exec-strain-head {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
      }}

      .exec-strain-label {{
        color: #D1D5DB;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        font-weight: 500;
      }}

      .exec-strain-value {{
        color: {EXEC_COLORS['red']};
        font-size: 12px;
        font-weight: 500;
      }}

      .exec-strain-track {{
        width: 100%;
        height: 8px;
        border-radius: 999px;
        border: 0.5px solid {EXEC_COLORS['border_strong']};
        background: rgba(255,255,255,0.08);
        overflow: hidden;
      }}

      .exec-strain-fill {{
        height: 100%;
        border-radius: 999px;
        background: {EXEC_COLORS['red']};
      }}

      .hero-brief {{
        background: {EXEC_COLORS['black']};
        border: 0.5px solid {EXEC_COLORS['border_strong']};
        border-left: 3px solid {EXEC_COLORS['red']};
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 14px;
      }}

      .hero-brief h3 {{
        color: {EXEC_COLORS['white']};
        font-weight: 500;
        margin: 0 0 8px 0;
      }}

      .hero-brief p {{
        color: #F3C9C9;
        line-height: 1.8;
        margin: 0;
      }}

      .hero-brief b {{
        color: {EXEC_COLORS['white']};
        font-weight: 500;
      }}

      .hero-brief i {{
        color: #F3C9C9;
      }}

      [data-testid="stVerticalBlockBorderWrapper"] {{
        border: 0.5px solid {EXEC_COLORS['border']} !important;
        border-radius: 12px !important;
        background: {EXEC_COLORS['white']} !important;
      }}

      .stButton > button, .stDownloadButton > button {{
        border-radius: 8px !important;
        border: 0.5px solid {EXEC_COLORS['border_strong']} !important;
        box-shadow: none !important;
      }}
    </style>
    """,
    unsafe_allow_html=True,
  )


def _compute_national_strain(global_risks):
  strain_score = min(10.0, len(global_risks) * 2.5)
  if strain_score > 7.0:
    return strain_score, "CRITICAL", EXEC_COLORS["red"]
  if strain_score > 3.0:
    return strain_score, "MODERATE", "#E05000"
  return strain_score, "LOW", EXEC_COLORS["green"]


def _apply_plotly_numeric_font(fig):
  fig.update_layout(
    font=dict(family="IBM Plex Sans, sans-serif"),
    hoverlabel=dict(font=dict(family="IBM Plex Mono, monospace")),
  )
  for axis_name in [
    "xaxis", "xaxis2", "xaxis3", "xaxis4", "xaxis5",
    "yaxis", "yaxis2", "yaxis3", "yaxis4", "yaxis5",
  ]:
    axis = getattr(fig.layout, axis_name, None)
    if axis is not None:
      axis.tickfont = dict(family="IBM Plex Mono, monospace")
  return fig


def _render_exec_content_chrome(
  active_section: str,
  strain_score: float,
  strain_label: str,
  strain_color: str,
  topbar_title: str = "Presidency Intelligence & Coordinated Response",
  dashboard_label: str = "National Executive Dashboard",
  command_subtitle: str = "Presidency Intelligence & Coordinated Response",
):
  strain_pct = max(0.0, min(100.0, (strain_score / 10.0) * 100.0))
  st.markdown(
    f"""
    <div class="exec-topbar">
      <div class="exec-topbar-left">
        <span class="exec-topbar-dot" style="background:{EXEC_COLORS['green']};"></span>
        <span class="exec-topbar-dot" style="background:{EXEC_COLORS['red']};"></span>
        <span class="exec-topbar-dot" style="background:{EXEC_COLORS['white']}; border:0.5px solid {EXEC_COLORS['border_strong']};"></span>
      </div>
      <div class="exec-topbar-center">{escape(topbar_title)}</div>
      <div class="exec-topbar-actions">
        <button class="exec-btn exec-btn-stop" type="button">Stop</button>
        <button class="exec-btn exec-btn-deploy" type="button">Deploy</button>
      </div>
    </div>
    <div class="exec-command-header">
      <div class="exec-command-label">{escape(dashboard_label)}</div>
      <div class="exec-command-title">{escape(active_section)}</div>
      <div class="exec-command-subtitle">{escape(command_subtitle)}</div>
      <div class="exec-strain-wrap">
        <div class="exec-strain-head">
          <span class="exec-strain-label">National Systemic Strain</span>
          <span class="exec-strain-value" style="color:{strain_color};">{escape(strain_label)} - {strain_score:.1f} / 10</span>
        </div>
        <div class="exec-strain-track">
          <div class="exec-strain-fill" style="width:{strain_pct:.1f}%; background:{strain_color};"></div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
  )


def _tail_file_lines(file_path: Path, max_lines: int = 120) -> str:
  try:
    with file_path.open("r", encoding="utf-8", errors="replace") as fh:
      lines = fh.readlines()
    return "".join(lines[-max_lines:])
  except Exception as exc:
    return f"Failed to read log file: {exc}"


def _render_developer_only_panels(
  active_projects,
  global_risks,
  unread_escs: int,
):
  st.markdown("### Developer Console")

  with st.expander("Debug Controls", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
      st.session_state["dev_show_raw_payloads"] = st.toggle(
        "Show Raw Payloads",
        value=bool(st.session_state.get("dev_show_raw_payloads", False)),
      )
    with c2:
      st.session_state["dev_trace_queries"] = st.toggle(
        "Trace Queries",
        value=bool(st.session_state.get("dev_trace_queries", False)),
      )
    with c3:
      st.session_state["dev_disable_cache"] = st.toggle(
        "Disable Cache",
        value=bool(st.session_state.get("dev_disable_cache", False)),
      )
    with c4:
      if st.button("Clear Cache", use_container_width=True, key="dev_clear_cache"):
        st.cache_data.clear()
        st.success("Streamlit cache cleared.")

  with st.expander("Model Diagnostics", expanded=False):
    d1, d2, d3, d4 = st.columns(4)
    with d1:
      st.metric("Active Projects", len(active_projects))
    with d2:
      st.metric("Promoted Risks", len(global_risks))
    with d3:
      st.metric("Unread Escalations", unread_escs)

    try:
      with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) AS c FROM users")
        users_count = int(c.fetchone()["c"])
        c.execute("SELECT COUNT(*) AS c FROM institutions")
        institutions_count = int(c.fetchone()["c"])
        c.execute(
          "SELECT timestamp, analysis_type, username, role FROM analysis_history ORDER BY timestamp DESC LIMIT 5"
        )
        recent_analysis = [dict(r) for r in c.fetchall()]
    except Exception:
      users_count = 0
      institutions_count = 0
      recent_analysis = []

    with d4:
      st.metric("Registered Users", users_count)
    st.caption(f"Institutions tracked: {institutions_count}")

    if recent_analysis:
      st.dataframe(pd.DataFrame(recent_analysis), use_container_width=True)
    else:
      st.caption("No recent analysis history found.")

  with st.expander("Internal Logs", expanded=False):
    log_root = Path(project_root) / "logs"
    log_files = []
    if log_root.exists():
      for p in log_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".log", ".txt", ".json"}:
          log_files.append(p)
    log_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    if not log_files:
      st.caption("No internal log files found under logs/.")
      return

    options = {str(p.relative_to(project_root)): p for p in log_files[:50]}
    selected_rel = st.selectbox("Log File", list(options.keys()), key="dev_log_file")
    selected_path = options[selected_rel]

    tail_text = _tail_file_lines(selected_path, max_lines=150)
    st.code(tail_text, language="text")
    st.caption(f"Showing tail of {selected_rel}")

def _resolve_risk_scores(risk: dict) -> dict:
  """Normalize composite_scores regardless of whether they come from the
  demo seeder (keys: severity, composite_risk, confidence, trend) or from
  the analysis pipeline (keys: A_Detection, B_Impact, C_Certainty).
  Returns a dict always containing B_Impact, A_Detection, C_Certainty."""
  raw = risk.get('composite_scores', {})
  if isinstance(raw, str):
    import json as _j
    try:
      raw = _j.loads(raw)
    except Exception:
      raw = {}
  if not raw:
    return {'B_Impact': 0.0, 'A_Detection': 0.0, 'C_Certainty': 0.0}
  # Pipeline schema
  if 'B_Impact' in raw or 'A_Detection' in raw:
    impact = float(raw.get('B_Impact', raw.get('A_Detection', 0.0)))
    return {
      'B_Impact': impact,
      'A_Detection': float(raw.get('A_Detection', impact)),
      'C_Certainty': float(raw.get('C_Certainty', 0.0)),
    }
  # Demo-seeder schema: composite_risk or severity are 0-1 floats â€” scale to /10
  composite = float(raw.get('composite_risk', raw.get('severity', 0.0)))
  confidence = float(raw.get('confidence', raw.get('trend', 0.8)))
  return {
    'B_Impact':  round(composite * 10, 2),
    'A_Detection': round(composite * 10, 2),
    'C_Certainty': round(confidence * 10, 2),
  }


@st.cache_data(ttl=3600)
def load_and_process_geojson(geojson_path, county_scores=None):
  import json
  with open(geojson_path, "r", encoding="utf-8") as f:
    geojson_data = json.load(f)

  for feature in geojson_data['features']:
    county_name = feature['properties'].get('shapeName', '').lower()

    # Use real data if available, otherwise show zero (honest absence)
    if county_scores and county_name in county_scores:
      stress = int(county_scores[county_name]['score'])
      has_data = True
    else:
      stress = 0
      has_data = False

    feature['properties']['stress'] = stress
    feature['properties']['has_data'] = has_data

    if not has_data:
      r, g, b = 148, 163, 184 # Slate gray â€” no data
    elif stress > 80:
      r, g, b = 239, 68, 68  # Red
    elif stress > 50:
      r, g, b = 245, 158, 11 # Amber
    elif stress > 20:
      r, g, b = 59, 130, 246 # Blue
    else:
      r, g, b = 16, 185, 129 # Green

    alpha = 140 if has_data else 60
    feature['properties']['color'] = [r, g, b, alpha]
    feature['properties']['line_color'] = [r, g, b, 255]
    feature['properties']['elevation'] = stress * 1200 if has_data else 200

  return geojson_data

@st.cache_data(ttl=300)
def cached_simulate_policy_shock(target_idx, magnitude, steps):
  return ExecutiveBridge.simulate_policy_shock(target_idx, magnitude, steps=steps)

@st.cache_data(ttl=3600)
def get_pulse_data():
  path = os.path.join(project_root, "data", "synthetic_kenya_policy", "tweets.csv")
  if not os.path.exists(path):
    return pd.DataFrame()
  cols = [
    "timestamp",
    "text",
    "intent",
    "topic_cluster",
    "location_county",
    "sentiment_score",
    "threat_score",
    "imperative_rate",
    "urgency_rate",
    "coordination_score",
    "escalation_score",
    "policy_event_id",
    "policy_phase",
    "policy_severity",
  ]
  try:
    df = pd.read_csv(path, usecols=cols).dropna(subset=['intent'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    df.rename(columns={
      'topic_cluster': 'Sector',
      'intent': 'Threat Category',
      'sentiment_score': 'Sentiment',
      'timestamp': 'Timestamp',
      'location_county': 'County',
      'text': 'Text',
    }, inplace=True)
    
    for col in (
      "Sentiment",
      "threat_score",
      "imperative_rate",
      "urgency_rate",
      "coordination_score",
      "escalation_score",
      "policy_severity",
    ):
      if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df['Sector'] = df['Sector'].fillna("General Systemic")
    df['Sentiment'] = df['Sentiment'].clip(lower=0.0, upper=1.0)
    df['threat_score'] = df['threat_score'].clip(lower=0.0, upper=1.0)
    df['Criticality'] = (df['threat_score'] * 8.0) + ((1.0 - df['Sentiment']) * 2.0)
    df['Criticality'] = df['Criticality'].clip(lower=0.0, upper=10.0).round(2)
    df["Scenario Relevance"] = 0.0
    df["Scenario Link"] = "generic_stream"
    
    return df
  except Exception as e:
    st.error(f"Failed to load pulse data: {e}")
    return pd.DataFrame()


def _to_tokens(value: str) -> set:
  text = str(value or "").lower()
  text = re.sub(r"[^a-z0-9\s]+", " ", text)
  return {tok for tok in text.split() if len(tok) >= 3}


def _extract_simulation_context(global_risks, all_baskets, active_projects) -> list[dict]:
  context: list[dict] = []
  for risk in global_risks or []:
    basket_id = risk.get("basket_id")
    sector_name = str((all_baskets or {}).get(basket_id, "General Systemic"))
    title = str(risk.get("title", ""))
    description = str(risk.get("description", ""))
    composite = _resolve_risk_scores(risk)
    severity = float(composite.get("B_Impact", 0.0) or 0.0) / 10.0
    tokens = _to_tokens(f"{title} {description} {sector_name}")
    if not tokens:
      continue
    context.append(
      {
        "sector": sector_name,
        "threat_label": title[:64] if title else "simulated_risk",
        "severity": max(0.0, min(1.0, severity)),
        "tokens": tokens,
      }
    )

  for project in active_projects or []:
    title = str(project.get("title", ""))
    description = str(project.get("description", ""))
    phase = str(project.get("current_phase", ""))
    severity = float(project.get("severity", 0.0) or 0.0) / 10.0
    tokens = _to_tokens(f"{title} {description} {phase}")
    if not tokens:
      continue
    context.append(
      {
        "sector": "Cross-Sector Operations",
        "threat_label": title[:64] if title else "operational_project",
        "severity": max(0.0, min(1.0, severity)),
        "tokens": tokens,
      }
    )

  return context


def _enrich_pulse_with_simulated_challenges(df: pd.DataFrame, global_risks, all_baskets, active_projects) -> pd.DataFrame:
  if df.empty:
    return df

  sim_context = _extract_simulation_context(global_risks, all_baskets, active_projects)
  if not sim_context:
    out = df.copy()
    out["Scenario Relevance"] = out.get("Scenario Relevance", 0.0)
    out["Scenario Link"] = out.get("Scenario Link", "generic_stream")
    return out

  negative_terms = {
    "fear", "panic", "unsafe", "danger", "violence", "attack", "hunger", "shortage", "riot", "crisis",
    "kill", "dead", "threat", "flee", "urgent", "collapse", "corrupt", "outbreak", "cholera", "flood",
  }
  positive_terms = {"safe", "stable", "calm", "recover", "resolved", "normal", "support", "peace", "secure"}

  out = df.copy()
  sentiments: list[float] = []
  threats: list[float] = []
  sectors: list[str] = []
  categories: list[str] = []
  criticalities: list[float] = []
  relevances: list[float] = []
  links: list[str] = []

  for row in out.to_dict(orient="records"):
    raw_text = str(row.get("Text") or "")
    raw_intent = str(row.get("Threat Category") or "")
    raw_sector = str(row.get("Sector") or "").strip()
    token_set = _to_tokens(f"{raw_text} {raw_intent} {raw_sector}")

    best_ctx = None
    best_overlap = 0.0
    for ctx in sim_context:
      overlap = len(token_set & ctx["tokens"]) / max(1.0, min(10.0, float(len(ctx["tokens"]))))
      if overlap > best_overlap:
        best_overlap = overlap
        best_ctx = ctx

    sentiment = float(row.get("Sentiment", 0.0) or 0.0)
    threat = float(row.get("threat_score", 0.0) or 0.0)
    urgency = float(row.get("urgency_rate", 0.0) or 0.0)
    escalation = float(row.get("escalation_score", 0.0) or 0.0)
    imperative = float(row.get("imperative_rate", 0.0) or 0.0)
    coordination = float(row.get("coordination_score", 0.0) or 0.0)
    policy_severity = float(row.get("policy_severity", 0.0) or 0.0)

    neg_hits = len(token_set & negative_terms)
    pos_hits = len(token_set & positive_terms)

    if sentiment <= 0.0:
      sentiment = 0.55 + (0.06 * pos_hits) - (0.08 * neg_hits)

    structural_threat = 0.38 * urgency + 0.30 * escalation + 0.20 * imperative + 0.12 * coordination
    lexical_threat = min(0.45, 0.06 * neg_hits)
    threat = max(threat, structural_threat + lexical_threat + (0.08 * policy_severity))

    relevance = best_overlap
    if best_ctx is not None:
      threat = min(1.0, threat + (0.22 * best_overlap) + (0.20 * float(best_ctx.get("severity", 0.0) or 0.0)))
      if raw_sector.lower() in {"", "general systemic", "nan"}:
        raw_sector = str(best_ctx.get("sector") or "General Systemic")
      if raw_intent.lower() in {"", "casual", "opinion", "satire_mockery"} and best_overlap >= 0.12:
        raw_intent = str(best_ctx.get("threat_label") or raw_intent)

    sentiment = min(1.0, max(0.0, sentiment - (0.22 * threat) + (0.02 * pos_hits)))
    relevance = min(1.0, max(0.0, 0.55 * relevance + 0.30 * threat + 0.15 * (1.0 - sentiment)))
    criticality = min(10.0, max(0.0, 10.0 * (0.55 * threat + 0.30 * (1.0 - sentiment) + 0.15 * relevance)))

    sentiments.append(round(float(sentiment), 3))
    threats.append(round(float(threat), 3))
    sectors.append(raw_sector or "General Systemic")
    categories.append(raw_intent or "unclassified")
    criticalities.append(round(float(criticality), 2))
    relevances.append(round(float(relevance), 3))
    links.append("simulated_challenge" if relevance >= 0.2 else "generic_stream")

  out["Sentiment"] = sentiments
  out["threat_score"] = threats
  out["Sector"] = sectors
  out["Threat Category"] = categories
  out["Criticality"] = criticalities
  out["Scenario Relevance"] = relevances
  out["Scenario Link"] = links
  return out


def render(mode: str = "executive"):
  enforce_role(Role.EXECUTIVE.value)
  inject_enterprise_theme()
  _inject_executive_shell_css()

  mode_name = str(mode or "executive").strip().lower()
  is_developer = mode_name == "developer"
  nav_state_key = "developer_nav" if is_developer else "executive_nav"
  nav_button_prefix = "developer_nav" if is_developer else "executive_nav"
  disconnect_key = "developer_disconnect" if is_developer else "executive_disconnect"

  profile_label = "Developer Command" if is_developer else "National Executive"
  profile_name = (
    st.session_state.get("username", "Developer Dashboard")
    if is_developer
    else "National Dashboard"
  )
  role_badge = "DEVELOPER" if is_developer else "EXECUTIVE"
  topbar_title = (
    "Developer Intelligence & Experiment Control"
    if is_developer
    else "Presidency Intelligence & Coordinated Response"
  )
  dashboard_label = "Dedicated Developer Dashboard" if is_developer else "National Executive Dashboard"
  command_subtitle = (
    "Full-fidelity executive stack in dedicated developer mode"
    if is_developer
    else "Presidency Intelligence & Coordinated Response"
  )
  
  # Fetch all baskets
  with get_connection() as conn:
    c = conn.cursor()
    c.execute("SELECT id, name FROM baskets")
    all_baskets = {r['id']: r['name'] for r in c.fetchall()}
    
  # Global state queries for Top-Level Metrics
  active_projects = ProjectManager.get_active_projects(None)
  global_risks = DeltaSyncManager.get_promoted_risks()
  memories = ProjectManager.get_institutional_memory()

  # Pre-fetch inbox so unread count is available everywhere
  esc_inbox_global = SecureMessaging.get_inbox(Role.EXECUTIVE.value, "ALL")
  unread_escs = sum(1 for m in esc_inbox_global if not m['is_read'])
  assurance_snapshot = build_quality_assurance_snapshot()

  sidebar_state = render_shared_sidebar(
    state_key=nav_state_key,
    default_key="national-briefing",
    nav_items=EXEC_NAV_ITEMS,
    group_order=["intelligence", "sectors", "command"],
    group_labels={
      "intelligence": ("Intelligence", EXEC_COLORS["red"]),
      "sectors": ("Sectors", EXEC_COLORS["green"]),
      "command": ("Command", EXEC_COLORS["black"]),
    },
    profile_label=profile_label,
    profile_name=profile_name,
    role_badge_text=role_badge,
    role_badge_bg=EXEC_COLORS["black"],
    role_badge_fg=EXEC_COLORS["white"],
    role_badge_border=EXEC_COLORS["black"],
    profile_bottom_border=EXEC_COLORS["red"],
    button_key_prefix=nav_button_prefix,
    badge_counts={"sector-reports": len(global_risks)},
    disconnect_button_key=disconnect_key,
    disconnect_label="Disconnect session",
  )

  overall_assurance = assurance_snapshot.get("overall_assurance", {}) if isinstance(assurance_snapshot, dict) else {}
  light = str(overall_assurance.get("traffic_light", "amber")).lower()
  light_color = "#006600" if light == "green" else "#b54708" if light == "amber" else "#BB0000"
  light_label = "GREEN" if light == "green" else "AMBER" if light == "amber" else "RED"
  st.sidebar.markdown(
    (
      f"<div style='margin-top:8px; border:0.5px solid rgba(26,26,26,0.2); border-left:3px solid {light_color}; "
      f"border-radius:7px; padding:7px 8px; background:#FFFFFF;'>"
      f"<div style='font-size:0.68rem; color:#6B6B6B; text-transform:uppercase; letter-spacing:0.4px;'>Assurance</div>"
      f"<div style='font-size:0.84rem; font-weight:600; color:{light_color};'>"
      f"{light_label} | {float(overall_assurance.get('score', 0.0) or 0.0) * 100:.0f}%"
      f"</div>"
      f"</div>"
    ),
    unsafe_allow_html=True,
  )

  if sidebar_state["disconnect_clicked"]:
    logout_user()
    st.rerun()

  active_key = str(sidebar_state["active_key"])
  if sidebar_state["changed"]:
    st.session_state[nav_state_key] = active_key
    st.rerun()

  active_section = EXEC_NAV_KEY_TO_SECTION.get(active_key, "National Briefing")
  strain_score, strain_label, strain_color = _compute_national_strain(global_risks)
  _render_exec_content_chrome(
    active_section,
    strain_score,
    strain_label,
    strain_color,
    topbar_title=topbar_title,
    dashboard_label=dashboard_label,
    command_subtitle=command_subtitle,
  )

  if is_developer:
    _render_developer_only_panels(
      active_projects=active_projects,
      global_risks=global_risks,
      unread_escs=unread_escs,
    )

  if active_section == "National Briefing":
    st.markdown("### National Briefing")
    _render_agency_onboarding_snapshot()
    _render_executive_assurance_snapshot(assurance_snapshot)
    df_social = _enrich_pulse_with_simulated_challenges(get_pulse_data(), global_risks, all_baskets, active_projects)
    avg_sent = df_social['Sentiment'].mean() if not df_social.empty else 0.50
    linked_share = float((df_social.get("Scenario Relevance", pd.Series(dtype=float)) >= 0.2).mean()) if not df_social.empty else 0.0
    
    # 1. THE EXECUTIVE BRIEFING (Hero Section)
    # Dynamic Morning Narrative
    brief_text = f"National Systemic Strain is currently <b>{strain_label.title()}</b> ({strain_score:.1f}/10). "
    if global_risks:
      primary = global_risks[0]
      brief_text += f"The primary driver is an escalating threat involving <i>{primary['title']}</i> within the <b>{all_baskets.get(primary['basket_id'], 'unknown')}</b> sector. "
      if len(global_risks) > 1:
        brief_text += f"There are {len(global_risks) - 1} secondary anomalies detected across the topography. "
      
    if active_projects:
      most_severe_proj = sorted(active_projects, key=lambda x: x['severity'], reverse=True)[0]
      brief_text += f"<br><br>The administration is currently tracking <b>{len(active_projects)} active National Projects</b>. "
      brief_text += f"Immediate executive oversight is required on Project <i>{most_severe_proj['title']}</i> (Severity: {most_severe_proj['severity']}, Phase: {most_severe_proj['current_phase']}). "
    else:
      brief_text += "<br><br>There are currently no active cross-sector Operational Projects requiring executive coordination."
      
    if unread_escs > 0:
      brief_text += f" <span style='color: #BB0000; font-weight: bold;'>You have {unread_escs} unread escalations awaiting Command & Control clearance.</span>"
    
    st.markdown(f"""
      <div class="hero-brief">
        <h3>Executive Summary</h3>
        <p>{brief_text}</p>
      </div>
    """, unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Active Threat Signals", len(global_risks))
    with m2: st.metric("Systemic Strain", f"{strain_score:.1f}/10")
    with m3: st.metric("Active Projects", len(active_projects))
    with m4: st.metric("National Sentiment", f"{avg_sent:.2f}", f"Scenario-linked {linked_share * 100:.0f}%")

    st.markdown("#### National Economic Health")
    _impact_values = [float(_resolve_risk_scores(r).get('B_Impact', 0.0)) for r in global_risks]
    _avg_impact = (sum(_impact_values) / len(_impact_values)) if _impact_values else 0.0
    _ops_pressure = min(10.0, len(active_projects) * 1.25)
    _sentiment_risk = (1.0 - float(avg_sent)) * 10.0
    _stability_score = max(0.0, min(10.0, 10.0 - (0.5 * _avg_impact + 0.3 * _ops_pressure + 0.2 * _sentiment_risk)))
    _econ_status = "Stable" if _stability_score >= 7.0 else "Watch" if _stability_score >= 4.5 else "Stressed"
    _econ_color = "#006600" if _econ_status == "Stable" else "#F59E0B" if _econ_status == "Watch" else "#BB0000"

    st.markdown(
      f"""
      <div style="background:#FFFFFF; border:0.5px solid rgba(26,26,26,0.12); border-left:4px solid {_econ_color}; border-radius:8px; padding:12px 14px; margin-top:8px; margin-bottom:8px;">
        <div style="font-size:0.8rem; color:#6B6B6B; text-transform:uppercase; letter-spacing:0.6px;">Executive Macro Health</div>
        <div style="font-size:1.05rem; color:#1a1a1a; font-weight:600; margin-top:2px;">National status: {_econ_status} ({_stability_score:.1f}/10)</div>
        <div style="font-size:0.86rem; color:#4B5563; margin-top:6px; line-height:1.45;">
          Composite view from live systemic impact, active operations pressure, and national sentiment telemetry.
          Higher stress implies tighter policy response windows.
        </div>
      </div>
      """,
      unsafe_allow_html=True,
    )

    _eh1, _eh2, _eh3 = st.columns(3)
    with _eh1:
      st.metric("Average Threat Impact", f"{_avg_impact:.1f}/10")
    with _eh2:
      st.metric("Operations Pressure", f"{_ops_pressure:.1f}/10")
    with _eh3:
      st.metric("Sentiment Risk", f"{_sentiment_risk:.1f}/10")
      
    st.write("---")
    
    # Layout for Visualizations & Top Priorities
    bc1, bc2 = st.columns([1.5, 1])
    
    with bc1:
      st.markdown("#### Strategic Threat Distribution")
      if global_risks:
        import plotly.express as px
        threat_data = []
        for r in global_risks:
          threat_data.append({
            "Sector": all_baskets.get(r['basket_id'], 'Unknown'),
            "Impact": _resolve_risk_scores(r)['B_Impact'] or 5.0,
            "Title": r['title']
          })
        df_threats = pd.DataFrame(threat_data)
        fig_bar = px.bar(df_threats, x="Sector", y="Impact", color="Impact", 
                 title="Cumulative Threat Impact by Sector",
                 color_continuous_scale="Reds", template="plotly_white")
        fig_bar.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0))
        _apply_plotly_numeric_font(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)
      else:
        st.success("No active threats.")
        
      st.markdown("#### National Projects by Phase")
      if active_projects:
        proj_data = [{"Phase": p['current_phase'], "Title": p['title']} for p in active_projects]
        df_proj = pd.DataFrame(proj_data)
        fig_donut = px.pie(df_proj, names="Phase", hole=0.5, 
                  title="Operational Projects Distribution",
                  color_discrete_sequence=px.colors.sequential.Greens_r, template="plotly_white")
        fig_donut.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0))
        _apply_plotly_numeric_font(fig_donut)
        st.plotly_chart(fig_donut, use_container_width=True)
      else:
        st.success("No active projects.")
        
    with bc2:
      st.markdown("#### Top Priorities")
      st.write("Immediate Attention Required:")
      
      if unread_escs > 0:
        st.markdown(f"""
        <div style="background: #FEF2F2; border-left: 4px solid #EF4444; padding: 12px; margin-bottom: 10px; border-radius: 4px;">
          <strong>{unread_escs} Unread Escalations</strong>
          <div style="font-size: 0.8rem; color: #7F1D1D;">Check Command & Control Inbox</div>
        </div>
        """, unsafe_allow_html=True)
        
      if active_projects:
        top_p = sorted(active_projects, key=lambda x: x['severity'], reverse=True)[0]
        st.markdown(f"""
        <div style="background: #FFFBEB; border-left: 4px solid #F59E0B; padding: 12px; margin-bottom: 10px; border-radius: 4px;">
          <strong>Project: {top_p['title']}</strong>
          <div style="font-size: 0.8rem; color: #92400E;">Phase: {top_p['current_phase']} | Severity: {top_p['severity']}</div>
        </div>
        """, unsafe_allow_html=True)
        
      if global_risks:
        top_r = sorted(global_risks, key=lambda x: _resolve_risk_scores(x)['B_Impact'], reverse=True)[0]
        st.markdown(f"""
        <div style="background: #F0FDF4; border-left: 4px solid #10B981; padding: 12px; margin-bottom: 10px; border-radius: 4px;">
          <strong>Signal: {top_r['title']}</strong>
          <div style="font-size: 0.8rem; color: #065F46;">Sector: {all_baskets.get(top_r['basket_id'], 'unknown')}</div>
        </div>
        """, unsafe_allow_html=True)
      
    # â”€â”€ Causal Intelligence â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    st.write("---")
    with st.expander(" Causal Intelligence â€” evidence from historical data", expanded=False):
      st.caption(
        "Structural causal estimates using archived World Bank national indicators. "
        "Defaults are pre-selected from the most common causal pathways. "
        "These are historical estimates, independent of the live simulation."
      )
      try:
        from kshiked.ui.kshield.causal import render_causal_evidence_panel, load_world_bank_data
        from kshiked.ui.theme import LIGHT_THEME as _exec_theme
        _wb = load_world_bank_data()
        if not _wb.empty:
          _avail = [c for c in _wb.columns if _wb[c].notna().sum() >= 15]
          _auto_t, _auto_o = _avail[0], _avail[1]
          for _t_kw, _o_kw in [
            ("inflation", "unemployment"), ("gdp", "poverty"),
            ("interest rate", "inflation"), ("exports", "gdp"),
          ]:
            _t_m = [c for c in _avail if _t_kw.lower() in c.lower()]
            _o_m = [c for c in _avail if _o_kw.lower() in c.lower()]
            if _t_m and _o_m:
              _auto_t, _auto_o = _t_m[0], _o_m[0]
              break
          _ei1, _ei2 = st.columns(2)
          with _ei1:
            _exec_t = st.selectbox(
              "Indicator (cause)", _avail,
              index=_avail.index(_auto_t), key="exec_causal_t",
            )
          with _ei2:
            _o_opts = [c for c in _avail if c != _exec_t]
            _o_def = _o_opts.index(_auto_o) if _auto_o in _o_opts else 0
            _exec_o = st.selectbox(
              "Indicator (effect)", _o_opts,
              index=_o_def, key="exec_causal_o",
            )
          render_causal_evidence_panel(
            df=_wb, treatment=_exec_t, outcome=_exec_o,
            theme=_exec_theme, key_prefix="exec_brief",
          )
        else:
          st.info("Historical causal dataset not loaded yet in this session.")
      except Exception as _ex:
        st.caption(f"Causal intelligence unavailable: {_ex}")

  # --- SECTOR REPORTS ---
  if active_section == "Sector Reports":
    st.markdown("### Sector Command Overview")
    st.caption("Live status of all active national sectors. Each sector admin reports validated risks upward to this view.")

    # Pre-fetch spoke counts per basket
    spoke_counts = {}
    try:
      with get_connection() as _conn:
        _cc = _conn.cursor()
        _cc.execute("SELECT basket_id, COUNT(*) as n FROM institutions GROUP BY basket_id")
        for _rr in _cc.fetchall():
          spoke_counts[_rr['basket_id']] = _rr['n']
    except Exception:
      pass

    _SECTOR_ICONS = {
      "Public Health": "",
      "Water & Sanitation": "",
      "Transport & Logistics": "",
      "Security & Border": "",
      "Displacement & IDP": "",
      "Food & Markets": "",
      "Communications & Information": "",
    }

    # â”€â”€ SECTION 1: STATUS GRID (always shows ALL sectors) â”€â”€
    basket_list = list(all_baskets.items())
    for row_start in range(0, len(basket_list), 3):
      cols = st.columns(3)
      for col_idx, (b_id, b_name) in enumerate(basket_list[row_start : row_start + 3]):
        s_risks = [r for r in global_risks if r.get('basket_id') == b_id]
        top_imp = max((_resolve_risk_scores(r)['B_Impact'] for r in s_risks), default=0.0)
        n_spokes = spoke_counts.get(b_id, 0)
        icon   = _SECTOR_ICONS.get(b_name, "")
        if top_imp >= 8:
          cborder = "#BB0000"; slabel = "CRITICAL"; sbg = "#BB0000"
        elif top_imp >= 6:
          cborder = "#E05000"; slabel = "HIGH";   sbg = "#E05000"
        elif top_imp >= 4:
          cborder = "#F59E0B"; slabel = "ELEVATED"; sbg = "#F59E0B"
        elif s_risks:
          cborder = "#3B82F6"; slabel = "ACTIVE";  sbg = "#3B82F6"
        else:
          cborder = "#10B981"; slabel = "CLEAR";  sbg = "#10B981"
        risk_tag = f"&nbsp;|&nbsp; {round(top_imp,1)}/10" if s_risks else ""
        cols[col_idx].markdown(
          f'<div style="background:#F8FAFC; border-radius:10px; border-top:4px solid {cborder}; '
          f'padding:14px 16px; margin-bottom:12px; min-height:100px;">'
          f'<div style="font-size:1.05rem; font-weight:700; color:#1F2937; margin-bottom:4px;">{icon} {b_name}</div>'
          f'<div style="margin-bottom:6px;"><span style="background:{sbg}; color:#fff; padding:2px 8px; '
          f'border-radius:4px; font-size:0.75rem; font-weight:600;">{slabel}</span></div>'
          f'<div style="font-size:0.79rem; color:#475569;">'
          f' {n_spokes} spoke{"s" if n_spokes != 1 else ""} &nbsp;|&nbsp; '
          f'{len(s_risks)} risk{"s" if len(s_risks) != 1 else ""} promoted{risk_tag}</div></div>',
          unsafe_allow_html=True
        )

    st.write("---")
    st.markdown("#### Validated Risks â€” All Sectors")
    st.markdown("#### Promoted Risks â€” All Sectors")
    # â”€â”€ SECTION 2: DETAILED RISK CARDS (one expander per sector, all visible) â”€â”€
    if True: # always enter â€” no early-exit guard

      for b_id, b_name in all_baskets.items():
        sector_risks = [r for r in global_risks if r.get('basket_id') == b_id]
        total_impact = sum(_resolve_risk_scores(r)['B_Impact'] for r in sector_risks)
        icon     = _SECTOR_ICONS.get(b_name, "")
        with st.expander(
          f"{icon} {b_name} | {len(sector_risks)} risk(s)"
          + (f" | Composite: {total_impact:.1f}/10" if sector_risks else " | CLEAR"),
          expanded=len(sector_risks) > 0
        ):
          if not sector_risks:
            st.success("No promoted risks from this sector. All clear.")
          else:
            for risk in sector_risks:
              scores  = _resolve_risk_scores(risk)
              impact  = scores['B_Impact']
              detection = scores['A_Detection']
              certainty = scores['C_Certainty']
              ts    = pd.to_datetime(risk.get('timestamp', 0), unit='s').strftime('%Y-%m-%d %H:%M')
              sev_color = "#BB0000" if impact > 7 else "#E05000" if impact > 5 else "#F59E0B" if impact > 3 else "#059669"
              st.markdown(
                f'<div style="background:{sev_color}; color:#fff; padding:8px 14px; '
                f'border-radius:6px 6px 0 0; font-weight:700; font-size:0.92rem; margin-top:8px;">'
                f'{risk.get("title","Untitled Risk")}</div>',
                unsafe_allow_html=True,
              )
              st.markdown(
                f'<div style="background:#F8FAFC; border:1px solid #E5E7EB; border-top:none; '
                f'padding:12px 16px; border-radius:0 0 6px 6px; font-size:0.88rem; line-height:1.6; margin-bottom:4px;">'
                f'{risk.get("description","")}</div>',
                unsafe_allow_html=True,
              )
              st.markdown(
                f'<div style="padding:4px 0 4px 2px; font-size:0.8rem; color:#475569;">'
                f'<b>Detection</b> {detection:.1f}/10 &middot; <b>Impact</b> {impact:.1f}/10 &middot; '
                f'<b>Certainty</b> {certainty:.1f}/10 &nbsp;|&nbsp; {ts}</div>',
                unsafe_allow_html=True,
              )
              projection = generate_inaction_projection(
                severity=impact / 10.0, incident_type='PROMOTED_RISK', composite_scores=scores,
              )
              if projection:
                st.markdown(
                  f'<div style="background:#FEF2F2; border-left:4px solid #DC2626; padding:8px 12px; '
                  f'border-radius:0 6px 6px 0; margin:4px 0; font-size:0.84rem;">'
                  f'<strong>So What?</strong> {projection}</div>',
                  unsafe_allow_html=True,
                )
              hist_ctx = get_historical_context(
                basket_id=b_id, severity=impact / 10.0, incident_type='PROMOTED_RISK',
              )
              st.markdown(
                f'<div style="background:#F0F9FF; border-left:4px solid #3B82F6; padding:8px 12px; '
                f'border-radius:0 6px 6px 0; margin:4px 0 8px 0; font-size:0.84rem;">'
                f'<strong>Compared to What?</strong> {hist_ctx}</div>',
                unsafe_allow_html=True,
              )
              rec = generate_recommendation(
                risk=risk, all_baskets=all_baskets, global_risks=global_risks,
              )
              st.markdown(
                f'<div style="background:#FFFBEB; border-left:4px solid {rec.level_color}; '
                f'padding:8px 12px; border-radius:0 6px 6px 0; margin:0 0 12px 0;">'
                f'<strong><span style="background:{rec.level_color}; color:#fff; padding:2px 8px; '
                f'border-radius:4px; font-size:0.78rem;">{rec.level}</span></strong> '
                f'<span style="font-size:0.84rem;">{rec.summary}</span><br/>'
                f'<span style="font-size:0.80rem; color:#64748b;">'
                f'<b>Who:</b> {", ".join(rec.who[:3])} &nbsp;|&nbsp; <b>Urgency:</b> {rec.urgency}</span>'
                f'</div>',
                unsafe_allow_html=True,
              )

  if active_section == "National Map":
    st.markdown("### National Map")
    st.write("Geospatial distribution of emerging hotspots across Kenyan counties â€” driven by real promoted risk data.")
    import pydeck as pdk

    # â”€â”€ PILLAR 3: "WHERE EXACTLY?" â€” Real geographic data â”€â”€
    county_scores = build_county_convergence(global_risks, all_baskets)

    if not county_scores:
      st.info(
        "**Geographic metadata unavailable.** No promoted risks contain "
        "county-level geographic data. The map will activate once risks "
        "with spatial metadata are promoted by sector admins."
      )

    # Load local Kenyan counties GeoJSON with real data
    geojson_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "kenya_adm1_simplified.geojson")
    try:
      geojson_data = load_and_process_geojson(geojson_path, county_scores or None)

      layer = pdk.Layer(
        "GeoJsonLayer",
        geojson_data,
        pickable=True,
        stroked=True,
        filled=True,
        extruded=True,
        wireframe=True,
        get_fill_color="properties.color",
        get_line_color="properties.line_color",
        get_elevation="properties.elevation",
        elevation_scale=1,
        line_width_min_pixels=2,
      )

      view_state = pdk.ViewState(
        latitude=0.0236,
        longitude=37.9062,
        zoom=5.2,
        pitch=55,
        bearing=15
      )

      tooltip_html = (
        "<b>County:</b> {shapeName} <br/> "
        "<b>Convergence Score:</b> {stress}/100 <br/>"
        "<b>Data:</b> {'Real signal data' if {has_data} else 'No data'}"
      )

      r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={
          "html": "<b>County:</b> {shapeName} <br/> <b>Convergence Score:</b> {stress}/100",
          "style": {
            "backgroundColor": "#ffffff",
            "color": "#0f172a",
            "border": "1px solid #e2e8f0",
            "borderRadius": "8px",
            "boxShadow": "0 4px 6px -1px rgba(0,0,0,0.1)",
            "fontFamily": "IBM Plex Sans, sans-serif",
            "fontWeight": "500",
            "padding": "10px"
          }
        },
        map_style="mapbox://styles/mapbox/light-v11"
      )
      st.pydeck_chart(r, height=450, use_container_width=True)

      # Legend
      if county_scores:
        data_counties = len(county_scores)
        total_risks = sum(c['risk_count'] for c in county_scores.values())
        st.caption(
          f"Low (&lt;20) &nbsp; Moderate (20-50) &nbsp; Elevated (50-80) &nbsp; Critical (&gt;80) &nbsp; No data \n\n"
          f"**{data_counties} counties** with signal data from **{total_risks} promoted risks**."
        )
      else:
        st.caption("All counties shown in grey â€” no active signal data.")

    except Exception as e:
      st.error(f"Error loading geospatial topography: {e}")
  
  if active_section == "Threat Intelligence":
    st.markdown("### Threat Intelligence")
    st.write("Validated risks promoted by sector admins. These have been reviewed and assessed at the sector level before reaching this dashboard.")
    if not global_risks:
      st.success("No active systemic signals detected.")
    else:
      st.markdown("<br>", unsafe_allow_html=True)
      for risk in global_risks:
        # 1. Normalize Scores
        scores = risk.get('composite_scores', {})
        if isinstance(scores, str):
           try:
               scores = json.loads(scores)
           except:
               scores = {}
        
        # Determine metrics (fallback to impact if others missing)
        b_impact = float(scores.get('B_Impact', scores.get('composite_risk', scores.get('severity', 0))) * 10 if scores.get('composite_risk') or scores.get('severity') else scores.get('B_Impact', 0))
        a_detect = float(scores.get('A_Detection', b_impact))
        c_cert = float(scores.get('C_Certainty', scores.get('confidence', scores.get('trend', 0.8)) * 10 if scores.get('confidence') or scores.get('trend') else scores.get('C_Certainty', 0)))

        # Visual Classing
        sev_class = "cr" if b_impact > 7 else "hi" if b_impact > 4 else "md"
        color_hex = theme.accent_danger if sev_class == "cr" else theme.accent_warning if sev_class == "hi" else theme.accent_primary
        bg_hex = f"{color_hex}11"
        border_hex = f"{color_hex}44"
        
        sector_name = all_baskets.get(risk['basket_id'], f"Sector {risk['basket_id']}")
        title = risk.get('title', 'Uncharacterized Risk Profile')
        detected = pd.to_datetime(risk.get('timestamp', time.time()), unit='s').strftime('%Y-%m-%d %H:%M')

        # Generate Strategic Narratives
        # 1. Inaction Projection
        p_str = generate_inaction_projection(severity=b_impact)
        projection_text = p_str if p_str else "Current severity level projects no immediate critical structural breaches."
        
        # 2. Strategic Recommendation
        rec = generate_recommendation(risk=risk, all_baskets=all_baskets, global_risks=global_risks)
        recommendation_text = f"<span style='color:{rec.level_color}; font-weight:700;'>[{rec.level}]</span> {rec.summary} <br><span style='font-size:0.8rem; color:{theme.text_muted};'>{rec.urgency}</span>"
        
        # Build UI Card
        html = f"""
        <div style="background:{bg_hex}; border:1px solid {border_hex}; border-radius:8px; padding:1.2rem; margin-bottom:1rem; font-family:'IBM Plex Sans', sans-serif;">
          <!-- Header -->
          <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:1rem;">
            <div>
              <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:1px; color:{color_hex}; font-weight:700; margin-bottom:0.2rem;">
                {sector_name}
              </div>
              <div style="font-size:1.1rem; font-weight:600; color:{theme.text_primary};">
                {title}
              </div>
            </div>
            <div style="font-size:0.8rem; color:{color_hex}; font-weight:600; text-align:right;">
              Impact: <span class="exec-num">{b_impact:.1f}/10</span><br>
              <span style="font-size:0.7rem; color:{theme.text_muted}; font-weight:400;">Detected: {detected}</span>
            </div>
          </div>
          
          <!-- Metrics Bar -->
          <div style="display:flex; gap:1.5rem; border-top:1px solid {theme.border_default}; border-bottom:1px solid {theme.border_default}; padding:0.8rem 0; margin-bottom:1rem;">
             <div style="flex:1;">
               <div style="font-size:0.7rem; color:{theme.text_muted}; text-transform:uppercase;">Detection</div>
               <div style="font-size:1.1rem; font-weight:600; color:{theme.text_primary};"><span class="exec-num">{a_detect:.1f}</span><span style="font-size:0.8rem; color:{theme.text_muted};">/10</span></div>
             </div>
             <div style="flex:1;">
               <div style="font-size:0.7rem; color:{theme.text_muted}; text-transform:uppercase;">Impact Profile</div>
               <div style="font-size:1.1rem; font-weight:600; color:{color_hex};"><span class="exec-num">{b_impact:.1f}</span><span style="font-size:0.8rem; color:{theme.text_muted};">/10</span></div>
             </div>
             <div style="flex:1;">
               <div style="font-size:0.7rem; color:{theme.text_muted}; text-transform:uppercase;">Certainty</div>
               <div style="font-size:1.1rem; font-weight:600; color:{theme.text_primary};"><span class="exec-num">{c_cert:.1f}</span><span style="font-size:0.8rem; color:{theme.text_muted};">/10</span></div>
             </div>
          </div>
          
          <!-- Narratives -->
          <div style="display:flex; flex-direction:column; gap:0.8rem;">
            <div>
              <div style="font-size:0.75rem; font-weight:600; color:{theme.text_muted}; margin-bottom:0.2rem;">PROJECTION OF INACTION</div>
              <div style="font-size:0.9rem; color:{theme.text_primary}; line-height:1.4;">{projection_text}</div>
            </div>
            <div>
              <div style="font-size:0.75rem; font-weight:600; color:{theme.text_muted}; margin-bottom:0.2rem;">STRATEGIC RECOMMENDATION</div>
              <div style="font-size:0.9rem; color:{theme.text_primary}; line-height:1.4;">{recommendation_text}</div>
            </div>
          </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
      
  if active_section == "Social Signals":
    st.markdown("### Public Safety & Social Signal Monitor")
    st.write("Executive-level telemetry aligned to active simulated challenges and affected sectors.")
    
    df_social = _enrich_pulse_with_simulated_challenges(get_pulse_data(), global_risks, all_baskets, active_projects)
    
    if df_social.empty:
      st.warning("No real-time social signals available to map.")
    else:
      avg_sent = df_social['Sentiment'].mean()
      linked_share = float((df_social.get("Scenario Relevance", pd.Series(dtype=float)) >= 0.2).mean())
      most_strained = df_social.groupby('Sector')['Criticality'].mean().idxmax()
      
      # 1. Plain-Language translation
      bg_color = "#FEF2F2" if avg_sent < 0.4 else "#F0FDF4" if avg_sent > 0.6 else "#FFFBEB"
      border_color = "#DC2626" if avg_sent < 0.4 else "#10B981" if avg_sent > 0.6 else "#F59E0B"
      sentiment_text = "Highly Negative" if avg_sent < 0.4 else "Positive" if avg_sent > 0.6 else "Mixed / Guarded"
      
      st.markdown(f"""
      <div style="background:{bg_color}; border-left:4px solid {border_color}; padding:15px; border-radius:4px; margin-bottom: 20px;">
        <strong>National Sentiment Verdict: {sentiment_text} (Score: {avg_sent:.2f})</strong><br>
        Based on real-time social streams, the public mood is currently skewed towards the {sentiment_text.lower()} end of the spectrum.
        Scenario-linked coverage is <b>{linked_share * 100:.0f}%</b> and the <b>{most_strained}</b> sector is currently bearing the highest level of social frustration and criticality.
        Focus executive communication and potential interventions in this sector.
      </div>
      """, unsafe_allow_html=True)

      # Filters
      sc1, sc2, sc3 = st.columns(3)
      with sc1:
        sel_sector = st.multiselect("Filter Sector", df_social['Sector'].unique(), default=df_social['Sector'].unique()[:3] if len(df_social['Sector'].unique()) > 3 else df_social['Sector'].unique())
      with sc2:
        sel_threat = st.multiselect("Filter Threat", df_social['Threat Category'].unique(), default=df_social['Threat Category'].unique()[:3] if len(df_social['Threat Category'].unique()) > 3 else df_social['Threat Category'].unique())
      with sc3:
        strict_scenario_mode = st.toggle(
          "Command View (Scenario-linked only)",
          value=True,
          key="exec_social_strict_scenario_mode",
          help="Show only signals tightly linked to active simulated challenges.",
        )
        relevance_threshold = st.slider(
          "Scenario relevance threshold",
          min_value=0.0,
          max_value=1.0,
          value=0.4,
          step=0.05,
          key="exec_social_scenario_relevance_threshold",
        )
        
      df_filtered = df_social[(df_social['Sector'].isin(sel_sector)) & (df_social['Threat Category'].isin(sel_threat))]
      if strict_scenario_mode and "Scenario Relevance" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["Scenario Relevance"] >= float(relevance_threshold)]

      command_linked_share = float((df_filtered.get("Scenario Relevance", pd.Series(dtype=float)) >= float(relevance_threshold)).mean()) if not df_filtered.empty else 0.0
      st.caption(
        f"Filtered signals: {len(df_filtered)} | Linked at threshold: {command_linked_share * 100:.0f}%"
      )
      
      if len(df_filtered) > 3000:
        df_filtered = df_filtered.sample(n=3000, random_state=42)
      
      import plotly.express as px
      if not df_filtered.empty:
        fig = px.scatter(
          df_filtered, x="Timestamp", y="Criticality", 
          color="Sector", size="Criticality", hover_data=["Threat Category", "Sentiment", "Scenario Relevance", "Scenario Link"],
          template="plotly_white", height=300,
          title="Real-Time Signal Criticality Scatter Plot"
        )
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        _apply_plotly_numeric_font(fig)
        st.plotly_chart(fig, use_container_width=True)
        
      st.write("---")
      st.markdown("#### Thematic Threat Tension Vectors")

      if df_filtered.empty:
        st.info("No social signals match the current filters. Lower the relevance threshold or disable strict mode.")
      else:
        tension_series = df_filtered.groupby('Threat Category')['Criticality'].mean().sort_values(ascending=False).head(4)
        t_cols = st.columns(len(tension_series))

        for i, (threat_name, threat_crit) in enumerate(tension_series.items()):
          explanation = get_threat_index_explanation(threat_name.lower().replace(" ", "_"))
          if not explanation:
            explanation = f"Tracks real-time public frustration and tension regarding {threat_name.lower()}."

          t_color = "#BB0000" if threat_crit > 7 else "#E05000" if threat_crit > 5 else "#006600"
          with t_cols[i]:
            st.markdown(f"""
            <div style="border-top:3px solid {t_color}; padding-top:10px;">
              <strong>{threat_name}</strong><br>
              <span style="font-size:1.5rem; color:{t_color}; font-weight:bold;">{threat_crit:.1f}</span><span style="font-size:0.8rem;"> /10</span><br>
              <p style="font-size:0.8rem; color:#475569; margin-top:5px;">{explanation}</p>
            </div>
            """, unsafe_allow_html=True)
        
  if active_section == "Policy Simulator":
    render_executive_simulator()
    
  if active_section == "Active Operations":
    with st.container(border=True):
      from kshiked.ui.institution.project_components import render_project_wizard
      # Executive doesn't have a specific basket ID, so pass 0 to show all baskets
      render_project_wizard(all_baskets, current_basket_id=0)
      
    with st.container(border=True):
      st.info("#### Active Operational Projects")
      st.write("Monitor multi-sector collaboration projects. Inject top-down Policy Actions directly into their intelligence streams.")
      
      if not active_projects:
        st.success("No active cross-sector Operational Projects.")
      else:
        for proj in active_projects:
          with st.expander(f"ACTIVE PROJECT: {proj['title']} (Severity: {proj['severity']})", expanded=True):
            project_data = ProjectManager.get_project_details(proj['id'])
            
            # 1. Phase Progression Banner
            current_phase = proj['current_phase']
            phases = ["EMERGENCE", "ESCALATION", "STABILIZATION", "RECOVERY"]
            try:
              p_idx = phases.index(current_phase)
            except ValueError:
              p_idx = 0
              
            with st.container(border=True):
              st.write(f"**Executive Oversight | Phase:** {' âž” '.join([f'*{p}*' if i < p_idx else f'**{p}**' if i == p_idx else p for i, p in enumerate(phases)])}")
            
            p_col1, p_col2, p_col3 = st.columns([1, 1.5, 1])
            with p_col1:
              st.write("**Reporting Sectors**")
              for p_b_id in project_data['participants']:
                st.write(f"- {all_baskets.get(p_b_id, f'Basket {p_b_id}')}")
              st.write("---")
              st.write(f"**Created:**<br>{pd.to_datetime(proj['created_at'], unit='s').strftime('%Y-%m-%d %H:%M')}", unsafe_allow_html=True)
              
              st.write("---")
              st.write("**Sector Disagreement Metrics**")
              dis_matrix = ProjectManager.get_disagreement_matrix(proj['id'])
              if dis_matrix:
                import plotly.express as px
                df_dis = pd.DataFrame(list(dis_matrix.items()), columns=["Sector Admin", "Certainty"])
                fig = px.bar(df_dis, x="Sector Admin", y="Certainty", range_y=[0.0, 1.0], color="Certainty", color_continuous_scale="RdYlGn")
                fig.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0), template="plotly_white")
                _apply_plotly_numeric_font(fig)
                st.plotly_chart(fig, use_container_width=True, key=f"exec_dis_{proj['id']}")
              else:
                st.caption("Awaiting sector updates.")
              
            with p_col2:
              st.write("**Shared Causal Stream**")
              st.markdown(f"*{proj['description']}*")
              st.write("---")
              
              for update in project_data.get('updates', []):
                u_color = "#BB0000" if update['update_type'] == 'POLICY_ACTION' else "#006600" if update['update_type'] == 'OBSERVATION' else "#1F2937"
                st.markdown(f"**<span style='color:{u_color};'>[{update['update_type']}]</span> {update['author_name']}**", unsafe_allow_html=True)
                st.write(update['content'])
                if update['certainty']:
                  st.caption(f"Certainty Index: {update['certainty']:.2f}/1.0")
                st.write("")
                
            with p_col3:
              with st.container(border=True):
                st.write("Executive Directives")
                new_policy = st.text_area("Post Policy Action", key=f"pol_text_{proj['id']}")
                if st.button("Inject Action to Project Stream", type="primary", key=f"pol_btn_{proj['id']}"):
                  if new_policy:
                    ProjectManager.add_update(proj['id'], st.session_state.get('username', "Executive Tier"), 'POLICY_ACTION', new_policy)
                    st.rerun()
                  else:
                    st.error("Text required.")
                    
              with st.expander("Executive Override: Force Archive"):
                st.write("Executive Override: Force resolve this project and execute network-wide trust recalibration.")
                res_state = st.selectbox("Resolution State", ['RESOLVED', 'FALSE_ALARM', 'INSUFFICIENT_EVIDENCE', 'CONFLICTING_SIGNALS'], key=f"exec_res_{proj['id']}")
                pol_score = st.slider("National Policy Effectiveness", 0.0, 10.0, 5.0, 0.5, key=f"exec_score_{proj['id']}")
                res_sum = st.text_area("Executive Debrief", key=f"exec_sum_{proj['id']}")
                
                if st.button("Force Archive & Calibrate Network Weights", key=f"exec_arch_{proj['id']}", type="primary"):
                  if not res_sum:
                    st.error("Executive debrief summary required.")
                  else:
                    payload = {"final_severity": proj['severity'], "active_participants": len(project_data['participants']), "archived_by": "Executive"}
                    ProjectManager.archive_project(proj['id'], res_state, pol_score, res_sum, payload)
                    st.rerun()

            st.write("---")
            st.markdown("#### Structured Project Health")
            from kshiked.ui.institution.project_components import render_project_overview
            render_project_overview(project_data, "Executive", 0, all_baskets)

  if active_section == "Archive":
    with st.container(border=True):
      st.markdown("#### National Archive")
      st.write("Closed operations and the outcomes they recorded.")
      
      if not memories:
        st.write("No closed projects in the national archive.")
      else:
        for mem in memories:
          res_color = "#006600" if mem['resolution_state'] == 'RESOLVED' else "#BB0000" if mem['resolution_state'] == 'FALSE_ALARM' else "#1F2937"
          with st.expander(f"National Archive: {mem['title']} [{mem['resolution_state']}]"):
            m_col1, m_col2 = st.columns([2, 1])
            with m_col1:
              st.markdown(f"**Executive Debrief:**<br>{mem['resolution_summary']}", unsafe_allow_html=True)
              st.write("---")

              # â”€â”€ PILLAR 5: "DID IT WORK?" â”€â”€
              try:
                participant_ids = []
                with get_connection() as conn_p5:
                  cp5 = conn_p5.cursor()
                  cp5.execute("SELECT basket_id FROM project_participants WHERE project_id = ?", (mem['id'],))
                  participant_ids = [r['basket_id'] for r in cp5.fetchall()]
                if participant_ids:
                  impact = compute_outcome_impact(
                    project_id=mem['id'],
                    project_created_at=mem.get('created_at', 0),
                    project_archived_at=mem.get('updated_at', 0),
                    participant_basket_ids=participant_ids,
                  )
                  bg = "#F0FDF4" if impact.get('delta_pct', 0) < 0 else "#FEF2F2"
                  border = "#10B981" if impact.get('delta_pct', 0) < 0 else "#EF4444"
                  st.markdown(
                    f'<div style="background:{bg}; border-left:4px solid {border}; '
                    f'padding:10px 14px; border-radius:0 6px 6px 0; margin:8px 0; font-size:0.88rem;">'
                    f'<strong> Did it Work?</strong> {impact["narrative"]}</div>',
                    unsafe_allow_html=True,
                  )
              except Exception:
                pass

              st.json(mem['learning_payload'])
            with m_col2:
              st.metric("Policy Outcome Score", f"{mem['policy_effectiveness_score']}/10.0")
              ttc_mins = mem['time_to_consensus_seconds'] / 60.0
              st.metric("Coordination Time", f"{ttc_mins:.1f} minutes")
              st.markdown(f"<span style='color:{res_color};'>**Network Weights Recalibrated**</span>", unsafe_allow_html=True)
              
  if active_section == "Command & Control":
    with st.container(border=True):
      st.markdown("### Command & Control")
      st.write("Read escalations from sector admins and issue directives downward.")
      
      c1, c2 = st.columns([1, 1])
      with c1:
        with st.container(border=True):
          st.write("National Escalations (Inbox)")
          esc_inbox = esc_inbox_global
          if not esc_inbox:
            st.caption("No pending escalations.")
          else:
            for msg in esc_inbox:
              with st.expander(f"From {msg['sender_id']} | {msg['timestamp']} {'(NEW)' if not msg['is_read'] else ''}"):
                st.write(msg['content'])
                if not msg['is_read']:
                  if st.button("Mark Cleared", key=f"ex_clear_{msg['id']}"):
                    SecureMessaging.mark_read(msg['id'])
                    st.rerun()

      with c2:
        with st.container(border=True):
          st.write("Issue directives (Downward)")
          target_level = st.selectbox("Send to", ["Sector Admins", "Local Institutions", "Broadcast to all"])
          target_id = st.text_input("Username / Target (or 'ALL' to broadcast)")
          directive_type = st.selectbox("Directive Category", ["BENCHMARK_ENFORCEMENT", "RESOURCE_REALLOCATION", "EMERGENCY_DECREE", "STRATEGIC_WARNING"])
          priority = st.selectbox("Priority Level", ["CRITICAL", "HIGH", "ROUTINE"])
          directive = st.text_area("Command Payload", height=100)
          requires_ack = st.checkbox("Require explicit acknowledgment from destination node(s)", value=True)
          
          if st.button("Transmit Command", type="primary", use_container_width=True):
            if directive and target_id:
              if target_level == "Sector Admins" and target_id.lower() != "all":
                # Try to resolve target to basket_id
                # Assuming target_id could be the sector name or admin username, handling gracefully
                target_basket_id = next((k for k, v in all_baskets.items() if v.lower() == target_id.lower()), None)
                if target_basket_id:
                  DataSharingManager.issue_directive(Role.EXECUTIVE.value, st.session_state.get('username', 'Executive'), directive, priority, directive_type, target_basket_id=target_basket_id, requires_ack=requires_ack)
                  st.success(f"Directive transmitted to {all_baskets[target_basket_id]}")
                else:
                  st.error("Invalid target Sector Admin.")
              elif target_level == "Local Institutions" and target_id.lower() != "all":
                st.warning("Targeting individual spokes currently passes through Sector Admins first.")
                # Assuming ID is provided
                try:
                  tid = int(target_id)
                  DataSharingManager.issue_directive(Role.EXECUTIVE.value, st.session_state.get('username', 'Executive'), directive, priority, directive_type, target_institution_id=tid, requires_ack=requires_ack)
                  st.success(f"Directive transmitted to Spoke {tid}")
                except ValueError:
                  st.error("Please enter a valid Spoke ID.")
              else:
                # Broadcast to ALL baskets (Admins) which then cascade
                for b_id in all_baskets.keys():
                  DataSharingManager.issue_directive(Role.EXECUTIVE.value, st.session_state.get('username', 'Executive'), directive, priority, directive_type, target_basket_id=b_id, requires_ack=requires_ack)
                st.success("National Broadcast Directive transmitted to all Sectors.")
          


  if active_section == "Sector Summaries":
    st.markdown("### Sector Summaries")
    st.write("Per-sector snapshot: validated risks, operational project participation, and public sentiment.")
    st.write("---")

    df_social_sum = _enrich_pulse_with_simulated_challenges(get_pulse_data(), global_risks, all_baskets, active_projects)

    if not all_baskets:
      st.info("No sectors registered in the system.")
    else:
      cols_per_row = 2
      basket_list = list(all_baskets.items())
      for row_start in range(0, len(basket_list), cols_per_row):
        row_items = basket_list[row_start:row_start + cols_per_row]
        sum_cols = st.columns(len(row_items))
        for col_idx, (b_id, b_name) in enumerate(row_items):
          sector_risks = [r for r in global_risks if r.get('basket_id') == b_id]
          sector_projects = [p for p in active_projects]
          risk_count = len(sector_risks)
          proj_count = len(sector_projects)

          # Aggregate composite impact
          total_impact = sum(r.get('composite_scores', {}).get('B_Impact', 0) for r in sector_risks)
          avg_impact = (total_impact / risk_count) if risk_count else 0.0

          # Social sentiment for this sector (match by sector name in pulse data)
          if not df_social_sum.empty and 'Sector' in df_social_sum.columns:
            sector_mask = df_social_sum['Sector'].str.contains(b_name.split()[0], case=False, na=False)
            sector_df = df_social_sum[sector_mask]
            avg_sentiment = sector_df['Sentiment'].mean() if not sector_df.empty else df_social_sum['Sentiment'].mean()
          else:
            avg_sentiment = None

          # Color-code card border by severity
          if risk_count == 0:
            border_color = "#10B981" # green â€” clear
            status_label = "Clear"
          elif avg_impact > 7:
            border_color = "#EF4444" # red â€” critical
            status_label = "Critical"
          elif avg_impact > 4:
            border_color = "#F59E0B" # amber â€” elevated
            status_label = "Elevated"
          else:
            border_color = "#3B82F6" # blue â€” low
            status_label = "Low"

          sentiment_str = f"{avg_sentiment:.2f}" if avg_sentiment is not None else "N/A"

          with sum_cols[col_idx]:
            st.markdown(
              f'<div style="border: 2px solid {border_color}; border-radius: 8px; padding: 1rem; margin-bottom: 0.5rem; background: #f8fafc;">'
              f'<div style="font-weight: bold; font-size: 1rem; color: #1F2937;">{b_name}</div>'
              f'<div style="margin-top: 0.4rem;">'
              f'<span style="background:{border_color}; color:#fff; border-radius:4px; padding:2px 8px; font-size:0.75rem; font-weight:bold;">{status_label}</span>'
              f'</div>'
              f'<div style="margin-top: 0.6rem; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.3rem;">'
              f'<div style="text-align:center;"><div style="font-size:1.4rem; font-weight:bold; color:{border_color};">{risk_count}</div><div style="font-size:0.7rem; color:#64748b;">Risks</div></div>'
              f'<div style="text-align:center;"><div style="font-size:1.4rem; font-weight:bold; color:#1F2937;">{proj_count}</div><div style="font-size:0.7rem; color:#64748b;">Projects</div></div>'
              f'<div style="text-align:center;"><div style="font-size:1.4rem; font-weight:bold; color:#1F2937;">{sentiment_str}</div><div style="font-size:0.7rem; color:#64748b;">Sentiment</div></div>'
              f'</div>'
              f'</div>',
              unsafe_allow_html=True
            )
            if sector_risks:
              top_risk = sorted(sector_risks, key=lambda x: x.get('composite_scores', {}).get('B_Impact', 0), reverse=True)[0]
              st.caption(f"Top risk: {top_risk['title'][:60]}")

  if active_section == "Collaboration Room":
    render_collab_room(
      role=Role.EXECUTIVE.value,
      basket_id=None,
      username=st.session_state.get('username', 'executive'),
      all_baskets=all_baskets,
    )

