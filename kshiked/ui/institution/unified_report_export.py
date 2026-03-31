import io
import json
import zipfile
from typing import Dict, Iterable, List, Optional

import pandas as pd
import streamlit as st


def _safe_file_slug(text: str) -> str:
  raw = "".join(ch if ch.isalnum() else "_" for ch in str(text or "report"))
  compact = "_".join(part for part in raw.split("_") if part)
  return compact.lower() or "report"


def _plain_language_summary(
  dashboard_name: str,
  section_name: str,
  highlights: List[str],
  cost_delay: Optional[Dict[str, float]],
) -> str:
  lines: List[str] = []
  lines.append(f"Dashboard: {dashboard_name}")
  lines.append(f"Section: {section_name}")
  lines.append("")
  lines.append("What this means in plain language")

  if highlights:
    for item in highlights[:6]:
      lines.append(f"- {str(item)}")
  else:
    lines.append("- The system has generated a live status snapshot for this dashboard.")

  if isinstance(cost_delay, dict):
    dn = int(float(cost_delay.get("do_nothing_loss_kes_b_rounded", 0.0) or 0.0))
    ea = int(float(cost_delay.get("act_early_loss_kes_b_rounded", 0.0) or 0.0))
    lp = int(float(cost_delay.get("late_penalty_kes_b_rounded", 0.0) or 0.0))
    lines.append("")
    lines.append("Cost of delay estimate")
    lines.append(f"- Do nothing: about KES {dn} billion lost.")
    lines.append(f"- Act early: about KES {ea} billion lost.")
    lines.append(f"- Price of being late: about KES {lp} billion.")

  lines.append("")
  lines.append("How to use this report")
  lines.append("- Prioritize the highest-impact items first.")
  lines.append("- Focus on actions that reduce delay and protect essential services.")
  lines.append("- Use the JSON appendix for technical trace details if needed.")
  return "\n".join(lines)


def render_unified_report_export(
  dashboard_name: str,
  section_name: str,
  metrics: Dict[str, object],
  highlights: Optional[Iterable[str]] = None,
  cost_delay: Optional[Dict[str, float]] = None,
  tables: Optional[Dict[str, pd.DataFrame]] = None,
  evidence: Optional[Dict[str, object]] = None,
  key_prefix: str = "unified_report",
) -> None:
  """Render one-click report export used consistently across dashboards."""
  highlights_list = [str(h) for h in (highlights or []) if str(h).strip()]
  export_payload: Dict[str, object] = {
    "dashboard": str(dashboard_name),
    "section": str(section_name),
    "exported_at_utc": str(pd.Timestamp.utcnow()),
    "metrics": dict(metrics or {}),
    "highlights": highlights_list,
    "cost_of_delay": dict(cost_delay or {}),
    "evidence": dict(evidence or {}),
  }

  summary_text = _plain_language_summary(
    dashboard_name=str(dashboard_name),
    section_name=str(section_name),
    highlights=highlights_list,
    cost_delay=cost_delay,
  )

  with st.expander("Unified Report Export", expanded=False):
    st.caption(
      "Single export for non-technical decision makers. Includes plain-language summary, "
      "key numbers, and technical appendix."
    )

    mcols = st.columns(3)
    mcols[0].metric("Dashboard", str(dashboard_name))
    mcols[1].metric("Section", str(section_name))
    mcols[2].metric("Metrics Included", len(export_payload.get("metrics", {})))

    st.markdown("##### Quick Summary")
    st.text(summary_text)

    if metrics:
      metric_rows = [{"metric": str(k), "value": v} for k, v in metrics.items()]
      st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

    zip_buf = io.BytesIO()
    slug = _safe_file_slug(f"{dashboard_name}_{section_name}")
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
      archive.writestr("report_summary.txt", summary_text)
      archive.writestr(
        "report_payload.json",
        json.dumps(export_payload, ensure_ascii=True, indent=2),
      )
      if metrics:
        archive.writestr(
          "metrics.csv",
          pd.DataFrame([{"metric": str(k), "value": v} for k, v in metrics.items()]).to_csv(index=False),
        )
      for table_name, df in (tables or {}).items():
        if isinstance(df, pd.DataFrame) and not df.empty:
          archive.writestr(f"tables/{_safe_file_slug(table_name)}.csv", df.to_csv(index=False))

    zip_buf.seek(0)
    st.download_button(
      "Export Unified Report Pack (.zip)",
      data=zip_buf.getvalue(),
      file_name=f"unified_report_{slug}.zip",
      mime="application/zip",
      key=f"{key_prefix}_{slug}_zip",
    )
