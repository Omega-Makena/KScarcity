import io
import json
import zipfile
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages


def _safe_file_slug(text: str) -> str:
  raw = "".join(ch if ch.isalnum() else "_" for ch in str(text or "report"))
  compact = "_".join(part for part in raw.split("_") if part)
  return compact.lower() or "report"


def _plain_language_summary(
  dashboard_name: str,
  section_name: str,
  highlights: List[str],
  interpretations: List[str],
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

  if interpretations:
    lines.append("")
    lines.append("Interpretation")
    for item in interpretations[:8]:
      lines.append(f"- {str(item)}")

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
  interpretations: Optional[Iterable[str]] = None,
  cost_delay: Optional[Dict[str, float]] = None,
  tables: Optional[Dict[str, pd.DataFrame]] = None,
  evidence: Optional[Dict[str, object]] = None,
  key_prefix: str = "unified_report",
) -> None:
  """Render one-click report export used consistently across dashboards."""
  highlights_list = [str(h) for h in (highlights or []) if str(h).strip()]
  interpretations_list = [str(i) for i in (interpretations or []) if str(i).strip()]
  export_payload: Dict[str, object] = {
    "dashboard": str(dashboard_name),
    "section": str(section_name),
    "exported_at_utc": str(pd.Timestamp.utcnow()),
    "metrics": dict(metrics or {}),
    "highlights": highlights_list,
    "interpretations": interpretations_list,
    "cost_of_delay": dict(cost_delay or {}),
    "evidence": dict(evidence or {}),
  }

  summary_text = _plain_language_summary(
    dashboard_name=str(dashboard_name),
    section_name=str(section_name),
    highlights=highlights_list,
    interpretations=interpretations_list,
    cost_delay=cost_delay,
  )

  st.markdown("#### Unified Report Export")
  st.caption(
    "PDF is the primary report for decision makers. Use ZIP when you need full metadata and raw tables."
  )

  mcols = st.columns(3)
  mcols[0].metric("Dashboard", str(dashboard_name))
  mcols[1].metric("Section", str(section_name))
  mcols[2].metric("Metrics Included", len(export_payload.get("metrics", {})))

  if metrics:
    metric_rows = [{"metric": str(k), "value": v} for k, v in metrics.items()]
    st.dataframe(pd.DataFrame(metric_rows), use_container_width=True, hide_index=True)

  def _as_float(value: object) -> Optional[float]:
    try:
      return float(value)
    except Exception:
      return None

  def _render_pdf_report() -> bytes:
    pdf_buf = io.BytesIO()
    with PdfPages(pdf_buf) as pdf:
      # Page 1: Plain-language summary
      fig = plt.figure(figsize=(8.27, 11.69))
      fig.patch.set_facecolor("white")
      y = 0.96
      fig.text(0.07, y, f"{dashboard_name} - {section_name}", fontsize=16, fontweight="bold")
      y -= 0.04
      fig.text(0.07, y, f"Generated: {export_payload.get('exported_at_utc')}", fontsize=9, color="#555555")
      y -= 0.05

      for line in summary_text.splitlines():
        fig.text(0.07, y, line, fontsize=10, color="#111111")
        y -= 0.024
        if y < 0.06:
          break
      pdf.savefig(fig, bbox_inches="tight")
      plt.close(fig)

      # Page 2: Headline charts
      fig2, axs = plt.subplots(2, 1, figsize=(8.27, 11.69))
      fig2.patch.set_facecolor("white")

      metric_pairs = []
      for k, v in (metrics or {}).items():
        fv = _as_float(v)
        if fv is not None:
          metric_pairs.append((str(k), fv))

      if metric_pairs:
        names = [m[0] for m in metric_pairs][:8]
        values = [m[1] for m in metric_pairs][:8]
        axs[0].barh(names, values, color="#006600")
        axs[0].set_title("Key Metrics")
        axs[0].invert_yaxis()
        axs[0].grid(axis="x", alpha=0.2)
      else:
        axs[0].text(0.5, 0.5, "No numeric key metrics available.", ha="center", va="center")
        axs[0].set_axis_off()

      dn = int(float((cost_delay or {}).get("do_nothing_loss_kes_b_rounded", 0.0) or 0.0))
      ea = int(float((cost_delay or {}).get("act_early_loss_kes_b_rounded", 0.0) or 0.0))
      lp = int(float((cost_delay or {}).get("late_penalty_kes_b_rounded", 0.0) or 0.0))
      if dn > 0 or ea > 0 or lp > 0:
        labels = ["Do Nothing", "Act Early", "Price Of Delay"]
        vals = [dn, ea, lp]
        colors = ["#BB0000", "#006600", "#E05000"]
        axs[1].bar(labels, vals, color=colors)
        axs[1].set_title("Cost Of Delay (KES Billions)")
        axs[1].set_ylabel("KES Billions")
        axs[1].grid(axis="y", alpha=0.2)
      else:
        axs[1].text(0.5, 0.5, "Cost-of-delay values unavailable for this section.", ha="center", va="center")
        axs[1].set_axis_off()

      fig2.tight_layout()
      pdf.savefig(fig2, bbox_inches="tight")
      plt.close(fig2)

      # Additional pages: table snapshots
      for table_name, df in (tables or {}).items():
        if not isinstance(df, pd.DataFrame) or df.empty:
          continue
        sample = df.head(20).copy()
        figt = plt.figure(figsize=(11.69, 8.27))
        figt.patch.set_facecolor("white")
        ax = figt.add_subplot(111)
        ax.axis("off")
        ax.set_title(f"Table Snapshot: {table_name}", fontsize=12, pad=12)
        table = ax.table(
          cellText=sample.astype(str).values,
          colLabels=[str(c) for c in sample.columns],
          loc="center",
          cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.2)
        pdf.savefig(figt, bbox_inches="tight")
        plt.close(figt)

    pdf_buf.seek(0)
    return pdf_buf.getvalue()

  pdf_bytes = _render_pdf_report()
  slug = _safe_file_slug(f"{dashboard_name}_{section_name}")
  st.download_button(
    "Download Decision Report (PDF)",
    data=pdf_bytes,
    file_name=f"unified_report_{slug}.pdf",
    mime="application/pdf",
    key=f"{key_prefix}_{slug}_pdf",
    type="primary",
    use_container_width=True,
  )

  with st.expander("Technical Metadata Package", expanded=False):
    st.caption("Full metadata export for audits, integration, and deep technical review.")
    st.markdown("##### Quick Summary")
    st.text(summary_text)

    zip_buf = io.BytesIO()

    table_profiles: Dict[str, object] = {}
    for table_name, df in (tables or {}).items():
      if not isinstance(df, pd.DataFrame):
        continue
      table_profiles[str(table_name)] = {
        "row_count": int(len(df.index)),
        "column_count": int(len(df.columns)),
        "columns": [str(c) for c in df.columns],
        "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
      }

    metadata_bundle = {
      "summary_text": summary_text,
      "export_payload": export_payload,
      "table_profiles": table_profiles,
    }

    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
      archive.writestr("report_summary.txt", summary_text)
      archive.writestr(
        "report_payload.json",
        json.dumps(export_payload, ensure_ascii=True, indent=2),
      )
      archive.writestr(
        "metadata_bundle.json",
        json.dumps(metadata_bundle, ensure_ascii=True, indent=2),
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
      "Export Full Metadata Package (.zip)",
      data=zip_buf.getvalue(),
      file_name=f"unified_report_{slug}.zip",
      mime="application/zip",
      key=f"{key_prefix}_{slug}_zip",
      use_container_width=True,
    )
