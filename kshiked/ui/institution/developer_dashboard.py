import json
import re
import io
import zipfile
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from kshiked.ui.institution.backend.auth import enforce_role, logout_user
from kshiked.ui.institution.backend.analytics_engine import compute_cost_of_delay_kes_b
from kshiked.ui.institution.backend.model_quality import (
  build_unified_model_metrics,
  get_quality_routing_defaults,
  build_quality_assurance_snapshot,
)
from kshiked.ui.institution.backend.models import Role
from kshiked.ui.institution.unified_report_export import render_unified_report_export
from scarcity.engine.engine_v2 import OnlineDiscoveryEngine


ROOT = Path(__file__).resolve().parents[3]
AUDITS_DIR = ROOT / "audits"
ARTIFACTS_RUNS_DIR = ROOT / "artifacts" / "runs"
BENCHMARKS_DIR = ROOT / "artifacts" / "benchmarks"
LOG_HISTORY_DIR = ROOT / "kshiked" / "logs" / "history"
FED_RUNTIME_DIR = ROOT / "federated_databases" / "runtime"

DEV_TABS = [
  "Developer Overview",
  "Backend Tests",
  "Model Metrics",
  "Training Runs",
  "Logs & Edge Cases",
  "Raw Artifacts",
  "tracnepalaretny and autditabilty",
  "summary",
  "misues",
]


def _safe_read_text(path: Path, max_chars: int = 300000) -> str:
  try:
    text = path.read_text(encoding="utf-8", errors="replace")
  except Exception as exc:
    return f"[Read error] {exc}"
  if len(text) > max_chars:
    return text[-max_chars:]
  return text


def _parse_pytest_counts(text: str) -> Dict[str, int]:
  counts = {
    "passed": 0,
    "failed": 0,
    "errors": 0,
    "skipped": 0,
    "xfailed": 0,
    "xpassed": 0,
  }
  lower = text.lower()
  for key in counts:
    pattern = rf"(\d+)\s+{key}"
    for match in re.findall(pattern, lower):
      try:
        counts[key] += int(match)
      except ValueError:
        continue
  return counts


def _list_existing(paths: List[Path]) -> List[Path]:
  return [p for p in paths if p.exists()]


def _load_test_files() -> List[Path]:
  return _list_existing(
    [
      AUDITS_DIR / "pytest_main_run.txt",
      AUDITS_DIR / "pytest_fast_tests.txt",
      AUDITS_DIR / "pytest_slow_tests.txt",
      AUDITS_DIR / "pytest_scarcity_tests.txt",
      AUDITS_DIR / "pytest_test_collection.txt",
    ]
  )


def _load_audit_reports() -> List[Path]:
  if not AUDITS_DIR.exists():
    return []
  return sorted(AUDITS_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)


def _load_run_dirs(limit: int = 20) -> List[Path]:
  if not ARTIFACTS_RUNS_DIR.exists():
    return []
  run_dirs = [d for d in ARTIFACTS_RUNS_DIR.iterdir() if d.is_dir()]
  run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
  return run_dirs[:limit]


def _summarize_run(run_dir: Path) -> Dict[str, object]:
  summary_path = run_dir / "summary.json"
  effects_path = run_dir / "effects.jsonl"
  errors_path = run_dir / "errors.jsonl"

  summary = {}
  metadata = {}
  if summary_path.exists():
    try:
      payload = json.loads(_safe_read_text(summary_path, max_chars=200000))
      summary = payload.get("summary", {})
      metadata = payload.get("metadata", {})
    except Exception:
      summary = {}
      metadata = {}

  effects_count = 0
  if effects_path.exists():
    effects_text = _safe_read_text(effects_path)
    effects_count = len([ln for ln in effects_text.splitlines() if ln.strip()])

  errors_count = 0
  if errors_path.exists():
    errors_text = _safe_read_text(errors_path)
    errors_count = len([ln for ln in errors_text.splitlines() if ln.strip()])

  return {
    "run_id": run_dir.name,
    "status": summary.get("status", "unknown"),
    "duration_sec": summary.get("duration_sec"),
    "started_at": summary.get("started_at"),
    "finished_at": summary.get("finished_at"),
    "succeeded": summary.get("succeeded"),
    "failed": summary.get("failed"),
    "effects_count": effects_count,
    "errors_count": errors_count,
    "python_version": metadata.get("python_version"),
    "platform": metadata.get("platform"),
    "summary_path": str(summary_path),
    "effects_path": str(effects_path),
    "errors_path": str(errors_path),
    "run_path": str(run_dir),
  }


def _load_recent_logs(limit: int = 30) -> List[Path]:
  if not LOG_HISTORY_DIR.exists():
    return []
  log_files = [f for f in LOG_HISTORY_DIR.iterdir() if f.is_file() and f.suffix.lower() == ".json"]
  log_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
  return log_files[:limit]


def _load_latest_drift_benchmark(mode: str) -> Dict[str, object]:
  if not BENCHMARKS_DIR.exists():
    return {}
  candidates = sorted(
    BENCHMARKS_DIR.glob(f"engine_drift_{mode}_*.json"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
  )
  if not candidates:
    return {}
  target = candidates[0]
  try:
    payload = json.loads(_safe_read_text(target, max_chars=800000))
  except Exception:
    return {}
  payload["_path"] = str(target.relative_to(ROOT))
  payload["_mtime"] = str(pd.to_datetime(target.stat().st_mtime, unit="s"))
  return payload


def _load_latest_fl_accuracy_benchmark() -> Dict[str, object]:
  if not BENCHMARKS_DIR.exists():
    return {}
  candidates = sorted(
    BENCHMARKS_DIR.glob("fl_model_accuracy_*.json"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
  )
  if not candidates:
    return {}
  target = candidates[0]
  try:
    payload = json.loads(_safe_read_text(target, max_chars=800000))
  except Exception:
    return {}
  payload["_path"] = str(target.relative_to(ROOT))
  payload["_mtime"] = str(pd.to_datetime(target.stat().st_mtime, unit="s"))
  return payload


def _load_latest_meta_accuracy_benchmark() -> Dict[str, object]:
  if not BENCHMARKS_DIR.exists():
    return {}
  candidates = sorted(
    BENCHMARKS_DIR.glob("meta_model_accuracy_*.json"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
  )
  if not candidates:
    return {}
  target = candidates[0]
  try:
    payload = json.loads(_safe_read_text(target, max_chars=800000))
  except Exception:
    return {}
  payload["_path"] = str(target.relative_to(ROOT))
  payload["_mtime"] = str(pd.to_datetime(target.stat().st_mtime, unit="s"))
  return payload


def _load_latest_statistical_accuracy_benchmark() -> Dict[str, object]:
  if not BENCHMARKS_DIR.exists():
    return {}
  candidates = sorted(
    BENCHMARKS_DIR.glob("statistical_model_accuracy_*.json"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
  )
  if not candidates:
    return {}
  target = candidates[0]
  try:
    payload = json.loads(_safe_read_text(target, max_chars=800000))
  except Exception:
    return {}
  payload["_path"] = str(target.relative_to(ROOT))
  payload["_mtime"] = str(pd.to_datetime(target.stat().st_mtime, unit="s"))
  return payload


def _load_latest_online_accuracy_benchmark() -> Dict[str, object]:
  if not BENCHMARKS_DIR.exists():
    return {}
  candidates = sorted(
    BENCHMARKS_DIR.glob("online_model_accuracy_*.json"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
  )
  if not candidates:
    return {}
  target = candidates[0]
  try:
    payload = json.loads(_safe_read_text(target, max_chars=800000))
  except Exception:
    return {}
  payload["_path"] = str(target.relative_to(ROOT))
  payload["_mtime"] = str(pd.to_datetime(target.stat().st_mtime, unit="s"))
  return payload


def _extract_edge_case_tests() -> List[str]:
  files = _list_existing(
    [
      ROOT / "scarcity" / "tests" / "test_engine_integration.py",
      ROOT / "tests" / "test_federated_databases_smoke.py",
      ROOT / "tests" / "test_kcollab_non_iid.py",
      ROOT / "tests" / "test_fl_model_metrics_over_time.py",
      ROOT / "tests" / "test_health_sector_resilience.py",
    ]
  )
  if not files:
    return []

  keywords = (
    "malformed",
    "invalid",
    "corrupt",
    "drift",
    "empty",
    "skip",
    "fails",
    "failure",
    "error",
    "non_iid",
    "robust",
  )
  found: List[str] = []
  for path in files:
    text = _safe_read_text(path, max_chars=600000)
    for name in re.findall(r"^\s*def\s+(test_[a-zA-Z0-9_]+)\s*\(", text, flags=re.MULTILINE):
      lname = name.lower()
      if any(k in lname for k in keywords):
        found.append(name)
  return sorted(set(found))


def _render_unified_model_metrics_table() -> None:
  st.markdown("### Unified Model Inventory and Metrics")
  all_metrics = build_unified_model_metrics()
  if not all_metrics:
    st.info("No discovered models or benchmark metrics found yet.")
    return

  df = pd.DataFrame(all_metrics)
  st.dataframe(df, use_container_width=True, hide_index=True)

  ok_count = int((df.get("data_status") == "ok").sum()) if "data_status" in df.columns else 0
  missing_count = int((df.get("data_status") != "ok").sum()) if "data_status" in df.columns else 0
  top_row = df.sort_values(by="composite_score", ascending=False).iloc[0] if "composite_score" in df.columns and not df.empty else None

  c1, c2, c3 = st.columns(3)
  c1.metric("Discovered Models", len(df))
  c2.metric("Models With Metrics", ok_count)
  c3.metric("Missing Metrics", missing_count)

  if top_row is not None:
    st.caption(
      f"Top composite model: {top_row.get('model')} ({top_row.get('family')}) score={float(top_row.get('composite_score') or 0.0):.3f}"
    )

  routing_defaults = get_quality_routing_defaults()
  if routing_defaults:
    st.markdown("#### Quality-First Routing Defaults")
    st.dataframe(pd.DataFrame(routing_defaults), use_container_width=True, hide_index=True)


def _load_quality_override_events(limit: int = 1000) -> List[Dict[str, object]]:
  log_path = AUDITS_DIR / "quality_override_events.jsonl"
  if not log_path.exists() or not log_path.is_file():
    return []

  events: List[Dict[str, object]] = []
  for line in _safe_read_text(log_path, max_chars=900000).splitlines():
    line = line.strip()
    if not line:
      continue
    try:
      obj = json.loads(line)
    except Exception:
      continue
    if isinstance(obj, dict):
      events.append(obj)

  events.sort(key=lambda e: float(e.get("created_at") or 0.0), reverse=True)
  return events[:limit]


def _render_quality_override_events() -> None:
  st.markdown("### Quality Overrides")
  events = _load_quality_override_events(limit=1200)
  if not events:
    st.info("No quality override events logged yet.")
    return

  rows: List[Dict[str, object]] = []
  for event in events:
    rows.append(
      {
        "timestamp": str(pd.to_datetime(float(event.get("created_at") or 0.0), unit="s")),
        "family": event.get("family"),
        "actor": event.get("actor"),
        "context": event.get("context"),
        "reason": event.get("reason"),
        "selected": event.get("selected"),
        "recommended": event.get("recommended"),
        "details": json.dumps(event.get("details") or {}, ensure_ascii=True),
      }
    )

  df = pd.DataFrame(rows)
  families = ["All"] + sorted(df["family"].fillna("unknown").astype(str).unique().tolist())
  actors = ["All"] + sorted(df["actor"].fillna("unknown").astype(str).unique().tolist())
  reasons = ["All"] + sorted(df["reason"].fillna("unknown").astype(str).unique().tolist())

  c1, c2, c3 = st.columns(3)
  selected_family = c1.selectbox("Filter family", families, key="quality_override_family")
  selected_actor = c2.selectbox("Filter actor", actors, key="quality_override_actor")
  selected_reason = c3.selectbox("Filter reason", reasons, key="quality_override_reason")

  filtered = df
  if selected_family != "All":
    filtered = filtered[filtered["family"].astype(str) == selected_family]
  if selected_actor != "All":
    filtered = filtered[filtered["actor"].astype(str) == selected_actor]
  if selected_reason != "All":
    filtered = filtered[filtered["reason"].astype(str) == selected_reason]

  m1, m2, m3 = st.columns(3)
  m1.metric("Override Events", len(filtered))
  m2.metric("Unique Actors", filtered["actor"].astype(str).nunique())
  m3.metric("Unique Families", filtered["family"].astype(str).nunique())

  st.dataframe(filtered, use_container_width=True, hide_index=True)


def _count_recent_quality_overrides(hours: int = 24) -> int:
  cutoff = time.time() - (max(1, int(hours)) * 3600)
  count = 0
  for event in _load_quality_override_events(limit=5000):
    try:
      ts = float(event.get("created_at") or 0.0)
    except Exception:
      ts = 0.0
    if ts >= cutoff:
      count += 1
  return count


def _extract_recent_decisions(run_summaries: List[Dict[str, object]], limit: int = 40) -> List[Dict[str, object]]:
  decisions: List[Dict[str, object]] = []
  ordered = sorted(
    run_summaries,
    key=lambda r: str(r.get("started_at") or ""),
    reverse=True,
  )
  for run in ordered:
    effects_path = Path(str(run.get("effects_path") or ""))
    if not effects_path.exists() or not effects_path.is_file():
      continue

    for line in _safe_read_text(effects_path, max_chars=500000).splitlines():
      line = line.strip()
      if not line:
        continue
      try:
        obj = json.loads(line)
      except Exception:
        continue

      spec = obj.get("spec", {}) if isinstance(obj.get("spec"), dict) else {}
      diagnostics = obj.get("diagnostics", {}) if isinstance(obj.get("diagnostics"), dict) else {}
      provenance = obj.get("provenance", {}) if isinstance(obj.get("provenance"), dict) else {}
      backend = obj.get("backend", {}) if isinstance(obj.get("backend"), dict) else {}
      temporal = obj.get("temporal_diagnostics", {}) if isinstance(obj.get("temporal_diagnostics"), dict) else {}
      refuters = obj.get("refuter_results", {}) if isinstance(obj.get("refuter_results"), dict) else {}

      decisions.append(
        {
          "run_id": provenance.get("run_id") or run.get("run_id"),
          "spec_id": obj.get("spec_id", ""),
          "estimand_type": obj.get("estimand_type"),
          "estimate": obj.get("estimate"),
          "treatment": spec.get("treatment"),
          "outcome": spec.get("outcome"),
          "confounders_count": len(spec.get("confounders") or []),
          "rows": diagnostics.get("rows"),
          "method": backend.get("method_name") or diagnostics.get("method_name"),
          "created_at": obj.get("created_at"),
          "data_hash": provenance.get("data_hash"),
          "temporal_valid": temporal.get("valid", True),
          "temporal_issues": len(temporal.get("issues") or []),
          "temporal_warnings": len(temporal.get("warnings") or []),
          "refuter_errors": sum(
            1
            for v in refuters.values()
            if isinstance(v, dict) and str(v.get("status", "")).lower() == "error"
          ),
          "_raw": obj,
        }
      )

      if len(decisions) >= limit:
        return decisions
  return decisions


def _decision_warning_flags(decision: Dict[str, object]) -> List[Dict[str, str]]:
  flags: List[Dict[str, str]] = []
  rows = decision.get("rows")
  confounders_count = int(decision.get("confounders_count") or 0)
  estimate = decision.get("estimate")
  refuter_errors = int(decision.get("refuter_errors") or 0)
  temporal_valid = bool(decision.get("temporal_valid", True))
  temporal_issues = int(decision.get("temporal_issues") or 0)
  temporal_warnings = int(decision.get("temporal_warnings") or 0)

  try:
    abs_estimate = abs(float(estimate))
  except Exception:
    abs_estimate = None

  if isinstance(rows, (int, float)) and rows < 40:
    flags.append(
      {
        "category": "risk",
        "severity": "high",
        "message": f"Very small sample size ({rows}) can destabilize decision confidence.",
      }
    )
  elif isinstance(rows, (int, float)) and rows < 100:
    flags.append(
      {
        "category": "risk",
        "severity": "medium",
        "message": f"Moderate sample size ({rows}) may limit robustness under shift.",
      }
    )

  if confounders_count == 0:
    flags.append(
      {
        "category": "bias",
        "severity": "high",
        "message": "No confounders listed; omitted-variable bias risk is elevated.",
      }
    )
  elif confounders_count < 2:
    flags.append(
      {
        "category": "bias",
        "severity": "medium",
        "message": "Very few confounders supplied; bias control may be weak.",
      }
    )

  if refuter_errors > 0:
    flags.append(
      {
        "category": "misuse",
        "severity": "high",
        "message": f"{refuter_errors} refuter checks errored; avoid operational use until resolved.",
      }
    )

  if abs_estimate is not None and abs_estimate > 1.0 and (rows is None or rows < 200):
    flags.append(
      {
        "category": "misuse",
        "severity": "medium",
        "message": "Large effect on limited data; high chance of over-interpretation.",
      }
    )

  if not temporal_valid or temporal_issues > 0:
    flags.append(
      {
        "category": "misconsideration",
        "severity": "high",
        "message": "Temporal diagnostics report invalidity/issues; sequence assumptions may be broken.",
      }
    )
  elif temporal_warnings > 0:
    flags.append(
      {
        "category": "misconsideration",
        "severity": "medium",
        "message": "Temporal warnings present; validate time-order assumptions before action.",
      }
    )

  return flags


def _render_summary_tab(test_files: List[Path], run_summaries: List[Dict[str, object]], logs: List[Path]):
  st.subheader("Summary")
  st.caption("Consolidated snapshot of executed tests, edge-case coverage, and latest benchmark metrics.")

  totals = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
  for tf in test_files:
    counts = _parse_pytest_counts(_safe_read_text(tf, max_chars=250000))
    for key in totals:
      totals[key] += counts.get(key, 0)

  total_runs = len(run_summaries)
  successful_runs = sum(1 for r in run_summaries if str(r.get("status", "")).lower() == "success")
  run_success_rate = (successful_runs / total_runs) if total_runs else 0.0
  run_errors = sum(int(r.get("errors_count") or 0) for r in run_summaries)

  c1, c2, c3, c4 = st.columns(4)
  c1.metric("Tests Passed", totals["passed"])
  c2.metric("Tests Failed+Errors", totals["failed"] + totals["errors"])
  c3.metric("Recent Run Success", f"{run_success_rate:.1%}")
  c4.metric("Run Error Records", run_errors)

  c5, c6, c7 = st.columns(3)
  c5.metric("Recent Run Folders", total_runs)
  c6.metric("Rendered Log Files", len(logs))
  c7.metric("Skipped/XFail", totals["skipped"] + totals["xfailed"])

  st.markdown("### Edge Cases Covered")
  edge_cases = _extract_edge_case_tests()
  if edge_cases:
    st.markdown("\n".join([f"- {name}" for name in edge_cases[:60]]))
    if len(edge_cases) > 60:
      st.caption(f"Showing 60 of {len(edge_cases)} detected edge-case tests.")
  else:
    st.info("No edge-case test signatures detected from selected files.")

  st.markdown("### Current Drift Benchmark Metrics")
  balanced_payload = _load_latest_drift_benchmark("balanced")
  performance_payload = _load_latest_drift_benchmark("performance")

  def _payload_to_rows(payload: Dict[str, object], mode: str) -> List[Dict[str, object]]:
    if not payload:
      return []
    rows = []
    results = payload.get("results", {})
    if not isinstance(results, dict):
      return rows
    for magnitude, entry in results.items():
      if not isinstance(entry, dict):
        continue
      summary = entry.get("summary", {})
      if not isinstance(summary, dict):
        continue
      rows.append(
        {
          "mode": mode,
          "magnitude": magnitude,
          "detect_rate": summary.get("detect_rate"),
          "split_rate": summary.get("split_rate"),
          "latency_mean": summary.get("latency_mean"),
          "latency_p95": summary.get("latency_p95"),
          "avg_pressure_delta": summary.get("avg_pressure_delta"),
        }
      )
    return rows

  benchmark_rows = _payload_to_rows(balanced_payload, "balanced") + _payload_to_rows(performance_payload, "performance")
  if benchmark_rows:
    st.dataframe(pd.DataFrame(benchmark_rows), use_container_width=True, hide_index=True)
    info_cols = st.columns(2)
    info_cols[0].caption(
      f"Latest balanced benchmark: {balanced_payload.get('_path', 'n/a')} ({balanced_payload.get('_mtime', 'n/a')})"
      if balanced_payload
      else "Latest balanced benchmark: n/a"
    )
    info_cols[1].caption(
      f"Latest performance benchmark: {performance_payload.get('_path', 'n/a')} ({performance_payload.get('_mtime', 'n/a')})"
      if performance_payload
      else "Latest performance benchmark: n/a"
    )
  else:
    st.info("No benchmark exports found yet in artifacts/benchmarks.")

  st.markdown("### FL Model Accuracy Metrics")
  fl_accuracy_payload = _load_latest_fl_accuracy_benchmark()
  rows = fl_accuracy_payload.get("results", []) if isinstance(fl_accuracy_payload, dict) else []
  if isinstance(rows, list) and rows:
    flat_rows: List[Dict[str, object]] = []
    for row in rows:
      if not isinstance(row, dict):
        continue
      mm = row.get("metrics_mean", {}) if isinstance(row.get("metrics_mean"), dict) else {}
      ci = row.get("metrics_ci95", {}) if isinstance(row.get("metrics_ci95"), dict) else {}
      flat_rows.append(
        {
          "model_name": row.get("model_name"),
          "objective": row.get("objective_metric"),
          "best_threshold": row.get("best_threshold_mean"),
          "fit_pos_rate": row.get("fit_positive_rate_avg"),
          "test_pos_rate": row.get("test_positive_rate_avg"),
          "accuracy": mm.get("accuracy"),
          "precision": mm.get("precision"),
          "recall": mm.get("recall"),
          "f1": mm.get("f1"),
          "balanced_accuracy": mm.get("balanced_accuracy"),
          "tpr": mm.get("tpr"),
          "tnr": mm.get("tnr"),
          "brier": mm.get("brier"),
          "ece": mm.get("ece"),
          "acc_ci95_low": (ci.get("accuracy", {}) or {}).get("low"),
          "acc_ci95_high": (ci.get("accuracy", {}) or {}).get("high"),
          "f1_ci95_low": (ci.get("f1", {}) or {}).get("low"),
          "f1_ci95_high": (ci.get("f1", {}) or {}).get("high"),
        }
      )

    if flat_rows:
      df = pd.DataFrame(flat_rows)
      st.dataframe(df, use_container_width=True, hide_index=True)

      best_bal_idx = df["balanced_accuracy"].astype(float).idxmax() if "balanced_accuracy" in df.columns else None
      best_rec_idx = df["recall"].astype(float).idxmax() if "recall" in df.columns else None
      avg_ece = float(pd.to_numeric(df.get("ece", pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean()) if "ece" in df.columns else 0.0

      m1, m2, m3 = st.columns(3)
      m1.metric(
        "Best Balanced Accuracy",
        f"{float(df.loc[best_bal_idx, 'balanced_accuracy']):.3f}" if best_bal_idx is not None else "n/a",
        str(df.loc[best_bal_idx, "model_name"]) if best_bal_idx is not None else None,
      )
      m2.metric(
        "Best Recall",
        f"{float(df.loc[best_rec_idx, 'recall']):.3f}" if best_rec_idx is not None else "n/a",
        str(df.loc[best_rec_idx, "model_name"]) if best_rec_idx is not None else None,
      )
      m3.metric("Average Calibration Error (ECE)", f"{avg_ece:.3f}")

    st.caption(
      f"Latest FL accuracy benchmark: {fl_accuracy_payload.get('_path', 'n/a')} ({fl_accuracy_payload.get('_mtime', 'n/a')})"
    )
  else:
    st.info("No FL model accuracy exports found yet in artifacts/benchmarks.")

  st.markdown("### Meta Model Accuracy Metrics")
  meta_payload = _load_latest_meta_accuracy_benchmark()
  meta_rows = meta_payload.get("results", []) if isinstance(meta_payload, dict) else []
  if isinstance(meta_rows, list) and meta_rows:
    mrows: List[Dict[str, object]] = []
    for row in meta_rows:
      if not isinstance(row, dict):
        continue
      summary = row.get("summary", {}) if isinstance(row.get("summary"), dict) else {}
      mrows.append(
        {
          "meta_model": row.get("model"),
          "sign_accuracy": summary.get("sign_accuracy_mean"),
          "transfer_gain_rate": summary.get("transfer_gain_rate_mean"),
          "rollback_accuracy": summary.get("rollback_accuracy_mean"),
          "rollback_precision": summary.get("rollback_precision_mean"),
          "rollback_recall": summary.get("rollback_recall_mean"),
          "participants": summary.get("participants_mean"),
          "confidence_mean": summary.get("confidence_mean_mean"),
        }
      )
    if mrows:
      st.dataframe(pd.DataFrame(mrows), use_container_width=True, hide_index=True)
      st.caption(
        f"Latest meta benchmark: {meta_payload.get('_path', 'n/a')} ({meta_payload.get('_mtime', 'n/a')})"
      )
  else:
    st.info("No meta-model accuracy exports found yet in artifacts/benchmarks.")

  st.markdown("### Statistical Model Accuracy Metrics")
  stat_payload = _load_latest_statistical_accuracy_benchmark()
  stat_rows = stat_payload.get("results", []) if isinstance(stat_payload, dict) else []
  if isinstance(stat_rows, list) and stat_rows:
    srows: List[Dict[str, object]] = []
    for row in stat_rows:
      if not isinstance(row, dict):
        continue
      summary = row.get("summary", {}) if isinstance(row.get("summary"), dict) else {}
      srows.append(
        {
          "model": row.get("model"),
          "overall_accuracy": summary.get("overall_accuracy_mean"),
          "direction_accuracy": summary.get("direction_accuracy_mean"),
          "magnitude_accuracy": summary.get("magnitude_accuracy_mean"),
          "rmse": summary.get("rmse_mean"),
          "mae": summary.get("mae_mean"),
        }
      )
    if srows:
      sdf = pd.DataFrame(srows).sort_values(by="overall_accuracy", ascending=False)
      st.dataframe(sdf, use_container_width=True, hide_index=True)
      st.caption(
        f"Latest statistical benchmark: {stat_payload.get('_path', 'n/a')} ({stat_payload.get('_mtime', 'n/a')})"
      )
  else:
    st.info("No statistical model accuracy exports found yet in artifacts/benchmarks.")

  st.markdown("### Online Model Accuracy Metrics")
  online_payload = _load_latest_online_accuracy_benchmark()
  online_rows = online_payload.get("results", []) if isinstance(online_payload, dict) else []
  if isinstance(online_rows, list) and online_rows:
    orows: List[Dict[str, object]] = []
    for row in online_rows:
      if not isinstance(row, dict):
        continue
      summary = row.get("summary", {}) if isinstance(row.get("summary"), dict) else {}
      orows.append(
        {
          "model": row.get("model"),
          "accuracy_like": summary.get("accuracy_like_mean"),
          "fit_score": summary.get("fit_score_mean"),
          "mae": summary.get("mae_mean"),
          "parameter_error": summary.get("parameter_error_mean"),
          "equilibrium": summary.get("equilibrium_mean"),
          "shift_error": summary.get("shift_error_mean"),
          "avg_latency_ms": summary.get("avg_latency_ms_mean"),
          "p99_latency_ms": summary.get("p99_latency_ms_mean"),
        }
      )
    if orows:
      odf = pd.DataFrame(orows).sort_values(by="accuracy_like", ascending=False)
      st.dataframe(odf, use_container_width=True, hide_index=True)
      st.caption(
        f"Latest online benchmark: {online_payload.get('_path', 'n/a')} ({online_payload.get('_mtime', 'n/a')})"
      )
  else:
    st.info("No online model accuracy exports found yet in artifacts/benchmarks.")

  _render_unified_model_metrics_table()

  st.markdown("### Recently Executed Validation Bundles")
  st.markdown(
    "\n".join(
      [
        "- scarcity/tests/test_engine_integration.py",
        "- scarcity/tests/test_meta.py",
        "- scarcity/tests/test_relationships.py",
        "- tests/test_fl_model_metrics_over_time.py",
        "- tests/test_kcollab_non_iid.py",
        "- tests/test_health_sector_resilience.py",
        "- tests/test_federated_databases_smoke.py",
        "- scripts/benchmark_engine_drift.py (balanced/performance)",
      ]
    )
  )

  st.markdown("### Decision Trace and Explainability")
  decisions = _extract_recent_decisions(run_summaries, limit=60)
  if not decisions:
    st.info("No decision artifacts found in recent effects.jsonl files.")
    return

  traced_rows = []
  warning_rows = []
  for d in decisions:
    flags = _decision_warning_flags(d)
    traced_rows.append(
      {
        "run_id": d.get("run_id"),
        "spec_id": str(d.get("spec_id", ""))[:90],
        "estimand_type": d.get("estimand_type"),
        "estimate": d.get("estimate"),
        "rows": d.get("rows"),
        "confounders": d.get("confounders_count"),
        "method": d.get("method"),
        "created_at": d.get("created_at"),
        "warnings": len(flags),
      }
    )
    for f in flags:
      warning_rows.append(
        {
          "run_id": d.get("run_id"),
          "spec_id": str(d.get("spec_id", ""))[:90],
          "category": f.get("category"),
          "severity": f.get("severity"),
          "message": f.get("message"),
        }
      )

  c1, c2, c3, c4 = st.columns(4)
  c1.metric("Decisions Traced", len(decisions))
  c2.metric("Warnings Raised", len(warning_rows))
  c3.metric("High Severity", sum(1 for w in warning_rows if str(w.get("severity")) == "high"))
  c4.metric("Bias Flags", sum(1 for w in warning_rows if str(w.get("category")) == "bias"))

  st.markdown("Decision Trace Table")
  st.dataframe(pd.DataFrame(traced_rows), use_container_width=True, hide_index=True)

  if warning_rows:
    st.markdown("Decision Warning Table")
    st.dataframe(pd.DataFrame(warning_rows), use_container_width=True, hide_index=True)
  else:
    st.success("No explainability warning flags detected in currently loaded decisions.")


def _render_misues_tab(run_summaries: List[Dict[str, object]], logs: List[Path]):
  st.subheader("misues")
  st.caption("Detects potential system misuse from decision traces, refuter failures, and risky audit patterns.")

  decisions = _extract_recent_decisions(run_summaries, limit=120)
  misuse_rows: List[Dict[str, object]] = []
  refuter_error_total = 0

  for d in decisions:
    flags = _decision_warning_flags(d)
    refuter_error_total += int(d.get("refuter_errors") or 0)
    for f in flags:
      if str(f.get("category")) != "misuse":
        continue
      misuse_rows.append(
        {
          "run_id": d.get("run_id"),
          "spec_id": str(d.get("spec_id", ""))[:90],
          "severity": f.get("severity"),
          "message": f.get("message"),
          "method": d.get("method"),
          "rows": d.get("rows"),
          "refuter_errors": d.get("refuter_errors"),
          "created_at": d.get("created_at"),
        }
      )

  risky_patterns_path = AUDITS_DIR / "operations_risky_patterns.txt"
  risky_lines = []
  if risky_patterns_path.exists():
    risky_lines = [ln for ln in _safe_read_text(risky_patterns_path, max_chars=120000).splitlines() if ln.strip()]

  c1, c2, c3, c4 = st.columns(4)
  c1.metric("Decisions Reviewed", len(decisions))
  c2.metric("Misuse Flags", len(misuse_rows))
  c3.metric("High Severity", sum(1 for r in misuse_rows if str(r.get("severity")) == "high"))
  c4.metric("Refuter Errors", refuter_error_total)

  c5, c6 = st.columns(2)
  c5.metric("Risky Pattern Lines", len(risky_lines))
  c6.metric("Log Files Reviewed", len(logs))

  if misuse_rows:
    mdf = pd.DataFrame(misuse_rows).sort_values(by=["severity", "created_at"], ascending=[True, False])
    st.markdown("Misuse Incidents")
    st.dataframe(mdf, use_container_width=True, hide_index=True)
  else:
    st.success("No misuse incidents detected from current decision traces.")

  st.markdown("Audit Signals")
  if risky_lines:
    st.code("\n".join(risky_lines[:200]), language="text")
    if len(risky_lines) > 200:
      st.caption(f"Showing 200 of {len(risky_lines)} risky pattern lines.")
  else:
    st.info("No operations_risky_patterns.txt signals found.")


def _count_high_severity_misues(run_summaries: List[Dict[str, object]], limit: int = 120) -> int:
  decisions = _extract_recent_decisions(run_summaries, limit=limit)
  count = 0
  for d in decisions:
    for flag in _decision_warning_flags(d):
      if str(flag.get("category")) == "misuse" and str(flag.get("severity")) == "high":
        count += 1
  return count


def _score_band(score: float) -> str:
  s = max(0.0, min(1.0, float(score)))
  if s >= 0.8:
    return "strong"
  if s >= 0.6:
    return "moderate"
  if s >= 0.4:
    return "weak"
  return "critical_gap"


def _render_assurance_scorecard(test_files: List[Path], run_summaries: List[Dict[str, object]], logs: List[Path]) -> None:
  st.subheader("Assurance Scorecard")
  st.caption("Evaluates metric meaning vs baseline, robustness under stress/partial failure, explainability/transparency for audits, and deployability realism.")

  snapshot = build_quality_assurance_snapshot()
  overall = snapshot.get("overall_assurance", {}) if isinstance(snapshot, dict) else {}
  metric = snapshot.get("metric_credibility", {}) if isinstance(snapshot, dict) else {}
  robustness = snapshot.get("robustness", {}) if isinstance(snapshot, dict) else {}
  traceability = snapshot.get("traceability", {}) if isinstance(snapshot, dict) else {}
  deployability = snapshot.get("deployment_realism", {}) if isinstance(snapshot, dict) else {}
  drg_allocator = deployability.get("dynamic_resource_allocator", {}) if isinstance(deployability.get("dynamic_resource_allocator"), dict) else {}

  metric_meaning_score = float(metric.get("score", 0.0) or 0.0)
  robustness_score = float(robustness.get("score", 0.0) or 0.0)
  traceability_score = float(traceability.get("score", 0.0) or 0.0)
  deployability_score = float(deployability.get("score", 0.0) or 0.0)
  drg_score = float(drg_allocator.get("score", 0.0) or 0.0)

  c1, c2, c3, c4, c5 = st.columns(5)
  c1.metric("Metric Meaning", f"{metric_meaning_score * 100:.0f}%", _score_band(metric_meaning_score))
  c2.metric("Robustness", f"{robustness_score * 100:.0f}%", _score_band(robustness_score))
  c3.metric("Traceability", f"{traceability_score * 100:.0f}%", _score_band(traceability_score))
  c4.metric("Deployment Realism", f"{deployability_score * 100:.0f}%", _score_band(deployability_score))
  c5.metric("DRG Allocator", f"{drg_score * 100:.0f}%", _score_band(drg_score))

  with st.expander("Why this score?", expanded=False):
    st.caption("Exact formulas and weighted contributions used to compute assurance.")
    overall_formula = str(overall.get("formula") or "")
    if overall_formula:
      st.code(overall_formula, language="text")

    export_payload = {
      "overall_assurance": overall,
      "metric_credibility": metric,
      "robustness": robustness,
      "traceability": traceability,
      "deployment_realism": deployability,
      "summary_rows": snapshot.get("summary_rows", []),
    }
    ex1, ex2, ex3 = st.columns(3)
    ex1.download_button(
      "Export Explainability JSON",
      data=json.dumps(export_payload, ensure_ascii=True, indent=2).encode("utf-8"),
      file_name="assurance_explainability_developer.json",
      mime="application/json",
      key="dev_assurance_explainability_json_export",
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
        file_name="assurance_components_developer.csv",
        mime="text/csv",
        key="dev_assurance_components_csv_export",
      )

    def _render_breakdown(title: str, obj: Dict[str, object]) -> None:
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
        file_name="assurance_signal_breakdown_developer.csv",
        mime="text/csv",
        key="dev_assurance_breakdown_csv_export",
      )

  baseline_rows = metric.get("baseline_rows", []) if isinstance(metric.get("baseline_rows"), list) else []
  if baseline_rows:
    st.markdown("#### Baseline Comparisons by Model Family")
    st.dataframe(pd.DataFrame(baseline_rows), use_container_width=True, hide_index=True)

  drift_rows = robustness.get("drift_rows", []) if isinstance(robustness.get("drift_rows"), list) else []
  if drift_rows:
    st.markdown("#### Robustness Benchmarks (Noise/Drift)")
    st.dataframe(pd.DataFrame(drift_rows), use_container_width=True, hide_index=True)

  decisions = _extract_recent_decisions(run_summaries, limit=120)
  docs = traceability.get("documentation_paths", []) if isinstance(traceability.get("documentation_paths"), list) else []
  source_artifacts = traceability.get("source_artifacts", []) if isinstance(traceability.get("source_artifacts"), list) else []
  fallback_signals = int(robustness.get("fallback_signals", 0) or 0)

  st.markdown("#### Traceability and Explainability Evidence")
  t1, t2, t3, t4 = st.columns(4)
  t1.metric("Decision Traces", len(decisions))
  t2.metric("Override Events", int(traceability.get("override_events", 0) or 0))
  t3.metric("Decision Artifacts", int(traceability.get("decision_artifacts", 0) or 0))
  t4.metric("Fallback Signals", fallback_signals)

  if docs:
    st.caption("Documentation paths used in this assurance view")
    st.code("\n".join(docs), language="text")

  if source_artifacts:
    st.caption("Benchmark artifacts feeding baseline and robustness scoring")
    st.code("\n".join(source_artifacts[:12]), language="text")

  breakdown = traceability.get("transparency_breakdown", {}) if isinstance(traceability.get("transparency_breakdown"), dict) else {}
  if breakdown:
    st.markdown("#### Transparency Score Decomposition")
    rows = []
    for name, payload in breakdown.items():
      if not isinstance(payload, dict):
        continue
      rows.append(
        {
          "signal": str(name),
          "weight": float(payload.get("weight", 0.0) or 0.0),
          "value": float(payload.get("value", 0.0) or 0.0),
          "contribution": float(payload.get("weight", 0.0) or 0.0) * float(payload.get("value", 0.0) or 0.0),
          "raw_count": int(float(payload.get("raw_count", 0) or 0)),
          "formula": str(payload.get("formula") or ""),
        }
      )
    if rows:
      st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

  recent_overrides = traceability.get("recent_override_samples", []) if isinstance(traceability.get("recent_override_samples"), list) else []
  if recent_overrides:
    st.markdown("#### Recent Override Decisions (Explainability Trail)")
    df_overrides = pd.DataFrame(recent_overrides)
    if "created_at" in df_overrides.columns:
      df_overrides["created_at"] = pd.to_datetime(df_overrides["created_at"], unit="s", errors="coerce")
    st.dataframe(df_overrides, use_container_width=True, hide_index=True)

  assumptions = deployability.get("assumptions", []) if isinstance(deployability.get("assumptions"), list) else []
  if assumptions:
    st.markdown("#### Deployment Assumptions (On-Prem/Sovereign/Edge/Low-Connectivity)")
    st.code("\n".join(assumptions), language="text")

  if drg_allocator:
    st.markdown("#### Dynamic Resource Allocator (DRG) Signals")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Readiness", f"{float(drg_allocator.get('readiness_score', 0.0) or 0.0) * 100:.0f}%")
    d2.metric("Activity", f"{float(drg_allocator.get('activity_score', 0.0) or 0.0) * 100:.0f}%")
    d3.metric("Runtime Files", int(drg_allocator.get("runtime_activity_files", 0) or 0))
    d4.metric("Keyword Hits", int(drg_allocator.get("runtime_keyword_hits", 0) or 0))

  rows = snapshot.get("summary_rows", []) if isinstance(snapshot.get("summary_rows"), list) else []
  if not rows:
    rows = [
      {
        "criterion": "Clarity and credibility of metrics",
        "score_pct": round(metric_meaning_score * 100.0, 1),
        "evidence": "No assurance summary rows were generated.",
        "main_gap": "Assurance summary generation returned no rows.",
      }
    ]

  st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_header():
  st.title("Developer Operations Console")
  st.caption("Live engineering view: tests, logs, edge cases, model diagnostics, and training artifacts.")


def _render_overview_tab(test_files: List[Path], run_summaries: List[Dict[str, object]], logs: List[Path]):
  test_totals = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
  for tf in test_files:
    counts = _parse_pytest_counts(_safe_read_text(tf, max_chars=250000))
    for k in test_totals:
      test_totals[k] += counts.get(k, 0)

  total_runs = len(run_summaries)
  failed_runs = sum(1 for run in run_summaries if str(run.get("status", "")).lower() != "success")
  total_run_errors = sum(int(run.get("errors_count") or 0) for run in run_summaries)

  c1, c2, c3, c4 = st.columns(4)
  c1.metric("Detected Test Passes", test_totals["passed"])
  c2.metric("Detected Test Failures", test_totals["failed"] + test_totals["errors"])
  c3.metric("Recent Training Runs", total_runs)
  c4.metric("Run Error Records", total_run_errors)

  c5, c6, c7 = st.columns(3)
  c5.metric("Recent Log Files", len(logs))
  c6.metric("Non-success Runs", failed_runs)
  c7.metric("Skipped/XFail", test_totals["skipped"] + test_totals["xfailed"])

  st.subheader("What This Page Is Rendering")
  st.markdown(
    "\n".join(
      [
        "- Backend pytest outputs in audits.",
        "- Causal/model training run summaries and effect records in artifacts/runs.",
        "- Structured simulator logs in kshiked/logs/history.",
        "- Full audit artifacts, risky patterns, and technical debt extracts.",
      ]
    )
  )

  _render_assurance_scorecard(test_files, run_summaries, logs)

  st.subheader("Pilot Simulation")
  st.caption("Run side-by-side benchmark of OnlineDiscoveryEngine modes on synthetic pilot data.")

  rows = st.slider("Pilot rows", min_value=200, max_value=5000, value=1200, step=200, key="pilot_rows")
  seed = st.number_input("Pilot random seed", min_value=1, max_value=999999, value=42, step=1, key="pilot_seed")

  def _run_pilot(mode: str, n_rows: int, random_seed: int) -> Dict[str, object]:
    rng = np.random.default_rng(random_seed)
    x = rng.normal(0.0, 1.0, n_rows)
    y = 0.85 * x + rng.normal(0.0, 0.25, n_rows)

    engine = OnlineDiscoveryEngine(mode=mode)
    schema = {"fields": [{"name": "X"}, {"name": "Y"}]}
    engine.initialize_v2(schema, use_causal=True)

    start = time.perf_counter()
    final_result = {}
    for i in range(n_rows):
      final_result = engine.process_row({"X": float(x[i]), "Y": float(y[i])})
    elapsed = max(1e-9, time.perf_counter() - start)

    return {
      "mode": mode,
      "rows": n_rows,
      "elapsed_sec": elapsed,
      "rows_per_sec": n_rows / elapsed,
      "total_hypotheses": final_result.get("total_hypotheses", 0),
      "active_hypotheses": final_result.get("active_hypotheses", 0),
      "knowledge_graph_edges": len(engine.get_knowledge_graph()),
    }

  if st.button("Run Pilot Benchmark", key="run_pilot_benchmark"):
    with st.spinner("Running balanced vs performance pilot simulation..."):
      balanced = _run_pilot("balanced", int(rows), int(seed))
      performance = _run_pilot("performance", int(rows), int(seed))
    st.session_state["pilot_results"] = {"balanced": balanced, "performance": performance}

  pilot_results = st.session_state.get("pilot_results")
  if pilot_results:
    balanced = pilot_results["balanced"]
    performance = pilot_results["performance"]

    speedup = performance["rows_per_sec"] / max(1e-9, balanced["rows_per_sec"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Balanced rows/sec", f"{balanced['rows_per_sec']:.1f}")
    c2.metric("Performance rows/sec", f"{performance['rows_per_sec']:.1f}")
    c3.metric("Speedup", f"{speedup:.2f}x")

    st.dataframe(
      pd.DataFrame([balanced, performance]),
      use_container_width=True,
      hide_index=True,
    )


def _render_backend_tests_tab(test_files: List[Path]):
  st.subheader("Backend Test Results")
  if not test_files:
    st.warning("No pytest output files were found in audits.")
    return

  summary_rows = []
  for tf in test_files:
    counts = _parse_pytest_counts(_safe_read_text(tf))
    summary_rows.append(
      {
        "file": tf.name,
        "passed": counts["passed"],
        "failed": counts["failed"],
        "errors": counts["errors"],
        "skipped": counts["skipped"],
        "xfailed": counts["xfailed"],
        "xpassed": counts["xpassed"],
      }
    )

  st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

  selected = st.selectbox("Select test output file", [p.name for p in test_files], key="dev_tests_file")
  selected_path = next(p for p in test_files if p.name == selected)
  st.code(_safe_read_text(selected_path), language="text")


def _render_model_metrics_tab(run_summaries: List[Dict[str, object]]):
  st.subheader("Model Metrics and Effect Diagnostics")
  _render_unified_model_metrics_table()
  _render_quality_override_events()

  fl_accuracy_payload = _load_latest_fl_accuracy_benchmark()
  fl_rows = fl_accuracy_payload.get("results", []) if isinstance(fl_accuracy_payload, dict) else []
  if isinstance(fl_rows, list) and fl_rows:
    leaderboard: List[Dict[str, object]] = []
    for row in fl_rows:
      if not isinstance(row, dict):
        continue
      mm = row.get("metrics_mean", {}) if isinstance(row.get("metrics_mean"), dict) else {}
      ci = row.get("metrics_ci95", {}) if isinstance(row.get("metrics_ci95"), dict) else {}
      leaderboard.append(
        {
          "model": row.get("model_name"),
          "objective": row.get("objective_metric"),
          "threshold": row.get("best_threshold_mean"),
          "balanced_accuracy": mm.get("balanced_accuracy"),
          "f1": mm.get("f1"),
          "recall": mm.get("recall"),
          "precision": mm.get("precision"),
          "tpr": mm.get("tpr"),
          "tnr": mm.get("tnr"),
          "ece": mm.get("ece"),
          "brier": mm.get("brier"),
          "f1_ci95_low": (ci.get("f1", {}) or {}).get("low"),
          "f1_ci95_high": (ci.get("f1", {}) or {}).get("high"),
        }
      )

    if leaderboard:
      ldf = pd.DataFrame(leaderboard)
      if "balanced_accuracy" in ldf.columns:
        ldf = ldf.sort_values(by="balanced_accuracy", ascending=False)

      best_bal_model = str(ldf.iloc[0]["model"]) if not ldf.empty else "n/a"
      best_bal_score = float(ldf.iloc[0]["balanced_accuracy"]) if not ldf.empty else 0.0
      best_recall_idx = ldf["recall"].astype(float).idxmax() if "recall" in ldf.columns and not ldf.empty else None
      best_recall_model = str(ldf.loc[best_recall_idx, "model"]) if best_recall_idx is not None else "n/a"
      best_recall_score = float(ldf.loc[best_recall_idx, "recall"]) if best_recall_idx is not None else 0.0
      avg_ece = float(pd.to_numeric(ldf.get("ece", pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean()) if "ece" in ldf.columns else 0.0

      st.markdown("Latest FL Accuracy Benchmark")
      c1, c2, c3 = st.columns(3)
      c1.metric("Best Balanced Accuracy", f"{best_bal_score:.3f}", best_bal_model)
      c2.metric("Best Recall", f"{best_recall_score:.3f}", best_recall_model)
      c3.metric("Average Calibration Error", f"{avg_ece:.3f}")

      st.dataframe(ldf, use_container_width=True, hide_index=True)
      st.caption(
        f"Source: {fl_accuracy_payload.get('_path', 'n/a')} ({fl_accuracy_payload.get('_mtime', 'n/a')})"
      )

  meta_payload = _load_latest_meta_accuracy_benchmark()
  meta_rows = meta_payload.get("results", []) if isinstance(meta_payload, dict) else []
  if isinstance(meta_rows, list) and meta_rows:
    mdf_rows: List[Dict[str, object]] = []
    for row in meta_rows:
      if not isinstance(row, dict):
        continue
      summary = row.get("summary", {}) if isinstance(row.get("summary"), dict) else {}
      mdf_rows.append(
        {
          "meta_model": row.get("model"),
          "sign_accuracy": summary.get("sign_accuracy_mean"),
          "transfer_gain_rate": summary.get("transfer_gain_rate_mean"),
          "rollback_accuracy": summary.get("rollback_accuracy_mean"),
          "rollback_precision": summary.get("rollback_precision_mean"),
          "rollback_recall": summary.get("rollback_recall_mean"),
        }
      )

    if mdf_rows:
      mdf = pd.DataFrame(mdf_rows)
      if "rollback_accuracy" in mdf.columns:
        mdf = mdf.sort_values(by="rollback_accuracy", ascending=False)

      st.markdown("Meta Model Accuracy")
      mc1, mc2, mc3 = st.columns(3)
      mc1.metric("Top Meta Rollback Accuracy", f"{float(mdf.iloc[0]['rollback_accuracy']):.3f}", str(mdf.iloc[0]["meta_model"]))
      mc2.metric("Top Meta Transfer Gain", f"{float(pd.to_numeric(mdf['transfer_gain_rate'], errors='coerce').fillna(0.0).max()):.3f}")
      mc3.metric("Top Meta Sign Accuracy", f"{float(pd.to_numeric(mdf['sign_accuracy'], errors='coerce').fillna(0.0).max()):.3f}")

      st.dataframe(mdf, use_container_width=True, hide_index=True)
      st.caption(
        f"Source: {meta_payload.get('_path', 'n/a')} ({meta_payload.get('_mtime', 'n/a')})"
      )

  stat_payload = _load_latest_statistical_accuracy_benchmark()
  stat_rows = stat_payload.get("results", []) if isinstance(stat_payload, dict) else []
  if isinstance(stat_rows, list) and stat_rows:
    stat_df_rows: List[Dict[str, object]] = []
    for row in stat_rows:
      if not isinstance(row, dict):
        continue
      summary = row.get("summary", {}) if isinstance(row.get("summary"), dict) else {}
      stat_df_rows.append(
        {
          "model": row.get("model"),
          "overall_accuracy": summary.get("overall_accuracy_mean"),
          "direction_accuracy": summary.get("direction_accuracy_mean"),
          "magnitude_accuracy": summary.get("magnitude_accuracy_mean"),
          "rmse": summary.get("rmse_mean"),
          "mae": summary.get("mae_mean"),
        }
      )

    if stat_df_rows:
      sdf = pd.DataFrame(stat_df_rows).sort_values(by="overall_accuracy", ascending=False)
      st.markdown("Statistical Model Accuracy")
      sc1, sc2, sc3 = st.columns(3)
      sc1.metric("Top Statistical Overall", f"{float(sdf.iloc[0]['overall_accuracy']):.3f}", str(sdf.iloc[0]["model"]))
      sc2.metric("Top Direction Accuracy", f"{float(pd.to_numeric(sdf['direction_accuracy'], errors='coerce').fillna(0.0).max()):.3f}")
      sc3.metric("Top Magnitude Accuracy", f"{float(pd.to_numeric(sdf['magnitude_accuracy'], errors='coerce').fillna(0.0).max()):.3f}")
      st.dataframe(sdf, use_container_width=True, hide_index=True)
      st.caption(
        f"Source: {stat_payload.get('_path', 'n/a')} ({stat_payload.get('_mtime', 'n/a')})"
      )

  online_payload = _load_latest_online_accuracy_benchmark()
  online_rows = online_payload.get("results", []) if isinstance(online_payload, dict) else []
  if isinstance(online_rows, list) and online_rows:
    online_df_rows: List[Dict[str, object]] = []
    for row in online_rows:
      if not isinstance(row, dict):
        continue
      summary = row.get("summary", {}) if isinstance(row.get("summary"), dict) else {}
      online_df_rows.append(
        {
          "model": row.get("model"),
          "accuracy_like": summary.get("accuracy_like_mean"),
          "fit_score": summary.get("fit_score_mean"),
          "mae": summary.get("mae_mean"),
          "parameter_error": summary.get("parameter_error_mean"),
          "shift_error": summary.get("shift_error_mean"),
          "avg_latency_ms": summary.get("avg_latency_ms_mean"),
          "p99_latency_ms": summary.get("p99_latency_ms_mean"),
        }
      )

    if online_df_rows:
      odf = pd.DataFrame(online_df_rows).sort_values(by="accuracy_like", ascending=False)
      st.markdown("Online Model Accuracy")
      oc1, oc2, oc3 = st.columns(3)
      oc1.metric("Top Online Accuracy-like", f"{float(odf.iloc[0]['accuracy_like']):.3f}", str(odf.iloc[0]["model"]))
      oc2.metric("Best Online Fit Score", f"{float(pd.to_numeric(odf['fit_score'], errors='coerce').fillna(0.0).max()):.3f}")
      latency_col = pd.to_numeric(odf.get("avg_latency_ms", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
      oc3.metric("Best Avg Latency (ms)", f"{float(latency_col[latency_col > 0].min() if (latency_col > 0).any() else 0.0):.3f}")
      st.dataframe(odf, use_container_width=True, hide_index=True)
      st.caption(
        f"Source: {online_payload.get('_path', 'n/a')} ({online_payload.get('_mtime', 'n/a')})"
      )

  if not run_summaries:
    st.warning("No run summaries found in artifacts/runs.")
    return

  df = pd.DataFrame(run_summaries)
  st.dataframe(
    df[["run_id", "status", "duration_sec", "succeeded", "failed", "effects_count", "errors_count", "started_at"]],
    use_container_width=True,
    hide_index=True,
  )

  selected_run = st.selectbox("Inspect run", df["run_id"].tolist(), key="dev_metrics_run")
  run = next(item for item in run_summaries if item["run_id"] == selected_run)
  effects_path = Path(run["effects_path"])
  summary_path = Path(run["summary_path"])

  c1, c2, c3 = st.columns(3)
  c1.metric("Effects", int(run.get("effects_count") or 0))
  c2.metric("Errors", int(run.get("errors_count") or 0))
  c3.metric("Duration (sec)", run.get("duration_sec") if run.get("duration_sec") is not None else "n/a")

  if summary_path.exists():
    st.markdown("Summary JSON")
    st.code(_safe_read_text(summary_path, max_chars=200000), language="json")

  if effects_path.exists() and effects_path.stat().st_size > 0:
    parsed = []
    for line in _safe_read_text(effects_path, max_chars=350000).splitlines()[:200]:
      line = line.strip()
      if not line:
        continue
      try:
        obj = json.loads(line)
      except Exception:
        continue
      parsed.append(
        {
          "spec_id": obj.get("spec_id", "")[:100],
          "estimate": obj.get("estimate"),
          "method": obj.get("backend", {}).get("method_name"),
          "created_at": obj.get("created_at"),
        }
      )
    if parsed:
      st.markdown("Effect Estimates")
      st.dataframe(pd.DataFrame(parsed), use_container_width=True, hide_index=True)


def _render_training_runs_tab(run_summaries: List[Dict[str, object]]):
  st.subheader("Training Runs")
  if not run_summaries:
    st.warning("No training run artifacts found.")
    return

  selected_run = st.selectbox("Select run folder", [r["run_id"] for r in run_summaries], key="dev_run_selector")
  run = next(item for item in run_summaries if item["run_id"] == selected_run)
  run_dir = Path(run["run_path"])

  files = sorted([p for p in run_dir.rglob("*") if p.is_file()])
  file_rows = [{"name": p.name, "path": str(p.relative_to(ROOT)), "size": p.stat().st_size} for p in files]
  st.dataframe(pd.DataFrame(file_rows), use_container_width=True, hide_index=True)

  previewable = [p for p in files if p.suffix.lower() in {".json", ".jsonl", ".txt", ".md", ".dot"}]
  if previewable:
    selected_file = st.selectbox("Preview run file", [str(p.relative_to(ROOT)) for p in previewable], key="dev_run_file")
    file_path = ROOT / selected_file
    lang = "json" if file_path.suffix.lower() == ".json" else "text"
    st.code(_safe_read_text(file_path), language=lang)


def _render_logs_and_edge_cases_tab(log_files: List[Path]):
  st.subheader("Logs and Edge Cases")
  if not log_files:
    st.warning("No log history files found in kshiked/logs/history.")
  else:
    log_rows = []
    for lf in log_files:
      log_rows.append(
        {
          "file": lf.name,
          "size": lf.stat().st_size,
          "modified": str(pd.to_datetime(lf.stat().st_mtime, unit="s")),
        }
      )
    st.dataframe(pd.DataFrame(log_rows), use_container_width=True, hide_index=True)

    selected_log = st.selectbox("Inspect log record", [p.name for p in log_files], key="dev_log_file")
    log_path = next(p for p in log_files if p.name == selected_log)
    st.code(_safe_read_text(log_path), language="json")

  st.markdown("Edge-case and risk artifacts")
  edge_files = _list_existing(
    [
      AUDITS_DIR / "operations_risky_patterns.txt",
      AUDITS_DIR / "tech_debt_todo_fixme.txt",
      AUDITS_DIR / "constants_hardcoded_values.txt",
      AUDITS_DIR / "kshield_forensic_audit.md",
      AUDITS_DIR / "scarcity_verification_report.md",
    ]
  )
  if edge_files:
    picked = st.selectbox("Select edge-case artifact", [p.name for p in edge_files], key="dev_edge_file")
    path = next(p for p in edge_files if p.name == picked)
    st.code(_safe_read_text(path), language="text")
  else:
    st.info("No edge-case audit files detected.")


def _render_raw_artifacts_tab(audit_reports: List[Path]):
  st.subheader("Raw Artifact Browser")
  if not audit_reports:
    st.warning("No files found in audits.")
    return

  rows = []
  for p in audit_reports:
    if p.is_file():
      rows.append({"name": p.name, "size": p.stat().st_size, "path": str(p.relative_to(ROOT))})
  if rows:
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

  file_choices = [p for p in audit_reports if p.is_file() and p.suffix.lower() in {".txt", ".md", ".json", ".csv"}]
  if not file_choices:
    st.info("No previewable files in audits.")
    return

  selected = st.selectbox("Inspect raw artifact", [p.name for p in file_choices], key="dev_raw_file")
  selected_path = next(p for p in file_choices if p.name == selected)
  language = "json" if selected_path.suffix.lower() == ".json" else "text"
  st.code(_safe_read_text(selected_path), language=language)


def _discover_transparency_log_rows(run_summaries: List[Dict[str, object]], log_files: List[Path]) -> List[Dict[str, object]]:
  rows: List[Dict[str, object]] = []
  default_account = _sanitize_account_label(str(st.session_state.get("username") or "unknown_account"))

  def _infer_account_from_relative_path(path: Path) -> str:
    parts = list(path.parts)
    try:
      idx = parts.index("accounts")
      if idx + 1 < len(parts):
        return _sanitize_account_label(parts[idx + 1])
    except ValueError:
      pass
    return default_account

  def _add(source: str, path: Path):
    if not path.exists() or not path.is_file():
      return
    rel_path = path.relative_to(ROOT)
    rows.append(
      {
        "source": source,
        "name": path.name,
        "path": str(rel_path),
        "account": _infer_account_from_relative_path(rel_path),
        "size": path.stat().st_size,
        "modified": str(pd.to_datetime(path.stat().st_mtime, unit="s")),
      }
    )

  # Structured application logs.
  for lf in log_files:
    _add("streamlit_history", lf)

  # Audit reports and pytest outputs.
  for ap in _load_audit_reports():
    if ap.is_file():
      _add("audits", ap)

  # Federated runtime logs and db artifacts.
  fed_files = [
    FED_RUNTIME_DIR / "audit_log.jsonl",
    FED_RUNTIME_DIR / "federation_control.sqlite",
  ]
  for ff in fed_files:
    _add("federation_runtime", ff)

  # Published benchmark outputs (FL, meta, statistical, online, drift).
  if BENCHMARKS_DIR.exists() and BENCHMARKS_DIR.is_dir():
    benchmark_files = sorted(
      [p for p in BENCHMARKS_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".json", ".csv", ".md"}],
      key=lambda p: p.stat().st_mtime,
      reverse=True,
    )
    for bf in benchmark_files[:80]:
      _add("benchmarks", bf)

  # Run-level logs and diagnostics.
  for run in run_summaries:
    for key in ("summary_path", "effects_path", "errors_path"):
      value = run.get(key)
      if value:
        _add("artifacts_runs", Path(str(value)))

  rows.sort(key=lambda r: r["modified"], reverse=True)
  return rows


def _sanitize_account_label(value: str) -> str:
  cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value or "").strip())
  cleaned = re.sub(r"_+", "_", cleaned).strip("_")
  return cleaned or "unknown_account"


def _current_account_context() -> Dict[str, object]:
  username = st.session_state.get("username")
  user_id = st.session_state.get("user_id")
  role = st.session_state.get("role")
  basket_id = st.session_state.get("basket_id")
  institution_id = st.session_state.get("institution_id")
  return {
    "username": username,
    "user_id": user_id,
    "role": role,
    "basket_id": basket_id,
    "institution_id": institution_id,
  }


def _render_transparency_and_auditability_tab(run_summaries: List[Dict[str, object]], log_files: List[Path]):
  st.subheader("tracnepalaretny and autditabilty")
  st.caption("Unified access to operational logs, audit trails, test outputs, and run diagnostics.")

  rows = _discover_transparency_log_rows(run_summaries, log_files)
  if not rows:
    st.warning("No log or audit artifacts discovered.")
    return

  df = pd.DataFrame(rows)
  account_options = ["All"] + sorted(df["account"].dropna().astype(str).unique().tolist())
  selected_account = st.selectbox("Filter account", account_options, key="dev_transparency_account")
  source_options = ["All"] + sorted(df["source"].unique().tolist())
  selected_source = st.selectbox("Filter source", source_options, key="dev_transparency_source")
  name_query = st.text_input("Search file name", key="dev_transparency_search").strip().lower()

  filtered = df
  if selected_account != "All":
    filtered = filtered[filtered["account"] == selected_account]
  if selected_source != "All":
    filtered = filtered[filtered["source"] == selected_source]
  if name_query:
    filtered = filtered[filtered["name"].str.lower().str.contains(name_query, regex=False)]

  st.dataframe(filtered, use_container_width=True, hide_index=True)
  if filtered.empty:
    st.info("No files match current transparency filters.")
    return

  # Export controls for incident forensics and offline sharing.
  selectable_paths = filtered["path"].tolist()
  all_rows = _discover_transparency_log_rows(run_summaries, log_files)

  def _top_paths(source: str, limit: int) -> List[str]:
    matches = [r["path"] for r in all_rows if r.get("source") == source]
    return matches[:limit]

  def _preset_last_incident() -> List[str]:
    result = []
    result.extend(_top_paths("streamlit_history", 2))
    result.extend(_top_paths("artifacts_runs", 4))
    result.extend(_top_paths("federation_runtime", 2))
    result.extend(_top_paths("benchmarks", 4))
    return [p for p in result if p in selectable_paths]

  def _preset_federated_audit() -> List[str]:
    result = []
    result.extend(_top_paths("federation_runtime", 10))
    # Include run summaries/errors for federation-related incident context.
    result.extend([p for p in _top_paths("artifacts_runs", 20) if p.endswith("summary.json") or p.endswith("errors.jsonl")])
    result.extend(_top_paths("benchmarks", 12))
    return [p for p in result if p in selectable_paths]

  def _preset_test_evidence() -> List[str]:
    selected = []
    for r in all_rows:
      source = str(r.get("source") or "")
      if source not in {"audits", "benchmarks"}:
        continue
      name = str(r.get("name", "")).lower()
      if (
        "pytest" in name
        or "verification" in name
        or "audit_report" in name
        or "forensic" in name
        or "accuracy" in name
        or "benchmark" in name
        or "drift" in name
      ):
        if r["path"] in selectable_paths:
          selected.append(r["path"])
    return selected

  if "dev_transparency_bundle_select" not in st.session_state:
    st.session_state["dev_transparency_bundle_select"] = selectable_paths[: min(5, len(selectable_paths))]

  current_selected = st.session_state.get("dev_transparency_bundle_select", [])
  st.session_state["dev_transparency_bundle_select"] = [p for p in current_selected if p in selectable_paths]

  st.markdown("Preset export bundles")
  c1, c2, c3 = st.columns(3)
  if c1.button("Last Incident Bundle", key="dev_preset_last_incident"):
    st.session_state["dev_transparency_bundle_select"] = _preset_last_incident() or selectable_paths[: min(5, len(selectable_paths))]
    st.rerun()
  if c2.button("Full Federated Audit Bundle", key="dev_preset_fed_audit"):
    st.session_state["dev_transparency_bundle_select"] = _preset_federated_audit() or selectable_paths[: min(5, len(selectable_paths))]
    st.rerun()
  if c3.button("Test Evidence Bundle", key="dev_preset_test_evidence"):
    st.session_state["dev_transparency_bundle_select"] = _preset_test_evidence() or selectable_paths[: min(5, len(selectable_paths))]
    st.rerun()

  selected_for_bundle = st.multiselect(
    "Select files to export as bundle",
    selectable_paths,
    key="dev_transparency_bundle_select",
  )

  account_context = _current_account_context()
  default_account_label = _sanitize_account_label(str(account_context.get("username") or "unknown_account"))
  bundle_account_label = _sanitize_account_label(
    st.text_input(
      "Bundle account classification",
      value=default_account_label,
      key="dev_transparency_bundle_account",
      help="Use the originating account label so incident bundles are traceable by account.",
    )
  )

  if selected_for_bundle:
    zip_buffer = io.BytesIO()
    row_by_path = {str(r.get("path")): r for r in all_rows}
    manifest_files: List[Dict[str, object]] = []
    account_prefix = f"accounts/{bundle_account_label}"

    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
      for rel_path in selected_for_bundle:
        abs_path = ROOT / rel_path
        if abs_path.exists() and abs_path.is_file():
          source = str(row_by_path.get(rel_path, {}).get("source") or "unknown_source")
          archive_path = f"{account_prefix}/{source}/{rel_path}"
          archive.writestr(archive_path, _safe_read_text(abs_path))
          manifest_files.append(
            {
              "account": bundle_account_label,
              "source": source,
              "original_path": rel_path,
              "archive_path": archive_path,
              "size": int(abs_path.stat().st_size),
            }
          )

      archive.writestr(
        f"{account_prefix}/manifest.json",
        json.dumps(
          {
            "bundle_classification": {
              "account": bundle_account_label,
              "exported_at": str(pd.Timestamp.utcnow()),
              "session_account_context": account_context,
            },
            "files": manifest_files,
          },
          indent=2,
        ),
      )

    zip_buffer.seek(0)
    st.download_button(
      "Download Selected Log Bundle (.zip)",
      data=zip_buffer.getvalue(),
      file_name=f"transparency_audit_bundle_{bundle_account_label}.zip",
      mime="application/zip",
      key="dev_transparency_bundle_download",
    )

  selected_path = st.selectbox("Open log/audit artifact", selectable_paths, key="dev_transparency_file")
  target_path = ROOT / selected_path
  suffix = target_path.suffix.lower()

  if target_path.exists() and target_path.is_file():
    st.download_button(
      "Download This Artifact",
      data=_safe_read_text(target_path),
      file_name=target_path.name,
      mime="text/plain",
      key="dev_transparency_single_download",
    )

  if suffix == ".sqlite":
    st.info("SQLite artifact detected. Use DB tooling for table-level inspection.")
  elif suffix in {".json", ".jsonl"}:
    st.code(_safe_read_text(target_path), language="json")
  else:
    st.code(_safe_read_text(target_path), language="text")


def render():
  """Dedicated developer dashboard focused on engineering operations and diagnostics."""
  enforce_role(Role.EXECUTIVE.value)
  st.session_state["dashboard_persona"] = "developer"

  test_files = _load_test_files()
  audit_reports = _load_audit_reports()
  run_summaries = [_summarize_run(rd) for rd in _load_run_dirs(limit=25)]
  logs = _load_recent_logs(limit=40)
  misues_high = _count_high_severity_misues(run_summaries, limit=120)
  quality_overrides_24h = _count_recent_quality_overrides(hours=24)

  st.sidebar.title("Developer Navigation")
  display_tabs: List[str] = []
  tab_lookup: Dict[str, str] = {}
  for tab in DEV_TABS:
    display = tab
    if tab == "misues" and misues_high > 0:
      display = f"misues [{misues_high}]"
    if tab == "Model Metrics" and quality_overrides_24h > 0:
      display = f"Model Metrics [{quality_overrides_24h}]"
    display_tabs.append(display)
    tab_lookup[display] = tab

  selected_display = st.sidebar.radio("Open section", display_tabs, key="dev_sidebar_tab")
  active_tab = tab_lookup.get(selected_display, selected_display)

  if misues_high > 0:
    st.sidebar.markdown(
      f"<div style='color:#d92d20; font-weight:700;'>High-severity misuse alerts: {misues_high}</div>",
      unsafe_allow_html=True,
    )
  if quality_overrides_24h > 0:
    color = "#d92d20" if quality_overrides_24h >= 10 else "#b54708"
    st.sidebar.markdown(
      f"<div style='color:{color}; font-weight:700;'>Quality overrides (24h): {quality_overrides_24h}</div>",
      unsafe_allow_html=True,
    )
  st.sidebar.caption("Auto-rendering backend diagnostics from audits, logs, and artifacts.")
  if st.sidebar.button("Log out", key="dev_logout"):
    logout_user()
    st.rerun()

  _render_header()

  export_snapshot = build_quality_assurance_snapshot()
  export_overall = export_snapshot.get("overall_assurance", {}) if isinstance(export_snapshot, dict) else {}
  assurance_score = float(export_overall.get("score", 0.0) or 0.0)
  derived_severity = max(0.0, min(10.0, (1.0 - assurance_score) * 10.0))
  export_cost_snapshot = compute_cost_of_delay_kes_b(severity=derived_severity, projection_steps=4)
  render_unified_report_export(
    dashboard_name="Developer Dashboard",
    section_name=active_tab,
    metrics={
      "recent_training_runs": len(run_summaries),
      "recent_logs": len(logs),
      "high_severity_misuse_alerts": int(misues_high),
      "quality_overrides_24h": int(quality_overrides_24h),
      "assurance_score_pct": round(assurance_score * 100.0, 1),
    },
    highlights=[
      f"Current assurance score is {assurance_score * 100.0:.0f}%.",
      f"High-severity misuse alerts in recent runs: {misues_high}.",
      f"Quality overrides observed in last 24h: {quality_overrides_24h}.",
    ],
    interpretations=[
      "Lower assurance means model governance and deployment controls need immediate review.",
      "Misuse alerts indicate potential policy or operational safety violations.",
      "Frequent overrides can signal process drift and reduced audit reliability.",
    ],
    cost_delay=export_cost_snapshot,
    tables={
      "assurance_summary": pd.DataFrame(export_snapshot.get("summary_rows", []))
      if isinstance(export_snapshot, dict)
      else pd.DataFrame(),
      "recent_training_runs": pd.DataFrame(run_summaries) if isinstance(run_summaries, list) else pd.DataFrame(),
      "recent_log_files": pd.DataFrame([{"path": str(p)} for p in logs]) if isinstance(logs, list) else pd.DataFrame(),
    },
    evidence={"overall_assurance": export_overall},
    key_prefix="dev_unified_report",
  )

  if active_tab == "Developer Overview":
    _render_overview_tab(test_files, run_summaries, logs)
  elif active_tab == "Backend Tests":
    _render_backend_tests_tab(test_files)
  elif active_tab == "Model Metrics":
    _render_model_metrics_tab(run_summaries)
  elif active_tab == "Training Runs":
    _render_training_runs_tab(run_summaries)
  elif active_tab == "Logs & Edge Cases":
    _render_logs_and_edge_cases_tab(logs)
  elif active_tab == "Raw Artifacts":
    _render_raw_artifacts_tab(audit_reports)
  elif active_tab == "tracnepalaretny and autditabilty":
    _render_transparency_and_auditability_tab(run_summaries, logs)
  elif active_tab == "summary":
    _render_summary_tab(test_files, run_summaries, logs)
  elif active_tab == "misues":
    _render_misues_tab(run_summaries, logs)
