import json
import time
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[4]
BENCHMARKS_DIR = ROOT / "artifacts" / "benchmarks"
QUALITY_OVERRIDE_LOG_PATH = ROOT / "audits" / "quality_override_events.jsonl"
ARTIFACTS_RUNS_DIR = ROOT / "artifacts" / "runs"
AUDITS_DIR = ROOT / "audits"
LOGS_DIR = ROOT / "logs"

FAMILY_BASELINE_COMPOSITE = {
  "fl": 0.62,
  "meta": 0.65,
  "statistical": 0.63,
  "online": 0.67,
}

TRACEABILITY_DOC_CANDIDATES = [
  ROOT / "README.md",
  ROOT / "ARCHITECTURE.md",
  ROOT / "AUDIT_REPORT.md",
  ROOT / "validation_report.md",
  ROOT / "documentation" / "SCARCITY_ARCHITECTURE.md",
  ROOT / "documentation" / "SIMULATION_ENGINE.md",
]

DEPLOYABILITY_DOC_CANDIDATES = [
  ROOT / "documentation" / "CONFIG_REFERENCE.md",
  ROOT / "documentation" / "SCARCITY_VS_SENTINEL.md",
  ROOT / "documentation" / "SCARCITY_ARCHITECTURE.md",
  ROOT / "ARCHITECTURE.md",
]

DRG_DOC_CANDIDATES = [
  ROOT / "documentation" / "CONFIG_REFERENCE.md",
  ROOT / "documentation" / "SCARCITY_ARCHITECTURE.md",
  ROOT / "documentation" / "scarcity-docs" / "governor" / "00_overview.md",
  ROOT / "documentation" / "scarcity-docs" / "engine" / "18_resource_manager.md",
]

INSTITUTION_DB_PATH = ROOT / "kshiked" / "ui" / "institution" / "backend" / "federated_registry.sqlite"
DRG_CORE_PATH = ROOT / "scarcity" / "governor" / "drg_core.py"
INSTITUTION_SCARCITY_BRIDGE_PATH = ROOT / "kshiked" / "ui" / "institution" / "backend" / "scarcity_bridge.py"
SHARED_SCARCITY_BRIDGE_PATH = ROOT / "kshiked" / "core" / "scarcity_bridge.py"
DRG_LOG_DIR = ROOT / "logs" / "drg"


def _safe_read_text(path: Path, max_chars: int = 800000) -> str:
  try:
    txt = path.read_text(encoding="utf-8", errors="replace")
  except Exception:
    return ""
  if len(txt) > max_chars:
    return txt[-max_chars:]
  return txt


def _latest_benchmark_payload(pattern: str) -> Tuple[Dict[str, object], str, str]:
  if not BENCHMARKS_DIR.exists():
    return {}, "", ""
  candidates = sorted(BENCHMARKS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
  if not candidates:
    return {}, "", ""
  target = candidates[0]
  try:
    payload = json.loads(_safe_read_text(target))
  except Exception:
    payload = {}
  rel = str(target.relative_to(ROOT))
  mtime = str(target.stat().st_mtime)
  return payload, rel, mtime


def _clamp01(value: float) -> float:
  if value < 0.0:
    return 0.0
  if value > 1.0:
    return 1.0
  return value


def _as_float(value, default: float = 0.0) -> float:
  try:
    return float(value)
  except Exception:
    return float(default)


def _score_band(score: float) -> str:
  s = _clamp01(_as_float(score, 0.0))
  if s >= 0.8:
    return "strong"
  if s >= 0.6:
    return "moderate"
  if s >= 0.4:
    return "weak"
  return "critical_gap"


def _traffic_light(score: float) -> str:
  s = _clamp01(_as_float(score, 0.0))
  if s >= 0.75:
    return "green"
  if s >= 0.5:
    return "amber"
  return "red"


def _loss_to_score(loss_value: float) -> float:
  # Smoothly maps positive losses to [0,1], where lower loss is better.
  return 1.0 / (1.0 + max(0.0, loss_value))


def _latency_to_score(latency_ms: float) -> float:
  # Penalize higher latency with a soft decay against a 1000ms reference.
  return 1.0 / (1.0 + max(0.0, latency_ms) / 1000.0)


def _composite_from_row(row: Dict[str, object]) -> float:
  terms: List[float] = []

  for key in ("balanced_accuracy", "accuracy", "f1", "recall", "precision", "overall_accuracy", "direction_accuracy", "magnitude_accuracy", "sign_accuracy", "transfer_gain_rate", "rollback_accuracy", "accuracy_like", "fit_score"):
    if key in row and row.get(key) is not None:
      terms.append(_clamp01(_as_float(row.get(key), 0.0)))

  for key in ("ece", "brier", "rmse", "mae", "parameter_error", "shift_error"):
    if key in row and row.get(key) is not None:
      terms.append(_loss_to_score(_as_float(row.get(key), 0.0)))

  if row.get("avg_latency_ms") is not None:
    terms.append(_latency_to_score(_as_float(row.get("avg_latency_ms"), 0.0)))

  if not terms:
    return 0.0
  return float(sum(terms) / len(terms))


def _fl_rows() -> List[Dict[str, object]]:
  payload, src_path, _ = _latest_benchmark_payload("fl_model_accuracy_*.json")
  rows = payload.get("results", []) if isinstance(payload, dict) else []
  out: List[Dict[str, object]] = []
  if not isinstance(rows, list):
    return out

  for row in rows:
    if not isinstance(row, dict):
      continue
    mm = row.get("metrics_mean", {}) if isinstance(row.get("metrics_mean"), dict) else {}
    rec = {
      "family": "fl",
      "model": row.get("model_name"),
      "accuracy": mm.get("accuracy"),
      "balanced_accuracy": mm.get("balanced_accuracy"),
      "f1": mm.get("f1"),
      "recall": mm.get("recall"),
      "precision": mm.get("precision"),
      "ece": mm.get("ece"),
      "brier": mm.get("brier"),
      "source_artifact": src_path,
      "data_status": "ok",
    }
    rec["composite_score"] = _composite_from_row(rec)
    out.append(rec)
  return out


def _meta_rows() -> List[Dict[str, object]]:
  payload, src_path, _ = _latest_benchmark_payload("meta_model_accuracy_*.json")
  rows = payload.get("results", []) if isinstance(payload, dict) else []
  out: List[Dict[str, object]] = []
  if not isinstance(rows, list):
    return out

  for row in rows:
    if not isinstance(row, dict):
      continue
    summary = row.get("summary", {}) if isinstance(row.get("summary"), dict) else {}
    rec = {
      "family": "meta",
      "model": row.get("model"),
      "sign_accuracy": summary.get("sign_accuracy_mean"),
      "transfer_gain_rate": summary.get("transfer_gain_rate_mean"),
      "rollback_accuracy": summary.get("rollback_accuracy_mean"),
      "precision": summary.get("rollback_precision_mean"),
      "recall": summary.get("rollback_recall_mean"),
      "source_artifact": src_path,
      "data_status": "ok",
    }
    rec["composite_score"] = _composite_from_row(rec)
    out.append(rec)
  return out


def _stat_rows() -> List[Dict[str, object]]:
  payload, src_path, _ = _latest_benchmark_payload("statistical_model_accuracy_*.json")
  rows = payload.get("results", []) if isinstance(payload, dict) else []
  out: List[Dict[str, object]] = []
  if not isinstance(rows, list):
    return out

  for row in rows:
    if not isinstance(row, dict):
      continue
    summary = row.get("summary", {}) if isinstance(row.get("summary"), dict) else {}
    rec = {
      "family": "statistical",
      "model": row.get("model"),
      "overall_accuracy": summary.get("overall_accuracy_mean"),
      "direction_accuracy": summary.get("direction_accuracy_mean"),
      "magnitude_accuracy": summary.get("magnitude_accuracy_mean"),
      "rmse": summary.get("rmse_mean"),
      "mae": summary.get("mae_mean"),
      "source_artifact": src_path,
      "data_status": "ok",
    }
    rec["composite_score"] = _composite_from_row(rec)
    out.append(rec)
  return out


def _online_rows() -> List[Dict[str, object]]:
  payload, src_path, _ = _latest_benchmark_payload("online_model_accuracy_*.json")
  rows = payload.get("results", []) if isinstance(payload, dict) else []
  out: List[Dict[str, object]] = []
  if not isinstance(rows, list):
    return out

  for row in rows:
    if not isinstance(row, dict):
      continue
    summary = row.get("summary", {}) if isinstance(row.get("summary"), dict) else {}
    rec = {
      "family": "online",
      "model": row.get("model"),
      "accuracy_like": summary.get("accuracy_like_mean"),
      "fit_score": summary.get("fit_score_mean"),
      "mae": summary.get("mae_mean"),
      "parameter_error": summary.get("parameter_error_mean"),
      "shift_error": summary.get("shift_error_mean"),
      "avg_latency_ms": summary.get("avg_latency_ms_mean"),
      "source_artifact": src_path,
      "data_status": "ok",
    }
    rec["composite_score"] = _composite_from_row(rec)
    out.append(rec)
  return out


def _fl_registry_placeholders(existing: List[Dict[str, object]]) -> List[Dict[str, object]]:
  known = {str(r.get("model")) for r in existing if str(r.get("family")) == "fl"}
  try:
    from federated_databases.model_registry import FLModelRegistry
    all_models = FLModelRegistry.list_models()
  except Exception:
    all_models = []

  placeholders: List[Dict[str, object]] = []
  for model in all_models:
    model_name = str(model)
    if model_name in known:
      continue
    placeholders.append(
      {
        "family": "fl",
        "model": model_name,
        "source_artifact": "",
        "data_status": "no_benchmark_data",
        "composite_score": 0.0,
      }
    )
  return placeholders


def build_unified_model_metrics() -> List[Dict[str, object]]:
  rows: List[Dict[str, object]] = []
  rows.extend(_fl_rows())
  rows.extend(_meta_rows())
  rows.extend(_stat_rows())
  rows.extend(_online_rows())
  rows.extend(_fl_registry_placeholders(rows))

  rows.sort(key=lambda r: (str(r.get("family")), -_as_float(r.get("composite_score"), 0.0), str(r.get("model") or "")))
  return rows


def get_family_quality_rows(family: str) -> List[Dict[str, object]]:
  fam = str(family or "").strip().lower()
  rows = [r for r in build_unified_model_metrics() if str(r.get("family", "")).lower() == fam]
  rows.sort(
    key=lambda r: (
      0 if str(r.get("data_status", "")).lower() == "ok" else 1,
      -_as_float(r.get("composite_score"), 0.0),
      str(r.get("model") or ""),
    )
  )
  return rows


def get_recommended_model_name(family: str) -> Tuple[str, str]:
  rows = get_family_quality_rows(family)
  if not rows:
    return "", "No discovered models found for this family."

  ok_rows = [r for r in rows if str(r.get("data_status", "")).lower() == "ok"]
  if ok_rows:
    best = ok_rows[0]
    score = _as_float(best.get("composite_score"), 0.0)
    return str(best.get("model") or ""), f"Top composite quality score ({score:.3f}) from latest benchmarks."

  best = rows[0]
  return str(best.get("model") or ""), "No benchmark metrics available; using discovered model fallback."


def get_quality_routing_defaults() -> List[Dict[str, str]]:
  out: List[Dict[str, str]] = []
  for family in ("fl", "meta", "statistical", "online"):
    model, reason = get_recommended_model_name(family)
    status = "ok" if model else "missing"
    out.append(
      {
        "family": family,
        "recommended_model": model or "n/a",
        "status": status,
        "reason": reason,
      }
    )
  return out


def _latest_payload_rows(pattern: str) -> Tuple[List[Dict[str, object]], str]:
  payload, src_path, _ = _latest_benchmark_payload(pattern)
  if not isinstance(payload, dict):
    return [], ""
  rows = payload.get("results", [])
  if not isinstance(rows, list):
    return [], src_path
  filtered = [r for r in rows if isinstance(r, dict)]
  return filtered, src_path


def _latest_drift_summary_rows() -> List[Dict[str, object]]:
  out: List[Dict[str, object]] = []
  for mode in ("balanced", "performance"):
    payload, src_path, _ = _latest_benchmark_payload(f"engine_drift_{mode}_*.json")
    results = payload.get("results", {}) if isinstance(payload, dict) else {}
    if not isinstance(results, dict):
      continue
    for magnitude, entry in results.items():
      if not isinstance(entry, dict):
        continue
      summary = entry.get("summary", {}) if isinstance(entry.get("summary"), dict) else {}
      out.append(
        {
          "mode": mode,
          "magnitude": str(magnitude),
          "detect_rate": _as_float(summary.get("detect_rate"), 0.0),
          "split_rate": _as_float(summary.get("split_rate"), 0.0),
          "latency_mean": _as_float(summary.get("latency_mean"), 0.0),
          "latency_p95": _as_float(summary.get("latency_p95"), 0.0),
          "avg_pressure_delta": _as_float(summary.get("avg_pressure_delta"), 0.0),
          "source_artifact": src_path,
        }
      )
  return out


def _count_fallback_signals(max_files: int = 80) -> int:
  keywords = ("fallback", "degrad", "graceful", "partial failure", "retry", "recover")
  candidates: List[Path] = []
  for root in (LOGS_DIR, AUDITS_DIR):
    if not root.exists() or not root.is_dir():
      continue
    for suffix in ("*.log", "*.txt", "*.json", "*.jsonl", "*.md"):
      candidates.extend(root.rglob(suffix))

  seen = set()
  ordered = sorted(candidates, key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
  count = 0
  scanned = 0
  for path in ordered:
    if scanned >= max_files:
      break
    if not path.exists() or not path.is_file():
      continue
    key = str(path)
    if key in seen:
      continue
    seen.add(key)
    scanned += 1
    text = _safe_read_text(path, max_chars=140000).lower()
    if any(k in text for k in keywords):
      count += 1
  return count


def _load_quality_override_events(limit: int = 5000) -> List[Dict[str, object]]:
  if not QUALITY_OVERRIDE_LOG_PATH.exists() or not QUALITY_OVERRIDE_LOG_PATH.is_file():
    return []
  events: List[Dict[str, object]] = []
  for line in _safe_read_text(QUALITY_OVERRIDE_LOG_PATH, max_chars=1200000).splitlines():
    line = str(line).strip()
    if not line:
      continue
    try:
      obj = json.loads(line)
    except Exception:
      continue
    if isinstance(obj, dict):
      events.append(obj)
    if len(events) >= limit:
      break
  events.sort(key=lambda e: _as_float(e.get("created_at"), 0.0), reverse=True)
  return events


def _recent_override_samples(events: List[Dict[str, object]], limit: int = 8) -> List[Dict[str, object]]:
  out: List[Dict[str, object]] = []
  for event in events[: max(1, int(limit))]:
    out.append(
      {
        "created_at": _as_float(event.get("created_at"), 0.0),
        "family": str(event.get("family") or ""),
        "selected": str(event.get("selected") or ""),
        "recommended": str(event.get("recommended") or ""),
        "actor": str(event.get("actor") or ""),
        "context": str(event.get("context") or ""),
        "reason": str(event.get("reason") or ""),
      }
    )
  return out


def _count_decision_artifacts(limit: int = 80) -> int:
  if not ARTIFACTS_RUNS_DIR.exists() or not ARTIFACTS_RUNS_DIR.is_dir():
    return 0
  run_dirs = [d for d in ARTIFACTS_RUNS_DIR.iterdir() if d.is_dir()]
  run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
  count = 0
  for run_dir in run_dirs[:limit]:
    effects_path = run_dir / "effects.jsonl"
    if effects_path.exists() and effects_path.is_file() and effects_path.stat().st_size > 0:
      count += 1
  return count


def _existing_documentation_paths() -> List[str]:
  out: List[str] = []
  for path in TRACEABILITY_DOC_CANDIDATES:
    if path.exists() and path.is_file():
      out.append(str(path.relative_to(ROOT)))
  return out


def _file_size_mb(path: Path) -> float:
  try:
    if path.exists() and path.is_file():
      return float(path.stat().st_size) / (1024.0 * 1024.0)
  except Exception:
    return 0.0
  return 0.0


def _recent_run_artifact_size_mb(limit: int = 40) -> float:
  if not ARTIFACTS_RUNS_DIR.exists() or not ARTIFACTS_RUNS_DIR.is_dir():
    return 0.0
  run_dirs = [d for d in ARTIFACTS_RUNS_DIR.iterdir() if d.is_dir()]
  run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
  total_mb = 0.0
  for run_dir in run_dirs[:limit]:
    for name in ("summary.json", "effects.jsonl", "errors.jsonl"):
      total_mb += _file_size_mb(run_dir / name)
  return total_mb


def _benchmark_artifact_size_mb() -> float:
  if not BENCHMARKS_DIR.exists() or not BENCHMARKS_DIR.is_dir():
    return 0.0
  total = 0.0
  for path in BENCHMARKS_DIR.glob("*"):
    total += _file_size_mb(path)
  return total


def _deployability_mode_signals() -> Dict[str, bool]:
  on_prem_ready = INSTITUTION_DB_PATH.exists()
  text = ""
  for path in DEPLOYABILITY_DOC_CANDIDATES:
    text += "\n" + _safe_read_text(path, max_chars=180000).lower()

  sovereign_signal = ("sovereign" in text) or ("on-prem" in text) or ("on prem" in text)
  edge_signal = ("edge" in text) or ("low-connectivity" in text) or ("offline" in text)
  offline_queue_signal = "asynchronous delta queue" in _safe_read_text(
    ROOT / "kshiked" / "ui" / "institution" / "backend" / "database.py",
    max_chars=220000,
  ).lower()

  return {
    "on_prem": bool(on_prem_ready),
    "sovereign_cloud": bool(sovereign_signal),
    "edge_low_connectivity": bool(edge_signal),
    "offline_queue": bool(offline_queue_signal),
  }


def _drg_signals() -> Dict[str, object]:
  drg_core_present = DRG_CORE_PATH.exists() and DRG_CORE_PATH.is_file()

  bridge_text = (
    _safe_read_text(INSTITUTION_SCARCITY_BRIDGE_PATH, max_chars=240000)
    + "\n"
    + _safe_read_text(SHARED_SCARCITY_BRIDGE_PATH, max_chars=240000)
  ).lower()
  bridge_connected = (
    "dynamicresourcegovernor" in bridge_text
    or "resource governor connected" in bridge_text
    or "scarcity.drg_extension_profile" in bridge_text
  )

  docs_text = ""
  existing_docs = 0
  for path in DRG_DOC_CANDIDATES:
    if path.exists() and path.is_file():
      existing_docs += 1
    docs_text += "\n" + _safe_read_text(path, max_chars=220000).lower()

  config_documented = (
    "scarce_scarcity_drg_enabled" in docs_text
    and "scarce_scarcity_drg_control_interval" in docs_text
    and "scarce_scarcity_drg_cpu_threshold" in docs_text
    and "scarce_scarcity_drg_memory_threshold" in docs_text
  )

  runtime_activity_files = 0
  runtime_keyword_hits = 0
  runtime_candidates: List[Path] = []

  if DRG_LOG_DIR.exists() and DRG_LOG_DIR.is_dir():
    for suffix in ("*.log", "*.txt", "*.json", "*.jsonl", "*.md"):
      runtime_candidates.extend(DRG_LOG_DIR.rglob(suffix))

  if ARTIFACTS_RUNS_DIR.exists() and ARTIFACTS_RUNS_DIR.is_dir():
    run_dirs = [d for d in ARTIFACTS_RUNS_DIR.iterdir() if d.is_dir()]
    run_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for run_dir in run_dirs[:40]:
      runtime_candidates.extend([
        run_dir / "effects.jsonl",
        run_dir / "errors.jsonl",
        run_dir / "summary.json",
      ])

  seen = set()
  keywords = ("drg", "resource governor", "scarcity.drg_", "throttle", "backpressure")
  for path in runtime_candidates:
    if not path.exists() or not path.is_file():
      continue
    key = str(path)
    if key in seen:
      continue
    seen.add(key)
    runtime_activity_files += 1
    text = _safe_read_text(path, max_chars=160000).lower()
    runtime_keyword_hits += sum(1 for kw in keywords if kw in text)

  readiness_score = _clamp01(
    0.40 * (1.0 if drg_core_present else 0.0)
    + 0.30 * (1.0 if bridge_connected else 0.0)
    + 0.20 * (1.0 if config_documented else 0.0)
    + 0.10 * _clamp01(existing_docs / max(1.0, float(len(DRG_DOC_CANDIDATES))))
  )
  activity_score = _clamp01(runtime_keyword_hits / 8.0)
  drg_score = _clamp01(0.70 * readiness_score + 0.30 * activity_score)

  return {
    "score": drg_score,
    "readiness_score": readiness_score,
    "activity_score": activity_score,
    "drg_core_present": bool(drg_core_present),
    "bridge_connected": bool(bridge_connected),
    "config_documented": bool(config_documented),
    "documentation_count": int(existing_docs),
    "runtime_activity_files": int(runtime_activity_files),
    "runtime_keyword_hits": int(runtime_keyword_hits),
  }


def build_quality_assurance_snapshot() -> Dict[str, object]:
  quality_rows = build_unified_model_metrics()
  ok_rows = [r for r in quality_rows if str(r.get("data_status", "")).lower() == "ok"]

  baseline_rows: List[Dict[str, object]] = []
  family_rows: Dict[str, List[Dict[str, object]]] = {}
  for row in ok_rows:
    family = str(row.get("family") or "").strip().lower()
    if not family:
      continue
    family_rows.setdefault(family, []).append(row)

  for family, baseline in FAMILY_BASELINE_COMPOSITE.items():
    rows = family_rows.get(family, [])
    if not rows:
      baseline_rows.append(
        {
          "family": family,
          "model": "n/a",
          "composite_score": 0.0,
          "baseline_score": baseline,
          "delta_vs_baseline": -baseline,
          "status": "missing",
          "source_artifact": "",
        }
      )
      continue
    best = sorted(rows, key=lambda r: _as_float(r.get("composite_score"), 0.0), reverse=True)[0]
    score = _as_float(best.get("composite_score"), 0.0)
    delta = score - baseline
    baseline_rows.append(
      {
        "family": family,
        "model": str(best.get("model") or "n/a"),
        "composite_score": score,
        "baseline_score": baseline,
        "delta_vs_baseline": delta,
        "status": "above_baseline" if delta >= 0.0 else "below_baseline",
        "source_artifact": str(best.get("source_artifact") or ""),
      }
    )

  baseline_win_rate = (
    sum(1 for row in baseline_rows if str(row.get("status")) == "above_baseline") / max(1, len(baseline_rows))
  )
  mean_delta = sum(_as_float(row.get("delta_vs_baseline"), 0.0) for row in baseline_rows) / max(1, len(baseline_rows))
  baseline_lift_score = _clamp01((mean_delta + 0.25) / 0.5)
  benchmark_coverage = len(ok_rows) / max(1, len(quality_rows))
  calibration_coverage = (
    sum(
      1
      for row in ok_rows
      if any(row.get(k) is not None for k in ("ece", "brier", "rmse", "mae", "avg_latency_ms", "parameter_error", "shift_error"))
    )
    / max(1, len(ok_rows))
  )
  metric_credibility_score = (
    0.35 * benchmark_coverage
    + 0.35 * baseline_win_rate
    + 0.20 * calibration_coverage
    + 0.10 * baseline_lift_score
  )
  metric_breakdown = {
    "benchmark_coverage": {
      "weight": 0.35,
      "value": _clamp01(benchmark_coverage),
      "formula": "ok_rows / total_rows",
    },
    "baseline_win_rate": {
      "weight": 0.35,
      "value": _clamp01(baseline_win_rate),
      "formula": "families_above_baseline / total_families",
    },
    "calibration_coverage": {
      "weight": 0.20,
      "value": _clamp01(calibration_coverage),
      "formula": "rows_with_calibration_or_error_metrics / ok_rows",
    },
    "baseline_lift_score": {
      "weight": 0.10,
      "value": _clamp01(baseline_lift_score),
      "formula": "clamp((mean_delta_vs_baseline + 0.25) / 0.5)",
    },
  }

  drift_rows = _latest_drift_summary_rows()
  detect_rate = sum(_as_float(r.get("detect_rate"), 0.0) for r in drift_rows) / max(1, len(drift_rows))
  split_rate = sum(_as_float(r.get("split_rate"), 0.0) for r in drift_rows) / max(1, len(drift_rows))
  latency_p95 = sum(_as_float(r.get("latency_p95"), 0.0) for r in drift_rows) / max(1, len(drift_rows))
  latency_score = _clamp01(1.0 / (1.0 + (latency_p95 / 40.0)))
  drift_score = 0.45 * detect_rate + 0.35 * split_rate + 0.20 * latency_score

  fallback_signals = _count_fallback_signals(max_files=80)
  fallback_signal_score = _clamp01(fallback_signals / 12.0)
  robustness_score = 0.70 * drift_score + 0.30 * fallback_signal_score
  robustness_breakdown = {
    "drift_score": {
      "weight": 0.70,
      "value": _clamp01(drift_score),
      "formula": "0.45*detect_rate + 0.35*split_rate + 0.20*latency_score",
    },
    "fallback_signal_score": {
      "weight": 0.30,
      "value": _clamp01(fallback_signal_score),
      "formula": "min(fallback_signals/12, 1.0)",
    },
  }

  override_events = _load_quality_override_events(limit=5000)
  override_signal_score = _clamp01(len(override_events) / 30.0)
  decision_artifacts = _count_decision_artifacts(limit=80)
  decision_signal_score = _clamp01(decision_artifacts / 20.0)
  docs = _existing_documentation_paths()
  documentation_score = _clamp01(len(docs) / 6.0)
  traceability_score = 0.35 * override_signal_score + 0.35 * decision_signal_score + 0.30 * documentation_score
  transparency_breakdown = {
    "override_signal": {
      "weight": 0.35,
      "value": _clamp01(override_signal_score),
      "raw_count": len(override_events),
      "formula": "min(override_events/30, 1.0)",
    },
    "decision_signal": {
      "weight": 0.35,
      "value": _clamp01(decision_signal_score),
      "raw_count": decision_artifacts,
      "formula": "min(decision_artifacts/20, 1.0)",
    },
    "documentation_signal": {
      "weight": 0.30,
      "value": _clamp01(documentation_score),
      "raw_count": len(docs),
      "formula": "min(documentation_paths/6, 1.0)",
    },
  }
  recent_override_rows = _recent_override_samples(override_events, limit=10)

  online_latency_values = [
    _as_float(r.get("avg_latency_ms"), 0.0)
    for r in ok_rows
    if str(r.get("family", "")).lower() == "online" and r.get("avg_latency_ms") is not None
  ]
  online_latency_mean = sum(online_latency_values) / max(1, len(online_latency_values))
  online_latency_score = _clamp01(1.0 / (1.0 + (online_latency_mean / 40.0)))

  benchmark_mb = _benchmark_artifact_size_mb()
  run_artifacts_mb = _recent_run_artifact_size_mb(limit=40)
  db_mb = _file_size_mb(INSTITUTION_DB_PATH)
  total_storage_mb = benchmark_mb + run_artifacts_mb + db_mb
  storage_score = _clamp01(1.0 / (1.0 + (total_storage_mb / 512.0)))

  mode_signals = _deployability_mode_signals()
  drg = _drg_signals()
  mode_score = (
    sum(1.0 for value in mode_signals.values() if value)
    / max(1.0, float(len(mode_signals)))
  )
  connectivity_score = (
    0.45 * fallback_signal_score
    + 0.35 * (1.0 if mode_signals.get("offline_queue") else 0.0)
    + 0.20 * latency_score
  )
  deployment_realism_score = (
    0.28 * mode_score
    + 0.20 * online_latency_score
    + 0.15 * storage_score
    + 0.15 * connectivity_score
    + 0.22 * _as_float(drg.get("score"), 0.0)
  )
  deployment_breakdown = {
    "mode_score": {
      "weight": 0.28,
      "value": _clamp01(mode_score),
      "formula": "enabled_mode_signals / total_mode_signals",
    },
    "online_latency_score": {
      "weight": 0.20,
      "value": _clamp01(online_latency_score),
      "formula": "1 / (1 + online_latency_mean_ms/40)",
    },
    "storage_score": {
      "weight": 0.15,
      "value": _clamp01(storage_score),
      "formula": "1 / (1 + storage_total_mb/512)",
    },
    "connectivity_score": {
      "weight": 0.15,
      "value": _clamp01(connectivity_score),
      "formula": "0.45*fallback_signal + 0.35*offline_queue + 0.20*latency_score",
    },
    "drg_allocator_score": {
      "weight": 0.22,
      "value": _clamp01(_as_float(drg.get("score"), 0.0)),
      "formula": "0.70*drg_readiness + 0.30*drg_activity",
    },
  }

  deployment_assumptions = [
    f"on_prem_supported={mode_signals.get('on_prem', False)}",
    f"sovereign_cloud_documented={mode_signals.get('sovereign_cloud', False)}",
    f"edge_or_low_connectivity_documented={mode_signals.get('edge_low_connectivity', False)}",
    f"offline_queue_supported={mode_signals.get('offline_queue', False)}",
    f"online_latency_mean_ms={online_latency_mean:.2f}",
    f"drift_latency_p95_steps={latency_p95:.2f}",
    f"storage_total_mb={total_storage_mb:.1f}",
    f"drg_ready={bool(drg.get('drg_core_present', False) and drg.get('bridge_connected', False))}",
    f"drg_config_documented={bool(drg.get('config_documented', False))}",
    f"drg_activity_hits={int(_as_float(drg.get('runtime_keyword_hits'), 0.0))}",
  ]

  overall_weights = {
    "metric_credibility": 0.30,
    "robustness": 0.25,
    "traceability": 0.20,
    "deployment_realism": 0.25,
  }
  overall_components = [
    {
      "criterion": "metric_credibility",
      "weight": 0.30,
      "score": _clamp01(metric_credibility_score),
      "contribution": 0.30 * _clamp01(metric_credibility_score),
    },
    {
      "criterion": "robustness",
      "weight": 0.25,
      "score": _clamp01(robustness_score),
      "contribution": 0.25 * _clamp01(robustness_score),
    },
    {
      "criterion": "traceability",
      "weight": 0.20,
      "score": _clamp01(traceability_score),
      "contribution": 0.20 * _clamp01(traceability_score),
    },
    {
      "criterion": "deployment_realism",
      "weight": 0.25,
      "score": _clamp01(deployment_realism_score),
      "contribution": 0.25 * _clamp01(deployment_realism_score),
    },
  ]
  overall_score = _clamp01(sum(_as_float(c.get("contribution"), 0.0) for c in overall_components))
  overall_light = _traffic_light(overall_score)
  if overall_light == "green":
    overall_note = "Assurance posture is ready for routine operational decisions."
  elif overall_light == "amber":
    overall_note = "Assurance posture is moderate; continue with targeted controls and review."
  else:
    overall_note = "Assurance posture is weak; restrict high-impact automation until gaps close."

  summary_rows = [
    {
      "criterion": "Clarity and credibility of metrics",
      "score_pct": round(metric_credibility_score * 100.0, 1),
      "evidence": (
        f"benchmark_coverage={benchmark_coverage:.2f}, baseline_win_rate={baseline_win_rate:.2f}, "
        f"calibration_coverage={calibration_coverage:.2f}, mean_delta_vs_baseline={mean_delta:.3f}"
      ),
      "main_gap": "Raise baseline lift in families below threshold." if baseline_win_rate < 0.75 else "Most families clear the baseline threshold.",
    },
    {
      "criterion": "Behavior under noise/edge cases/partial failure",
      "score_pct": round(robustness_score * 100.0, 1),
      "evidence": (
        f"drift_detect_rate={detect_rate:.2f}, split_rate={split_rate:.2f}, latency_p95={latency_p95:.1f}, "
        f"fallback_signals={fallback_signals}"
      ),
      "main_gap": "Increase fallback/degradation evidence in recent operational logs." if fallback_signals < 4 else "Fallback/degradation evidence present in logs.",
    },
    {
      "criterion": "Traceability and explainability",
      "score_pct": round(traceability_score * 100.0, 1),
      "evidence": (
        f"override_events={len(override_events)}, decision_artifacts={decision_artifacts}, docs={len(docs)}"
      ),
      "main_gap": "Expand per-decision rationale templates across runs." if decision_artifacts < 12 else "Decision and audit artifacts support reconstruction.",
    },
    {
      "criterion": "Operational deployment realism",
      "score_pct": round(deployment_realism_score * 100.0, 1),
      "evidence": (
        f"mode_score={mode_score:.2f}, online_latency_mean_ms={online_latency_mean:.1f}, "
        f"storage_total_mb={total_storage_mb:.1f}, connectivity_score={connectivity_score:.2f}, "
        f"drg_score={_as_float(drg.get('score'), 0.0):.2f}"
      ),
      "main_gap": "Strengthen low-connectivity operational evidence and footprint controls." if deployment_realism_score < 0.7 else "Deployment assumptions are reasonably aligned to constrained environments.",
    },
  ]

  return {
    "overall_assurance": {
      "score": overall_score,
      "band": _score_band(overall_score),
      "traffic_light": overall_light,
      "note": overall_note,
      "weights": overall_weights,
      "formula": "overall = 0.30*metric_credibility + 0.25*robustness + 0.20*traceability + 0.25*deployment_realism",
      "components": overall_components,
    },
    "metric_credibility": {
      "score": _clamp01(metric_credibility_score),
      "band": _score_band(metric_credibility_score),
      "formula": "metric_credibility = 0.35*benchmark_coverage + 0.35*baseline_win_rate + 0.20*calibration_coverage + 0.10*baseline_lift_score",
      "score_breakdown": metric_breakdown,
      "benchmark_coverage": benchmark_coverage,
      "baseline_win_rate": baseline_win_rate,
      "calibration_coverage": calibration_coverage,
      "mean_delta_vs_baseline": mean_delta,
      "baseline_rows": baseline_rows,
    },
    "robustness": {
      "score": _clamp01(robustness_score),
      "band": _score_band(robustness_score),
      "formula": "robustness = 0.70*drift_score + 0.30*fallback_signal_score",
      "score_breakdown": robustness_breakdown,
      "drift_score": _clamp01(drift_score),
      "detect_rate": detect_rate,
      "split_rate": split_rate,
      "latency_p95": latency_p95,
      "fallback_signals": fallback_signals,
      "drift_rows": drift_rows,
    },
    "traceability": {
      "score": _clamp01(traceability_score),
      "band": _score_band(traceability_score),
      "formula": "traceability = 0.35*override_signal + 0.35*decision_signal + 0.30*documentation_signal",
      "override_events": len(override_events),
      "decision_artifacts": decision_artifacts,
      "documentation_paths": docs,
      "transparency_breakdown": transparency_breakdown,
      "recent_override_samples": recent_override_rows,
      "override_log_path": str(QUALITY_OVERRIDE_LOG_PATH.relative_to(ROOT)) if QUALITY_OVERRIDE_LOG_PATH.exists() else "",
      "source_artifacts": sorted(
        {
          str(row.get("source_artifact") or "")
          for row in ok_rows
          if str(row.get("source_artifact") or "")
        }
      ),
    },
    "deployment_realism": {
      "score": _clamp01(deployment_realism_score),
      "band": _score_band(deployment_realism_score),
      "formula": "deployment_realism = 0.28*mode_score + 0.20*online_latency_score + 0.15*storage_score + 0.15*connectivity_score + 0.22*drg_allocator_score",
      "score_breakdown": deployment_breakdown,
      "mode_signals": mode_signals,
      "mode_score": _clamp01(mode_score),
      "online_latency_mean_ms": online_latency_mean,
      "online_latency_score": _clamp01(online_latency_score),
      "storage_total_mb": total_storage_mb,
      "storage_score": _clamp01(storage_score),
      "connectivity_score": _clamp01(connectivity_score),
      "dynamic_resource_allocator": drg,
      "assumptions": deployment_assumptions,
    },
    "summary_rows": summary_rows,
  }


def log_quality_override(
  family: str,
  selected: str,
  recommended: str,
  actor: str,
  context: str,
  reason: str,
  details: Dict[str, object] = None,
) -> None:
  payload = {
    "created_at": time.time(),
    "family": str(family or "").strip().lower(),
    "selected": str(selected or "").strip(),
    "recommended": str(recommended or "").strip(),
    "actor": str(actor or "system").strip(),
    "context": str(context or "").strip(),
    "reason": str(reason or "").strip(),
    "details": details or {},
  }

  try:
    QUALITY_OVERRIDE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with QUALITY_OVERRIDE_LOG_PATH.open("a", encoding="utf-8") as f:
      f.write(json.dumps(payload, ensure_ascii=True) + "\n")
  except Exception:
    # Best-effort logging only; do not break runtime flows.
    return
