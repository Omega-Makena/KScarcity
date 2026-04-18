import json
import time
from typing import Any, Dict, List

from kshiked.ui.institution.backend.database import get_connection


class ResourceControlManager:
  """Persistent policy manager for model governance and spoke compute allocations."""

  DEFAULT_POLICY = {
    "default_cpu_cores": 1.0,
    "default_memory_gb": 2.0,
    "default_gpu_units": 0.0,
    "default_daily_training_rounds": 3,
    "default_max_rows_per_round": 12000,
    "default_priority_tier": "normal",
    "allowed_models": [],
    "blocked_models": [],
    "enforce_model_allowlist": True,
    "auto_sync_directives": True,
    "fair_share_mode": "balanced",
  }

  @staticmethod
  def _ensure_tables() -> None:
    with get_connection() as conn:
      c = conn.cursor()
      c.execute("""
        CREATE TABLE IF NOT EXISTS resource_control_policies (
          basket_id INTEGER PRIMARY KEY,
          policy_json TEXT NOT NULL,
          updated_at REAL NOT NULL,
          updated_by TEXT,
          FOREIGN KEY (basket_id) REFERENCES baskets (id)
        )
      """)
      c.execute("""
        CREATE TABLE IF NOT EXISTS spoke_resource_allocations (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          basket_id INTEGER NOT NULL,
          institution_id INTEGER NOT NULL,
          allocation_json TEXT NOT NULL,
          updated_at REAL NOT NULL,
          updated_by TEXT,
          UNIQUE(basket_id, institution_id),
          FOREIGN KEY (basket_id) REFERENCES baskets (id),
          FOREIGN KEY (institution_id) REFERENCES institutions (id)
        )
      """)
      conn.commit()

  @staticmethod
  def list_available_models() -> List[str]:
    try:
      from federated_databases.model_registry import FLModelRegistry
      from kshiked.ui.institution.backend.model_quality import get_family_quality_rows
      models = [str(m) for m in FLModelRegistry.list_models()]
      models = sorted({m.strip() for m in models if m and str(m).strip()})
      if models:
        ranked = []
        for row in get_family_quality_rows("fl"):
          model = str(row.get("model") or "").strip()
          if model and model in models and model not in ranked:
            ranked.append(model)
        for model in models:
          if model not in ranked:
            ranked.append(model)
        if ranked:
          return ranked
        return models
    except Exception:
      pass
    return ["logistic", "hypothesis_ensemble", "rls_online", "bayesian_varx"]

  @staticmethod
  def _as_float(value: Any, fallback: float, min_v: float = 0.0) -> float:
    try:
      parsed = float(value)
      if parsed < min_v:
        return float(min_v)
      return parsed
    except Exception:
      return float(fallback)

  @staticmethod
  def _as_int(value: Any, fallback: int, min_v: int = 0) -> int:
    try:
      parsed = int(value)
      if parsed < min_v:
        return int(min_v)
      return parsed
    except Exception:
      return int(fallback)

  @staticmethod
  def _dedupe_strs(values: Any) -> List[str]:
    if not isinstance(values, list):
      return []
    out = []
    seen = set()
    for item in values:
      val = str(item).strip()
      if not val:
        continue
      if val not in seen:
        seen.add(val)
        out.append(val)
    return out

  @staticmethod
  def _normalize_policy(policy: Dict[str, Any], available_models: List[str]) -> Dict[str, Any]:
    src = policy or {}
    out = dict(ResourceControlManager.DEFAULT_POLICY)
    out.update(src)

    out["default_cpu_cores"] = ResourceControlManager._as_float(
      out.get("default_cpu_cores"), 1.0, min_v=0.1,
    )
    out["default_memory_gb"] = ResourceControlManager._as_float(
      out.get("default_memory_gb"), 2.0, min_v=0.25,
    )
    out["default_gpu_units"] = ResourceControlManager._as_float(
      out.get("default_gpu_units"), 0.0, min_v=0.0,
    )
    out["default_daily_training_rounds"] = ResourceControlManager._as_int(
      out.get("default_daily_training_rounds"), 3, min_v=0,
    )
    out["default_max_rows_per_round"] = ResourceControlManager._as_int(
      out.get("default_max_rows_per_round"), 12000, min_v=500,
    )

    priority = str(out.get("default_priority_tier", "normal")).strip().lower()
    if priority not in {"critical", "high", "normal", "low"}:
      priority = "normal"
    out["default_priority_tier"] = priority

    fair_share_mode = str(out.get("fair_share_mode", "balanced")).strip().lower()
    if fair_share_mode not in {"balanced", "conservative", "performance"}:
      fair_share_mode = "balanced"
    out["fair_share_mode"] = fair_share_mode

    allowed = ResourceControlManager._dedupe_strs(out.get("allowed_models"))
    blocked = ResourceControlManager._dedupe_strs(out.get("blocked_models"))

    if available_models:
      allowed = [m for m in allowed if m in available_models]
      blocked = [m for m in blocked if m in available_models]

    if not allowed:
      allowed = list(available_models)

    blocked_set = set(blocked)
    allowed = [m for m in allowed if m not in blocked_set]
    if not allowed and available_models:
      fallback = [m for m in available_models if m not in blocked_set]
      allowed = fallback if fallback else list(available_models)

    out["allowed_models"] = allowed
    out["blocked_models"] = blocked
    out["enforce_model_allowlist"] = bool(out.get("enforce_model_allowlist", True))
    out["auto_sync_directives"] = bool(out.get("auto_sync_directives", True))
    return out

  @staticmethod
  def get_policy(basket_id: int) -> Dict[str, Any]:
    ResourceControlManager._ensure_tables()
    available_models = ResourceControlManager.list_available_models()

    with get_connection() as conn:
      c = conn.cursor()
      c.execute(
        "SELECT policy_json FROM resource_control_policies WHERE basket_id = ?",
        (basket_id,),
      )
      row = c.fetchone()
      if row:
        try:
          parsed = json.loads(row["policy_json"])
        except Exception:
          parsed = {}
      else:
        parsed = {}

      normalized = ResourceControlManager._normalize_policy(parsed, available_models)

      if not row:
        c.execute(
          "INSERT INTO resource_control_policies (basket_id, policy_json, updated_at, updated_by) VALUES (?, ?, ?, ?)",
          (basket_id, json.dumps(normalized), time.time(), "system"),
        )
        conn.commit()

    return normalized

  @staticmethod
  def save_policy(basket_id: int, policy: Dict[str, Any], updated_by: str = "admin") -> Dict[str, Any]:
    ResourceControlManager._ensure_tables()
    available_models = ResourceControlManager.list_available_models()
    normalized = ResourceControlManager._normalize_policy(policy, available_models)

    with get_connection() as conn:
      c = conn.cursor()
      c.execute(
        """
        INSERT INTO resource_control_policies (basket_id, policy_json, updated_at, updated_by)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(basket_id)
        DO UPDATE SET policy_json = excluded.policy_json,
                      updated_at = excluded.updated_at,
                      updated_by = excluded.updated_by
        """,
        (basket_id, json.dumps(normalized), time.time(), updated_by),
      )
      conn.commit()

    return normalized

  @staticmethod
  def get_spoke_overrides(basket_id: int) -> Dict[int, Dict[str, Any]]:
    ResourceControlManager._ensure_tables()
    out: Dict[int, Dict[str, Any]] = {}
    with get_connection() as conn:
      c = conn.cursor()
      c.execute(
        "SELECT institution_id, allocation_json FROM spoke_resource_allocations WHERE basket_id = ?",
        (basket_id,),
      )
      rows = c.fetchall()
      for row in rows:
        try:
          payload = json.loads(row["allocation_json"])
        except Exception:
          payload = {}
        out[int(row["institution_id"])] = payload
    return out

  @staticmethod
  def _normalize_spoke_override(allocation: Dict[str, Any], policy: Dict[str, Any]) -> Dict[str, Any]:
    src = allocation or {}
    allowed_from_policy = list(policy.get("allowed_models", []))
    blocked_from_policy = set(policy.get("blocked_models", []))

    allowed_models = ResourceControlManager._dedupe_strs(src.get("allowed_models"))
    if allowed_models:
      allowed_models = [m for m in allowed_models if m in allowed_from_policy and m not in blocked_from_policy]
    else:
      allowed_models = list(allowed_from_policy)

    priority = str(src.get("priority_tier", policy.get("default_priority_tier", "normal"))).strip().lower()
    if priority not in {"critical", "high", "normal", "low"}:
      priority = "normal"

    return {
      "enabled": bool(src.get("enabled", True)),
      "cpu_cores": ResourceControlManager._as_float(
        src.get("cpu_cores", policy.get("default_cpu_cores", 1.0)),
        policy.get("default_cpu_cores", 1.0),
        min_v=0.1,
      ),
      "memory_gb": ResourceControlManager._as_float(
        src.get("memory_gb", policy.get("default_memory_gb", 2.0)),
        policy.get("default_memory_gb", 2.0),
        min_v=0.25,
      ),
      "gpu_units": ResourceControlManager._as_float(
        src.get("gpu_units", policy.get("default_gpu_units", 0.0)),
        policy.get("default_gpu_units", 0.0),
        min_v=0.0,
      ),
      "daily_training_rounds": ResourceControlManager._as_int(
        src.get("daily_training_rounds", policy.get("default_daily_training_rounds", 3)),
        policy.get("default_daily_training_rounds", 3),
        min_v=0,
      ),
      "max_rows_per_round": ResourceControlManager._as_int(
        src.get("max_rows_per_round", policy.get("default_max_rows_per_round", 12000)),
        policy.get("default_max_rows_per_round", 12000),
        min_v=500,
      ),
      "priority_tier": priority,
      "allowed_models": allowed_models,
      "note": str(src.get("note", "")).strip(),
    }

  @staticmethod
  def save_spoke_override(
    basket_id: int,
    institution_id: int,
    allocation: Dict[str, Any],
    updated_by: str = "admin",
  ) -> Dict[str, Any]:
    ResourceControlManager._ensure_tables()
    policy = ResourceControlManager.get_policy(basket_id)
    normalized = ResourceControlManager._normalize_spoke_override(allocation, policy)

    with get_connection() as conn:
      c = conn.cursor()
      c.execute(
        """
        INSERT INTO spoke_resource_allocations (basket_id, institution_id, allocation_json, updated_at, updated_by)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(basket_id, institution_id)
        DO UPDATE SET allocation_json = excluded.allocation_json,
                      updated_at = excluded.updated_at,
                      updated_by = excluded.updated_by
        """,
        (basket_id, institution_id, json.dumps(normalized), time.time(), updated_by),
      )
      conn.commit()

    return normalized

  @staticmethod
  def clear_spoke_override(basket_id: int, institution_id: int) -> None:
    ResourceControlManager._ensure_tables()
    with get_connection() as conn:
      c = conn.cursor()
      c.execute(
        "DELETE FROM spoke_resource_allocations WHERE basket_id = ? AND institution_id = ?",
        (basket_id, institution_id),
      )
      conn.commit()

  @staticmethod
  def list_spoke_metadata(basket_id: int) -> List[Dict[str, Any]]:
    ResourceControlManager._ensure_tables()

    with get_connection() as conn:
      c = conn.cursor()
      # Backward compatibility: legacy registries may not yet have institutions.trust_weight.
      try:
        c.execute(
          "SELECT id, name, trust_weight FROM institutions WHERE basket_id = ? ORDER BY name ASC",
          (basket_id,),
        )
      except Exception:
        c.execute(
          "SELECT id, name, 1.0 AS trust_weight FROM institutions WHERE basket_id = ? ORDER BY name ASC",
          (basket_id,),
        )
      institutions = [dict(row) for row in c.fetchall()]

      c.execute(
        """
        SELECT
          institution_id,
          SUM(CASE WHEN status = 'PENDING' THEN 1 ELSE 0 END) AS pending_reports,
          SUM(CASE WHEN status = 'PROCESSED' THEN 1 ELSE 0 END) AS processed_reports,
          SUM(CASE WHEN status = 'REJECTED' THEN 1 ELSE 0 END) AS rejected_reports,
          MAX(timestamp) AS last_sync_at
        FROM delta_queue
        WHERE basket_id = ?
        GROUP BY institution_id
        """,
        (basket_id,),
      )
      sync_rows = {int(r["institution_id"]): dict(r) for r in c.fetchall()}

    for inst in institutions:
      sync = sync_rows.get(int(inst["id"]), {})
      inst["pending_reports"] = int(sync.get("pending_reports") or 0)
      inst["processed_reports"] = int(sync.get("processed_reports") or 0)
      inst["rejected_reports"] = int(sync.get("rejected_reports") or 0)
      inst["last_sync_at"] = sync.get("last_sync_at")
    return institutions

  @staticmethod
  def build_effective_allocations(basket_id: int) -> List[Dict[str, Any]]:
    policy = ResourceControlManager.get_policy(basket_id)
    overrides = ResourceControlManager.get_spoke_overrides(basket_id)
    institutions = ResourceControlManager.list_spoke_metadata(basket_id)

    effective = []
    for inst in institutions:
      inst_id = int(inst["id"])
      override = overrides.get(inst_id)
      if override:
        normalized = ResourceControlManager._normalize_spoke_override(override, policy)
      else:
        normalized = {
          "enabled": True,
          "cpu_cores": float(policy.get("default_cpu_cores", 1.0)),
          "memory_gb": float(policy.get("default_memory_gb", 2.0)),
          "gpu_units": float(policy.get("default_gpu_units", 0.0)),
          "daily_training_rounds": int(policy.get("default_daily_training_rounds", 3)),
          "max_rows_per_round": int(policy.get("default_max_rows_per_round", 12000)),
          "priority_tier": str(policy.get("default_priority_tier", "normal")),
          "allowed_models": list(policy.get("allowed_models", [])),
          "note": "",
        }

      effective.append({
        "institution_id": inst_id,
        "institution_name": inst.get("name"),
        "trust_weight": float(inst.get("trust_weight") or 1.0),
        "pending_reports": int(inst.get("pending_reports") or 0),
        "processed_reports": int(inst.get("processed_reports") or 0),
        "rejected_reports": int(inst.get("rejected_reports") or 0),
        "last_sync_at": inst.get("last_sync_at"),
        "has_override": bool(override),
        "allocation": normalized,
      })

    return effective

  @staticmethod
  def build_policy_directive_text(policy: Dict[str, Any], override: Dict[str, Any] = None) -> str:
    if override:
      return (
        "Resource control update for your institution: "
        f"enabled={override.get('enabled')}, "
        f"cpu={override.get('cpu_cores')} cores, "
        f"memory={override.get('memory_gb')} GB, "
        f"gpu={override.get('gpu_units')} units, "
        f"daily_rounds={override.get('daily_training_rounds')}, "
        f"max_rows={override.get('max_rows_per_round')}, "
        f"priority={override.get('priority_tier')}, "
        f"allowed_models={override.get('allowed_models', [])}."
      )

    return (
      "Sector resource control baseline updated: "
      f"default_cpu={policy.get('default_cpu_cores')} cores, "
      f"default_memory={policy.get('default_memory_gb')} GB, "
      f"default_gpu={policy.get('default_gpu_units')} units, "
      f"default_daily_rounds={policy.get('default_daily_training_rounds')}, "
      f"default_max_rows={policy.get('default_max_rows_per_round')}, "
      f"fair_share_mode={policy.get('fair_share_mode')}, "
      f"allowed_models={policy.get('allowed_models', [])}, "
      f"blocked_models={policy.get('blocked_models', [])}."
    )
