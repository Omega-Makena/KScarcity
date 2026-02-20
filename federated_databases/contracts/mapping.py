"""Canonical schema mapping and data-quality metadata for federated datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from k_collab.common.versioned_store import VersionedJSONStore


@dataclass
class CanonicalFieldMapping:
    canonical_name: str
    local_name: str
    dtype: str = "text"
    nullable: bool = True


@dataclass
class DatasetCanonicalMapping:
    dataset_id: str
    fields: List[CanonicalFieldMapping] = field(default_factory=list)
    quality: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> Dict[str, Any]:
        return {
            "dataset_id": str(self.dataset_id).strip(),
            "fields": [
                {
                    "canonical_name": f.canonical_name,
                    "local_name": f.local_name,
                    "dtype": f.dtype,
                    "nullable": bool(f.nullable),
                }
                for f in self.fields
            ],
            "quality": dict(self.quality or {}),
        }


class CanonicalSchemaRegistry:
    """Versioned canonical mapping registry."""

    def __init__(self, base_dir: Path | str):
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        self._store = VersionedJSONStore(base / "canonical_mapping_versions.jsonl", kind="canonical_mapping")

    def _current(self) -> Dict[str, Any]:
        latest = self._store.latest()
        return latest.payload if latest else {"datasets": {}}

    def upsert(self, mapping: DatasetCanonicalMapping, actor: str = "system") -> Dict[str, Any]:
        payload = self._current()
        datasets = dict(payload.get("datasets", {}))
        normalized = mapping.normalized()
        dataset_id = normalized["dataset_id"]
        if not dataset_id:
            raise ValueError("dataset_id cannot be empty")
        datasets[dataset_id] = normalized
        rec = self._store.save({"datasets": datasets}, actor=actor, message=f"canonical_mapping:{dataset_id}")
        return rec.__dict__

    def get(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        return self._current().get("datasets", {}).get(str(dataset_id).strip())

    def all(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._current().get("datasets", {}))

    def list_versions(self, limit: int = 30) -> List[Dict[str, Any]]:
        return [v.__dict__ for v in self._store.list(limit)]


def resolve_field_map(
    mapping: Optional[Dict[str, Any]],
    connector_field_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build canonical->local field map, with connector override taking precedence."""
    result: Dict[str, str] = {}
    if mapping:
        for item in mapping.get("fields", []):
            canonical = str(item.get("canonical_name", "")).strip()
            local = str(item.get("local_name", "")).strip()
            if canonical and local:
                result[canonical] = local

    for canonical, local in (connector_field_mapping or {}).items():
        c = str(canonical).strip()
        l = str(local).strip()
        if c and l:
            result[c] = l
    return result


def map_canonical_query(
    group_by: List[str],
    filters: Dict[str, Any],
    metric_field: str,
    field_map: Dict[str, str],
) -> Tuple[List[str], Dict[str, Any], str]:
    mapped_group = [field_map.get(col, col) for col in group_by]
    mapped_filters = {field_map.get(k, k): v for k, v in filters.items()}
    mapped_metric = field_map.get(metric_field, metric_field)
    return mapped_group, mapped_filters, mapped_metric


def quality_summary(rows: List[Dict[str, Any]], required_fields: List[str]) -> Dict[str, Any]:
    total = len(rows)
    if total == 0:
        return {"total_rows": 0, "completeness": 1.0, "missing_counts": {k: 0 for k in required_fields}}

    missing_counts = {k: 0 for k in required_fields}
    for row in rows:
        for key in required_fields:
            if row.get(key) in {None, ""}:
                missing_counts[key] += 1

    field_completeness = []
    for key in required_fields:
        field_completeness.append(1.0 - (missing_counts[key] / total))

    return {
        "total_rows": total,
        "completeness": float(sum(field_completeness) / len(field_completeness)) if field_completeness else 1.0,
        "missing_counts": missing_counts,
    }
