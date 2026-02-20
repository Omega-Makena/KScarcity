"""Versioned topology persistence and schema validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from k_collab.common.versioned_store import VersionedJSONStore
from .schema import validate_topology, diff_topologies

try:
    import yaml
except Exception:  # pragma: no cover - optional
    yaml = None


class TopologyStore:
    """Stores versioned federation topology definitions."""

    def __init__(self, base_dir: Path | str = "federated_databases/runtime/k_collab"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._versions = VersionedJSONStore(self.base_dir / "topology_versions.jsonl", kind="topology")

    def load_text(self, content: str, fmt: str = "json") -> Dict[str, Any]:
        fmt_normalized = (fmt or "json").strip().lower()
        if fmt_normalized == "json":
            return json.loads(content)
        if fmt_normalized in {"yaml", "yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML not installed; YAML parsing unavailable")
            loaded = yaml.safe_load(content)
            return loaded or {}
        raise ValueError(f"Unsupported topology format: {fmt}")

    def save(self, payload: Dict[str, Any], actor: str = "system", message: str = "") -> Dict[str, Any]:
        normalized = validate_topology(payload)
        rec = self._versions.save(normalized, actor=actor, message=message)
        latest_file = self.base_dir / "topology_latest.json"
        latest_file.write_text(json.dumps(normalized, indent=2), encoding="utf-8")
        return rec.__dict__

    def latest(self) -> Optional[Dict[str, Any]]:
        rec = self._versions.latest()
        return rec.__dict__ if rec else None

    def get_payload(self, version_id: Optional[str] = None) -> Dict[str, Any]:
        rec = self._versions.get(version_id) if version_id else self._versions.latest()
        return rec.payload if rec else {"name": "empty", "nodes": [], "trust_edges": []}

    def list_versions(self, limit: int = 30) -> List[Dict[str, Any]]:
        return [r.__dict__ for r in self._versions.list(limit=limit)]

    def diff(self, old_version_id: str, new_version_id: str) -> Dict[str, Any]:
        old = self._versions.get(old_version_id)
        new = self._versions.get(new_version_id)
        if not old or not new:
            raise ValueError("Both versions must exist for diff")
        return diff_topologies(old.payload, new.payload)
