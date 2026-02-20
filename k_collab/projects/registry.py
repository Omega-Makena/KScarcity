"""Versioned registry for collaboration projects."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from k_collab.common.versioned_store import VersionedJSONStore


@dataclass
class CollaborationProject:
    """Defines participants, objective, allowed data/domains, and governance rules."""

    project_id: str
    name: str
    objective: str
    participants: List[str] = field(default_factory=list)
    allowed_datasets: List[str] = field(default_factory=list)
    allowed_domains: List[str] = field(default_factory=list)
    allowed_computations: List[str] = field(default_factory=lambda: ["analytics"])
    governance: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> Dict[str, Any]:
        return {
            "project_id": str(self.project_id).strip(),
            "name": str(self.name).strip(),
            "objective": str(self.objective).strip(),
            "participants": sorted({str(x).strip() for x in self.participants if str(x).strip()}),
            "allowed_datasets": sorted({str(x).strip() for x in self.allowed_datasets if str(x).strip()}),
            "allowed_domains": sorted({str(x).strip().lower() for x in self.allowed_domains if str(x).strip()}),
            "allowed_computations": sorted({str(x).strip().lower() for x in self.allowed_computations if str(x).strip()}),
            "governance": dict(self.governance or {}),
        }


class CollaborationProjectRegistry:
    """Stores collaboration projects with version history."""

    def __init__(self, base_dir: Path | str):
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        self._store = VersionedJSONStore(base / "project_versions.jsonl", kind="projects")

    def _current(self) -> Dict[str, Any]:
        latest = self._store.latest()
        return latest.payload if latest else {"projects": {}}

    def upsert(self, project: CollaborationProject, actor: str = "system") -> Dict[str, Any]:
        payload = self._current()
        projects = dict(payload.get("projects", {}))
        normalized = project.normalized()
        if not normalized["project_id"]:
            raise ValueError("project_id cannot be empty")
        projects[normalized["project_id"]] = normalized
        record = self._store.save({"projects": projects}, actor=actor, message=f"project:{normalized['project_id']}")
        return record.__dict__

    def get(self, project_id: str) -> Optional[Dict[str, Any]]:
        return self._current().get("projects", {}).get(str(project_id).strip())

    def all(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._current().get("projects", {}))

    def list_versions(self, limit: int = 30) -> List[Dict[str, Any]]:
        return [x.__dict__ for x in self._store.list(limit)]
