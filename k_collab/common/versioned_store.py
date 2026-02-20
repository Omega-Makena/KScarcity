"""Generic JSONL versioned config store."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class VersionRecord:
    version_id: str
    created_at: str
    actor: str
    kind: str
    payload: Dict[str, Any]
    message: str = ""


class VersionedJSONStore:
    """Append-only JSONL store with deterministic version IDs."""

    def __init__(self, path: Path | str, kind: str):
        self.path = Path(path)
        self.kind = kind
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _serialize(self, payload: Dict[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _version_id(self, payload: Dict[str, Any]) -> str:
        body = f"{self.kind}:{self._serialize(payload)}"
        return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]

    def save(self, payload: Dict[str, Any], actor: str = "system", message: str = "") -> VersionRecord:
        version_id = self._version_id(payload)
        record = VersionRecord(
            version_id=version_id,
            created_at=_utc_now(),
            actor=actor,
            kind=self.kind,
            payload=payload,
            message=message,
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.__dict__, ensure_ascii=True) + "\n")
        return record

    def list(self, limit: int = 50) -> List[VersionRecord]:
        if not self.path.exists():
            return []
        rows: List[VersionRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    rows.append(VersionRecord(**payload))
                except Exception:
                    continue
        rows.sort(key=lambda x: x.created_at, reverse=True)
        return rows[: max(0, int(limit))]

    def latest(self) -> Optional[VersionRecord]:
        items = self.list(limit=1)
        return items[0] if items else None

    def get(self, version_id: str) -> Optional[VersionRecord]:
        target = (version_id or "").strip()
        if not target or not self.path.exists():
            return None
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if payload.get("version_id") == target:
                    return VersionRecord(**payload)
        return None
