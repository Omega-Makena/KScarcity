"""Append-only audit log with chain hash for tamper-evidence."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AppendOnlyAuditLog:
    """Writes immutable-style audit events to JSONL."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        if not self.path.exists():
            return ""
        last = ""
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                last = line
        if not last:
            return ""
        try:
            payload = json.loads(last)
            return str(payload.get("chain_hash", ""))
        except Exception:
            return ""

    def append(
        self,
        event_type: str,
        actor: str,
        payload: Optional[Dict[str, Any]] = None,
        scope: str = "k_collab",
    ) -> Dict[str, Any]:
        payload = payload or {}
        event = {
            "event_type": str(event_type),
            "timestamp": _utc_now(),
            "actor": str(actor),
            "scope": str(scope),
            "payload": payload,
            "previous_hash": self._last_hash(),
        }
        hash_input = json.dumps(event, sort_keys=True, separators=(",", ":"))
        event["chain_hash"] = hashlib.sha256(hash_input.encode("utf-8")).hexdigest()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True) + "\n")
        return event

    def list(self, limit: int = 200) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        rows.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return rows[: max(0, int(limit))]
