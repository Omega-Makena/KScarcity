"""Versioned data contract store."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from k_collab.common.versioned_store import VersionedJSONStore
from .models import DataContract


class DataContractRegistry:
    """Stores dataset contracts with version history."""

    def __init__(self, base_dir: Path | str):
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        self._store = VersionedJSONStore(base / "contract_versions.jsonl", kind="contracts")

    def _current(self) -> Dict[str, Any]:
        latest = self._store.latest()
        return latest.payload if latest else {"contracts": {}}

    def upsert(self, contract: DataContract, actor: str = "system") -> Dict[str, Any]:
        current = self._current()
        contracts = dict(current.get("contracts", {}))
        contracts[contract.dataset_id] = contract.normalized()
        payload = {"contracts": contracts}
        rec = self._store.save(payload, actor=actor, message=f"contract:{contract.dataset_id}")
        return rec.__dict__

    def get(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        return self._current().get("contracts", {}).get(dataset_id)

    def all(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._current().get("contracts", {}))

    def list_versions(self, limit: int = 30) -> List[Dict[str, Any]]:
        return [r.__dict__ for r in self._store.list(limit)]
