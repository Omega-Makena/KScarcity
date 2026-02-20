"""Connector and dataset registry for federated query routing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from k_collab.common.versioned_store import VersionedJSONStore
from federated_databases.connectors.base import ConnectorSpec


class FederatedCatalog:
    """Maintains node connector specs and dataset placement."""

    def __init__(self, base_dir: Path | str):
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        self._store = VersionedJSONStore(base / "catalog_versions.jsonl", kind="catalog")

    def _current(self) -> Dict[str, Any]:
        latest = self._store.latest()
        if latest:
            return latest.payload
        return {"connectors": [], "datasets": {}}

    def save(self, payload: Dict[str, Any], actor: str = "system", message: str = "") -> Dict[str, Any]:
        rec = self._store.save(payload, actor=actor, message=message)
        return rec.__dict__

    def list_versions(self, limit: int = 30) -> List[Dict[str, Any]]:
        return [r.__dict__ for r in self._store.list(limit)]

    def register_connector(self, spec: ConnectorSpec, actor: str = "system") -> Dict[str, Any]:
        current = self._current()
        connectors = [c for c in current.get("connectors", []) if c.get("connector_id") != spec.connector_id]
        connectors.append(spec.__dict__)

        datasets = dict(current.get("datasets", {}))
        for dataset_id in spec.dataset_ids:
            rows = [r for r in datasets.get(dataset_id, []) if r.get("connector_id") != spec.connector_id]
            rows.append({"connector_id": spec.connector_id, "node_id": spec.node_id})
            datasets[dataset_id] = rows

        updated = {"connectors": connectors, "datasets": datasets}
        self.save(updated, actor=actor, message=f"register_connector:{spec.connector_id}")
        return updated

    def connectors(self) -> List[Dict[str, Any]]:
        return list(self._current().get("connectors", []))

    def dataset_locations(self, dataset_id: str) -> List[Dict[str, Any]]:
        placements = list(self._current().get("datasets", {}).get(dataset_id, []))
        by_connector = {c.get("connector_id"): c for c in self.connectors()}
        result: List[Dict[str, Any]] = []
        for row in placements:
            connector_id = row.get("connector_id")
            connector = by_connector.get(connector_id, {})
            result.append(
                {
                    **row,
                    "connector_options": dict(connector.get("options", {})),
                }
            )
        return result

    def connector(self, connector_id: str) -> Optional[Dict[str, Any]]:
        for row in self.connectors():
            if row.get("connector_id") == connector_id:
                return row
        return None
