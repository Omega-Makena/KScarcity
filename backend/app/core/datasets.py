"""Dataset registry backed by the local ingestion report."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


logger = logging.getLogger(__name__)


def _slugify(name: str) -> str:
    """Return a filesystem-friendly slug."""
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in name.lower())
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-_") or "dataset"


class DatasetRegistry:
    """Load dataset metadata derived from the local ingestion pipeline."""

    def __init__(self, report_path: Path) -> None:
        self.report_path = report_path
        self._datasets: Dict[str, Dict[str, Any]] = {}
        self._domains: Dict[str, Dict[str, Any]] = {}
        self._totals: Dict[str, Any] = {}
        self._last_refresh: Optional[datetime] = None
        self.refresh()

    # ------------------------------------------------------------------ #
    # Public API

    def refresh(self) -> None:
        """Reload the JSON report from disk."""
        if not self.report_path.exists():
            logger.warning("Dataset report %s not found; registry will be empty.", self.report_path)
            self._clear_state()
            return

        try:
            payload = json.loads(self.report_path.read_text())
        except Exception as exc:  # pragma: no cover - defensive only
            logger.error("Failed to parse dataset report %s: %s", self.report_path, exc)
            self._clear_state()
            return

        datasets = payload.get("datasets_loaded", [])
        totals = {
            "hypergraph_edges": int(payload.get("hypergraph_edges", 0) or 0),
            "hypergraph_hyperedges": int(payload.get("hypergraph_hyperedges", 0) or 0),
            "total_windows": int(payload.get("total_windows", 0) or 0),
        }

        parsed = {}
        domain_index: Dict[str, Dict[str, Any]] = {}

        for raw in datasets:
            dataset = self._normalise_record(raw)
            parsed[dataset["dataset_id"]] = dataset

            domain_key = dataset["domain"]
            domain_entry = domain_index.setdefault(
                domain_key,
                {
                    "domain": dataset["domain"],
                    "domain_id": dataset["domain_id"],
                    "datasets": 0,
                    "rows": 0,
                    "windows": 0,
                },
            )
            domain_entry["datasets"] += 1
            domain_entry["rows"] += dataset["rows"]
            domain_entry["windows"] += dataset["windows_emitted"]

        totals["dataset_count"] = len(parsed)
        totals["rows_total"] = sum(ds["rows"] for ds in parsed.values())
        totals["columns_total"] = sum(ds["columns"] for ds in parsed.values())

        self._datasets = parsed
        self._domains = domain_index
        self._totals = totals
        self._last_refresh = datetime.now(timezone.utc)
        logger.info("Dataset registry refreshed with %d datasets.", len(parsed))

    def summaries(self) -> List[Dict[str, Any]]:
        """Return lightweight summaries sorted by domain then filename."""
        return [
            {
                "dataset_id": data["dataset_id"],
                "filename": data["filename"],
                "path": data["path"],
                "domain": data["domain"],
                "domain_id": data["domain_id"],
                "rows": data["rows"],
                "columns": data["columns"],
                "windows_emitted": data["windows_emitted"],
                "schema_version": data["schema_version"],
                "last_ingested": data["last_ingested"],
            }
            for data in sorted(self._datasets.values(), key=lambda item: (item["domain"], item["filename"]))
        ]

    def detail(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Return full dataset record for API responses."""
        return self._datasets.get(dataset_id)

    def domain_breakdown(self) -> List[Dict[str, Any]]:
        """Return aggregate metrics for each domain."""
        return sorted(self._domains.values(), key=lambda entry: (-entry["rows"], entry["domain"]))

    def totals(self) -> Dict[str, Any]:
        """Return aggregate stats for UI badges."""
        payload = dict(self._totals)
        payload["refreshed_at"] = self._last_refresh.isoformat() if self._last_refresh else None
        return payload

    # ------------------------------------------------------------------ #
    # Helpers

    def _clear_state(self) -> None:
        self._datasets = {}
        self._domains = {}
        self._totals = {"dataset_count": 0, "rows_total": 0, "columns_total": 0, "total_windows": 0}
        self._last_refresh = None

    def _normalise_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        path = Path(record.get("path", "unknown"))
        filename = path.name or str(path)
        slug = _slugify(filename)
        digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:8]
        dataset_id = f"{slug}-{digest}"

        missing = record.get("missing", {}) or {}
        field_types = record.get("field_types", {}) or {}

        fields = [
            {
                "name": name,
                "dtype": field_types.get(name, "unknown"),
                "missing": int(missing.get(name, 0)),
            }
            for name in field_types.keys()
        ]

        timestamp = None
        try:
            stat = path.stat()
            timestamp = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
        except Exception:
            timestamp = None

        return {
            "dataset_id": dataset_id,
            "path": str(path),
            "filename": filename,
            "domain": record.get("domain", "unknown"),
            "domain_id": int(record.get("domain_id", 0) or 0),
            "rows": int(record.get("rows", 0) or 0),
            "columns": int(record.get("columns", 0) or 0),
            "windows_emitted": int(record.get("windows_emitted", 0) or 0),
            "schema_version": record.get("schema_version", "unknown"),
            "fields": fields,
            "last_ingested": timestamp,
            "missing_total": sum(int(value) for value in missing.values()),
        }


__all__ = ["DatasetRegistry"]
