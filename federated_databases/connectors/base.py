"""Connector interfaces for federated data sources."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ConnectorSpec:
    """Registered connector metadata for a federation node."""

    connector_id: str
    node_id: str
    source_type: str
    location: str
    dataset_ids: List[str] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)
