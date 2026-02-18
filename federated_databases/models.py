"""Core models for Scarcity federated database orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FederatedNode:
    """Registered node metadata."""

    node_id: str
    backend: str
    db_path: str
    county_filter: Optional[str] = None
    created_at: str = field(default_factory=_utc_now_iso)


@dataclass
class LocalTrainingMetrics:
    """Local node training output for a sync round."""

    node_id: str
    sample_count: int
    loss: float
    gradient_norm: float
    mean_criticality: float
    feature_count: int


@dataclass
class SyncRoundResult:
    """Federation round summary."""

    round_number: int
    participants: int
    total_samples: int
    global_loss: float
    global_gradient_norm: float
    aggregation_method: str = "weighted_fedavg"
    started_at: str = field(default_factory=_utc_now_iso)
    completed_at: str = field(default_factory=_utc_now_iso)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExchangeAuditRecord:
    """Record for model/signal exchange events."""

    round_number: int
    from_node: str
    to_node: str
    payload_type: str
    payload_size: int
    created_at: str = field(default_factory=_utc_now_iso)
    details: Dict[str, Any] = field(default_factory=dict)
