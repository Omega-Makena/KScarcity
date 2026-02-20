"""Query request and plan models for federated databases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class QueryRequest:
    dataset_id: str
    operation: str
    group_by: List[str] = field(default_factory=list)
    metric: str = "count"
    metric_field: str = "*"
    filters: Dict[str, Any] = field(default_factory=dict)
    joins: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class ExecutionStep:
    node_id: str
    connector_id: str
    dataset_id: str
    connector_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    query: QueryRequest
    steps: List[ExecutionStep]
    cross_agency: bool
    topology_version: str
