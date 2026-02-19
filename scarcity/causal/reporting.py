"""
Reporting Layer.

Defines the standardized output formats for the production causal pipeline.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from scarcity.causal.specs import EstimandSpec


@dataclass
class EffectArtifact:
    """Per-spec causal effect artifact."""
    spec: EstimandSpec
    spec_id: str
    index: int
    estimand_type: str
    estimate: Any
    confidence_intervals: Optional[Any]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    refuter_results: Dict[str, Any] = field(default_factory=dict)
    temporal_diagnostics: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    graph_edge_payload: List[Dict[str, Any]] = field(default_factory=list)
    backend: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "index": self.index,
            "estimand_type": self.estimand_type,
            "estimate": self.estimate,
            "confidence_intervals": self.confidence_intervals,
            "diagnostics": self.diagnostics,
            "refuter_results": self.refuter_results,
            "temporal_diagnostics": self.temporal_diagnostics,
            "provenance": self.provenance,
            "graph_edge_payload": self.graph_edge_payload,
            "backend": self.backend,
            "created_at": self.created_at,
            "spec": {
                "treatment": self.spec.treatment,
                "outcome": self.spec.outcome,
                "confounders": list(self.spec.confounders),
                "effect_modifiers": list(self.spec.effect_modifiers),
                "instrument": self.spec.instrument,
                "mediator": self.spec.mediator,
                "type": self.spec.type.value,
                "time_column": self.spec.time_column,
                "lag": self.spec.lag,
                "mediator_lag": self.spec.mediator_lag,
                "panel_keys": list(self.spec.panel_keys),
                "dot_path": self.spec.dot_path,
                "metadata": dict(self.spec.metadata),
            },
        }


@dataclass
class SpecError:
    """Failure record for a specific estimand spec."""
    spec_id: str
    index: int
    stage: str
    message: str
    exception_type: str
    traceback: Optional[str] = None
    fatal: bool = False
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "index": self.index,
            "stage": self.stage,
            "message": self.message,
            "exception_type": self.exception_type,
            "traceback": self.traceback,
            "fatal": self.fatal,
            "created_at": self.created_at,
        }


@dataclass
class RunSummary:
    """Run-level summary statistics."""
    run_id: str
    total_specs: int
    succeeded: int
    failed: int
    started_at: str
    finished_at: str
    duration_sec: float
    status: str
    fail_policy: str
    parallelism: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "total_specs": self.total_specs,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_sec": self.duration_sec,
            "status": self.status,
            "fail_policy": self.fail_policy,
            "parallelism": self.parallelism,
        }


@dataclass
class RuntimeMetadata:
    """Runtime metadata for traceability."""
    run_id: str
    runtime: Dict[str, Any]
    python_version: str
    platform: str
    dowhy_version: Optional[str]
    econml_version: Optional[str]
    data_signature: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "runtime": self.runtime,
            "python_version": self.python_version,
            "platform": self.platform,
            "dowhy_version": self.dowhy_version,
            "econml_version": self.econml_version,
            "data_signature": self.data_signature,
        }


@dataclass
class CausalRunResult:
    """Bundled result for a run of one or more estimands."""
    results: List[EffectArtifact] = field(default_factory=list)
    errors: List[SpecError] = field(default_factory=list)
    summary: Optional[RunSummary] = None
    metadata: Optional[RuntimeMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "errors": [e.to_dict() for e in self.errors],
            "summary": self.summary.to_dict() if self.summary else None,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }
