"""Types for K-Shield causal adapter layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from scarcity.causal.specs import EstimandType


@dataclass(frozen=True)
class TaskWindow:
    start_year: int
    end_year: int


@dataclass
class CausalTaskSpec:
    """K-Shield domain task spec that will be converted to Scarcity EstimandSpecs."""
    treatment: str
    outcome: str
    confounders: List[str] = field(default_factory=list)
    effect_modifiers: List[str] = field(default_factory=list)
    instrument: Optional[str] = None
    mediator: Optional[str] = None
    time_column: Optional[str] = None
    lag: Optional[int] = None
    mediator_lag: Optional[int] = None
    panel_keys: List[str] = field(default_factory=list)
    dot_path: Optional[str] = None
    window: Optional[TaskWindow] = None
    county: Optional[str] = None
    sector: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def signature(self) -> str:
        parts = [
            f"treatment={self.treatment}",
            f"outcome={self.outcome}",
            f"confounders={sorted(self.confounders)}",
            f"effect_modifiers={sorted(self.effect_modifiers)}",
            f"instrument={self.instrument}",
            f"mediator={self.mediator}",
            f"time_column={self.time_column}",
            f"lag={self.lag}",
            f"mediator_lag={self.mediator_lag}",
            f"panel_keys={sorted(self.panel_keys)}",
            f"dot_path={self.dot_path}",
            f"window={self.window}",
            f"county={self.county}",
            f"sector={self.sector}",
        ]
        return "|".join(parts)


@dataclass
class EstimandDecision:
    estimands: List[EstimandType]
    reasons: Dict[EstimandType, str] = field(default_factory=dict)


@dataclass
class KnowledgeGraphEdge:
    edge_id: str
    source: str
    target: str
    sign: str
    weight: float
    confidence: float
    lag: Optional[int] = None
    window: Optional[TaskWindow] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationParameterUpdate:
    parameter: str
    delta: float
    reason: str
    edge_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchContext:
    window: Optional[TaskWindow] = None
    county: Optional[str] = None
    sector: Optional[str] = None


@dataclass
class AdapterRunResult:
    run_id: str
    results: List[Any]
    edges: List[KnowledgeGraphEdge]
    simulation_updates: List[SimulationParameterUpdate]
    errors: List[Any]
    metadata: Dict[str, Any]
