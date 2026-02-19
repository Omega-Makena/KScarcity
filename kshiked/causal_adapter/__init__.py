"""K-Shield adapter for Scarcity causal pipeline."""
from .artifacts import KShieldArtifactStore
from .config import (
    AdapterConfig,
    AdapterEdgeConfig,
    AdapterPolicyConfig,
    AdapterRuntimeConfig,
    AdapterSelectionConfig,
    AdapterSimulationConfig,
)
from .dataset import load_unified_dataset, segment_dataset
from .integration import artifact_to_edge, edge_to_simulation_update
from .policy import select_estimands
from .spec_builder import build_estimand_specs, build_task_specs
from .types import (
    AdapterRunResult,
    BatchContext,
    CausalTaskSpec,
    EstimandDecision,
    KnowledgeGraphEdge,
    SimulationParameterUpdate,
    TaskWindow,
)

__all__ = [
    "AdapterConfig",
    "AdapterEdgeConfig",
    "AdapterPolicyConfig",
    "AdapterRuntimeConfig",
    "AdapterSelectionConfig",
    "AdapterSimulationConfig",
    "AdapterRunResult",
    "BatchContext",
    "CausalTaskSpec",
    "EstimandDecision",
    "KnowledgeGraphEdge",
    "KShieldArtifactStore",
    "SimulationParameterUpdate",
    "TaskWindow",
    "artifact_to_edge",
    "build_estimand_specs",
    "build_task_specs",
    "edge_to_simulation_update",
    "load_unified_dataset",
    "segment_dataset",
    "select_estimands",
]
