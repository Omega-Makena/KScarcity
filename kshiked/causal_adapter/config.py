"""Configuration for K-Shield causal adapter."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from scarcity.causal.specs import FailPolicy, ParallelismMode, RuntimeSpec, TimeSeriesPolicy

from .types import TaskWindow


@dataclass
class AdapterRuntimeConfig:
    refute_random_common_cause: bool = True
    refute_placebo_treatment: bool = True
    refute_data_subset: bool = True
    refutation_simulations: int = 100
    parallelism: ParallelismMode = ParallelismMode.PROCESS
    n_jobs: int = 2
    chunk_size: int = 1
    fail_policy: FailPolicy = FailPolicy.CONTINUE
    time_series_policy: TimeSeriesPolicy = TimeSeriesPolicy.STRICT
    blas_thread_policy: Dict[str, int] = field(default_factory=lambda: {
        "OMP_NUM_THREADS": 1,
        "MKL_NUM_THREADS": 1,
        "OPENBLAS_NUM_THREADS": 1,
        "NUMEXPR_NUM_THREADS": 1,
        "VECLIB_MAXIMUM_THREADS": 1,
    })
    seed: int = 42
    random_seed: Optional[int] = None
    confidence_level: float = 0.95
    estimator_method: Optional[str] = None
    estimator_params: Dict[str, Any] = field(default_factory=dict)
    artifact_root: str = "artifacts"
    dot_path: Optional[str] = None
    export_graphs: bool = True

    def to_runtime_spec(self) -> RuntimeSpec:
        return RuntimeSpec(
            refute_random_common_cause=self.refute_random_common_cause,
            refute_placebo_treatment=self.refute_placebo_treatment,
            refute_data_subset=self.refute_data_subset,
            refutation_simulations=self.refutation_simulations,
            n_jobs=self.n_jobs,
            seed=self.seed,
            random_seed=self.random_seed,
            confidence_level=self.confidence_level,
            estimator_method=self.estimator_method,
            estimator_params=dict(self.estimator_params),
            parallelism=self.parallelism,
            fail_policy=self.fail_policy,
            chunk_size=self.chunk_size,
            time_series_policy=self.time_series_policy,
            blas_thread_policy=self.blas_thread_policy,
            artifact_root=self.artifact_root,
            dot_path=self.dot_path,
            export_graphs=self.export_graphs,
        )


@dataclass
class AdapterPolicyConfig:
    allow_late: bool = True
    allow_mediation: bool = True
    require_dot_for_iv: bool = True
    require_dot_for_mediation: bool = True


@dataclass
class AdapterSelectionConfig:
    treatments: List[str] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)
    confounders: List[str] = field(default_factory=list)
    effect_modifiers: List[str] = field(default_factory=list)
    instrument: Optional[str] = None
    mediator: Optional[str] = None
    lag: Optional[int] = None
    mediator_lag: Optional[int] = None
    time_column: Optional[str] = None
    panel_keys: List[str] = field(default_factory=list)
    windows: List[TaskWindow] = field(default_factory=list)
    counties: List[str] = field(default_factory=list)
    sectors: List[str] = field(default_factory=list)
    dot_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterEdgeConfig:
    weight_scale: float = 1.0
    min_confidence: float = 0.05
    max_weight: float = 1.0
    min_weight: float = 0.0


@dataclass
class AdapterSimulationConfig:
    mapping: Dict[str, str] = field(default_factory=dict)
    scale: float = 1.0


@dataclass
class AdapterConfig:
    runtime: AdapterRuntimeConfig = field(default_factory=AdapterRuntimeConfig)
    policy: AdapterPolicyConfig = field(default_factory=AdapterPolicyConfig)
    selection: AdapterSelectionConfig = field(default_factory=AdapterSelectionConfig)
    edge: AdapterEdgeConfig = field(default_factory=AdapterEdgeConfig)
    simulation: AdapterSimulationConfig = field(default_factory=AdapterSimulationConfig)
    kshield_artifact_root: str = "kshiked_artifacts"
