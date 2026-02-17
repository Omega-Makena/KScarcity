"""
Causal Specification Module.

Defines the data structures and runtime configuration for Scarcity's
production causal inference pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EstimandType(str, Enum):
    """
    Taxonomy of Causal Estimands.

    Reflects the distinct statistical quantities we can estimate.
    """
    ATE = "ATE"  # Average Treatment Effect
    ATT = "ATT"  # Average Treatment Effect on the Treated
    ATC = "ATC"  # Average Treatment Effect on the Control
    CATE = "CATE"  # Conditional Average Treatment Effect (Heterogeneous)
    ITE = "ITE"    # Individual Treatment Effect (Synonym for CATE in this context)
    LATE = "LATE"  # Local Average Treatment Effect (IV)
    MEDIATION_NDE = "MEDIATION_NDE"  # Natural Direct Effect
    MEDIATION_NIE = "MEDIATION_NIE"  # Natural Indirect Effect


class TimeSeriesPolicy(str, Enum):
    STRICT = "strict"
    WARN = "warn"
    NONE = "none"


class ParallelismMode(str, Enum):
    PROCESS = "process"
    THREAD = "thread"
    NONE = "none"


class FailPolicy(str, Enum):
    FAIL_FAST = "fail_fast"
    CONTINUE = "continue"


@dataclass
class EstimandSpec:
    """
    Defines the 'What' of the Causal Inference task.

    This spec encapsulates the causal graph structure and the specific
    quantity of interest. It is data-agnostic in definition but relates
    to column names in the provided dataset.
    """
    treatment: str
    outcome: str

    # Common confounding structure
    confounders: List[str] = field(default_factory=list)

    # For Heterogeneity (CATE/ITE)
    effect_modifiers: List[str] = field(default_factory=list)

    # For Instrumental Variable (LATE)
    instrument: Optional[str] = None

    # For Mediation
    mediator: Optional[str] = None

    # Target quantity
    type: EstimandType = EstimandType.ATE

    # Time-series and panel support
    time_column: Optional[str] = None
    lag: Optional[int] = None
    mediator_lag: Optional[int] = None
    panel_keys: List[str] = field(default_factory=list)

    # Optional DOT graph for identification
    dot_path: Optional[str] = None

    # Arbitrary metadata for provenance
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validates the internal consistency of the specification.
        Raises ValueError if the spec is invalid.
        """
        if not self.treatment:
            raise ValueError("Treatment variable is required.")
        if not self.outcome:
            raise ValueError("Outcome variable is required.")

        if self.type == EstimandType.LATE and not self.instrument:
            raise ValueError("Instrument is required for LATE estimation.")

        if self.type in (EstimandType.MEDIATION_NDE, EstimandType.MEDIATION_NIE) and not self.mediator:
            raise ValueError("Mediator is required for Mediation analysis.")

    @property
    def target_units(self) -> str:
        """Translates type to DoWhy target_units string if applicable."""
        if self.type == EstimandType.ATT:
            return "att"
        if self.type == EstimandType.ATC:
            return "atc"
        return "ate"

    def signature(self) -> str:
        """Stable identifier for this spec (for artifacts and errors)."""
        parts = [
            f"type={self.type.value}",
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
        ]
        return "|".join(parts)


@dataclass
class RuntimeSpec:
    """
    Defines the 'How' of the execution.

    Configuration for the estimation process, refutations, and computational constraints.
    """
    # Refutation Configuration
    refute_random_common_cause: bool = True
    refute_placebo_treatment: bool = True
    refute_data_subset: bool = True
    refutation_simulations: int = 100

    # Computational
    n_jobs: int = 1
    seed: int = 42
    random_seed: Optional[int] = None

    # Estimation Quality
    confidence_level: float = 0.95

    # Estimator Preference (Optional override)
    estimator_method: Optional[str] = None
    estimator_params: Dict[str, Any] = field(default_factory=dict)

    # Execution policies
    parallelism: ParallelismMode = ParallelismMode.PROCESS
    fail_policy: FailPolicy = FailPolicy.CONTINUE
    chunk_size: int = 1

    # Time-series policy
    time_series_policy: TimeSeriesPolicy = TimeSeriesPolicy.NONE

    # BLAS thread policy
    blas_thread_policy: Dict[str, int] = field(default_factory=lambda: {
        "OMP_NUM_THREADS": 1,
        "MKL_NUM_THREADS": 1,
        "OPENBLAS_NUM_THREADS": 1,
        "NUMEXPR_NUM_THREADS": 1,
        "VECLIB_MAXIMUM_THREADS": 1,
    })

    # DOT graph support
    dot_path: Optional[str] = None

    # Artifact output
    artifact_root: str = "artifacts"
    run_id: Optional[str] = None
    export_graphs: bool = True

    def resolved_seed(self) -> int:
        """Return the effective seed (prefers random_seed when set)."""
        return int(self.random_seed if self.random_seed is not None else self.seed)

    def normalize(self) -> None:
        """Normalize enum-like fields when provided as strings."""
        if isinstance(self.parallelism, str):
            self.parallelism = ParallelismMode(self.parallelism)
        if isinstance(self.fail_policy, str):
            self.fail_policy = FailPolicy(self.fail_policy)
        if isinstance(self.time_series_policy, str):
            self.time_series_policy = TimeSeriesPolicy(self.time_series_policy)

    def normalized_blas_policy(self) -> Dict[str, int]:
        """Return a normalized BLAS thread policy mapping."""
        if self.blas_thread_policy is None:
            return {
                "OMP_NUM_THREADS": 1,
                "MKL_NUM_THREADS": 1,
                "OPENBLAS_NUM_THREADS": 1,
                "NUMEXPR_NUM_THREADS": 1,
                "VECLIB_MAXIMUM_THREADS": 1,
            }
        if isinstance(self.blas_thread_policy, dict):
            return {str(k): int(v) for k, v in self.blas_thread_policy.items()}
        return {
            "OMP_NUM_THREADS": int(self.blas_thread_policy),
            "MKL_NUM_THREADS": int(self.blas_thread_policy),
            "OPENBLAS_NUM_THREADS": int(self.blas_thread_policy),
            "NUMEXPR_NUM_THREADS": int(self.blas_thread_policy),
            "VECLIB_MAXIMUM_THREADS": int(self.blas_thread_policy),
        }

    def as_dict(self) -> Dict[str, Any]:
        """Serialize runtime configuration for metadata."""
        self.normalize()
        return {
            "refute_random_common_cause": self.refute_random_common_cause,
            "refute_placebo_treatment": self.refute_placebo_treatment,
            "refute_data_subset": self.refute_data_subset,
            "refutation_simulations": self.refutation_simulations,
            "n_jobs": self.n_jobs,
            "seed": self.resolved_seed(),
            "confidence_level": self.confidence_level,
            "estimator_method": self.estimator_method,
            "estimator_params": dict(self.estimator_params),
            "parallelism": self.parallelism.value,
            "fail_policy": self.fail_policy.value,
            "chunk_size": self.chunk_size,
            "time_series_policy": self.time_series_policy.value,
            "blas_thread_policy": self.normalized_blas_policy(),
            "dot_path": self.dot_path,
            "artifact_root": self.artifact_root,
            "run_id": self.run_id,
            "export_graphs": self.export_graphs,
        }
