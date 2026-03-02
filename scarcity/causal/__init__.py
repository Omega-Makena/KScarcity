"""
Scarcity Causal Inference Package.

Public API surface for the production causal inference pipeline.
"""

from scarcity.causal.engine import run_causal
from scarcity.causal.specs import (
    EstimandSpec,
    EstimandType,
    FailPolicy,
    ParallelismMode,
    RuntimeSpec,
    TimeSeriesPolicy,
)
from scarcity.causal.reporting import (
    CausalRunResult,
    EffectArtifact,
    RunSummary,
    RuntimeMetadata,
    SpecError,
)
from scarcity.causal.graph import (
    parse_dot_edges,
    build_dot,
    merge_edges,
    validate_temporal_edges,
)
from scarcity.causal.feature_layer import FeatureBuilder
from scarcity.causal.identification import Identifier
from scarcity.causal.validation import Validator

__all__ = [
    # Pipeline entry-point
    "run_causal",
    # Spec types
    "EstimandSpec",
    "EstimandType",
    "FailPolicy",
    "ParallelismMode",
    "RuntimeSpec",
    "TimeSeriesPolicy",
    # Output types
    "CausalRunResult",
    "EffectArtifact",
    "RunSummary",
    "RuntimeMetadata",
    "SpecError",
    # Graph utilities
    "parse_dot_edges",
    "build_dot",
    "merge_edges",
    "validate_temporal_edges",
    # Pipeline stages
    "FeatureBuilder",
    "Identifier",
    "Validator",
]
