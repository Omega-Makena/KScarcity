"""
Configuration classes for relationship hypothesis types.

All magic numbers and thresholds are centralized here as configurable parameters.
Each hypothesis class accepts an optional config object with sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CausalConfig:
    """
    Configuration for CausalHypothesis (Granger Causality).
    
    Attributes:
        direction_threshold: Minimum gain difference to declare a direction.
            Higher values require stronger evidence for directionality.
        confidence_multiplier: Scales the gain into confidence score.
        min_samples_for_eval: Minimum buffer length before computing Granger test.
        ridge_alpha: Regularization for regression (prevents singularity).
        min_prediction_samples: Minimum samples before predictions are generated.
    """
    direction_threshold: float = 0.02
    confidence_multiplier: float = 2.0
    min_samples_for_eval: int = 10  # lag + this value
    ridge_alpha: float = 1e-3
    min_prediction_samples: int = 5  # lag + this value


@dataclass
class CorrelationalConfig:
    """
    Configuration for CorrelationalHypothesis (Pearson Correlation).
    
    Attributes:
        min_samples: Minimum observations before computing correlation.
        confidence_scale: Samples needed for full confidence (n/confidence_scale).
        stability_threshold: Correlation magnitude above which stability is high.
    """
    min_samples: int = 10
    confidence_scale: int = 50
    stability_threshold: float = 0.3


@dataclass
class TemporalConfig:
    """
    Configuration for TemporalHypothesis (AR/VAR model).
    
    Attributes:
        forgetting_factor: RLS forgetting factor (0.9-1.0). Lower = faster adaptation.
        initial_covariance: Initial P matrix diagonal scaling.
        min_samples_for_eval: Minimum buffer length for evaluation.
        autocorr_stability_threshold: Lag-1 autocorrelation for high stability.
    """
    forgetting_factor: float = 0.99
    initial_covariance: float = 100.0
    min_samples_for_eval: int = 5  # lag + this value
    autocorr_stability_threshold: float = 0.2


@dataclass
class FunctionalConfig:
    """
    Configuration for FunctionalHypothesis (Polynomial Regression).
    
    Attributes:
        forgetting_factor: RLS forgetting factor.
        initial_covariance: Initial P matrix scaling.
        min_samples: Minimum samples for evaluation.
        deterministic_threshold: Residual std / total std ratio for determinism.
        confidence_scale: Samples needed for full confidence.
    """
    forgetting_factor: float = 0.99
    initial_covariance: float = 100.0
    min_samples: int = 10
    deterministic_threshold: float = 0.1
    confidence_scale: int = 30


@dataclass
class EquilibriumConfig:
    """
    Configuration for EquilibriumHypothesis (Kalman Filter).
    
    Attributes:
        process_noise: Kalman Q parameter (state evolution noise).
        observation_noise: Kalman R parameter (measurement noise).
        reversion_threshold: Minimum reversion rate to declare mean-reverting.
        min_samples_for_eval: Minimum buffer length for evaluation.
        min_samples_for_prediction: Minimum samples before predictions.
        confidence_scale: Samples needed for full confidence.
    """
    process_noise: float = 0.01
    observation_noise: float = 0.1
    reversion_threshold: float = 0.05
    min_samples_for_eval: int = 20
    min_samples_for_prediction: int = 20
    confidence_scale: int = 50


@dataclass
class CompositionalConfig:
    """
    Configuration for CompositionalHypothesis (Sum Constraints).
    
    Attributes:
        min_samples: Minimum observations for evaluation.
        error_threshold: Max mean error for constraint to hold.
        error_scaling: Multiplier for fit score from error.
    """
    min_samples: int = 5
    error_threshold: float = 0.05
    error_scaling: float = 10.0


@dataclass
class CompetitiveConfig:
    """
    Configuration for CompetitiveHypothesis (Trade-off/Zero-sum).
    
    Attributes:
        min_samples: Minimum observations for evaluation.
        cv_threshold: Coefficient of variation threshold for constant sum.
        correlation_threshold: Negative correlation threshold for competitive.
    """
    min_samples: int = 10
    cv_threshold: float = 0.1
    correlation_threshold: float = -0.5


@dataclass
class SynergisticConfig:
    """
    Configuration for SynergisticHypothesis (Interaction Effects).
    
    Attributes:
        min_samples: Minimum observations for evaluation.
        interaction_threshold: Min |coefficient| to declare interaction.
    """
    min_samples: int = 20
    interaction_threshold: float = 0.1


@dataclass
class ProbabilisticConfig:
    """
    Configuration for ProbabilisticHypothesis (Distribution Shift).
    
    Attributes:
        min_samples_per_group: Minimum in each condition group.
        split_threshold: Condition threshold for splitting groups.
        effect_size_threshold: Cohen's d threshold for significance.
    """
    min_samples_per_group: int = 10
    split_threshold: float = 0.5
    effect_size_threshold: float = 0.5


@dataclass
class StructuralConfig:
    """
    Configuration for StructuralHypothesis (ICC/Hierarchical).
    
    Attributes:
        min_groups: Minimum number of groups for evaluation.
        confidence_scale: Samples needed for full confidence.
    """
    min_groups: int = 2
    confidence_scale: int = 50


# =============================================================================
# Extended Relationship Configs
# =============================================================================

@dataclass
class MediatingConfig:
    """
    Configuration for MediatingHypothesis (Baron-Kenny).
    
    Attributes:
        min_samples: Minimum observations for evaluation.
        path_significance_threshold: Minimum R² for path significance.
        mediation_reduction_threshold: Min c-c' for partial mediation.
    """
    min_samples: int = 30
    path_significance_threshold: float = 0.1
    mediation_reduction_threshold: float = 0.1


@dataclass
class ModeratingConfig:
    """
    Configuration for ModeratingHypothesis (Interaction).
    
    Attributes:
        min_samples: Minimum observations for evaluation.
        interaction_threshold: Min |interaction coefficient| for moderation.
    """
    min_samples: int = 30
    interaction_threshold: float = 0.1


@dataclass
class GraphConfig:
    """
    Configuration for GraphHypothesis (Network Structure).
    
    Attributes:
        min_edges: Minimum edges for evaluation.
    """
    min_edges: int = 5


@dataclass
class SimilarityConfig:
    """
    Configuration for SimilarityHypothesis (Clustering).
    
    Attributes:
        min_samples: Minimum samples for clustering.
        min_explained_variance: Threshold for cluster structure.
    """
    min_samples: int = 50
    min_explained_variance: float = 0.2


@dataclass
class LogicalConfig:
    """
    Configuration for LogicalHypothesis (Boolean Rules).
    
    Attributes:
        min_samples: Minimum observations for rule detection.
        accuracy_threshold: Min accuracy to confirm rule.
    """
    min_samples: int = 20
    accuracy_threshold: float = 0.9


# =============================================================================
# Master Config (combines all)
# =============================================================================

@dataclass
class HypothesisConfig:
    """
    Master configuration containing all hypothesis-specific configs.
    
    Use this for centralized configuration management.
    """
    causal: CausalConfig = field(default_factory=CausalConfig)
    correlational: CorrelationalConfig = field(default_factory=CorrelationalConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    functional: FunctionalConfig = field(default_factory=FunctionalConfig)
    equilibrium: EquilibriumConfig = field(default_factory=EquilibriumConfig)
    compositional: CompositionalConfig = field(default_factory=CompositionalConfig)
    competitive: CompetitiveConfig = field(default_factory=CompetitiveConfig)
    synergistic: SynergisticConfig = field(default_factory=SynergisticConfig)
    probabilistic: ProbabilisticConfig = field(default_factory=ProbabilisticConfig)
    structural: StructuralConfig = field(default_factory=StructuralConfig)
    mediating: MediatingConfig = field(default_factory=MediatingConfig)
    moderating: ModeratingConfig = field(default_factory=ModeratingConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    logical: LogicalConfig = field(default_factory=LogicalConfig)
