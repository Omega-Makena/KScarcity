"""
Configuration dataclasses for the Meta-Integrative Layer.

Replaces the Dict-based DEFAULT_CONFIG with type-safe dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class MetaScoreWeights:
    """Weights for positive contributions to meta-score."""
    accept: float = 0.35
    stability: float = 0.25
    contrast: float = 0.1


@dataclass
class MetaScorePenalties:
    """Penalties for negative contributions to meta-score."""
    latency: float = 0.15
    vram: float = 0.1
    oom: float = 0.2


@dataclass
class MetaScoreConfig:
    """Configuration for meta-score computation."""
    weights: MetaScoreWeights = field(default_factory=MetaScoreWeights)
    penalties: MetaScorePenalties = field(default_factory=MetaScorePenalties)
    ema_alpha: float = 0.3
    latency_normalization: float = 120.0  # Divisor for latency normalization
    latency_clip_max: float = 2.0  # Max clipped latency value
    reward_clip_min: float = -1.0
    reward_clip_max: float = 1.0


@dataclass
class ControllerPolicyConfig:
    """Configuration for controller policy adjustments."""
    tau_bounds: Tuple[float, float] = (0.5, 1.2)
    gamma_bounds: Tuple[float, float] = (0.1, 0.5)
    # Adjustment step sizes
    tau_increase_step: float = 0.1
    tau_decrease_step: float = 0.1
    gamma_increase_step: float = 0.05
    # Thresholds for triggering adjustments
    accept_low_threshold: float = 0.06
    stability_high_threshold: float = 0.8


@dataclass
class EvaluatorPolicyConfig:
    """Configuration for evaluator policy adjustments."""
    g_min_bounds: Tuple[float, float] = (0.006, 0.02)
    lambda_ci_bounds: Tuple[float, float] = (0.4, 0.6)
    # Adjustment step sizes
    g_min_decrease_step: float = 0.002
    lambda_ci_decrease_step: float = 0.05
    # Thresholds
    accept_low_windows_threshold: int = 5
    accept_very_low_threshold: float = 0.03


@dataclass
class DRGPolicyConfig:
    """Configuration for Dynamic Resource Governance policy."""
    sketch_dim_set: List[int] = field(default_factory=lambda: [512, 1024, 2048])
    n_paths_max: int = 128
    # Resource thresholds
    vram_high_threshold: float = 0.85
    vram_low_threshold: float = 0.55
    latency_comfort_threshold: float = 100.0
    latency_high_threshold: float = 120.0
    # Adjustment factors
    n_paths_decrease_factor: float = 0.15
    n_paths_increase_factor: float = 0.10
    resamples_increase: int = 2


@dataclass
class SafetyConfig:
    """Configuration for safety mechanisms."""
    rollback_delta: float = 0.1  # EMA drop threshold for rollback
    cooldown_cycles: int = 5  # Cycles before a knob can be changed again
    max_history: int = 10  # Maximum history entries to retain


@dataclass
class IntegrativeMetaConfig:
    """
    Complete configuration for the MetaIntegrativeLayer.
    
    This dataclass replaces the Dict-based DEFAULT_CONFIG for type safety
    and IDE support.
    """
    meta_score: MetaScoreConfig = field(default_factory=MetaScoreConfig)
    controller_policy: ControllerPolicyConfig = field(default_factory=ControllerPolicyConfig)
    evaluator_policy: EvaluatorPolicyConfig = field(default_factory=EvaluatorPolicyConfig)
    drg_policy: DRGPolicyConfig = field(default_factory=DRGPolicyConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    
    def to_dict(self) -> dict:
        """Convert to dictionary format for backward compatibility."""
        return {
            'meta_score': {
                'weights': {
                    'accept': self.meta_score.weights.accept,
                    'stability': self.meta_score.weights.stability,
                    'contrast': self.meta_score.weights.contrast,
                },
                'penalties': {
                    'latency': self.meta_score.penalties.latency,
                    'vram': self.meta_score.penalties.vram,
                    'oom': self.meta_score.penalties.oom,
                },
                'ema_alpha': self.meta_score.ema_alpha,
            },
            'controller_policy': {
                'tau_bounds': list(self.controller_policy.tau_bounds),
                'gamma_bounds': list(self.controller_policy.gamma_bounds),
            },
            'evaluator_policy': {
                'g_min_bounds': list(self.evaluator_policy.g_min_bounds),
                'lambda_ci_bounds': list(self.evaluator_policy.lambda_ci_bounds),
            },
            'drg_policy': {
                'sketch_dim_set': self.drg_policy.sketch_dim_set,
                'n_paths_max': self.drg_policy.n_paths_max,
            },
            'safety': {
                'rollback_delta': self.safety.rollback_delta,
                'cooldown_cycles': self.safety.cooldown_cycles,
            },
        }


# Default configuration instance
DEFAULT_INTEGRATIVE_CONFIG = IntegrativeMetaConfig()
