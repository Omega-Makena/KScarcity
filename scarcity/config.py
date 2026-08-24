"""
Central engine configuration.

Single home for the tunable constants used across the online discovery /
orchestration path. Change a value here instead of hunting for a hardcoded
literal in the middle of an algorithm.

Usage
-----
    from scarcity.config import ENGINE_CONFIG

    n = ENGINE_CONFIG.proposer.n_paths_default

To override at runtime (e.g. in a benchmark or test) mutate the singleton::

    from scarcity.config import ENGINE_CONFIG
    ENGINE_CONFIG.proposer.max_arms = 20000

or build a fresh one and pass it where a config is accepted::

    from scarcity.config import EngineConfig, ProposerConfig
    cfg = EngineConfig(proposer=ProposerConfig(n_paths_default=64))

Per-component configs that already live next to their component
(`BanditConfig` in bandit_router, `StructuralConfig` in relationships, the
resource profile) are intentionally left where they are; this module holds the
cross-cutting orchestration tunables only.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ProposerConfig:
    """Path-proposal and candidate-generation tunables (BanditRouter, Orchestrator)."""

    # Default number of candidate paths proposed per data window when the
    # resource profile does not specify one.
    n_paths_default: int = 200

    # How many hypothesis-derived candidates the orchestrator pulls from an
    # attached OnlineDiscoveryEngine each window.
    discovery_top_k: int = 30

    # Source lags tried for each directed variable pair (target is contemporaneous).
    candidate_lags: Tuple[int, ...] = (0, 1)

    # Per-variable transform ops for a generated pair candidate.
    candidate_ops: Tuple[str, str] = ("identity", "identity")

    # Hard cap on distinct bandit arms retained (bounds unbounded growth).
    max_arms: int = 5000

    # Variable count assumed only when a window carries no schema at all.
    fallback_n_vars: int = 8


@dataclass
class DiversityConfig:
    """Structural-novelty scoring for candidate paths."""

    # Number of recently accepted variable-sets remembered for novelty scoring.
    recent_memory: int = 64

    # Blend weight: frequency-novelty vs variable-set novelty, in [0, 1].
    # 1.0 = purely inverse-frequency, 0.0 = purely set-difference.
    novelty_weight: float = 0.5


@dataclass
class EngineConfig:
    """Top-level container for cross-cutting engine tunables."""

    proposer: ProposerConfig = field(default_factory=ProposerConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)


# Process-wide singleton. Import this; mutate it if you must override.
ENGINE_CONFIG = EngineConfig()


def get_config() -> EngineConfig:
    """Return the process-wide engine configuration singleton."""
    return ENGINE_CONFIG
