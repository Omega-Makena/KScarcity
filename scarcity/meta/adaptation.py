"""
Adaptation Engine — Phase 1c of meta-learning.

Wires ContextEncoder → EpisodicMemory → parameter adaptation.

Memory hit path:
  encode(context) → retrieve top-k episodes → similarity-weighted blend of
  stored deltas → base_params + blended_delta

REPTILE fallback path (cold start / memory miss):
  Return the optimizer's current prior, or base_params if the prior is empty.

Callers should call record() after observing actual performance outcomes so
that future queries can benefit from this episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .encoder import ContextEncoder, ContextEncoderConfig
from .memory import EpisodicMemory, EpisodicMemoryConfig
from .optimizer import OnlineReptileOptimizer, MetaOptimizerConfig


@dataclass
class AdaptationConfig:
    min_similarity: float = 0.3   # cosine threshold; miss if best < this
    top_k: int = 5                # max episodes to blend
    blend_mode: str = "weighted"  # "weighted" (sim-normalised) | "top1"


@dataclass
class AdaptationResult:
    adapted_params: Dict[str, float]
    source: str                   # "memory" | "reptile" | "passthrough"
    similarities: List[float]     # per-retrieved episode; empty on fallback
    n_retrieved: int
    query_key: np.ndarray         # embedding used for retrieval


class AdaptationEngine:
    """
    Combines episodic memory retrieval with REPTILE fallback for fast adaptation.

    Parameters not present in retrieved deltas are left at their base_params
    value. Parameters not in base_params but present in deltas are added.
    """

    def __init__(
        self,
        encoder: Optional[ContextEncoder] = None,
        memory: Optional[EpisodicMemory] = None,
        optimizer: Optional[OnlineReptileOptimizer] = None,
        config: Optional[AdaptationConfig] = None,
    ):
        self.encoder = encoder or ContextEncoder()
        self.memory = memory or EpisodicMemory()
        self.optimizer = optimizer or OnlineReptileOptimizer()
        self.config = config or AdaptationConfig()

        # Running counters for external monitoring
        self.hits: int = 0
        self.misses: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def adapt(
        self,
        context: Dict[str, Any],
        base_params: Dict[str, float],
        drg_profile: Optional[Dict[str, float]] = None,
    ) -> AdaptationResult:
        """
        Produce adapted parameters for the given context.

        Args:
            context:     Raw context dict (same schema fed to ContextEncoder).
            base_params: Current parameter snapshot to adapt from.
            drg_profile: Resource pressure profile forwarded to REPTILE if
                         the fallback path is taken (optional).

        Returns:
            AdaptationResult with adapted parameters and provenance metadata.
        """
        cfg = self.config
        query_key = self.encoder.encode(context)

        results = self.memory.retrieve(
            query_key,
            top_k=cfg.top_k,
            min_similarity=cfg.min_similarity,
        )

        if results:
            self.hits += 1
            adapted = self._blend(base_params, results, cfg.blend_mode)
            return AdaptationResult(
                adapted_params=adapted,
                source="memory",
                similarities=[r.similarity for r in results],
                n_retrieved=len(results),
                query_key=query_key,
            )

        # --- fallback ---
        self.misses += 1
        prior = self.optimizer.state.prior
        if prior:
            # Merge prior into base_params: prior wins for keys it knows about
            adapted = dict(base_params)
            adapted.update(prior)
            return AdaptationResult(
                adapted_params=adapted,
                source="reptile",
                similarities=[],
                n_retrieved=0,
                query_key=query_key,
            )

        return AdaptationResult(
            adapted_params=dict(base_params),
            source="passthrough",
            similarities=[],
            n_retrieved=0,
            query_key=query_key,
        )

    def record(
        self,
        context: Dict[str, Any],
        base_params: Dict[str, float],
        adapted_params: Dict[str, float],
        observed_delta: Dict[str, float],
        policy: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Store an episode after observing its outcome.

        Args:
            context:        The context used during adapt().
            base_params:    Parameters before adaptation.
            adapted_params: Parameters after adaptation (the value to store).
            observed_delta: Measured performance deltas (e.g. {"gain": 0.12}).
            policy:         Freeform metadata (domain_id, step, source, ...).
        """
        key = self.encoder.encode(context)
        self.memory.store(
            key=key,
            value=adapted_params,
            context=context,
            delta=observed_delta,
            policy=policy or {},
        )

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _blend(
        base_params: Dict[str, float],
        results: list,
        mode: str,
    ) -> Dict[str, float]:
        """Blend stored deltas into base_params."""
        if mode == "top1":
            delta = results[0].entry.delta
            adapted = dict(base_params)
            for k, v in delta.items():
                adapted[k] = adapted.get(k, 0.0) + v
            return adapted

        # weighted: normalise by sum of similarities
        total_sim = sum(r.similarity for r in results)
        if total_sim < 1e-8:
            return dict(base_params)

        adapted = dict(base_params)
        for r in results:
            w = r.similarity / total_sim
            for k, v in r.entry.delta.items():
                adapted[k] = adapted.get(k, 0.0) + w * v
        return adapted
