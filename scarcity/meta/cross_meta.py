"""
Cross-domain aggregation for meta learning.

Two aggregators are provided:

CrossDomainMetaAggregator  (Phase 1 / fallback)
    Pure statistical aggregation of domain deltas — trimmed mean or median.
    No memory. Used as fallback when global memory is unavailable or empty.

CrossDomainMetaLearner  (Phase 5b / true meta-learner)
    Memory-backed aggregation. Queries GlobalMetaMemory for a cross-domain prior
    and blends it with the statistical baseline according to memory quality
    (= memory saturation ratio).  Falls back to CrossDomainMetaAggregator when
    memory is empty.

    Memory quality grows from 0 → 1 as the episode buffer fills. When quality=0
    the result is identical to the fallback. When quality=1 the result is a blend
    biased toward the historical optimal parameter configuration stored in memory.

    This implements "learning to learn": the aggregator gets better at picking a
    global prior as it accumulates cross-domain episodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .domain_meta import DomainMetaUpdate


@dataclass
class CrossMetaConfig:
    """Configuration for cross-domain aggregation."""
    method: str = "trimmed_mean"
    trim_alpha: float = 0.1
    min_confidence: float = 0.05


class CrossDomainMetaAggregator:
    """
    Combines domain-level meta updates into a global update vector.
    
    This class takes a list of domain-specific updates, filters them based on
    confidence, aligns them to a common set of parameter keys, and computes
    an aggregated update vector using the specified method (e.g., trimmed mean).
    """

    def __init__(self, config: Optional[CrossMetaConfig] = None):
        """
        Initialize the aggregator.
        
        Args:
            config: Configuration object. Defaults to default settings.
        """
        self.config = config or CrossMetaConfig()

    def aggregate(self, updates: Sequence[DomainMetaUpdate]) -> Tuple[np.ndarray, List[str], Dict[str, float]]:
        """
        Aggregate a sequence of domain updates.

        Args:
            updates: A list of DomainMetaUpdate objects.

        Returns:
            A tuple containing:
            - The aggregated update vector (np.ndarray).
            - The list of parameter keys corresponding to the vector (List[str]).
            - A dictionary of metadata about the aggregation process (Dict[str, float]).
        """
        cfg = self.config
        filtered = [u for u in updates if u.confidence >= cfg.min_confidence and len(u.vector) > 0]
        if not filtered:
            return np.zeros(0, dtype=np.float32), [], {"participants": 0}

        keys = self._union_keys(filtered)
        matrix = self._stack_vectors(filtered, keys)

        if cfg.method == "median":
            aggregate = np.median(matrix, axis=0)
        else:
            aggregate = self._trimmed_mean(matrix, cfg.trim_alpha)

        meta = {
            "participants": len(filtered),
            "method": cfg.method,
            "trim_alpha": cfg.trim_alpha if cfg.method != "median" else 0.0,
            "confidence_mean": float(np.mean([u.confidence for u in filtered])),
        }
        return aggregate.astype(np.float32), keys, meta

    def _union_keys(self, updates: Sequence[DomainMetaUpdate]) -> List[str]:
        """Collect all unique keys from the updates."""
        key_set = set()
        for update in updates:
            key_set.update(update.keys)
        return sorted(key_set)

    def _stack_vectors(self, updates: Sequence[DomainMetaUpdate], keys: List[str]) -> np.ndarray:
        """
        Stack update vectors into a matrix, aligning them to the union of keys.
        Missing values are filled with zeros.
        """
        key_index = {key: idx for idx, key in enumerate(keys)}
        matrix = np.zeros((len(updates), len(keys)), dtype=np.float32)
        for row, update in enumerate(updates):
            for key, value in zip(update.keys, update.vector):
                matrix[row, key_index[key]] = value
        return matrix

    def _trimmed_mean(self, matrix: np.ndarray, alpha: float) -> np.ndarray:
        """Compute the trimmed mean of the matrix along axis 0."""
        if matrix.shape[0] == 1:
            return matrix[0]
        k = int(np.floor(alpha * matrix.shape[0]))
        if k == 0:
            return matrix.mean(axis=0)
        sorted_vals = np.sort(matrix, axis=0)
        trimmed = sorted_vals[k : matrix.shape[0] - k]
        if trimmed.size == 0:
            trimmed = sorted_vals
        return trimmed.mean(axis=0)


# ---------------------------------------------------------------------------
# Phase 5b — Memory-backed cross-domain meta-learner
# ---------------------------------------------------------------------------

@dataclass
class CrossDomainMetaLearnerConfig:
    """Configuration for the memory-backed cross-domain meta-learner."""
    # Inherit fallback aggregator settings
    fallback_method: str = "trimmed_mean"
    fallback_trim_alpha: float = 0.1
    fallback_min_confidence: float = 0.05

    # Memory blending
    memory_reference_capacity: int = 256   # denominator for quality ratio
    min_memory_quality: float = 0.0        # floor (0 = pure fallback when empty)
    max_memory_quality: float = 0.8        # ceiling — always retain some signal


class CrossDomainMetaLearner:
    """
    Memory-backed cross-domain meta-learner (Phase 5b).

    Wraps CrossDomainMetaAggregator as a fallback and blends its output with a
    prior retrieved from GlobalMetaMemory, weighted by memory quality.

    quality = min(memory_size / memory_reference_capacity, max_memory_quality)

    result_vec = (1 - quality) * fallback_vec
               + quality      * prior_vec_aligned_to_keys

    Args:
        config:             Learner configuration.
        global_meta_memory: A GlobalMetaMemory instance (typed as Any to avoid
                            circular imports between meta ↔ federation packages).
                            May be None — degrades gracefully to fallback.
    """

    def __init__(
        self,
        config: Optional[CrossDomainMetaLearnerConfig] = None,
        global_meta_memory: Optional[Any] = None,
    ):
        self.config = config or CrossDomainMetaLearnerConfig()
        self._memory = global_meta_memory

        fallback_cfg = CrossMetaConfig(
            method=self.config.fallback_method,
            trim_alpha=self.config.fallback_trim_alpha,
            min_confidence=self.config.fallback_min_confidence,
        )
        self._fallback = CrossDomainMetaAggregator(fallback_cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def aggregate(
        self,
        updates: Sequence[DomainMetaUpdate],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Aggregate domain updates with memory-backed blending.

        Args:
            updates: DomainMetaUpdate list — one per active domain server.
            context: Optional extra context forwarded to the memory query.
                     If None, an aggregate context is derived from the updates.

        Returns:
            (result_vec, keys, meta) — same shape as CrossDomainMetaAggregator.
            meta additionally contains:
                source:         "memory_backed" | "fallback"
                memory_quality: float blend weight used
                prior_keys_matched: int keys shared between prior and current keys
        """
        # Always compute fallback — it's the baseline and a safety net
        fallback_vec, keys, meta = self._fallback.aggregate(updates)

        quality = self._memory_quality()
        prior = self._query_prior(updates, context) if quality > 0 else None

        if prior is None or not keys:
            meta["source"] = "fallback"
            meta["memory_quality"] = 0.0
            meta["prior_keys_matched"] = 0
            return fallback_vec, keys, meta

        # Align prior to current keys (zero-fill missing)
        prior_vec = np.array([prior.get(k, 0.0) for k in keys], dtype=np.float32)
        matched = sum(1 for k in keys if k in prior)

        result_vec = ((1.0 - quality) * fallback_vec + quality * prior_vec).astype(np.float32)

        meta["source"] = "memory_backed"
        meta["memory_quality"] = float(quality)
        meta["prior_keys_matched"] = matched
        return result_vec, keys, meta

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _memory_quality(self) -> float:
        """0 when memory is empty/unavailable, grows toward max_memory_quality as it fills."""
        if self._memory is None:
            return 0.0
        cfg = self.config
        # Works for both EpisodicMemory (has __len__) and GlobalMetaMemory (has .memory_size)
        size = self._memory.memory_size if hasattr(self._memory, "memory_size") else len(self._memory)
        cap = max(cfg.memory_reference_capacity, 1)
        raw = size / cap
        return float(np.clip(raw, cfg.min_memory_quality, cfg.max_memory_quality))

    def _query_prior(
        self,
        updates: Sequence[DomainMetaUpdate],
        extra_context: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, float]]:
        """Build aggregate context and query GlobalMetaMemory for the best prior."""
        if self._memory is None:
            return None

        filtered = [u for u in updates if len(u.vector) > 0]
        ctx: Dict[str, Any] = {
            "n_domains": float(len(filtered)),
        }
        if filtered:
            ctx["confidence_mean"] = float(np.mean([u.confidence for u in filtered]))
            ctx["score_delta_mean"] = float(np.mean([u.score_delta for u in filtered]))

        if extra_context:
            ctx.update(extra_context)

        return self._memory.suggest_prior("cross_domain", ctx)
