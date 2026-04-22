"""
DomainServerMeta — bridges DomainServer episodic state into DomainMetaUpdates.

Phase 5a of the meta-learning build plan.

The existing DomainMetaLearner (adaptive fallback) observes raw metrics dicts.
DomainServerMeta observes a DomainServer directly — extracting the richer signal
available from its episodic memory, hit_rate, and REPTILE prior — and produces a
DomainMetaUpdate compatible with both:
  - CrossDomainMetaAggregator  (fallback, trimmed-mean)
  - CrossDomainMetaLearner     (true meta-learner, memory-backed)

Confidence derivation:
    confidence = hit_rate_weight * hit_rate
               + memory_weight * log1p(memory_size) / log1p(memory_reference)
    optionally boosted by positive performance.gain

Delta vector:
    meta_lr   = meta_lr_min + (meta_lr_max - meta_lr_min) * confidence
    delta_vec = meta_lr * (current_base_params - previous_base_params)

Uses duck-typing for DomainServer/DomainServerRegistry to avoid circular imports
with the federation package.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .domain_meta import DomainMetaUpdate


@dataclass
class DomainServerMetaConfig:
    hit_rate_weight: float = 0.6        # share of confidence from hit_rate
    memory_weight: float = 0.4          # share of confidence from memory saturation
    memory_reference: int = 64          # reference memory size for log normalization
    min_confidence: float = 0.05        # floor — always emit a non-zero update
    meta_lr_min: float = 0.05
    meta_lr_max: float = 0.2
    performance_gain_boost: float = 0.05  # confidence boost per unit of positive gain


@dataclass
class _DomainState:
    """Per-basket running state for delta computation."""
    prev_base_params: Dict[str, float] = field(default_factory=dict)
    last_hit_rate: float = 0.0
    last_round_id: int = -1


class DomainServerMeta:
    """
    Observes DomainServers and produces DomainMetaUpdates for the meta-learner.

    One instance lives alongside HierarchicalFederation and is called after each
    Layer2 aggregation (or on-demand).

    Usage:
        observer = DomainServerMeta()
        updates  = observer.observe_registry(fed.domain_registry)
        vec, keys, meta = learner.aggregate(updates)
    """

    def __init__(self, config: Optional[DomainServerMetaConfig] = None):
        self.config = config or DomainServerMetaConfig()
        self._states: Dict[str, _DomainState] = {}   # basket_id → state

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(
        self,
        server: Any,                              # DomainServer (duck-typed)
        performance: Optional[Dict[str, float]] = None,
    ) -> DomainMetaUpdate:
        """
        Compute a DomainMetaUpdate from the server's current state.

        Args:
            server:      A DomainServer instance (accesses .basket_id, .domain_id,
                         .base_params, .hit_rate, .memory_size, .round_id).
            performance: Optional caller-supplied metrics (gain, stability, ...).
                         Positive 'gain' boosts the computed confidence.

        Returns:
            DomainMetaUpdate compatible with CrossDomainMetaAggregator /
            CrossDomainMetaLearner.
        """
        cfg = self.config
        perf = performance or {}
        basket_id = server.basket_id

        state = self._states.get(basket_id, _DomainState())

        # --- Confidence ---
        mem_norm = math.log1p(server.memory_size) / math.log1p(max(cfg.memory_reference, 1))
        mem_norm = min(mem_norm, 1.0)
        confidence = cfg.hit_rate_weight * server.hit_rate + cfg.memory_weight * mem_norm

        gain = float(perf.get("gain", 0.0))
        if gain > 0.0:
            confidence = min(confidence + cfg.performance_gain_boost * gain, 1.0)

        confidence = max(confidence, cfg.min_confidence)

        # --- Delta vector ---
        meta_lr = cfg.meta_lr_min + (cfg.meta_lr_max - cfg.meta_lr_min) * confidence
        curr = server.base_params          # Dict[str, float]
        prev = state.prev_base_params
        keys = sorted(curr.keys())

        if keys:
            curr_vec = np.array([curr[k] for k in keys], dtype=np.float32)
            prev_vec = np.array([prev.get(k, 0.0) for k in keys], dtype=np.float32)
            delta_vec = (meta_lr * (curr_vec - prev_vec)).astype(np.float32)
        else:
            delta_vec = np.zeros(0, dtype=np.float32)

        score_delta = server.hit_rate - state.last_hit_rate

        # --- Update state ---
        state.prev_base_params = dict(curr)
        state.last_hit_rate = server.hit_rate
        state.last_round_id = server.round_id
        self._states[basket_id] = state

        return DomainMetaUpdate(
            domain_id=server.domain_id,
            vector=delta_vec,
            keys=keys,
            confidence=confidence,
            timestamp=time.time(),
            score_delta=score_delta,
        )

    def observe_registry(
        self,
        registry: Any,                            # DomainServerRegistry (duck-typed)
        performance_map: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[DomainMetaUpdate]:
        """
        Observe every server in the registry and return a list of updates.

        Args:
            registry:        DomainServerRegistry — iterated via .all_servers().
            performance_map: basket_id → performance dict (optional).

        Returns:
            One DomainMetaUpdate per server (empty list if registry is empty).
        """
        perf_map = performance_map or {}
        updates: List[DomainMetaUpdate] = []
        for basket_id, server in registry.all_servers().items():
            update = self.observe(server, perf_map.get(basket_id))
            updates.append(update)
        return updates

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    @property
    def n_domains_tracked(self) -> int:
        return len(self._states)

    def status(self) -> Dict[str, Any]:
        return {
            "n_domains_tracked": self.n_domains_tracked,
            "basket_ids": list(self._states.keys()),
        }
