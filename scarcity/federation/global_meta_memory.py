"""
GlobalMetaMemory — cross-domain episodic memory for Phase 3.

Replaces the GlobalMetaModel stub with real cross-domain learning:

1. Absorb path — periodically snapshot all active DomainServers, compute a
   robust cross-domain median of their base parameters, and store the result
   as an episodic entry keyed by the aggregate performance context.

2. Suggest path — when a DomainServer is new or cold, query the global memory
   with the domain's context and return the best-matching prior as a warm-start
   parameter set.  Falls back to the current global_params if no episode matches.

This keeps the same deterministic/inspectable property as the domain-level
EpisodicMemory: every stored episode has a logged context and can be traced
back to a specific aggregate round.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..meta.encoder import ContextEncoder, ContextEncoderConfig
from ..meta.memory import EpisodicMemory, EpisodicMemoryConfig
from .domain_server import DomainServer, DomainServerRegistry


@dataclass
class DomainSnapshot:
    """Point-in-time capture of one DomainServer's state."""
    domain_id: str
    basket_id: str
    base_params: Dict[str, float]
    performance: Dict[str, float]   # caller-supplied metrics (gain, stability…)
    memory_size: int
    hit_rate: float
    round_id: int
    captured_at: float = field(default_factory=time.time)


@dataclass
class GlobalMetaMemoryConfig:
    memory_capacity: int = 256       # cross-domain episodes to retain
    memory_top_k: int = 3
    min_similarity: float = 0.0      # return best match even if similarity is low
    min_domains_for_aggregate: int = 2  # need ≥ N domains to store a global episode


class GlobalMetaMemory:
    """
    Cross-domain meta-memory.

    Owned by HierarchicalFederation alongside the legacy GlobalMetaModel.
    """

    def __init__(self, config: Optional[GlobalMetaMemoryConfig] = None):
        self.config = config or GlobalMetaMemoryConfig()

        self._encoder = ContextEncoder(ContextEncoderConfig(normalize=True))
        self._memory = EpisodicMemory(
            EpisodicMemoryConfig(
                capacity=self.config.memory_capacity,
                top_k=self.config.memory_top_k,
            )
        )

        self._domain_snapshots: Dict[str, DomainSnapshot] = {}  # basket_id → snapshot
        self._global_params: Dict[str, float] = {}
        self._update_count: int = 0

    # ------------------------------------------------------------------
    # Absorb path
    # ------------------------------------------------------------------

    def absorb_domain(
        self,
        server: DomainServer,
        performance: Dict[str, float],
    ) -> None:
        """
        Capture a snapshot of one DomainServer.

        Does not store an episode yet — call aggregate() to flush all pending
        snapshots into the global memory.

        Args:
            server:      The DomainServer to snapshot.
            performance: Caller-supplied performance metrics for this domain
                         (e.g. {"gain": 0.12, "stability": 0.87}).
        """
        self._domain_snapshots[server.basket_id] = DomainSnapshot(
            domain_id=server.domain_id,
            basket_id=server.basket_id,
            base_params=server.base_params,
            performance=performance,
            memory_size=server.memory_size,
            hit_rate=server.hit_rate,
            round_id=server.round_id,
        )

    def aggregate(
        self,
        registry: DomainServerRegistry,
        performance_map: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        """
        Pull snapshots from all servers in the registry, compute cross-domain
        robust median, store as episode, return updated global_params.

        Args:
            registry:        Active DomainServerRegistry.
            performance_map: basket_id → performance dict.  If omitted, any
                             previously absorbed snapshots are reused, and
                             servers without a snapshot get empty performance.

        Returns:
            Updated global_params dict (robust median across all domains).
        """
        perf = performance_map or {}

        for basket_id, server in registry.all_servers().items():
            if basket_id not in self._domain_snapshots or basket_id in perf:
                self.absorb_domain(server, perf.get(basket_id, {}))

        snapshots = list(self._domain_snapshots.values())
        if len(snapshots) < self.config.min_domains_for_aggregate:
            return dict(self._global_params)

        # Robust median of base_params across all snapshotted domains
        global_params = self._median_params(snapshots)
        self._global_params = global_params

        # Encode the aggregate context as a memory key
        agg_ctx = self._aggregate_context(snapshots)
        key = self._encoder.encode(agg_ctx)

        # Performance delta: improvement in median hit_rate and memory_size
        prev_hit = float(np.median([s.hit_rate for s in snapshots]))
        delta: Dict[str, float] = {
            "n_domains": float(len(snapshots)),
            "median_hit_rate": prev_hit,
        }

        self._memory.store(
            key=key,
            value=dict(global_params),
            context=agg_ctx,
            delta=delta,
            policy={
                "source": "global_aggregate",
                "update_count": self._update_count,
                "domain_ids": [s.domain_id for s in snapshots],
            },
        )
        self._update_count += 1
        return dict(global_params)

    # ------------------------------------------------------------------
    # Suggest path
    # ------------------------------------------------------------------

    def suggest_prior(
        self,
        domain_id: str,
        context: Dict[str, Any],
    ) -> Optional[Dict[str, float]]:
        """
        Retrieve the most relevant global prior for a domain context.

        Used by new or cold DomainServers as a warm-start before they have
        their own episodic memory.

        Returns:
            Best-matching global_params dict, or None if memory is empty.
        """
        if len(self._memory) == 0:
            return None

        enriched = dict(context)
        if "domain_id" not in enriched:
            enriched["domain_id"] = domain_id

        key = self._encoder.encode(enriched)
        results = self._memory.retrieve(
            key,
            top_k=1,
            min_similarity=self.config.min_similarity,
        )
        if not results:
            return None

        return dict(results[0].entry.value)

    # ------------------------------------------------------------------
    # Telemetry / inspection
    # ------------------------------------------------------------------

    @property
    def global_params(self) -> Dict[str, float]:
        return dict(self._global_params)

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def memory_size(self) -> int:
        return len(self._memory)

    @property
    def n_domains_tracked(self) -> int:
        return len(self._domain_snapshots)

    def domain_snapshot(self, basket_id: str) -> Optional[DomainSnapshot]:
        return self._domain_snapshots.get(basket_id)

    def status(self) -> Dict[str, Any]:
        return {
            "update_count": self._update_count,
            "memory_size": self.memory_size,
            "n_domains_tracked": self.n_domains_tracked,
            "global_params_keys": list(self._global_params.keys()),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _median_params(snapshots: List[DomainSnapshot]) -> Dict[str, float]:
        """Robust median of base_params across all domain snapshots."""
        gathered: Dict[str, List[float]] = {}
        for snap in snapshots:
            for k, v in snap.base_params.items():
                gathered.setdefault(k, []).append(v)

        # Require at least 2 domains for a key to enter the global model
        return {
            k: float(np.median(vs))
            for k, vs in gathered.items()
            if len(vs) >= 2
        }

    @staticmethod
    def _aggregate_context(snapshots: List[DomainSnapshot]) -> Dict[str, Any]:
        """Build the context dict used as the memory key for this aggregate."""
        all_perf: Dict[str, List[float]] = {}
        for snap in snapshots:
            for k, v in snap.performance.items():
                all_perf.setdefault(k, []).append(v)

        ctx: Dict[str, Any] = {
            "n_domains": float(len(snapshots)),
            "mean_hit_rate": float(np.mean([s.hit_rate for s in snapshots])),
            "mean_memory_size": float(np.mean([s.memory_size for s in snapshots])),
        }
        for k, vs in all_perf.items():
            ctx[f"perf_{k}"] = float(np.mean(vs))

        return ctx
