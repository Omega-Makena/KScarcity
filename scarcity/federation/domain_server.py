"""
DomainServer — elevated BasketModel with episodic meta-learning.

Each basket now owns a DomainServer that wraps an AdaptationEngine
(ContextEncoder + EpisodicMemory + REPTILE fallback).  DomainServers are
logical agents: they live as objects inside HierarchicalFederation, not as
separate network processes.

Responsibilities:
- Serve adapt() requests from clients during their local inference step
- Absorb incoming client parameter deltas as episodic memory entries
- Evolve the domain base model via REPTILE after each Layer1 aggregation
- Expose status/hit-rate telemetry for monitoring

The DomainServerRegistry manages one DomainServer per basket and is the
single access point for HierarchicalFederation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..meta.encoder import ContextEncoder, ContextEncoderConfig
from ..meta.memory import EpisodicMemory, EpisodicMemoryConfig
from ..meta.optimizer import OnlineReptileOptimizer, MetaOptimizerConfig
from ..meta.adaptation import AdaptationEngine, AdaptationConfig, AdaptationResult


@dataclass
class DomainServerConfig:
    """Configuration for a single DomainServer."""
    memory_capacity: int = 512
    memory_top_k: int = 5
    min_similarity: float = 0.25   # cosine threshold; below → REPTILE fallback
    blend_mode: str = "weighted"   # "weighted" | "top1"
    reptile_beta_init: float = 0.05  # conservative for domain-level REPTILE


class DomainServer:
    """
    Logical domain agent for a single basket.

    Owns the full meta-learning stack for one domain:
    ContextEncoder → EpisodicMemory → AdaptationEngine → REPTILE

    Two write paths:
    - receive_client_update(): called when a client submits a round's delta
    - record(): called after observing an outcome directly
    """

    def __init__(
        self,
        domain_id: str,
        basket_id: str,
        config: Optional[DomainServerConfig] = None,
    ):
        self.domain_id = domain_id
        self.basket_id = basket_id
        self.config = config or DomainServerConfig()

        cfg = self.config
        encoder = ContextEncoder(ContextEncoderConfig(normalize=True))
        memory = EpisodicMemory(
            EpisodicMemoryConfig(capacity=cfg.memory_capacity, top_k=cfg.memory_top_k)
        )
        optimizer = OnlineReptileOptimizer(
            MetaOptimizerConfig(beta_init=cfg.reptile_beta_init)
        )
        adaptation_cfg = AdaptationConfig(
            min_similarity=cfg.min_similarity,
            top_k=cfg.memory_top_k,
            blend_mode=cfg.blend_mode,
        )
        self.engine = AdaptationEngine(
            encoder=encoder,
            memory=memory,
            optimizer=optimizer,
            config=adaptation_cfg,
        )

        # Domain base model — evolves via REPTILE after each aggregation round
        self._base_params: Dict[str, float] = {}
        self._round_id: int = 0
        self._client_update_count: int = 0

    # ------------------------------------------------------------------
    # Meta-learning interface
    # ------------------------------------------------------------------

    def adapt(
        self,
        context: Dict[str, Any],
        base_params: Optional[Dict[str, float]] = None,
        drg_profile: Optional[Dict[str, float]] = None,
    ) -> AdaptationResult:
        """
        Produce adapted parameters for the given context.

        If base_params is not supplied the server's own domain base model is
        used (safe default for client calls that don't have local params yet).
        """
        params = base_params if base_params is not None else self._base_params
        enriched = self._enrich(context)
        return self.engine.adapt(enriched, params, drg_profile)

    def record(
        self,
        context: Dict[str, Any],
        base_params: Dict[str, float],
        adapted_params: Dict[str, float],
        observed_delta: Dict[str, float],
        policy: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store an episode after observing its outcome."""
        enriched = self._enrich(context)
        self.engine.record(enriched, base_params, adapted_params, observed_delta, policy)

    # ------------------------------------------------------------------
    # Federation write path
    # ------------------------------------------------------------------

    def receive_client_update(
        self,
        client_id: str,
        delta_params: Dict[str, float],
        context: Dict[str, Any],
        observed_delta: Dict[str, float],
    ) -> None:
        """
        Accept a parameter delta from a client node after a local training round.

        The delta is stored as an episodic memory entry so future adaptation
        queries can retrieve it.
        """
        enriched = self._enrich(context)
        adapted = {k: self._base_params.get(k, 0.0) + v for k, v in delta_params.items()}
        self.engine.record(
            context=enriched,
            base_params=self._base_params,
            adapted_params=adapted,
            observed_delta=observed_delta,
            policy={
                "source": "client",
                "client_id": client_id,
                "round": self._round_id,
            },
        )
        self._client_update_count += 1

    def update_base_params(
        self,
        aggregated_vector: np.ndarray,
        keys: List[str],
        reward: float,
        drg_profile: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Apply a REPTILE update to the domain base model.

        Called by HierarchicalFederation after Layer1 aggregation for this
        basket.  The updated prior becomes the new base_params for future
        adapt() calls.

        Returns:
            The updated base params dict.
        """
        updated = self.engine.optimizer.apply(
            aggregated_vector, keys, reward, drg_profile or {}
        )
        self._base_params = updated
        self._round_id += 1
        return updated

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    @property
    def base_params(self) -> Dict[str, float]:
        return dict(self._base_params)

    @property
    def memory_size(self) -> int:
        return len(self.engine.memory)

    @property
    def hit_rate(self) -> float:
        return self.engine.hit_rate

    @property
    def round_id(self) -> int:
        return self._round_id

    def status(self) -> Dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "basket_id": self.basket_id,
            "round_id": self._round_id,
            "memory_size": self.memory_size,
            "client_updates": self._client_update_count,
            "hit_rate": self.hit_rate,
            "hits": self.engine.hits,
            "misses": self.engine.misses,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enrich(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Inject domain_id unless the caller already set one."""
        if "domain_id" not in context:
            return {"domain_id": self.domain_id, **context}
        return context


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class DomainServerRegistry:
    """
    Manages one DomainServer per basket.

    HierarchicalFederation holds a registry and calls get_or_create() when a
    basket becomes active.  Servers are created lazily on first access.
    """

    def __init__(self, server_config: Optional[DomainServerConfig] = None):
        self._config = server_config or DomainServerConfig()
        self._servers: Dict[str, DomainServer] = {}  # basket_id → DomainServer

    def get_or_create(self, domain_id: str, basket_id: str) -> DomainServer:
        """Return existing server or create a new one."""
        if basket_id not in self._servers:
            self._servers[basket_id] = DomainServer(domain_id, basket_id, self._config)
        return self._servers[basket_id]

    def get(self, basket_id: str) -> Optional[DomainServer]:
        """Return server for basket_id, or None if not yet created."""
        return self._servers.get(basket_id)

    def all_servers(self) -> Dict[str, DomainServer]:
        """Snapshot of all active servers keyed by basket_id."""
        return dict(self._servers)

    def domain_ids(self) -> List[str]:
        return [s.domain_id for s in self._servers.values()]

    def __len__(self) -> int:
        return len(self._servers)

    def aggregate_status(self) -> List[Dict[str, Any]]:
        """Return status dicts for all servers, for telemetry."""
        return [s.status() for s in self._servers.values()]
