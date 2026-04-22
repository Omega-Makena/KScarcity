"""
Packet schemas exchanged by the SCARCITY federated learning layer.

All packets are dataclasses with helper methods for serialisation and validation.
This module defines the structures for PathPacks, EdgeDeltas, PolicyPacks,
and CausalSemanticPacks, ensuring type safety and consistency across the federation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import hashlib


def _hash_schema(schema_hash: str, domain_id: int) -> str:
    """Create a composite hash for schema-domain pairing."""
    composite = f"{schema_hash}:{domain_id}".encode("utf-8")
    return hashlib.blake2b(composite, digest_size=12).hexdigest()


@dataclass
class Provenance:
    """Metadata describing the origin of a packet."""
    config_hash: str
    tier_set: Sequence[str]
    encoder_profile: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "config_hash": self.config_hash,
            "tier_set": list(self.tier_set),
            "encoder_profile": dict(self.encoder_profile),
        }


@dataclass
class PathPack:
    """Represents a set of discovered causal paths."""
    schema_hash: str
    window_range: Tuple[int, int]
    domain_id: int
    revision: int
    edges: List[Tuple[str, str, float, float, float, int]]
    hyperedges: List[Dict[str, Any]]
    operator_stats: Dict[str, float]
    provenance: Provenance

    def composite_id(self) -> str:
        """Generate a unique ID for this pack based on schema and domain."""
        return _hash_schema(self.schema_hash, self.domain_id)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "schema_hash": self.schema_hash,
            "window_range": list(self.window_range),
            "domain_id": self.domain_id,
            "revision": self.revision,
            "edges": [list(edge) for edge in self.edges],
            "hyper": self.hyperedges,
            "operator_stats": dict(self.operator_stats),
            "provenance": self.provenance.to_dict(),
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "PathPack":
        """Create from dictionary."""
        provenance = Provenance(**payload["provenance"])
        edges = [tuple(edge) for edge in payload.get("edges", [])]
        return PathPack(
            schema_hash=payload["schema_hash"],
            window_range=tuple(payload["window_range"]),
            domain_id=payload["domain_id"],
            revision=payload["revision"],
            edges=edges,  # type: ignore[arg-type]
            hyperedges=list(payload.get("hyper", [])),
            operator_stats=dict(payload.get("operator_stats", {})),
            provenance=provenance,
        )


@dataclass
class EdgeDelta:
    """Represents incremental updates (upserts/prunes) to edges."""
    schema_hash: str
    domain_id: int
    revision: int
    upserts: List[Tuple[str, float, float, int, int, int]]
    prunes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "schema_hash": self.schema_hash,
            "domain_id": self.domain_id,
            "revision": self.revision,
            "upserts": [list(u) for u in self.upserts],
            "prunes": list(self.prunes),
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "EdgeDelta":
        """Create from dictionary."""
        upserts = [tuple(item) for item in payload.get("upserts", [])]
        return EdgeDelta(
            schema_hash=payload["schema_hash"],
            domain_id=payload["domain_id"],
            revision=payload["revision"],
            upserts=upserts,  # type: ignore[arg-type]
            prunes=list(payload.get("prunes", [])),
        )


@dataclass
class PolicyPack:
    """Represents a set of policy parameters."""
    controller: Dict[str, float]
    evaluator: Dict[str, float]
    drg: Dict[str, float]
    evidence: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "PolicyPack":
        """Create from dictionary."""
        return PolicyPack(
            controller=dict(payload.get("controller", {})),
            evaluator=dict(payload.get("evaluator", {})),
            drg=dict(payload.get("drg", {})),
            evidence=dict(payload.get("evidence", {})),
        )


@dataclass
class CausalPair:
    """A pair of causally related variables."""
    source: str
    target: str
    probability: float
    direction: int
    regime: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "src": self.source,
            "dst": self.target,
            "prob": float(self.probability),
            "direction": int(self.direction),
            "regime": self.regime,
        }


@dataclass
class ConceptLink:
    """A link between an abstract concept and variables."""
    concept_id: str
    score: float
    links: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.concept_id,
            "score": float(self.score),
            "links": list(self.links),
        }


@dataclass
class CausalSemanticPack:
    """Represents high-level causal and semantic knowledge."""
    schema_hash: str
    domain_id: int
    revision: int
    pairs: List[CausalPair]
    concepts: List[ConceptLink]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "schema_hash": self.schema_hash,
            "domain_id": self.domain_id,
            "revision": self.revision,
            "pairs": [pair.to_dict() for pair in self.pairs],
            "concepts": [concept.to_dict() for concept in self.concepts],
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "CausalSemanticPack":
        """Create from dictionary."""
        pairs = [
            CausalPair(
                source=item["src"],
                target=item["dst"],
                probability=item["prob"],
                direction=item.get("direction", 1),
                regime=item.get("regime"),
            )
            for item in payload.get("pairs", [])
        ]
        concepts = [
            ConceptLink(concept_id=item["id"], score=item["score"], links=item.get("links", []))
            for item in payload.get("concepts", [])
        ]
        return CausalSemanticPack(
            schema_hash=payload["schema_hash"],
            domain_id=payload["domain_id"],
            revision=payload["revision"],
            pairs=pairs,
            concepts=concepts,
        )


# ---------------------------------------------------------------------------
# Meta-learning / adaptation packets (Phase 4)
# ---------------------------------------------------------------------------

@dataclass
class AdaptationRequest:
    """
    Sent by a DomainServer (or any caller) to request a warm-start prior from
    the global meta-memory.

    basket_id:   the requesting basket
    domain_id:   human-readable domain label
    context:     feature dict describing the current task context
    round_id:    optional round for ordering
    """
    basket_id: str
    domain_id: str
    context: Dict[str, Any]
    round_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "basket_id": self.basket_id,
            "domain_id": self.domain_id,
            "context": dict(self.context),
            "round_id": self.round_id,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "AdaptationRequest":
        return AdaptationRequest(
            basket_id=payload["basket_id"],
            domain_id=payload["domain_id"],
            context=dict(payload.get("context", {})),
            round_id=int(payload.get("round_id", 0)),
        )


@dataclass
class AdaptationResponse:
    """
    Reply to an AdaptationRequest.

    basket_id:     echoes the requesting basket
    domain_id:     echoes the requesting domain
    prior_params:  warm-start parameter dict (may be empty if no prior available)
    source:        "global_memory" | "passthrough"
    round_id:      echoes round from request
    """
    basket_id: str
    domain_id: str
    prior_params: Dict[str, float]
    source: str
    round_id: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "basket_id": self.basket_id,
            "domain_id": self.domain_id,
            "prior_params": dict(self.prior_params),
            "source": self.source,
            "round_id": self.round_id,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "AdaptationResponse":
        return AdaptationResponse(
            basket_id=payload["basket_id"],
            domain_id=payload["domain_id"],
            prior_params=dict(payload.get("prior_params", {})),
            source=str(payload.get("source", "passthrough")),
            round_id=int(payload.get("round_id", 0)),
        )


@dataclass
class DomainSyncPacket:
    """
    Periodic snapshot broadcast from one DomainServer to the global coordinator.

    basket_id:    source basket
    domain_id:    source domain
    base_params:  current REPTILE prior
    performance:  caller-supplied metrics (gain, stability, …)
    memory_size:  current episodic memory size
    hit_rate:     adapter hit rate
    round_id:     current domain round
    """
    basket_id: str
    domain_id: str
    base_params: Dict[str, float]
    performance: Dict[str, float]
    memory_size: int
    hit_rate: float
    round_id: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "basket_id": self.basket_id,
            "domain_id": self.domain_id,
            "base_params": dict(self.base_params),
            "performance": dict(self.performance),
            "memory_size": self.memory_size,
            "hit_rate": float(self.hit_rate),
            "round_id": self.round_id,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "DomainSyncPacket":
        return DomainSyncPacket(
            basket_id=payload["basket_id"],
            domain_id=payload["domain_id"],
            base_params=dict(payload.get("base_params", {})),
            performance=dict(payload.get("performance", {})),
            memory_size=int(payload.get("memory_size", 0)),
            hit_rate=float(payload.get("hit_rate", 0.0)),
            round_id=int(payload.get("round_id", 0)),
        )


PacketType = Tuple[str, Dict[str, Any]]


def serialise_packet(packet: Any) -> PacketType:
    """
    Return typed payload for bus transport.
    
    Args:
        packet: The packet object to serialize.
        
    Returns:
        A tuple of (topic_string, payload_dict).
        
    Raises:
        TypeError: If the packet type is not supported.
    """
    if isinstance(packet, PathPack):
        return ("federation.path_pack", packet.to_dict())
    if isinstance(packet, EdgeDelta):
        return ("federation.edge_delta", packet.to_dict())
    if isinstance(packet, PolicyPack):
        return ("federation.policy_pack", packet.to_dict())
    if isinstance(packet, CausalSemanticPack):
        return ("federation.causal_pack", packet.to_dict())
    if isinstance(packet, AdaptationRequest):
        return ("federation.adaptation_request", packet.to_dict())
    if isinstance(packet, AdaptationResponse):
        return ("federation.adaptation_response", packet.to_dict())
    if isinstance(packet, DomainSyncPacket):
        return ("federation.domain_sync", packet.to_dict())
    raise TypeError(f"Unsupported packet type: {type(packet)}")


def normalise_packets(payloads: Sequence[PacketType]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group packets by topic for efficient batching.
    
    Args:
        payloads: Sequence of (topic, payload) tuples.
        
    Returns:
        Dictionary mapping topics to lists of payloads.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for topic, payload in payloads:
        grouped.setdefault(topic, []).append(payload)
    return grouped
