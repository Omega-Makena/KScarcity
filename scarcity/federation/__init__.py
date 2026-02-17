"""
Federated learning layer for SCARCITY.

Exposes high-level orchestrators and packet schemas used by the online,
model-free federation pipeline. Includes hierarchical federation with
domain baskets, gossip learning, and two-layer aggregation.
"""

from .packets import PathPack, EdgeDelta, PolicyPack, CausalSemanticPack
from .aggregator import AggregationMethod, FederatedAggregator
from .trust_scorer import TrustScorer
from .privacy_guard import PrivacyGuard
from .validator import PacketValidator
from .scheduler import FederationScheduler
from .client_agent import FederationClientAgent
from .coordinator import FederationCoordinator
from .reconciler import StoreReconciler, build_reconciler
from .codec import PayloadCodec

# Hierarchical federation components
from .basket import BasketManager, BasketConfig, BasketStatus, ClientInfo, BasketInfo
from .gossip import (
    GossipProtocol, 
    GossipConfig, 
    GossipMessage, 
    LocalDPMechanism,
    MaterialityDetector,
    PeerSampler,
)
from .buffer import (
    UpdateBuffer, 
    BufferConfig, 
    BufferedUpdate, 
    TriggerEngine, 
    PrivacyAccountant,
    ReplayGuard,
)
from .layers import (
    Layer1Aggregator, 
    Layer1Config,
    Layer2Aggregator, 
    Layer2Config,
    SecureAggregator,
    CentralDPMechanism,
    BasketModel,
    GlobalMetaModel,
)
from .secure_aggregation import (
    IdentityKeyPair,
    EphemeralKeyPair,
    EphemeralKeyRecord,
    SecureAggClient,
    SecureAggCoordinator,
)
from .hierarchical import HierarchicalFederation, HierarchicalFederationConfig

__all__ = [
    # Original exports
    "PathPack",
    "EdgeDelta",
    "PolicyPack",
    "CausalSemanticPack",
    "AggregationMethod",
    "FederatedAggregator",
    "TrustScorer",
    "PrivacyGuard",
    "PacketValidator",
    "FederationScheduler",
    "FederationClientAgent",
    "FederationCoordinator",
    "StoreReconciler",
    "build_reconciler",
    "PayloadCodec",
    # Hierarchical federation
    "HierarchicalFederation",
    "HierarchicalFederationConfig",
    "BasketManager",
    "BasketConfig",
    "BasketStatus",
    "ClientInfo",
    "BasketInfo",
    "GossipProtocol",
    "GossipConfig",
    "GossipMessage",
    "LocalDPMechanism",
    "MaterialityDetector",
    "PeerSampler",
    "UpdateBuffer",
    "BufferConfig",
    "BufferedUpdate",
    "TriggerEngine",
    "PrivacyAccountant",
    "ReplayGuard",
    "Layer1Aggregator",
    "Layer1Config",
    "Layer2Aggregator",
    "Layer2Config",
    "SecureAggregator",
    "CentralDPMechanism",
    "BasketModel",
    "GlobalMetaModel",
    "IdentityKeyPair",
    "EphemeralKeyPair",
    "EphemeralKeyRecord",
    "SecureAggClient",
    "SecureAggCoordinator",
]
