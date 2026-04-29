"""
SCARCITY Engine — Online Relationship Discovery.

Top-level exports for the new Relationship Discovery Engine.
"""

from .discovery import Hypothesis, RelationshipType
from .engine_v2 import OnlineDiscoveryEngine
from .grouping import AdaptiveGrouper
from .bandit_router import BanditRouter, BanditConfig, BanditAlgorithm
from .baskets import Basket, BasketRegistry, REGISTRY as BasketRegistrySingleton
from .federation_node import FederationNode
from .federation_hub import FederationHub

# Expose the new engine as the default
Engine = OnlineDiscoveryEngine

__all__ = [
    'OnlineDiscoveryEngine',
    'Engine',
    'Hypothesis',
    'RelationshipType',
    'AdaptiveGrouper',
    'BanditRouter',
    'BanditConfig',
    'BanditAlgorithm',
    'Basket',
    'BasketRegistry',
    'BasketRegistrySingleton',
    'FederationNode',
    'FederationHub',
]
