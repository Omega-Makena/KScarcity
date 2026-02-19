"""Federation coordinator for multi-domain model sharing."""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class AggregationStrategy(str, Enum):
    """Model aggregation strategies."""
    FEDAVG = "fedavg"  # Federated averaging
    WEIGHTED = "weighted"  # Weighted by domain size
    ADAPTIVE = "adaptive"  # Adaptive based on performance


@dataclass
class ModelUpdate:
    """Model update from a domain."""
    domain_id: int
    timestamp: datetime
    weights: np.ndarray
    num_samples: int
    loss: float
    metadata: Dict


@dataclass
class Connection:
    """P2P connection between domains."""
    from_domain: int
    to_domain: int
    established_at: datetime
    updates_shared: int
    last_update_at: Optional[datetime] = None


@dataclass
class ConnectionInfo:
    """Connection information for API."""
    from_domain: int
    to_domain: int
    established_at: str
    updates_shared: int
    last_update_at: Optional[str]


class ModelAggregator:
    """Aggregates model updates from multiple domains."""
    
    def __init__(self, strategy: AggregationStrategy = AggregationStrategy.FEDAVG):
        self.strategy = strategy
    
    def aggregate(self, updates: List[ModelUpdate]) -> Optional[np.ndarray]:
        """
        Aggregate model updates using configured strategy.
        
        Args:
            updates: List of model updates from domains
            
        Returns:
            Aggregated model weights or None if no updates
        """
        if not updates:
            return None
        
        if self.strategy == AggregationStrategy.FEDAVG:
            return self._federated_average(updates)
        elif self.strategy == AggregationStrategy.WEIGHTED:
            return self._weighted_average(updates)
        elif self.strategy == AggregationStrategy.ADAPTIVE:
            return self._adaptive_average(updates)
        
        return self._federated_average(updates)
    
    def _federated_average(self, updates: List[ModelUpdate]) -> np.ndarray:
        """Simple federated averaging."""
        total_samples = sum(u.num_samples for u in updates)
        
        if total_samples == 0:
            # Unweighted average
            return np.mean([u.weights for u in updates], axis=0)
        
        # Weighted by number of samples
        aggregated = np.zeros_like(updates[0].weights)
        for update in updates:
            weight = update.num_samples / total_samples
            aggregated += weight * update.weights
        
        return aggregated
    
    def _weighted_average(self, updates: List[ModelUpdate]) -> np.ndarray:
        """Weighted average by domain size."""
        return self._federated_average(updates)
    
    def _adaptive_average(self, updates: List[ModelUpdate]) -> np.ndarray:
        """Adaptive averaging based on loss."""
        # Weight by inverse loss (better models get more weight)
        total_inv_loss = sum(1.0 / (u.loss + 1e-6) for u in updates)
        
        aggregated = np.zeros_like(updates[0].weights)
        for update in updates:
            weight = (1.0 / (update.loss + 1e-6)) / total_inv_loss
            aggregated += weight * update.weights
        
        return aggregated


class DifferentialPrivacy:
    """Differential privacy for federated learning."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
    
    def add_noise(self, weights: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        """
        Add Gaussian noise for differential privacy.
        
        Args:
            weights: Model weights
            sensitivity: Sensitivity of the query
            
        Returns:
            Noised weights
        """
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon
        noise = np.random.normal(0, sigma, weights.shape)
        return weights + noise


class FederationCoordinator:
    """
    Coordinates federated learning across domains.
    
    Manages P2P connections, model sharing, and aggregation.
    """
    
    def __init__(
        self,
        strategy: AggregationStrategy = AggregationStrategy.FEDAVG,
        enable_privacy: bool = False
    ):
        self.active = False
        self.connections: Dict[Tuple[int, int], Connection] = {}
        self.aggregator = ModelAggregator(strategy)
        self.privacy_engine = DifferentialPrivacy() if enable_privacy else None
        
        # Update buffers per domain
        self.pending_updates: Dict[int, List[ModelUpdate]] = {}
        
        # Statistics
        self.total_rounds = 0
        self.total_updates_shared = 0
        
        logger.info(f"Federation coordinator initialized with strategy={strategy}, privacy={enable_privacy}")
    
    def enable_federation(self):
        """Activate federated learning."""
        if self.active:
            logger.warning("Federation already active")
            return
        
        self.active = True
        logger.info("Federation enabled")
    
    def disable_federation(self):
        """Deactivate federated learning."""
        if not self.active:
            logger.warning("Federation already inactive")
            return
        
        self.active = False
        logger.info("Federation disabled")
    
    def create_connection(self, from_domain: int, to_domain: int):
        """
        Create P2P connection between domains.
        
        Args:
            from_domain: Source domain ID
            to_domain: Target domain ID
        """
        key = (from_domain, to_domain)
        
        if key in self.connections:
            logger.warning(f"Connection {from_domain} -> {to_domain} already exists")
            return
        
        connection = Connection(
            from_domain=from_domain,
            to_domain=to_domain,
            established_at=datetime.utcnow(),
            updates_shared=0
        )
        
        self.connections[key] = connection
        logger.info(f"Created connection: {from_domain} -> {to_domain}")
    
    def remove_connection(self, from_domain: int, to_domain: int):
        """Remove P2P connection."""
        key = (from_domain, to_domain)
        
        if key not in self.connections:
            logger.warning(f"Connection {from_domain} -> {to_domain} does not exist")
            return
        
        del self.connections[key]
        logger.info(f"Removed connection: {from_domain} -> {to_domain}")
    
    async def share_update(
        self,
        from_domain: int,
        to_domain: int,
        update: ModelUpdate
    ):
        """
        Share model update between domains.
        
        Args:
            from_domain: Source domain
            to_domain: Target domain
            update: Model update to share
        """
        if not self.active:
            logger.warning("Federation not active, skipping update")
            return
        
        key = (from_domain, to_domain)
        
        if key not in self.connections:
            logger.warning(f"No connection from {from_domain} to {to_domain}")
            return
        
        # Apply privacy if enabled
        if self.privacy_engine:
            update.weights = self.privacy_engine.add_noise(update.weights)
        
        # Store update for target domain
        if to_domain not in self.pending_updates:
            self.pending_updates[to_domain] = []
        
        self.pending_updates[to_domain].append(update)
        
        # Update connection stats
        connection = self.connections[key]
        connection.updates_shared += 1
        connection.last_update_at = datetime.utcnow()
        
        self.total_updates_shared += 1
        
        logger.debug(f"Shared update from domain {from_domain} to {to_domain}")
    
    def aggregate_updates(self, domain_id: int) -> Optional[np.ndarray]:
        """
        Aggregate pending updates for a domain.
        
        Args:
            domain_id: Domain to aggregate updates for
            
        Returns:
            Aggregated weights or None if no updates
        """
        if domain_id not in self.pending_updates:
            return None
        
        updates = self.pending_updates[domain_id]
        
        if not updates:
            return None
        
        # Aggregate
        aggregated = self.aggregator.aggregate(updates)
        
        # Clear pending updates
        self.pending_updates[domain_id] = []
        
        self.total_rounds += 1
        
        logger.info(f"Aggregated {len(updates)} updates for domain {domain_id}")
        
        return aggregated
    
    def get_connections(self) -> List[ConnectionInfo]:
        """Get all active P2P connections."""
        return [
            ConnectionInfo(
                from_domain=conn.from_domain,
                to_domain=conn.to_domain,
                established_at=conn.established_at.isoformat() + "Z",
                updates_shared=conn.updates_shared,
                last_update_at=conn.last_update_at.isoformat() + "Z" if conn.last_update_at else None
            )
            for conn in self.connections.values()
        ]
    
    def get_metrics(self) -> Dict:
        """Get federation metrics."""
        return {
            "active": self.active,
            "total_connections": len(self.connections),
            "total_rounds": self.total_rounds,
            "total_updates_shared": self.total_updates_shared,
            "strategy": self.aggregator.strategy.value,
            "privacy_enabled": self.privacy_engine is not None,
            "pending_updates": {
                domain_id: len(updates)
                for domain_id, updates in self.pending_updates.items()
            }
        }
    
    def create_full_mesh(self, domain_ids: List[int]):
        """
        Create full mesh topology (all-to-all connections).
        
        Args:
            domain_ids: List of domain IDs to connect
        """
        for from_domain in domain_ids:
            for to_domain in domain_ids:
                if from_domain != to_domain:
                    self.create_connection(from_domain, to_domain)
        
        logger.info(f"Created full mesh topology for {len(domain_ids)} domains")
    
    def create_ring_topology(self, domain_ids: List[int]):
        """
        Create ring topology (each domain connects to next).
        
        Args:
            domain_ids: List of domain IDs to connect in order
        """
        for i in range(len(domain_ids)):
            from_domain = domain_ids[i]
            to_domain = domain_ids[(i + 1) % len(domain_ids)]
            self.create_connection(from_domain, to_domain)
        
        logger.info(f"Created ring topology for {len(domain_ids)} domains")
