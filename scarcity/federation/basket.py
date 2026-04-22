"""
Domain Basket System for Hierarchical Federated Learning.

This module implements domain-based grouping of clients for federated learning.
Clients are assigned to baskets based on their domain (manual assignment) with
optional auto-refinement using noisy fingerprints for sub-basket clustering.
"""

from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

import numpy as np


class BasketStatus(str, Enum):
    """Status of a basket."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    FORMING = "forming"  # Not yet minimum size


@dataclass
class BasketConfig:
    """Configuration for domain baskets."""
    auto_refine: bool = False             # Disabled for simple defaults
    fingerprint_noise: float = 0.1        # DP noise for fingerprints
    min_basket_size: int = 3              # Minimum clients for a basket
    max_sub_baskets: int = 5              # Maximum sub-basket depth
    fingerprint_dim: int = 32             # Dimension of fingerprint vectors


@dataclass
class ClientInfo:
    """Information about a registered client."""
    client_id: str
    domain_id: str
    basket_id: str
    fingerprint: Optional[np.ndarray] = None
    registered_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    participation_count: int = 0


@dataclass
class BasketInfo:
    """Information about a basket."""
    basket_id: str
    domain_id: str
    parent_basket_id: Optional[str] = None  # For sub-baskets
    status: BasketStatus = BasketStatus.FORMING
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        self._client_ids: Set[str] = set()
    
    @property
    def client_ids(self) -> Set[str]:
        return self._client_ids
    
    @property
    def size(self) -> int:
        return len(self._client_ids)
    
    def add_client(self, client_id: str) -> None:
        self._client_ids.add(client_id)
    
    def remove_client(self, client_id: str) -> None:
        self._client_ids.discard(client_id)


class BasketManager:
    """
    Manages domain baskets and client membership.
    
    Clients are assigned to baskets based on their domain_id. The basket_id
    is derived from the domain_id (one basket per domain by default).
    
    Features:
    - Manual domain assignment via register_client()
    - Optional auto-refinement with noisy fingerprints (disabled by default)
    - Minimum basket size enforcement
    - Client activity tracking
    """
    
    def __init__(self, config: Optional[BasketConfig] = None):
        """
        Initialize the basket manager.
        
        Args:
            config: Configuration object. Defaults to simple settings.
        """
        self.config = config or BasketConfig()
        self._clients: Dict[str, ClientInfo] = {}
        self._baskets: Dict[str, BasketInfo] = {}
        self._domain_to_basket: Dict[str, str] = {}  # domain_id -> basket_id
        self._rng = np.random.default_rng(42)
    
    def register_client(
        self, 
        client_id: str, 
        domain_id: str,
        features: Optional[np.ndarray] = None
    ) -> str:
        """
        Register a client with manual domain_id assignment.
        
        Args:
            client_id: Unique identifier for the client.
            domain_id: Domain identifier (e.g., 'healthcare', 'finance').
            features: Optional feature vector for fingerprinting.
            
        Returns:
            The basket_id the client was assigned to.
        """
        # Get or create basket for this domain
        basket_id = self._get_or_create_basket(domain_id)
        
        # Create noisy fingerprint if features provided
        fingerprint = None
        if features is not None:
            fingerprint = self._create_noisy_fingerprint(features)
        
        # Register client
        client_info = ClientInfo(
            client_id=client_id,
            domain_id=domain_id,
            basket_id=basket_id,
            fingerprint=fingerprint,
        )
        self._clients[client_id] = client_info
        
        # Add to basket
        self._baskets[basket_id].add_client(client_id)
        
        # Update basket status
        self._update_basket_status(basket_id)
        
        return basket_id
    
    def unregister_client(self, client_id: str) -> bool:
        """
        Remove a client from the system.
        
        Args:
            client_id: Client to remove.
            
        Returns:
            True if client was found and removed.
        """
        if client_id not in self._clients:
            return False
        
        client_info = self._clients[client_id]
        basket_id = client_info.basket_id
        
        # Remove from basket
        if basket_id in self._baskets:
            self._baskets[basket_id].remove_client(client_id)
            self._update_basket_status(basket_id)
        
        # Remove client record
        del self._clients[client_id]
        return True
    
    def get_client(self, client_id: str) -> Optional[ClientInfo]:
        """Get client info by ID."""
        return self._clients.get(client_id)
    
    def get_basket(self, basket_id: str) -> Optional[BasketInfo]:
        """Get basket info by ID."""
        return self._baskets.get(basket_id)
    
    def get_basket_peers(self, basket_id: str) -> List[str]:
        """
        Get all client IDs in a basket.
        
        Args:
            basket_id: Basket identifier.
            
        Returns:
            List of client IDs in the basket.
        """
        basket = self._baskets.get(basket_id)
        if basket is None:
            return []
        return list(basket.client_ids)
    
    def get_active_baskets(self) -> List[str]:
        """Get IDs of all active baskets (meeting minimum size)."""
        return [
            basket_id for basket_id, basket in self._baskets.items()
            if basket.status == BasketStatus.ACTIVE
        ]
    
    def get_basket_for_client(self, client_id: str) -> Optional[str]:
        """Get the basket_id for a client."""
        client = self._clients.get(client_id)
        return client.basket_id if client else None
    
    def get_basket_summary(self, basket_id: str) -> Dict[str, any]:
        """
        Get aggregated basket state for gossip.
        
        Args:
            basket_id: Basket identifier.
            
        Returns:
            Dictionary with basket metadata and client count.
        """
        basket = self._baskets.get(basket_id)
        if basket is None:
            return {}
        
        return {
            "basket_id": basket_id,
            "domain_id": basket.domain_id,
            "status": basket.status.value,
            "client_count": basket.size,
            "created_at": basket.created_at,
        }
    
    def record_participation(self, client_id: str) -> None:
        """Record that a client participated in a round."""
        if client_id in self._clients:
            self._clients[client_id].participation_count += 1
            self._clients[client_id].last_active = time.time()
    
    def get_all_baskets(self) -> Dict[str, BasketInfo]:
        """Get all baskets."""
        return dict(self._baskets)
    
    def get_all_clients(self) -> Dict[str, ClientInfo]:
        """Get all clients."""
        return dict(self._clients)
    
    def _get_or_create_basket(self, domain_id: str) -> str:
        """Get existing basket for domain or create new one."""
        if domain_id in self._domain_to_basket:
            return self._domain_to_basket[domain_id]
        
        # Create new basket
        basket_id = self._generate_basket_id(domain_id)
        basket_info = BasketInfo(
            basket_id=basket_id,
            domain_id=domain_id,
        )
        # Initialize the client_ids set (done in __post_init__ but we need to be explicit)
        basket_info._client_ids = set()
        
        self._baskets[basket_id] = basket_info
        self._domain_to_basket[domain_id] = basket_id
        
        return basket_id
    
    def _generate_basket_id(self, domain_id: str) -> str:
        """Generate a unique basket ID from domain ID."""
        # Simple hash-based ID for determinism
        hash_input = f"basket:{domain_id}"
        hash_bytes = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
        return f"basket_{domain_id}_{hash_bytes}"
    
    def _create_noisy_fingerprint(self, features: np.ndarray) -> np.ndarray:
        """Create a DP-noised fingerprint from features."""
        # Resize/project to fingerprint dimension
        if len(features) != self.config.fingerprint_dim:
            # Simple projection via random hashing
            np.random.seed(hash(tuple(features.flatten()[:10])) % (2**31))
            proj = np.random.randn(len(features), self.config.fingerprint_dim)
            fingerprint = features @ proj / np.sqrt(len(features))
        else:
            fingerprint = features.copy()
        
        # Normalize
        norm = np.linalg.norm(fingerprint)
        if norm > 0:
            fingerprint = fingerprint / norm
        
        # Add DP noise
        noise = self._rng.normal(0, self.config.fingerprint_noise, 
                                  size=self.config.fingerprint_dim)
        fingerprint = fingerprint + noise
        
        return fingerprint.astype(np.float32)
    
    def _update_basket_status(self, basket_id: str) -> None:
        """Update basket status based on size."""
        basket = self._baskets.get(basket_id)
        if basket is None:
            return
        
        if basket.size >= self.config.min_basket_size:
            basket.status = BasketStatus.ACTIVE
        else:
            basket.status = BasketStatus.FORMING
    
    # =========================================================================
    # Auto-refinement (disabled by default, for future use)
    # =========================================================================
    
    def refine_baskets(self) -> Dict[str, List[str]]:
        """
        Auto-refine sub-baskets using noisy fingerprints.
        
        Returns:
            Dictionary mapping new sub-basket IDs to client lists.
            
        Note:
            Disabled by default (config.auto_refine=False).
            When enabled, clusters clients within a domain into sub-baskets
            based on fingerprint similarity.
        """
        if not self.config.auto_refine:
            return {}

        refined: Dict[str, List[str]] = {}
        domains = list(self._domain_to_basket.keys())

        for domain_id in domains:
            base_basket_id = self._domain_to_basket[domain_id]
            domain_clients = [
                client for client in self._clients.values() if client.domain_id == domain_id
            ]

            # Refinement is only meaningful if we can form at least two viable baskets.
            if len(domain_clients) < self.config.min_basket_size * 2:
                continue

            fingerprinted = [c for c in domain_clients if c.fingerprint is not None]
            if len(fingerprinted) < self.config.min_basket_size * 2:
                continue

            matrix = np.stack([c.fingerprint for c in fingerprinted], axis=0).astype(np.float32)
            if not np.isfinite(matrix).all():
                continue

            k_target = max(2, len(fingerprinted) // self.config.min_basket_size)
            k = min(self.config.max_sub_baskets, max(2, k_target), len(fingerprinted))
            assignments = self._kmeans_assign(matrix, k)
            clusters = self._normalize_clusters(assignments, matrix, k)
            viable = [cluster for cluster in clusters if len(cluster) >= self.config.min_basket_size]

            # Keep domain untouched unless we can create at least two viable refined baskets.
            if len(viable) < 2:
                continue

            # Remove old sub-baskets for this domain before rebuilding.
            self._drop_domain_sub_baskets(base_basket_id)

            # Clear current membership for all domain clients.
            for client in domain_clients:
                old_basket = self._baskets.get(client.basket_id)
                if old_basket is not None:
                    old_basket.remove_client(client.client_id)

            # Build new sub-baskets and reassign clients.
            for sub_index, member_indices in enumerate(viable, start=1):
                sub_basket_id = f"{base_basket_id}_sub_{sub_index}"
                basket = BasketInfo(
                    basket_id=sub_basket_id,
                    domain_id=domain_id,
                    parent_basket_id=base_basket_id,
                )
                basket._client_ids = set()
                self._baskets[sub_basket_id] = basket

                member_ids: List[str] = []
                for idx in member_indices:
                    client = fingerprinted[idx]
                    client.basket_id = sub_basket_id
                    basket.add_client(client.client_id)
                    member_ids.append(client.client_id)

                self._update_basket_status(sub_basket_id)
                refined[sub_basket_id] = member_ids

            # Any domain client without fingerprint falls back to the base basket.
            base_basket = self._baskets.get(base_basket_id)
            if base_basket is None:
                base_basket = BasketInfo(basket_id=base_basket_id, domain_id=domain_id)
                base_basket._client_ids = set()
                self._baskets[base_basket_id] = base_basket

            fingerprinted_ids = {client.client_id for client in fingerprinted}
            for client in domain_clients:
                if client.client_id in fingerprinted_ids:
                    continue
                client.basket_id = base_basket_id
                base_basket.add_client(client.client_id)

            self._update_basket_status(base_basket_id)

        return refined

    def _kmeans_assign(self, matrix: np.ndarray, k: int, max_iter: int = 12) -> np.ndarray:
        """Compute cluster assignments using a lightweight k-means loop."""
        n = matrix.shape[0]
        if k <= 1 or n <= 1:
            return np.zeros(n, dtype=np.int32)

        init_idx = self._rng.choice(n, size=k, replace=False)
        centroids = matrix[init_idx].copy()

        assignments = np.zeros(n, dtype=np.int32)
        for _ in range(max_iter):
            dists = np.linalg.norm(matrix[:, None, :] - centroids[None, :, :], axis=2)
            new_assignments = np.argmin(dists, axis=1).astype(np.int32)

            if np.array_equal(assignments, new_assignments):
                break

            assignments = new_assignments
            for cluster_id in range(k):
                members = matrix[assignments == cluster_id]
                if len(members) == 0:
                    refill_idx = int(self._rng.integers(0, n))
                    centroids[cluster_id] = matrix[refill_idx]
                else:
                    centroids[cluster_id] = members.mean(axis=0)

        return assignments

    def _normalize_clusters(
        self,
        assignments: np.ndarray,
        matrix: np.ndarray,
        k: int,
    ) -> List[List[int]]:
        """Merge undersized clusters into nearest larger clusters."""
        clusters: List[List[int]] = [np.where(assignments == cluster_id)[0].tolist() for cluster_id in range(k)]

        if k <= 1:
            return clusters

        centroids: List[np.ndarray] = []
        for members in clusters:
            if members:
                centroids.append(matrix[members].mean(axis=0))
            else:
                centroids.append(np.zeros(matrix.shape[1], dtype=np.float32))

        min_size = self.config.min_basket_size
        for cluster_id in range(k):
            members = clusters[cluster_id]
            if len(members) == 0 or len(members) >= min_size:
                continue

            candidates = [
                target
                for target in range(k)
                if target != cluster_id and len(clusters[target]) >= min_size
            ]
            if not candidates:
                continue

            src_centroid = centroids[cluster_id]
            target = min(
                candidates,
                key=lambda c: float(np.linalg.norm(src_centroid - centroids[c])),
            )
            clusters[target].extend(members)
            clusters[cluster_id] = []

        return clusters

    def _drop_domain_sub_baskets(self, base_basket_id: str) -> None:
        """Remove existing sub-baskets for a domain base basket."""
        to_drop = [
            basket_id
            for basket_id, basket in self._baskets.items()
            if basket.parent_basket_id == base_basket_id
        ]
        for basket_id in to_drop:
            del self._baskets[basket_id]
