"""
Two-Layer Aggregation for Hierarchical Federated Learning.

This module implements:
- Layer 1 (Intra-Basket): Gossip consensus within a domain basket
- Layer 2 (Cross-Basket): Global aggregation with bounded influence + DP

Privacy Model (Fork 2 - Secure Aggregation + DP):
- Secure aggregation uses masking-based summation (non-cryptographic in-process)
- Server only sees masked updates; unmasking uses aggregate seeds in coordinator
- Bounded influence per basket via L2 clipping
- Central DP noise added to final global aggregate
- Minimum support filtering for minority protection
"""

from __future__ import annotations

import time
import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Sequence
from enum import Enum

import numpy as np

from .aggregator import FederatedAggregator, AggregationConfig, AggregationMethod
from .buffer import UpdateBuffer, BufferConfig, BufferedUpdate, PrivacyAccountant
from .gossip import GossipProtocol, GossipConfig, GossipMessage
from .privacy_guard import PrivacyGuard, PrivacyConfig
from .secure_aggregation import (
    IdentityKeyPair,
    SecureAggClient,
    SecureAggCoordinator,
)

logger = logging.getLogger(__name__)

@dataclass
class Layer1Config:
    """Configuration for intra-basket aggregation."""
    aggregation_method: AggregationMethod = AggregationMethod.TRIMMED_MEAN
    trim_alpha: float = 0.1
    clip_norm: float = 1.0                # Per-update clipping


@dataclass
class Layer2Config:
    """Configuration for cross-basket aggregation."""
    aggregation_method: AggregationMethod = AggregationMethod.BULYAN
    basket_clip_norm: float = 5.0         # Bounded influence per basket
    min_basket_support: int = 2           # Suppress if seen in <N baskets
    multi_krum_m: int = 5                 # For Krum/Bulyan
    
    # Central DP at global layer
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    # Secure aggregation (masking-based)
    secure_aggregation: bool = True
    sa_seed_length: int = 16
    secure_aggregation_mode: str = "masking"  # masking | crypto
    auto_generate_identities: bool = True


@dataclass
class BasketModel:
    """Per-basket specialized state."""
    basket_id: str
    aggregate_vector: np.ndarray
    update_count: int
    last_updated: float = field(default_factory=time.time)
    round_id: int = 0
    
    # Knowledge edges discovered in this basket
    knowledge_edges: Dict[Tuple[str, str], float] = field(default_factory=dict)
    
    # Basket-specific hypothesis parameters
    hypothesis_params: Dict[str, float] = field(default_factory=dict)


class Layer1Aggregator:
    """
    Intra-basket aggregation using gossip consensus.
    
    Aggregates all updates within a basket using robust methods
    (trimmed mean by default). No DP noise at this layer since
    gossip already applied local DP.
    """
    
    def __init__(
        self, 
        config: Layer1Config,
        buffer: UpdateBuffer
    ):
        """
        Initialize Layer 1 aggregator.
        
        Args:
            config: Layer 1 configuration
            buffer: Update buffer containing client updates
        """
        self.config = config
        self.buffer = buffer
        
        # Create aggregator
        agg_config = AggregationConfig(
            method=config.aggregation_method,
            trim_alpha=config.trim_alpha,
        )
        self.aggregator = FederatedAggregator(agg_config)
        
        # Per-basket models
        self._basket_models: Dict[str, BasketModel] = {}
        self._round_id: int = 0
    
    def aggregate_basket(self, basket_id: str) -> Optional[np.ndarray]:
        """
        Aggregate all updates within a basket.
        
        Args:
            basket_id: Basket to aggregate
            
        Returns:
            Aggregated vector, or None if no updates
        """
        # Get weighted aggregate from buffer
        aggregate, count = self.buffer.get_weighted(basket_id)
        
        if count == 0:
            return None
        
        # Apply L2 clipping to the basket aggregate
        aggregate = self._clip_vector(aggregate, self.config.clip_norm)
        
        # Update basket model
        self._basket_models[basket_id] = BasketModel(
            basket_id=basket_id,
            aggregate_vector=aggregate,
            update_count=count,
            round_id=self._round_id,
        )
        
        # Clear basket buffer
        self.buffer.clear_basket(basket_id)
        
        return aggregate
    
    def get_basket_model(self, basket_id: str) -> Optional[BasketModel]:
        """Get the current model for a basket."""
        return self._basket_models.get(basket_id)
    
    def get_basket_models(self) -> Dict[str, BasketModel]:
        """Get all per-basket models."""
        return dict(self._basket_models)
    
    def get_ready_baskets(self) -> List[str]:
        """Get baskets that have fresh aggregates since last global."""
        return [
            bid for bid, model in self._basket_models.items()
            if model.round_id == self._round_id
        ]
    
    def advance_round(self) -> None:
        """Advance to next round."""
        self._round_id += 1
    
    def _clip_vector(self, vector: np.ndarray, clip_norm: float) -> np.ndarray:
        """L2 clip a vector."""
        norm = np.linalg.norm(vector)
        if norm > clip_norm:
            return vector * (clip_norm / norm)
        return vector


class SecureAggregator:
    """
    Secure aggregation protocol (simplified simulation).
    
    In production, this would use cryptographic protocols like:
    - Pairwise masking (Bonawitz et al.)
    - Threshold secret sharing
    - Trusted Execution Environments
    
    For this implementation, we simulate the sum-only property:
    the server only sees the sum of inputs, not individual values.
    """
    
    def __init__(self, min_participants: int = 2):
        """
        Initialize secure aggregator.
        
        Args:
            min_participants: Minimum participants required for aggregation
        """
        self.min_participants = min_participants
        self._pending_shares: List[np.ndarray] = []
        self._participant_count: int = 0
        logger.warning(
            "SecureAggregator uses masking-based summation in-process; "
            "this is not a cryptographic secure aggregation protocol."
        )
    
    def submit_share(self, share: np.ndarray) -> None:
        """
        Submit a masked share for aggregation.
        
        In production: this would be a cryptographically masked value.
        Here we just collect the values.
        """
        self._pending_shares.append(share.copy())
        self._participant_count += 1
    
    def aggregate(self) -> Optional[np.ndarray]:
        """
        Perform secure aggregation.
        
        Returns:
            Sum of all shares, or None if not enough participants
        """
        if self._participant_count < self.min_participants:
            return None
        
        if not self._pending_shares:
            return None
        
        # In production: cryptographic unmask + sum
        # Here: simple sum
        result = np.zeros_like(self._pending_shares[0])
        for share in self._pending_shares:
            result += share
        
        # Clear for next round
        self._pending_shares = []
        self._participant_count = 0
        
        return result
    
    def reset(self) -> None:
        """Reset the aggregator."""
        self._pending_shares = []
        self._participant_count = 0
    
    @property
    def pending_count(self) -> int:
        """Get number of pending shares."""
        return self._participant_count


class CentralDPMechanism:
    """
    Central DP mechanism for global aggregates.
    
    Adds calibrated Gaussian noise to the final aggregate
    after secure aggregation.
    """
    
    def __init__(
        self, 
        epsilon: float, 
        delta: float, 
        sensitivity: float,
        accountant: Optional[PrivacyAccountant] = None
    ):
        """
        Initialize central DP mechanism.
        
        Args:
            epsilon: Privacy parameter
            delta: Privacy failure probability
            sensitivity: L2 sensitivity (basket_clip_norm * num_baskets)
            accountant: Optional privacy accountant for tracking
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.accountant = accountant
        self._rng = np.random.default_rng()
        
        # Compute noise scale
        self.sigma = self._compute_sigma()
    
    def _compute_sigma(self) -> float:
        """Compute noise standard deviation for (ε,δ)-DP."""
        if self.epsilon <= 0:
            return float('inf')
        c = math.sqrt(2 * math.log(1.25 / self.delta))
        return self.sensitivity * c / self.epsilon
    
    def add_noise(self, aggregate: np.ndarray) -> Optional[np.ndarray]:
        """
        Add calibrated DP noise to aggregate.
        
        Args:
            aggregate: Sum from secure aggregation
            
        Returns:
            Noised aggregate, or None if budget exhausted
        """
        # Check budget
        if self.accountant is not None:
            if not self.accountant.spend(self.epsilon, self.delta):
                return None
        
        # Add noise
        noise = self._rng.normal(0, self.sigma, size=aggregate.shape)
        return aggregate + noise.astype(aggregate.dtype)
    
    def update_sensitivity(self, sensitivity: float) -> None:
        """Update sensitivity and recompute sigma."""
        self.sensitivity = sensitivity
        self.sigma = self._compute_sigma()


class Layer2Aggregator:
    """
    Cross-basket global aggregation with Secure Aggregation + DP.
    
    Features:
    - Bounded influence: clip each basket contribution
    - Secure aggregation: server only sees sum
    - Central DP: noise added to final aggregate
    - Minimum support: suppress edges seen in too few baskets
    """
    
    def __init__(
        self, 
        config: Layer2Config,
        privacy_accountant: Optional[PrivacyAccountant] = None
    ):
        """
        Initialize Layer 2 aggregator.
        
        Args:
            config: Layer 2 configuration
            privacy_accountant: Optional DP budget tracker
        """
        self.config = config
        self.privacy_accountant = privacy_accountant
        
        # Secure aggregator (masking mode only)
        self.secure_agg: Optional[SecureAggregator]
        if config.secure_aggregation_mode == "crypto":
            self.secure_agg = None
        else:
            self.secure_agg = SecureAggregator(
                min_participants=config.min_basket_support
            )

        self.privacy_guard = PrivacyGuard(
            PrivacyConfig(
                secure_aggregation=config.secure_aggregation,
                dp_noise_sigma=0.0,
                seed_length=config.sa_seed_length,
            )
        )

        # Crypto secure aggregation (pairwise masks)
        self._identity_keys: Dict[str, IdentityKeyPair] = {}
        self._crypto_clients: Dict[str, SecureAggClient] = {}
        
        # Central DP (initialized with placeholder sensitivity)
        self.dp_mechanism = CentralDPMechanism(
            epsilon=config.dp_epsilon,
            delta=config.dp_delta,
            sensitivity=config.basket_clip_norm,  # Will update based on basket count
            accountant=privacy_accountant,
        )
        
        # Global model
        self._global_model: Optional[np.ndarray] = None
        self._round_id: int = 0
        
        # Edge support tracking for minimum support filtering
        self._edge_basket_counts: Dict[Tuple[str, str], Set[str]] = {}
    
    def aggregate_global(
        self,
        basket_updates: Dict[str, np.ndarray],
        expected_participants: Optional[Sequence[str]] = None,
    ) -> Optional[np.ndarray]:
        """
        Aggregate basket updates into global model.
        
        Args:
            basket_updates: Dict mapping basket_id to aggregate vector
            
        Returns:
            DP-noised global aggregate, or None if failed
        """
        if len(basket_updates) < self.config.min_basket_support:
            return None
        
        # Update sensitivity based on number of baskets
        total_sensitivity = self.config.basket_clip_norm * len(basket_updates)
        self.dp_mechanism.update_sensitivity(total_sensitivity)
        
        if self.config.secure_aggregation_mode == "crypto":
            aggregate_sum = self._aggregate_crypto(basket_updates, expected_participants)
            if aggregate_sum is None:
                return None
        else:
            aggregate_sum = self._aggregate_masked(basket_updates)
            if aggregate_sum is None:
                return None

        # Average
        aggregate = aggregate_sum / len(basket_updates)

        # Apply central DP
        noised = self.dp_mechanism.add_noise(aggregate)
        if noised is None:
            return None

        # Update global model
        self._global_model = noised
        self._round_id += 1

        return noised

    def _aggregate_masked(self, basket_updates: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Masking-based in-process secure aggregation."""
        if self.secure_agg is None:
            self.secure_agg = SecureAggregator(min_participants=self.config.min_basket_support)

        # Apply bounded influence to each basket (optionally mask)
        clipped_updates = {}
        seeds: List[bytes] = []
        for basket_id, update in basket_updates.items():
            clipped = self.apply_bounded_influence(update)
            if self.config.secure_aggregation:
                masked, seed = self.privacy_guard.secure_mask(clipped)
                clipped_updates[basket_id] = masked
                if seed:
                    seeds.append(seed)
                # Submit masked share
                self.secure_agg.submit_share(masked)
            else:
                clipped_updates[basket_id] = clipped
                # Submit raw share
                self.secure_agg.submit_share(clipped)
        
        # Secure aggregation (sum only)
        aggregate_sum = self.secure_agg.aggregate()
        if aggregate_sum is None:
            return None

        if self.config.secure_aggregation and seeds:
            aggregate_sum = self.privacy_guard.unmask(aggregate_sum, seeds, len(basket_updates))

        return aggregate_sum

    def _aggregate_crypto(
        self,
        basket_updates: Dict[str, np.ndarray],
        expected_participants: Optional[Sequence[str]] = None,
    ) -> Optional[np.ndarray]:
        """Cryptographic pairwise-mask secure aggregation."""
        participants = list(expected_participants) if expected_participants else list(basket_updates.keys())
        dropped = [pid for pid in participants if pid not in basket_updates]

        # Ensure identities and clients for all participants (including dropouts)
        for pid in participants:
            self._ensure_crypto_client(pid)

        identity_registry = {pid: kp.public_bytes() for pid, kp in self._identity_keys.items()}
        coordinator = SecureAggCoordinator(identity_registry)

        round_id = f"round-{self._round_id}"
        records = []
        for pid in participants:
            record = self._crypto_clients[pid].start_round(round_id)
            coordinator.verify_record(record, round_id)
            records.append(record)

        masked_updates: Dict[str, np.ndarray] = {}
        for pid, update in basket_updates.items():
            clipped = self.apply_bounded_influence(update)
            masked = self._crypto_clients[pid].build_masked_update(
                clipped, round_id, records, identity_registry
            )
            masked_updates[pid] = masked

        if not masked_updates:
            return None

        aggregate_sum = np.zeros_like(next(iter(masked_updates.values())), dtype=np.float32)
        for masked in masked_updates.values():
            aggregate_sum += masked

        if dropped:
            reveal_map: Dict[str, Dict[str, str]] = {}
            for pid in masked_updates.keys():
                reveal_map[pid] = self._crypto_clients[pid].reveal_mask_seeds(dropped)
            aggregate_sum = coordinator.unmask_for_dropouts(
                aggregate_sum, round_id, dropped, reveal_map
            )

        return aggregate_sum

    def register_identity(self, peer_id: str, keypair: IdentityKeyPair) -> None:
        """Register a long-term identity keypair for secure aggregation."""
        self._identity_keys[peer_id] = keypair
        self._crypto_clients[peer_id] = SecureAggClient(peer_id, keypair)

    def _ensure_crypto_client(self, peer_id: str) -> None:
        if peer_id in self._crypto_clients:
            return
        if self.config.auto_generate_identities:
            keypair = IdentityKeyPair.generate()
            self.register_identity(peer_id, keypair)
            logger.warning("Auto-generated identity key for peer %s", peer_id)
            return
        raise RuntimeError(f"Missing identity key for peer {peer_id}")
    
    def apply_bounded_influence(self, update: np.ndarray) -> np.ndarray:
        """
        Clip basket update to bounded influence.
        
        Args:
            update: Basket aggregate
            
        Returns:
            Clipped update
        """
        norm = np.linalg.norm(update)
        if norm > self.config.basket_clip_norm:
            return update * (self.config.basket_clip_norm / norm)
        return update
    
    def record_edge(self, edge: Tuple[str, str], basket_id: str) -> None:
        """
        Record that an edge was seen in a basket.
        
        Args:
            edge: (source, target) tuple
            basket_id: Basket that contributed the edge
        """
        if edge not in self._edge_basket_counts:
            self._edge_basket_counts[edge] = set()
        self._edge_basket_counts[edge].add(basket_id)
    
    def check_minimum_support(self) -> List[Tuple[str, str]]:
        """
        Return edges meeting minimum basket support threshold.
        
        Returns:
            List of (source, target) edges with sufficient support
        """
        return [
            edge for edge, baskets in self._edge_basket_counts.items()
            if len(baskets) >= self.config.min_basket_support
        ]
    
    def get_global_model(self) -> Optional[np.ndarray]:
        """Get current global model."""
        return self._global_model
    
    @property
    def round_id(self) -> int:
        """Get current round ID."""
        return self._round_id


class GlobalMetaModel:
    """
    Shared meta-parameters learned across baskets.
    
    Extracts parameters that generalize across domains:
    - Learning rates
    - Forgetting factors
    - Lag defaults
    - Threshold values
    """
    
    def __init__(self):
        """Initialize the global meta-model."""
        self._meta_params: Dict[str, float] = {}
        self._param_history: Dict[str, List[float]] = {}
        self._update_count: int = 0
    
    def extract_shared(
        self, 
        basket_models: List[BasketModel]
    ) -> Dict[str, float]:
        """
        Extract meta-parameters that generalize across baskets.
        
        Args:
            basket_models: List of per-basket models
            
        Returns:
            Dictionary of shared meta-parameters
        """
        # Collect all parameter names
        all_params: Dict[str, List[float]] = {}
        
        for model in basket_models:
            for name, value in model.hypothesis_params.items():
                if name not in all_params:
                    all_params[name] = []
                all_params[name].append(value)
        
        # Compute robust estimates (median)
        shared = {}
        for name, values in all_params.items():
            if len(values) >= 2:  # Need multiple baskets
                shared[name] = float(np.median(values))
        
        return shared
    
    def update(
        self, 
        basket_contributions: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Update shared parameters using robust merge.
        
        Args:
            basket_contributions: Dict mapping basket_id to param dicts
        """
        # Collect all parameters
        param_values: Dict[str, List[float]] = {}
        
        for basket_id, params in basket_contributions.items():
            for name, value in params.items():
                if name not in param_values:
                    param_values[name] = []
                param_values[name].append(value)
        
        # Update meta params with median
        for name, values in param_values.items():
            if values:
                self._meta_params[name] = float(np.median(values))
                
                # Track history
                if name not in self._param_history:
                    self._param_history[name] = []
                self._param_history[name].append(self._meta_params[name])
        
        self._update_count += 1
    
    def get_meta_params(self) -> Dict[str, float]:
        """Get current meta-parameters."""
        return dict(self._meta_params)
    
    def get_param(self, name: str, default: float = 0.0) -> float:
        """Get a specific meta-parameter."""
        return self._meta_params.get(name, default)
