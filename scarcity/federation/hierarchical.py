"""
Hierarchical Federation Orchestrator.

This module provides the main entry point for hierarchical federated learning,
integrating all components:
- Domain Baskets: Client grouping by domain
- Gossip Protocol: Decentralized intra-basket communication with local DP
- Memory Buffers: Staleness-aware storage with triggers
- Two-Layer Aggregation: Layer 1 (basket) + Layer 2 (global with secure agg + DP)

Privacy Model:
- Untrusted gossip: All messages are clipped and locally DP-noised
- Secure aggregation: Server only sees sums, not individual updates
- Central DP: Gaussian noise added to global aggregate
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from .basket import BasketManager, BasketConfig, BasketStatus
from .gossip import GossipProtocol, GossipConfig, GossipMessage
from .buffer import (
    UpdateBuffer, 
    BufferConfig, 
    BufferedUpdate, 
    TriggerEngine, 
    PrivacyAccountant
)
from .layers import (
    Layer1Aggregator, 
    Layer1Config,
    Layer2Aggregator, 
    Layer2Config,
    BasketModel,
    GlobalMetaModel
)
from .secure_aggregation import IdentityKeyPair


@dataclass
class HierarchicalFederationConfig:
    """Configuration for hierarchical federation."""
    # Component configs
    basket: BasketConfig = field(default_factory=BasketConfig)
    gossip: GossipConfig = field(default_factory=GossipConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    layer1: Layer1Config = field(default_factory=Layer1Config)
    layer2: Layer2Config = field(default_factory=Layer2Config)
    
    # Global privacy budget
    total_epsilon: float = 10.0           # Total ε budget
    total_delta: float = 1e-4             # Total δ budget
    
    # Operational settings
    auto_aggregate: bool = True           # Auto-run aggregation when triggers fire
    vector_dim: int = 64                  # Default vector dimension


class HierarchicalFederation:
    """
    Main entry point for hierarchical federated learning.
    
    Integrates:
    - BasketManager: Domain grouping
    - GossipProtocol: Intra-basket communication with local DP
    - UpdateBuffer: Staleness-aware storage
    - Layer1/Layer2Aggregator: Hierarchical aggregation with secure agg + DP
    - PrivacyAccountant: DP budget tracking
    
    Usage:
        fed = HierarchicalFederation(config)
        
        # Register clients
        basket_id = fed.register_client("client_1", "healthcare")
        
        # Submit updates
        fed.submit_update("client_1", update_vector)
        
        # Run gossip
        fed.run_gossip_round()
        
        # Check for aggregation
        global_model = fed.maybe_aggregate()
    """
    
    def __init__(self, config: Optional[HierarchicalFederationConfig] = None):
        """
        Initialize hierarchical federation.
        
        Args:
            config: Configuration object
        """
        self.config = config or HierarchicalFederationConfig()
        
        # Initialize privacy accountant
        self.privacy_accountant = PrivacyAccountant(
            total_epsilon=self.config.total_epsilon,
            total_delta=self.config.total_delta,
        )
        
        # Initialize components
        self.basket_manager = BasketManager(self.config.basket)
        
        self.buffer = UpdateBuffer(self.config.buffer)
        
        self.gossip = GossipProtocol(
            self.config.gossip, 
            self.basket_manager
        )
        
        self.trigger_engine = TriggerEngine(
            self.config.buffer,
            self.privacy_accountant,
        )
        
        self.layer1 = Layer1Aggregator(
            self.config.layer1,
            self.buffer,
        )
        
        self.layer2 = Layer2Aggregator(
            self.config.layer2,
            self.privacy_accountant,
        )
        
        self.meta_model = GlobalMetaModel()
        
        # State tracking
        self._round_id: int = 0
        self._global_model: Optional[np.ndarray] = None
    
    def register_client(
        self, 
        client_id: str, 
        domain_id: str,
        features: Optional[np.ndarray] = None,
        identity_keypair: Optional[IdentityKeyPair] = None,
    ) -> str:
        """
        Register a client with a domain.
        
        Args:
            client_id: Unique client identifier
            domain_id: Domain identifier (e.g., 'healthcare')
            features: Optional features for fingerprinting
            
        Returns:
            The basket_id the client was assigned to
        """
        basket_id = self.basket_manager.register_client(client_id, domain_id, features)
        if identity_keypair is not None:
            self.layer2.register_identity(basket_id, identity_keypair)
        return basket_id

    def register_basket_identity(self, basket_id: str, identity_keypair: IdentityKeyPair) -> None:
        """Register a basket-level identity keypair for secure aggregation."""
        self.layer2.register_identity(basket_id, identity_keypair)
    
    def unregister_client(self, client_id: str) -> bool:
        """
        Remove a client from the federation.
        
        Args:
            client_id: Client to remove
            
        Returns:
            True if client was found and removed
        """
        return self.basket_manager.unregister_client(client_id)
    
    def submit_update(
        self, 
        client_id: str, 
        update: np.ndarray,
        round_id: Optional[int] = None
    ) -> bool:
        """
        Submit a client update to the federation.
        
        The update is:
        1. Validated for replay/participation cap
        2. Stored in the buffer
        3. Optionally triggers Layer 1 aggregation
        
        Args:
            client_id: Client submitting the update
            update: Update vector (already protected by local DP if from gossip)
            round_id: Optional round ID
            
        Returns:
            True if update was accepted
        """
        client_info = self.basket_manager.get_client(client_id)
        if client_info is None:
            return False
        
        # Create buffered update
        buffered = BufferedUpdate(
            client_id=client_id,
            basket_id=client_info.basket_id,
            vector=update.astype(np.float32),
            sequence_number=client_info.participation_count,
            round_id=round_id or self._round_id,
        )
        
        # Add to buffer
        if not self.buffer.add(buffered):
            return False
        
        # Record participation
        self.basket_manager.record_participation(client_id)
        
        # Update trigger counts
        self.trigger_engine.increment_count(client_info.basket_id)
        
        # Auto-aggregate if enabled and triggered
        if self.config.auto_aggregate:
            self._check_and_aggregate_basket(client_info.basket_id)
        
        return True
    
    def run_gossip_round(self) -> Dict[str, int]:
        """
        Run one gossip round across all active baskets.
        
        Returns:
            Dict mapping basket_id to number of messages exchanged
        """
        message_counts: Dict[str, int] = {}
        
        for basket_id in self.basket_manager.get_active_baskets():
            count = self._run_basket_gossip(basket_id)
            message_counts[basket_id] = count
        
        # Advance gossip round
        self.gossip.advance_round()
        self.trigger_engine.advance_round()
        
        return message_counts
    
    def _run_basket_gossip(self, basket_id: str) -> int:
        """
        Run gossip for a single basket.
        
        Args:
            basket_id: Basket to gossip in
            
        Returns:
            Number of messages exchanged
        """
        message_count = 0
        peers = self.basket_manager.get_basket_peers(basket_id)
        
        for client_id in peers:
            # Pull round: get peers to request from
            pull_peers = self.gossip.pull_round(client_id)
            
            # Get messages from inbox for this basket
            messages = self.gossip.get_inbox_messages(basket_id, clear=False)
            
            if messages:
                # Merge received messages
                merged = self.gossip.merge_messages(messages)
                
                # Submit as update (if valid)
                if merged.size > 0:
                    self.submit_update(client_id, merged)
                    message_count += len(messages)
        
        return message_count
    
    def _check_and_aggregate_basket(self, basket_id: str) -> Optional[np.ndarray]:
        """
        Check trigger and aggregate basket if needed.
        
        Args:
            basket_id: Basket to check/aggregate
            
        Returns:
            Basket aggregate if triggered, None otherwise
        """
        count = self.buffer.get_basket_update_count(basket_id)
        
        if self.trigger_engine.check_layer1(basket_id, sample_count=count):
            aggregate = self.layer1.aggregate_basket(basket_id)
            self.trigger_engine.mark_triggered(1, basket_id)
            return aggregate
        
        return None
    
    def maybe_aggregate(self) -> Optional[np.ndarray]:
        """
        Check triggers and run aggregation if conditions met.
        
        Automatically runs Layer 1 for any triggered baskets,
        then Layer 2 if global trigger conditions are met.
        
        Returns:
            Global model if Layer 2 ran, None otherwise
        """
        # Run Layer 1 for all triggered baskets
        for basket_id in self.basket_manager.get_active_baskets():
            self._check_and_aggregate_basket(basket_id)
        
        # Check Layer 2 trigger
        active_baskets = self.basket_manager.get_active_baskets()
        ready_baskets = self.layer1.get_ready_baskets()
        
        if not self.trigger_engine.check_layer2(
            total_baskets=len(active_baskets),
            contributed_baskets=len(ready_baskets)
        ):
            return None
        
        # Collect basket aggregates
        basket_updates: Dict[str, np.ndarray] = {}
        for basket_id in ready_baskets:
            model = self.layer1.get_basket_model(basket_id)
            if model is not None:
                basket_updates[basket_id] = model.aggregate_vector
        
        if not basket_updates:
            return None
        
        # Run Layer 2 aggregation
        global_model = self.layer2.aggregate_global(basket_updates)
        
        if global_model is not None:
            self._global_model = global_model
            self.trigger_engine.mark_triggered(2)
            
            # Update meta-model
            basket_models = list(self.layer1.get_basket_models().values())
            meta_params = self.meta_model.extract_shared(basket_models)
            self.meta_model.update({
                bm.basket_id: bm.hypothesis_params 
                for bm in basket_models
            })
        
        return global_model
    
    def force_aggregate(self) -> Optional[np.ndarray]:
        """
        Force aggregation regardless of triggers.
        
        Returns:
            Global model, or None if not enough data
        """
        # Force Layer 1 for all baskets with data
        for basket_id in self.buffer.get_buffered_baskets():
            self.layer1.aggregate_basket(basket_id)
            self.trigger_engine.mark_triggered(1, basket_id)
        
        # Force Layer 2
        basket_models = self.layer1.get_basket_models()
        basket_updates = {
            bid: model.aggregate_vector 
            for bid, model in basket_models.items()
        }
        
        if not basket_updates:
            return None
        
        global_model = self.layer2.aggregate_global(basket_updates)
        if global_model is not None:
            self._global_model = global_model
            self.trigger_engine.mark_triggered(2)
        
        return global_model
    
    def get_global_model(self) -> Optional[np.ndarray]:
        """Get current global model."""
        return self._global_model
    
    def get_basket_model(self, basket_id: str) -> Optional[np.ndarray]:
        """Get basket-specific model."""
        model = self.layer1.get_basket_model(basket_id)
        return model.aggregate_vector if model else None
    
    def get_meta_params(self) -> Dict[str, float]:
        """Get shared meta-parameters."""
        return self.meta_model.get_meta_params()
    
    def get_privacy_budget(self) -> Tuple[float, float]:
        """
        Get remaining privacy budget.
        
        Returns:
            Tuple of (remaining_epsilon, remaining_delta)
        """
        return self.privacy_accountant.remaining()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get federation statistics.
        
        Returns:
            Dictionary of statistics
        """
        eps_remaining, delta_remaining = self.privacy_accountant.remaining()
        
        return {
            "round_id": self._round_id,
            "total_clients": len(self.basket_manager.get_all_clients()),
            "total_baskets": len(self.basket_manager.get_all_baskets()),
            "active_baskets": len(self.basket_manager.get_active_baskets()),
            "layer1_round": self.layer1._round_id,
            "layer2_round": self.layer2.round_id,
            "gossip_round": self.gossip.current_round,
            "epsilon_remaining": eps_remaining,
            "delta_remaining": delta_remaining,
            "epsilon_spent": self.privacy_accountant.spent_epsilon,
            "releases_made": self.privacy_accountant.release_count,
        }
    
    def advance_round(self) -> None:
        """Advance to next round."""
        self._round_id += 1
        self.gossip.advance_round()
        self.trigger_engine.advance_round()
        self.layer1.advance_round()
        self.buffer.prune_stale()
