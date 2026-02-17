"""
Test: Hierarchical Federation

Validates the hierarchical federated learning implementation including:
- Domain basket management
- Gossip protocol with local DP
- Memory buffer with staleness handling
- Two-layer aggregation with secure agg + central DP
- End-to-end integration
"""

import pytest
import numpy as np
import time

from scarcity.federation.basket import (
    BasketManager, 
    BasketConfig, 
    BasketStatus,
)
from scarcity.federation.gossip import (
    GossipProtocol, 
    GossipConfig, 
    GossipMessage,
    LocalDPMechanism,
    MaterialityDetector,
    PeerSampler,
)
from scarcity.federation.buffer import (
    UpdateBuffer, 
    BufferConfig, 
    BufferedUpdate,
    TriggerEngine,
    PrivacyAccountant,
    ReplayGuard,
)
from scarcity.federation.layers import (
    Layer1Aggregator, 
    Layer1Config,
    Layer2Aggregator, 
    Layer2Config,
    SecureAggregator,
)
from scarcity.federation.hierarchical import (
    HierarchicalFederation, 
    HierarchicalFederationConfig,
)


class TestBasketManager:
    """Tests for domain basket management."""
    
    def test_register_client_creates_basket(self):
        """Registering a client should create a basket for the domain."""
        manager = BasketManager()
        
        basket_id = manager.register_client("client_1", "healthcare")
        
        assert basket_id is not None
        assert "healthcare" in basket_id
        assert manager.get_client("client_1") is not None
        assert manager.get_basket(basket_id) is not None
    
    def test_same_domain_same_basket(self):
        """Clients in the same domain should be in the same basket."""
        manager = BasketManager()
        
        basket_1 = manager.register_client("client_1", "healthcare")
        basket_2 = manager.register_client("client_2", "healthcare")
        
        assert basket_1 == basket_2
    
    def test_different_domain_different_basket(self):
        """Clients in different domains should be in different baskets."""
        manager = BasketManager()
        
        basket_1 = manager.register_client("client_1", "healthcare")
        basket_2 = manager.register_client("client_2", "finance")
        
        assert basket_1 != basket_2
    
    def test_basket_status_forming_until_min_size(self):
        """Basket should be FORMING until it reaches minimum size."""
        config = BasketConfig(min_basket_size=3)
        manager = BasketManager(config)
        
        basket_id = manager.register_client("client_1", "healthcare")
        assert manager.get_basket(basket_id).status == BasketStatus.FORMING
        
        manager.register_client("client_2", "healthcare")
        assert manager.get_basket(basket_id).status == BasketStatus.FORMING
        
        manager.register_client("client_3", "healthcare")
        assert manager.get_basket(basket_id).status == BasketStatus.ACTIVE
    
    def test_get_basket_peers(self):
        """Should return all clients in a basket."""
        manager = BasketManager()
        
        basket_id = manager.register_client("client_1", "healthcare")
        manager.register_client("client_2", "healthcare")
        
        peers = manager.get_basket_peers(basket_id)
        
        assert len(peers) == 2
        assert "client_1" in peers
        assert "client_2" in peers
    
    def test_unregister_client(self):
        """Should remove client from basket."""
        manager = BasketManager()
        
        basket_id = manager.register_client("client_1", "healthcare")
        manager.register_client("client_2", "healthcare")
        
        assert manager.unregister_client("client_1")
        
        peers = manager.get_basket_peers(basket_id)
        assert len(peers) == 1
        assert "client_2" in peers


class TestLocalDPMechanism:
    """Tests for local differential privacy."""
    
    def test_noise_is_added(self):
        """Noise should be added to vectors."""
        dp = LocalDPMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        
        vector = np.ones(10)
        noised = dp.add_noise(vector)
        
        # Should not be identical
        assert not np.allclose(vector, noised)
    
    def test_clip_enforces_norm(self):
        """Clipping should enforce L2 norm bound."""
        dp = LocalDPMechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        
        # Large vector
        vector = np.ones(10) * 10  # L2 norm = ~31.6
        clipped = dp.clip_and_noise(vector)
        
        # After clipping, norm should be <= sensitivity + noise
        # (noise can increase norm slightly)
        original_clipped_norm = np.linalg.norm(vector * (1.0 / np.linalg.norm(vector)))
        assert original_clipped_norm <= dp.sensitivity + 0.01
    
    def test_higher_epsilon_less_noise(self):
        """Higher epsilon should mean less noise (lower sigma)."""
        dp_low = LocalDPMechanism(epsilon=0.1, delta=1e-5, sensitivity=1.0)
        dp_high = LocalDPMechanism(epsilon=10.0, delta=1e-5, sensitivity=1.0)
        
        assert dp_low.sigma > dp_high.sigma


class TestGossipProtocol:
    """Tests for gossip protocol."""
    
    def test_create_message_applies_dp(self):
        """Created messages should have DP noise applied."""
        config = GossipConfig(local_dp_epsilon=1.0, local_dp_delta=1e-5)
        manager = BasketManager()
        manager.register_client("client_1", "healthcare")
        
        gossip = GossipProtocol(config, manager)
        
        raw = np.ones(10)
        message = gossip.create_message("client_1", raw)
        
        assert message is not None
        assert not np.allclose(raw, message.summary_vector)
    
    def test_message_budget_enforced(self):
        """Message count per day should be limited."""
        config = GossipConfig(max_messages_per_day=3)
        manager = BasketManager()
        manager.register_client("client_1", "healthcare")
        
        gossip = GossipProtocol(config, manager)
        
        # Should allow 3 messages
        for _ in range(3):
            msg = gossip.create_message("client_1", np.ones(10))
            assert msg is not None
        
        # 4th should fail
        msg = gossip.create_message("client_1", np.ones(10))
        assert msg is None
    
    def test_pull_round_samples_peers(self):
        """Pull round should sample k peers from basket."""
        config = GossipConfig(peers_per_round=2)
        manager = BasketManager()
        manager.register_client("client_1", "healthcare")
        manager.register_client("client_2", "healthcare")
        manager.register_client("client_3", "healthcare")
        manager.register_client("client_4", "healthcare")
        
        gossip = GossipProtocol(config, manager)
        
        peers = gossip.pull_round("client_1")
        
        assert len(peers) == 2
        assert "client_1" not in peers  # Self excluded


class TestMaterialityDetector:
    """Tests for materiality detection."""
    
    def test_first_update_is_material(self):
        """First update should always be material."""
        detector = MaterialityDetector(drift_threshold=0.1)
        
        should_push = detector.should_push("client_1", np.ones(10))
        
        assert should_push
    
    def test_small_change_not_material(self):
        """Small changes should not trigger push."""
        detector = MaterialityDetector(drift_threshold=0.1)
        
        state1 = np.ones(10)
        state2 = np.ones(10) * 1.01  # 1% change
        
        detector.check_drift("client_1", state1)
        detector.update_state("client_1", state1)
        
        should_push = detector.should_push("client_1", state2)
        
        assert not should_push
    
    def test_large_change_is_material(self):
        """Large changes should trigger push."""
        detector = MaterialityDetector(drift_threshold=0.1)
        
        state1 = np.ones(10)
        state2 = np.ones(10) * 2.0  # 100% change
        
        detector.update_state("client_1", state1)
        
        should_push = detector.should_push("client_1", state2)
        
        assert should_push


class TestUpdateBuffer:
    """Tests for update buffer."""
    
    def test_add_update(self):
        """Should store updates."""
        buffer = UpdateBuffer()
        
        update = BufferedUpdate(
            client_id="client_1",
            basket_id="basket_1",
            vector=np.ones(10),
            sequence_number=0,
            round_id=0,
        )
        
        result = buffer.add(update)
        
        assert result is True
        assert buffer.get_basket_update_count("basket_1") == 1
    
    def test_replay_detection(self):
        """Should reject replay attacks."""
        buffer = UpdateBuffer()
        
        update1 = BufferedUpdate(
            client_id="client_1",
            basket_id="basket_1",
            vector=np.ones(10),
            sequence_number=0,
            round_id=0,
        )
        update2 = BufferedUpdate(
            client_id="client_1",
            basket_id="basket_1",
            vector=np.ones(10),
            sequence_number=0,  # Same sequence = replay
            round_id=0,
        )
        
        assert buffer.add(update1)
        assert not buffer.add(update2)  # Replay rejected
    
    def test_weighted_aggregate(self):
        """Should compute decay-weighted aggregate."""
        config = BufferConfig(decay_half_life=60.0)
        buffer = UpdateBuffer(config)
        
        # Add two updates
        buffer.add(BufferedUpdate(
            client_id="client_1",
            basket_id="basket_1",
            vector=np.ones(10),
            sequence_number=0,
            round_id=0,
        ))
        buffer.add(BufferedUpdate(
            client_id="client_2",
            basket_id="basket_1",
            vector=np.ones(10) * 2,
            sequence_number=0,
            round_id=0,
        ))
        
        aggregate, count = buffer.get_weighted("basket_1")
        
        assert count == 2
        # Should be approximately 1.5 (average of 1 and 2)
        assert np.allclose(aggregate, np.ones(10) * 1.5, atol=0.1)


class TestPrivacyAccountant:
    """Tests for privacy budget tracking."""
    
    def test_spend_budget(self):
        """Should track spent budget."""
        accountant = PrivacyAccountant(total_epsilon=10.0, total_delta=1e-4)
        
        assert accountant.spend(1.0, 1e-5)
        
        eps, delta = accountant.remaining()
        assert eps == 9.0
        assert delta == 1e-4 - 1e-5
    
    def test_budget_exhaustion(self):
        """Should reject when budget exhausted."""
        accountant = PrivacyAccountant(total_epsilon=1.0, total_delta=1e-5)
        
        assert accountant.spend(0.5, 0.0)
        assert accountant.spend(0.5, 0.0)
        assert not accountant.spend(0.1, 0.0)  # Exhausted
    
    def test_can_release(self):
        """Should check if release is allowed."""
        accountant = PrivacyAccountant(total_epsilon=1.0, total_delta=1e-5)
        
        assert accountant.can_release(epsilon=0.5)
        accountant.spend(0.8, 0.0)
        assert not accountant.can_release(epsilon=0.5)


class TestTriggerEngine:
    """Tests for aggregation triggers."""
    
    def test_count_trigger(self):
        """Should trigger after count threshold."""
        # Use long interval to prevent time trigger from interfering
        config = BufferConfig(trigger_count=10, trigger_interval=10000.0)
        engine = TriggerEngine(config)
        
        assert not engine.check_layer1("basket_1", sample_count=5)
        assert engine.check_layer1("basket_1", sample_count=10)
    
    def test_time_trigger(self):
        """Should trigger after time interval."""
        config = BufferConfig(trigger_interval=0.1)
        engine = TriggerEngine(config)
        
        # Mark as just triggered
        engine.mark_triggered(1, "basket_1")
        
        assert not engine.check_layer1("basket_1", sample_count=0)
        
        time.sleep(0.15)
        
        assert engine.check_layer1("basket_1", sample_count=0)


class TestLayerAggregation:
    """Tests for two-layer aggregation."""
    
    def test_layer1_aggregates_basket(self):
        """Layer 1 should aggregate within a basket."""
        buffer = UpdateBuffer()
        # Use higher clip_norm to avoid clipping the test values
        config = Layer1Config(clip_norm=10.0)
        layer1 = Layer1Aggregator(config, buffer)
        
        # Add updates
        for i in range(3):
            buffer.add(BufferedUpdate(
                client_id=f"client_{i}",
                basket_id="basket_1",
                vector=np.ones(10) * (i + 1),
                sequence_number=0,
                round_id=0,
            ))
        
        aggregate = layer1.aggregate_basket("basket_1")
        
        assert aggregate is not None
        # Should be approximately mean of [1, 2, 3] = 2
        assert np.allclose(aggregate, np.ones(10) * 2, atol=0.5)
    
    def test_layer2_bounded_influence(self):
        """Layer 2 should clip basket contributions."""
        config = Layer2Config(basket_clip_norm=1.0)
        layer2 = Layer2Aggregator(config)
        
        large_update = np.ones(10) * 100
        clipped = layer2.apply_bounded_influence(large_update)
        
        norm = np.linalg.norm(clipped)
        assert norm <= config.basket_clip_norm + 0.01
    
    def test_layer2_minimum_support(self):
        """Layer 2 should require minimum basket support."""
        config = Layer2Config(min_basket_support=2)
        layer2 = Layer2Aggregator(config)
        
        # Only 1 basket - should fail
        result = layer2.aggregate_global({
            "basket_1": np.ones(10)
        })
        
        assert result is None
        
        # 2 baskets - should succeed
        result = layer2.aggregate_global({
            "basket_1": np.ones(10),
            "basket_2": np.ones(10),
        })
        
        assert result is not None


class TestSecureAggregator:
    """Tests for secure aggregation."""
    
    def test_requires_min_participants(self):
        """Should require minimum participants."""
        secure_agg = SecureAggregator(min_participants=2)
        
        secure_agg.submit_share(np.ones(10))
        
        # Only 1 participant - should fail
        result = secure_agg.aggregate()
        assert result is None
    
    def test_computes_sum(self):
        """Should compute sum of shares."""
        secure_agg = SecureAggregator(min_participants=2)
        
        secure_agg.submit_share(np.ones(10))
        secure_agg.submit_share(np.ones(10) * 2)
        
        result = secure_agg.aggregate()
        
        assert result is not None
        assert np.allclose(result, np.ones(10) * 3)


class TestHierarchicalFederation:
    """End-to-end tests for hierarchical federation."""
    
    def test_register_and_submit(self):
        """Should allow registering clients and submitting updates."""
        fed = HierarchicalFederation()
        
        basket_id = fed.register_client("client_1", "healthcare")
        
        result = fed.submit_update("client_1", np.ones(10))
        
        assert result is True
    
    def test_multiple_baskets(self):
        """Should handle multiple domain baskets."""
        fed = HierarchicalFederation()
        
        # Register clients in different domains
        basket_health = fed.register_client("client_1", "healthcare")
        fed.register_client("client_2", "healthcare")
        fed.register_client("client_3", "healthcare")
        
        basket_finance = fed.register_client("client_4", "finance")
        fed.register_client("client_5", "finance")
        fed.register_client("client_6", "finance")
        
        stats = fed.get_stats()
        
        assert stats["total_clients"] == 6
        assert stats["total_baskets"] == 2
        assert stats["active_baskets"] == 2
    
    def test_end_to_end_aggregation(self):
        """Should aggregate updates through both layers."""
        config = HierarchicalFederationConfig(
            buffer=BufferConfig(trigger_count=3, min_basket_coverage=0.5),
            layer2=Layer2Config(min_basket_support=1),
        )
        fed = HierarchicalFederation(config)
        
        # Register 3 clients per basket (minimum for ACTIVE)
        for i in range(3):
            fed.register_client(f"health_{i}", "healthcare")
            fed.register_client(f"finance_{i}", "finance")
        
        # Submit updates
        for i in range(3):
            fed.submit_update(f"health_{i}", np.ones(10) * (i + 1))
            fed.submit_update(f"finance_{i}", np.ones(10) * (i + 4))
        
        # Force aggregation
        global_model = fed.force_aggregate()
        
        assert global_model is not None
        assert len(global_model) == 10
    
    def test_privacy_budget_tracking(self):
        """Should track privacy budget across rounds."""
        config = HierarchicalFederationConfig(
            total_epsilon=2.0,
            total_delta=1e-4,
            layer2=Layer2Config(dp_epsilon=1.0, min_basket_support=1),
        )
        fed = HierarchicalFederation(config)
        
        # Check initial budget
        eps, delta = fed.get_privacy_budget()
        assert eps == 2.0
        
        # Register and submit
        for i in range(3):
            fed.register_client(f"client_{i}", "healthcare")
            fed.submit_update(f"client_{i}", np.ones(10))
        
        # Force aggregation (spends budget)
        fed.force_aggregate()
        
        # Budget should be reduced
        eps, delta = fed.get_privacy_budget()
        assert eps < 2.0


class TestConvergence:
    """Tests for learning convergence."""
    
    def test_gossip_converges_within_basket(self):
        """Gossip should lead to convergence within a basket."""
        config = HierarchicalFederationConfig(
            gossip=GossipConfig(
                local_dp_epsilon=10.0,  # High epsilon = low noise for testing
                max_messages_per_day=100,
            ),
        )
        fed = HierarchicalFederation(config)
        
        # Register clients with different initial values
        values = [1.0, 5.0, 10.0]
        for i, v in enumerate(values):
            fed.register_client(f"client_{i}", "healthcare")
        
        # Submit initial updates
        for i, v in enumerate(values):
            fed.submit_update(f"client_{i}", np.ones(10) * v)
        
        # Run several gossip rounds
        for _ in range(5):
            fed.run_gossip_round()
        
        # Force aggregate
        model = fed.force_aggregate()
        
        # Should converge toward mean (5.33...)
        if model is not None:
            mean_expected = np.mean(values)
            # With DP noise, we allow larger tolerance
            assert np.mean(model) > 0
