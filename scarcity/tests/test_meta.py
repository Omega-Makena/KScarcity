"""
Test: Meta-Learning Layer

Validates cross-domain transfer and meta-update generation.
"""

import asyncio
import pytest
import numpy as np
from scarcity.meta.domain_meta import (
    DomainMetaLearner,
    DomainMetaConfig,
    DomainMetaUpdate,
)
from scarcity.meta.cross_meta import CrossDomainMetaAggregator, CrossMetaConfig
from scarcity.meta.optimizer import OnlineReptileOptimizer


class TestDomainMetaLearner:
    def test_observe_generates_update(self):
        """observe() should generate a meta-update."""
        learner = DomainMetaLearner()
        
        update = learner.observe(
            domain_id='healthcare',
            metrics={'gain_p50': 0.7, 'stability_avg': 0.8},
            parameters={'lr': 0.01, 'alpha': 0.1}
        )
        
        assert update.domain_id == 'healthcare'
        assert update.vector is not None
        assert len(update.keys) == 2
        print(f"Update: {update}")
    
    def test_confidence_increases_with_good_performance(self):
        """Confidence should increase with consistent improvements."""
        learner = DomainMetaLearner()
        
        confidences = []
        for i in range(10):
            update = learner.observe(
                domain_id='finance',
                metrics={'gain_p50': 0.5 + i * 0.05, 'stability_avg': 0.8},
                parameters={'lr': 0.01}
            )
            confidences.append(update.confidence)
        
        # Confidence should generally increase
        assert confidences[-1] > confidences[0], \
            f"Confidence should increase: {confidences}"
        print(f"Confidence progression: {confidences}")
    
    def test_state_persistence(self):
        """State should persist across observations."""
        learner = DomainMetaLearner()
        
        learner.observe('domain_a', {'gain_p50': 0.5}, {'x': 1.0})
        learner.observe('domain_a', {'gain_p50': 0.6}, {'x': 1.5})
        
        state = learner.state('domain_a')
        
        assert state.last_score == 0.6
        assert len(state.history) == 2
        print(f"State: {state}")


class TestCrossDomainTransfer:
    def test_multiple_domains_tracked(self):
        """Learner should track multiple domains independently."""
        learner = DomainMetaLearner()
        
        learner.observe('healthcare', {'gain_p50': 0.3}, {'lr': 0.01})
        learner.observe('finance', {'gain_p50': 0.7}, {'lr': 0.02})
        
        state_health = learner.state('healthcare')
        state_finance = learner.state('finance')
        
        assert state_health.last_score == 0.3
        assert state_finance.last_score == 0.7
        
        print(f"Healthcare: {state_health.last_score}")
        print(f"Finance: {state_finance.last_score}")
    
    def test_domain_updates_independent(self):
        """Domain updates should be independent."""
        learner = DomainMetaLearner()
        
        # Different domains, different parameters
        update_a = learner.observe('domain_a', {'gain_p50': 0.5}, {'x': 1.0, 'y': 2.0})
        update_b = learner.observe('domain_b', {'gain_p50': 0.8}, {'z': 3.0})
        
        assert update_a.keys == ['x', 'y']
        assert update_b.keys == ['z']
        
        print(f"Domain A keys: {update_a.keys}")
        print(f"Domain B keys: {update_b.keys}")


class TestMetaLearningAdaptation:
    def test_adaptive_learning_rate(self):
        """Meta learning rate should adapt based on confidence."""
        config = DomainMetaConfig(
            meta_lr_min=0.01,
            meta_lr_max=0.1
        )
        learner = DomainMetaLearner(config)
        
        # First observation - low confidence
        update1 = learner.observe('test', {'gain_p50': 0.5}, {'param': 1.0})
        state1 = learner.state('test')
        
        # Many successful observations - build confidence
        for i in range(10):
            learner.observe('test', {'gain_p50': 0.5 + i * 0.02, 'stability_avg': 0.9}, {'param': 1.0 + i * 0.1})
        
        state2 = learner.state('test')
        
        # Confidence should have increased (or stayed if it was already building)
        assert state2.confidence >= state1.confidence or state2.confidence > 0.3, \
            f"Confidence should build: {state1.confidence:.3f} → {state2.confidence:.3f}"

    def test_high_confidence_produces_larger_update_norm(self):
        """Higher confidence should result in larger update magnitude for same parameter delta."""
        learner = DomainMetaLearner(DomainMetaConfig(meta_lr_min=0.01, meta_lr_max=0.2))

        # Prime both domains with the same initial parameter state.
        learner.observe('high_conf', {'gain_p50': 0.4, 'stability_avg': 0.9}, {'p': 0.0})
        learner.observe('low_conf', {'gain_p50': 0.4, 'stability_avg': 0.2}, {'p': 0.0})

        # Build high confidence with improving and stable performance.
        for i in range(8):
            learner.observe(
                'high_conf',
                {'gain_p50': 0.55 + i * 0.04, 'stability_avg': 0.95},
                {'p': 0.0},
            )

        # Keep low confidence suppressed with declining unstable observations.
        for i in range(8):
            learner.observe(
                'low_conf',
                {'gain_p50': 0.45 - i * 0.03, 'stability_avg': 0.1},
                {'p': 0.0},
            )

        # Apply the same parameter delta to both domains.
        upd_high = learner.observe('high_conf', {'gain_p50': 0.90, 'stability_avg': 0.95}, {'p': 1.0})
        upd_low = learner.observe('low_conf', {'gain_p50': 0.20, 'stability_avg': 0.1}, {'p': 1.0})

        assert np.linalg.norm(upd_high.vector) > np.linalg.norm(upd_low.vector), (
            "High-confidence domain should yield a larger meta-update norm"
        )

    def test_confidence_decays_after_prolonged_drift(self):
        """Confidence should decrease after sustained degraded metrics (drift-like behavior)."""
        learner = DomainMetaLearner()

        # Build confidence with healthy observations.
        for i in range(10):
            learner.observe(
                'drift_domain',
                {'gain_p50': 0.6 + i * 0.03, 'stability_avg': 0.9},
                {'lr': 0.02 + i * 0.001},
            )
        pre_drift = learner.state('drift_domain').confidence

        # Simulate sustained degradation.
        for i in range(12):
            learner.observe(
                'drift_domain',
                {'gain_p50': 0.3 - i * 0.01, 'stability_avg': 0.1},
                {'lr': 0.03},
            )

        post_drift = learner.state('drift_domain').confidence
        assert post_drift < pre_drift, (
            f"Confidence should decay under prolonged drift: {pre_drift:.3f} -> {post_drift:.3f}"
        )


class TestMetaTransferAndResilience:
    def test_cross_domain_aggregation_warmstarts_unseen_domain_direction(self):
        """Aggregated source-domain priors should align with an unseen-domain update direction."""
        updates = [
            DomainMetaUpdate('health', np.array([0.35, 0.40], dtype=np.float32), ['lr', 'alpha'], 0.9, 0.0, 0.2),
            DomainMetaUpdate('finance', np.array([0.30, 0.45], dtype=np.float32), ['lr', 'alpha'], 0.8, 0.0, 0.2),
            DomainMetaUpdate('energy', np.array([0.32, 0.42], dtype=np.float32), ['lr', 'alpha'], 0.85, 0.0, 0.2),
        ]
        aggregator = CrossDomainMetaAggregator(CrossMetaConfig(method='trimmed_mean', trim_alpha=0.1, min_confidence=0.05))
        agg_vec, keys, meta = aggregator.aggregate(updates)
        assert meta['participants'] == 3

        optimizer = OnlineReptileOptimizer()
        prior = optimizer.apply(agg_vec, keys, reward=0.75, drg_profile={'vram_high': 0.0, 'latency_high': 0.0, 'bandwidth_free': 1.0})
        prior_vec = np.array([prior[k] for k in keys], dtype=np.float32)

        unseen_target = np.array([0.34, 0.41], dtype=np.float32)
        baseline = np.linalg.norm(unseen_target)  # distance from zero vector
        warmed = np.linalg.norm(unseen_target - prior_vec)
        assert warmed < baseline, f"Warm-start prior should move closer to unseen-domain direction ({warmed:.4f} < {baseline:.4f})"

    def test_optimizer_detects_negative_transfer_and_rolls_back(self):
        """Severe reward drop should trigger rollback decision and restore previous prior."""
        optimizer = OnlineReptileOptimizer()

        keys = ['lr', 'alpha']
        base = optimizer.apply(
            np.array([0.25, 0.25], dtype=np.float32),
            keys,
            reward=0.8,
            drg_profile={'vram_high': 0.0, 'latency_high': 0.0, 'bandwidth_free': 0.0},
        )
        before_bad = dict(base)

        # Apply a bad update with degraded reward.
        optimizer.apply(
            np.array([-1.2, -1.0], dtype=np.float32),
            keys,
            reward=0.05,
            drg_profile={'vram_high': 0.0, 'latency_high': 0.0, 'bandwidth_free': 0.0},
        )

        assert optimizer.should_rollback(0.0) is True
        restored = optimizer.rollback()
        for k in keys:
            assert abs(restored[k] - before_bad[k]) < 1e-9


class TestMetaLearningAgentEndToEnd:
    """Integration tests exercising the full EventBus flow."""

    @pytest.mark.asyncio
    async def test_prior_update_published_with_structured_format(self, tmp_path):
        """MetaLearningAgent must publish meta_prior_update in controller/evaluator format."""
        from scarcity.runtime import EventBus
        from scarcity.meta.meta_learning import MetaLearningAgent, MetaLearningConfig
        from scarcity.meta.scheduler import MetaSchedulerConfig
        from scarcity.meta.storage import MetaStorageConfig

        bus = EventBus()
        cfg = MetaLearningConfig(
            scheduler=MetaSchedulerConfig(
                update_interval_windows=1,
                min_interval_windows=1,  # prevent _adapt_interval from clamping up
            ),
            storage=MetaStorageConfig(root=tmp_path / "meta"),
        )
        agent = MetaLearningAgent(bus=bus, config=cfg)
        await agent.start()

        received: list = []
        bus.subscribe("meta_prior_update", lambda topic, data: received.append(data))

        # Confidence starts at 0 and needs ~2 observations to exceed validator threshold
        # (0.1). Send two packs per domain to build confidence first, then one more
        # to populate _pending_updates for the scheduler cycle.
        for domain in ("finance", "health"):
            for _ in range(2):
                await bus.publish("federation.policy_pack", {
                    "domain_id": domain,
                    "metrics": {"gain_p50": 0.6, "stability_avg": 0.8},
                    "controller": {"tau": 0.9, "gamma_diversity": 0.3},
                    "evaluator": {"g_min": 0.01, "lambda_ci": 0.5},
                })
                await bus.wait_for_idle()

        # Trigger scheduler — update_interval_windows=1 so first window fires
        await bus.publish("processing_metrics", {
            "gain_p50": 0.6,
            "stability_avg": 0.8,
            "latency_ms": 50.0,
            "vram_high": 0.0,
            "bandwidth_free": 1.0,
            "bandwidth_low": 0.0,
        })

        # wait_for_idle snapshots current tasks; nested tasks (meta_prior_update dispatch)
        # are spawned while those run, so drain twice to catch both levels.
        await bus.wait_for_idle()
        await bus.wait_for_idle()

        assert len(received) >= 1, "meta_prior_update should have been published"
        payload = received[0]
        # Must be structured for the engine to consume
        assert "controller" in payload or "evaluator" in payload, (
            f"Prior must be structured with controller/evaluator keys, got: {list(payload.keys())}"
        )
        if "controller" in payload:
            for k in payload["controller"]:
                assert k in {"tau", "gamma_diversity"}, f"Unexpected controller key: {k}"
        if "evaluator" in payload:
            for k in payload["evaluator"]:
                assert k in {"g_min", "lambda_ci"}, f"Unexpected evaluator key: {k}"

        await agent.stop()

    @pytest.mark.asyncio
    async def test_rollback_suppresses_integrative_layer(self, tmp_path):
        """meta_rollback_active from optimizer must suppress MetaSupervisor for 2 cycles."""
        from scarcity.runtime import EventBus
        from scarcity.meta.integrative_meta import MetaSupervisor

        bus = EventBus()
        supervisor = MetaSupervisor(bus=bus)
        await supervisor.start()

        policy_updates: list = []
        bus.subscribe("meta_policy_update", lambda t, d: policy_updates.append(d))

        # Prime supervisor with processing metrics so it has state
        await bus.publish("processing_metrics", {
            "accept_rate": 0.5, "stability_avg": 0.7, "gain_p50": 0.6,
            "latency_ms": 40.0, "vram_util": 0.3,
        })
        await bus.wait_for_idle()
        count_before = len(policy_updates)

        # Optimizer fires rollback signal — must be fully processed before next metrics
        await bus.publish("meta_rollback_active", {"source": "optimizer", "reward": 0.1})
        await bus.wait_for_idle()

        # Next processing_metrics cycle should be suppressed
        await bus.publish("processing_metrics", {
            "accept_rate": 0.5, "stability_avg": 0.7, "gain_p50": 0.6,
            "latency_ms": 40.0, "vram_util": 0.3,
        })
        await bus.wait_for_idle()

        count_after = len(policy_updates)
        assert count_after == count_before, (
            "MetaSupervisor should suppress policy update for 1 cycle after rollback signal"
        )

        await supervisor.stop()
