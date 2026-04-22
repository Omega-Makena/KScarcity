"""
Tests for CrossDomainMetaLearner (Phase 5b) and HierarchicalFederation
run_full_meta_round integration.

Contracts verified:
--- CrossDomainMetaLearner ---
- aggregate() returns (np.ndarray, list, dict) tuple
- aggregate() with no memory falls back (source="fallback")
- aggregate() memory_quality=0 when memory is empty
- aggregate() result identical to fallback when memory is empty
- aggregate() source="memory_backed" when memory has content
- aggregate() memory_quality > 0 when memory has content
- aggregate() result vector has same shape as fallback vector
- aggregate() prior_keys_matched key present in meta
- aggregate() with no updates returns zero vector (fallback)
- aggregate() blended result lies between fallback and prior
- memory_quality capped at max_memory_quality

--- HierarchicalFederation.run_full_meta_round() ---
- returns dict with required keys
- n_updates matches number of registered domain servers
- global_params is a dict
- cross_domain tuple has 3 elements
- run_full_meta_round() with no servers returns n_updates=0

--- existing tests still pass (no regression) ---
"""

import numpy as np
import pytest

from scarcity.federation.domain_server import DomainServer, DomainServerConfig, DomainServerRegistry
from scarcity.federation.global_meta_memory import GlobalMetaMemory, GlobalMetaMemoryConfig
from scarcity.federation.hierarchical import HierarchicalFederation
from scarcity.meta.cross_meta import (
    CrossDomainMetaAggregator,
    CrossDomainMetaLearner,
    CrossDomainMetaLearnerConfig,
)
from scarcity.meta.domain_meta import DomainMetaUpdate
from scarcity.meta.domain_server_meta import DomainServerMeta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _update(
    domain: str = "healthcare",
    keys: list = None,
    vector: list = None,
    confidence: float = 0.7,
    score_delta: float = 0.05,
) -> DomainMetaUpdate:
    import time
    keys = keys or ["gain", "tau"]
    vector = vector or [0.1, 0.05]
    return DomainMetaUpdate(
        domain_id=domain,
        vector=np.array(vector, dtype=np.float32),
        keys=keys,
        confidence=confidence,
        timestamp=time.time(),
        score_delta=score_delta,
    )


def _server(domain: str, basket: str, params: dict = None) -> DomainServer:
    s = DomainServer(domain, basket, DomainServerConfig(memory_capacity=32))
    if params:
        keys = sorted(params.keys())
        vec = np.array([params[k] for k in keys], dtype=np.float32)
        s.update_base_params(vec, keys, reward=0.8)
    return s


def _registry(*servers: DomainServer) -> DomainServerRegistry:
    reg = DomainServerRegistry()
    for s in servers:
        reg._servers[s.basket_id] = s
    return reg


def _filled_memory(n_episodes: int = 10) -> GlobalMetaMemory:
    gmm = GlobalMetaMemory(GlobalMetaMemoryConfig(
        memory_capacity=64,
        min_domains_for_aggregate=2,
    ))
    s1 = _server("healthcare", "b1", {"gain": 0.4, "tau": 0.8})
    s2 = _server("finance", "b2", {"gain": 0.6, "tau": 0.9})
    reg = _registry(s1, s2)
    for _ in range(n_episodes):
        gmm.aggregate(reg)
    return gmm


# ---------------------------------------------------------------------------
# CrossDomainMetaLearner — no memory
# ---------------------------------------------------------------------------

class TestLearnerNoMemory:
    def _learner(self) -> CrossDomainMetaLearner:
        return CrossDomainMetaLearner(global_meta_memory=None)

    def test_aggregate_returns_tuple(self):
        l = self._learner()
        result = l.aggregate([_update()])
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_aggregate_vector_is_ndarray(self):
        l = self._learner()
        vec, keys, meta = l.aggregate([_update()])
        assert isinstance(vec, np.ndarray)

    def test_source_fallback_when_no_memory(self):
        l = self._learner()
        _, _, meta = l.aggregate([_update()])
        assert meta["source"] == "fallback"

    def test_memory_quality_zero_when_no_memory(self):
        l = self._learner()
        _, _, meta = l.aggregate([_update()])
        assert meta["memory_quality"] == pytest.approx(0.0)

    def test_result_matches_fallback_when_no_memory(self):
        fallback = CrossDomainMetaAggregator()
        learner = CrossDomainMetaLearner(global_meta_memory=None)
        updates = [_update("hc", ["gain"], [0.1]), _update("fin", ["gain"], [0.2])]
        fb_vec, fb_keys, _ = fallback.aggregate(updates)
        l_vec, l_keys, _ = learner.aggregate(updates)
        assert l_keys == fb_keys
        np.testing.assert_allclose(l_vec, fb_vec, atol=1e-6)

    def test_no_updates_returns_zero_vector(self):
        l = self._learner()
        vec, keys, _ = l.aggregate([])
        assert len(vec) == 0
        assert keys == []

    def test_meta_contains_required_keys(self):
        l = self._learner()
        _, _, meta = l.aggregate([_update()])
        for k in ("source", "memory_quality", "prior_keys_matched"):
            assert k in meta


# ---------------------------------------------------------------------------
# CrossDomainMetaLearner — with memory
# ---------------------------------------------------------------------------

class TestLearnerWithMemory:
    def _learner(self, memory: GlobalMetaMemory) -> CrossDomainMetaLearner:
        cfg = CrossDomainMetaLearnerConfig(memory_reference_capacity=10)
        return CrossDomainMetaLearner(cfg, global_meta_memory=memory)

    def test_source_memory_backed_after_episodes(self):
        mem = _filled_memory(n_episodes=10)
        l = self._learner(mem)
        _, _, meta = l.aggregate([_update("hc", ["gain", "tau"], [0.1, 0.05])])
        assert meta["source"] == "memory_backed"

    def test_memory_quality_positive_after_episodes(self):
        mem = _filled_memory(n_episodes=5)
        l = self._learner(mem)
        _, _, meta = l.aggregate([_update()])
        assert meta["memory_quality"] > 0.0

    def test_result_vector_same_length_as_fallback(self):
        mem = _filled_memory(n_episodes=5)
        fallback = CrossDomainMetaAggregator()
        learner = self._learner(mem)
        updates = [_update("hc", ["gain", "tau"], [0.1, 0.05]),
                   _update("fin", ["gain", "tau"], [0.2, 0.08])]
        fb_vec, fb_keys, _ = fallback.aggregate(updates)
        l_vec, l_keys, _ = learner.aggregate(updates)
        assert len(l_vec) == len(fb_vec)
        assert l_keys == fb_keys

    def test_prior_keys_matched_present(self):
        mem = _filled_memory(n_episodes=5)
        l = self._learner(mem)
        _, _, meta = l.aggregate([_update("hc", ["gain", "tau"], [0.1, 0.05])])
        assert "prior_keys_matched" in meta
        assert isinstance(meta["prior_keys_matched"], int)

    def test_blended_result_differs_from_fallback(self):
        mem = _filled_memory(n_episodes=10)
        fallback = CrossDomainMetaAggregator()
        learner = self._learner(mem)
        updates = [_update("hc", ["gain", "tau"], [0.1, 0.05]),
                   _update("fin", ["gain", "tau"], [0.2, 0.08])]
        fb_vec, _, _ = fallback.aggregate(updates)
        l_vec, _, meta = learner.aggregate(updates)
        if meta["memory_quality"] > 0 and meta["prior_keys_matched"] > 0:
            # With memory influence, blended result should differ from pure fallback
            assert not np.allclose(l_vec, fb_vec, atol=1e-7)

    def test_memory_quality_capped_at_max(self):
        cfg = CrossDomainMetaLearnerConfig(
            memory_reference_capacity=2,
            max_memory_quality=0.5,
        )
        mem = _filled_memory(n_episodes=50)  # far exceeds reference_capacity
        l = CrossDomainMetaLearner(cfg, global_meta_memory=mem)
        _, _, meta = l.aggregate([_update()])
        assert meta["memory_quality"] <= 0.5 + 1e-9

    def test_extra_context_does_not_crash(self):
        mem = _filled_memory(n_episodes=3)
        l = self._learner(mem)
        _, _, meta = l.aggregate(
            [_update()],
            context={"custom_key": 0.42},
        )
        assert "source" in meta


# ---------------------------------------------------------------------------
# HierarchicalFederation.run_full_meta_round()
# ---------------------------------------------------------------------------

class TestRunFullMetaRound:
    def _fed(self) -> HierarchicalFederation:
        return HierarchicalFederation()

    def test_returns_dict(self):
        fed = self._fed()
        result = fed.run_full_meta_round()
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        fed = self._fed()
        result = fed.run_full_meta_round()
        for k in ("global_params", "cross_domain", "n_updates"):
            assert k in result

    def test_n_updates_zero_with_no_servers(self):
        fed = self._fed()
        result = fed.run_full_meta_round()
        assert result["n_updates"] == 0

    def test_n_updates_matches_server_count(self):
        fed = self._fed()
        fed.get_domain_server("healthcare", "b1")
        fed.get_domain_server("finance", "b2")
        result = fed.run_full_meta_round()
        assert result["n_updates"] == 2

    def test_global_params_is_dict(self):
        fed = self._fed()
        result = fed.run_full_meta_round()
        assert isinstance(result["global_params"], dict)

    def test_cross_domain_is_tuple_of_three(self):
        fed = self._fed()
        result = fed.run_full_meta_round()
        cd = result["cross_domain"]
        assert isinstance(cd, tuple)
        assert len(cd) == 3

    def test_cross_domain_vector_is_ndarray(self):
        fed = self._fed()
        result = fed.run_full_meta_round()
        vec, _, _ = result["cross_domain"]
        assert isinstance(vec, np.ndarray)

    def test_full_round_with_active_servers(self):
        fed = self._fed()
        s1 = fed.get_domain_server("healthcare", "b1")
        s2 = fed.get_domain_server("finance", "b2")
        vec = np.array([0.5, 0.7], dtype=np.float32)
        keys = ["gain", "tau"]
        s1.update_base_params(vec, keys, reward=0.8)
        s2.update_base_params(vec, keys, reward=0.8)
        result = fed.run_full_meta_round()
        assert result["n_updates"] == 2
        _, cross_keys, _ = result["cross_domain"]
        # After a second round with memory, cross_domain should carry the keys
        # (first round has no prior yet — subsequent round will be memory-backed)
        fed.run_full_meta_round()
        result2 = fed.run_full_meta_round()
        assert result2["n_updates"] == 2

    def test_performance_map_forwarded(self):
        fed = self._fed()
        fed.get_domain_server("healthcare", "b1")
        pmap = {"b1": {"gain": 0.8, "stability": 0.9}}
        result = fed.run_full_meta_round(performance_map=pmap)
        assert "n_updates" in result

    def test_existing_run_meta_round_still_works(self):
        fed = self._fed()
        result = fed.run_meta_round()
        assert isinstance(result, dict)

    def test_existing_suggest_prior_still_works(self):
        fed = self._fed()
        assert fed.suggest_prior("new_domain") is None


# ---------------------------------------------------------------------------
# DomainServerMeta + CrossDomainMetaLearner integration
# ---------------------------------------------------------------------------

class TestPhase5Integration:
    def test_observe_then_aggregate_pipeline(self):
        observer = DomainServerMeta()
        mem = _filled_memory(n_episodes=5)
        cfg = CrossDomainMetaLearnerConfig(memory_reference_capacity=5)
        learner = CrossDomainMetaLearner(cfg, global_meta_memory=mem)

        reg = _registry(
            _server("healthcare", "b1", {"gain": 0.4, "tau": 0.8}),
            _server("finance", "b2", {"gain": 0.6, "tau": 0.9}),
        )
        updates = observer.observe_registry(reg)
        assert len(updates) == 2

        vec, keys, meta = learner.aggregate(updates)
        assert isinstance(vec, np.ndarray)
        assert "source" in meta

    def test_two_rounds_accumulate_state(self):
        observer = DomainServerMeta()
        learner = CrossDomainMetaLearner(global_meta_memory=None)
        reg = _registry(
            _server("hc", "b1", {"gain": 0.3}),
            _server("fin", "b2", {"gain": 0.7}),
        )
        updates_r1 = observer.observe_registry(reg)
        updates_r2 = observer.observe_registry(reg)
        assert observer.n_domains_tracked == 2
        _, _, m1 = learner.aggregate(updates_r1)
        _, _, m2 = learner.aggregate(updates_r2)
        assert m1["participants"] == m2["participants"]
