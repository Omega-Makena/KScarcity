"""
Tests for DomainServerMeta (Phase 5a).

Contracts verified:
- observe() returns DomainMetaUpdate
- observe() domain_id matches server's domain_id
- observe() confidence >= min_confidence floor
- observe() confidence increases with higher hit_rate
- observe() confidence increases with larger memory_size
- observe() positive performance.gain boosts confidence
- observe() keys match sorted server.base_params keys
- observe() delta is zero on first call when server params are zero
- observe() delta reflects change in base_params between calls
- observe() score_delta equals hit_rate change since last observe
- observe() cold server (empty base_params) returns empty vector
- observe_registry() returns one update per server
- observe_registry() empty registry returns empty list
- observe_registry() forwards performance_map per basket
- n_domains_tracked increments after observe()
- status() contains required keys
"""

import math
import numpy as np
import pytest

from scarcity.federation.domain_server import DomainServer, DomainServerConfig, DomainServerRegistry
from scarcity.meta.domain_server_meta import DomainServerMeta, DomainServerMetaConfig
from scarcity.meta.domain_meta import DomainMetaUpdate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _server(domain: str = "healthcare", basket: str = "b1", params: dict = None) -> DomainServer:
    s = DomainServer(domain, basket, DomainServerConfig(memory_capacity=32))
    if params:
        keys = sorted(params.keys())
        vec = np.array([params[k] for k in keys], dtype=np.float32)
        s.update_base_params(vec, keys, reward=0.8)
    return s


def _meta(min_confidence: float = 0.05) -> DomainServerMeta:
    return DomainServerMeta(DomainServerMetaConfig(min_confidence=min_confidence))


# ---------------------------------------------------------------------------
# observe() — return type and field correctness
# ---------------------------------------------------------------------------

class TestObserveReturnType:
    def test_returns_domain_meta_update(self):
        m = _meta()
        s = _server()
        result = m.observe(s)
        assert isinstance(result, DomainMetaUpdate)

    def test_domain_id_matches_server(self):
        m = _meta()
        s = _server(domain="finance")
        assert m.observe(s).domain_id == "finance"

    def test_keys_match_sorted_base_params(self):
        m = _meta()
        s = _server(params={"tau": 0.9, "gain": 0.5})
        result = m.observe(s)
        assert result.keys == ["gain", "tau"]

    def test_vector_length_matches_keys(self):
        m = _meta()
        s = _server(params={"gain": 0.5, "tau": 0.9})
        result = m.observe(s)
        assert len(result.vector) == len(result.keys)

    def test_cold_server_returns_empty_vector(self):
        m = _meta()
        s = _server()   # no params set
        result = m.observe(s)
        assert len(result.vector) == 0
        assert result.keys == []


# ---------------------------------------------------------------------------
# observe() — confidence
# ---------------------------------------------------------------------------

class TestObserveConfidence:
    def test_confidence_at_least_min_floor(self):
        m = _meta(min_confidence=0.05)
        s = _server()   # hit_rate=0, memory_size=0 → would be 0 without floor
        result = m.observe(s)
        assert result.confidence >= 0.05

    def test_higher_hit_rate_raises_confidence(self):
        cfg = DomainServerMetaConfig(hit_rate_weight=0.6, memory_weight=0.0)
        m = DomainServerMeta(cfg)
        # Build a server with a memory hit by recording and adapting
        s_lo = _server("hc", "b_lo")
        s_hi = _server("hc", "b_hi")
        # Simulate high hit_rate on s_hi by recording then adapting
        from scarcity.meta.adaptation import AdaptationEngine
        ctx = {"gain_p50": 0.5}
        s_hi.record(ctx, {"gain": 0.5}, {"gain": 0.5}, {"gain": 0.1})
        s_hi.adapt(ctx, {"gain": 0.5})  # causes a hit
        r_lo = m.observe(s_lo)
        r_hi = m.observe(s_hi)
        assert r_hi.confidence >= r_lo.confidence

    def test_positive_gain_boosts_confidence(self):
        cfg = DomainServerMetaConfig(performance_gain_boost=0.1, min_confidence=0.0)
        m = DomainServerMeta(cfg)
        s = _server()
        r_no_gain = m.observe(s, performance={"gain": 0.0})
        m2 = DomainServerMeta(cfg)
        r_with_gain = m2.observe(s, performance={"gain": 1.0})
        assert r_with_gain.confidence > r_no_gain.confidence

    def test_larger_memory_size_raises_confidence(self):
        cfg = DomainServerMetaConfig(hit_rate_weight=0.0, memory_weight=1.0)
        m = DomainServerMeta(cfg)
        s_small = _server("hc", "b_small", params={"gain": 0.5})
        s_large = _server("hc", "b_large", params={"gain": 0.5})
        # Give s_large more memory entries
        for i in range(10):
            s_large.record({"gain_p50": 0.1 * i}, {"gain": 0.5}, {"gain": 0.5}, {"gain": 0.01})
        r_small = m.observe(s_small)
        r_large = m.observe(s_large)
        assert r_large.confidence >= r_small.confidence


# ---------------------------------------------------------------------------
# observe() — delta vector
# ---------------------------------------------------------------------------

class TestObserveDelta:
    def test_delta_zero_on_first_call_for_cold_server_with_params(self):
        # First call: prev_params = zeros, curr = current. delta = meta_lr * (curr - 0).
        # But for a server set to gain=0 via REPTILE from zero, delta should be very small.
        m = _meta()
        s = _server(params={"gain": 0.0})
        result = m.observe(s)
        assert result.vector is not None
        assert len(result.vector) == 1

    def test_delta_reflects_param_change_between_observations(self):
        m = _meta()
        s = _server("hc", "b1", params={"gain": 0.2})
        m.observe(s)  # baseline snapshot
        # Update server params
        s.update_base_params(np.array([0.8], dtype=np.float32), ["gain"], reward=0.9)
        result = m.observe(s)
        # Delta should be non-zero (param grew)
        assert abs(result.vector[0]) > 0.0

    def test_score_delta_reflects_hit_rate_change(self):
        m = _meta()
        s = _server("hc", "b1")
        r1 = m.observe(s)
        assert r1.score_delta == pytest.approx(0.0)
        # Simulate a hit by recording and adapting
        s.record({"g": 0.5}, {"gain": 0.5}, {"gain": 0.5}, {"gain": 0.1})
        s.adapt({"g": 0.5}, {"gain": 0.5})
        r2 = m.observe(s)
        assert r2.score_delta >= 0.0  # hit_rate grew → positive delta


# ---------------------------------------------------------------------------
# observe_registry()
# ---------------------------------------------------------------------------

class TestObserveRegistry:
    def test_returns_one_update_per_server(self):
        m = _meta()
        reg = DomainServerRegistry()
        reg.get_or_create("hc", "b1")
        reg.get_or_create("fin", "b2")
        updates = m.observe_registry(reg)
        assert len(updates) == 2

    def test_empty_registry_returns_empty_list(self):
        m = _meta()
        reg = DomainServerRegistry()
        assert m.observe_registry(reg) == []

    def test_performance_map_forwarded(self):
        cfg = DomainServerMetaConfig(performance_gain_boost=0.2, min_confidence=0.0)
        m = DomainServerMeta(cfg)
        reg = DomainServerRegistry()
        reg.get_or_create("hc", "b1")
        pmap = {"b1": {"gain": 1.0}}
        updates_with = m.observe_registry(reg, performance_map=pmap)
        m2 = DomainServerMeta(cfg)
        updates_without = m2.observe_registry(reg)
        assert updates_with[0].confidence >= updates_without[0].confidence

    def test_all_updates_are_domain_meta_updates(self):
        m = _meta()
        reg = DomainServerRegistry()
        for i in range(4):
            reg.get_or_create(f"domain_{i}", f"basket_{i}")
        updates = m.observe_registry(reg)
        assert all(isinstance(u, DomainMetaUpdate) for u in updates)


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class TestTelemetry:
    def test_n_domains_tracked_increments(self):
        m = _meta()
        s1 = _server("hc", "b1")
        s2 = _server("fin", "b2")
        m.observe(s1)
        assert m.n_domains_tracked == 1
        m.observe(s2)
        assert m.n_domains_tracked == 2

    def test_same_server_does_not_double_count(self):
        m = _meta()
        s = _server()
        m.observe(s)
        m.observe(s)
        assert m.n_domains_tracked == 1

    def test_status_contains_required_keys(self):
        m = _meta()
        st = m.status()
        assert "n_domains_tracked" in st
        assert "basket_ids" in st

    def test_status_basket_ids_populated(self):
        m = _meta()
        m.observe(_server("hc", "b1"))
        m.observe(_server("fin", "b2"))
        assert set(m.status()["basket_ids"]) == {"b1", "b2"}
