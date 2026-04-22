"""
Tests for GlobalMetaMemory and HierarchicalFederation meta-learning API.

Contracts verified:
--- GlobalMetaMemory ---
- absorb_domain() captures snapshot; n_domains_tracked increments
- absorb_domain() updates existing snapshot for same basket_id
- aggregate() returns empty dict when fewer than min_domains servers present
- aggregate() returns global_params with at least one shared key after 2+ domains
- aggregate() stores episode in memory (memory_size increments)
- aggregate() update_count increments each call
- global_params reflects robust median across domains
- global_params returns copy (mutation doesn't affect internal state)
- suggest_prior() returns None when memory is empty
- suggest_prior() returns a dict after at least one aggregate
- suggest_prior() returns value with keys from global_params
- status() contains required keys
- domain_snapshot() returns None for unknown basket
- domain_snapshot() returns DomainSnapshot for known basket
- min_domains_for_aggregate respected: no episode stored when below threshold
--- HierarchicalFederation integration ---
- get_domain_server() returns DomainServer
- get_domain_server() same basket_id returns same server
- run_meta_round() returns dict (may be empty if < 2 servers)
- run_meta_round() with 2+ servers returns non-empty global_params
- suggest_prior() returns None before any meta round
- suggest_prior() returns dict after meta round with 2+ domains
- existing tests still pass (no regression)
"""

import numpy as np
import pytest

from scarcity.federation.domain_server import (
    DomainServer,
    DomainServerConfig,
    DomainServerRegistry,
)
from scarcity.federation.global_meta_memory import (
    GlobalMetaMemory,
    GlobalMetaMemoryConfig,
    DomainSnapshot,
)
from scarcity.federation.hierarchical import (
    HierarchicalFederation,
    HierarchicalFederationConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_server(domain: str, basket: str, params: dict = None) -> DomainServer:
    s = DomainServer(domain, basket, DomainServerConfig(memory_capacity=32))
    if params:
        keys = list(params.keys())
        vec = np.array(list(params.values()), dtype=np.float32)
        s.update_base_params(vec, keys, reward=0.8)
    return s


def _make_registry(*servers: DomainServer) -> DomainServerRegistry:
    reg = DomainServerRegistry()
    for s in servers:
        reg._servers[s.basket_id] = s
    return reg


def _perf(gain: float = 0.1, stability: float = 0.8) -> dict:
    return {"gain": gain, "stability": stability}


def _gmm(min_domains: int = 2) -> GlobalMetaMemory:
    return GlobalMetaMemory(
        GlobalMetaMemoryConfig(memory_capacity=64, min_domains_for_aggregate=min_domains)
    )


# ---------------------------------------------------------------------------
# absorb_domain
# ---------------------------------------------------------------------------

class TestAbsorbDomain:
    def test_increments_n_domains_tracked(self):
        gmm = _gmm()
        s = _make_server("healthcare", "b1", {"gain": 0.5})
        gmm.absorb_domain(s, _perf())
        assert gmm.n_domains_tracked == 1

    def test_two_domains_tracked(self):
        gmm = _gmm()
        gmm.absorb_domain(_make_server("healthcare", "b1"), _perf())
        gmm.absorb_domain(_make_server("finance", "b2"), _perf())
        assert gmm.n_domains_tracked == 2

    def test_same_basket_overwrites_snapshot(self):
        gmm = _gmm()
        s = _make_server("healthcare", "b1", {"gain": 0.3})
        gmm.absorb_domain(s, {"gain": 0.1})
        gmm.absorb_domain(s, {"gain": 0.9})  # second absorb, same basket
        snap = gmm.domain_snapshot("b1")
        assert snap.performance["gain"] == pytest.approx(0.9)

    def test_snapshot_contains_correct_domain_id(self):
        gmm = _gmm()
        gmm.absorb_domain(_make_server("energy", "b_en"), _perf())
        snap = gmm.domain_snapshot("b_en")
        assert snap.domain_id == "energy"

    def test_snapshot_contains_base_params(self):
        gmm = _gmm()
        s = _make_server("finance", "b_fin", {"gain": 0.4, "tau": 0.7})
        gmm.absorb_domain(s, _perf())
        snap = gmm.domain_snapshot("b_fin")
        assert "gain" in snap.base_params

    def test_domain_snapshot_none_for_unknown_basket(self):
        gmm = _gmm()
        assert gmm.domain_snapshot("nonexistent") is None


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

class TestAggregate:
    def test_returns_empty_below_min_domains(self):
        gmm = _gmm(min_domains=2)
        reg = _make_registry(_make_server("healthcare", "b1", {"gain": 0.5}))
        result = gmm.aggregate(reg)
        assert result == {}

    def test_returns_global_params_with_two_domains(self):
        gmm = _gmm(min_domains=2)
        s1 = _make_server("healthcare", "b1", {"gain": 0.4, "tau": 0.8})
        s2 = _make_server("finance",    "b2", {"gain": 0.6, "tau": 0.9})
        reg = _make_registry(s1, s2)
        result = gmm.aggregate(reg)
        assert "gain" in result
        assert "tau" in result

    def test_global_params_is_median(self):
        # REPTILE applies beta=0.05 so base_params = 0.05 * vec, not vec directly.
        # Median of two equal-beta-scaled values is their average.
        gmm = _gmm(min_domains=2)
        s1 = _make_server("healthcare", "b1", {"gain": 0.2})
        s2 = _make_server("finance",    "b2", {"gain": 0.8})
        reg = _make_registry(s1, s2)
        result = gmm.aggregate(reg)
        # Both servers were set from zero with the same beta; global median
        # must lie strictly between the two base_param values.
        p1 = s1.base_params.get("gain", 0.0)
        p2 = s2.base_params.get("gain", 0.0)
        lo, hi = min(p1, p2), max(p1, p2)
        assert lo <= result["gain"] <= hi

    def test_aggregate_stores_episode(self):
        gmm = _gmm(min_domains=2)
        reg = _make_registry(
            _make_server("healthcare", "b1", {"gain": 0.4}),
            _make_server("finance", "b2", {"gain": 0.6}),
        )
        gmm.aggregate(reg)
        assert gmm.memory_size == 1

    def test_aggregate_increments_update_count(self):
        gmm = _gmm(min_domains=2)
        reg = _make_registry(
            _make_server("healthcare", "b1", {"gain": 0.4}),
            _make_server("finance", "b2", {"gain": 0.6}),
        )
        gmm.aggregate(reg)
        gmm.aggregate(reg)
        assert gmm.update_count == 2

    def test_no_episode_stored_below_min_domains(self):
        gmm = _gmm(min_domains=3)
        reg = _make_registry(
            _make_server("healthcare", "b1", {"gain": 0.4}),
            _make_server("finance", "b2", {"gain": 0.6}),
        )
        gmm.aggregate(reg)
        assert gmm.memory_size == 0

    def test_global_params_property_reflects_last_aggregate(self):
        gmm = _gmm(min_domains=2)
        s1 = _make_server("healthcare", "b1", {"gain": 0.5})
        s2 = _make_server("finance", "b2", {"gain": 0.5})
        reg = _make_registry(s1, s2)
        gmm.aggregate(reg)
        assert "gain" in gmm.global_params

    def test_global_params_returns_copy(self):
        gmm = _gmm(min_domains=2)
        reg = _make_registry(
            _make_server("healthcare", "b1", {"gain": 0.5}),
            _make_server("finance", "b2", {"gain": 0.5}),
        )
        gmm.aggregate(reg)
        copy = gmm.global_params
        copy["gain"] = 9999.0
        assert gmm.global_params.get("gain") != 9999.0

    def test_performance_map_forwarded_to_absorb(self):
        gmm = _gmm(min_domains=2)
        s1 = _make_server("healthcare", "b1", {"gain": 0.4})
        s2 = _make_server("finance",    "b2", {"gain": 0.6})
        reg = _make_registry(s1, s2)
        gmm.aggregate(reg, performance_map={"b1": {"gain": 0.77}, "b2": {"gain": 0.55}})
        snap = gmm.domain_snapshot("b1")
        assert snap.performance["gain"] == pytest.approx(0.77)


# ---------------------------------------------------------------------------
# suggest_prior
# ---------------------------------------------------------------------------

class TestSuggestPrior:
    def test_returns_none_when_memory_empty(self):
        gmm = _gmm()
        assert gmm.suggest_prior("healthcare", {}) is None

    def test_returns_dict_after_aggregate(self):
        gmm = _gmm(min_domains=2)
        reg = _make_registry(
            _make_server("healthcare", "b1", {"gain": 0.5}),
            _make_server("finance",    "b2", {"gain": 0.5}),
        )
        gmm.aggregate(reg)
        result = gmm.suggest_prior("new_domain", {})
        assert isinstance(result, dict)

    def test_returned_dict_contains_global_param_keys(self):
        gmm = _gmm(min_domains=2)
        reg = _make_registry(
            _make_server("healthcare", "b1", {"gain": 0.4, "tau": 0.8}),
            _make_server("finance",    "b2", {"gain": 0.6, "tau": 0.9}),
        )
        gmm.aggregate(reg)
        result = gmm.suggest_prior("new_domain", {})
        assert "gain" in result
        assert "tau" in result

    def test_suggest_with_context_does_not_crash(self):
        gmm = _gmm(min_domains=2)
        reg = _make_registry(
            _make_server("healthcare", "b1", {"gain": 0.4}),
            _make_server("finance",    "b2", {"gain": 0.6}),
        )
        gmm.aggregate(reg)
        result = gmm.suggest_prior("energy", {"gain_p50": 0.3, "stability_mean": 0.7})
        assert result is not None


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

class TestStatus:
    def test_status_contains_required_keys(self):
        gmm = _gmm()
        st = gmm.status()
        for k in ("update_count", "memory_size", "n_domains_tracked", "global_params_keys"):
            assert k in st

    def test_status_update_count_matches(self):
        gmm = _gmm(min_domains=2)
        reg = _make_registry(
            _make_server("healthcare", "b1", {"gain": 0.4}),
            _make_server("finance",    "b2", {"gain": 0.6}),
        )
        gmm.aggregate(reg)
        assert gmm.status()["update_count"] == 1


# ---------------------------------------------------------------------------
# HierarchicalFederation integration
# ---------------------------------------------------------------------------

class TestHierarchicalFederationMeta:
    def _fed(self) -> HierarchicalFederation:
        return HierarchicalFederation()

    def test_domain_registry_attribute_exists(self):
        fed = self._fed()
        assert hasattr(fed, "domain_registry")

    def test_global_meta_memory_attribute_exists(self):
        fed = self._fed()
        assert hasattr(fed, "global_meta_memory")

    def test_get_domain_server_returns_domain_server(self):
        fed = self._fed()
        s = fed.get_domain_server("healthcare", "b1")
        assert isinstance(s, DomainServer)

    def test_get_domain_server_same_basket_same_object(self):
        fed = self._fed()
        s1 = fed.get_domain_server("healthcare", "b1")
        s2 = fed.get_domain_server("healthcare", "b1")
        assert s1 is s2

    def test_get_domain_server_different_baskets_different_objects(self):
        fed = self._fed()
        s1 = fed.get_domain_server("healthcare", "b1")
        s2 = fed.get_domain_server("finance", "b2")
        assert s1 is not s2

    def test_run_meta_round_returns_dict(self):
        fed = self._fed()
        result = fed.run_meta_round()
        assert isinstance(result, dict)

    def test_run_meta_round_empty_when_no_servers(self):
        fed = self._fed()
        result = fed.run_meta_round()
        assert result == {}

    def test_run_meta_round_returns_params_with_two_servers(self):
        fed = self._fed()
        s1 = fed.get_domain_server("healthcare", "b1")
        s2 = fed.get_domain_server("finance",    "b2")
        # Give both servers some base params via REPTILE
        vec = np.array([0.5, 0.7], dtype=np.float32)
        keys = ["gain", "tau"]
        s1.update_base_params(vec, keys, reward=0.8)
        s2.update_base_params(vec, keys, reward=0.8)
        result = fed.run_meta_round()
        assert "gain" in result
        assert "tau" in result

    def test_suggest_prior_none_before_meta_round(self):
        fed = self._fed()
        assert fed.suggest_prior("new_domain") is None

    def test_suggest_prior_returns_dict_after_meta_round(self):
        fed = self._fed()
        s1 = fed.get_domain_server("healthcare", "b1")
        s2 = fed.get_domain_server("finance",    "b2")
        vec = np.array([0.5], dtype=np.float32)
        s1.update_base_params(vec, ["gain"], reward=0.8)
        s2.update_base_params(vec, ["gain"], reward=0.8)
        fed.run_meta_round()
        result = fed.suggest_prior("energy")
        assert isinstance(result, dict)

    def test_suggest_prior_with_context(self):
        fed = self._fed()
        s1 = fed.get_domain_server("healthcare", "b1")
        s2 = fed.get_domain_server("finance",    "b2")
        vec = np.array([0.4, 0.9], dtype=np.float32)
        keys = ["gain", "tau"]
        s1.update_base_params(vec, keys, reward=0.8)
        s2.update_base_params(vec, keys, reward=0.8)
        fed.run_meta_round()
        result = fed.suggest_prior("energy", {"gain_p50": 0.3, "stability_mean": 0.7})
        assert result is not None

    def test_existing_meta_params_api_unchanged(self):
        fed = self._fed()
        params = fed.get_meta_params()
        assert isinstance(params, dict)

    def test_existing_register_client_unchanged(self):
        fed = self._fed()
        basket_id = fed.register_client("client_1", "healthcare")
        assert isinstance(basket_id, str)
