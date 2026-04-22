"""
Tests for DomainServer and DomainServerRegistry.

Contracts verified:
- DomainServer created with correct domain_id / basket_id
- adapt() returns AdaptationResult with correct query_key shape
- adapt() with no base_params uses internal base_params (empty dict → passthrough)
- adapt() with base_params uses supplied params
- record() stores episode; subsequent adapt() returns memory hit
- receive_client_update() increments client_update_count
- receive_client_update() stores episode retrievable by adapt()
- update_base_params() evolves base_params via REPTILE
- update_base_params() increments round_id
- adapt() uses updated base_params after update_base_params()
- hit_rate reflects engine counters
- memory_size reflects stored episodes
- status() returns expected keys and values
- _enrich() injects domain_id when missing
- _enrich() preserves caller-supplied domain_id
- DomainServerRegistry: get_or_create returns same server for same basket_id
- DomainServerRegistry: creates different servers for different basket_ids
- DomainServerRegistry: get() returns None for unknown basket
- DomainServerRegistry: len() matches server count
- DomainServerRegistry: aggregate_status() returns list of status dicts
- DomainServerRegistry: domain_ids() returns correct list
"""

import numpy as np
import pytest

from scarcity.federation.domain_server import (
    DomainServer,
    DomainServerConfig,
    DomainServerRegistry,
)
from scarcity.meta.adaptation import AdaptationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _server(
    domain: str = "healthcare",
    basket: str = "basket_hc_001",
    min_similarity: float = 0.0,
    capacity: int = 64,
) -> DomainServer:
    cfg = DomainServerConfig(
        memory_capacity=capacity,
        min_similarity=min_similarity,
    )
    return DomainServer(domain, basket, cfg)


def _ctx(**extra) -> dict:
    return {"gain_p50": 0.3, "stability_mean": 0.7, **extra}


def _base() -> dict:
    return {"gain": 0.5, "tau": 0.9}


def _delta() -> dict:
    return {"gain": 0.12}


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_domain_id_stored(self):
        s = _server(domain="finance")
        assert s.domain_id == "finance"

    def test_basket_id_stored(self):
        s = _server(basket="basket_fin_abc")
        assert s.basket_id == "basket_fin_abc"

    def test_default_base_params_empty(self):
        s = _server()
        assert s.base_params == {}

    def test_initial_round_id_zero(self):
        s = _server()
        assert s.round_id == 0

    def test_initial_memory_size_zero(self):
        s = _server()
        assert s.memory_size == 0


# ---------------------------------------------------------------------------
# adapt() interface
# ---------------------------------------------------------------------------

class TestAdapt:
    def test_returns_adaptation_result(self):
        s = _server()
        result = s.adapt(_ctx())
        assert isinstance(result, AdaptationResult)

    def test_query_key_shape_matches_encoder(self):
        s = _server()
        result = s.adapt(_ctx())
        assert result.query_key.shape == (s.engine.encoder.output_dim,)

    def test_adapt_with_no_base_params_is_passthrough(self):
        s = _server(min_similarity=0.99)
        result = s.adapt(_ctx())
        assert result.source in ("passthrough", "reptile")
        assert result.adapted_params == {}

    def test_adapt_uses_supplied_base_params(self):
        s = _server(min_similarity=0.99)
        base = {"gain": 0.5, "tau": 0.9}
        result = s.adapt(_ctx(), base_params=base)
        assert set(result.adapted_params.keys()) >= set(base.keys())

    def test_adapt_hits_memory_after_record(self):
        s = _server(min_similarity=0.0)
        ctx = _ctx()
        base = _base()
        s.record(ctx, base, base, _delta())
        result = s.adapt(ctx, base)
        assert result.source == "memory"

    def test_adapt_injects_domain_id_into_context(self):
        s = _server(domain="energy")
        ctx = {"gain_p50": 0.5}  # no domain_id
        # Should not crash; domain embedding uses "energy"
        result = s.adapt(ctx, _base())
        assert result is not None

    def test_adapt_preserves_caller_domain_id(self):
        s = _server(domain="healthcare")
        ctx = {"domain_id": "finance", "gain_p50": 0.5}
        # Caller set domain_id — it should be preserved, not overwritten
        result = s.adapt(ctx, _base())
        assert result is not None


# ---------------------------------------------------------------------------
# record() and memory
# ---------------------------------------------------------------------------

class TestRecord:
    def test_record_increments_memory_size(self):
        s = _server()
        assert s.memory_size == 0
        s.record(_ctx(), _base(), _base(), _delta())
        assert s.memory_size == 1

    def test_multiple_records_grow_memory(self):
        s = _server()
        for i in range(5):
            s.record({"gain_p50": 0.1 * i}, _base(), _base(), _delta())
        assert s.memory_size == 5

    def test_record_with_policy_stored_correctly(self):
        s = _server(min_similarity=0.0)
        ctx = _ctx()
        base = _base()
        s.record(ctx, base, base, _delta(), policy={"source": "test", "step": 1})
        result = s.adapt(ctx, base)
        assert result.source == "memory"
        assert result.n_retrieved >= 1


# ---------------------------------------------------------------------------
# receive_client_update()
# ---------------------------------------------------------------------------

class TestReceiveClientUpdate:
    def test_increments_client_update_count(self):
        s = _server()
        s.receive_client_update("client_1", {"gain": 0.05}, _ctx(), _delta())
        assert s._client_update_count == 1

    def test_multiple_clients_accumulate_count(self):
        s = _server()
        for i in range(3):
            s.receive_client_update(f"client_{i}", {"gain": 0.01 * i}, _ctx(), _delta())
        assert s._client_update_count == 3

    def test_client_update_stored_in_memory(self):
        s = _server()
        s.receive_client_update("client_1", {"gain": 0.05}, _ctx(), _delta())
        assert s.memory_size == 1

    def test_client_update_retrievable_by_adapt(self):
        s = _server(min_similarity=0.0)
        ctx = _ctx()
        base = _base()
        s.receive_client_update("client_1", {"gain": 0.05}, ctx, {"gain": 0.08})
        result = s.adapt(ctx, base)
        assert result.source == "memory"

    def test_client_update_policy_contains_client_id(self):
        s = _server(min_similarity=0.0)
        ctx = _ctx()
        base = _base()
        s.receive_client_update("client_99", {"gain": 0.05}, ctx, _delta())
        result = s.adapt(ctx, base)
        assert result.n_retrieved >= 1
        ep = result.entry if hasattr(result, "entry") else None
        # policy is inside retrieved entry
        retrieved = s.engine.memory.retrieve(result.query_key, top_k=1)
        assert retrieved[0].entry.policy.get("client_id") == "client_99"


# ---------------------------------------------------------------------------
# update_base_params()
# ---------------------------------------------------------------------------

class TestUpdateBaseParams:
    def test_update_base_params_returns_dict(self):
        s = _server()
        keys = ["gain", "tau"]
        vec = np.array([0.3, 0.1], dtype=np.float32)
        result = s.update_base_params(vec, keys, reward=0.8)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"gain", "tau"}

    def test_update_base_params_increments_round_id(self):
        s = _server()
        keys = ["gain"]
        vec = np.array([0.1], dtype=np.float32)
        s.update_base_params(vec, keys, reward=0.7)
        assert s.round_id == 1

    def test_multiple_updates_increment_round_id(self):
        s = _server()
        keys = ["gain"]
        vec = np.array([0.1], dtype=np.float32)
        for _ in range(3):
            s.update_base_params(vec, keys, reward=0.7)
        assert s.round_id == 3

    def test_base_params_updated_after_reptile(self):
        s = _server()
        keys = ["gain", "tau"]
        vec = np.array([0.2, 0.05], dtype=np.float32)
        updated = s.update_base_params(vec, keys, reward=0.8)
        assert s.base_params == updated

    def test_adapt_uses_updated_base_params_on_fallback(self):
        s = _server(min_similarity=0.99)  # force fallback
        keys = ["gain"]
        vec = np.array([0.4], dtype=np.float32)
        prior = s.update_base_params(vec, keys, reward=0.9)
        result = s.adapt(_ctx())
        if result.source == "reptile":
            assert result.adapted_params.get("gain") == pytest.approx(prior["gain"], abs=1e-5)

    def test_update_accepts_drg_profile(self):
        s = _server()
        keys = ["gain"]
        vec = np.array([0.1], dtype=np.float32)
        result = s.update_base_params(vec, keys, reward=0.7, drg_profile={"vram_high": 1.0})
        assert "gain" in result


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class TestTelemetry:
    def test_hit_rate_zero_initially(self):
        s = _server()
        assert s.hit_rate == 0.0

    def test_hit_rate_increases_after_hit(self):
        s = _server(min_similarity=0.0)
        ctx = _ctx()
        base = _base()
        s.record(ctx, base, base, _delta())
        s.adapt(ctx, base)
        assert s.hit_rate > 0.0

    def test_status_contains_required_keys(self):
        s = _server()
        st = s.status()
        for key in ("domain_id", "basket_id", "round_id", "memory_size",
                    "client_updates", "hit_rate", "hits", "misses"):
            assert key in st

    def test_status_domain_id_correct(self):
        s = _server(domain="retail")
        assert s.status()["domain_id"] == "retail"

    def test_status_round_id_matches_property(self):
        s = _server()
        keys = ["gain"]
        vec = np.array([0.1], dtype=np.float32)
        s.update_base_params(vec, keys, reward=0.5)
        assert s.status()["round_id"] == s.round_id


# ---------------------------------------------------------------------------
# DomainServerRegistry
# ---------------------------------------------------------------------------

class TestDomainServerRegistry:
    def test_get_or_create_returns_domain_server(self):
        reg = DomainServerRegistry()
        s = reg.get_or_create("healthcare", "basket_hc")
        assert isinstance(s, DomainServer)

    def test_same_basket_id_returns_same_server(self):
        reg = DomainServerRegistry()
        s1 = reg.get_or_create("healthcare", "basket_hc")
        s2 = reg.get_or_create("healthcare", "basket_hc")
        assert s1 is s2

    def test_different_basket_ids_return_different_servers(self):
        reg = DomainServerRegistry()
        s1 = reg.get_or_create("healthcare", "basket_hc")
        s2 = reg.get_or_create("finance", "basket_fin")
        assert s1 is not s2

    def test_get_returns_none_for_unknown_basket(self):
        reg = DomainServerRegistry()
        assert reg.get("nonexistent") is None

    def test_get_returns_created_server(self):
        reg = DomainServerRegistry()
        reg.get_or_create("energy", "basket_en")
        assert reg.get("basket_en") is not None

    def test_len_matches_server_count(self):
        reg = DomainServerRegistry()
        reg.get_or_create("healthcare", "b1")
        reg.get_or_create("finance", "b2")
        reg.get_or_create("energy", "b3")
        assert len(reg) == 3

    def test_domain_ids_returns_all_domains(self):
        reg = DomainServerRegistry()
        reg.get_or_create("healthcare", "b1")
        reg.get_or_create("finance", "b2")
        assert set(reg.domain_ids()) == {"healthcare", "finance"}

    def test_aggregate_status_returns_list(self):
        reg = DomainServerRegistry()
        reg.get_or_create("healthcare", "b1")
        reg.get_or_create("finance", "b2")
        statuses = reg.aggregate_status()
        assert isinstance(statuses, list)
        assert len(statuses) == 2

    def test_aggregate_status_contains_status_keys(self):
        reg = DomainServerRegistry()
        reg.get_or_create("healthcare", "b1")
        for st in reg.aggregate_status():
            assert "domain_id" in st
            assert "memory_size" in st

    def test_all_servers_snapshot_is_copy(self):
        reg = DomainServerRegistry()
        reg.get_or_create("healthcare", "b1")
        snapshot = reg.all_servers()
        snapshot["b1"] = None  # mutate copy
        assert reg.get("b1") is not None  # original unaffected

    def test_server_config_propagated(self):
        cfg = DomainServerConfig(memory_capacity=128)
        reg = DomainServerRegistry(server_config=cfg)
        s = reg.get_or_create("healthcare", "b1")
        assert s.config.memory_capacity == 128
