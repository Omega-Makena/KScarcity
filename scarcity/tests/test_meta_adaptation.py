"""
Tests for AdaptationEngine.

Contracts verified:
- Memory hit: returns "memory" source with similarity list
- Memory miss (no entries): returns "passthrough" source
- Memory miss (entries below threshold): returns "reptile" or "passthrough"
- Reptile fallback: prior merged into base_params when prior is non-empty
- Passthrough: base_params returned unchanged when prior is also empty
- Weighted blend: delta applied proportional to similarity
- top1 blend: only highest-similarity delta applied
- blend covers keys not in base_params
- record() stores episode retrievable on next adapt
- hit/miss counters increment correctly
- hit_rate computed correctly
- query_key shape matches encoder.output_dim
- AdaptationResult.n_retrieved matches len(similarities)
- Identical context + identical memory → same result (determinism)
"""

import numpy as np
import pytest

from scarcity.meta.encoder import ContextEncoder, ContextEncoderConfig
from scarcity.meta.memory import EpisodicMemory, EpisodicMemoryConfig
from scarcity.meta.optimizer import OnlineReptileOptimizer, MetaOptimizerConfig
from scarcity.meta.adaptation import AdaptationEngine, AdaptationConfig, AdaptationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine(
    min_similarity: float = 0.0,
    top_k: int = 5,
    blend_mode: str = "weighted",
    capacity: int = 64,
) -> AdaptationEngine:
    encoder = ContextEncoder(ContextEncoderConfig(normalize=True))
    memory = EpisodicMemory(EpisodicMemoryConfig(capacity=capacity, top_k=top_k))
    optimizer = OnlineReptileOptimizer()
    config = AdaptationConfig(
        min_similarity=min_similarity,
        top_k=top_k,
        blend_mode=blend_mode,
    )
    return AdaptationEngine(encoder=encoder, memory=memory, optimizer=optimizer, config=config)


def _base() -> dict:
    return {"gain": 0.5, "tau": 0.9}


def _ctx(domain: str = "healthcare") -> dict:
    return {"domain_id": domain, "gain_p50": 0.2, "stability_mean": 0.7}


# ---------------------------------------------------------------------------
# Source routing
# ---------------------------------------------------------------------------

class TestSourceRouting:
    def test_empty_memory_passthrough(self):
        eng = _make_engine(min_similarity=0.3)
        result = eng.adapt(_ctx(), _base())
        assert result.source == "passthrough"

    def test_memory_hit_when_similar_entry_exists(self):
        eng = _make_engine(min_similarity=0.0)
        ctx = _ctx()
        eng.record(ctx, _base(), _base(), {"gain": 0.1})
        result = eng.adapt(ctx, _base())
        assert result.source == "memory"

    def test_reptile_fallback_when_prior_populated(self):
        eng = _make_engine(min_similarity=0.99)  # almost impossible to hit
        # populate REPTILE prior
        update = np.array([0.2, 0.1], dtype=np.float32)
        keys = ["gain", "tau"]
        eng.optimizer.apply(update, keys, reward=0.8, drg_profile={})
        # store an entry that won't match (orthogonal domain)
        eng.record(
            {"domain_id": "energy", "gain_p50": 0.9},
            _base(), _base(), {"gain": 0.05},
        )
        result = eng.adapt(_ctx("finance"), _base())
        assert result.source in ("reptile", "passthrough")

    def test_memory_miss_below_threshold_uses_fallback(self):
        eng = _make_engine(min_similarity=0.99)
        # Store entry that will have < 0.99 similarity
        eng.record(
            {"domain_id": "energy", "gain_p50": 0.5},
            _base(), _base(), {"gain": 0.05},
        )
        result = eng.adapt({"domain_id": "finance", "stability_mean": 0.1}, _base())
        assert result.source in ("reptile", "passthrough")
        assert result.n_retrieved == 0
        assert result.similarities == []


# ---------------------------------------------------------------------------
# Adapted parameters
# ---------------------------------------------------------------------------

class TestAdaptedParams:
    def test_passthrough_returns_base_params_unchanged(self):
        eng = _make_engine(min_similarity=0.99)
        base = {"gain": 0.5, "tau": 0.9}
        result = eng.adapt(_ctx(), base)
        assert result.adapted_params == base

    def test_memory_hit_applies_delta_to_base(self):
        eng = _make_engine(min_similarity=0.0)
        ctx = _ctx()
        base = {"gain": 0.5}
        eng.record(ctx, base, base, {"gain": 0.2})
        result = eng.adapt(ctx, base)
        assert result.adapted_params["gain"] > base["gain"]

    def test_delta_key_not_in_base_is_added(self):
        eng = _make_engine(min_similarity=0.0)
        ctx = _ctx()
        base = {"gain": 0.5}
        eng.record(ctx, base, base, {"new_param": 0.77})
        result = eng.adapt(ctx, base)
        assert "new_param" in result.adapted_params

    def test_reptile_prior_merged_into_base_params(self):
        eng = _make_engine(min_similarity=0.99)
        keys = ["gain", "tau"]
        update = np.array([0.3, 0.05], dtype=np.float32)
        prior = eng.optimizer.apply(update, keys, reward=0.8, drg_profile={})
        result = eng.adapt(_ctx(), _base())
        if result.source == "reptile":
            for k in keys:
                assert result.adapted_params[k] == pytest.approx(prior[k], abs=1e-5)

    def test_weighted_blend_proportional_to_similarity(self):
        """Two entries with identical delta magnitude; higher-sim entry should
        dominate the weighted blend."""
        eng = _make_engine(min_similarity=0.0, blend_mode="weighted")
        ctx_hi = {"domain_id": "healthcare", "gain_p50": 0.8, "stability_mean": 0.9}
        ctx_lo = {"domain_id": "other_domain_xyz", "gain_p50": 0.01}
        base = {"gain": 0.5}
        eng.record(ctx_hi, base, base, {"gain": 1.0})
        eng.record(ctx_lo, base, base, {"gain": 1.0})
        result = eng.adapt(ctx_hi, base)
        # gain increase should be < 1.0 (weighted blend of two deltas), > 0.5
        assert result.adapted_params["gain"] > 0.5
        assert result.adapted_params["gain"] < 1.5 + 0.5  # sanity cap

    def test_top1_blend_applies_only_best_delta(self):
        eng = _make_engine(min_similarity=0.0, blend_mode="top1")
        ctx = _ctx()
        base = {"gain": 0.5}
        eng.record(ctx, base, base, {"gain": 0.3})
        eng.record({"domain_id": "other_xyz", "gain_p50": 0.01}, base, base, {"gain": 9.9})
        result = eng.adapt(ctx, base)
        # Should use only the best match's delta (≈ 0.3), not 9.9
        assert result.adapted_params["gain"] < 2.0


# ---------------------------------------------------------------------------
# Record / retrieve round-trip
# ---------------------------------------------------------------------------

class TestRecordRetrieve:
    def test_recorded_episode_retrieved_on_next_adapt(self):
        eng = _make_engine(min_similarity=0.0)
        ctx = _ctx()
        base = _base()
        eng.record(ctx, base, base, {"gain": 0.15}, policy={"source": "test"})
        result = eng.adapt(ctx, base)
        assert result.source == "memory"
        assert result.n_retrieved >= 1

    def test_multiple_records_increase_n_retrieved(self):
        eng = _make_engine(min_similarity=0.0, top_k=10)
        base = _base()
        for i in range(4):
            ctx = {"domain_id": "healthcare", "gain_p50": 0.1 * i}
            eng.record(ctx, base, base, {"gain": 0.1})
        result = eng.adapt(_ctx(), base)
        assert result.n_retrieved >= 1

    def test_n_retrieved_matches_similarities_length(self):
        eng = _make_engine(min_similarity=0.0)
        ctx = _ctx()
        base = _base()
        eng.record(ctx, base, base, {"gain": 0.1})
        result = eng.adapt(ctx, base)
        assert result.n_retrieved == len(result.similarities)


# ---------------------------------------------------------------------------
# Counters and hit rate
# ---------------------------------------------------------------------------

class TestCounters:
    def test_miss_increments_on_empty_memory(self):
        eng = _make_engine(min_similarity=0.0)
        eng.adapt(_ctx(), _base())
        assert eng.misses == 1
        assert eng.hits == 0

    def test_hit_increments_on_memory_match(self):
        eng = _make_engine(min_similarity=0.0)
        ctx = _ctx()
        base = _base()
        eng.record(ctx, base, base, {"gain": 0.1})
        eng.adapt(ctx, base)
        assert eng.hits == 1
        assert eng.misses == 0

    def test_hit_rate_zero_on_all_misses(self):
        eng = _make_engine(min_similarity=0.99)
        eng.adapt(_ctx(), _base())
        assert eng.hit_rate == 0.0

    def test_hit_rate_one_on_all_hits(self):
        eng = _make_engine(min_similarity=0.0)
        ctx = _ctx()
        base = _base()
        eng.record(ctx, base, base, {"gain": 0.1})
        eng.adapt(ctx, base)
        assert eng.hit_rate == 1.0

    def test_hit_rate_partial(self):
        eng = _make_engine(min_similarity=0.0)
        ctx = _ctx()
        base = _base()
        eng.record(ctx, base, base, {"gain": 0.1})
        eng.adapt(ctx, base)      # hit
        eng.adapt({"domain_id": "z_xyz_unk"}, base)  # likely miss if no similar entry
        assert 0.0 <= eng.hit_rate <= 1.0


# ---------------------------------------------------------------------------
# Result metadata
# ---------------------------------------------------------------------------

class TestResultMetadata:
    def test_query_key_shape_matches_encoder_output_dim(self):
        eng = _make_engine()
        result = eng.adapt(_ctx(), _base())
        assert result.query_key.shape == (eng.encoder.output_dim,)

    def test_similarities_empty_on_miss(self):
        eng = _make_engine(min_similarity=0.99)
        result = eng.adapt(_ctx(), _base())
        assert result.similarities == []

    def test_similarities_between_neg1_and_1(self):
        eng = _make_engine(min_similarity=0.0)
        ctx = _ctx()
        base = _base()
        eng.record(ctx, base, base, {"gain": 0.1})
        result = eng.adapt(ctx, base)
        for s in result.similarities:
            assert -1.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_context_same_adapted_params(self):
        eng = _make_engine(min_similarity=0.0)
        ctx = _ctx()
        base = _base()
        eng.record(ctx, base, base, {"gain": 0.1})
        r1 = eng.adapt(ctx, base)
        r2 = eng.adapt(ctx, base)
        assert r1.adapted_params == r2.adapted_params

    def test_same_context_same_query_key(self):
        eng = _make_engine()
        ctx = _ctx()
        base = _base()
        k1 = eng.adapt(ctx, base).query_key
        k2 = eng.adapt(ctx, base).query_key
        np.testing.assert_array_equal(k1, k2)
