"""
Tests for ContextEncoder.

Contracts verified:
- Output shape matches output_dim
- Deterministic: same input → same output always
- Domain identity: different domain_ids produce different embeddings
- Domain identity: same domain_id always produces same embedding region
- Named slots: known keys land in the correct index and direction
- Overflow: unknown numeric keys contribute signal to overflow region
- Unknown non-numeric keys are silently ignored
- Missing keys produce zeros in their named slot
- L2-normalized output has unit norm (when normalize=True)
- Batch encoding matches individual encoding
- Config with custom dims produces correct output_dim
"""

import numpy as np
import pytest

from scarcity.meta.encoder import (
    ContextEncoder,
    ContextEncoderConfig,
    _NAMED_DIM,
    _SLOT_INDEX,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def enc() -> ContextEncoder:
    return ContextEncoder()


@pytest.fixture
def basic_context() -> dict:
    return {
        "domain_id": "healthcare",
        "gain_p50": 0.15,
        "stability_mean": 0.8,
        "vram_high": 0.0,
        "tau": 0.91,
        "confidence": 0.6,
    }


# ---------------------------------------------------------------------------
# Shape and type
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_default_output_dim(self, enc, basic_context):
        vec = enc.encode(basic_context)
        assert vec.shape == (enc.output_dim,)

    def test_dtype_is_float32(self, enc, basic_context):
        vec = enc.encode(basic_context)
        assert vec.dtype == np.float32

    def test_custom_dims_output_dim(self, basic_context):
        cfg = ContextEncoderConfig(domain_dim=4, overflow_dim=4)
        enc = ContextEncoder(cfg)
        assert enc.output_dim == _NAMED_DIM + 4 + 4
        vec = enc.encode(basic_context)
        assert vec.shape == (enc.output_dim,)

    def test_empty_context_produces_correct_shape(self, enc):
        vec = enc.encode({})
        assert vec.shape == (enc.output_dim,)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_context_same_output(self, enc, basic_context):
        v1 = enc.encode(basic_context)
        v2 = enc.encode(basic_context)
        np.testing.assert_array_equal(v1, v2)

    def test_different_encoders_same_output(self, basic_context):
        v1 = ContextEncoder().encode(basic_context)
        v2 = ContextEncoder().encode(basic_context)
        np.testing.assert_array_equal(v1, v2)

    def test_domain_embedding_cached_equals_fresh(self, enc):
        ctx = {"domain_id": "finance"}
        v1 = enc.encode(ctx)
        enc._domain_cache.clear()
        v2 = enc.encode(ctx)
        np.testing.assert_array_equal(v1, v2)


# ---------------------------------------------------------------------------
# Domain identity
# ---------------------------------------------------------------------------

class TestDomainIdentity:
    def test_different_domains_produce_different_embeddings(self, enc):
        v_health = enc.encode({"domain_id": "healthcare"})
        v_finance = enc.encode({"domain_id": "finance"})
        assert not np.allclose(v_health, v_finance)

    def test_same_domain_same_embedding_region(self, enc):
        cfg = enc.config
        d_start = _NAMED_DIM
        d_end = _NAMED_DIM + cfg.domain_dim

        ctx_a = {"domain_id": "energy", "gain_p50": 0.1}
        ctx_b = {"domain_id": "energy", "gain_p50": 0.9}

        raw_a = ContextEncoder(ContextEncoderConfig(normalize=False)).encode(ctx_a)
        raw_b = ContextEncoder(ContextEncoderConfig(normalize=False)).encode(ctx_b)

        np.testing.assert_array_equal(raw_a[d_start:d_end], raw_b[d_start:d_end])

    def test_no_domain_id_leaves_domain_region_zero(self):
        enc = ContextEncoder(ContextEncoderConfig(normalize=False))
        vec = enc.encode({"gain_p50": 0.5})
        d_start = _NAMED_DIM
        d_end = _NAMED_DIM + enc.config.domain_dim
        np.testing.assert_array_equal(vec[d_start:d_end], np.zeros(enc.config.domain_dim))


# ---------------------------------------------------------------------------
# Named slots
# ---------------------------------------------------------------------------

class TestNamedSlots:
    def test_known_key_lands_in_correct_index(self):
        enc = ContextEncoder(ContextEncoderConfig(normalize=False))
        idx = _SLOT_INDEX["gain_p50"]
        vec = enc.encode({"gain_p50": 0.5})
        assert vec[idx] != 0.0

    def test_positive_value_produces_positive_slot(self):
        enc = ContextEncoder(ContextEncoderConfig(normalize=False))
        idx = _SLOT_INDEX["stability_mean"]
        vec = enc.encode({"stability_mean": 0.8})
        assert vec[idx] > 0.0

    def test_negative_value_produces_negative_slot(self):
        enc = ContextEncoder(ContextEncoderConfig(normalize=False))
        idx = _SLOT_INDEX["score_delta"]
        vec = enc.encode({"score_delta": -0.3})
        assert vec[idx] < 0.0

    def test_larger_value_produces_larger_slot_magnitude(self):
        enc = ContextEncoder(ContextEncoderConfig(normalize=False))
        idx = _SLOT_INDEX["confidence"]
        low = enc.encode({"confidence": 0.1})[idx]
        high = enc.encode({"confidence": 0.9})[idx]
        assert high > low

    def test_missing_key_slot_is_zero(self):
        enc = ContextEncoder(ContextEncoderConfig(normalize=False))
        idx = _SLOT_INDEX["tau"]
        vec = enc.encode({"gain_p50": 0.5})
        assert vec[idx] == 0.0

    def test_latency_ms_uses_scale(self):
        enc = ContextEncoder(ContextEncoderConfig(normalize=False))
        idx = _SLOT_INDEX["latency_ms"]
        low = enc.encode({"latency_ms": 100.0})[idx]
        high = enc.encode({"latency_ms": 2000.0})[idx]
        assert high > low
        # Both should be within tanh range
        assert -1.0 < low < 1.0
        assert -1.0 < high < 1.0


# ---------------------------------------------------------------------------
# Overflow region
# ---------------------------------------------------------------------------

class TestOverflow:
    def test_unknown_numeric_key_contributes_to_overflow(self):
        enc = ContextEncoder(ContextEncoderConfig(normalize=False))
        o_start = _NAMED_DIM + enc.config.domain_dim
        with_unknown = enc.encode({"unknown_metric": 1.0})
        without_unknown = enc.encode({})
        assert not np.allclose(with_unknown[o_start:], without_unknown[o_start:])

    def test_unknown_string_value_is_ignored(self):
        enc = ContextEncoder(ContextEncoderConfig(normalize=False))
        v1 = enc.encode({"gain_p50": 0.5})
        v2 = enc.encode({"gain_p50": 0.5, "label": "some_string"})
        np.testing.assert_array_equal(v1, v2)

    def test_domain_id_does_not_contribute_to_overflow(self):
        enc = ContextEncoder(ContextEncoderConfig(normalize=False))
        o_start = _NAMED_DIM + enc.config.domain_dim
        v1 = enc.encode({})
        v2 = enc.encode({"domain_id": "healthcare"})
        np.testing.assert_array_equal(v1[o_start:], v2[o_start:])


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

class TestNormalization:
    def test_normalized_output_has_unit_norm(self, enc, basic_context):
        vec = enc.encode(basic_context)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5

    def test_unnormalized_output_not_necessarily_unit(self, basic_context):
        enc = ContextEncoder(ContextEncoderConfig(normalize=False))
        vec = enc.encode(basic_context)
        # should not be unit norm in general
        assert abs(np.linalg.norm(vec) - 1.0) > 1e-3

    def test_empty_context_with_normalization_does_not_crash(self, enc):
        vec = enc.encode({})
        assert np.all(np.isfinite(vec))


# ---------------------------------------------------------------------------
# Batch encoding
# ---------------------------------------------------------------------------

class TestBatchEncoding:
    def test_batch_shape(self, enc, basic_context):
        contexts = [basic_context, {"domain_id": "finance"}, {}]
        batch = enc.encode_batch(contexts)
        assert batch.shape == (3, enc.output_dim)

    def test_batch_matches_individual(self, enc, basic_context):
        ctx2 = {"domain_id": "energy", "gain_p50": 0.3}
        batch = enc.encode_batch([basic_context, ctx2])
        np.testing.assert_array_equal(batch[0], enc.encode(basic_context))
        np.testing.assert_array_equal(batch[1], enc.encode(ctx2))


# ---------------------------------------------------------------------------
# named_slots helper
# ---------------------------------------------------------------------------

class TestNamedSlotsHelper:
    def test_named_slots_returns_list_of_strings(self, enc):
        slots = enc.named_slots()
        assert isinstance(slots, list)
        assert all(isinstance(s, str) for s in slots)

    def test_named_slots_length_matches_named_dim(self, enc):
        assert len(enc.named_slots()) == _NAMED_DIM
