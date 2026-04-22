"""
Tests for EpisodicMemory.

Contracts verified:
- Empty memory returns no results
- Stored entry is retrievable
- Retrieval is ordered by similarity descending
- Tie-breaking by recency (higher timestamp first)
- top_k limits results
- min_similarity threshold filters results
- Per-call overrides for top_k and min_similarity work
- Capacity eviction is FIFO (oldest entry dropped)
- clear() resets state completely
- Thread safety: concurrent stores do not corrupt the buffer
- keys_matrix() and timestamps() return consistent snapshots
- Zero-norm query returns empty list (no crash)
- Zero-norm stored key is excluded from results
- Value, context, delta, policy are preserved faithfully
"""

import threading
import time

import numpy as np
import pytest

from scarcity.meta.memory import (
    EpisodicMemory,
    EpisodicMemoryConfig,
    EpisodicEntry,
    RetrievalResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v: list) -> np.ndarray:
    a = np.array(v, dtype=np.float32)
    n = np.linalg.norm(a)
    return (a / n) if n > 1e-8 else a


def _entry_kwargs(key: np.ndarray, tag: str = "x") -> dict:
    return dict(
        key=key,
        value={"param": tag},
        context={"domain_id": tag},
        delta={"gain": 0.1},
        policy={"source": tag},
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mem() -> EpisodicMemory:
    return EpisodicMemory(EpisodicMemoryConfig(capacity=8, top_k=3, min_similarity=0.0))


# ---------------------------------------------------------------------------
# Basic store / retrieve
# ---------------------------------------------------------------------------

class TestStoreRetrieve:
    def test_empty_returns_empty(self, mem):
        q = _unit([1.0, 0.0, 0.0])
        assert mem.retrieve(q) == []

    def test_single_entry_retrieved(self, mem):
        k = _unit([1.0, 0.0, 0.0])
        mem.store(**_entry_kwargs(k, "a"))
        results = mem.retrieve(k)
        assert len(results) == 1
        assert results[0].entry.value == {"param": "a"}

    def test_similarity_is_close_to_one_for_identical_key(self, mem):
        k = _unit([1.0, 2.0, 3.0])
        mem.store(**_entry_kwargs(k))
        r = mem.retrieve(k)[0]
        assert abs(r.similarity - 1.0) < 1e-5

    def test_payload_fields_preserved(self, mem):
        k = _unit([0.5, 0.5, 0.0])
        mem.store(
            key=k,
            value={"w": [1.0, 2.0]},
            context={"domain_id": "energy", "tau": 0.9},
            delta={"gain": 0.3, "stability": 0.05},
            policy={"source": "domain_server", "step": 42},
        )
        r = mem.retrieve(k)[0]
        assert r.entry.value == {"w": [1.0, 2.0]}
        assert r.entry.context == {"domain_id": "energy", "tau": 0.9}
        assert r.entry.delta == {"gain": 0.3, "stability": 0.05}
        assert r.entry.policy == {"source": "domain_server", "step": 42}


# ---------------------------------------------------------------------------
# Ordering and ranking
# ---------------------------------------------------------------------------

class TestOrdering:
    def test_results_ordered_by_similarity_descending(self, mem):
        q = _unit([1.0, 0.0, 0.0])
        for v in [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.6, 0.8, 0.0]]:
            mem.store(**_entry_kwargs(_unit(v), str(v)))
        results = mem.retrieve(q)
        sims = [r.similarity for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_rank_field_matches_position(self, mem):
        q = _unit([1.0, 0.0, 0.0])
        for v in [[1.0, 0.0, 0.0], [0.7, 0.7, 0.0]]:
            mem.store(**_entry_kwargs(_unit(v)))
        results = mem.retrieve(q)
        for i, r in enumerate(results):
            assert r.rank == i

    def test_tie_broken_by_recency(self, mem):
        """Two entries with identical keys: the newer one should come first."""
        k = _unit([1.0, 0.0, 0.0])
        mem.store(**_entry_kwargs(k, "old"))
        mem.store(**_entry_kwargs(k, "new"))
        results = mem.retrieve(k)
        assert results[0].entry.value == {"param": "new"}


# ---------------------------------------------------------------------------
# top_k and min_similarity
# ---------------------------------------------------------------------------

class TestFiltering:
    def test_top_k_limits_results(self, mem):
        q = _unit([1.0, 0.0, 0.0])
        for i in range(6):
            mem.store(**_entry_kwargs(_unit([1.0, float(i) * 0.1, 0.0])))
        results = mem.retrieve(q, top_k=2)
        assert len(results) <= 2

    def test_top_k_override_per_call(self, mem):
        q = _unit([1.0, 0.0, 0.0])
        for i in range(5):
            mem.store(**_entry_kwargs(_unit([1.0, 0.0, 0.0])))
        assert len(mem.retrieve(q, top_k=2)) == 2
        assert len(mem.retrieve(q, top_k=4)) == 4

    def test_min_similarity_filters_low_matches(self, mem):
        q = _unit([1.0, 0.0, 0.0])
        mem.store(**_entry_kwargs(_unit([1.0, 0.0, 0.0])))    # sim ≈ 1.0
        mem.store(**_entry_kwargs(_unit([0.0, 1.0, 0.0])))    # sim ≈ 0.0
        results = mem.retrieve(q, min_similarity=0.5)
        assert all(r.similarity >= 0.5 for r in results)

    def test_min_similarity_zero_returns_all_within_top_k(self, mem):
        q = _unit([1.0, 0.0, 0.0])
        for v in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]:
            mem.store(**_entry_kwargs(_unit(v)))
        results = mem.retrieve(q, top_k=10, min_similarity=0.0)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Capacity eviction (FIFO)
# ---------------------------------------------------------------------------

class TestCapacity:
    def test_len_never_exceeds_capacity(self):
        mem = EpisodicMemory(EpisodicMemoryConfig(capacity=4))
        k = _unit([1.0, 0.0, 0.0])
        for _ in range(10):
            mem.store(**_entry_kwargs(k))
        assert len(mem) <= 4

    def test_oldest_entry_evicted_first(self):
        mem = EpisodicMemory(EpisodicMemoryConfig(capacity=3, top_k=10))
        keys = [_unit([1.0, 0.0, 0.0]) for _ in range(4)]
        mem.store(**_entry_kwargs(keys[0], "first"))
        mem.store(**_entry_kwargs(keys[1], "second"))
        mem.store(**_entry_kwargs(keys[2], "third"))
        mem.store(**_entry_kwargs(keys[3], "fourth"))  # evicts "first"

        q = _unit([1.0, 0.0, 0.0])
        values = [r.entry.value["param"] for r in mem.retrieve(q, top_k=10)]
        assert "first" not in values
        assert "fourth" in values

    def test_config_capacity_one_stores_single_entry(self):
        mem = EpisodicMemory(EpisodicMemoryConfig(capacity=1, top_k=5))
        k1 = _unit([1.0, 0.0, 0.0])
        k2 = _unit([0.0, 1.0, 0.0])
        mem.store(**_entry_kwargs(k1, "a"))
        mem.store(**_entry_kwargs(k2, "b"))
        assert len(mem) == 1
        results = mem.retrieve(k2, top_k=5)
        assert results[0].entry.value == {"param": "b"}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_norm_query_returns_empty(self, mem):
        k = _unit([1.0, 0.0, 0.0])
        mem.store(**_entry_kwargs(k))
        q = np.zeros(3, dtype=np.float32)
        assert mem.retrieve(q) == []

    def test_zero_norm_stored_key_excluded(self, mem):
        zero_key = np.zeros(3, dtype=np.float32)
        good_key = _unit([1.0, 0.0, 0.0])
        mem.store(**_entry_kwargs(zero_key, "zero"))
        mem.store(**_entry_kwargs(good_key, "good"))
        results = mem.retrieve(good_key, top_k=10)
        values = [r.entry.value["param"] for r in results]
        assert "zero" not in values
        assert "good" in values

    def test_clear_resets_state(self, mem):
        k = _unit([1.0, 0.0, 0.0])
        mem.store(**_entry_kwargs(k))
        mem.clear()
        assert len(mem) == 0
        assert mem.retrieve(k) == []

    def test_clear_resets_timestamp_counter(self):
        mem = EpisodicMemory(EpisodicMemoryConfig(capacity=4))
        k = _unit([1.0, 0.0, 0.0])
        mem.store(**_entry_kwargs(k))
        mem.clear()
        mem.store(**_entry_kwargs(k, "after_clear"))
        assert mem.timestamps() == [0]

    def test_invalid_capacity_raises(self):
        with pytest.raises(ValueError):
            EpisodicMemory(EpisodicMemoryConfig(capacity=0))

    def test_invalid_top_k_raises(self):
        with pytest.raises(ValueError):
            EpisodicMemory(EpisodicMemoryConfig(top_k=0))


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------

class TestInspection:
    def test_len_matches_stored_count(self, mem):
        k = _unit([1.0, 0.0, 0.0])
        for i in range(5):
            mem.store(**_entry_kwargs(k, str(i)))
        assert len(mem) == 5

    def test_keys_matrix_shape(self, mem):
        for v in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]:
            mem.store(**_entry_kwargs(_unit(v)))
        km = mem.keys_matrix()
        assert km.shape == (2, 3)
        assert km.dtype == np.float32

    def test_keys_matrix_empty_when_no_entries(self, mem):
        km = mem.keys_matrix()
        assert len(km) == 0

    def test_timestamps_monotonic(self, mem):
        k = _unit([1.0, 0.0, 0.0])
        for _ in range(4):
            mem.store(**_entry_kwargs(k))
        ts = mem.timestamps()
        assert ts == sorted(ts)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_stores_do_not_corrupt(self):
        mem = EpisodicMemory(EpisodicMemoryConfig(capacity=128, top_k=10))
        k = _unit([1.0, 0.0, 0.0])
        errors = []

        def writer():
            try:
                for _ in range(50):
                    mem.store(**_entry_kwargs(k))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent writes: {errors}"
        assert len(mem) <= mem.capacity

    def test_concurrent_store_and_retrieve(self):
        mem = EpisodicMemory(EpisodicMemoryConfig(capacity=64, top_k=5))
        k = _unit([1.0, 0.0, 0.0])
        errors = []

        def writer():
            try:
                for _ in range(30):
                    mem.store(**_entry_kwargs(k))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(30):
                    mem.retrieve(k)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(2)]
        threads += [threading.Thread(target=reader) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent access: {errors}"
