"""The engine's public views must reflect the backend that holds the evidence.

Regression for a real landmine: with the default tensor backend (vectorized=True
when torch is present) process_row feeds the internal GPUDiscoveryEngine and the
Python hypothesis pool is never updated. get_knowledge_graph()/get_candidate_paths()
used to read that frozen pool, returning a stale graph (every confidence 0.5, no
true edge). They must delegate to the tensor backend instead.
"""
import numpy as np
import pytest

from scarcity.engine.engine_v2 import OnlineDiscoveryEngine


def _stream(engine, seed=0, n=600):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 0.9 * x + 0.2 * rng.normal(size=n)
    z = rng.normal(size=n)
    engine.initialize_v2({"fields": [{"name": c} for c in ("x", "y", "z")]}, use_causal=True)
    for t in range(n):
        engine.process_row({"x": float(x[t]), "y": float(y[t]), "z": float(z[t])})


@pytest.mark.parametrize("vectorized", [True, False])
def test_knowledge_graph_recovers_true_edge_on_both_backends(vectorized):
    e = OnlineDiscoveryEngine(vectorized=vectorized)
    _stream(e)
    kg = e.get_knowledge_graph()
    xy = [h for h in kg if set(h["variables"]) == {"x", "y"}
          and h["metrics"]["confidence"] > 0.55]
    assert xy, f"true x-y edge not recovered (vectorized={e.vectorized}); stale pool?"


def test_default_engine_is_not_stale():
    """The default constructor (tensor backend when torch present) must not return
    a graph frozen at initialization (the bug: every confidence == 0.5)."""
    e = OnlineDiscoveryEngine()
    _stream(e)
    kg = e.get_knowledge_graph()
    confs = [h["metrics"]["confidence"] for h in kg]
    assert not all(abs(c - 0.5) < 1e-9 for c in confs), "graph is frozen at init (stale pool)"


def test_calibrated_graph_exposed_through_cpu_api_when_vectorized():
    e = OnlineDiscoveryEngine(vectorized=True)
    if not e.vectorized:
        pytest.skip("torch not available; no tensor backend")
    _stream(e)
    kg = e.get_knowledge_graph(calibrated=True, q=0.05)
    assert all("p_value" in h["metrics"] for h in kg)


def test_candidate_paths_nonempty_under_tensor_backend():
    e = OnlineDiscoveryEngine(vectorized=True)
    if not e.vectorized:
        pytest.skip("torch not available; no tensor backend")
    _stream(e)
    cands = e.get_candidate_paths(top_k=30)
    assert len(cands) > 0, "candidate paths empty under tensor backend (stale pool?)"
