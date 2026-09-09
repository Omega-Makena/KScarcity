"""Reproducibility (#5): a seeded run is bit-reproducible, and set_seeds enforces
deterministic kernels (so the guarantee holds on GPU, not just CPU)."""
import numpy as np
import pytest

from scarcity.experiment.runner import set_seeds
from scarcity.engine.gpu_engine import GPUDiscoveryEngine


def test_set_seeds_enables_deterministic_algorithms():
    import torch
    set_seeds(0)
    assert torch.are_deterministic_algorithms_enabled()
    import os
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG")  # set for deterministic cuBLAS


def _run(seed_data=0, n=1200):
    rng = np.random.default_rng(seed_data)
    x = rng.normal(size=n)
    y = 0.9 * x + 0.3 * rng.normal(size=n)
    z = rng.normal(size=n)
    e = GPUDiscoveryEngine(device="cpu")
    e.initialize_v2({"fields": [{"name": c} for c in ("x", "y", "z")]}, use_causal=True)
    for t in range(n):
        e.process_row({"x": float(x[t]), "y": float(y[t]), "z": float(z[t])})
    return e


def test_engine_state_is_bit_reproducible():
    set_seeds(0)
    a = _run()
    pa, r2a = a._coef_stats()
    confa = a.get_hyp_metrics()[0]

    set_seeds(0)
    b = _run()
    pb, r2b = b._coef_stats()
    confb = b.get_hyp_metrics()[0]

    assert np.array_equal(pa, pb, equal_nan=True)
    assert np.array_equal(r2a, r2b, equal_nan=True)
    assert np.array_equal(confa, confb, equal_nan=True)


def test_calibrated_graph_is_reproducible():
    set_seeds(0)
    kg1 = _run().get_knowledge_graph(top_k=200, calibrated=True, q=0.05)
    set_seeds(0)
    kg2 = _run().get_knowledge_graph(top_k=200, calibrated=True, q=0.05)
    sig1 = sorted((h["type"], tuple(sorted(h["variables"])),
                   round(h["metrics"]["p_value"], 12)) for h in kg1)
    sig2 = sorted((h["type"], tuple(sorted(h["variables"])),
                   round(h["metrics"]["p_value"], 12)) for h in kg2)
    assert sig1 == sig2
