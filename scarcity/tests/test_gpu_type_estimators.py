"""GPU batched type-specific estimators (Step 2 of the GPU port).

Each relationship type must be evaluated with its own statistic in the GPU
backend, not a generic goodness-of-fit. These tests validate the ported
estimators by construction: the estimator fires on data that contains the
relationship and stays quiet on data that does not.

The GPU engine runs on device='cpu' (torch, no CUDA needed).
"""
import numpy as np
import pytest

pytest.importorskip("torch")

from scarcity.engine.gpu_engine import GPUDiscoveryEngine


def _max_conf(kind_rows, rel_type, cols=("a", "b", "y")):
    rows, n = kind_rows
    schema = {"fields": [{"name": c} for c in cols]}
    e = GPUDiscoveryEngine(device="cpu")
    e.initialize_v2(schema, use_causal=True)
    for r in rows:
        e.process_row(r)
    return max(
        (h["metrics"]["confidence"] for h in e.get_knowledge_graph() if h["type"] == rel_type),
        default=0.0,
    )


def _triple(fn, n=500, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    y = fn(a, b, rng)
    return [{"a": float(a[i]), "b": float(b[i]), "y": float(y[i])} for i in range(n)], n


# --- synergistic / moderating: the interaction term a*b matters ---------------

def test_synergistic_fires_on_real_interaction():
    rows = _triple(lambda a, b, rng: a * b + 0.1 * rng.normal(size=len(a)))
    assert _max_conf(rows, "synergistic") > 0.55


def test_synergistic_quiet_on_additive_only():
    rows = _triple(lambda a, b, rng: a + b + 0.1 * rng.normal(size=len(a)))
    assert _max_conf(rows, "synergistic") < 0.55


def test_moderating_fires_on_real_interaction():
    # Moderating shares the interaction mechanism: b moderates a's effect on y.
    rows = _triple(lambda a, b, rng: a * b + 0.1 * rng.normal(size=len(a)))
    assert _max_conf(rows, "moderating") > 0.55


def test_moderating_quiet_without_interaction():
    rows = _triple(lambda a, b, rng: a + b + 0.1 * rng.normal(size=len(a)))
    assert _max_conf(rows, "moderating") < 0.55
