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


# --- competitive: a substitute (significant negative) relationship ------------

def _pair(fn, n=400, seed=1):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = fn(a, rng)
    return [{"a": float(a[i]), "b": float(b[i])} for i in range(n)], n


def test_competitive_fires_on_substitute():
    rows = _pair(lambda a, rng: -a + 0.1 * rng.normal(size=len(a)))
    assert _max_conf(rows, "competitive", cols=("a", "b")) > 0.55


def test_competitive_quiet_on_complement():
    # A positive (complementary) relationship is not competitive.
    rows = _pair(lambda a, rng: a + 0.1 * rng.normal(size=len(a)))
    assert _max_conf(rows, "competitive", cols=("a", "b")) < 0.55


# --- mediating: a significant indirect path a -> b -> c (Sobel) ---------------

def _chain(kind, n=500, seed=2):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    if kind == "mediation":
        b = a + 0.3 * rng.normal(size=n)
        c = b + 0.3 * rng.normal(size=n)
    elif kind == "direct":            # c depends on a directly, not through b
        b = rng.normal(size=n)
        c = a + 0.3 * rng.normal(size=n)
    else:                             # independent
        b = rng.normal(size=n)
        c = rng.normal(size=n)
    return [{"a": float(a[i]), "b": float(b[i]), "c": float(c[i])} for i in range(n)], n


def test_mediating_fires_on_real_chain():
    assert _max_conf(_chain("mediation"), "mediating", cols=("a", "b", "c")) > 0.55


def test_mediating_quiet_when_independent():
    assert _max_conf(_chain("independent"), "mediating", cols=("a", "b", "c")) < 0.55


def test_mediating_quiet_on_direct_effect():
    # A direct a->c effect with no a->b path is not mediation.
    assert _max_conf(_chain("direct"), "mediating", cols=("a", "b", "c")) < 0.55


# --- equilibrium: a mean-reverting (stationary) series, ADF-style -------------

def _series(phi, n=600, seed=3):
    rng = np.random.default_rng(seed)
    v = np.zeros(n)
    for t in range(1, n):
        v[t] = phi * v[t - 1] + rng.normal()
    w = rng.normal(size=n)
    rows = [{"v": float(v[i]), "w": float(w[i])} for i in range(n)]
    return rows, n


def _eq_conf(rows_n):
    rows, n = rows_n
    e = GPUDiscoveryEngine(device="cpu")
    e.initialize_v2({"fields": [{"name": "v"}, {"name": "w"}]}, use_causal=True)
    for r in rows:
        e.process_row(r)
    return max(
        (h["metrics"]["confidence"] for h in e.get_knowledge_graph()
         if h["type"] == "equilibrium" and "v" in h["variables"]),
        default=0.0,
    )


def test_equilibrium_fires_on_mean_reversion():
    assert _eq_conf(_series(0.5)) > 0.55        # phi < 1: stationary


def test_equilibrium_quiet_on_random_walk():
    assert _eq_conf(_series(1.0)) < 0.55        # phi ~ 1: unit root, not equilibrium
