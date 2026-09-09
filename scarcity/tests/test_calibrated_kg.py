"""The streaming engine's calibrated knowledge graph controls false positives.

The default (raw-confidence) knowledge graph emits null edges at a high rate; the
calibrated graph (analytic F-test p-value + Benjamini-Hochberg) must keep true
edges while driving the null false-edge rate to ~q.
"""
import numpy as np
import pytest

from scarcity.engine.gpu_engine import GPUDiscoveryEngine


def _stream(seed, n=800):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 0.9 * x + 0.2 * rng.normal(size=n)   # true edge x-y
    z = rng.normal(size=n)                    # z independent -> x-z, y-z are null
    return [{"x": float(x[t]), "y": float(y[t]), "z": float(z[t])} for t in range(n)]


def _run(seed):
    e = GPUDiscoveryEngine(device="cpu")
    e.initialize_v2({"fields": [{"name": c} for c in ("x", "y", "z")]}, use_causal=True)
    for r in _stream(seed):
        e.process_row(r)
    return e


def _pairs(kg):
    return {frozenset(h["variables"]) for h in kg if len(h["variables"]) == 2}


def test_calibrated_kg_keeps_true_edge_and_controls_null_fpr():
    xy = frozenset(("x", "y"))
    nulls = [frozenset(("x", "z")), frozenset(("y", "z"))]
    raw_fp = cal_fp = tp = 0
    seeds = range(6)
    for s in seeds:
        e = _run(s)
        raw = _pairs([h for h in e.get_knowledge_graph(top_k=200)
                      if h["metrics"]["confidence"] >= 0.55])
        cal = _pairs(e.get_knowledge_graph(top_k=200, calibrated=True, q=0.05))
        assert xy in cal                       # true edge always retained
        tp += 1
        raw_fp += sum(p in raw for p in nulls)
        cal_fp += sum(p in cal for p in nulls)
    n_null = 2 * len(list(seeds))
    assert cal_fp <= 0.05 * n_null + 1          # calibrated controls FPR near q
    assert cal_fp < raw_fp                       # and strictly improves on raw


def test_calibrated_edges_carry_significance_annotations():
    e = _run(0)
    kg = e.get_knowledge_graph(top_k=200, calibrated=True, q=0.05)
    assert kg, "calibrated graph should not be empty on a strong true edge"
    for h in kg:
        m = h["metrics"]
        assert "p_value" in m and "q_value" in m and m.get("significant") is True
        assert 0.0 <= m["p_value"] <= 1.0
        assert m["q_value"] <= 0.05 + 1e-9       # survivors are below the BH threshold


def test_default_graph_unchanged():
    """calibrated=False keeps the legacy raw-confidence behaviour (no p-values)."""
    e = _run(0)
    kg = e.get_knowledge_graph(top_k=50)
    assert kg and "p_value" not in kg[0]["metrics"]


def _ar1(rng, n, phi=0.7, scale=1.0):
    v = np.zeros(n)
    for t in range(1, n):
        v[t] = phi * v[t - 1] + rng.normal(scale=scale)
    return v


def test_calibrated_gate_is_autocorrelation_robust():
    """Two INDEPENDENT autocorrelated (AR1) series must not be linked: the gate
    scores the predictor coefficient, not the whole-regression R² (which the
    target's own autoregressive term would inflate). The true contemporaneous edge
    is still recovered."""
    fp = 0
    seeds = range(6)
    for s in seeds:
        rng = np.random.default_rng(100 + s)
        n = 900
        x = _ar1(rng, n, 0.7)
        z = _ar1(rng, n, 0.7)                 # independent of x, but autocorrelated
        y = 0.9 * x + 0.3 * rng.normal(size=n)  # true edge x-y
        e = GPUDiscoveryEngine(device="cpu")
        e.initialize_v2({"fields": [{"name": c} for c in ("x", "y", "z")]}, use_causal=True)
        for t in range(n):
            e.process_row({"x": float(x[t]), "y": float(y[t]), "z": float(z[t])})
        cal = _pairs(e.get_knowledge_graph(top_k=200, calibrated=True, q=0.05))
        assert frozenset(("x", "y")) in cal           # true edge kept
        fp += sum(p in cal for p in (frozenset(("x", "z")), frozenset(("y", "z"))))
    # x-z, y-z are independent AR series; the gate must not spuriously link them
    assert fp <= 0.05 * (2 * len(list(seeds))) + 1, f"autocorrelation false links: {fp}"
