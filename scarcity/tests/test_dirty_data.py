"""Dirty-data robustness tests — the engine must ingest corrupted streams without
crashing and still recover a strong true edge under light corruption.

Complements benchmark/scripts/dirty_data.py (which quantifies degradation across
severities); these are the fast always-on guards.
"""
import numpy as np
import pytest

from scarcity.engine.gpu_engine import GPUDiscoveryEngine
from scarcity.engine.engine_v2 import OnlineDiscoveryEngine

SCHEMA = {"fields": [{"name": c} for c in ("x", "y", "z")]}
ENGINES = [lambda: GPUDiscoveryEngine(device="cpu"), OnlineDiscoveryEngine]


def _run(engine_factory, rows):
    e = engine_factory()
    e.initialize_v2(SCHEMA, use_causal=True)
    for r in rows:
        e.process_row(r)
    return e


@pytest.mark.parametrize("engine_factory", ENGINES)
@pytest.mark.parametrize("kind", ["all_nan", "missing_keys", "extra_key", "inf_huge",
                                  "empty_rows", "mixed"])
def test_no_crash_on_dirty_input(engine_factory, kind):
    """Every corruption must be ingested without raising."""
    rng = np.random.default_rng(0)
    n = 80
    if kind == "all_nan":
        rows = [{"x": float("nan"), "y": 1.0, "z": 2.0} for _ in range(n)]
    elif kind == "missing_keys":
        rows = [{"x": float(rng.normal())} for _ in range(n)]          # y, z absent
    elif kind == "extra_key":
        rows = [{"x": 1.0, "y": 2.0, "z": 3.0, "w": 4.0} for _ in range(n)]
    elif kind == "inf_huge":
        rows = [{"x": float("inf"), "y": 1e300, "z": -1e300} for _ in range(n)]
    elif kind == "empty_rows":
        rows = [{} for _ in range(n)]
    else:  # mixed: a bit of everything
        rows = []
        for t in range(n):
            r = {"x": float(rng.normal()), "y": float(rng.normal()), "z": float(rng.normal())}
            if t % 5 == 0:
                r["y"] = float("nan")
            if t % 7 == 0:
                r.pop("z", None)
            if t % 11 == 0:
                r["w"] = 1.0
            rows.append(r)
    _run(engine_factory, rows)   # must not raise


def _edge(engine, a, b):
    best = 0.0
    for h in engine.get_knowledge_graph(top_k=200):
        if set(h["variables"]) == {a, b}:
            best = max(best, h["metrics"]["confidence"])
    return best


def test_true_edge_survives_light_missingness():
    """A strong x->y edge is still recovered under 10% missing-completely-at-random."""
    rng = np.random.default_rng(1)
    n = 800
    x = rng.normal(size=n)
    y = 0.9 * x + 0.2 * rng.normal(size=n)
    z = rng.normal(size=n)
    rows = []
    for t in range(n):
        r = {"x": float(x[t]), "y": float(y[t]), "z": float(z[t])}
        for c in ("x", "y", "z"):
            if rng.random() < 0.10:            # 10% MCAR
                r.pop(c, None)
        rows.append(r)
    e = _run(lambda: GPUDiscoveryEngine(device="cpu"), rows)
    assert _edge(e, "x", "y") >= 0.55, "true edge lost under light missingness"


def test_out_of_order_rows_do_not_crash_and_keep_true_edge():
    """Locally shuffled (out-of-order) arrival is tolerated."""
    rng = np.random.default_rng(2)
    n = 800
    x = rng.normal(size=n)
    y = 0.9 * x + 0.2 * rng.normal(size=n)
    z = rng.normal(size=n)
    arr = np.column_stack([x, y, z])
    for s in range(0, n, 20):                  # shuffle within 20-row windows
        blk = arr[s:s + 20]
        arr[s:s + 20] = blk[rng.permutation(len(blk))]
    rows = [{"x": float(arr[t, 0]), "y": float(arr[t, 1]), "z": float(arr[t, 2])} for t in range(n)]
    e = _run(lambda: GPUDiscoveryEngine(device="cpu"), rows)
    assert _edge(e, "x", "y") >= 0.55   # a contemporaneous edge is order-invariant
