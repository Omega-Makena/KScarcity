"""Forgetting is uniform across hypothesis types, not only correlational.

forgetting_window>0 sets the effective window for every type (buffer-based types
window their buffer; RLS types map it to a forgetting factor). After a regime
break the stale edge must decay for functional/causal edges too, while the
cumulative default (forgetting_window=0) keeps them.
"""
import numpy as np
import pytest

from scarcity.engine.engine_v2 import OnlineDiscoveryEngine


def _regime_break_stream(seed=1, half=800):
    rng = np.random.default_rng(seed)
    n = 2 * half
    x = rng.normal(size=n)
    y = np.empty(n)
    y[:half] = 0.9 * x[:half] + 0.2 * rng.normal(size=half)   # phase 1: x -> y
    y[half:] = 0.2 * rng.normal(size=half)                     # phase 2: dead
    return x, y, n, half


def _max_xy_conf(engine, types):
    best = 0.0
    for h in engine.hypotheses.population.values():
        d = h.to_dict()
        if set(d.get("variables", [])) == {"x", "y"} and d.get("type") in types:
            best = max(best, d["metrics"].get("confidence", 0.0))
    return best


def _run(fw):
    x, y, n, half = _regime_break_stream()
    e = OnlineDiscoveryEngine(vectorized=False, forgetting_window=fw)
    e.initialize_v2({"fields": [{"name": "x"}, {"name": "y"}]}, use_causal=True)
    p1 = None
    types = ("correlational", "functional")
    for t in range(n):
        e.process_row({"x": float(x[t]), "y": float(y[t])})
        if t == half - 1:
            p1 = _max_xy_conf(e, types)
    return p1, _max_xy_conf(e, types)


def test_forgetting_window_decays_non_correlational_edges():
    p1, p2 = _run(fw=200)
    assert p1 >= 0.55, f"edge not established in phase 1 (conf {p1})"
    assert p2 < 0.30, f"stale edge did not decay with forgetting_window (conf {p2})"


def test_cumulative_default_keeps_stale_edge():
    p1, p2 = _run(fw=0)
    assert p1 >= 0.55
    assert p2 >= 0.55, f"cumulative default should stay stale (conf {p2})"
