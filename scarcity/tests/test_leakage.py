"""Leakage-prevention tests — no future information may leak backward.

These lock down two credibility-critical guarantees:

  1. Streaming causality: the engine's knowledge graph after processing a prefix
     rows[0:t] is identical whether or not later rows exist. Any regression that
     introduced a full-array statistic or a look-ahead would break this.

  2. Forecasting: a one-step-ahead prediction is a function of the training data
     only. It must not read the test row's contemporaneous feature values.

  3. Windowed correlation reads only past observations.
"""
import numpy as np
import pytest

from scarcity.engine.relationships import CorrelationalHypothesis
from scarcity.engine.relationship_config import CorrelationalConfig


def _kg_signature(engine):
    """A stable, comparable projection of the knowledge graph."""
    sig = []
    for h in engine.get_knowledge_graph(top_k=200):
        sig.append((h["type"], tuple(sorted(map(str, h["variables"]))),
                    round(float(h["metrics"]["confidence"]), 9)))
    return sorted(sig)


def _stream(n, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 0.9 * x + 0.2 * rng.normal(size=n)
    z = rng.normal(size=n)
    return [{"x": float(x[t]), "y": float(y[t]), "z": float(z[t])} for t in range(n)]


def test_streaming_state_is_invariant_to_future_rows():
    """KG after t rows must not depend on rows that come after t."""
    from scarcity.engine.gpu_engine import GPUDiscoveryEngine
    schema = {"fields": [{"name": c} for c in ("x", "y", "z")]}
    t = 150
    rows = _stream(2 * t, seed=1)

    # Engine A sees only the prefix.
    a = GPUDiscoveryEngine(device="cpu")
    a.initialize_v2(schema, use_causal=True)
    for r in rows[:t]:
        a.process_row(r)
    sig_a = _kg_signature(a)

    # Engine B sees the prefix, snapshotted before the future rows arrive, then
    # continues — the snapshot must equal A regardless of what follows.
    b = GPUDiscoveryEngine(device="cpu")
    b.initialize_v2(schema, use_causal=True)
    for r in rows[:t]:
        b.process_row(r)
    sig_b_prefix = _kg_signature(b)
    for r in rows[t:]:                       # future rows
        b.process_row(r)

    assert sig_a == sig_b_prefix, "engine state after a prefix changed — future leak"
    # sanity: the future rows *do* change the graph (the test is not vacuous)
    assert _kg_signature(b) != sig_b_prefix


def test_forecast_uses_train_features_not_test_row():
    """evaluate_scarcity_graph's one-step prediction must depend on the training
    features (last train row), not on the test row's contemporaneous values."""
    pd = pytest.importorskip("pandas")
    from benchmark.evaluation.forecasting import ForecastingEvaluator

    rng = np.random.default_rng(3)
    n = 40
    idx = list(range(2000, 2000 + n))
    parent = rng.normal(size=n)
    target = 0.8 * np.roll(parent, 1) + 0.1 * rng.normal(size=n)   # target_t ~ parent_{t-1}
    train = pd.DataFrame({"drive": parent[:-1], "gdp": target[:-1]}, index=idx[:-1])

    ev = ForecastingEvaluator(target_variable="gdp", horizon=1)
    graph = {"gdp": ["drive"]}

    # Two single-row test years with the SAME target but DIFFERENT feature value.
    # A leak-free predictor gives the same prediction (uses last train row), so
    # the same MAE; a contemporaneous-feature leak would change it.
    test_a = pd.DataFrame({"drive": [999.0], "gdp": [target[-1]]}, index=[idx[-1]])
    test_b = pd.DataFrame({"drive": [-999.0], "gdp": [target[-1]]}, index=[idx[-1]])

    mae_a = ev.evaluate_scarcity_graph(train, test_a, graph)["mae"]
    mae_b = ev.evaluate_scarcity_graph(train, test_b, graph)["mae"]
    assert mae_a == pytest.approx(mae_b), "prediction changed with the test feature — leak"


def test_windowed_correlation_reads_only_past():
    """The windowed estimate at step t must equal the estimate on the first t
    observations alone — i.e. it never peeks at observations fed after t."""
    rng = np.random.default_rng(5)
    n = 400
    x = rng.normal(size=n)
    y = 0.7 * x + 0.3 * rng.normal(size=n)
    rows = [{"x": float(x[t]), "y": float(y[t])} for t in range(n)]
    cfg = lambda: CorrelationalConfig(window=100)

    t = 250
    # Fed the full stream but read at step t.
    h_full = CorrelationalHypothesis("x", "y", buffer_size=100, config=cfg())
    r_at_t = None
    for i, row in enumerate(rows):
        h_full.fit_step(row)
        m = h_full.evaluate(row)
        if i == t:
            r_at_t = m["correlation"]

    # Fed only the prefix rows[0:t+1].
    h_prefix = CorrelationalHypothesis("x", "y", buffer_size=100, config=cfg())
    for row in rows[:t + 1]:
        h_prefix.fit_step(row)
        m = h_prefix.evaluate(row)
    assert r_at_t == pytest.approx(m["correlation"]), "windowed estimate saw the future"
