import numpy as np

from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
from scarcity.engine.store import HypergraphStore


def test_audit_engine_v2_smoke():
    engine = OnlineDiscoveryEngine()
    schema = {"fields": [{"name": "x"}, {"name": "y"}, {"name": "z"}]}
    engine.initialize_v2(schema, use_causal=False)
    for i in range(12):
        engine.process_row({"x": i, "y": i + 1, "z": i + 2})
    hyps = list(engine.hypotheses.population.values())
    assert hyps and all(np.isfinite(h.confidence) for h in hyps[:5])


def test_audit_store_roundtrip():
    store = HypergraphStore()
    a = store.get_or_create_node("Foo ")
    b = store.get_or_create_node(" foo")
    store.upsert_edge(a, b, 1.0, 0.0, 1.0, 0.9)
    assert a == b and store.get_edge(a, b) is not None
