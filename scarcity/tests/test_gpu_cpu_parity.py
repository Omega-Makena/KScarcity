"""CPU vs GPU backend parity.

The pure-Python OnlineDiscoveryEngine (vectorized=False) is the authoritative
path. These tests pin that the GPU/torch backend (GPUDiscoveryEngine) exposes
the SAME knowledge-graph interface and recovers the SAME structure on controlled
data, so vectorized=True can be trusted as a drop-in replacement.

Runs the GPU engine on device='cpu' (torch, no CUDA needed).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
from scarcity.engine.gpu_engine import GPUDiscoveryEngine

_SCHEMA = {"fields": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}


def _data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = np.concatenate([[0.0], a[:-1]]) + 0.1 * rng.normal(size=n)  # b[t] ~ a[t-1]
    c = rng.normal(size=n)                                          # independent
    return [{"a": float(a[i]), "b": float(b[i]), "c": float(c[i])} for i in range(n)]


def _run(engine, rows):
    for r in rows:
        engine.process_row(r)
    return engine.get_knowledge_graph()


def _strong_edges(kg, conf=0.5):
    return {
        (e["type"], tuple(e["variables"]))
        for e in kg
        if e["metrics"]["confidence"] >= conf and e["state"] != "dead"
    }


def test_gpu_knowledge_graph_matches_cpu_schema():
    rows = _data()
    cpu = OnlineDiscoveryEngine(vectorized=False)
    cpu.initialize_v2(_SCHEMA, use_causal=True)
    cpu_kg = _run(cpu, rows)

    gpu = GPUDiscoveryEngine(device="cpu")
    gpu.initialize_v2(_SCHEMA, use_causal=True)
    gpu_kg = _run(gpu, rows)

    assert cpu_kg and gpu_kg
    keys = {"id", "type", "state", "created_at", "generation", "variables", "metrics"}
    mkeys = {"fit_score", "confidence", "evidence", "stability"}
    for kg in (cpu_kg, gpu_kg):
        e = kg[0]
        assert keys <= set(e)
        assert mkeys <= set(e["metrics"])


def test_gpu_and_cpu_recover_the_core_causal_edge():
    # Both backends must recover the injected a<->b coupling as a strong causal
    # edge. (The GPU pool currently covers the causal/RLS core; the other 13
    # types' batched estimators are the Step-2 port, so full multi-type parity
    # is not yet asserted here — only that where the GPU engine speaks, it agrees
    # with the CPU engine on the primary structure.)
    rows = _data()
    cpu = OnlineDiscoveryEngine(vectorized=False)
    cpu.initialize_v2(_SCHEMA, use_causal=True)
    cpu_causal = {v for t, v in _strong_edges(_run(cpu, rows)) if t == "causal"}

    gpu = GPUDiscoveryEngine(device="cpu")
    gpu.initialize_v2(_SCHEMA, use_causal=True)
    gpu_causal = {v for t, v in _strong_edges(_run(gpu, rows)) if t == "causal"}

    def _has_ab(edges):
        return any({"a", "b"} <= set(v) for v in edges)

    assert _has_ab(cpu_causal), f"CPU missed a-b causal edge: {cpu_causal}"
    assert _has_ab(gpu_causal), f"GPU missed a-b causal edge: {gpu_causal}"
