"""Experiment + provenance layer: reproducible records and auditable edges."""
import os
import numpy as np
import pytest

from scarcity.experiment import (
    DatasetSpec, EnvInfo, HardwareInfo, ExperimentRecord,
    experiment, set_seeds, build_edge_provenance, explain_edge,
)
from scarcity.experiment.record import EdgeProvenance


def _rows(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    z = rng.standard_normal(n)
    y = 0.9 * x + 0.1 * rng.standard_normal(n)
    return [{"x": float(x[t]), "y": float(y[t]), "z": float(z[t])} for t in range(n)]


def _run_engine(rows):
    from scarcity.engine import OnlineDiscoveryEngine
    eng = OnlineDiscoveryEngine(vectorized=False, small_dataset_mode=True)
    eng.initialize({"fields": [{"name": "x"}, {"name": "y"}, {"name": "z"}]})
    for r in rows:
        eng.process_row(r)
    return eng


# --- record capture / serialization ------------------------------------------

def test_env_and_hardware_capture():
    env = EnvInfo.capture()
    assert env.python_version and env.scarcity_version
    hw = HardwareInfo.capture()
    assert hw.cpu_count >= 1 and hw.platform


def test_experiment_captures_envelope(tmp_path):
    ds = DatasetSpec(name="ds", version="1", n_observations=300, n_variables=3,
                     variables=["x", "y", "z"], observation_range=(0, 300))
    with experiment("t", dataset=ds, config={"vectorized": False}, seed=7,
                    out_dir=str(tmp_path)) as rec:
        rec.metrics = {"ok": True}
    assert rec.status == "ok"
    assert rec.seed == 7
    assert rec.runtime_seconds >= 0.0
    assert rec.peak_memory_mb >= 0.0
    assert rec.env is not None and rec.hardware is not None
    assert rec.gpu_used is False                      # config vectorized=False
    files = os.listdir(tmp_path)
    assert len(files) == 1 and files[0].endswith(".json")


def test_experiment_records_and_reraises_failure(tmp_path):
    with pytest.raises(ValueError):
        with experiment("boom", out_dir=str(tmp_path)) as rec:
            raise ValueError("kaboom")
    assert rec.status == "failed"
    assert any("kaboom" in f for f in rec.failures)
    assert os.listdir(tmp_path)                        # still saved on failure


def test_record_save_load_roundtrip(tmp_path):
    rec = ExperimentRecord(name="r", seed=1, dataset=DatasetSpec(name="d"),
                           env=EnvInfo.capture(), hardware=HardwareInfo.capture())
    rec.edges = [EdgeProvenance(source="a", target="b", rel_type="causal",
                                variables=["a", "b"], metrics={"confidence": 0.9})]
    path = str(tmp_path / "rec.json")
    rec.save(path)
    loaded = ExperimentRecord.load(path)
    assert loaded["name"] == "r" and loaded["seed"] == 1
    assert loaded["edges"][0]["rel_type"] == "causal"


# --- provenance --------------------------------------------------------------

def test_build_edge_provenance_and_explain():
    eng = _run_engine(_rows())
    provs = build_edge_provenance(eng)
    assert provs and all(isinstance(p, EdgeProvenance) for p in provs)
    prov = explain_edge(eng, "x", "y")
    assert prov is not None
    assert {"x", "y"} <= set(prov.variables)
    assert prov.metrics.get("confidence", 0) > 0
    text = prov.explain()
    assert "x -> y" in text and "confidence" in text


def test_explain_edge_absent_returns_none():
    eng = _run_engine(_rows())
    # 'w' is not a variable in the schema.
    assert explain_edge(eng, "x", "w") is None


def test_calibration_and_causal_fold_into_provenance():
    eng = _run_engine(_rows())
    kg = eng.get_knowledge_graph()
    top = max(kg, key=lambda e: e["metrics"].get("confidence", 0))
    key = (top["type"], tuple(top["variables"]))
    provs = build_edge_provenance(
        eng,
        calibration={key: {"perm_p": 0.001, "q_value": 0.01}},
        causal={key: {"estimand": "ATE", "estimate": 0.9}},
    )
    match = [p for p in provs if p.rel_type == top["type"] and p.variables == top["variables"]][0]
    assert match.calibration["perm_p"] == 0.001
    assert match.causal["estimand"] == "ATE"


# --- reproducibility ---------------------------------------------------------

def test_set_seeds_makes_discovery_reproducible():
    rows = _rows(seed=3)
    set_seeds(0)
    g1 = {(e["type"], tuple(e["variables"])) for e in _run_engine(rows).get_knowledge_graph()}
    set_seeds(0)
    g2 = {(e["type"], tuple(e["variables"])) for e in _run_engine(rows).get_knowledge_graph()}
    assert g1 == g2
