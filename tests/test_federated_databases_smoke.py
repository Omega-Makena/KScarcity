from __future__ import annotations

from pathlib import Path
import sys
import json
import sqlite3

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from federated_databases import ScarcityFederationManager


def _read_round_update(node_db_path: str, round_number: int, mode: str):
    with sqlite3.connect(node_db_path) as conn:
        row = conn.execute(
            """
            SELECT weights_json, sample_count
            FROM model_updates
            WHERE round_number = ? AND json_extract(metrics_json, '$.mode') = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (int(round_number), mode),
        ).fetchone()
    assert row is not None, f"Missing update for mode={mode}, round={round_number}"
    weights = np.array(json.loads(row[0]), dtype=np.float64)
    return weights, int(row[1])


def test_federated_round_creates_audit(tmp_path: Path):
    manager = ScarcityFederationManager(base_dir=tmp_path / "fed_runtime")
    manager.register_node("org_a")
    manager.register_node("org_b")

    # Seed minimal samples directly into nodes.
    sample = {
        "sample_uid": "seed-1",
        "timestamp": "2026-02-18T00:00:00",
        "county": "Nairobi",
        "sector": "taxation",
        "criticality": 0.8,
        "threat_score": 0.9,
        "escalation_score": 0.7,
        "coordination_score": 0.6,
        "urgency_rate": 0.4,
        "imperative_rate": 0.2,
        "policy_severity": 0.5,
        "label": 1,
    }
    manager._node_store("org_a").add_samples([sample])
    sample_b = dict(sample)
    sample_b["sample_uid"] = "seed-2"
    sample_b["criticality"] = 0.2
    sample_b["label"] = 0
    manager._node_store("org_b").add_samples([sample_b])

    result = manager.run_sync_round(lookback_hours=24, source_path=tmp_path / "missing.csv")
    assert result.participants >= 2
    assert result.total_samples >= 2

    audit = manager.get_exchange_log(limit=20)
    assert audit, "Expected exchange audit records after sync"


def test_register_node_rejects_invalid_backend(tmp_path: Path):
    manager = ScarcityFederationManager(base_dir=tmp_path / "fed_runtime")
    with pytest.raises(ValueError, match="Only sqlite backend"):
        manager.register_node("org_x", backend="postgres")


def test_register_node_rejects_empty_node_id(tmp_path: Path):
    manager = ScarcityFederationManager(base_dir=tmp_path / "fed_runtime")
    with pytest.raises(ValueError, match="node_id cannot be empty"):
        manager.register_node("   ")


def test_sync_round_uses_sample_weighted_aggregation(tmp_path: Path):
    manager = ScarcityFederationManager(base_dir=tmp_path / "fed_runtime")
    manager.register_node("org_a")
    manager.register_node("org_b")

    # Node A has many positive samples, Node B has fewer negative samples.
    a_rows = []
    b_rows = []
    for i in range(90):
        a_rows.append(
            {
                "sample_uid": f"a-{i}",
                "timestamp": "2026-02-18T00:00:00",
                "county": "Nairobi",
                "sector": "taxation",
                "criticality": 0.9,
                "threat_score": 0.9,
                "escalation_score": 0.85,
                "coordination_score": 0.8,
                "urgency_rate": 0.75,
                "imperative_rate": 0.7,
                "policy_severity": 0.9,
                "label": 1,
            }
        )
    for i in range(10):
        b_rows.append(
            {
                "sample_uid": f"b-{i}",
                "timestamp": "2026-02-18T00:00:00",
                "county": "Mombasa",
                "sector": "taxation",
                "criticality": 0.1,
                "threat_score": 0.1,
                "escalation_score": 0.15,
                "coordination_score": 0.2,
                "urgency_rate": 0.25,
                "imperative_rate": 0.3,
                "policy_severity": 0.1,
                "label": 0,
            }
        )

    manager._node_store("org_a").add_samples(a_rows)
    manager._node_store("org_b").add_samples(b_rows)

    result = manager.run_sync_round(lookback_hours=24, source_path=tmp_path / "missing.csv")
    assert result.participants == 2
    assert result.total_samples == 100

    nodes = {n["node_id"]: n for n in manager.list_nodes()}
    a_local_w, a_n = _read_round_update(nodes["org_a"]["db_path"], result.round_number, "federated_local")
    b_local_w, b_n = _read_round_update(nodes["org_b"]["db_path"], result.round_number, "federated_local")

    expected = (a_local_w * a_n + b_local_w * b_n) / float(a_n + b_n)
    global_state = manager.control.get_global_state("global_weights")
    assert global_state and "weights" in global_state
    observed = np.array(global_state["weights"], dtype=np.float64)

    assert observed.shape == expected.shape
    assert np.allclose(observed, expected, atol=1e-10), "Global weights are not sample-weighted average"


def test_sync_round_is_deterministic_for_same_seeded_input(tmp_path: Path):
    def _build_manager(base: Path) -> ScarcityFederationManager:
        m = ScarcityFederationManager(base_dir=base)
        m.register_node("org_a")
        m.register_node("org_b")
        sample_a = {
            "sample_uid": "same-a",
            "timestamp": "2026-02-18T00:00:00",
            "county": "Nairobi",
            "sector": "taxation",
            "criticality": 0.8,
            "threat_score": 0.9,
            "escalation_score": 0.7,
            "coordination_score": 0.6,
            "urgency_rate": 0.4,
            "imperative_rate": 0.2,
            "policy_severity": 0.5,
            "label": 1,
        }
        sample_b = dict(sample_a)
        sample_b["sample_uid"] = "same-b"
        sample_b["criticality"] = 0.2
        sample_b["threat_score"] = 0.15
        sample_b["label"] = 0
        m._node_store("org_a").add_samples([sample_a])
        m._node_store("org_b").add_samples([sample_b])
        return m

    m1 = _build_manager(tmp_path / "runtime_1")
    m2 = _build_manager(tmp_path / "runtime_2")

    r1 = m1.run_sync_round(lookback_hours=24, source_path=tmp_path / "missing1.csv")
    r2 = m2.run_sync_round(lookback_hours=24, source_path=tmp_path / "missing2.csv")

    w1 = np.array(m1.control.get_global_state("global_weights")["weights"], dtype=np.float64)
    w2 = np.array(m2.control.get_global_state("global_weights")["weights"], dtype=np.float64)

    assert r1.participants == r2.participants
    assert r1.total_samples == r2.total_samples
    assert np.allclose(w1, w2, atol=1e-12), "Same inputs should produce same global weights"


def test_sync_round_skips_empty_node_but_succeeds(tmp_path: Path):
    manager = ScarcityFederationManager(base_dir=tmp_path / "fed_runtime")
    manager.register_node("org_a")
    manager.register_node("org_b")

    # Seed only one node; the other remains empty and should be skipped.
    sample = {
        "sample_uid": "single-node-seed",
        "timestamp": "2026-02-18T00:00:00",
        "county": "Nairobi",
        "sector": "taxation",
        "criticality": 0.75,
        "threat_score": 0.8,
        "escalation_score": 0.7,
        "coordination_score": 0.6,
        "urgency_rate": 0.5,
        "imperative_rate": 0.4,
        "policy_severity": 0.7,
        "label": 1,
    }
    manager._node_store("org_a").add_samples([sample])

    result = manager.run_sync_round(lookback_hours=24, source_path=tmp_path / "missing.csv")
    assert result.participants == 1
    assert result.total_samples == 1


def test_sync_round_raises_when_all_nodes_empty(tmp_path: Path):
    manager = ScarcityFederationManager(base_dir=tmp_path / "fed_runtime")
    manager.register_node("org_a")
    manager.register_node("org_b")

    with pytest.raises(RuntimeError, match="No node had usable samples"):
        manager.run_sync_round(lookback_hours=24, source_path=tmp_path / "missing.csv")


def test_sync_round_skips_corrupted_local_update_and_completes(tmp_path: Path, monkeypatch):
    manager = ScarcityFederationManager(base_dir=tmp_path / "fed_runtime")
    manager.register_node("org_a")
    manager.register_node("org_b")

    good = {
        "sample_uid": "good-a",
        "timestamp": "2026-02-18T00:00:00",
        "county": "Nairobi",
        "sector": "taxation",
        "criticality": 0.8,
        "threat_score": 0.8,
        "escalation_score": 0.7,
        "coordination_score": 0.6,
        "urgency_rate": 0.5,
        "imperative_rate": 0.4,
        "policy_severity": 0.7,
        "label": 1,
    }
    good_b = dict(good)
    good_b["sample_uid"] = "good-b"
    good_b["label"] = 0
    manager._node_store("org_a").add_samples([good])
    manager._node_store("org_b").add_samples([good_b])

    real_train = manager._train_local_step

    def _patched_train(weights, x, y, learning_rate=0.12, model_name="logistic"):
        # First node update is corrupted; second should proceed normally.
        if float(np.mean(y)) > 0.5:
            return np.full_like(weights, np.nan), float("nan"), float("nan")
        return real_train(weights, x, y, learning_rate=learning_rate, model_name=model_name)

    monkeypatch.setattr(manager, "_train_local_step", _patched_train)

    result = manager.run_sync_round(lookback_hours=24, source_path=tmp_path / "missing.csv")
    assert result.participants == 1
    assert result.total_samples == 1

    exchange_log = manager.get_exchange_log(limit=20)
    skipped_records = [r for r in exchange_log if r.get("payload_type") == "model_update_skipped"]
    assert skipped_records, "Expected a model_update_skipped record in exchange audit"
    assert skipped_records[0].get("details", {}).get("reason") == "non_finite_local_update"

    history = manager.get_round_history(limit=1)
    assert history, "Expected round history entry"
    skipped = history[0].get("metrics", {}).get("skipped_updates", [])
    assert skipped and skipped[0].get("reason") == "non_finite_local_update"

    # Ensure raw runtime audit file also captures the skip event.
    with manager.audit_path.open("r", encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    skip_events = [e for e in events if e.get("event_type") == "local_update_skipped"]
    assert skip_events, "Expected local_update_skipped in audit_log.jsonl"
    assert skip_events[-1].get("payload", {}).get("reason") == "non_finite_local_update"
