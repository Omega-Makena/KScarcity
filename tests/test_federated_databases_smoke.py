from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from federated_databases import ScarcityFederationManager


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

    result = manager.run_sync_round(lookback_hours=24)
    assert result.participants >= 2
    assert result.total_samples >= 2

    audit = manager.get_exchange_log(limit=20)
    assert audit, "Expected exchange audit records after sync"
