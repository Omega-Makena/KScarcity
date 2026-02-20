from __future__ import annotations

from federated_databases.executor.non_iid import summarize_non_iid
from federated_ml.orchestration.non_iid import update_heterogeneity


def test_db_non_iid_summary_detects_skew():
    rows = [
        {"county": "A", "_node_contrib": {"n1": 100.0, "n2": 1.0}},
        {"county": "B", "_node_contrib": {"n1": 90.0, "n2": 0.5}},
    ]
    summary = summarize_non_iid(rows)
    assert summary["groups"] == 2
    assert summary["highly_skewed_groups"] >= 1


def test_fl_non_iid_summary_detects_drift():
    updates = {
        "c1": [10.0, 0.0, 0.0],
        "c2": [0.0, 10.0, 0.0],
        "c3": [0.0, 0.0, 10.0],
    }
    diag = update_heterogeneity(updates)
    assert diag["client_count"] == 3
    assert diag["mean_cosine_distance"] > 0.9
    assert diag["is_highly_non_iid"] is True
