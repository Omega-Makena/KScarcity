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


def test_fl_non_iid_drift_trend_increases_across_rounds():
    low_drift = {
        "c1": [1.0, 1.0, 1.0],
        "c2": [1.02, 0.98, 1.01],
        "c3": [0.99, 1.01, 1.0],
    }
    high_drift = {
        "c1": [10.0, 0.0, 0.0],
        "c2": [0.0, 10.0, 0.0],
        "c3": [0.0, 0.0, 10.0],
    }

    d1 = update_heterogeneity(low_drift)
    d2 = update_heterogeneity(high_drift)

    assert d2["mean_cosine_distance"] > d1["mean_cosine_distance"]
    assert d2["mean_cosine_distance"] > 0.9
    assert d1["is_highly_non_iid"] is False
    assert d2["is_highly_non_iid"] is True


def test_db_non_iid_skew_trend_increases_with_dominance():
    balanced_rows = [
        {"county": "A", "_node_contrib": {"n1": 10.0, "n2": 9.0, "n3": 11.0}},
        {"county": "B", "_node_contrib": {"n1": 8.0, "n2": 7.5, "n3": 9.0}},
    ]
    skewed_rows = [
        {"county": "A", "_node_contrib": {"n1": 100.0, "n2": 2.0, "n3": 1.0}},
        {"county": "B", "_node_contrib": {"n1": 90.0, "n2": 3.0, "n3": 2.0}},
    ]

    s1 = summarize_non_iid(balanced_rows)
    s2 = summarize_non_iid(skewed_rows)

    assert s2["mean_dominant_share"] > s1["mean_dominant_share"]
    assert s2["highly_skewed_groups"] > s1["highly_skewed_groups"]
