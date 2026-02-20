"""Non-IID diagnostics for federated database result composition."""

from __future__ import annotations

from typing import Any, Dict, List


def distribution_skew(node_contributions: Dict[str, float]) -> Dict[str, Any]:
    """Compute simple skew metrics over node contribution weights."""
    total = float(sum(max(0.0, v) for v in node_contributions.values()))
    if total <= 0:
        return {
            "node_count": len(node_contributions),
            "dominant_share": 0.0,
            "entropy": 0.0,
            "is_highly_skewed": False,
        }

    shares = []
    for value in node_contributions.values():
        s = max(0.0, float(value)) / total
        shares.append(s)

    dominant = max(shares) if shares else 0.0
    entropy = 0.0
    for s in shares:
        if s > 0:
            # natural log entropy
            import math

            entropy -= s * math.log(s)

    return {
        "node_count": len(shares),
        "dominant_share": float(dominant),
        "entropy": float(entropy),
        "is_highly_skewed": bool(dominant >= 0.8),
    }


def summarize_non_iid(group_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize non-IID effects from per-group node contribution vectors."""
    if not group_rows:
        return {"groups": 0, "highly_skewed_groups": 0, "mean_dominant_share": 0.0}

    dominant_scores: List[float] = []
    skewed = 0
    for row in group_rows:
        node_contrib = row.get("_node_contrib", {})
        diag = distribution_skew(node_contrib)
        dominant_scores.append(float(diag["dominant_share"]))
        if diag["is_highly_skewed"]:
            skewed += 1

    return {
        "groups": len(group_rows),
        "highly_skewed_groups": skewed,
        "mean_dominant_share": float(sum(dominant_scores) / len(dominant_scores)),
    }
