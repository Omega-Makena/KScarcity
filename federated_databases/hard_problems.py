"""DB-side hard-problem assessor for federated data access coordination."""

from __future__ import annotations

from typing import Any, Dict, List


def _status(ok: bool, warn: bool = False) -> str:
    if ok:
        return "pass"
    if warn:
        return "warn"
    return "fail"


class FederatedDBHardProblemAssessor:
    """Summarize seven production hard problems on the federated DB side."""

    def assess(
        self,
        *,
        health: List[Dict[str, Any]],
        compatibility: Dict[str, Any],
        query_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        rows = query_result.get("rows", []) if isinstance(query_result, dict) else []
        trace = query_result.get("execution_trace", []) if isinstance(query_result, dict) else []
        provenance = query_result.get("provenance", {}) if isinstance(query_result, dict) else {}
        non_iid = query_result.get("non_iid_diagnostics", {}) if isinstance(query_result, dict) else {}
        allowed = bool(query_result.get("allowed")) if isinstance(query_result, dict) else False

        healthy_count = len([h for h in health if h.get("healthy")]) if health else 0
        total_health = len(health or [])
        node_scores = compatibility.get("node_scores", []) if isinstance(compatibility, dict) else []
        baskets = compatibility.get("baskets", []) if isinstance(compatibility, dict) else []

        non_iid_skew = float(non_iid.get("mean_dominant_share", 0.0) or 0.0)
        non_iid_warn = non_iid_skew >= 0.80

        problems = [
            {
                "id": "db_1_connectivity_heterogeneity",
                "name": "Heterogeneous source connectivity",
                "status": _status(total_health > 0 and healthy_count == total_health),
                "evidence": {"healthy_connectors": healthy_count, "total_connectors": total_health},
                "action": "Repair failed connector health checks before collaboration.",
            },
            {
                "id": "db_2_schema_contract_alignment",
                "name": "Schema/contract/canonical alignment",
                "status": _status(bool(node_scores)),
                "evidence": {"compatibility_nodes_scored": len(node_scores)},
                "action": "Ensure dataset contracts and canonical mappings are published for all participants.",
            },
            {
                "id": "db_3_basket_formation",
                "name": "Compatibility baskets and partial participation",
                "status": _status(bool(baskets)),
                "evidence": {"basket_count": len(baskets), "excluded_nodes": len(compatibility.get("excluded_nodes", []))},
                "action": "Tune governance thresholds or mappings for excluded nodes.",
            },
            {
                "id": "db_4_planner_pushdown",
                "name": "Planner routing and local pushdown",
                "status": _status(bool(trace) and all("pushdown" in t for t in trace)),
                "evidence": {"execution_steps": len(trace)},
                "action": "Validate connector capabilities and pushdown mapping fields.",
            },
            {
                "id": "db_5_privacy_suppression",
                "name": "Suppression and safe output controls",
                "status": _status(allowed and "suppressed" in query_result),
                "evidence": {"suppressed_groups": len(query_result.get("suppressed", [])), "rows_returned": len(rows)},
                "action": "Adjust k-threshold/governance if unsafe small groups can leak.",
            },
            {
                "id": "db_6_provenance_coverage",
                "name": "Provenance, exclusion reasoning, and coverage",
                "status": _status(bool(provenance) and provenance.get("coverage_score", 0.0) > 0.0),
                "evidence": {
                    "coverage_score": provenance.get("coverage_score", 0.0),
                    "quality_score": provenance.get("quality_score", 0.0),
                    "excluded_institutions": len(provenance.get("excluded_institutions", [])),
                },
                "action": "Ensure every excluded institution carries reason metadata in provenance.",
            },
            {
                "id": "db_7_non_iid_visibility",
                "name": "Non-IID skew visibility (DB side)",
                "status": _status(bool(non_iid), warn=non_iid_warn),
                "evidence": non_iid,
                "action": "Use weighting/coverage notes when dominant-share skew is high across institutions.",
            },
        ]

        passed = len([p for p in problems if p["status"] == "pass"])
        warned = len([p for p in problems if p["status"] == "warn"])
        failed = len([p for p in problems if p["status"] == "fail"])
        overall = "pass" if failed == 0 and warned == 0 else ("warn" if failed == 0 else "fail")

        return {
            "overall_status": overall,
            "summary": {"passed": passed, "warned": warned, "failed": failed},
            "problems": problems,
        }
