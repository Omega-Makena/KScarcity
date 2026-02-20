"""FL-side hard-problem assessor for nested federated orchestration."""

from __future__ import annotations

from typing import Any, Dict, List


def _status(ok: bool, warn: bool = False) -> str:
    if ok:
        return "pass"
    if warn:
        return "warn"
    return "fail"


class FederatedMLHardProblemAssessor:
    """Summarize seven production hard problems on the federated ML side."""

    def assess(
        self,
        *,
        readiness: Dict[str, Any],
        round_output: Dict[str, Any] | None = None,
        status_snapshot: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        round_output = round_output or {}
        status_snapshot = status_snapshot or {}
        usable_baskets = list(readiness.get("usable_baskets", []) or [])
        excluded_baskets = list(readiness.get("excluded_baskets", []) or [])
        privacy = dict(readiness.get("privacy", {}) or {})
        reasons = set(readiness.get("reasons", []) or [])

        non_iid = dict(round_output.get("non_iid_diagnostics", {}) or {})
        non_iid_available = bool(non_iid)
        non_iid_warn = bool(non_iid.get("is_highly_non_iid", False))

        models = list(status_snapshot.get("models", []) or [])

        non_iid_status = "fail"
        if non_iid_available and non_iid_warn:
            non_iid_status = "warn"
        elif non_iid_available:
            non_iid_status = "pass"

        problems = [
            {
                "id": "fl_1_topology_alignment",
                "name": "Nested topology to participant alignment",
                "status": _status(len(usable_baskets) > 0),
                "evidence": {"usable_baskets": len(usable_baskets), "excluded_baskets": len(excluded_baskets)},
                "action": "Ensure basket members map to active topology participants.",
            },
            {
                "id": "fl_2_participation_viability",
                "name": "Participation viability across baskets",
                "status": _status(len(usable_baskets) > 0, warn=len(excluded_baskets) > 0),
                "evidence": {"usable_baskets": usable_baskets, "excluded_baskets": excluded_baskets},
                "action": "Increase eligible participants or relax minimum participants per basket carefully.",
            },
            {
                "id": "fl_3_non_iid_drift",
                "name": "Non-IID gradient/update drift (FL side)",
                "status": non_iid_status,
                "evidence": non_iid,
                "action": "Enable drift-aware weighting or basket-level adaptation for high non-IID conditions.",
            },
            {
                "id": "fl_4_privacy_budget",
                "name": "Privacy budget compatibility and runtime guards",
                "status": _status(bool(privacy.get("privacy_budget_compatible", False))),
                "evidence": privacy,
                "action": "Lower per-round spend or increase planned budget under governance approval.",
            },
            {
                "id": "fl_5_secure_aggregation_posture",
                "name": "Secure aggregation posture",
                "status": _status(True, warn=not round_output),
                "evidence": {
                    "runtime_round_observed": bool(round_output),
                    "note": "Secure aggregation remains enforced by existing scarcity federation layer.",
                },
                "action": "Run a supervised round and verify secure aggregation telemetry in audit logs.",
            },
            {
                "id": "fl_6_registry_traceability",
                "name": "Model registry traceability",
                "status": _status(True, warn=len(models) == 0),
                "evidence": {"registered_models": len(models)},
                "action": "Complete at least one job to validate end-to-end model lineage.",
            },
            {
                "id": "fl_7_governance_scope",
                "name": "Governance and project computation scope",
                "status": _status("project_not_authorized_for_federated_ml" not in reasons),
                "evidence": {"readiness_reasons": sorted(reasons)},
                "action": "Update project allowed computations/participants before training.",
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
