"""ABAC/RBAC hybrid policy checks for federated queries."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from k_collab.common.versioned_store import VersionedJSONStore
from federated_databases.contracts.models import CLASSIFICATION_ORDER


@dataclass
class AccessContext:
    user_id: str
    role: str
    clearance: str
    purpose: str


@dataclass
class PolicyDecision:
    allowed: bool
    reasons: List[str] = field(default_factory=list)


class PolicyEngine:
    """Policy authoring + evaluation."""

    def __init__(self, base_dir: Path | str):
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        self._store = VersionedJSONStore(base / "policy_versions.jsonl", kind="policy")
        if not self._store.latest():
            self._store.save(
                {
                    "deny_operations": [
                        "raw_export",
                        "raw_export_cross_agency",
                        "row_level_cross_agency",
                        "full_table_dump",
                        "extract_raw_records",
                    ],
                    "roles": {
                        "analyst": {"max_clearance": "RESTRICTED", "purposes": ["monitoring", "research", "casework"]},
                        "supervisor": {"max_clearance": "SECRET", "purposes": ["monitoring", "research", "casework", "oversight"]},
                        "auditor": {"max_clearance": "SECRET", "purposes": ["audit", "oversight"]},
                    },
                },
                actor="system",
                message="default_policy",
            )

    def latest(self) -> Dict[str, Any]:
        rec = self._store.latest()
        return rec.payload if rec else {}

    def set_policy(self, payload: Dict[str, Any], actor: str = "system", message: str = "") -> Dict[str, Any]:
        rec = self._store.save(payload, actor=actor, message=message or "policy_update")
        return rec.__dict__

    def list_versions(self, limit: int = 30) -> List[Dict[str, Any]]:
        return [r.__dict__ for r in self._store.list(limit)]

    def evaluate(
        self,
        context: AccessContext,
        dataset_contracts: List[Dict[str, Any]],
        operation: str,
        cross_agency: bool,
    ) -> PolicyDecision:
        policy = self.latest()
        reasons: List[str] = []
        role_rules = policy.get("roles", {}).get(context.role, None)
        if not role_rules:
            reasons.append(f"role_not_allowed:{context.role}")
            return PolicyDecision(False, reasons)

        user_clearance = context.clearance.upper()
        max_for_role = str(role_rules.get("max_clearance", "PUBLIC")).upper()
        if CLASSIFICATION_ORDER.get(user_clearance, -1) > CLASSIFICATION_ORDER.get(max_for_role, -1):
            reasons.append("context_clearance_exceeds_role_limit")
            return PolicyDecision(False, reasons)

        allowed_purposes = {str(p).lower() for p in role_rules.get("purposes", [])}
        if context.purpose.lower() not in allowed_purposes:
            reasons.append(f"purpose_not_allowed:{context.purpose}")
            return PolicyDecision(False, reasons)

        deny_ops = {str(v).lower() for v in policy.get("deny_operations", [])}
        if operation.lower() in deny_ops:
            reasons.append(f"operation_denied:{operation}")
            return PolicyDecision(False, reasons)

        for contract in dataset_contracts:
            required = str(contract.get("classification", "PUBLIC")).upper()
            if CLASSIFICATION_ORDER.get(user_clearance, -1) < CLASSIFICATION_ORDER.get(required, -1):
                reasons.append(f"clearance_too_low_for_dataset:{contract.get('dataset_id', 'unknown')}")

        if cross_agency and operation.lower() not in {"aggregate", "time_bucket"}:
            reasons.append("cross_agency_requires_aggregate_only")

        if reasons:
            return PolicyDecision(False, reasons)
        return PolicyDecision(True, ["allowed"])
