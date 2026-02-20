"""Audit helper for federated ML jobs and compliance summaries."""

from __future__ import annotations

from typing import Any, Dict, List


class FederatedMLAuditHooks:
    """Thin wrapper to log ML job lifecycle events."""

    def __init__(self, audit_log):
        self.audit_log = audit_log

    def job_started(self, actor: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.audit_log.append("ml_job_started", actor=actor, payload=payload, scope="federated_ml")

    def round_completed(self, actor: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.audit_log.append("ml_round_completed", actor=actor, payload=payload, scope="federated_ml")

    def job_completed(self, actor: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.audit_log.append("ml_job_completed", actor=actor, payload=payload, scope="federated_ml")

    def compliance_report(self, actor: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.audit_log.append("ml_compliance_report", actor=actor, payload=payload, scope="federated_ml")
