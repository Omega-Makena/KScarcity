"""Federated data access control plane with policy/contract/planner/executor pipeline.

This layer virtualizes access and execution across existing institutional systems.
It does not create or replace storage engines, data lakes, or warehouses.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from federated_databases.scarcity_federation import get_scarcity_federation
from federated_databases.catalog.registry import FederatedCatalog
from federated_databases.compatibility.engine import CompatibilityEngine
from federated_databases.hard_problems import FederatedDBHardProblemAssessor
from federated_databases.connectors.base import ConnectorSpec
from federated_databases.contracts.models import DataContract
from federated_databases.contracts.registry import DataContractRegistry
from federated_databases.contracts.mapping import (
    CanonicalFieldMapping,
    DatasetCanonicalMapping,
    CanonicalSchemaRegistry,
    quality_summary,
    resolve_field_map,
)
from federated_databases.executor.engine import FederatedExecutor
from federated_databases.executor.engine import apply_k_suppression
from federated_databases.executor.non_iid import summarize_non_iid
from federated_databases.planner.models import ExecutionPlan
from federated_databases.planner.query_parser import parse_query
from federated_databases.planner.router import plan_query
from federated_databases.policy.engine import AccessContext, PolicyEngine
from k_collab.projects.registry import CollaborationProjectRegistry, CollaborationProject
from k_collab.trust.controls import validate_connector_trust


class FederatedDatabaseControlPlane:
    """Planner-router-executor orchestration with governance checks.

    K-Collab operates above source systems and only returns approved/derived outputs.
    Raw data remains at institutional sources.
    """

    def __init__(self, base_dir: Path | str, audit_log, topology_store, project_registry: CollaborationProjectRegistry | None = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log = audit_log
        self.topology_store = topology_store

        self.catalog = FederatedCatalog(self.base_dir)
        self.contracts = DataContractRegistry(self.base_dir)
        self.canonical = CanonicalSchemaRegistry(self.base_dir)
        self.policy = PolicyEngine(self.base_dir)
        self.compatibility = CompatibilityEngine(self.base_dir)
        self.hard_problems = FederatedDBHardProblemAssessor()
        self.projects = project_registry or CollaborationProjectRegistry(self.base_dir)

    def register_default_from_manager(self, actor: str = "system") -> Dict[str, Any]:
        manager = get_scarcity_federation()
        nodes = manager.list_nodes()
        for node in nodes:
            connector_id = f"sqlite::{node['node_id']}"
            self.register_connector(
                ConnectorSpec(
                    connector_id=connector_id,
                    node_id=node["node_id"],
                    source_type="sqlite",
                    location=node["db_path"],
                    dataset_ids=["local_samples"],
                    options={
                        "channel_security": "internal_mesh",
                        "attestation_status": "verified",
                        "max_classification": "RESTRICTED",
                    },
                ),
                actor=actor,
            )

        if not self.contracts.get("local_samples"):
            self.contracts.upsert(
                DataContract(
                    dataset_id="local_samples",
                    schema={
                        "timestamp": "text",
                        "county": "text",
                        "sector": "text",
                        "criticality": "real",
                        "threat_score": "real",
                        "escalation_score": "real",
                        "coordination_score": "real",
                        "urgency_rate": "real",
                        "imperative_rate": "real",
                        "policy_severity": "real",
                        "label": "integer",
                    },
                    classification="RESTRICTED",
                    pii_fields=["sample_uid"],
                    allowed_operations=["aggregate", "time_bucket"],
                    approved_join_keys=["county", "sector"],
                ),
                actor=actor,
            )
        if not self.canonical.get("local_samples"):
            self.canonical.upsert(
                DatasetCanonicalMapping(
                    dataset_id="local_samples",
                    fields=[
                        CanonicalFieldMapping("county", "county"),
                        CanonicalFieldMapping("sector", "sector"),
                        CanonicalFieldMapping("criticality", "criticality", dtype="real"),
                        CanonicalFieldMapping("threat_score", "threat_score", dtype="real"),
                    ],
                    quality={"freshness_sla_hours": 24, "completeness_min": 0.90},
                ),
                actor=actor,
            )
        if not self.projects.get("default_monitoring"):
            self.projects.upsert(
                CollaborationProject(
                    project_id="default_monitoring",
                    name="Default Monitoring Project",
                    objective="Cross-institution monitoring with aggregate-only analytics",
                    participants=[n["node_id"] for n in nodes],
                    allowed_datasets=["local_samples"],
                    allowed_domains=["intel", "finance", "security"],
                    allowed_computations=["analytics", "federated_ml"],
                    governance={"k_threshold_min": 3, "purpose_allowlist": ["monitoring", "research", "casework"]},
                ),
                actor=actor,
            )

        self.audit_log.append(
            event_type="connector_registry_refreshed",
            actor=actor,
            payload={"node_count": len(nodes)},
            scope="federated_databases",
        )
        return {"nodes": len(nodes), "connectors": len(self.catalog.connectors())}

    def register_connector(self, spec: ConnectorSpec, actor: str = "system") -> Dict[str, Any]:
        issues = validate_connector_trust(spec.__dict__)
        if issues:
            raise ValueError(f"Connector trust validation failed: {issues}")
        return self.catalog.register_connector(spec, actor=actor)

    def register_external_source(self, spec: ConnectorSpec, actor: str = "system") -> Dict[str, Any]:
        """Alias emphasizing that sources are external systems, not internal storage."""
        return self.register_connector(spec, actor=actor)

    def upsert_project(self, project: CollaborationProject, actor: str = "system") -> Dict[str, Any]:
        return self.projects.upsert(project, actor=actor)

    def connector_health(self, dataset_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Basic health probe for registered connectors."""
        connectors = self.catalog.connectors()
        if dataset_id:
            dataset_nodes = {x.get("node_id") for x in self.catalog.dataset_locations(dataset_id)}
            connectors = [c for c in connectors if c.get("node_id") in dataset_nodes]

        out: List[Dict[str, Any]] = []
        for connector in connectors:
            source_type = str(connector.get("source_type", "")).lower()
            location = str(connector.get("location", ""))
            healthy = True
            reason = "ok"
            if source_type == "sqlite":
                healthy = Path(location).exists()
                reason = "sqlite_path_exists" if healthy else "sqlite_path_missing"
            elif source_type in {"postgres", "oracle", "sqlserver", "mssql", "azure", "http", "http_api", "api"}:
                healthy = bool(location)
                reason = "location_present" if healthy else "missing_location"
            out.append(
                {
                    "connector_id": connector.get("connector_id"),
                    "node_id": connector.get("node_id"),
                    "source_type": source_type,
                    "healthy": healthy,
                    "reason": reason,
                }
            )
        return out

    def run_compatibility_analysis(
        self,
        *,
        dataset_id: str,
        operation: str = "aggregate",
        project_id: Optional[str] = None,
        actor: str = "system",
    ) -> Dict[str, Any]:
        project_key = project_id or "default_monitoring"
        project = self.projects.get(project_key)
        if not project:
            raise ValueError(f"Project not found: {project_key}")

        all_connectors = [c for c in self.catalog.connectors() if dataset_id in set(c.get("dataset_ids", []))]
        allowed_nodes = set(project.get("participants", []))
        candidate_connectors = [c for c in all_connectors if c.get("node_id") in allowed_nodes] if allowed_nodes else all_connectors
        project_excluded = sorted(
            [
                {
                    "node_id": str(c.get("node_id", "")),
                    "connector_id": str(c.get("connector_id", "")),
                    "reason": "node_not_in_project_participants",
                }
                for c in all_connectors
                if allowed_nodes and c.get("node_id") not in allowed_nodes
            ],
            key=lambda x: (x["node_id"], x["connector_id"]),
        )

        report = self.compatibility.analyze(
            project=project,
            dataset_id=dataset_id,
            operation=operation,
            connectors=candidate_connectors,
            contract=self.contracts.get(dataset_id),
            canonical_mapping=self.canonical.get(dataset_id),
        )
        report["project_id"] = project_key
        report["project_excluded_nodes"] = project_excluded
        self.audit_log.append(
            event_type="compatibility_analyzed",
            actor=actor,
            payload={
                "project_id": project_key,
                "dataset_id": dataset_id,
                "operation": operation,
                "version_id": report.get("version_id"),
                "candidate_connectors": len(candidate_connectors),
                "excluded_by_project": len(project_excluded),
                "baskets": len(report.get("baskets", [])),
            },
            scope="federated_databases",
        )
        return report

    def evaluate_policy(
        self,
        context: AccessContext,
        query,
        cross_agency: bool,
    ):
        contract = self.contracts.get(query.dataset_id)
        if not contract:
            return False, [f"missing_contract:{query.dataset_id}"]
        decision = self.policy.evaluate(
            context=context,
            dataset_contracts=[contract],
            operation=query.operation,
            cross_agency=cross_agency,
        )
        if query.operation not in set(contract.get("allowed_operations", [])):
            decision.allowed = False
            decision.reasons.append("operation_not_allowed_by_contract")
        schema = set(contract.get("schema", {}).keys())
        for key in query.group_by:
            if key not in schema:
                decision.allowed = False
                decision.reasons.append(f"invalid_group_by_field:{key}")
        if query.metric_field != "*" and query.metric_field not in schema:
            decision.allowed = False
            decision.reasons.append(f"invalid_metric_field:{query.metric_field}")
        for key in query.filters.keys():
            if key not in schema:
                decision.allowed = False
                decision.reasons.append(f"invalid_filter_field:{key}")
        approved_join_keys = set(contract.get("approved_join_keys", []))
        for join in query.joins:
            join_key = str(join.get("on", ""))
            if join_key and join_key not in approved_join_keys:
                decision.allowed = False
                decision.reasons.append(f"join_key_not_approved:{join_key}")
        if cross_agency and query.joins:
            decision.allowed = False
            decision.reasons.append("cross_agency_join_denied")
        return decision.allowed, decision.reasons

    def _plan_hash(self, plan_payload: Dict[str, Any]) -> str:
        body = str(sorted(plan_payload.items()))
        return hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]

    def _node_to_agency(self, topology_payload: Dict[str, Any]) -> Dict[str, str]:
        return {str(n.get("node_id", "")): str(n.get("agency_id", n.get("node_id", ""))) for n in topology_payload.get("nodes", [])}

    def _cross_agency_for_steps(self, steps: List[Any], topology_payload: Dict[str, Any]) -> bool:
        node_to_agency = self._node_to_agency(topology_payload)
        agencies = {str(node_to_agency.get(str(step.node_id), str(step.node_id))) for step in steps}
        return len(agencies) > 1

    def run_query(
        self,
        query_text: str,
        context: AccessContext,
        actor: Optional[str] = None,
        topology_version_id: Optional[str] = None,
        k_threshold: int = 3,
        project_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        actor_value = actor or context.user_id

        query = parse_query(query_text)
        topology_record = self.topology_store.latest()
        topology_payload = self.topology_store.get_payload(topology_version_id)
        topology_version = topology_version_id or (topology_record["version_id"] if topology_record else "none")

        locations = self.catalog.dataset_locations(query.dataset_id)
        try:
            initial_plan = plan_query(query, locations, topology_payload, topology_version)
        except Exception as exc:
            event = self.audit_log.append(
                event_type="query_denied",
                actor=actor_value,
                payload={
                    "reasons": [f"planning_failed:{exc}"],
                    "dataset": query.dataset_id,
                    "topology_version": topology_version,
                },
                scope="federated_databases",
            )
            return {
                "allowed": False,
                "reasons": [f"planning_failed:{exc}"],
                "audit": event,
                "plan": {"dataset": query.dataset_id, "nodes": [], "cross_agency": False},
            }
        project = self.projects.get(project_id or "default_monitoring")

        project_reasons: List[str] = []
        excluded_by_project: List[Dict[str, str]] = []
        filtered_steps = list(initial_plan.steps)
        if project:
            if query.dataset_id not in set(project.get("allowed_datasets", [])):
                project_reasons.append(f"dataset_not_allowed_by_project:{query.dataset_id}")
            if query.operation == "aggregate":
                if "analytics" not in set(project.get("allowed_computations", [])):
                    project_reasons.append("analytics_not_allowed_by_project")
            allowed_nodes = set(project.get("participants", []))
            if allowed_nodes:
                filtered_steps = [step for step in initial_plan.steps if step.node_id in allowed_nodes]
                excluded_by_project = [
                    {"node_id": step.node_id, "reason": "node_not_in_project_participants"}
                    for step in initial_plan.steps
                    if step.node_id not in allowed_nodes
                ]
                if not filtered_steps:
                    project_reasons.append("no_project_permitted_nodes_for_dataset")
            purposes = set(project.get("governance", {}).get("purpose_allowlist", []))
            if purposes and context.purpose.lower() not in {p.lower() for p in purposes}:
                project_reasons.append(f"purpose_not_allowed_by_project:{context.purpose}")
        else:
            project_reasons.append("project_not_found")

        plan = ExecutionPlan(
            query=initial_plan.query,
            steps=filtered_steps,
            cross_agency=self._cross_agency_for_steps(filtered_steps, topology_payload),
            topology_version=initial_plan.topology_version,
        )

        allowed, reasons = self.evaluate_policy(context, query, cross_agency=plan.cross_agency)
        reasons = [r for r in reasons if str(r).lower() != "allowed"]
        reasons = list(reasons) + project_reasons
        if not allowed:
            event = self.audit_log.append(
                event_type="query_denied",
                actor=actor_value,
                payload={
                    "reasons": reasons,
                    "dataset": query.dataset_id,
                    "topology_version": topology_version,
                },
                scope="federated_databases",
            )
            return {
                "allowed": False,
                "reasons": reasons,
                "audit": event,
                "plan": {
                    "dataset": query.dataset_id,
                    "nodes": [s.node_id for s in initial_plan.steps],
                    "cross_agency": plan.cross_agency,
                },
            }

        if reasons:
            event = self.audit_log.append(
                event_type="query_denied",
                actor=actor_value,
                payload={
                    "reasons": reasons,
                    "dataset": query.dataset_id,
                    "topology_version": topology_version,
                    "project_id": project_id or "default_monitoring",
                },
                scope="federated_databases",
            )
            return {
                "allowed": False,
                "reasons": reasons,
                "audit": event,
                "plan": {
                    "dataset": query.dataset_id,
                    "nodes": [s.node_id for s in initial_plan.steps],
                    "cross_agency": plan.cross_agency,
                },
            }

        connector_specs = {c["connector_id"]: c for c in self.catalog.connectors()}
        executor = FederatedExecutor(connector_specs)
        dataset_mapping = self.canonical.get(query.dataset_id)
        canonical_map = resolve_field_map(dataset_mapping)
        compatibility_report = self.run_compatibility_analysis(
            dataset_id=query.dataset_id,
            operation=query.operation,
            project_id=project_id or "default_monitoring",
            actor=actor_value,
        )

        compatible_nodes = sorted({node for basket in compatibility_report.get("baskets", []) for node in basket.get("members", [])})
        if not compatible_nodes:
            event = self.audit_log.append(
                event_type="query_denied",
                actor=actor_value,
                payload={
                    "reasons": ["no_compatible_nodes_for_query"],
                    "dataset": query.dataset_id,
                    "topology_version": topology_version,
                    "project_id": project_id or "default_monitoring",
                },
                scope="federated_databases",
            )
            return {
                "allowed": False,
                "reasons": ["no_compatible_nodes_for_query"],
                "audit": event,
                "plan": {
                    "dataset": query.dataset_id,
                    "nodes": [s.node_id for s in plan.steps],
                    "cross_agency": plan.cross_agency,
                    "baskets": compatibility_report.get("baskets", []),
                },
                "compatibility": compatibility_report,
            }

        step_by_node = {step.node_id: step for step in plan.steps}
        merged: Dict[tuple, Dict[str, Any]] = {}
        execution_trace: List[Dict[str, Any]] = []
        basket_summaries: List[Dict[str, Any]] = []

        for basket in compatibility_report.get("baskets", []):
            basket_id = str(basket.get("basket_id", "unknown"))
            members = [m for m in basket.get("members", []) if m in step_by_node]
            if not members:
                continue
            basket_steps = [step_by_node[m] for m in members]
            basket_plan = ExecutionPlan(
                query=query,
                steps=basket_steps,
                cross_agency=self._cross_agency_for_steps(basket_steps, topology_payload),
                topology_version=topology_version,
            )
            basket_result = executor.execute(basket_plan, k_threshold=1, canonical_field_map=canonical_map)
            basket_rows = basket_result.get("combined_rows", [])
            basket_summaries.append(
                {
                    "basket_id": basket_id,
                    "tier": basket.get("tier", "partial"),
                    "members": members,
                    "row_count": len(basket_rows),
                    "suppressed_count": 0,
                }
            )

            for trace in basket_result.get("execution_trace", []):
                trace_row = dict(trace)
                trace_row["basket_id"] = basket_id
                trace_row["basket_tier"] = basket.get("tier", "partial")
                execution_trace.append(trace_row)

            for row in basket_rows:
                key = tuple(row.get(col) for col in query.group_by)
                metric_val = float(row.get("metric_value", 0.0) or 0.0)
                support_val = int(row.get("_support", 0) or 0)
                node_contrib = dict(row.get("_node_contrib", {}))
                if key not in merged:
                    merged[key] = {col: row.get(col) for col in query.group_by}
                    merged[key]["metric_value"] = 0.0
                    merged[key]["_support"] = 0
                    merged[key]["_node_contrib"] = {}
                merged[key]["metric_value"] += metric_val
                merged[key]["_support"] += support_val
                existing = merged[key]["_node_contrib"]
                for node_id, node_value in node_contrib.items():
                    existing[node_id] = float(existing.get(node_id, 0.0) + float(node_value or 0.0))

        combined_rows = list(merged.values())
        visible_rows, suppressed = apply_k_suppression(combined_rows, group_by=query.group_by, k_threshold=k_threshold)
        non_iid = summarize_non_iid(combined_rows)
        for row in visible_rows:
            row.pop("_node_contrib", None)

        required_fields = [x.get("canonical_name") for x in (dataset_mapping or {}).get("fields", []) if x.get("canonical_name")]
        dq_summary = quality_summary(visible_rows, required_fields=required_fields[:5])

        candidate_nodes = sorted({s.node_id for s in plan.steps})
        contributing_nodes = sorted({t.get("node_id") for t in execution_trace if int(t.get("returned_rows", 0) or 0) > 0})
        compatible_by_node = {n.get("node_id"): float(n.get("score", 0.0) or 0.0) for n in compatibility_report.get("node_scores", [])}
        contribution_score = 0.0
        if contributing_nodes:
            contribution_score = sum(compatible_by_node.get(node, 0.0) for node in contributing_nodes) / len(contributing_nodes)
        coverage_score = float(len(contributing_nodes) / max(1, len(candidate_nodes)))
        quality_score = float((dq_summary.get("completeness", 0.0) + contribution_score) / 2.0)

        node_to_agency = self._node_to_agency(topology_payload)
        contributing_institutions = sorted({node_to_agency.get(node, node) for node in contributing_nodes})

        excluded_nodes: Dict[str, Dict[str, Any]] = {}
        for row in excluded_by_project:
            node = str(row.get("node_id", ""))
            if node and node not in excluded_nodes:
                excluded_nodes[node] = {"node_id": node, "reason": row.get("reason", "project_excluded")}
        for row in compatibility_report.get("excluded_nodes", []):
            node = str(row.get("node_id", ""))
            if node and node not in excluded_nodes:
                excluded_nodes[node] = {
                    "node_id": node,
                    "reason": row.get("reason", "compatibility_below_threshold"),
                    "score": row.get("score"),
                }
        for node in candidate_nodes:
            if node not in compatible_nodes and node not in excluded_nodes:
                excluded_nodes[node] = {"node_id": node, "reason": "not_in_compatible_basket"}

        excluded_institutions = sorted(
            [
                {
                    "institution_id": node_to_agency.get(row["node_id"], row["node_id"]),
                    "node_id": row["node_id"],
                    "reason": row.get("reason", "excluded"),
                    "score": row.get("score"),
                }
                for row in excluded_nodes.values()
            ],
            key=lambda x: (x["institution_id"], x["node_id"]),
        )

        provenance = {
            "contributing_institutions": contributing_institutions,
            "contributing_nodes": contributing_nodes,
            "excluded_institutions": excluded_institutions,
            "coverage_score": round(coverage_score, 4),
            "quality_score": round(quality_score, 4),
            "aggregation_level_applied": "cross_agency_aggregate_only" if plan.cross_agency else "intra_agency_aggregate",
        }

        plan_payload = {
            "dataset": query.dataset_id,
            "metric": query.metric,
            "group_by": query.group_by,
            "nodes": [s.node_id for s in plan.steps],
            "cross_agency": plan.cross_agency,
            "topology_version": topology_version,
            "project_id": project_id or "default_monitoring",
            "baskets": basket_summaries,
            "execution_trace": execution_trace,
        }
        plan_hash = self._plan_hash(plan_payload)
        event = self.audit_log.append(
            event_type="query_executed",
            actor=actor_value,
            payload={
                "who": context.user_id,
                "purpose": context.purpose,
                "role": context.role,
                "policy_version": self.policy._store.latest().version_id if self.policy._store.latest() else "none",
                "topology_version": topology_version,
                "contract_version": self.contracts._store.latest().version_id if self.contracts._store.latest() else "none",
                "canonical_mapping_version": self.canonical._store.latest().version_id if self.canonical._store.latest() else "none",
                "project_version": self.projects._store.latest().version_id if self.projects._store.latest() else "none",
                "compatibility_version": compatibility_report.get("version_id", "none"),
                "datasets": [query.dataset_id],
                "plan_hash": plan_hash,
                "output_row_count": len(visible_rows),
                "suppression_events": len(suppressed),
                "coverage_score": provenance["coverage_score"],
                "quality_score": provenance["quality_score"],
                "excluded_nodes": excluded_institutions,
                "non_iid_diagnostics": non_iid,
                "data_quality": dq_summary,
            },
            scope="federated_databases",
        )

        return {
            "allowed": True,
            "rows": visible_rows,
            "suppressed": suppressed,
            "plan": plan_payload,
            "plan_hash": plan_hash,
            "execution_trace": execution_trace,
            "compatibility": compatibility_report,
            "baskets": basket_summaries,
            "provenance": provenance,
            "non_iid_diagnostics": non_iid,
            "data_quality": dq_summary,
            "audit": event,
        }

    def run_guided_walkthrough(
        self,
        *,
        actor: str,
        context: AccessContext,
        project_id: str = "default_monitoring",
        dataset_id: str = "local_samples",
        query_text: str = "SELECT county, COUNT(*) FROM local_samples GROUP BY county",
        k_threshold: int = 3,
    ) -> Dict[str, Any]:
        """Runnable 6-step walkthrough for registration -> analytics -> FL readiness handoff."""
        registration = self.register_default_from_manager(actor=actor)
        health = self.connector_health(dataset_id=dataset_id)

        connectors = self.catalog.connectors()
        participants = sorted({c.get("node_id", "") for c in connectors if c.get("node_id")})
        project = self.projects.get(project_id)
        if not project:
            self.upsert_project(
                CollaborationProject(
                    project_id=project_id,
                    name="Monitoring Collaboration",
                    objective="Cross-institution federated analytics",
                    participants=participants,
                    allowed_datasets=[dataset_id],
                    allowed_domains=["intel", "finance", "security"],
                    allowed_computations=["analytics", "federated_ml"],
                    governance={"k_threshold_min": max(2, int(k_threshold)), "purpose_allowlist": ["monitoring", "research", "casework"]},
                ),
                actor=actor,
            )
            project = self.projects.get(project_id)

        contract = self.contracts.get(dataset_id)
        mapping = self.canonical.get(dataset_id)
        compatibility = self.run_compatibility_analysis(
            dataset_id=dataset_id,
            operation="aggregate",
            project_id=project_id,
            actor=actor,
        )
        query_result: Dict[str, Any]
        try:
            query_result = self.run_query(
                query_text=query_text,
                context=context,
                actor=actor,
                k_threshold=k_threshold,
                project_id=project_id,
            )
        except Exception as exc:
            query_result = {"allowed": False, "reasons": [f"query_execution_error:{exc}"]}

        checks = {
            "heterogeneous_sources_connect": all(x.get("healthy", False) for x in health) and bool(health),
            "compatibility_scoring_runs": bool(compatibility.get("node_scores") is not None),
            "baskets_form_automatically": bool(compatibility.get("baskets")),
            "federated_analytics_executes": bool(query_result.get("allowed")),
            "provenance_visible": bool(query_result.get("provenance")),
            "excluded_nodes_explained": bool((query_result.get("provenance") or {}).get("excluded_institutions") is not None),
        }
        checks["walkthrough_complete_without_manual_fixes"] = all(checks.values())
        db_hard_problems = self.hard_problems.assess(
            health=health,
            compatibility=compatibility,
            query_result=query_result,
        )

        return {
            "steps": {
                "1_registration": {"synced": registration, "health": health},
                "2_dataset_publishing": {
                    "contract_present": bool(contract),
                    "mapping_present": bool(mapping),
                    "mapping_quality": (mapping or {}).get("quality", {}),
                },
                "3_project_creation": {
                    "project_id": project_id,
                    "participants": (project or {}).get("participants", []),
                    "allowed_datasets": (project or {}).get("allowed_datasets", []),
                },
                "4_compatibility_analysis": compatibility,
                "5_federated_analytics": query_result,
                "6_fl_ready_input": {
                    "baskets": compatibility.get("baskets", []),
                    "coverage_score": (query_result.get("provenance") or {}).get("coverage_score"),
                    "quality_score": (query_result.get("provenance") or {}).get("quality_score"),
                    "aggregation_level_applied": (query_result.get("provenance") or {}).get("aggregation_level_applied"),
                },
                "7_db_hard_problems": db_hard_problems,
            },
            "quality_gates": checks,
        }

    def audit_rows(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.audit_log.list(limit=limit)
