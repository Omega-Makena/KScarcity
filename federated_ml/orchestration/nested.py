"""Nested federated ML orchestration that wraps existing scarcity federation logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from k_collab.common.versioned_store import VersionedJSONStore
from federated_ml.audit_hooks.hooks import FederatedMLAuditHooks
from federated_ml.hard_problems import FederatedMLHardProblemAssessor
from federated_ml.registry.model_registry import FederatedModelRegistry
from federated_ml.topology_adapter.adapter import NestedTopologyAdapter
from federated_ml.orchestration.non_iid import update_heterogeneity
from scarcity.federation.hierarchical import HierarchicalFederation, HierarchicalFederationConfig


@dataclass
class ActiveJob:
    job_id: str
    topology_version: str
    task_name: str
    participants: List[Dict[str, str]]


class NestedFederatedMLOrchestrator:
    """K-Collab ML control plane using HierarchicalFederation under the hood."""

    def __init__(self, base_dir: Path | str, audit_log, topology_store, project_registry=None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.audit = FederatedMLAuditHooks(audit_log)
        self.topology_store = topology_store
        self.project_registry = project_registry
        self.registry = FederatedModelRegistry(self.base_dir / "ml_registry")
        self.hard_problems = FederatedMLHardProblemAssessor()
        self.jobs = VersionedJSONStore(self.base_dir / "ml_job_versions.jsonl", kind="ml_job")
        self._federation: Optional[HierarchicalFederation] = None
        self._active: Optional[ActiveJob] = None

    def start_job(
        self,
        actor: str,
        task_name: str,
        selected_nodes: Optional[List[str]] = None,
        topology_version_id: Optional[str] = None,
        vector_dim: int = 8,
        project_id: Optional[str] = None,
        min_remaining_epsilon: float = 0.5,
    ) -> Dict[str, Any]:
        topology_record = self.topology_store.latest()
        topology_version = topology_version_id or (topology_record["version_id"] if topology_record else "none")
        topology_payload = self.topology_store.get_payload(topology_version_id)
        project = self.project_registry.get(project_id or "default_monitoring") if self.project_registry else None
        if project and "federated_ml" not in set(project.get("allowed_computations", [])):
            raise ValueError("Project does not allow federated_ml computation")

        adapter = NestedTopologyAdapter(topology_payload)
        participants = [p.__dict__ for p in adapter.participants(selected_nodes=selected_nodes)]
        if not participants:
            raise ValueError("No participants resolved from selected topology nodes")
        if project:
            allowed = set(project.get("participants", []))
            denied = [p["client_id"] for p in participants if p["client_id"] not in allowed]
            if denied:
                raise ValueError(f"Project does not permit nodes: {denied}")

        config = HierarchicalFederationConfig(vector_dim=int(vector_dim))
        self._federation = HierarchicalFederation(config)
        self._federation._kcollab_min_remaining_epsilon = float(min_remaining_epsilon)  # lightweight runtime guard

        for p in participants:
            self._federation.register_client(p["client_id"], p["domain_id"])

        job_payload = {
            "task_name": task_name,
            "topology_version": topology_version,
            "participants": participants,
            "vector_dim": int(vector_dim),
            "project_id": project_id or "default_monitoring",
            "min_remaining_epsilon": float(min_remaining_epsilon),
        }
        record = self.jobs.save(job_payload, actor=actor, message="ml_job_start")
        self._active = ActiveJob(
            job_id=record.version_id,
            topology_version=topology_version,
            task_name=task_name,
            participants=participants,
        )

        self.audit.job_started(
            actor=actor,
            payload={
                "job_id": self._active.job_id,
                "topology_version": topology_version,
                "participants": len(participants),
            },
        )

        return {
            "job_id": self._active.job_id,
            "participants": participants,
            "topology_version": topology_version,
        }

    def run_round(self, actor: str, updates: Dict[str, List[float]]) -> Dict[str, Any]:
        if not self._active or not self._federation:
            raise RuntimeError("No active ML job")

        eps_remaining, _ = self._federation.get_privacy_budget()
        min_eps = float(getattr(self._federation, "_kcollab_min_remaining_epsilon", 0.5))
        if eps_remaining < min_eps:
            raise RuntimeError(
                f"Privacy budget too low to continue training: remaining_epsilon={eps_remaining:.4f}, minimum_required={min_eps:.4f}"
            )

        participant_ids = {p["client_id"] for p in self._active.participants}
        excluded: List[Dict[str, str]] = []

        for client_id, vector in (updates or {}).items():
            if client_id not in participant_ids:
                excluded.append({"node": client_id, "reason": "not_in_job_topology"})
                continue
            arr = np.asarray(vector, dtype=float)
            self._federation.submit_update(client_id, arr)

        global_model = self._federation.force_aggregate()
        if global_model is None:
            global_model = np.zeros(1, dtype=float)

        basket_models = self._federation.layer1.get_basket_models()

        dept_metrics: Dict[str, Dict[str, Any]] = {}
        agency_metrics: Dict[str, Dict[str, Any]] = {}

        for domain_id, model in basket_models.items():
            dept_metrics[domain_id] = {
                "updates": int(model.update_count),
                "norm": float(np.linalg.norm(model.aggregate_vector)),
            }
            agency_id = domain_id.split(":", 1)[0]
            agency_metrics.setdefault(agency_id, {"updates": 0, "norm_total": 0.0})
            agency_metrics[agency_id]["updates"] += int(model.update_count)
            agency_metrics[agency_id]["norm_total"] += float(np.linalg.norm(model.aggregate_vector))

        global_metrics = {
            "vector_dim": int(global_model.shape[0]),
            "global_norm": float(np.linalg.norm(global_model)),
            "privacy": self._federation.get_stats(),
        }
        non_iid = update_heterogeneity({k: v for k, v in (updates or {}).items() if k in participant_ids})

        round_payload = {
            "job_id": self._active.job_id,
            "department_metrics": dept_metrics,
            "agency_metrics": agency_metrics,
            "global_metrics": global_metrics,
            "non_iid_diagnostics": non_iid,
            "excluded_nodes": excluded,
        }

        self.audit.round_completed(actor=actor, payload=round_payload)
        self.audit.compliance_report(
            actor=actor,
            payload={
                "job_id": self._active.job_id,
                "participating_nodes": sorted(list(participant_ids & set(updates.keys()))),
                "excluded_nodes": excluded,
                "privacy_toggles": {
                    "local_dp": True,
                    "secure_aggregation": bool(self._federation.config.layer2.secure_aggregation),
                    "central_dp": True,
                },
                "non_iid_diagnostics": non_iid,
            },
        )

        return round_payload

    def complete_job(self, actor: str, model_id: str = "") -> Dict[str, Any]:
        if not self._active or not self._federation:
            raise RuntimeError("No active ML job")

        global_model = self._federation.get_global_model()
        if global_model is None:
            global_model = np.zeros(1, dtype=float)

        model_name = model_id or f"ml_{self._active.job_id}"
        rec = self.registry.register(
            model_vector=global_model,
            metadata={
                "model_id": model_name,
                "job_id": self._active.job_id,
                "topology_version": self._active.topology_version,
                "task_name": self._active.task_name,
                "participant_count": len(self._active.participants),
            },
            actor=actor,
        )
        self.audit.job_completed(
            actor=actor,
            payload={
                "job_id": self._active.job_id,
                "model_id": model_name,
                "registry_version": rec["version_id"],
            },
        )
        done = {
            "job_id": self._active.job_id,
            "model_id": model_name,
            "registry_version": rec["version_id"],
        }
        self._active = None
        self._federation = None
        return done

    def readiness_from_baskets(
        self,
        *,
        baskets: List[Dict[str, Any]],
        project_id: Optional[str] = None,
        topology_version_id: Optional[str] = None,
        min_participants_per_basket: int = 2,
        min_remaining_epsilon: float = 0.5,
    ) -> Dict[str, Any]:
        """Validate whether current baskets are usable inputs for FL orchestration."""
        topology_payload = self.topology_store.get_payload(topology_version_id)
        adapter = NestedTopologyAdapter(topology_payload)
        participants = [p.__dict__ for p in adapter.participants()]
        participant_ids = {p["client_id"] for p in participants}

        project = self.project_registry.get(project_id or "default_monitoring") if self.project_registry else None
        project_ok = bool(project and "federated_ml" in set(project.get("allowed_computations", [])))
        project_allowed_nodes = set(project.get("participants", [])) if project else set()

        usable_baskets: List[Dict[str, Any]] = []
        excluded_baskets: List[Dict[str, Any]] = []
        for basket in baskets or []:
            members = [m for m in basket.get("members", []) if m in participant_ids]
            if project_allowed_nodes:
                members = [m for m in members if m in project_allowed_nodes]
            if len(members) >= int(min_participants_per_basket):
                usable_baskets.append(
                    {
                        "basket_id": basket.get("basket_id"),
                        "members": members,
                        "tier": basket.get("tier", "partial"),
                    }
                )
            else:
                excluded_baskets.append(
                    {
                        "basket_id": basket.get("basket_id"),
                        "reason": "insufficient_participants_for_fl",
                        "members_after_filters": members,
                    }
                )

        base_cfg = HierarchicalFederationConfig()
        privacy = {
            "planned_total_epsilon": float(base_cfg.total_epsilon),
            "planned_total_delta": float(base_cfg.total_delta),
            "min_remaining_epsilon_required": float(min_remaining_epsilon),
            "privacy_budget_compatible": float(base_cfg.total_epsilon) >= float(min_remaining_epsilon),
        }

        ready = bool(usable_baskets) and project_ok and privacy["privacy_budget_compatible"]
        reasons: List[str] = []
        if not project_ok:
            reasons.append("project_not_authorized_for_federated_ml")
        if not usable_baskets:
            reasons.append("no_baskets_meet_min_participant_threshold")
        if not privacy["privacy_budget_compatible"]:
            reasons.append("min_remaining_epsilon_exceeds_planned_budget")

        return {
            "ready": ready,
            "usable_baskets": usable_baskets,
            "excluded_baskets": excluded_baskets,
            "participant_pool_size": len(participant_ids),
            "privacy": privacy,
            "reasons": reasons,
        }

    def assess_hard_problems(
        self,
        *,
        readiness: Dict[str, Any],
        round_output: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate FL-side hard-problem status from current orchestration state."""
        return self.hard_problems.assess(
            readiness=readiness,
            round_output=round_output or {},
            status_snapshot=self.status(),
        )

    def status(self) -> Dict[str, Any]:
        return {
            "active_job": self._active.__dict__ if self._active else None,
            "models": self.registry.list_models(),
        }
