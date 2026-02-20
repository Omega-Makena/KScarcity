"""Shared singleton-style services for Streamlit integration."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

from k_collab.topology.store import TopologyStore
from k_collab.audit.log import AppendOnlyAuditLog
from k_collab.projects.registry import CollaborationProjectRegistry
from federated_databases.control_plane import FederatedDatabaseControlPlane
from federated_ml.orchestration.nested import NestedFederatedMLOrchestrator


def get_kcollab_services(base_dir: Path | str = "federated_databases/runtime/k_collab") -> Dict[str, Any]:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    audit = AppendOnlyAuditLog(base / "audit_chain.jsonl")
    topology = TopologyStore(base)
    projects = CollaborationProjectRegistry(base)
    fed_db = FederatedDatabaseControlPlane(base_dir=base, audit_log=audit, topology_store=topology, project_registry=projects)
    fed_ml = NestedFederatedMLOrchestrator(base_dir=base, audit_log=audit, topology_store=topology, project_registry=projects)
    return {
        "audit": audit,
        "topology": topology,
        "projects": projects,
        "fed_db": fed_db,
        "fed_ml": fed_ml,
    }
