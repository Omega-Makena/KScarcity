from __future__ import annotations

import sqlite3
from pathlib import Path

from federated_databases.connectors.base import ConnectorSpec
from federated_databases.contracts.mapping import CanonicalFieldMapping, DatasetCanonicalMapping
from federated_databases.contracts.models import DataContract
from federated_databases.control_plane import FederatedDatabaseControlPlane
from federated_databases.policy.engine import AccessContext
from federated_ml.orchestration.nested import NestedFederatedMLOrchestrator
from k_collab.audit.log import AppendOnlyAuditLog
from k_collab.projects.registry import CollaborationProject
from k_collab.topology.store import TopologyStore


def _seed_node_db(path: Path, county: str, criticality: float) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS local_samples (
            timestamp TEXT,
            county TEXT,
            sector TEXT,
            criticality REAL
        );
        DELETE FROM local_samples;
        """
    )
    for idx in range(6):
        conn.execute(
            "INSERT INTO local_samples(timestamp, county, sector, criticality) VALUES (?, ?, ?, ?)",
            (f"2026-02-1{idx}T00:00:00", county, "security", float(criticality + idx * 0.01)),
        )
    conn.commit()
    conn.close()


def _seed_topology(store: TopologyStore) -> None:
    store.save(
        {
            "name": "kcollab_test_topology",
            "nodes": [
                {
                    "node_id": "agency_a",
                    "level": 1,
                    "node_type": "agency",
                    "parent_id": None,
                    "agency_id": "agency_a",
                    "domains": ["intel"],
                    "clearance": "SECRET",
                },
                {
                    "node_id": "agency_b",
                    "level": 1,
                    "node_type": "agency",
                    "parent_id": None,
                    "agency_id": "agency_b",
                    "domains": ["intel"],
                    "clearance": "SECRET",
                },
                {
                    "node_id": "dep_a",
                    "level": 2,
                    "node_type": "department",
                    "parent_id": "agency_a",
                    "agency_id": "agency_a",
                    "domains": ["intel"],
                    "clearance": "RESTRICTED",
                },
                {
                    "node_id": "dep_b",
                    "level": 2,
                    "node_type": "department",
                    "parent_id": "agency_b",
                    "agency_id": "agency_b",
                    "domains": ["intel"],
                    "clearance": "RESTRICTED",
                },
            ],
            "trust_edges": [{"source": "dep_a", "target": "dep_b", "channel": "cross_agency_aggregate"}],
        },
        actor="test",
        message="seed_topology",
    )


def _build_control_plane(tmp_path: Path) -> FederatedDatabaseControlPlane:
    topology = TopologyStore(tmp_path / "topology")
    _seed_topology(topology)
    audit = AppendOnlyAuditLog(tmp_path / "audit.jsonl")
    cp = FederatedDatabaseControlPlane(base_dir=tmp_path / "cp", audit_log=audit, topology_store=topology)

    node_a = tmp_path / "dep_a.sqlite"
    node_b = tmp_path / "dep_b.sqlite"
    _seed_node_db(node_a, county="Nairobi", criticality=0.85)
    _seed_node_db(node_b, county="Mombasa", criticality=0.15)

    cp.register_connector(
        ConnectorSpec(
            connector_id="sqlite::dep_a",
            node_id="dep_a",
            source_type="sqlite",
            location=str(node_a),
            dataset_ids=["local_samples"],
            options={
                "channel_security": "internal_mesh",
                "attestation_status": "verified",
                "max_classification": "SECRET",
                "time_grain": "day",
                "quality": {"completeness": 0.95, "missing_rate": 0.05, "freshness_hours": 6, "timestamp_completeness": 0.95},
                "profile": {
                    "iid_score": 0.9,
                    "features": ["county", "criticality", "timestamp"],
                    "feature_stats": {"criticality": {"mean": 0.86, "std": 0.06}},
                },
                "supported_operations": ["aggregate", "time_bucket"],
                "supported_pushdowns": ["aggregation", "group_by", "filters", "time_bucket"],
                "supports_aggregation": True,
            },
        ),
        actor="test",
    )
    cp.register_connector(
        ConnectorSpec(
            connector_id="sqlite::dep_b",
            node_id="dep_b",
            source_type="sqlite",
            location=str(node_b),
            dataset_ids=["local_samples"],
            options={
                "channel_security": "internal_mesh",
                "attestation_status": "verified",
                "max_classification": "INTERNAL",
                "time_grain": "month",
                "quality": {"completeness": 0.40, "missing_rate": 0.60, "freshness_hours": 160, "timestamp_completeness": 0.4},
                "profile": {
                    "iid_score": 0.15,
                    "features": ["county"],
                    "feature_stats": {"criticality": {"mean": 0.2, "std": 0.8}},
                },
                "supported_operations": ["time_bucket"],
                "supported_pushdowns": ["aggregation"],
                "supports_aggregation": False,
            },
        ),
        actor="test",
    )

    cp.contracts.upsert(
        DataContract(
            dataset_id="local_samples",
            schema={"timestamp": "text", "county": "text", "sector": "text", "criticality": "real"},
            classification="RESTRICTED",
            allowed_operations=["aggregate", "time_bucket"],
            approved_join_keys=["county"],
        ),
        actor="test",
    )
    cp.canonical.upsert(
        DatasetCanonicalMapping(
            dataset_id="local_samples",
            fields=[
                CanonicalFieldMapping("timestamp", "timestamp", dtype="text"),
                CanonicalFieldMapping("county", "county", dtype="text"),
                CanonicalFieldMapping("criticality", "criticality", dtype="real"),
                CanonicalFieldMapping("geo_code", "geo_code", dtype="text"),
            ],
            quality={"freshness_sla_hours": 24, "completeness_min": 0.9},
        ),
        actor="test",
    )
    cp.upsert_project(
        CollaborationProject(
            project_id="p_compat",
            name="Compatibility Project",
            objective="Heterogeneous federated analytics",
            participants=["dep_a", "dep_b"],
            allowed_datasets=["local_samples"],
            allowed_domains=["intel"],
            allowed_computations=["analytics", "federated_ml"],
            governance={
                "purpose_allowlist": ["monitoring"],
                "required_fields": ["timestamp", "county", "criticality", "geo_code"],
                "required_features": ["county", "criticality", "timestamp"],
                "reference_profile": {"criticality": {"mean": 0.85, "std": 0.05}},
                "required_clearance": "SECRET",
                "required_pushdowns": ["aggregation", "group_by", "filters"],
                "max_freshness_hours": 24,
                "time_grain": "day",
            },
        ),
        actor="test",
    )
    return cp


def test_compatibility_scoring_and_basket_formation(tmp_path: Path):
    cp = _build_control_plane(tmp_path)
    report = cp.run_compatibility_analysis(dataset_id="local_samples", operation="aggregate", project_id="p_compat", actor="tester")

    assert report["version_id"]
    assert len(report["node_scores"]) == 2
    assert report["baskets"], "Expected at least one compatible basket"
    excluded_nodes = {x["node_id"] for x in report["excluded_nodes"]}
    assert "dep_b" in excluded_nodes
    assert any("dep_a" in basket["members"] for basket in report["baskets"])


def test_query_execution_uses_baskets_and_returns_provenance(tmp_path: Path):
    cp = _build_control_plane(tmp_path)
    result = cp.run_query(
        query_text="SELECT county, COUNT(*) FROM local_samples GROUP BY county",
        context=AccessContext(user_id="boss", role="supervisor", clearance="SECRET", purpose="monitoring"),
        actor="boss",
        k_threshold=2,
        project_id="p_compat",
    )

    assert result["allowed"] is True
    assert result["rows"]
    assert result["baskets"]
    assert result["provenance"]["coverage_score"] < 1.0
    assert result["provenance"]["contributing_institutions"]
    assert any(item["node_id"] == "dep_b" for item in result["provenance"]["excluded_institutions"])


def test_fl_readiness_consumes_baskets_from_data_layer(tmp_path: Path):
    cp = _build_control_plane(tmp_path)
    compatibility = cp.run_compatibility_analysis(dataset_id="local_samples", operation="aggregate", project_id="p_compat", actor="tester")

    fl = NestedFederatedMLOrchestrator(
        base_dir=tmp_path / "ml",
        audit_log=cp.audit_log,
        topology_store=cp.topology_store,
        project_registry=cp.projects,
    )
    readiness = fl.readiness_from_baskets(
        baskets=compatibility["baskets"],
        project_id="p_compat",
        min_participants_per_basket=1,
        min_remaining_epsilon=0.5,
    )

    assert readiness["privacy"]["privacy_budget_compatible"] is True
    assert readiness["ready"] is True
    assert readiness["usable_baskets"]


def test_guided_walkthrough_completes_quality_gates(tmp_path: Path, monkeypatch):
    cp = _build_control_plane(tmp_path)
    monkeypatch.setattr(cp, "register_default_from_manager", lambda actor="system": {"nodes": 2, "connectors": 2})

    result = cp.run_guided_walkthrough(
        actor="coordinator",
        context=AccessContext(user_id="boss", role="supervisor", clearance="SECRET", purpose="monitoring"),
        project_id="p_compat",
        query_text="SELECT county, COUNT(*) FROM local_samples GROUP BY county",
        k_threshold=2,
    )

    gates = result["quality_gates"]
    assert gates["heterogeneous_sources_connect"] is True
    assert gates["compatibility_scoring_runs"] is True
    assert gates["baskets_form_automatically"] is True
    assert gates["federated_analytics_executes"] is True
    assert gates["provenance_visible"] is True
    assert gates["excluded_nodes_explained"] is True
    assert gates["walkthrough_complete_without_manual_fixes"] is True
    assert result["steps"]["7_db_hard_problems"]["overall_status"] in {"pass", "warn"}
    assert len(result["steps"]["7_db_hard_problems"]["problems"]) == 7


def test_fl_hard_problem_split_report(tmp_path: Path):
    cp = _build_control_plane(tmp_path)
    compatibility = cp.run_compatibility_analysis(dataset_id="local_samples", operation="aggregate", project_id="p_compat", actor="tester")

    fl = NestedFederatedMLOrchestrator(
        base_dir=tmp_path / "ml",
        audit_log=cp.audit_log,
        topology_store=cp.topology_store,
        project_registry=cp.projects,
    )
    readiness = fl.readiness_from_baskets(
        baskets=compatibility["baskets"],
        project_id="p_compat",
        min_participants_per_basket=1,
        min_remaining_epsilon=0.5,
    )
    report = fl.assess_hard_problems(
        readiness=readiness,
        round_output={"non_iid_diagnostics": {"is_highly_non_iid": True, "mean_cosine_distance": 0.95}},
    )

    assert report["summary"]["passed"] + report["summary"]["warned"] + report["summary"]["failed"] == 7
    assert any(p["id"] == "fl_3_non_iid_drift" and p["status"] == "warn" for p in report["problems"])
