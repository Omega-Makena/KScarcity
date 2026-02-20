from __future__ import annotations

import sqlite3
from pathlib import Path

from federated_databases.connectors.base import ConnectorSpec
from federated_databases.contracts.models import DataContract
from federated_databases.control_plane import FederatedDatabaseControlPlane
from federated_databases.executor.engine import apply_k_suppression
from federated_databases.planner.models import QueryRequest
from federated_databases.planner.router import plan_query
from federated_databases.policy.engine import AccessContext, PolicyEngine
from k_collab.projects.registry import CollaborationProject
from k_collab.audit.log import AppendOnlyAuditLog
from k_collab.topology.store import TopologyStore


def _seed_node_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS local_samples (
            county TEXT,
            sector TEXT,
            criticality REAL,
            threat_score REAL,
            escalation_score REAL,
            coordination_score REAL,
            urgency_rate REAL,
            imperative_rate REAL,
            policy_severity REAL,
            label INTEGER
        );
        DELETE FROM local_samples;
        INSERT INTO local_samples VALUES
            ('Nairobi', 'security', 0.8, 0.8, 0.7, 0.6, 0.3, 0.2, 0.5, 1),
            ('Nairobi', 'security', 0.9, 0.9, 0.8, 0.6, 0.4, 0.3, 0.6, 1),
            ('Mombasa', 'finance', 0.3, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0);
        """
    )
    conn.commit()
    conn.close()


def _build_topology(store: TopologyStore) -> None:
    store.save(
        {
            "name": "test_topology",
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
                    "node_id": "dep_a",
                    "level": 2,
                    "node_type": "department",
                    "parent_id": "agency_a",
                    "agency_id": "agency_a",
                    "domains": ["intel"],
                    "clearance": "RESTRICTED",
                },
            ],
            "trust_edges": [],
        },
        actor="test",
        message="seed",
    )


def test_contract_enforcement_denies_disallowed_operation(tmp_path: Path):
    topology = TopologyStore(tmp_path / "topology")
    _build_topology(topology)

    db_path = tmp_path / "node_a.sqlite"
    _seed_node_db(db_path)

    audit = AppendOnlyAuditLog(tmp_path / "audit.jsonl")
    cp = FederatedDatabaseControlPlane(base_dir=tmp_path / "cp", audit_log=audit, topology_store=topology)

    cp.catalog.register_connector(
        ConnectorSpec(
            connector_id="sqlite::dep_a",
            node_id="dep_a",
            source_type="sqlite",
            location=str(db_path),
            dataset_ids=["local_samples"],
        ),
        actor="test",
    )
    cp.contracts.upsert(
        DataContract(
            dataset_id="local_samples",
            schema={"county": "text", "criticality": "real"},
            classification="RESTRICTED",
            allowed_operations=["aggregate"],
        ),
        actor="test",
    )

    result = cp.run_query(
        query_text='{"dataset_id":"local_samples","operation":"raw_export","group_by":["county"],"metric":"count"}',
        context=AccessContext(user_id="u1", role="analyst", clearance="RESTRICTED", purpose="monitoring"),
        actor="u1",
    )

    assert result["allowed"] is False
    assert "operation_not_allowed_by_contract" in result["reasons"]


def test_policy_evaluation_clearance_gate(tmp_path: Path):
    engine = PolicyEngine(tmp_path / "policy")

    decision = engine.evaluate(
        context=AccessContext(user_id="u2", role="analyst", clearance="INTERNAL", purpose="monitoring"),
        dataset_contracts=[{"dataset_id": "d1", "classification": "SECRET"}],
        operation="aggregate",
        cross_agency=False,
    )

    assert decision.allowed is False
    assert any("clearance_too_low_for_dataset" in r for r in decision.reasons)


def test_planner_routes_nodes_and_marks_cross_agency():
    query = QueryRequest(dataset_id="local_samples", operation="aggregate", group_by=["county"], metric="count")
    plan = plan_query(
        query=query,
        dataset_locations=[
            {"connector_id": "c1", "node_id": "dep_a"},
            {"connector_id": "c2", "node_id": "dep_b"},
        ],
        topology_payload={
            "nodes": [
                {"node_id": "dep_a", "agency_id": "agency_a"},
                {"node_id": "dep_b", "agency_id": "agency_b"},
            ]
        },
        topology_version="v1",
    )

    assert len(plan.steps) == 2
    assert plan.cross_agency is True
    assert {s.node_id for s in plan.steps} == {"dep_a", "dep_b"}


def test_k_suppression_hides_small_groups():
    rows = [
        {"county": "Nairobi", "metric_value": 10.0, "_support": 5},
        {"county": "Mombasa", "metric_value": 2.0, "_support": 1},
    ]

    visible, suppressed = apply_k_suppression(rows, group_by=["county"], k_threshold=3)

    assert len(visible) == 1
    assert visible[0]["county"] == "Nairobi"
    assert len(suppressed) == 1
    assert suppressed[0]["group"]["county"] == "Mombasa"


def test_connector_registration_blocks_inline_secret(tmp_path: Path):
    topology = TopologyStore(tmp_path / "topology")
    _build_topology(topology)
    audit = AppendOnlyAuditLog(tmp_path / "audit.jsonl")
    cp = FederatedDatabaseControlPlane(base_dir=tmp_path / "cp", audit_log=audit, topology_store=topology)

    try:
        cp.register_connector(
            ConnectorSpec(
                connector_id="pg::dep_a",
                node_id="dep_a",
                source_type="postgres",
                location="postgresql://example",
                dataset_ids=["local_samples"],
                options={"password": "secret123", "channel_security": "mtls"},
            ),
            actor="test",
        )
    except ValueError as exc:
        assert "inline_secret_not_allowed" in str(exc)
    else:
        raise AssertionError("Expected inline secret connector validation to fail")


def test_project_enforcement_denies_unlisted_dataset(tmp_path: Path):
    topology = TopologyStore(tmp_path / "topology")
    _build_topology(topology)
    db_path = tmp_path / "node_a.sqlite"
    _seed_node_db(db_path)
    audit = AppendOnlyAuditLog(tmp_path / "audit.jsonl")
    cp = FederatedDatabaseControlPlane(base_dir=tmp_path / "cp", audit_log=audit, topology_store=topology)

    cp.register_connector(
        ConnectorSpec(
            connector_id="sqlite::dep_a",
            node_id="dep_a",
            source_type="sqlite",
            location=str(db_path),
            dataset_ids=["local_samples"],
            options={"channel_security": "internal_mesh", "attestation_status": "verified"},
        ),
        actor="test",
    )
    cp.contracts.upsert(
        DataContract(
            dataset_id="local_samples",
            schema={"county": "text"},
            classification="RESTRICTED",
            allowed_operations=["aggregate"],
        ),
        actor="test",
    )
    cp.upsert_project(
        CollaborationProject(
            project_id="p1",
            name="Project 1",
            objective="Restricted project",
            participants=["dep_a"],
            allowed_datasets=["another_dataset"],
            allowed_domains=["intel"],
            allowed_computations=["analytics"],
            governance={"purpose_allowlist": ["monitoring"]},
        ),
        actor="test",
    )

    result = cp.run_query(
        query_text="SELECT county, COUNT(*) FROM local_samples GROUP BY county",
        context=AccessContext(user_id="u1", role="analyst", clearance="RESTRICTED", purpose="monitoring"),
        project_id="p1",
        actor="u1",
    )

    assert result["allowed"] is False
    assert any("dataset_not_allowed_by_project" in reason for reason in result["reasons"])
