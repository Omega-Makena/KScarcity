"""Federated query routing planner."""

from __future__ import annotations

from typing import Dict, List

from .models import QueryRequest, ExecutionPlan, ExecutionStep


def plan_query(
    query: QueryRequest,
    dataset_locations: List[Dict[str, str]],
    topology_payload: Dict[str, object],
    topology_version: str,
) -> ExecutionPlan:
    if not dataset_locations:
        raise ValueError(f"No dataset locations found for {query.dataset_id}")

    steps: List[ExecutionStep] = []
    agencies: set[str] = set()
    node_to_agency = {n["node_id"]: n.get("agency_id", n["node_id"]) for n in topology_payload.get("nodes", [])}

    for loc in dataset_locations:
        node_id = str(loc.get("node_id"))
        connector_id = str(loc.get("connector_id"))
        steps.append(
            ExecutionStep(
                node_id=node_id,
                connector_id=connector_id,
                dataset_id=query.dataset_id,
                connector_options=dict(loc.get("connector_options", {})),
            )
        )
        agencies.add(str(node_to_agency.get(node_id, node_id)))

    cross_agency = len(agencies) > 1
    return ExecutionPlan(query=query, steps=steps, cross_agency=cross_agency, topology_version=topology_version)
