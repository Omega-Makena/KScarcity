"""Executor for planner-router output with suppression controls."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from federated_databases.connectors.registry import ConnectorFactory
from federated_databases.contracts.mapping import map_canonical_query
from federated_databases.planner.models import ExecutionPlan
from federated_databases.executor.non_iid import summarize_non_iid


def apply_k_suppression(rows: List[Dict[str, Any]], group_by: List[str], k_threshold: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if k_threshold <= 1 or not rows or not group_by:
        return rows, []

    suppressed: List[Dict[str, Any]] = []
    kept: List[Dict[str, Any]] = []
    for row in rows:
        count_value = int(row.get("_support", 0) or 0)
        if count_value < k_threshold:
            suppressed.append({"group": {k: row.get(k) for k in group_by}, "support": count_value})
            continue
        cleaned = dict(row)
        cleaned.pop("_support", None)
        kept.append(cleaned)
    return kept, suppressed


class FederatedExecutor:
    """Runs execution plans via registered connectors."""

    def __init__(self, connector_specs: Dict[str, Dict[str, Any]]):
        self.connector_specs = connector_specs

    def _connector(self, connector_id: str):
        spec = self.connector_specs.get(connector_id)
        if not spec:
            raise ValueError(f"Connector not found: {connector_id}")
        return ConnectorFactory.create(spec)

    def execute(self, plan: ExecutionPlan, k_threshold: int = 3, canonical_field_map: Dict[str, str] | None = None) -> Dict[str, Any]:
        query = plan.query
        grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = defaultdict(dict)
        execution_trace: List[Dict[str, Any]] = []
        effective_group_by: List[str] = list(query.group_by)

        for step in plan.steps:
            connector = self._connector(step.connector_id)
            group_by_local, filters_local, metric_field_local = map_canonical_query(
                group_by=query.group_by,
                filters=query.filters,
                metric_field=query.metric_field,
                field_map=canonical_field_map or {},
            )
            if step.connector_options.get("field_mapping"):
                group_by_local, filters_local, metric_field_local = map_canonical_query(
                    group_by=group_by_local,
                    filters=filters_local,
                    metric_field=metric_field_local,
                    field_map=dict(step.connector_options.get("field_mapping", {})),
                )
            effective_group_by = list(group_by_local)

            rows = connector.execute_aggregate(
                table=query.dataset_id,
                group_by=group_by_local,
                metric=query.metric,
                metric_field=metric_field_local,
                filters=filters_local,
            )
            execution_trace.append(
                {
                    "node_id": step.node_id,
                    "connector_id": step.connector_id,
                    "pushdown": {
                        "table": query.dataset_id,
                        "group_by": group_by_local,
                        "metric": query.metric,
                        "metric_field": metric_field_local,
                        "filters": filters_local,
                    },
                    "returned_rows": len(rows),
                }
            )
            for row in rows:
                key = tuple(row.get(col) for col in group_by_local)
                metric_value = float(row.get("metric_value", 0.0) or 0.0)
                if not grouped[key]:
                    grouped[key] = {col: row.get(col) for col in group_by_local}
                    grouped[key]["metric_value"] = 0.0
                    grouped[key]["_support"] = 0
                    grouped[key]["_node_contrib"] = {}
                grouped[key]["metric_value"] += metric_value
                grouped[key]["_support"] += int(row.get("metric_value", 0) if query.metric == "count" else 1)
                node_contrib = grouped[key]["_node_contrib"]
                node_contrib[step.node_id] = float(node_contrib.get(step.node_id, 0.0) + metric_value)

        combined = list(grouped.values())
        visible, suppressed = apply_k_suppression(combined, effective_group_by, k_threshold)
        non_iid = summarize_non_iid(combined)
        for row in visible:
            row.pop("_node_contrib", None)

        return {
            "rows": visible,
            "combined_rows": combined,
            "suppressed": suppressed,
            "row_count": len(visible),
            "suppressed_count": len(suppressed),
            "execution_trace": execution_trace,
            "non_iid_diagnostics": non_iid,
        }
