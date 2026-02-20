"""Safe query parser supporting JSON DSL and small SQL subset."""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from .models import QueryRequest

_SQL_PATTERN = re.compile(
    r"^SELECT\s+(?P<select>.+?)\s+FROM\s+(?P<table>[a-zA-Z_][\w]*)"
    r"(?:\s+WHERE\s+(?P<where>.+?))?"
    r"(?:\s+GROUP\s+BY\s+(?P<groupby>.+))?$",
    re.IGNORECASE,
)


def _parse_where(where_clause: str) -> Dict[str, Any]:
    filters: Dict[str, Any] = {}
    if not where_clause:
        return filters
    for fragment in where_clause.split("AND"):
        fragment = fragment.strip()
        if "=" not in fragment:
            continue
        key, value = fragment.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            filters[key] = value
    return filters


def parse_query(raw_query: str) -> QueryRequest:
    query = (raw_query or "").strip()
    if not query:
        raise ValueError("Query cannot be empty")

    if query.startswith("{"):
        payload = json.loads(query)
        return QueryRequest(
            dataset_id=str(payload.get("dataset_id", "local_samples")),
            operation=str(payload.get("operation", "aggregate")).lower(),
            group_by=[str(v) for v in payload.get("group_by", [])],
            metric=str(payload.get("metric", "count")).lower(),
            metric_field=str(payload.get("metric_field", "*")).lower(),
            filters=dict(payload.get("filters", {})),
            joins=list(payload.get("joins", [])),
        )

    match = _SQL_PATTERN.match(query)
    if not match:
        raise ValueError("Only JSON DSL or simple SELECT SQL subset is supported")

    select_expr = match.group("select")
    table = match.group("table")
    where_clause = match.group("where") or ""
    groupby = [s.strip() for s in (match.group("groupby") or "").split(",") if s.strip()]

    metric = "count"
    metric_field = "*"

    select_lower = select_expr.lower()
    if "count(" in select_lower:
        metric = "count"
    elif "avg(" in select_lower:
        metric = "avg"
        inner = re.findall(r"avg\(([^\)]+)\)", select_lower)
        metric_field = inner[0].strip() if inner else "criticality"
    elif "sum(" in select_lower:
        metric = "sum"
        inner = re.findall(r"sum\(([^\)]+)\)", select_lower)
        metric_field = inner[0].strip() if inner else "criticality"

    return QueryRequest(
        dataset_id=table,
        operation="aggregate",
        group_by=groupby,
        metric=metric,
        metric_field=metric_field,
        filters=_parse_where(where_clause),
        joins=[],
    )
