"""Postgres connector (real implementation when psycopg is available)."""

from __future__ import annotations

from typing import Any, Dict, List


class PostgresNodeConnector:
    """Executes aggregate queries against a Postgres source."""

    SUPPORTED_METRICS = {"count", "avg", "sum"}

    def __init__(self, dsn: str):
        self.dsn = dsn

    def execute_aggregate(
        self,
        table: str,
        group_by: List[str],
        metric: str,
        metric_field: str,
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        try:
            import psycopg  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("psycopg is required for postgres connector") from exc

        metric_normalized = str(metric or "count").strip().lower()
        if metric_normalized not in self.SUPPORTED_METRICS:
            raise ValueError(f"Unsupported metric: {metric}")

        metric_expr = "COUNT(*)"
        if metric_normalized == "avg":
            metric_expr = f"AVG({metric_field})"
        elif metric_normalized == "sum":
            metric_expr = f"SUM({metric_field})"

        group_cols = [c for c in group_by if c]
        select_cols = ", ".join(group_cols) if group_cols else ""

        where_parts = []
        args = []
        for key, val in (filters or {}).items():
            where_parts.append(f"{key} = %s")
            args.append(val)
        where_sql = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""

        if select_cols:
            sql = f"SELECT {select_cols}, {metric_expr} AS metric_value FROM {table}{where_sql} GROUP BY {select_cols}"
        else:
            sql = f"SELECT {metric_expr} AS metric_value FROM {table}{where_sql}"

        with psycopg.connect(self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, args)
                rows = cur.fetchall()
                cols = [d.name for d in cur.description]

        return [dict(zip(cols, row)) for row in rows]
