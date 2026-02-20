"""SQLite connector used for node-local federated queries."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List


class SQLiteNodeConnector:
    """Executes limited aggregate queries against node-local SQLite tables."""

    SUPPORTED_METRICS = {"count", "avg", "sum"}

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def execute_aggregate(
        self,
        table: str,
        group_by: List[str],
        metric: str,
        metric_field: str,
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
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

        where_parts: List[str] = []
        args: List[Any] = []
        for key, val in (filters or {}).items():
            where_parts.append(f"{key} = ?")
            args.append(val)
        where_sql = f" WHERE {' AND '.join(where_parts)}" if where_parts else ""

        if select_cols:
            sql = f"SELECT {select_cols}, {metric_expr} AS metric_value FROM {table}{where_sql} GROUP BY {select_cols}"
        else:
            sql = f"SELECT {metric_expr} AS metric_value FROM {table}{where_sql}"

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(args)).fetchall()

        return [dict(row) for row in rows]
