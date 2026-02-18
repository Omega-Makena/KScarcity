"""SQLite storage backends for federated nodes and control-plane state."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class NodeStorage:
    """Per-node storage backend (SQLite)."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS local_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sample_uid TEXT UNIQUE,
                    timestamp TEXT,
                    county TEXT,
                    sector TEXT,
                    criticality REAL,
                    threat_score REAL,
                    escalation_score REAL,
                    coordination_score REAL,
                    urgency_rate REAL,
                    imperative_rate REAL,
                    policy_severity REAL,
                    label INTEGER,
                    ingested_at TEXT
                );

                CREATE TABLE IF NOT EXISTS model_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    round_number INTEGER,
                    created_at TEXT,
                    weights_json TEXT,
                    gradient_norm REAL,
                    loss REAL,
                    sample_count INTEGER,
                    metrics_json TEXT
                );

                CREATE TABLE IF NOT EXISTS shared_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    round_number INTEGER,
                    signal_key TEXT,
                    signal_value REAL,
                    payload_json TEXT,
                    created_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_local_samples_time ON local_samples(timestamp);
                CREATE INDEX IF NOT EXISTS idx_model_updates_round ON model_updates(round_number);
                CREATE INDEX IF NOT EXISTS idx_shared_signals_round ON shared_signals(round_number);
                """
            )
            conn.commit()

    def add_samples(self, samples: List[Dict[str, Any]]) -> int:
        if not samples:
            return 0
        inserted = 0
        with self._connect() as conn:
            cur = conn.cursor()
            for row in samples:
                try:
                    cur.execute(
                        """
                        INSERT OR IGNORE INTO local_samples (
                            sample_uid, timestamp, county, sector, criticality,
                            threat_score, escalation_score, coordination_score,
                            urgency_rate, imperative_rate, policy_severity,
                            label, ingested_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            row.get("sample_uid"),
                            row.get("timestamp"),
                            row.get("county"),
                            row.get("sector"),
                            float(row.get("criticality", 0.0)),
                            float(row.get("threat_score", 0.0)),
                            float(row.get("escalation_score", 0.0)),
                            float(row.get("coordination_score", 0.0)),
                            float(row.get("urgency_rate", 0.0)),
                            float(row.get("imperative_rate", 0.0)),
                            float(row.get("policy_severity", 0.0)),
                            int(row.get("label", 0)),
                            _utc_now(),
                        ),
                    )
                    if cur.rowcount > 0:
                        inserted += 1
                except Exception:
                    continue
            conn.commit()
        return inserted

    def get_training_matrix(self, limit: int = 10000) -> Tuple[List[List[float]], List[int]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    threat_score,
                    escalation_score,
                    coordination_score,
                    urgency_rate,
                    imperative_rate,
                    policy_severity,
                    label
                FROM local_samples
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()

        features: List[List[float]] = []
        labels: List[int] = []
        for row in rows:
            features.append(
                [
                    float(row["threat_score"]),
                    float(row["escalation_score"]),
                    float(row["coordination_score"]),
                    float(row["urgency_rate"]),
                    float(row["imperative_rate"]),
                    float(row["policy_severity"]),
                ]
            )
            labels.append(int(row["label"]))
        return features, labels

    def latest_weights(self) -> Optional[List[float]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT weights_json
                FROM model_updates
                ORDER BY round_number DESC, id DESC
                LIMIT 1
                """
            ).fetchone()
        if not row:
            return None
        try:
            return [float(x) for x in json.loads(row["weights_json"])]
        except Exception:
            return None

    def record_model_update(
        self,
        round_number: int,
        weights: List[float],
        gradient_norm: float,
        loss: float,
        sample_count: int,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO model_updates (
                    round_number, created_at, weights_json,
                    gradient_norm, loss, sample_count, metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(round_number),
                    _utc_now(),
                    json.dumps([float(v) for v in weights]),
                    float(gradient_norm),
                    float(loss),
                    int(sample_count),
                    json.dumps(metrics or {}),
                ),
            )
            conn.commit()

    def record_shared_signal(
        self,
        round_number: int,
        signal_key: str,
        signal_value: float,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO shared_signals (
                    round_number, signal_key, signal_value, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    int(round_number),
                    str(signal_key),
                    float(signal_value),
                    json.dumps(payload or {}),
                    _utc_now(),
                ),
            )
            conn.commit()

    def stats(self) -> Dict[str, Any]:
        with self._connect() as conn:
            sample_count = conn.execute("SELECT COUNT(*) FROM local_samples").fetchone()[0]
            update_count = conn.execute("SELECT COUNT(*) FROM model_updates").fetchone()[0]
            signal_count = conn.execute("SELECT COUNT(*) FROM shared_signals").fetchone()[0]
            latest = conn.execute(
                """
                SELECT round_number, loss, sample_count, created_at
                FROM model_updates
                ORDER BY round_number DESC, id DESC
                LIMIT 1
                """
            ).fetchone()

        result: Dict[str, Any] = {
            "sample_count": int(sample_count),
            "update_count": int(update_count),
            "shared_signal_count": int(signal_count),
        }
        if latest:
            result.update(
                {
                    "last_round": int(latest["round_number"]),
                    "last_loss": float(latest["loss"]),
                    "last_update_samples": int(latest["sample_count"]),
                    "last_update_at": str(latest["created_at"]),
                }
            )
        return result


class ControlPlaneStorage:
    """Federation control-plane storage (SQLite)."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    backend TEXT NOT NULL,
                    db_path TEXT NOT NULL,
                    county_filter TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS sync_rounds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    round_number INTEGER,
                    started_at TEXT,
                    completed_at TEXT,
                    participants INTEGER,
                    total_samples INTEGER,
                    aggregation_method TEXT,
                    global_loss REAL,
                    global_gradient_norm REAL,
                    metrics_json TEXT
                );

                CREATE TABLE IF NOT EXISTS exchange_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    round_number INTEGER,
                    from_node TEXT,
                    to_node TEXT,
                    payload_type TEXT,
                    payload_size INTEGER,
                    created_at TEXT,
                    details_json TEXT
                );

                CREATE TABLE IF NOT EXISTS global_state (
                    state_key TEXT PRIMARY KEY,
                    value_json TEXT,
                    updated_at TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_sync_rounds_round ON sync_rounds(round_number);
                CREATE INDEX IF NOT EXISTS idx_exchange_round ON exchange_audit(round_number);
                """
            )
            conn.commit()

    def upsert_node(
        self,
        node_id: str,
        backend: str,
        db_path: str,
        county_filter: Optional[str] = None,
    ) -> None:
        now = _utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO nodes (node_id, backend, db_path, county_filter, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    backend=excluded.backend,
                    db_path=excluded.db_path,
                    county_filter=excluded.county_filter,
                    updated_at=excluded.updated_at
                """,
                (node_id, backend, db_path, county_filter, now, now),
            )
            conn.commit()

    def list_nodes(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT node_id, backend, db_path, county_filter, created_at, updated_at
                FROM nodes
                ORDER BY node_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def next_round_number(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COALESCE(MAX(round_number), 0) AS r FROM sync_rounds").fetchone()
        return int(row["r"]) + 1

    def record_sync_round(
        self,
        round_number: int,
        started_at: str,
        completed_at: str,
        participants: int,
        total_samples: int,
        aggregation_method: str,
        global_loss: float,
        global_gradient_norm: float,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sync_rounds (
                    round_number, started_at, completed_at, participants,
                    total_samples, aggregation_method, global_loss,
                    global_gradient_norm, metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(round_number),
                    started_at,
                    completed_at,
                    int(participants),
                    int(total_samples),
                    aggregation_method,
                    float(global_loss),
                    float(global_gradient_norm),
                    json.dumps(metrics or {}),
                ),
            )
            conn.commit()

    def record_exchange(
        self,
        round_number: int,
        from_node: str,
        to_node: str,
        payload_type: str,
        payload_size: int,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO exchange_audit (
                    round_number, from_node, to_node, payload_type,
                    payload_size, created_at, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(round_number),
                    from_node,
                    to_node,
                    payload_type,
                    int(payload_size),
                    _utc_now(),
                    json.dumps(details or {}),
                ),
            )
            conn.commit()

    def get_round_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    round_number,
                    started_at,
                    completed_at,
                    participants,
                    total_samples,
                    aggregation_method,
                    global_loss,
                    global_gradient_norm,
                    metrics_json
                FROM sync_rounds
                ORDER BY round_number DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()

        result: List[Dict[str, Any]] = []
        for row in rows:
            payload = dict(row)
            try:
                payload["metrics"] = json.loads(payload.pop("metrics_json") or "{}")
            except Exception:
                payload["metrics"] = {}
            result.append(payload)
        return result

    def get_exchange_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    round_number,
                    from_node,
                    to_node,
                    payload_type,
                    payload_size,
                    created_at,
                    details_json
                FROM exchange_audit
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()

        result: List[Dict[str, Any]] = []
        for row in rows:
            payload = dict(row)
            try:
                payload["details"] = json.loads(payload.pop("details_json") or "{}")
            except Exception:
                payload["details"] = {}
            result.append(payload)
        return result

    def set_global_state(self, key: str, value: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO global_state (state_key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(state_key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at
                """,
                (key, json.dumps(value), _utc_now()),
            )
            conn.commit()

    def get_global_state(self, key: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value_json FROM global_state WHERE state_key = ?",
                (key,),
            ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["value_json"])
        except Exception:
            return None
