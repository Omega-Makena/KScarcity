"""Scarcity federation manager combining federated databases and ML sync rounds."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .models import ExchangeAuditRecord, FederatedNode, LocalTrainingMetrics, SyncRoundResult
from .storage import ControlPlaneStorage, NodeStorage

logger = logging.getLogger("scarcity.federated_databases")

FEATURE_KEYS = [
    "threat_score",
    "escalation_score",
    "coordination_score",
    "urgency_rate",
    "imperative_rate",
    "policy_severity",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ScarcityFederationManager:
    """Stateful federation manager with node DBs, ML rounds, and audit trail."""

    def __init__(self, base_dir: Union[Path, str] = "federated_databases/runtime"):
        self.base_dir = Path(base_dir)
        self.nodes_dir = self.base_dir / "nodes"
        self.audit_path = self.base_dir / "audit_log.jsonl"
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self.control = ControlPlaneStorage(self.base_dir / "federation_control.sqlite")
        self._node_stores: Dict[str, NodeStorage] = {}
        self._bootstrap_stores()

    def _bootstrap_stores(self) -> None:
        for node in self.control.list_nodes():
            self._node_stores[node["node_id"]] = NodeStorage(Path(node["db_path"]))

    def _node_store(self, node_id: str) -> NodeStorage:
        if node_id not in self._node_stores:
            raise KeyError(f"Node {node_id} not registered")
        return self._node_stores[node_id]

    def register_node(
        self,
        node_id: str,
        backend: str = "sqlite",
        county_filter: Optional[str] = None,
    ) -> FederatedNode:
        node_id_clean = (node_id or "").strip().lower().replace(" ", "_")
        if not node_id_clean:
            raise ValueError("node_id cannot be empty")
        if backend != "sqlite":
            raise ValueError("Only sqlite backend is currently supported")

        db_path = self.nodes_dir / f"{node_id_clean}.sqlite"
        self.control.upsert_node(node_id_clean, backend, str(db_path), county_filter)
        self._node_stores[node_id_clean] = NodeStorage(db_path)

        node = FederatedNode(
            node_id=node_id_clean,
            backend=backend,
            db_path=str(db_path),
            county_filter=county_filter,
        )
        self._append_audit_event("node_registered", asdict(node))
        logger.info("Registered federation node %s (%s)", node_id_clean, backend)
        return node

    def list_nodes(self) -> List[Dict[str, Any]]:
        nodes = self.control.list_nodes()
        result: List[Dict[str, Any]] = []
        for node in nodes:
            stats = self._node_stores.get(node["node_id"], NodeStorage(Path(node["db_path"]))).stats()
            merged = dict(node)
            merged.update(stats)
            result.append(merged)
        return result

    def _parse_timestamp(self, value: str) -> Optional[datetime]:
        def _normalize(dt: datetime) -> datetime:
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)

        raw = (value or "").strip()
        if not raw:
            return None
        raw = raw.replace("Z", "+00:00")
        try:
            return _normalize(datetime.fromisoformat(raw))
        except Exception:
            fmts = ["%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
            for fmt in fmts:
                try:
                    return _normalize(datetime.strptime(raw, fmt))
                except Exception:
                    continue
        return None

    def _criticality(self, row: Dict[str, Any]) -> float:
        def _f(name: str) -> float:
            try:
                return float(row.get(name, 0.0) or 0.0)
            except Exception:
                return 0.0

        score = (
            0.45 * _f("threat_score")
            + 0.25 * _f("escalation_score")
            + 0.15 * _f("coordination_score")
            + 0.10 * _f("urgency_rate")
            + 0.05 * _f("policy_severity")
        )
        return float(max(0.0, min(1.0, score)))

    def _load_live_samples(
        self,
        source_path: Path,
        lookback_hours: int = 24,
        max_rows: int = 30000,
    ) -> List[Dict[str, Any]]:
        if not source_path.exists():
            logger.warning("Live source file missing: %s", source_path)
            return []

        now = _utc_now()
        cutoff = now - timedelta(hours=max(1, int(lookback_hours)))
        rows: List[Dict[str, Any]] = []

        with source_path.open("r", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                ts = self._parse_timestamp(str(row.get("timestamp", "")))
                if ts is None:
                    continue

                criticality = self._criticality(row)
                sample = {
                    "timestamp": ts.isoformat(),
                    "county": str(row.get("location_county", "Unknown") or "Unknown"),
                    "sector": str(row.get("topic_cluster", "general") or "general"),
                    "criticality": criticality,
                    "threat_score": float(row.get("threat_score", 0.0) or 0.0),
                    "escalation_score": float(row.get("escalation_score", 0.0) or 0.0),
                    "coordination_score": float(row.get("coordination_score", 0.0) or 0.0),
                    "urgency_rate": float(row.get("urgency_rate", 0.0) or 0.0),
                    "imperative_rate": float(row.get("imperative_rate", 0.0) or 0.0),
                    "policy_severity": float(row.get("policy_severity", 0.0) or 0.0),
                    "label": 1 if criticality >= 0.6 else 0,
                }
                uid_base = str(row.get("post_id") or f"{sample['timestamp']}|{sample['county']}|{sample['criticality']:.4f}")
                sample["sample_uid"] = hashlib.sha256(uid_base.encode("utf-8")).hexdigest()
                rows.append(sample)

        live_rows = [r for r in rows if self._parse_timestamp(r["timestamp"]) and self._parse_timestamp(r["timestamp"]) >= cutoff]
        if not live_rows:
            live_rows = rows[-max_rows:]
        return live_rows[-max_rows:]

    def ingest_live_batch(
        self,
        lookback_hours: int = 24,
        source_path: Union[Path, str] = "data/synthetic_kenya_policy/tweets.csv",
    ) -> Dict[str, int]:
        nodes = self.control.list_nodes()
        if not nodes:
            return {}

        samples = self._load_live_samples(Path(source_path), lookback_hours=lookback_hours)
        if not samples:
            return {node["node_id"]: 0 for node in nodes}

        assignments: Dict[str, List[Dict[str, Any]]] = {node["node_id"]: [] for node in nodes}
        open_nodes = [node for node in nodes if not node.get("county_filter")]

        for idx, sample in enumerate(samples):
            placed = False
            for node in nodes:
                filt = node.get("county_filter")
                if filt and sample["county"].lower() == str(filt).lower():
                    assignments[node["node_id"]].append(sample)
                    placed = True
            if not placed and open_nodes:
                target = open_nodes[idx % len(open_nodes)]["node_id"]
                assignments[target].append(sample)

        inserted: Dict[str, int] = {}
        for node_id, bucket in assignments.items():
            inserted[node_id] = self._node_store(node_id).add_samples(bucket)

        self._append_audit_event(
            "live_batch_ingested",
            {
                "lookback_hours": lookback_hours,
                "source_path": str(source_path),
                "total_samples": len(samples),
                "inserted": inserted,
            },
        )
        return inserted

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        clipped = np.clip(x, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _train_local_step(
        self,
        weights: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.12,
    ) -> Tuple[np.ndarray, float, float]:
        logits = x @ weights
        probs = self._sigmoid(logits)
        eps = 1e-8
        loss = -np.mean(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
        gradient = (x.T @ (probs - y)) / max(1, len(y))
        next_weights = weights - learning_rate * gradient
        grad_norm = float(np.linalg.norm(gradient))
        return next_weights.astype(np.float64), float(loss), grad_norm

    def _global_weights(self, feature_count: int) -> np.ndarray:
        state = self.control.get_global_state("global_weights")
        if not state:
            return np.zeros(feature_count, dtype=np.float64)
        values = state.get("weights") or []
        if len(values) != feature_count:
            return np.zeros(feature_count, dtype=np.float64)
        return np.array([float(v) for v in values], dtype=np.float64)

    def _set_global_weights(self, weights: np.ndarray, round_number: int) -> None:
        payload = {
            "weights": [float(v) for v in weights.tolist()],
            "round_number": int(round_number),
            "updated_at": _utc_now().isoformat(),
        }
        self.control.set_global_state("global_weights", payload)

    def run_single_node_training(self, node_id: str, learning_rate: float = 0.12) -> Dict[str, Any]:
        node = node_id.strip().lower()
        store = self._node_store(node)
        features, labels = store.get_training_matrix(limit=15000)
        if not features:
            raise RuntimeError(f"Node {node} has no samples; ingest data first")

        x = np.array(features, dtype=np.float64)
        y = np.array(labels, dtype=np.float64)
        weights = self._global_weights(x.shape[1])
        next_weights, loss, grad_norm = self._train_local_step(weights, x, y, learning_rate=learning_rate)

        store.record_model_update(
            round_number=0,
            weights=next_weights.tolist(),
            gradient_norm=grad_norm,
            loss=loss,
            sample_count=len(y),
            metrics={"mode": "single_node"},
        )

        metric = {
            "node_id": node,
            "sample_count": int(len(y)),
            "loss": float(loss),
            "gradient_norm": float(grad_norm),
            "mode": "single_node",
        }
        self._append_audit_event("single_node_training", metric)
        return metric

    def run_sync_round(
        self,
        learning_rate: float = 0.12,
        lookback_hours: int = 24,
        source_path: Union[Path, str] = "data/synthetic_kenya_policy/tweets.csv",
    ) -> SyncRoundResult:
        nodes = self.control.list_nodes()
        if not nodes:
            raise RuntimeError("No registered federation nodes")

        self.ingest_live_batch(lookback_hours=lookback_hours, source_path=source_path)

        feature_count = len(FEATURE_KEYS)
        global_weights = self._global_weights(feature_count)
        round_number = self.control.next_round_number()
        started_at = _utc_now().isoformat()

        local_metrics: List[LocalTrainingMetrics] = []
        weighted_updates: List[np.ndarray] = []
        sample_weights: List[int] = []

        for node in nodes:
            node_id = node["node_id"]
            store = self._node_store(node_id)
            features, labels = store.get_training_matrix(limit=15000)
            if not features:
                continue

            x = np.array(features, dtype=np.float64)
            y = np.array(labels, dtype=np.float64)
            local_weights, loss, grad_norm = self._train_local_step(global_weights, x, y, learning_rate=learning_rate)
            sample_count = int(len(y))
            mean_criticality = float(np.mean(y)) if sample_count else 0.0

            store.record_model_update(
                round_number=round_number,
                weights=local_weights.tolist(),
                gradient_norm=grad_norm,
                loss=loss,
                sample_count=sample_count,
                metrics={"mode": "federated_local", "round": round_number},
            )
            store.record_shared_signal(
                round_number=round_number,
                signal_key="mean_criticality",
                signal_value=mean_criticality,
                payload={"sample_count": sample_count},
            )

            local_metrics.append(
                LocalTrainingMetrics(
                    node_id=node_id,
                    sample_count=sample_count,
                    loss=loss,
                    gradient_norm=grad_norm,
                    mean_criticality=mean_criticality,
                    feature_count=feature_count,
                )
            )
            weighted_updates.append(local_weights)
            sample_weights.append(sample_count)

            self.control.record_exchange(
                round_number=round_number,
                from_node=node_id,
                to_node="scarcity_aggregator",
                payload_type="model_update",
                payload_size=int(local_weights.size),
                details={"loss": loss, "gradient_norm": grad_norm, "sample_count": sample_count},
            )

        if not local_metrics:
            raise RuntimeError("No node had usable samples for sync round")

        total_samples = int(sum(sample_weights))
        if total_samples <= 0:
            total_samples = len(sample_weights)
            sample_weights = [1 for _ in sample_weights]

        stacked = np.vstack(weighted_updates)
        weight_array = np.array(sample_weights, dtype=np.float64)
        normalized = weight_array / np.sum(weight_array)
        aggregated_weights = np.sum(stacked * normalized[:, None], axis=0)

        global_loss = float(np.average([m.loss for m in local_metrics], weights=sample_weights))
        global_grad = float(np.average([m.gradient_norm for m in local_metrics], weights=sample_weights))

        self._set_global_weights(aggregated_weights, round_number)

        for node in nodes:
            node_id = node["node_id"]
            store = self._node_store(node_id)
            store.record_model_update(
                round_number=round_number,
                weights=aggregated_weights.tolist(),
                gradient_norm=global_grad,
                loss=global_loss,
                sample_count=total_samples,
                metrics={"mode": "federated_global", "round": round_number},
            )
            self.control.record_exchange(
                round_number=round_number,
                from_node="scarcity_aggregator",
                to_node=node_id,
                payload_type="global_model",
                payload_size=int(aggregated_weights.size),
                details={"global_loss": global_loss, "global_gradient_norm": global_grad},
            )

        completed_at = _utc_now().isoformat()
        round_result = SyncRoundResult(
            round_number=round_number,
            participants=len(local_metrics),
            total_samples=total_samples,
            global_loss=global_loss,
            global_gradient_norm=global_grad,
            started_at=started_at,
            completed_at=completed_at,
            metadata={
                "local_metrics": [asdict(m) for m in local_metrics],
                "learning_rate": learning_rate,
                "lookback_hours": lookback_hours,
            },
        )

        self.control.record_sync_round(
            round_number=round_number,
            started_at=started_at,
            completed_at=completed_at,
            participants=round_result.participants,
            total_samples=total_samples,
            aggregation_method=round_result.aggregation_method,
            global_loss=global_loss,
            global_gradient_norm=global_grad,
            metrics=round_result.metadata,
        )

        self._append_audit_event("sync_round_completed", asdict(round_result))
        logger.info(
            "Federation sync round %s complete: participants=%s, samples=%s, loss=%.4f",
            round_number,
            round_result.participants,
            round_result.total_samples,
            round_result.global_loss,
        )
        return round_result

    def get_round_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.control.get_round_history(limit=limit)

    def get_exchange_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        return self.control.get_exchange_log(limit=limit)

    def get_status(self) -> Dict[str, Any]:
        nodes = self.list_nodes()
        history = self.get_round_history(limit=1)
        latest = history[0] if history else None
        return {
            "node_count": len(nodes),
            "nodes": nodes,
            "latest_round": latest,
            "round_count": len(self.get_round_history(limit=10000)),
            "audit_path": str(self.audit_path),
            "control_db": str(self.control.db_path),
        }

    def run_smoke_round(self) -> Dict[str, Any]:
        nodes = self.control.list_nodes()
        if not nodes:
            self.register_node("org_a")
            self.register_node("org_b")

        single_metrics = self.run_single_node_training(self.control.list_nodes()[0]["node_id"])
        sync_metrics = asdict(self.run_sync_round())
        status = self.get_status()
        return {
            "single_node": single_metrics,
            "sync_round": sync_metrics,
            "status": status,
        }

    def _append_audit_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "event_type": event_type,
            "timestamp": _utc_now().isoformat(),
            "payload": payload,
        }
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


_MANAGER: Optional[ScarcityFederationManager] = None


def get_scarcity_federation(base_dir: Union[Path, str] = "federated_databases/runtime") -> ScarcityFederationManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = ScarcityFederationManager(base_dir=base_dir)
    return _MANAGER
