"""Unified ML pipeline runner for single-node and federated modes."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

from .scarcity_federation import ScarcityFederationManager, get_scarcity_federation


class ScarcityMLPipeline:
    """Run the same criticality model in single-node or federated orchestration mode."""

    def __init__(self, manager: Optional[ScarcityFederationManager] = None):
        self.manager = manager or get_scarcity_federation()

    def run(
        self,
        mode: str = "single_node",
        node_id: Optional[str] = None,
        learning_rate: float = 0.12,
        lookback_hours: int = 24,
    ) -> Dict[str, Any]:
        mode_normalized = (mode or "single_node").strip().lower()

        if mode_normalized in {"single", "single_node", "single-node"}:
            target = node_id
            if not target:
                nodes = self.manager.list_nodes()
                if not nodes:
                    created = self.manager.register_node("org_a")
                    target = created.node_id
                else:
                    target = nodes[0]["node_id"]
            self.manager.ingest_live_batch(lookback_hours=lookback_hours)
            return {
                "mode": "single_node",
                "result": self.manager.run_single_node_training(
                    node_id=str(target),
                    learning_rate=learning_rate,
                ),
            }

        if mode_normalized in {"federated", "federation", "multi_node", "multi-node"}:
            result = self.manager.run_sync_round(
                learning_rate=learning_rate,
                lookback_hours=lookback_hours,
            )
            return {
                "mode": "federated",
                "result": asdict(result),
            }

        raise ValueError(f"Unknown pipeline mode: {mode}")
