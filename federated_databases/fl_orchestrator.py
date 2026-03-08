"""
Event-Driven Federated Learning Orchestrator.

Connects the EventBus to the ScarcityFederationManager so that data
uploads automatically trigger local training → aggregation → global
model distribution.

Flow
----
1. Data arrives (upload/ingest) → EventBus publishes ``fl.data_ready``
2. Orchestrator triggers local training on the reporting node
3. Node training completes → ``fl.weights_ready``
4. When enough nodes report (≥ ``min_nodes_per_round``) → aggregation
5. Aggregated global model pushed to all nodes → ``fl.global_updated``

Usage::

    from federated_databases.fl_orchestrator import FLOrchestrator
    from federated_databases.fl_config import FLOrchestratorConfig
    from federated_databases.scarcity_federation import get_scarcity_federation
    from scarcity.runtime import get_bus

    fm = get_scarcity_federation()
    bus = get_bus()
    orch = FLOrchestrator(fm, bus, FLOrchestratorConfig(model_name="hypothesis_ensemble"))
    await orch.start()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .fl_config import FLOrchestratorConfig
from .scarcity_federation import ScarcityFederationManager

logger = logging.getLogger("scarcity.fl_orchestrator")


class FLOrchestrator:
    """
    Event-driven federated learning orchestrator.

    Listens for data upload events on the EventBus and automatically
    triggers the local training → aggregation → global model pipeline.
    """

    def __init__(
        self,
        federation_manager: ScarcityFederationManager,
        bus,
        config: Optional[FLOrchestratorConfig] = None,
    ):
        self.fm = federation_manager
        self.bus = bus
        self.config = config or FLOrchestratorConfig()

        # State
        self._running = False
        self._pending_updates: Dict[str, np.ndarray] = {}
        self._pending_losses: Dict[str, float] = {}
        self._pending_samples: Dict[str, int] = {}
        self._round_count = 0
        self._aggregation_timer: Optional[asyncio.Task] = None
        self._stats = {
            "rounds_completed": 0,
            "total_updates_received": 0,
            "total_samples_trained": 0,
            "last_global_loss": None,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Subscribe to events and start the orchestrator."""
        if self._running:
            logger.warning("FL Orchestrator already running")
            return

        self._running = True

        # Subscribe to data events
        self.bus.subscribe("fl.data_ready", self._on_data_ready)
        self.bus.subscribe("fl.weights_ready", self._on_weights_ready)
        self.bus.subscribe("fl.force_round", self._on_force_round)

        logger.info(
            "FL Orchestrator started (model=%s, min_nodes=%d, ws_port=%d)",
            self.config.model_name,
            self.config.min_nodes_per_round,
            self.config.ws_port,
        )

    async def stop(self) -> None:
        """Stop the orchestrator and unsubscribe from events."""
        self._running = False

        self.bus.unsubscribe("fl.data_ready", self._on_data_ready)
        self.bus.unsubscribe("fl.weights_ready", self._on_weights_ready)
        self.bus.unsubscribe("fl.force_round", self._on_force_round)

        if self._aggregation_timer and not self._aggregation_timer.done():
            self._aggregation_timer.cancel()

        logger.info("FL Orchestrator stopped")

    # ------------------------------------------------------------------
    # Event Handlers
    # ------------------------------------------------------------------

    async def _on_data_ready(self, topic: str, payload: Dict[str, Any]) -> None:
        """
        Handle data upload event.

        Expected payload:
            {"node_id": "nairobi", "source_path": "...", "sample_count": 150}
        """
        if not self._running:
            return

        node_id = payload.get("node_id")
        if not node_id:
            logger.warning("fl.data_ready received without node_id")
            return

        logger.info(f"Data ready on node '{node_id}', triggering local training...")

        # Optional: auto-ingest from source
        if self.config.auto_ingest and payload.get("source_path"):
            try:
                self.fm.ingest_live_batch(
                    lookback_hours=self.config.lookback_hours,
                    source_path=payload["source_path"],
                )
            except Exception as e:
                logger.error(f"Auto-ingest failed for {node_id}: {e}")

        # Run local training
        try:
            result = self.fm.run_single_node_training(
                node_id=node_id,
                learning_rate=self.config.learning_rate,
                model_name=self.config.model_name,
            )

            # Publish weights ready event
            store = self.fm._node_store(node_id.strip().lower())
            features, labels = store.get_training_matrix(limit=15000)
            x = np.array(features, dtype=np.float64)
            weights = self.fm._global_weights(x.shape[1])

            # Use the model to get actual trained weights
            from .model_registry import FLModelRegistry

            if FLModelRegistry.has(self.config.model_name):
                model = FLModelRegistry.create(
                    self.config.model_name,
                    n_features=x.shape[1],
                    learning_rate=self.config.learning_rate,
                )
                fl_result = model.train_local(
                    x,
                    np.array(labels, dtype=np.float64),
                    global_weights=weights,
                )
                trained_weights = fl_result.weights
            else:
                trained_weights = weights

            await self.bus.publish(
                "fl.weights_ready",
                {
                    "node_id": node_id,
                    "weights": trained_weights.tolist(),
                    "loss": result.get("loss", 0.0),
                    "sample_count": result.get("sample_count", 0),
                },
            )

            logger.info(
                f"Node '{node_id}' training complete: "
                f"loss={result.get('loss', 0):.4f}, "
                f"samples={result.get('sample_count', 0)}"
            )

        except Exception as e:
            logger.error(f"Training failed for node '{node_id}': {e}")
            await self.bus.publish(
                "fl.training_error",
                {"node_id": node_id, "error": str(e)},
            )

    async def _on_weights_ready(self, topic: str, payload: Dict[str, Any]) -> None:
        """
        Collect weight updates from nodes.

        When enough nodes have reported, trigger aggregation.
        """
        if not self._running:
            return

        node_id = payload.get("node_id")
        weights = payload.get("weights")
        if not node_id or weights is None:
            return

        self._pending_updates[node_id] = np.array(weights, dtype=np.float64)
        self._pending_losses[node_id] = payload.get("loss", 0.0)
        self._pending_samples[node_id] = payload.get("sample_count", 0)
        self._stats["total_updates_received"] += 1
        self._stats["total_samples_trained"] += payload.get("sample_count", 0)

        logger.info(
            f"Weights received from '{node_id}' "
            f"({len(self._pending_updates)}/{self.config.min_nodes_per_round} nodes ready)"
        )

        # Check if we have enough nodes
        if len(self._pending_updates) >= self.config.min_nodes_per_round:
            await self._run_aggregation_round()
        else:
            # Start or reset the max-wait timer
            self._start_aggregation_timer()

    async def _on_force_round(self, topic: str, payload: Dict[str, Any]) -> None:
        """Force an aggregation round with whatever updates are pending."""
        if self._pending_updates:
            logger.info("Forced aggregation round triggered")
            await self._run_aggregation_round()

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    async def _run_aggregation_round(self) -> None:
        """Aggregate pending updates and distribute the global model."""
        if not self._pending_updates:
            return

        # Cancel timer
        if self._aggregation_timer and not self._aggregation_timer.done():
            self._aggregation_timer.cancel()

        participants = list(self._pending_updates.keys())
        n_participants = len(participants)

        logger.info(
            f"Starting aggregation round with {n_participants} participants: "
            f"{participants}"
        )

        try:
            # Stack weight updates
            all_weights = [self._pending_updates[nid] for nid in participants]
            sample_counts = [self._pending_samples.get(nid, 1) for nid in participants]

            # Try Byzantine-robust aggregation
            try:
                from scarcity.federation.aggregator import (
                    FederatedAggregator,
                    AggregationConfig,
                    AggregationMethod,
                )

                method_map = {
                    "trimmed_mean": AggregationMethod.TRIMMED_MEAN,
                    "fedavg": AggregationMethod.FEDAVG,
                    "krum": AggregationMethod.KRUM,
                    "bulyan": AggregationMethod.BULYAN,
                    "median": AggregationMethod.MEDIAN,
                    "weighted": AggregationMethod.WEIGHTED,
                }
                method = method_map.get(
                    self.config.aggregation_method, AggregationMethod.TRIMMED_MEAN
                )

                agg_config = AggregationConfig(
                    method=method,
                    trim_alpha=self.config.trim_alpha,
                )
                aggregator = FederatedAggregator(agg_config)
                aggregated, meta = aggregator.aggregate(all_weights)

                logger.info(f"Aggregation method: {meta.get('method', 'unknown')}")

            except ImportError:
                # Fallback: weighted average
                stacked = np.vstack(all_weights)
                w = np.array(sample_counts, dtype=np.float64)
                w = w / np.sum(w)
                aggregated = np.sum(stacked * w[:, None], axis=0)
                meta = {"method": "weighted_average_fallback"}

            # Update global model
            self._round_count += 1
            round_number = self._round_count

            global_loss = float(
                np.average(
                    [self._pending_losses.get(nid, 0) for nid in participants],
                    weights=sample_counts,
                )
            )

            self.fm._set_global_weights(aggregated, round_number)
            self._stats["rounds_completed"] = round_number
            self._stats["last_global_loss"] = global_loss

            # Clear pending
            self._pending_updates.clear()
            self._pending_losses.clear()
            self._pending_samples.clear()

            # Publish global update event
            await self.bus.publish(
                "fl.global_updated",
                {
                    "round_number": round_number,
                    "participants": participants,
                    "n_participants": n_participants,
                    "global_loss": global_loss,
                    "aggregation_method": meta.get("method", "unknown"),
                    "weight_size": int(aggregated.size),
                },
            )

            logger.info(
                f"Round {round_number} complete: "
                f"participants={n_participants}, "
                f"loss={global_loss:.4f}, "
                f"method={meta.get('method', 'unknown')}"
            )

            # Check max rounds
            if 0 < self.config.max_rounds <= round_number:
                logger.info(f"Reached max rounds ({self.config.max_rounds}), stopping")
                await self.stop()

        except Exception as e:
            logger.error(f"Aggregation round failed: {e}", exc_info=True)
            await self.bus.publish(
                "fl.aggregation_error",
                {"error": str(e), "pending_nodes": list(self._pending_updates.keys())},
            )

    # ------------------------------------------------------------------
    # Timer
    # ------------------------------------------------------------------

    def _start_aggregation_timer(self) -> None:
        """Start a timer that forces aggregation after max_wait_seconds."""
        if self._aggregation_timer and not self._aggregation_timer.done():
            self._aggregation_timer.cancel()

        async def _timeout():
            await asyncio.sleep(self.config.max_wait_seconds)
            if self._pending_updates and self._running:
                logger.info(
                    f"Max wait ({self.config.max_wait_seconds}s) reached, "
                    f"aggregating with {len(self._pending_updates)} nodes"
                )
                await self._run_aggregation_round()

        self._aggregation_timer = asyncio.create_task(_timeout())

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def stats(self) -> Dict[str, Any]:
        """Return orchestrator statistics."""
        return {
            **self._stats,
            "pending_nodes": list(self._pending_updates.keys()),
            "is_running": self._running,
            "config": {
                "model_name": self.config.model_name,
                "min_nodes_per_round": self.config.min_nodes_per_round,
                "aggregation_method": self.config.aggregation_method,
            },
        }

    @property
    def is_running(self) -> bool:
        return self._running
