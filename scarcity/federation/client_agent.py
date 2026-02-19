"""
Federated client agent coordinating local exports and incoming aggregates.

This module implements the FederationClientAgent, which runs on each node in the
federated network. It handles the export of local updates (EdgeDeltas, PathPacks)
and the reception and integration of aggregated updates from the federation.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple, List

import numpy as np

from scarcity.runtime import EventBus, get_bus
from .aggregator import FederatedAggregator, AggregationConfig
from .packets import (
    PathPack,
    EdgeDelta,
    PolicyPack,
    CausalSemanticPack,
    serialise_packet,
)
from .privacy_guard import PrivacyGuard, PrivacyConfig
from .validator import PacketValidator, ValidatorConfig
from .scheduler import FederationScheduler, SchedulerConfig
from .trust_scorer import TrustScorer, TrustConfig
from .codec import PayloadCodec, CodecConfig
from .reconciler import StoreReconciler
from .transport import BaseTransport, TransportConfig, build_transport


@dataclass
class ClientAgentConfig:
    """Configuration for FederationClientAgent."""
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    validator: ValidatorConfig = field(default_factory=ValidatorConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    codec: CodecConfig = field(default_factory=CodecConfig)
    transport: TransportConfig = field(default_factory=lambda: TransportConfig(protocol="loopback"))
    trust: TrustConfig = field(default_factory=TrustConfig)


class FederationClientAgent:
    """
    Manages federation duties for a single node.
    
    Responsibilities include:
    - Transporting packets to/from peers/coordinator.
    - Applying differential privacy to outgoing updates.
    - Validating incoming packets.
    - Reconciling updates with the local data store.
    - Scheduling exports based on system metrics.
    """

    def __init__(
        self,
        node_id: str,
        reconciler: StoreReconciler,
        bus: Optional[EventBus] = None,
        config: Optional[ClientAgentConfig] = None,
        transport: Optional[BaseTransport] = None,
    ):
        """
        Initialize the client agent.

        Args:
            node_id: Unique identifier for this node.
            reconciler: Reconciler instance to manage store updates.
            bus: EventBus for local communication.
            config: Configuration object.
            transport: Transport layer for network communication.
        """
        self.node_id = node_id
        self.reconciler = reconciler
        self.bus = bus or get_bus()
        self.config = config or ClientAgentConfig()

        self.aggregator = FederatedAggregator(self.config.aggregation)
        self.privacy_guard = PrivacyGuard(self.config.privacy)
        self.validator = PacketValidator(self.config.validator)
        self.scheduler = FederationScheduler(self.config.scheduler)
        self.trust = TrustScorer(self.config.trust)
        self.codec = PayloadCodec(self.config.codec)
        self.transport = transport or build_transport(self.config.transport)
        self.transport.register_handler(self._handle_remote_packet)

        self._lock = asyncio.Lock()
        self._outbound_queue: asyncio.Queue[Tuple[str, Dict[str, Any]]] = asyncio.Queue()
        self._export_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the agent, transport, and export loop."""
        if self._running:
            return
        self._running = True
        self.bus.subscribe("processing_metrics", self._on_processing_metrics)
        await self.transport.start()
        self._export_task = asyncio.create_task(self._export_loop())

    async def stop(self) -> None:
        """Stop the agent and cleanup resources."""
        if not self._running:
            return
        self._running = False
        self.bus.unsubscribe("processing_metrics", self._on_processing_metrics)
        if self._export_task:
            self._export_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._export_task
        await self.transport.stop()

    async def publish_packets(self, packets: Sequence[Any]) -> None:
        """
        Queue packets for export to the federation.

        Args:
            packets: Sequence of packet objects (e.g., EdgeDelta, PathPack).
        """
        for packet in packets:
            topic, payload = serialise_packet(packet)
            await self._outbound_queue.put((topic, payload))

    async def aggregate_updates(self, updates: Sequence[Sequence[float]]) -> Tuple[np.ndarray, dict]:
        """
        Aggregate updates from multiple sources (if acting as an aggregator).
        Applies privacy noise before aggregation.
        """
        noisy_updates = self.privacy_guard.apply_noise(updates)
        masked_updates = []
        seeds: List[bytes] = []
        if self.config.privacy.secure_aggregation:
            for update in noisy_updates:
                masked, seed = self.privacy_guard.secure_mask(np.asarray(update, dtype=np.float32))
                masked_updates.append(masked)
                if seed:
                    seeds.append(seed)
        else:
            masked_updates = list(noisy_updates)

        aggregate, meta = self.aggregator.aggregate(masked_updates)
        meta["secure_aggregation"] = bool(self.config.privacy.secure_aggregation)
        if seeds:
            meta["mask_seeds_b64"] = [base64.b64encode(seed).decode("ascii") for seed in seeds]
        return aggregate, meta

    async def receive_aggregated(self, packet: Dict[str, Any], trust: float) -> Dict[str, int]:
        """
        Apply validated aggregated packet to the local store.

        Args:
            packet: The received packet payload.
            trust: Trust score of the sender.

        Returns:
            Dictionary of reconciliation stats (e.g., number of upserts).
        """
        if "edges" in packet:
            pack = PathPack.from_dict(packet)
            if not self.validator.validate_path_pack(pack, trust):
                return {}
            stats = self.reconciler.merge_path_pack(pack)
            await self.bus.publish("federation_update", {"type": "path_pack", "stats": stats, "domain_id": pack.domain_id})
            return stats
        if "upserts" in packet:
            delta = EdgeDelta.from_dict(packet)
            if not self.validator.validate_edge_delta(delta, trust):
                return {}
            stats = self.reconciler.merge_edge_delta(delta)
            await self.bus.publish("federation_update", {"type": "edge_delta", "stats": stats, "domain_id": delta.domain_id})
            return stats
        if "pairs" in packet:
            causal = CausalSemanticPack.from_dict(packet)
            if not self.validator.validate_causal_pack(causal, trust):
                return {}
            stats = self.reconciler.merge_causal_pack(causal)
            await self.bus.publish("federation_update", {"type": "causal_pack", "stats": stats, "domain_id": causal.domain_id})
            return stats
        return {}

    async def _export_loop(self) -> None:
        """Background loop to process and send outbound packets."""
        try:
            while self._running:
                topic, payload = await self._outbound_queue.get()
                await self.transport.send(topic, payload)
                self.scheduler.mark_export()
        except asyncio.CancelledError:
            pass

    async def _handle_remote_packet(self, topic: str, payload: Dict[str, Any]) -> None:
        """Callback for handling incoming packets from the transport."""
        trust = self.trust.score(payload.get("peer_id", "unknown"))
        await self.receive_aggregated(payload, trust)

    async def _on_processing_metrics(self, topic: str, metrics: Dict[str, Any]) -> None:
        """Handle processing metrics to trigger health reports."""
        telemetry = {
            "latency_ms": metrics.get("engine_latency_ms", 0.0),
            "bandwidth_free": metrics.get("bandwidth_free", 0.0),
            "vram_high": metrics.get("vram_high", 0.0),
        }
        if self.scheduler.should_export(telemetry):
            await self._outbound_queue.put(("federation.health", metrics))
