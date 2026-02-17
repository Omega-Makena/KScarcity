"""Background runner that boots the MPIE engine stack for the dashboard."""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from scarcity.engine.engine import MPIEOrchestrator
from scarcity.meta.integrative_meta import MetaSupervisor
from scarcity.meta.meta_learning import MetaLearningAgent
from scarcity.runtime import Telemetry, get_bus
from scarcity.simulation import AgentRegistry, SimulationConfig, SimulationEngine
from scarcity.stream import SchemaManager, StreamSource, WindowBuilder

logger = logging.getLogger(__name__)


class EngineRunner:
    """Manage lifecycle of the MPIE engine and its supporting subsystems."""

    def __init__(
        self,
        *,
        feature_dim: int = 32,
        chunk_rows: int = 512,
        window_size: int = 128,
        stride: int = 64,
        stream_interval: float = 0.25,
        telemetry_interval: float = 5.0,
        seed: int = 2025,
        history_size: int = 40,
    ) -> None:
        self.feature_dim = feature_dim
        self.chunk_rows = chunk_rows
        self.window_size = window_size
        self.stride = stride
        self.stream_interval = stream_interval
        self.telemetry_interval = telemetry_interval
        self._rng = np.random.default_rng(seed)

        self._running = False
        self._producer_task: Optional[asyncio.Task[None]] = None
        self._window_id = 0

        self.bus = get_bus()
        self.telemetry: Optional[Telemetry] = None
        self.meta_supervisor: Optional[MetaSupervisor] = None
        self.meta_agent: Optional[MetaLearningAgent] = None
        self.orchestrator: Optional[MPIEOrchestrator] = None
        self.source: Optional[StreamSource] = None
        self.builder: Optional[WindowBuilder] = None
        self.schema_manager: Optional[SchemaManager] = None

        self.sim_registry: Optional[AgentRegistry] = None
        self.sim_engine: Optional[SimulationEngine] = None
        self._registry_task: Optional[asyncio.Task[None]] = None
        self.registry_refresh_seconds = 5.0

        self._state_lock = asyncio.Lock()
        self._metrics_history: Deque[Dict[str, Any]] = deque(maxlen=history_size)
        self._latest_metrics: Optional[Dict[str, Any]] = None
        self._latest_resource_profile: Optional[Dict[str, Any]] = None
        self._latest_meta_policy: Optional[Dict[str, Any]] = None

        self._processing_handler = self._handle_processing_metrics
        self._resource_handler = self._handle_resource_profile
        self._policy_handler = self._handle_meta_policy

    async def start(self) -> None:
        """Start telemetry, meta-control, orchestrator, and synthetic stream."""
        if self._running:
            logger.info("EngineRunner already active.")
            return

        logger.info("Bootstrapping MPIE engine stack for dashboard telemetry.")

        self.telemetry = Telemetry(bus=self.bus, publish_interval=self.telemetry_interval)
        self.meta_supervisor = MetaSupervisor(bus=self.bus)
        self.meta_agent = MetaLearningAgent(bus=self.bus)
        self.orchestrator = MPIEOrchestrator(bus=self.bus)

        self.source = StreamSource(
            data_source=self._generate_chunk,
            window_size=self.chunk_rows,
            name="dashboard_synthetic_source",
        )
        self.builder = WindowBuilder(window_size=self.window_size, stride=self.stride)
        self.schema_manager = SchemaManager()

        self.bus.subscribe("processing_metrics", self._processing_handler)
        self.bus.subscribe("resource_profile", self._resource_handler)
        self.bus.subscribe("meta_policy_update", self._policy_handler)

        await self.telemetry.start()
        await self.meta_supervisor.start()
        await self.meta_agent.start()
        await self.orchestrator.start()
        await self._start_simulation()

        self._running = True
        self._producer_task = asyncio.create_task(self._stream_loop())
        logger.info("MPIE engine stack started.")

    async def stop(self) -> None:
        """Stop background stream and subsystems."""
        if not self._running:
            return

        logger.info("Stopping MPIE engine stack.")
        self._running = False

        if self._producer_task:
            self._producer_task.cancel()
            try:
                await self._producer_task
            except asyncio.CancelledError:
                pass

        self.bus.unsubscribe("processing_metrics", self._processing_handler)
        self.bus.unsubscribe("resource_profile", self._resource_handler)
        self.bus.unsubscribe("meta_policy_update", self._policy_handler)

        if self.orchestrator:
            await self.orchestrator.stop()
        if self.meta_agent:
            await self.meta_agent.stop()
        if self.meta_supervisor:
            await self.meta_supervisor.stop()
        if self.telemetry:
            await self.telemetry.stop()

        if self.source:
            self.source.stop()

        if self._registry_task:
            self._registry_task.cancel()
            try:
                await self._registry_task
            except asyncio.CancelledError:
                pass
            self._registry_task = None

        if self.sim_engine:
            await self.sim_engine.stop()
            self.sim_engine = None
        self.sim_registry = None

        logger.info("MPIE engine stack stopped.")

    async def snapshot(self) -> Dict[str, Any]:
        """Return a serialisable view of the engine state."""
        async with self._state_lock:
            history = [dict(entry) for entry in self._metrics_history]
            latest_metric = history[0] if history else None
            return {
                "running": self._running,
                "windows_ingested": self._window_id,
                "latest_metric": latest_metric,
                "history": history,
                "resource_profile": self._clone_entry(self._latest_resource_profile),
                "meta_policy": self._clone_entry(self._latest_meta_policy),
            }

    async def simulation_snapshot(self) -> Dict[str, Any]:
        """Return the latest simulation frame."""
        if not self.sim_engine:
            return self._empty_frame_payload()
        frame = self.sim_engine.visualizer_snapshot()
        payload = self._frame_to_payload(frame)
        payload["timestamp"] = datetime.now(timezone.utc).isoformat()
        return payload

    async def _stream_loop(self) -> None:
        """Continuously emit synthetic windows into the MPIE bus."""
        assert self.source and self.builder and self.schema_manager

        try:
            while self._running:
                chunk = await self.source.read_chunk()
                if chunk is None:
                    await asyncio.sleep(self.stream_interval)
                    continue

                windows = self.builder.process_chunk(chunk)
                if not windows:
                    await asyncio.sleep(self.stream_interval)
                    continue

                loop = asyncio.get_running_loop()

                for window in windows:
                    schema = self.schema_manager.infer_schema(window)
                    schema_payload: Dict[str, Any] = {}
                    if schema:
                        schema_payload = schema.to_dict()
                        fields = schema_payload.get("fields", [])
                        if isinstance(fields, list):
                            schema_payload["fields"] = {
                                str(field.get("name", f"feature_{idx}")): field for idx, field in enumerate(fields)
                            }
                    event = {
                        "data": window,
                        "schema": schema_payload,
                        "window_id": self._window_id,
                        "timestamp": loop.time(),
                    }
                    self._window_id += 1
                    await self.bus.publish("data_window", event)

                await asyncio.sleep(self.stream_interval)
        except asyncio.CancelledError:
            logger.info("EngineRunner stream loop cancelled.")
        except Exception:
            logger.exception("EngineRunner stream loop failed.")

    async def _handle_processing_metrics(self, topic: str, data: Dict[str, Any]) -> None:
        snapshot = {
            "timestamp": datetime.now(timezone.utc),
            "values": dict(data),
        }
        async with self._state_lock:
            self._latest_metrics = snapshot
            self._metrics_history.appendleft(snapshot)

    async def _handle_resource_profile(self, topic: str, data: Dict[str, Any]) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc),
            "values": dict(data),
        }
        async with self._state_lock:
            self._latest_resource_profile = entry

    async def _handle_meta_policy(self, topic: str, data: Dict[str, Any]) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc),
            "values": dict(data),
        }
        async with self._state_lock:
            self._latest_meta_policy = entry

    def _clone_entry(self, entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if entry is None:
            return None
        return {"timestamp": entry["timestamp"], "values": dict(entry["values"])}

    def _generate_chunk(self) -> np.ndarray:
        """Return a synthetic chunk for the stream source."""
        return self._rng.standard_normal(size=(self.chunk_rows, self.feature_dim)).astype(np.float32)

    async def ingest_feature_matrix(
        self,
        features: np.ndarray,
        window_size: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Stream a precomputed feature matrix through the engine as data windows.

        Intended for UI-driven dataset uploads (e.g. Excel/CSV) so that the
        hypergraph store and simulation reflect the uploaded data without
        exposing any training code to the user.
        """
        if not self.orchestrator:
            raise RuntimeError("Orchestrator not initialised")

        if not self.builder or not self.schema_manager:
            raise RuntimeError("Stream components not initialised")

        w_size = window_size or self.window_size
        s_stride = stride or self.stride
        builder = WindowBuilder(window_size=w_size, stride=s_stride)
        schema_manager = self.schema_manager

        windows = builder.process_chunk(features)
        loop = asyncio.get_running_loop()

        count = 0
        for window in windows:
            schema = schema_manager.infer_schema(window)
            schema_payload: Dict[str, Any] = {}
            if schema:
                schema_payload = schema.to_dict()
            event = {
                "data": window,
                "schema": schema_payload,
                "window_id": self._window_id,
                "timestamp": loop.time(),
            }
            self._window_id += 1
            await self.bus.publish("data_window", event)
            count += 1

        return {"windows_ingested": count}

    async def _start_simulation(self) -> None:
        if self.sim_engine or not self.orchestrator:
            return
        try:
            self.sim_registry = AgentRegistry()
            self._refresh_simulation_registry()
            sim_config = SimulationConfig()
            self.sim_engine = SimulationEngine(self.sim_registry, sim_config, bus=self.bus)
            await self.sim_engine.start()
            self._registry_task = asyncio.create_task(self._registry_sync_loop())
            logger.info("Simulation engine started.")
        except Exception:
            logger.exception("Failed to start simulation engine.")
            self.sim_engine = None
            self.sim_registry = None

    def _refresh_simulation_registry(self) -> None:
        if not self.sim_registry or not self.orchestrator or not getattr(self.orchestrator, "store", None):
            return
        snapshot = self.orchestrator.store.snapshot()
        self.sim_registry.load_from_store_snapshot(snapshot, default_value=0.0)
        if not self.sim_registry.nodes():
            self.sim_registry.seed_placeholder_graph()

    async def _registry_sync_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(self.registry_refresh_seconds)
                self._refresh_simulation_registry()
        except asyncio.CancelledError:
            pass

    def _frame_to_payload(self, frame: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "frame_id": int(frame.get("frame_id", 0)),
            "positions": self._array_to_list(frame.get("positions")),
            "colors": self._array_to_list(frame.get("colors")),
            "edges": self._array_to_list(frame.get("edges")),
        }

    def _empty_frame_payload(self) -> Dict[str, Any]:
        return {
            "frame_id": 0,
            "positions": [],
            "colors": [],
            "edges": [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def _array_to_list(value: Any) -> List[List[float]]:
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, list):
            return value
        return []
