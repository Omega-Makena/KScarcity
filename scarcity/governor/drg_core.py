"""
Dynamic Resource Governor core loop.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from scarcity.runtime import EventBus, get_bus

from .sensors import ResourceSensors, SensorConfig
from .profiler import ResourceProfiler, ProfilerConfig
from .policies import PolicyRule, default_policies
from .actuators import ResourceActuators
from .registry import SubsystemRegistry, SubsystemHandle
from .monitor import DRGMonitor, MonitorConfig
from .hooks import DRGHooks

import contextlib


@dataclass
class DRGConfig:
    """
    Configuration for the Dynamic Resource Governor.

    Attributes:
        sensor: Configuration for telemetry sensors.
        profiler: Configuration for the resource usage profiler/forecaster.
        control_interval: Time in seconds between control loop iterations.
        policies: Dictionary mapping subsystem names to lists of policy rules.
        monitor: Configuration for the monitoring system.
    """
    sensor: SensorConfig = field(default_factory=SensorConfig)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    control_interval: float = 0.5
    policies: Dict[str, List[PolicyRule]] = None  # type: ignore
    monitor: MonitorConfig = field(default_factory=MonitorConfig)

    def __post_init__(self):
        if self.policies is None:
            self.policies = default_policies()


class DynamicResourceGovernor:
    """
    Central controller for resource stability.

    Coordinates system telemetry (Sensors), predictive profiling (Profiler),
    rule evaluation (Policies), and corrective actions (Actuators) to keep
    the system stable under load.
    
    The governor runs an autonomous control loop that samples metrics,
    forecasts resource pressure, and dispatches backpressure or scaling signals
    to registered subsystems.
    """

    def __init__(
        self,
        config: DRGConfig,
        bus: Optional[EventBus] = None,
    ):
        """
        Initializes the Governor.

        Args:
            config: Configuration object.
            bus: EventBus for publishing telemetry and signals.
        """
        self.config = config
        self.bus = bus or get_bus()

        self.sensors = ResourceSensors(config.sensor)
        self.profiler = ResourceProfiler(config.profiler)
        self.registry = SubsystemRegistry()
        self.actuators = ResourceActuators(self.registry)
        self.monitor = DRGMonitor(config.monitor)
        self.hooks = DRGHooks(self.bus)

        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_metrics: Dict[str, float] = {}
        self._ema: Dict[str, float] = {}
        self._forecast: Dict[str, float] = {}

    def register_subsystem(self, name: str, handle: SubsystemHandle | object) -> None:
        """
        Registers a subsystem to be managed by the governor.

        Args:
            name: Unique name of the subsystem (e.g., 'inference_engine').
            handle: An object or handle implementing the tunable interface
                (e.g., `set_parameter`).
        """
        if isinstance(handle, SubsystemHandle):
            self.registry.register(name, handle.handle)
        else:
            self.registry.register(name, handle)

    async def start(self) -> None:
        """
        Starts the governor's control loop.
        """
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """
        Stops the governor's control loop.
        """
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def _loop(self) -> None:
        """
        The main control loop.

        Steps:
        1. Sample current system metrics (CPU, RAM, etc.).
        2. Update profiles and forecasts.
        3. Evaluate policies against forecasts.
        4. Dispatch control signals (actuations).
        5. Record monitoring data.
        """
        try:
            while self._running:
                metrics = self.sensors.sample()
                ema, forecast = self.profiler.update(metrics)
                decisions = self._evaluate_policies(metrics, ema, forecast)
                await self._dispatch_signals(metrics, decisions)
                self.monitor.record({**metrics, **ema})
                await asyncio.sleep(self.config.control_interval)
        except asyncio.CancelledError:  # pragma: no cover
            pass

    def _evaluate_policies(
        self,
        metrics: Dict[str, float],
        ema: Dict[str, float],
        forecast: Dict[str, float],
    ) -> List[Tuple[str, PolicyRule]]:
        """
        Checks active policies against current and projected metrics.

        Args:
            metrics: Instantaneous metrics.
            ema: Smoothed metrics.
            forecast: Projected metrics.

        Returns:
            A list of triggered (subsystem_name, rule) tuples.
        """
        decisions: List[Tuple[str, PolicyRule]] = []
        for subsystem, rules in self.config.policies.items():
            for rule in rules:
                value = forecast.get(rule.metric, metrics.get(rule.metric, 0.0))
                if rule.triggered(value):
                    if self.actuators.execute(subsystem, rule.action, rule.factor):
                        decisions.append((subsystem, rule))
        return decisions

    async def _dispatch_signals(self, metrics: Dict[str, float], decisions: List[Tuple[str, PolicyRule]]) -> None:
        """
        Publishes telemetry and control signals to the event bus.
        """
        telemetry_payload = {
            "metrics": metrics,
            "ema": self.profiler._ema,
            "forecast": self.profiler._kalman,
        }
        await self.hooks.publish_telemetry(telemetry_payload)
        for subsystem, rule in decisions:
            payload = {
                "subsystem": subsystem,
                "action": rule.action,
                "metric": rule.metric,
                "threshold": rule.threshold,
                "factor": rule.factor,
            }
            await self.hooks.publish_signal(rule.action, payload)
