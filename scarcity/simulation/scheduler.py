"""
Simulation scheduler for pacing the rendering and physics loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict


@dataclass
class SimulationSchedulerConfig:
    step_time_ms: float = 33.0  # ~30 FPS
    drift_check_interval: int = 10
    latency_target_ms: float = 120.0
    jitter: float = 0.1
    min_step_ms: float = 16.0
    max_step_ms: float = 100.0


class SimulationScheduler:
    def __init__(self, config: SimulationSchedulerConfig):
        self.config = config
        self._last_step_ts = time.time()
        self._steps_since_drift_check = 0
        self._step_time_ms = config.step_time_ms

    def should_step(self) -> bool:
        now = time.time()
        elapsed_ms = (now - self._last_step_ts) * 1000.0
        return elapsed_ms >= self._step_time_ms

    def mark_step(self) -> None:
        self._last_step_ts = time.time()
        self._steps_since_drift_check += 1

    def should_check_drift(self) -> bool:
        if self._steps_since_drift_check >= self.config.drift_check_interval:
            self._steps_since_drift_check = 0
            return True
        return False

    def adapt(self, telemetry: Dict[str, float]) -> None:
        latency_ms = telemetry.get("latency_ms", 0.0)
        fps = telemetry.get("fps", 30.0)
        vram_high = telemetry.get("vram_high", 0.0)

        step_ms = self.config.step_time_ms

        if latency_ms > self.config.latency_target_ms or vram_high:
            step_ms = min(self.config.max_step_ms, step_ms * 1.2)
        elif fps > 45:
            step_ms = max(self.config.min_step_ms, step_ms * 0.9)

        jitter = self.config.jitter * step_ms
        self._step_time_ms = max(self.config.min_step_ms, min(self.config.max_step_ms, step_ms + jitter))

