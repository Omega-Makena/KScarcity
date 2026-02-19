"""
Telemetry monitor for the simulation loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict


@dataclass
class MonitorConfig:
    ema_alpha: float = 0.2


class SimulationMonitor:
    def __init__(self, config: MonitorConfig):
        self.config = config
        self._last_ts = time.time()
        self._fps_ema = 0.0
        self._latency_ema = 0.0
        self._frame_count = 0

    def tick(self) -> Dict[str, float]:
        now = time.time()
        elapsed = now - self._last_ts
        self._last_ts = now
        fps = 1.0 / max(elapsed, 1e-6)
        latency_ms = elapsed * 1000.0

        alpha = self.config.ema_alpha
        if self._frame_count == 0:
            self._fps_ema = fps
            self._latency_ema = latency_ms
        else:
            self._fps_ema = (1 - alpha) * self._fps_ema + alpha * fps
            self._latency_ema = (1 - alpha) * self._latency_ema + alpha * latency_ms
        self._frame_count += 1

        return {
            "fps": self._fps_ema,
            "latency_ms": self._latency_ema,
        }

