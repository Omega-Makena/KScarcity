"""
Scheduler for meta-learning updates.

This module controls the timing of meta-learning updates. It adapts the update
frequency based on system performance metrics (latency, VRAM usage) to prevent
overloading the system while ensuring timely model improvements.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class MetaSchedulerConfig:
    """Configuration for the MetaScheduler."""
    update_interval_windows: int = 10
    latency_target_ms: float = 80.0
    latency_headroom_factor: float = 0.7  # Fraction of target below which to speed up
    jitter: float = 0.1
    min_interval_windows: int = 3
    max_interval_windows: int = 20
    # Interval adjustment factors
    interval_decay_factor: float = 0.7  # Multiplier when over latency target
    interval_speedup_factor: float = 0.8  # Multiplier when under latency headroom
    interval_load_increment: int = 2  # Added when bandwidth is low


class MetaScheduler:
    """
    Maintains cadence for meta updates using window counts and telemetry.
    
    The scheduler dynamically adjusts the interval between meta-updates based
    on observed system latency and resource availability. It can slow down
    updates under load or speed them up when resources are free.
    """

    def __init__(self, config: Optional[MetaSchedulerConfig] = None):
        """
        Initialize the scheduler.

        Args:
            config: Configuration object. Defaults to default settings.
        """
        self.config = config or MetaSchedulerConfig()
        self._window_counter = 0
        self._last_update_ts = 0.0
        self._interval_windows = self.config.update_interval_windows

    def record_window(self) -> None:
        """Increment the window counter (called after processing a window)."""
        self._window_counter += 1

    def should_update(self, telemetry: Dict[str, float]) -> bool:
        """
        Determine if a meta-update should be triggered.

        Args:
            telemetry: Dictionary of current system telemetry metrics.

        Returns:
            True if an update is due, False otherwise.
        """
        latency = telemetry.get("latency_ms", self.config.latency_target_ms)
        vram_high = telemetry.get("vram_high", 0.0)
        bandwidth_low = telemetry.get("bandwidth_low", 0.0)

        self._adapt_interval(latency, vram_high, bandwidth_low)
        if self._window_counter >= self._interval_windows:
            self._window_counter = 0
            self._last_update_ts = time.time()
            return True
        return False

    def _adapt_interval(self, latency: float, vram_high: float, bandwidth_low: float) -> None:
        """
        Adjust the update interval based on system load.
        
        - Increases interval if latency is high or VRAM is tight.
        - Decreases interval if latency is low and bandwidth is available.
        - Adds jitter to avoid synchronization issues.
        """
        cfg = self.config
        interval = self._interval_windows

        if latency > cfg.latency_target_ms or vram_high:
            interval = max(cfg.min_interval_windows, int(max(1, math.floor(interval * cfg.interval_decay_factor))))
        if bandwidth_low:
            interval = min(cfg.max_interval_windows, interval + cfg.interval_load_increment)
        if latency < cfg.latency_target_ms * cfg.latency_headroom_factor and not vram_high:
            interval = max(cfg.min_interval_windows, int(round(interval * cfg.interval_speedup_factor)))

        if cfg.jitter > 0.0:
            jitter = random.uniform(-cfg.jitter, cfg.jitter)
            interval = int(round(interval * (1 + jitter)))
        interval = max(cfg.min_interval_windows, min(cfg.max_interval_windows, interval))
        self._interval_windows = interval
