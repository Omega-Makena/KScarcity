"""
Adaptive scheduler for the SCARCITY federation layer.

This module controls when to trigger federated exports. It adjusts the export interval
dynamically based on system telemetry (latency, bandwidth, VRAM, CPU) to avoid
congesting the node or the network.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Dict


@dataclass
class SchedulerConfig:
    """Configuration for FederationScheduler."""
    base_export_interval: float = 10.0
    min_export_interval: float = 2.0
    max_export_interval: float = 120.0
    latency_target_ms: float = 120.0
    high_latency_backoff: float = 1.25
    vram_penalty: float = 1.4
    bandwidth_boost: float = 0.7
    jitter: float = 0.1
    ema_alpha: float = 0.2
    max_payload_kb: int = 256
    min_payload_kb: int = 64


class FederationScheduler:
    """
    Decides when federation exports should happen based on latency and resource metrics.
    
    It maintains an Exponential Moving Average (EMA) of system latency and adjusts
    the time between exports accordingly. High load increases the interval (backoff),
    while free resources decrease it (boost).
    """

    def __init__(self, config: SchedulerConfig):
        """
        Initialize the scheduler.

        Args:
            config: Scheduler configuration object.
        """
        self.config = config
        self._last_export_ts = 0.0
        self._interval = config.base_export_interval
        self._latency_ema = config.latency_target_ms

    def should_export(self, telemetry: Dict[str, float]) -> bool:
        """
        Determine if it is time to perform an export.

        Args:
            telemetry: Dictionary containing current system metrics 
                       (latency_ms, bandwidth_free, etc.).

        Returns:
            True if an export should occur, False otherwise.
        """
        now = time.time()
        elapsed = now - self._last_export_ts
        self._update_interval(telemetry)
        return elapsed >= self._interval

    def mark_export(self) -> None:
        """Record the timestamp of a successful export attempt."""
        self._last_export_ts = time.time()

    def max_payload_bytes(self, drg_util: Dict[str, float]) -> int:
        """
        Calculate the maximum allowed payload size based on current resource usage.

        Args:
            drg_util: DRG utility metrics (bandwidth_low, bandwidth_free, cpu_util_high).

        Returns:
            Maximum payload size in bytes.
        """
        cfg = self.config
        limit_kb = float(cfg.max_payload_kb)

        if drg_util.get("bandwidth_low", 0.0):
            limit_kb = max(float(cfg.min_payload_kb), limit_kb * 0.5)
        elif drg_util.get("bandwidth_free", 0.0):
            limit_kb = limit_kb * 1.2

        if drg_util.get("cpu_util_high", 0.0):
            limit_kb = max(float(cfg.min_payload_kb), limit_kb * 0.7)

        return int(limit_kb * 1024)

    def notify_success(self, payload_bytes: int) -> None:
        """
        Feedback mechanism for successful transmission.
        
        If the transmitted payload was small relative to the limit, we might
        reduce the interval to communicate more frequently.

        Args:
            payload_bytes: Size of the transmitted payload.
        """
        cfg = self.config
        fraction = payload_bytes / float(cfg.max_payload_kb * 1024)
        if fraction < 0.3:
            self._interval = max(cfg.min_export_interval, self._interval * 0.9)

    def _update_interval(self, telemetry: Dict[str, float]) -> None:
        """Update the adaptive export interval based on telemetry."""
        cfg = self.config
        latency = telemetry.get("latency_ms", cfg.latency_target_ms)
        bandwidth_free = telemetry.get("bandwidth_free", 0.0)
        bandwidth_low = telemetry.get("bandwidth_low", 0.0)
        vram_high = telemetry.get("vram_high", 0.0)

        self._latency_ema = (1 - cfg.ema_alpha) * self._latency_ema + cfg.ema_alpha * latency

        interval = cfg.base_export_interval
        if self._latency_ema > cfg.latency_target_ms:
            interval *= cfg.high_latency_backoff
        if vram_high:
            interval *= cfg.vram_penalty
        if bandwidth_free:
            interval *= cfg.bandwidth_boost
        if bandwidth_low:
            interval *= 1.2

        jitter = random.uniform(-cfg.jitter, cfg.jitter) * interval
        interval = interval + jitter
        interval = max(cfg.min_export_interval, min(cfg.max_export_interval, interval))

        self._interval = interval