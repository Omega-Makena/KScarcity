"""
Telemetry helpers for the meta-learning layer.

This module provides utility functions for constructing and publishing telemetry
snapshots related to the meta-learning process, such as reward, update rates,
and storage usage.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

from scarcity.runtime import EventBus, get_bus


def build_meta_metrics_snapshot(
    reward: float,
    update_rate: float,
    gain: float,
    confidence: float,
    drift_score: float,
    latency_ms: float,
    storage_mb: float,
    extras: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Construct a standardized dictionary of meta-learning metrics.

    Args:
        reward: The current meta-reward.
        update_rate: The rate of meta-updates (updates per window).
        gain: The average gain from the aggregated updates.
        confidence: The average confidence of the updates.
        drift_score: The optimizer's EMA reward (drift metric).
        latency_ms: Current system latency in milliseconds.
        storage_mb: Estimated storage usage in megabytes.
        extras: Optional dictionary of additional metrics.

    Returns:
        A dictionary containing the collected metrics and a timestamp.
    """
    snapshot = {
        "meta_reward": float(reward),
        "meta_update_rate": float(update_rate),
        "meta_gain": float(gain),
        "meta_confidence": float(confidence),
        "meta_drift_score": float(drift_score),
        "meta_latency_ms": float(latency_ms),
        "meta_storage_mb": float(storage_mb),
        "timestamp": time.time(),
    }
    if extras:
        snapshot.update({k: float(v) for k, v in extras.items()})
    return snapshot


async def publish_meta_metrics(bus: Optional[EventBus], snapshot: Dict[str, float]) -> None:
    """
    Publish a metrics snapshot to the 'meta_metrics' topic on the event bus.

    Args:
        bus: The EventBus instance relative to which the event should be published.
        snapshot: The metrics dictionary to publish.
    """
    bus = bus or get_bus()
    await bus.publish("meta_metrics", snapshot)
