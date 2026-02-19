"""
Bus integration helpers for the DRG.
"""

from __future__ import annotations

from typing import Any, Dict

from scarcity.runtime import EventBus, get_bus


class DRGHooks:
    def __init__(self, bus: EventBus | None = None):
        self.bus = bus or get_bus()

    async def publish_signal(self, signal: str, payload: Dict[str, Any]) -> None:
        await self.bus.publish(f"drg.signal.{signal}", payload)

    async def publish_telemetry(self, payload: Dict[str, Any]) -> None:
        await self.bus.publish("drg.telemetry", payload)

