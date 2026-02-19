"""
Actuator implementations for DRG decisions.
"""

from __future__ import annotations

from typing import Dict, List

from .registry import SubsystemRegistry


class ResourceActuators:
    def __init__(self, registry: SubsystemRegistry):
        self.registry = registry

    def execute(self, subsystem: str, action: str, factor: float) -> bool:
        handle = self.registry.get(subsystem)
        if handle is None:
            return False
        method_map = {
            "scale_down": "scale_down",
            "scale_up": "scale_up",
            "reduce_batch": "reduce_batch",
            "drop_low_priority": "drop_low_priority",
            "delay_sync": "delay_sync",
            "flush_cache": "flush_cache",
            "increase_lod": "increase_lod",
        }
        method = method_map.get(action)
        if method is None:
            return False
        return handle.call(method, factor=factor)

