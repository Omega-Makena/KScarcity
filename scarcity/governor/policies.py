"""
Scaling policies for the DRG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class PolicyRule:
    metric: str
    threshold: float
    action: str
    direction: str = ">"
    factor: float = 0.5
    priority: int = 1

    def triggered(self, value: float) -> bool:
        if self.direction == ">":
            return value >= self.threshold
        return value <= self.threshold


def default_policies() -> Dict[str, List[PolicyRule]]:
    return {
        "simulation": [
            PolicyRule(metric="vram_util", threshold=0.90, action="scale_down", factor=0.5, priority=3),
            PolicyRule(metric="fps", threshold=25.0, action="increase_lod", direction="<", factor=0.75, priority=2),
        ],
        "mpie": [
            PolicyRule(metric="cpu_util", threshold=0.85, action="reduce_batch", factor=0.5, priority=2),
        ],
        "meta": [
            PolicyRule(metric="vram_util", threshold=0.85, action="drop_low_priority", priority=1),
        ],
        "federation": [
            PolicyRule(metric="latency_ms", threshold=150.0, action="delay_sync", priority=1),
        ],
        "memory": [
            PolicyRule(metric="mem_util", threshold=0.90, action="flush_cache", priority=1),
        ],
    }

