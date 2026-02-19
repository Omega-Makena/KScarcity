"""
Telemetry support for FMI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

from .contracts import PacketBase, PacketType


@dataclass
class TelemetryCounters:
    packets_in: int = 0
    packets_out: int = 0
    drops: int = 0
    quarantines: int = 0
    last_emit_ts: float = field(default_factory=time.time)
    cohorts_active: int = 0


class FMITelemetry:
    """
    Aggregates operational metrics for FMI processing.
    """

    def __init__(self) -> None:
        self._counters = TelemetryCounters()
        self._type_counters: Dict[str, int] = {ptype.value: 0 for ptype in PacketType}
        self._latency: Dict[str, float] = {}
        self._meta_gain: float = 0.0

    def record_ingress(
        self,
        packet: PacketBase,
        *,
        size_kb: float,
        trust: float,
        dropped: bool = False,
        quarantined: bool = False,
    ) -> None:
        self._counters.packets_in += 1
        self._type_counters[packet.type.value] += 1
        if dropped:
            self._counters.drops += 1
        if quarantined:
            self._counters.quarantines += 1
        self._latency["last_ingress_ts"] = time.time()
        self._latency["last_packet_kb"] = size_kb
        self._latency["last_trust"] = trust

    def record_emit(self, result: Mapping[str, Any]) -> None:
        self._counters.packets_out += 1
        self._counters.last_emit_ts = time.time()
        if "meta_gain_delta" in result:
            try:
                self._meta_gain = float(result["meta_gain_delta"])
            except (TypeError, ValueError):  # pragma: no cover - defensive
                pass

    def update_active_cohorts(self, count: int) -> None:
        self._counters.cohorts_active = count

    def snapshot(self) -> Dict[str, Any]:
        return {
            "packets_in": self._counters.packets_in,
            "packets_out": self._counters.packets_out,
            "drops": self._counters.drops,
            "quarantines": self._counters.quarantines,
            "cohorts_active": self._counters.cohorts_active,
            "type_breakdown": dict(self._type_counters),
            "latency": dict(self._latency),
            "meta_gain_delta": self._meta_gain,
        }

    def reset(self) -> None:
        self._counters = TelemetryCounters()
        self._type_counters = {ptype.value: 0 for ptype in PacketType}
        self._latency.clear()
        self._meta_gain = 0.0


__all__ = [
    "FMITelemetry",
    "TelemetryCounters",
]


